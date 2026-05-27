# stochastic_superhuman_fairness/core/diagnostic_sets.py

from __future__ import annotations

from typing import Callable, Mapping, Sequence
import numpy as np
import torch


ArrayLike = np.ndarray | torch.Tensor


# ---------------------------------------------------------------------
# Basic helpers
# ---------------------------------------------------------------------

def _to_numpy_1d(x, *, dtype=None):
    if torch.is_tensor(x):
        x = x.detach().cpu().numpy()
    x = np.asarray(x)
    if dtype is not None:
        x = x.astype(dtype)
    return x.reshape(-1)


def _get_sensitive_1d(A):
    if torch.is_tensor(A):
        A = A.detach().cpu().numpy()
    A = np.asarray(A)
    if A.ndim > 1:
        A = A[:, 0]
    return A.reshape(-1)


def _maybe_numpy(x):
    if torch.is_tensor(x):
        return x.detach().cpu().numpy()
    return x


def _as_float_tensor(x):
    return torch.as_tensor(_maybe_numpy(x), dtype=torch.float32)


def _compute_feats(
    *,
    y_true,
    y_demo,
    A,
    X,
    metrics,
    compute_fairness_features,
):
    ff = compute_fairness_features(
        _as_float_tensor(y_true),
        _as_float_tensor(y_demo),
        _as_float_tensor(A),
        metrics=metrics,
        X=None if X is None else _as_float_tensor(X),
    )
    if torch.is_tensor(ff):
        ff = ff.detach().cpu().numpy()
    return np.asarray(ff, dtype=np.float32)


# ---------------------------------------------------------------------
# Semi-random diagnostic set: old 50/50-style behavior
# ---------------------------------------------------------------------
def make_semi_random_diagnostic_split(
    *,
    demos,
    metrics,
    compute_fairness_features,
    zero_one_loss,
    train_frac=0.5,
    random_mult=0.5,
    replace_train=False,
    p_one=0.5,
    seed=None,
):
    """
    Mix a fraction of real demos with random-decision demos.

    Equivalent to your previous `_make_diagnostic_split`, but moved out of
    Demonstrator.
    """
    if not (0.0 <= train_frac <= 1.0):
        raise ValueError("train_frac must be in [0,1]")
    if not (0.0 <= random_mult <= 10.0):
        raise ValueError("random_mult must be in [0,10]")
    if demos is None or len(demos) == 0:
        return []

    rng = np.random.default_rng(seed)
    n_total = len(demos)

    n_keep = int(round(train_frac * n_total))
    n_rand = int(round(random_mult * n_total))

    if n_keep > 0:
        chosen = rng.choice(n_total, size=n_keep, replace=replace_train)
        real_demos = [demos[int(i)] for i in chosen]
    else:
        real_demos = []

    random_demos = []

    for rid in range(n_rand):
        src_id = int(rng.integers(0, n_total))
        src = demos[src_id]

        y_ref = src["y"]
        X_ref = src.get("X", None)
        A_ref = src["A"]
        idx_ref = src.get("indices", None)

        y_shape = np.shape(_maybe_numpy(y_ref))
        y_demo = (rng.random(y_shape) < p_one).astype(np.float32)

        fairness_feats = _compute_feats(
            y_true=y_ref,
            y_demo=y_demo,
            A=A_ref,
            X=X_ref,
            metrics=metrics,
            compute_fairness_features=compute_fairness_features,
        )

        random_demos.append({
            "indices": idx_ref,
            "X": X_ref,
            "y": y_ref,
            "A": A_ref,
            "y_demo": y_demo,
            "fairness_feats": fairness_feats,
            "zero_one_loss": zero_one_loss(y_ref, y_demo),
            "source_demo_id": src_id,
            "random_demo_id": rid,
            "is_random_demo": True,
            "diagnostic_mode": "semi_random",
        })

    for d in real_demos:
        d.setdefault("is_random_demo", False)
        d.setdefault("diagnostic_mode", "semi_random_base")

    mixed = real_demos + random_demos
    rng.shuffle(mixed)
    return mixed


# ---------------------------------------------------------------------
# Pareto-clean diagnostic set
# ---------------------------------------------------------------------

def _metric_objective(actual: Mapping[str, float], target: Mapping[str, float], weights=None):
    if weights is None:
        weights = {k: 1.0 for k in target}

    return float(sum(
        weights.get(k, 1.0) * (actual[k] - target[k]) ** 2
        for k in target
        if k in actual
    ))


def _score_from_compute_fairness(
    *,
    y,
    y_demo,
    A,
    X,
    metrics,
    compute_fairness_features,
):
    ff = _compute_feats(
        y_true=y,
        y_demo=y_demo,
        A=A,
        X=X,
        metrics=metrics,
        compute_fairness_features=compute_fairness_features,
    )
    return {m: float(ff[i]) for i, m in enumerate(metrics)}, ff


def _construct_dp_error_seed(
    y,
    A,
    *,
    target_err: float,
    target_dp: float,
    seed: int = 0,
):
    """
    Construct a binary y_demo seed approximately matching error and DP.

    This is only a seed. Greedy repair improves the requested metric targets.
    """
    rng = np.random.default_rng(seed)

    y = _to_numpy_1d(y, dtype=np.float32)
    A = _get_sensitive_1d(A)

    N = len(y)
    y_demo = y.copy()

    n_flip = int(round(target_err * N))
    n_flip = min(max(0, n_flip), N)

    if n_flip == 0:
        return y_demo.astype(np.float32)

    g0 = np.where(A == 0)[0]
    g1 = np.where(A == 1)[0]

    if len(g0) == 0 or len(g1) == 0:
        idx = rng.choice(N, size=n_flip, replace=False)
        y_demo[idx] = 1.0 - y_demo[idx]
        return y_demo.astype(np.float32)

    # Pick direction: make one group more positive, the other less positive.
    hi_group, lo_group = (g1, g0) if rng.random() < 0.5 else (g0, g1)

    hi_candidates = hi_group[y_demo[hi_group] == 0]  # 0 -> 1
    lo_candidates = lo_group[y_demo[lo_group] == 1]  # 1 -> 0

    chosen = []

    # More target_dp => more structured group-opposing flips.
    structured_frac = min(1.0, max(0.0, target_dp / 0.5))
    n_structured = int(round(structured_frac * n_flip))

    n_hi = min(len(hi_candidates), n_structured // 2)
    n_lo = min(len(lo_candidates), n_structured - n_hi)

    if n_hi > 0:
        chosen.extend(rng.choice(hi_candidates, size=n_hi, replace=False).tolist())
    if n_lo > 0:
        chosen.extend(rng.choice(lo_candidates, size=n_lo, replace=False).tolist())

    remaining = n_flip - len(chosen)
    if remaining > 0:
        chosen_arr = np.asarray(chosen, dtype=int) if chosen else np.array([], dtype=int)
        unused = np.setdiff1d(np.arange(N), chosen_arr, assume_unique=False)
        extra = rng.choice(unused, size=min(remaining, len(unused)), replace=False)
        chosen.extend(extra.tolist())

    chosen = np.asarray(chosen, dtype=int)
    y_demo[chosen] = 1.0 - y_demo[chosen]
    return y_demo.astype(np.float32)


def _greedy_repair_demo(
    *,
    y,
    A,
    X,
    metrics,
    target,
    compute_fairness_features,
    seed: int,
    n_steps: int = 20_000,
    weights=None,
    preserve_error_count: bool = True,
):
    rng = np.random.default_rng(seed)

    y_np = _to_numpy_1d(y, dtype=np.float32)
    A_np = _get_sensitive_1d(A)

    target_err = float(target.get("L.ZeroOne", 0.2))
    target_dp = float(target.get("D.DP", 0.2))

    y_demo = _construct_dp_error_seed(
        y_np,
        A_np,
        target_err=target_err,
        target_dp=target_dp,
        seed=seed,
    )

    actual, ff = _score_from_compute_fairness(
        y=y,
        y_demo=y_demo,
        A=A,
        X=X,
        metrics=metrics,
        compute_fairness_features=compute_fairness_features,
    )
    best_obj = _metric_objective(actual, target, weights)

    for _ in range(n_steps):
        cand = y_demo.copy()

        if preserve_error_count and "L.ZeroOne" in target:
            wrong = np.where(cand != y_np)[0]
            right = np.where(cand == y_np)[0]

            if len(wrong) > 0 and len(right) > 0:
                i = int(rng.choice(wrong))
                j = int(rng.choice(right))
                cand[i] = 1.0 - cand[i]
                cand[j] = 1.0 - cand[j]
            else:
                i = int(rng.integers(0, len(cand)))
                cand[i] = 1.0 - cand[i]
        else:
            i = int(rng.integers(0, len(cand)))
            cand[i] = 1.0 - cand[i]

        cand_actual, cand_ff = _score_from_compute_fairness(
            y=y,
            y_demo=cand,
            A=A,
            X=X,
            metrics=metrics,
            compute_fairness_features=compute_fairness_features,
        )
        cand_obj = _metric_objective(cand_actual, target, weights)

        if cand_obj < best_obj:
            y_demo = cand
            actual = cand_actual
            ff = cand_ff
            best_obj = cand_obj

    return y_demo.astype(np.float32), np.asarray(ff, dtype=np.float32), actual, best_obj


def default_pareto_targets(
    *,
    n_demos: int = 7,
    err_low: float = 0.05,
    err_high: float = 0.35,
    dp_low: float = 0.05,
    dp_high: float = 0.35,
    metrics: Sequence[str] = ("D.DP", "L.ZeroOne"),
):
    """
    Default clean 2D frontier:
      low error / high DP  ->  high error / low DP.

    Extra metrics are left uncontrolled unless supplied through custom targets.
    """
    n_demos = int(n_demos)
    if n_demos <= 0:
        raise ValueError("n_demos must be positive.")

    errs = np.linspace(err_low, err_high, n_demos)
    dps = np.linspace(dp_high, dp_low, n_demos)

    targets = []
    for e, dp in zip(errs, dps):
        t = {}
        if "L.ZeroOne" in metrics:
            t["L.ZeroOne"] = float(e)
        if "D.DP" in metrics:
            t["D.DP"] = float(dp)
        targets.append(t)

    return targets


def make_pareto_clean_diagnostic_split(
    *,
    base_demo,
    metrics,
    compute_fairness_features,
    zero_one_loss,
    n_demos: int = 7,
    targets=None,
    seed: int | None = None,
    n_steps: int = 20_000,
    weights=None,
    verbose: bool = True,
):
    """
    Build artificial demos that approximately lie on a clean Pareto frontier.

    Each output demo shares X/y/A from base_demo but has a synthetic y_demo.
    By default, the frontier trades off:
      L.ZeroOne increasing, D.DP decreasing.

    For more metrics, pass explicit `targets`, e.g.
      [
        {"L.ZeroOne": 0.10, "D.DP": 0.30, "D.EqOdds": 0.25},
        ...
      ]
    """
    if base_demo is None:
        raise ValueError("base_demo cannot be None.")

    seed = 0 if seed is None else int(seed)

    y = base_demo["y"]
    A = base_demo["A"]
    X = base_demo.get("X", None)

    if targets is None:
        targets = default_pareto_targets(n_demos=n_demos, metrics=metrics)

    out = []
    if verbose:
        print("\n🧪 Generating Pareto-clean diagnostic demos")
        print(f"   Total demos: {len(targets)}")
        print("   Targets:")

        for i, t in enumerate(targets):
            t_str = ", ".join(f"{k}={v:.3f}" for k, v in t.items())
            print(f"     [{i+1}] {t_str}")
        print()
    for i, target in enumerate(targets):
        if verbose:
            print(f"▶ Making demo {i+1}/{len(targets)}", end="", flush=True)
        y_demo, fairness_feats, actual, obj = _greedy_repair_demo(
            y=y,
            A=A,
            X=X,
            metrics=metrics,
            target=target,
            compute_fairness_features=compute_fairness_features,
            seed=seed + i,
            n_steps=n_steps,
            weights=weights,
            preserve_error_count=True,
        )

        d = {
            "indices": base_demo.get("indices", None),
            "X": X,
            "y": y,
            "A": A,
            "y_demo": y_demo,
            "fairness_feats": fairness_feats,
            "zero_one_loss": zero_one_loss(y, y_demo),
            "is_artificial_frontier_demo": True,
            "is_random_demo": False,
            "diagnostic_mode": "pareto_clean",
            "pareto_target": dict(target),
            "pareto_actual": dict(actual),
            "pareto_objective": float(obj),
            "pareto_id": i,
        }
        out.append(d)
        if verbose:
            actual_str = ", ".join(f"{k}={actual[k]:.3f}" for k in target if k in actual)
            target_str = ", ".join(f"{k}={target[k]:.3f}" for k in target)

            print(f" | target: [{target_str}] → actual: [{actual_str}]")

    return out
