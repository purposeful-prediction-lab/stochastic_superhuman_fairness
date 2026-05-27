import torch
import numpy as np
import mosek
import matplotlib.pyplot as plt
from typing import Literal, Union
from stochastic_superhuman_fairness.core.gamma_utils import  smooth_gamma
# -----------------------------------------------------------------------------
# Utils
# -----------------------------------------------------------------------------

def minmax_normalize(x, eps=1e-12):
    x = np.asarray(x, dtype=float)
    lo, hi = x.min(), x.max()
    return (x - lo) / (hi - lo + eps)


def _normalize_marginal(x, n, name):
    if x is None:
        return np.ones(n) / n

    x = np.asarray(x, dtype=float).reshape(-1)
    if x.size != n:
        raise ValueError(f"{name} length {x.size} != {n}")

    x = np.clip(x, 0.0, None)
    s = x.sum()
    if s <= 0:
        raise ValueError(f"{name} must have positive sum")

    return x / s


def _prepare_cost(S, normalize_s_matrix, tau):
    S = np.asarray(S, dtype=float)
    if normalize_s_matrix:
        S = minmax_normalize(S)
    return S * tau

def bj_from_beatrates_nocollapse(
    beat_rates,
    tau: float = 5.0,
    uniform_mix: float = 0.0,    # lambda
    b_min: float = 0.0,          # e.g. 1e-3 / D
    center: str = "half",        # "none" | "mean" | "half"
    eps: float = 1e-12,
    rank_type: Literal['rev_ranking', 'ranking', None] = 'ranking',
)-> Union[None, np.ndarray]:

    if rank_type is None:
        return None
    br = np.asarray(beat_rates, dtype=float)

    if center == "mean":
        x = br - br.mean()
    elif center == "half":
        x = br - 0.5
    else:
        x = br
    sign = -1 if rank_type == 'rev_ranking' else 1
    w = np.exp(sign*tau * x)
    b = w / (w.sum() + eps)

    # uniform mixing
    if uniform_mix > 0:
        D = len(b)
        b = (1 - uniform_mix) * b + uniform_mix * (np.ones(D) / D)

    # floor
    if b_min > 0:
        b = np.maximum(b, b_min)
        b = b / (b.sum() + eps)

    return b


# -----------------------------------------------------------------------------
# MOSEK QP Solver
# -----------------------------------------------------------------------------
# ============================================================
# 🔹 MOSEK QP solver  (samples × demos)
# ============================================================
def _solve_qp_mosek_core(
    subdom_matrix: np.ndarray,
    rollout_marginals: np.ndarray | list = None,
    demo_marginals: np.ndarray | list = None,
    lambda_reg: float | None = None,
    normalize_s_matrix: bool = False,
    tau: float = 1.0,
    verbose: bool = True,
    row_constraints: bool = False,
    col_constraints: bool = True,
):
    """
    Solve QP:
        minimize_γ   <γ, S> + (λ/2)||γ||²
        s.t.          γ ≥ 0,
                      (optional) γ 1 = p_samples
                      (optional) γᵀ 1 = q_demos
    """

    # --- Input preparation ---
    S = np.asarray(subdom_matrix, dtype=float)
    num_samples, num_demos = S.shape
    n = num_samples * num_demos
    # --- Scaling ---
    if normalize_s_matrix:
        S = minmax_normalize(S)
    if tau != 1.0:
        S = S * tau
    if lambda_reg is None:
        lambda_reg = 1e-6
    #  import ipdb;ipdb.set_trace()
    # Reduce lampbda in unconstrained cases
    if not (row_constraints and col_constraints):
        lambda_reg = 0.0
        #  lambda_reg = 1e-12

    if verbose:
        print(f"[MOSEK] Solving QP with λ={lambda_reg:.1e}, τ={tau}, "
              f"rows={row_constraints}, cols={col_constraints}")
    # --- Marginals ---
    if rollout_marginals is None:
        p_samples = np.ones(num_samples) / num_samples
    else:
        p_samples = np.asarray(rollout_marginals, dtype=float)
        p_samples /= p_samples.sum()

    if demo_marginals is None:
        q_demos = np.ones(num_demos) / num_demos
    else:
        q_demos = np.asarray(demo_marginals, dtype=float)
        q_demos /= q_demos.sum()

    # --- MOSEK setup ---
    with mosek.Env() as env, env.Task(0, 0) as task:
        task.putobjsense(mosek.objsense.minimize)
        #  task.putintparam(mosek.iparam.optimizer, mosek.optimizertype.primal_simplex)
        task.putdouparam(mosek.dparam.intpnt_co_tol_rel_gap, 1e-8)

        # Variables γ_ij ≥ 0
        task.appendvars(n)
        task.putvarboundslice(0, n,
                              [mosek.boundkey.lo]*n,
                              [0.0]*n,
                              [float("inf")]*n)

        # --- Constraints ---
        constraint_idx = 0
        task.appendcons(1)
        idx = list(range(n))
        task.putarow(constraint_idx, idx, [1.0]*n)
        task.putconbound(constraint_idx, mosek.boundkey.fx, 1.0, 1.0)
        constraint_idx += 1
        # Row constraints (samples)
        if row_constraints:
            task.appendcons(num_samples)
            for i in range(num_samples):
                idx = [i*num_demos + j for j in range(num_demos)]
                task.putarow(constraint_idx + i, idx, [1.0]*num_demos)
                task.putconbound(constraint_idx + i, mosek.boundkey.fx,
                                 p_samples[i], p_samples[i])
            constraint_idx += num_samples

        # Column constraints (demos)
        if col_constraints:
            task.appendcons(num_demos)
            for j in range(num_demos):
                idx = [i*num_demos + j for i in range(num_samples)]
                task.putarow(constraint_idx + j, idx, [1.0]*num_samples)
                task.putconbound(constraint_idx + j, mosek.boundkey.fx,
                                 q_demos[j], q_demos[j])
            constraint_idx += num_demos

        # Linear objective
        task.putcslice(0, n, S.flatten())

        # Optional quadratic regularization
        if lambda_reg > 0:
            qsubi = qsubj = list(range(n))
            qval = [lambda_reg] * n
            task.putqobj(qsubi, qsubj, qval)

        # --- Solve ---
        # print("Constraints added:", task.getnumcon())
        # print("Expect:", (num_samples if row_constraints else 0) + (num_demos if col_constraints else 0))
        task.optimize()
        solsta = task.getsolsta(mosek.soltype.itr)
        if solsta != mosek.solsta.optimal:
            raise RuntimeError(f"[MOSEK] optimization failed: {solsta}")

        # --- Extract primal solution ---
        gamma = np.zeros(n)
        task.getxx(mosek.soltype.itr, gamma)
        gamma_matrix = gamma.reshape((num_samples, num_demos))
        if verbose:
            if (gamma < 0 ).any():
                print("Some gamma matrix values were negative; rectifying to 0. Numerical error in very small values? (<1e-6)")
                gamma_matrix = np.clip(gamma_matrix, a_min=0.0, a_max=None)
        # print("γ column sums:", gamma_matrix.sum(axis=0))
        # print("Target q:", q_demos)

        # --- Extract duals safely ---
        duals = np.zeros(constraint_idx)
        try:
            task.gety(mosek.soltype.itr, duals)
        except Exception:
            duals[:] = 0.0

        # Partition duals based on which constraints are active
        dual_rows = np.zeros(num_samples)
        dual_cols = np.zeros(num_demos)
        offset = 1
        #  offset = 0
        #  if row_constraints:
        #      dual_rows = duals[offset:offset + num_samples]
        #      offset += num_samples
        if col_constraints:
            dual_cols = duals[offset:offset + num_demos]


    # --- Diagnostics ---
    row_sum = gamma_matrix.sum(axis=1)
    col_sum = gamma_matrix.sum(axis=0)
    if verbose:
        print(f"γ mean/std: {gamma_matrix.mean():.6f} / {gamma_matrix.std():.6f}")
        print(f"γ row-sum std: {row_sum.std():.3e}, col-sum std: {col_sum.std():.3e}")

    return gamma_matrix, dual_rows, dual_cols

# -----------------------------------------------------------------------------
# Sinkhorn
# -----------------------------------------------------------------------------

def _solve_sinkhorn_core(
    subdom_matrix,
    rollout_marginals=None,
    demo_marginals=None,
    normalize_s_matrix=True,
    tau=1.0,
    epsilon=0.05,
    max_iters=1000,
    tol=1e-8,
    verbose=True,
    row_constraints=True,
    col_constraints=True,
):
    S = _prepare_cost(subdom_matrix, normalize_s_matrix, tau)
    R, D = S.shape

    p = _normalize_marginal(rollout_marginals, R, "rollout")
    q = _normalize_marginal(demo_marginals, D, "demo")

    K = np.exp(-S / max(epsilon, 1e-12)) + 1e-300

    u = np.ones(R)
    v = np.ones(D)

    for _ in range(max_iters):
        if row_constraints:
            u = p / (K @ v + 1e-12)
        if col_constraints:
            v = q / (K.T @ u + 1e-12)

    gamma = (u[:, None] * K) * v[None, :]

    dual_rows = epsilon * np.log(u + 1e-300)
    dual_cols = epsilon * np.log(v + 1e-300)

    return gamma, dual_rows, dual_cols


# -----------------------------------------------------------------------------
# Unified Solver
# -----------------------------------------------------------------------------

def solve_qp_superhuman(
    subdom_matrix,
    rollout_marginals=None,
    demo_marginals=None,
    solver="mosek",
    lambda_reg=None,
    normalize_s_matrix=True,
    tau=1.0,
    epsilon=0.05,
    row_constraints=True,
    col_constraints=True,
    verbose=True,
):
    solver = solver.lower()

    if solver == "mosek":
        return _solve_qp_mosek_core(
            subdom_matrix,
            rollout_marginals,
            demo_marginals,
            lambda_reg,
            normalize_s_matrix,
            tau,
            verbose,
            row_constraints,
            col_constraints,
        )

    elif solver == "sinkhorn":
        return _solve_sinkhorn_core(
            subdom_matrix,
            rollout_marginals,
            demo_marginals,
            normalize_s_matrix,
            tau,
            epsilon,
            verbose=verbose,
            row_constraints=row_constraints,
            col_constraints=col_constraints,
        )

    else:
        raise ValueError(f"Unknown solver {solver}")


# -----------------------------------------------------------------------------
# Diagnostics
# -----------------------------------------------------------------------------

def cost_gap_diagnostics(S, k=2, eps=1e-12):
    # S: [R, D], lower is better
    sorted_S = np.sort(S, axis=0)
    best = sorted_S[0]
    second = sorted_S[1] if S.shape[0] > 1 else sorted_S[0]
    gap = second - best

    return {
        "mean_best": float(best.mean()),
        "mean_gap": float(gap.mean()),
        "median_gap": float(np.median(gap)),
        "frac_near_tie_1e_4": float((gap < 1e-4).mean()),
        "frac_near_tie_1e_3": float((gap < 1e-3).mean()),
        "frac_near_tie_1e_2": float((gap < 1e-2).mean()),
        "gap_over_best": float((gap / (np.abs(best) + eps)).mean()),
    }

def gamma_objective_and_stability(
    gamma,
    cost,
    *,
    prev_gamma=None,
    eps=1e-12,
    support_eps=1e-9,
):
    gamma = np.asarray(gamma, dtype=float)
    cost = np.asarray(cost, dtype=float)

    if gamma.shape != cost.shape:
        raise ValueError(f"gamma and cost shape mismatch: {gamma.shape} vs {cost.shape}")

    row_mass = gamma.sum(axis=1)
    col_mass = gamma.sum(axis=0)

    out = {
        "objective": float((gamma * cost).sum()),
        "support_size": int((gamma > support_eps).sum()),
        "top1": np.argmax(gamma, axis=0).tolist(),  # best rollout per demo
        "row_mass_std": float(row_mass.std()),
        "col_mass_std": float(col_mass.std()),
        "total_mass": float(gamma.sum()),
    }

    if prev_gamma is not None:

        prev_gamma = np.asarray(prev_gamma, dtype=float)

        if prev_gamma.shape != gamma.shape:
            raise ValueError(
                f"prev_gamma shape mismatch: {prev_gamma.shape} vs {gamma.shape}"
            )

        diff = gamma - prev_gamma

        abs_delta = float(np.linalg.norm(diff))
        rel_delta = float(abs_delta / (np.linalg.norm(prev_gamma) + eps))

        # per-demo deltas, useful for your plots
        abs_delta_per_demo = np.linalg.norm(diff, axis=0)
        prev_norm_per_demo = np.linalg.norm(prev_gamma, axis=0)
        rel_delta_per_demo = abs_delta_per_demo / (prev_norm_per_demo + eps)

        prev_top1 = np.argmax(prev_gamma, axis=0)
        curr_top1 = np.argmax(gamma, axis=0)
        top1_flip = curr_top1 != prev_top1

        out.update({
            "abs_delta": abs_delta,
            "rel_delta": rel_delta,
            "abs_delta_per_demo": abs_delta_per_demo.tolist(),
            "rel_delta_per_demo": rel_delta_per_demo.tolist(),
            "top1_flip_rate": float(top1_flip.mean()),
            "top1_flips": top1_flip.astype(int).tolist(),
        })

    else:
        out.update({
            "abs_delta": 0.0,
            "rel_delta": 0.0,
            "abs_delta_per_demo": [0.0] * gamma.shape[1],
            "rel_delta_per_demo": [0.0] * gamma.shape[1],
            "top1_flip_rate": 0.0,
            "top1_flips": [0] * gamma.shape[1],
        })

    return out

def compute_weighted_flip(gamma, prev_gamma):
    curr_top1 = np.argmax(gamma, axis=0)
    prev_top1 = np.argmax(prev_gamma, axis=0)

    flips = curr_top1 != prev_top1

    # weight by confidence gap
    curr_max = gamma.max(axis=0)
    prev_max = prev_gamma.max(axis=0)

    weight = np.abs(curr_max - prev_max)

    return (flips * weight).mean()

# -----------------------------------------------------------------------------
# Main API (fixed preprocessing)
# -----------------------------------------------------------------------------

def solve_stochastic_subdom_coupling(
    S,
    solver="mosek",
    prev_gamma = None,
    rollout_marginals=None,
    demo_marginals=None,
    normalize_s_matrix=True,
    S_tau=1.0,
    lambda_reg=None,
    epsilon=0.05,
    row_constraints=True,
    col_constraints=True,
    gamma_smoothing_ema: float  = None,
    verbose=False,
):
    if torch.is_tensor(S):
        device = S.device
        S_np = S.detach().cpu().numpy()
        backend = "torch"
    else:
        S_np = np.asarray(S, dtype=float)
        backend = "numpy"
        device = None

    gamma_np, dual_r, dual_c = solve_qp_superhuman(
        S_np,
        rollout_marginals,
        demo_marginals,
        solver,
        lambda_reg,
        normalize_s_matrix,
        S_tau,
        epsilon,
        row_constraints,
        col_constraints,
        verbose,
    )
    
    #  import ipdb;ipdb.set_trace()
    if gamma_smoothing_ema is not None:
        gamma_np = smooth_gamma(gamma_np, prev_gamma, ema = gamma_smoothing_ema)

    diag = gamma_objective_and_stability(gamma_np, S_np, prev_gamma = prev_gamma)
    S_diagnostics = cost_gap_diagnostics(S_np)

    if backend == "torch":
        gamma_t = torch.tensor(gamma_np, device=device, dtype=torch.float32)
    else:
        gamma_t = None

    return {
        "gamma_np": gamma_np,
        "gamma_torch": gamma_t,
        "dual_rows_np": dual_r,
        "dual_cols_np": dual_c,
        "gamma_diagnostics": diag,
        "S_diagnostics": S_diagnostics,
    }
