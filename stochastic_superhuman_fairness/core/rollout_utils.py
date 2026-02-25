from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Union, Sequence
from stochastic_superhuman_fairness.core.fairness.fairness_metrics import zero_one_loss, compute_fairness_features
from stochastic_superhuman_fairness.core.utils import sample_actions_from_policy
import numpy as np
import warnings
import torch

@dataclass
class RolloutBatch:
    probs: List[torch.Tensor]
    y_hat: List[torch.Tensor]
    feats: torch.Tensor
    zero_one_losses: List[torch.Tensor] = None
    batch_zero_one_loss: float = None
    aux_list: Optional[List[Dict[str, Any]]] = None  # optional extension
    logits: Optional[List[torch.Tensor]] = None  # optional

# ========================================================================================================================
# Collection Functions
# ========================================================================================================================
def _normalize_policies(policy):
    # single module (but allow wrapper with .policies)
    if isinstance(policy, torch.nn.Module) and not isinstance(policy, torch.nn.ModuleList):
        if hasattr(policy, "policies"):
            ps = getattr(policy, "policies")
            if isinstance(ps, (list, tuple, torch.nn.ModuleList)):
                return list(ps)
        return [policy]

    if isinstance(policy, (list, tuple, torch.nn.ModuleList)):
        return list(policy)

    if hasattr(policy, "policies"):
        ps = getattr(policy, "policies")
        if isinstance(ps, (list, tuple, torch.nn.ModuleList)):
            return list(ps)

    raise TypeError(f"Unsupported policy type: {type(policy)}")

def collect_rollouts(
    *,
    policy,
    demonstrator,
    demos,
    metrics_list,
    n_rollouts: int = None,
    decision_threshold: float | None = 0.5,
    demo_idxs=None,
    before_demo=None,
    require_grad: bool = False,
    detach_outputs: bool = True,
    stochastic: bool = True,
    return_logits: bool = False,
    sample_actions_fn=None,   # <--- kwarg; if None and stochastic=True, use model.sample_actions
):
    """
    - If stochastic=True:
        - if sample_actions_fn is not None: use it. it must have the same signature as the base model's.
        - else: use pol.sample_actions(probs, threshold=decision_threshold)
    - If stochastic=False:
        y_hat = (probs > decision_threshold).float()

    Supports single policy or ensemble (cycled). Appends policy_perf to aux_list[-1].
    """

    policies = _normalize_policies(policy)
    m = len(policies)
    device = next(policies[0].parameters()).device

    d_idxs = np.arange(len(demos)) if demo_idxs is None else np.asarray(demo_idxs, dtype=int).reshape(-1)
    N = int(n_rollouts) if n_rollouts is not None else len(d_idxs)

    if m > 1 and (N % m != 0):
        warnings.warn(
            f"collect_rollouts: n_rollouts={N} not divisible by #policies={m}. Policies will be cycled."
        )

    if N != len(d_idxs):
        reps = int(np.ceil(N / len(d_idxs)))
        d_idxs = np.tile(d_idxs, reps)[:N]

    ctx = torch.enable_grad() if require_grad else torch.no_grad()

    probs_list, yhat_list, feats_list = [], [], []
    logits_list = [] if return_logits else None
    aux_list = []
    zero_one_losses = []

    pol_err = np.zeros(m, dtype=float)
    pol_count = np.zeros(m, dtype=float)
    total_err, total_count = 0.0, 0.0

    with ctx:
        for t, demo_i in enumerate(d_idxs):
            pol_id = t % m
            pol = policies[pol_id]
            d = demos[int(demo_i)]

            aux = (before_demo(int(demo_i), d) or {}) if before_demo is not None else {}
            aux["policy_id"] = int(pol_id)
            aux["demo_idx"] = int(demo_i)
            aux_list.append(aux)

            Xd = d["X"].to(device)
            yt = d["y"].to(device).view(-1)  # ground truth
            yd = yt if "y_demo" not in d else d["y_demo"].to(device).view(-1)  # demo labeling if present

            Ad = d["A"]
            Ad = Ad.to(device) if torch.is_tensor(Ad) else Ad

            # --- decisions ---
            #  import ipdb;ipdb.set_trace()
            if not stochastic:
                if decision_threshold is None:
                        raise ValueError("decision_threshold must be set for deterministic collection.")
            decision_threshold = decision_threshold if not stochastic else None

            y_hat, logits, probs = sample_actions_from_policy(pol, Xd, decision_threshold = decision_threshold, return_logits = True, return_probs = True, require_grad = require_grad)

            # fairness features (usually non-diff; ok)
            f_r = compute_fairness_features(yd, y_hat, Ad, metrics_list)

            # zero-one vs ground truth (logging only)
            mism = (y_hat.view(-1) != yt).float().sum().item()
            n_el = float(yt.numel())
            pol_err[pol_id] += mism
            pol_count[pol_id] += n_el
            total_err += mism
            total_count += n_el
            zero_one_losses.append(mism / max(1.0, n_el))

            if detach_outputs:
                probs = probs.detach()
                y_hat = y_hat.detach()
                f_r = f_r.detach()
                if return_logits:
                    logits = logits.detach()

            probs_list.append(probs)
            yhat_list.append(y_hat)
            feats_list.append(f_r)
            if return_logits:
                logits_list.append(logits)

    batch_zero_one = total_err / max(1.0, total_count)
    feats = torch.stack(feats_list, dim=0)

    policy_perf = {f"policy_{i}/zero_one": float(pol_err[i] / max(1.0, pol_count[i])) for i in range(m)}
    policy_perf["batch/zero_one"] = float(batch_zero_one)
    policy_perf["n_rollouts"] = int(N)
    policy_perf["n_policies"] = int(m)
    aux_list.append({"policy_perf": policy_perf})

    return RolloutBatch(
        probs=probs_list,
        y_hat=yhat_list,
        feats=feats,
        zero_one_losses=zero_one_losses,
        batch_zero_one_loss=float(batch_zero_one),
        aux_list=aux_list,
        logits=logits_list,
    )

#------------------------------------------------------------------------------------------------

def collect_rollouts_old(
    *,
    policy,
    demonstrator,
    demos,
    metrics_list,
    sample_actions_fn,                 # (probs, threshold) -> y_hat
    n_rollouts:int = None,
    decision_threshold: Optional[float] = None,
    demo_idxs: Union[List, np.ndarray] = None,
    before_demo: Optional[Callable[[int, Dict[str, Any]], Dict[str, Any]]] = None,
    require_grad: bool = False,
    detach_outputs: bool = True,
    return_logits: bool = False,
):
    """
    Collect rollouts for a fixed policy over a set of demos.

    For each demo:
      - runs the policy forward to obtain probabilities
      - samples decisions from those probabilities
      - computes fairness features

    This function is distribution-agnostic. It supports optional
    extension via the `before_demo` hook, which can mutate the policy
    (e.g. by loading sampled parameters) and return auxiliary data
    stored per rollout.

    Args:
        policy: Torch module used for forward passes.
        demonstrator: Provides targets and sensitive attributes.
        demos: Iterable of demo dicts with keys {"X", "A", ...}.
        metrics_list: Fairness metrics to compute per rollout.
        sample_actions_fn: Function mapping probabilities to actions.
        decision_threshold: Optional threshold for deterministic decisions.
        demo_idxs: Optional demo idxs to get Xs from and pass through policy.
        before_demo: Optional callback executed before each rollout.
                     Should return auxiliary info (or None).
        require_grad: Whether grad computation is enabled or not (i.e for eval or inference)

    Returns:
        RolloutBatch with:
          - probs: list of probability tensors per demo
          - y_hat: list of sampled decision tensors per demo
          - feats: stacked fairness features [R, K]
          - aux_list: None if before_demo is None, else list of aux dicts
    """
    ctx = torch.enable_grad() if require_grad else torch.no_grad()
    device = next(policy.parameters()).device

    probs_list, yhat_list, feats_list = [], [], []
    aux_list = [] if before_demo is not None else None
    logits_list = [] if return_logits else None
    zero_one_list = []
    total_zero_one, total_count = 0., 0.
    d_idxs = np.arange(len(demos))if demo_idxs is None else demo_idxs 
    n_rollouts = int(n_rollouts) if n_rollouts is not None else len(demos)
    
    with ctx:
        for i in d_idxs:
        #  for i, d in enumerate(demos):
            d = demos[i]
            if before_demo is not None:
                aux_list.append(before_demo(i, d))

            Xd = d["X"].to(device)
            #  yd = demonstrator.get_targets(d).to(device)
            yt = d['y'].to(device) # ground truth data
            yd = yt if 'y_demo' not in d.keys() else d['y_demo'].to(device)
            Ad = d["A"]
            Ad = Ad.to(device) if torch.is_tensor(Ad) else Ad

            logits = policy(Xd).squeeze(-1)
            probs = torch.sigmoid(logits)

            y_hat = sample_actions_fn(probs, threshold=decision_threshold)
            f_r = compute_fairness_features(yd, y_hat, Ad, metrics_list)
            import ipdb;ipdb.set_trace()
            # ---- ZERO-ONE LOSS ----
            pred_loss = zero_one_loss(yt, y_hat)
            zero_one = (y_hat != yt).float().sum()
            #  import ipdb;ipdb.set_trace()
            total_zero_one += zero_one.item()
            total_count += yt.numel()

            if detach_outputs:
                probs = probs.detach()
                y_hat = y_hat.detach()
                f_r = f_r.detach()
                zone_d = zero_one.detach()

            probs_list.append(probs)
            yhat_list.append(y_hat)
            feats_list.append(f_r)
            zero_one_list.append(zone_d / yt.numel())

            if return_logits:
                logits_list.append(logits)

    batch_zero_one = total_zero_one / max(1, total_count) # zero one loss for the entire batch
    feats = torch.stack(feats_list, dim=0)
    return RolloutBatch(probs=probs_list, y_hat=yhat_list, feats=feats, aux_list=aux_list, zero_one_losses = zero_one_list, batch_zero_one_loss = batch_zero_one)

@torch.no_grad()
def collect_bayesian_rollouts(
    self,
    demonstrator,
    *,
    n_rollouts: int = None,
    demos=None,
    dist_mode="per_param_diag",
    decision_threshold=None,
    init_var=1e-4,
    restore_policy_to_mean=True,
):
    """
    Collect rollouts under a Bayesian (stochastic-parameter) policy.

    For each demo:
      - samples a parameter draw from the current parameter distribution
      - loads the sampled parameters into the live policy
      - runs a rollout using the base rollout collector
      - records distribution-specific auxiliary data (e.g. eps or theta_vec)

    This is a thin wrapper around `collect_rollouts_base` that injects
    parameter sampling via the `before_demo` hook and optionally restores
    the policy to its mean parameters after collection.

    Args:
        demonstrator: Provides targets and sensitive attributes.
        demos: Optional list of demos (defaults to training demos).
        dist_mode: Distribution mode ("per_param_diag" or "full_param_mvn").
        n_rollouts: how many rollouts to collect. None = num_demos.
        decision_threshold: Optional threshold for action sampling.
        init_if_needed: Whether to initialize the parameter distribution.
        init_var: Initial variance used if distribution is initialized.
        restore_policy_to_mean: Whether to reset the policy to mean params
                                after all rollouts.

    Returns:
        RolloutBatch with populated aux_list containing
        distribution-specific data for each rollout.
    """

    if demos is None:
        demos = demonstrator.train_demos

    if getattr(self, "dist_initialized", 0) == 0:
        self.init_dist(dist_mode, init_var=init_var)

    def before_demo(i, d):
        theta, aux = self.sample_dist(dist_mode)
        self.load_params_into_policy(theta)
        aux["dist_mode"] = dist_mode
        return aux

    demo_idxs = sample_demo_idxs(len(demos), n_rollouts = n_rollouts, mode = 'cycle')
    rb = collect_rollouts(
        policy=self.policy,
        demonstrator=demonstrator,
        demos=demos,
        metrics_list=self.metrics_list,
        sample_actions_fn=self._sample_actions_from_probs,
        decision_threshold=decision_threshold,
        demo_idxs = demo_idxs,
        before_demo=before_demo,
        require_grad=False,
        detach_outputs=True,
    )

    if restore_policy_to_mean:
        if dist_mode == "per_param_diag" and hasattr(self, "per_param_mean"):
            self.load_params_into_policy(self.per_param_mean)
        elif dist_mode == "full_param_mvn" and hasattr(self, "full_mu"):
            theta_mean = self._unflatten_to_theta_dict(self.full_mu, self.full_template, self.full_names)
            self.load_params_into_policy(theta_mean)

    return rb


def sample_demo_idxs(
    n_demos: int,
    n_rollouts: Optional[int] = None,
    mode: str = "one_per_demo",          # "one_per_demo" | "cycle" | "stochastic"
    probs: Optional[Sequence[float]] = None,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """
    Returns an int array of demo indices of length n_rollouts (or n_demos if n_rollouts is None).

    Rules:
      - if n_rollouts is None: return [0,1,...,n_demos-1] (one per demo)
      - mode="cycle": repeat 0..n_demos-1 until length n_rollouts
      - mode="stochastic": sample with replacement using probs (default uniform)
    """
    if rng is None:
        rng = np.random.default_rng()

    if n_demos <= 0:
        raise ValueError("n_demos must be > 0")

    if n_rollouts is None:
        return np.arange(n_demos, dtype=int)

    if n_rollouts <= 0:
        raise ValueError("n_rollouts must be > 0")

    mode = str(mode).lower()

    if mode in ("one_per_demo", "one", "default"):
        # if user explicitly requests a number, fall back to cycle behavior
        base = np.arange(n_demos, dtype=int)
        reps = int(np.ceil(n_rollouts / n_demos))
        return np.tile(base, reps)[:n_rollouts]

    if mode == "cycle":
        base = np.arange(n_demos, dtype=int)
        reps = int(np.ceil(n_rollouts / n_demos))
        return np.tile(base, reps)[:n_rollouts]

    if mode == "stochastic":
        if probs is None:
            p = None  # numpy uses uniform when p=None
        else:
            p = np.asarray(probs, dtype=float).reshape(-1)
            if p.size != n_demos:
                raise ValueError(f"probs must have length n_demos={n_demos}, got {p.size}")
            s = p.sum()
            if not np.isfinite(s) or s <= 0:
                raise ValueError("probs must sum to a positive finite value")
            p = p / s
        return rng.choice(n_demos, size=n_rollouts, replace=True, p=p).astype(int)

    raise ValueEgtrror(f"Unknown mode: {mode}")
