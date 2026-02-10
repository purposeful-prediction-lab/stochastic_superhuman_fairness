from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional
from stochastic_superhuman_fairness.core.fairness.compute_fairness_utils import compute_fairness_features
import torch

@dataclass
class RolloutBatch:
    probs: List[torch.Tensor]
    y_hat: List[torch.Tensor]
    feats: torch.Tensor
    aux_list: Optional[List[Dict[str, Any]]] = None  # optional extension
    logits: Optional[List[torch.Tensor]] = None  # optional
# ========================================================================================================================
# Collection Functions
# ========================================================================================================================
def collect_rollouts(
    *,
    policy,
    demonstrator,
    demos,
    metrics_list,
    sample_actions_fn,                 # (probs, threshold) -> y_hat
    decision_threshold: Optional[float] = None,
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

    with ctx:
        for i, d in enumerate(demos):
            if before_demo is not None:
                aux_list.append(before_demo(i, d))

            Xd = d["X"].to(device)
            yd = demonstrator.get_targets(d).to(device)
            Ad = d["A"]
            Ad = Ad.to(device) if torch.is_tensor(Ad) else Ad

            logits = policy(Xd).squeeze(-1)
            probs = torch.sigmoid(logits)

            y_hat = sample_actions_fn(probs, threshold=decision_threshold)
            f_r = compute_fairness_features(Xd, yd, y_hat, Ad, metrics_list)

            if detach_outputs:
                probs = probs.detach()
                y_hat = y_hat.detach()
                f_r = f_r.detach()

            probs_list.append(probs)
            yhat_list.append(y_hat)
            feats_list.append(f_r)
            if return_logits:
                logits_list.append(logits)

    feats = torch.stack(feats_list, dim=0)
    return RolloutBatch(probs=probs_list, y_hat=yhat_list, feats=feats, aux_list=aux_list)

@torch.no_grad()
def collect_bayesian_rollouts(
    self,
    demonstrator,
    *,
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

    rb = collect_rollouts(
        policy=self.policy,
        demonstrator=demonstrator,
        demos=demos,
        metrics_list=self.metrics_list,
        sample_actions_fn=self._sample_actions_from_probs,
        decision_threshold=decision_threshold,
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
