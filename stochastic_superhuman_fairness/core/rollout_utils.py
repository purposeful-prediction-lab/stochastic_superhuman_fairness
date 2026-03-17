from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Union, Sequence
from stochastic_superhuman_fairness.core.fairness.fairness_metrics import zero_one_loss, compute_fairness_features
from stochastic_superhuman_fairness.core.utils import sample_actions_from_policy
from stochastic_superhuman_fairness.core.utils import sample_binary_from_probs
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

    def get_rollout_policy_idxs(self):
        #  if not hasattr(self, 'rollout_policy_idxs'):
        self.rollout_policy_idxs = [r['policy_id'] for r in self.aux_list[:-1]]
        return self.rollout_policy_idxs

    def get_rollout_groupings_by_policy(self):
        """
        Convert group-id vector into row index groups.

        Example
        -------
        [1,1,1,1,1,1,2,2,3,4]
        -> [[0,1,2,3,4,5], [6,7], [8], [9]]
        """
        ids = self.get_rollout_policy_idxs()
        ids = np.asarray(ids).reshape(-1)

        groups = []
        for g in np.unique(ids):
            groups.append(np.where(ids == g)[0].tolist())

        return groups

    def feats_per_policy(self, as_numpy: bool = False):
        """
        Groups rollout features by policy_id.

        Args:
            as_numpy: if True, returns np.ndarray instead of torch.Tensor

        Returns:
            {
                policy_id: (n_i, K) tensor or ndarray
            }
        """

        if self.aux_list is None:
            raise ValueError("aux_list required to group feats by policy.")

        if self.feats.ndim != 2:
            raise ValueError("feats must be 2D (T, K).")

        T = self.feats.shape[0]

        # collect policy ids aligned with rollout rows
        policy_ids = []
        for d in self.aux_list[:T]:  # ignore trailing summary entry
            if not isinstance(d, dict) or "policy_id" not in d:
                raise ValueError("Each aux entry must contain 'policy_id'.")
            policy_ids.append(int(d["policy_id"]))

        policy_ids = torch.tensor(policy_ids, device=self.feats.device)

        out = {}
        for pid in torch.unique(policy_ids):
            mask = policy_ids == pid
            feats_group = self.feats[mask]

            if as_numpy:
                feats_group = feats_group.detach().cpu().numpy()

            out[int(pid.item())] = feats_group

        return out

    #------------

    def feats_by_mode(self, as_numpy: bool = False)->List[Union[torch.Tensor, np.ndarray]]:
        """
        Returns rollout feats grouped by policy_id (mode).

        Output shape conceptually:
            [m][r_m][K]

        Args:
            as_numpy: if True return numpy arrays inside lists

        Returns:
            List[List[Tensor or ndarray]]
        """

        if self.aux_list is None:
            raise ValueError("aux_list required to group rollouts by mode.")

        if self.feats.ndim != 2:
            raise ValueError("feats must be 2D (T, K).")

        T, K = self.feats.shape

        # collect policy_ids aligned with feats rows
        policy_ids = []
        for d in self.aux_list[:T]:   # ignore trailing summary
            if not isinstance(d, dict) or "policy_id" not in d:
                raise ValueError("Each aux entry must contain 'policy_id'.")
            policy_ids.append(int(d["policy_id"]))

        policy_ids = np.asarray(policy_ids, dtype=int)
        unique_ids = sorted(np.unique(policy_ids))

        result = []

        for pid in unique_ids:
            mask = policy_ids == pid
            feats_group = self.feats[mask]

            if as_numpy:
                feats_group = feats_group.detach().cpu().numpy() if torch.is_tensor(feats_group) else np.asarray(feats_group)

            # convert to list of rollouts
            #  result.append([feats_group[i] for i in range(feats_group.shape[0])])
            result.append(feats_group)

        return result

    def overall_feature_mean(self, as_numpy: bool = False):
        """
        Computes aggregate mean over ALL rollouts for each feature dimension.

        Returns:
            Tensor (K,) or ndarray (K,)
        """

        X = self.feats

        # convert to numpy if requested later
        if isinstance(X, (list, tuple)):
            # list of (r_m, K)
            if len(X) == 0:
                raise ValueError("No rollout data.")
            X_cat = []
            for Xm in X:
                Xm_np = Xm.detach().cpu().numpy() if torch.is_tensor(Xm) else np.asarray(Xm)
                X_cat.append(Xm_np)
            X_all = np.concatenate(X_cat, axis=0)
            mean = X_all.mean(axis=0)
            return mean if as_numpy else torch.as_tensor(mean, device=self.feats[0].device)

        # tensor or array case
        if torch.is_tensor(X):
            mean = X.mean(dim=0)
            return mean.detach().cpu().numpy() if as_numpy else mean

        # numpy
        X_np = np.asarray(X)
        mean = X_np.mean(axis=0)
        return mean
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

def collect_rollouts_ref(
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
    sample_actions_fn=None,   # if None and stochastic=True, use pol.sample_actions
    use_demos_as_gtruth: bool = False,
    **kwargs,
):
    policies = _normalize_policies(policy)
    m = len(policies)
    device = next(policies[0].parameters()).device

    d_idxs = np.arange(len(demos)) if demo_idxs is None else np.asarray(demo_idxs, dtype=int).reshape(-1)
    N = int(n_rollouts) if n_rollouts is not None else len(d_idxs)

    if m > 1 and (N % m != 0):
        warnings.warn(f"collect_rollouts: n_rollouts={N} not divisible by #policies={m}. Policies will be cycled.\n")

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
            if use_demos_as_gtruth:
                y_ref = d["y_demo"].to(device).view(-1)
            else:
                y_ref = yt

            Ad = d["A"]
            Ad = Ad.to(device) if torch.is_tensor(Ad) else Ad

            # --- decisions ---
            if not stochastic and decision_threshold is None:
                raise ValueError("decision_threshold must be set for deterministic collection.")

            # use provided sampler if given, else default
            #  if sample_actions_fn is not None:
            #      y_hat, logits, probs = sample_actions_fn(
            #          pol, Xd,
            #          decision_threshold=(None if stochastic else decision_threshold),
            #          return_logits=True, return_probs=True,
            #          require_grad=require_grad,
            #      )
            #  else:
            # your existing helper; respects stochastic/deterministic via threshold
            y_hat, logits, probs = sample_actions_from_policy(
                pol, Xd,
                decision_threshold=(None if stochastic else decision_threshold),
                return_logits=True, return_probs=True,
                require_grad=require_grad,
            )

            # fairness features
            f_r = compute_fairness_features(y_ref, y_hat, Ad, metrics_list)

            # zero-one vs ground truth (logging)
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
def collect_rollouts(
    *,
    policy,
    demonstrator,
    demos,
    metrics_list,
    shared_x: bool = False, # True if  demos share X for faster computation
    n_rollouts: int = None,              # now: rollouts per policy
    decision_threshold: float | None = 0.5,
    demo_idxs=None,
    before_demo=None,
    require_grad: bool = False,
    detach_outputs: bool = True,
    stochastic: bool = True,
    return_logits: bool = False,
    sample_actions_fn=None,
    use_demos_as_gtruth: bool = False,
):
    policies = _normalize_policies(policy)
    m = len(policies)
    device = next(policies[0].parameters()).device

    # number of rollouts PER policy
    n_per_policy = int(n_rollouts) if n_rollouts is not None else len(demos)
    if n_per_policy <= 0:
        raise ValueError("n_rollouts must be > 0")

    # demo schedule per policy
    if demo_idxs is None:
        d_idxs = sample_demo_idxs(len(demos), n_rollouts=n_per_policy, mode="cycle")
    else:
        d_idxs = np.asarray(demo_idxs, dtype=int).reshape(-1)
        if len(d_idxs) != n_per_policy:
            reps = int(np.ceil(n_per_policy / len(d_idxs)))
            d_idxs = np.tile(d_idxs, reps)[:n_per_policy]

    ctx = torch.enable_grad() if require_grad else torch.no_grad()

    probs_list, yhat_list, feats_list = [], [], []
    logits_list = [] if return_logits else None
    aux_list = []
    zero_one_losses = []

    pol_err = np.zeros(m, dtype=float)
    pol_count = np.zeros(m, dtype=float)
    total_err, total_count = 0.0, 0.0
    #  import ipdb;ipdb.set_trace()
    with ctx:
        if shared_x:
            cached_logits, cached_probs = None, None
            X_shared = demos[0]["X"].to(device)
        for pol_id, pol in enumerate(policies):
            if shared_x:
                cached_logits = pol(X_shared).view(-1)
                cached_probs = torch.sigmoid(cached_logits)

            for t, demo_i in enumerate(d_idxs):
                d = demos[int(demo_i)]

                aux = (before_demo(int(demo_i), d) or {}) if before_demo is not None else {}
                aux["policy_id"] = int(pol_id)
                aux["demo_idx"] = int(demo_i)
                aux["rollout_idx_within_policy"] = int(t)
                aux_list.append(aux)

                #  Xd = d["X"].to(device)
                yt = d["y"].to(device).view(-1)

                if use_demos_as_gtruth:
                    y_ref = d["y_demo"].to(device).view(-1)
                else:
                    y_ref = yt

                Ad = d["A"]
                Ad = Ad.to(device) if torch.is_tensor(Ad) else Ad

                if not stochastic and decision_threshold is None:
                    raise ValueError("decision_threshold must be set for deterministic collection.")

                if shared_x:
                    logits = cached_logits
                    probs = cached_probs
                    if stochastic:
                        y_hat = sample_binary_from_probs(probs)
                    else:
                        if decision_threshold is None:
                            raise ValueError("decision_threshold must be set for deterministic collection.")
                        y_hat = (probs >= decision_threshold).float()
                else:
                    Xd = d["X"].to(device)
                    if sample_actions_fn is not None:
                        y_hat, logits, probs = sample_actions_fn(
                            pol,
                            Xd,
                            decision_threshold=(None if stochastic else decision_threshold),
                            return_logits=True,
                            return_probs=True,
                            require_grad=require_grad,
                        )
                    else:
                        y_hat, logits, probs = sample_actions_from_policy(
                            pol,
                            Xd,
                            decision_threshold=(None if stochastic else decision_threshold),
                            return_logits=True,
                            return_probs=True,
                            require_grad=require_grad,
                        )

                f_r = compute_fairness_features(y_ref, y_hat, Ad, metrics_list)

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

    policy_perf = {
        f"policy_{i}/zero_one": float(pol_err[i] / max(1.0, pol_count[i]))
        for i in range(m)
    }
    policy_perf["batch/zero_one"] = float(batch_zero_one)
    policy_perf["n_rollouts_per_policy"] = int(n_per_policy)
    policy_perf["n_rollouts_total"] = int(m * n_per_policy)
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
