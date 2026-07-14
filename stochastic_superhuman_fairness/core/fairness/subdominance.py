from __future__ import annotations
from typing import Literal, Optional, Tuple, Dict, Union
from dataclasses import dataclass
import torch
import torch.nn.functional as F
from core.fairness.losses import ensemble_bc_loss

try:
    import torch
    _has_torch = True
except Exception:
    _has_torch = False

import numpy as np


Mode = Literal["absolute", "relative"]
Agg  = Literal["sum", "mean", "max"]

@dataclass
class SubdomLossOut:
    loss: torch.Tensor
    info: dict
# ================================================================================================================
# Losses 
# ================================================================================================================

def subdominant_logloss_shared_X_multi_rollout(
    *,
    logits_rollouts: torch.Tensor,    # [R,N] logits for shared X under each sampled theta_i
    yhat_rollouts: torch.Tensor,      # [R,N] pseudo labels for each rollout i
    y_demo: torch.Tensor,             # [D,N] demo labels for each demo j
    gamma: torch.Tensor,              # [R,D]
    indicator_win: torch.Tensor,      # [R,D] 1 if S_ij <= Srev_ji else 0
    eps: float = 1e-12,
    normalize_gamma: bool = False,
) -> SubdomLossOut:
    """
    Implements:

      L = sum_i (sum_j gamma_ij * Iwin_ij) * BCE(logits_i, yhat_i)
        + sum_{i,j} gamma_ij * (1-Iwin_ij) * BCE(logits_i, y_demo_j)

    Shared X across all i and j.
    """
    device = logits_rollouts.device
    logits_rollouts = logits_rollouts.to(device).float()  # [R,N]
    yhat_rollouts = yhat_rollouts.to(device).float()      # [R,N]
    y_demo = y_demo.to(device).float()                    # [D,N]
    gamma = gamma.to(device).float()                      # [R,D]
    I = indicator_win.to(device).float()                  # [R,D]

    if normalize_gamma:
        gamma = gamma / (gamma.sum() + eps)

    R, N = logits_rollouts.shape
    D = y_demo.shape[0]

    # --- term 1: rollout wins -> fit yhat_i under logits_i ---
    # per-rollout BCE: [R]
    bce_roll = F.binary_cross_entropy_with_logits(
        logits_rollouts, yhat_rollouts, reduction="none"
    ).sum(dim=1)  # [R]

    w_win_i = (gamma * I).sum(dim=1)  # [R]
    term1 = (w_win_i * bce_roll).sum()

    # --- term 2: rollout loses to demo -> fit y_demo_j under logits_i ---
    # BCE for every (i,j): compute BCE(logits_i, y_demo_j) -> [R,D]
    # Expand:
    #   logits: [R,1,N]
    #   y_demo: [1,D,N]
    logits_ = logits_rollouts[:, None, :]   # [R,1,N]
    ydemo_  = y_demo[None, :, :]            # [1,D,N]

    bce_pair = F.binary_cross_entropy_with_logits(
        logits_.expand(R, D, N),
        ydemo_.expand(R, D, N),
        reduction="none",
    ).sum(dim=2)  # [R,D]

    w_lose_ij = gamma * (1.0 - I)           # [R,D]
    term2 = (w_lose_ij * bce_pair).sum()

    loss = term1 + term2
    #  import ipdb;ipdb.set_trace()

    info = {
        "w_win_sum": float(w_win_i.sum().detach().cpu()),
        "w_lose_sum": float(w_lose_ij.sum().detach().cpu()),
        "indicator_mean": float(I.mean().detach().cpu()),
        "term1": float(term1.detach().cpu()),
        "term2": float(term2.detach().cpu()),
    }
    return SubdomLossOut(loss=loss, info=info)

# ---------------------------------------------------------------------------------
def subdominant_weighted_logloss_shared_X_multi_rollout(
    S,                                # [R,D]  rollout-demo subdominance matrix
    S_rev,                                # [R,D]  rollout-demo subdominance matrix
    logits_rollouts: torch.Tensor,    # [R,N] logits for shared X under each sampled theta_i
    yhat_rollouts: torch.Tensor,      # [R,N] pseudo labels for each rollout i
    y_demo: torch.Tensor,             # [D,N] demo labels for each demo j
    gamma: torch.Tensor,              # [R,D]
    indicator_win: torch.Tensor,      # [R,D] 1 if S_ij <= Srev_ji else 0
    criterion: torch.Tensor,
    eps: float = 1e-12,
    normalize_gamma: bool = False,
    term_weights: list = [1., 1.],
    **kwargs,
) -> SubdomLossOut:
    """
    Implements:

      L = sum_i (sum_j gamma_ij * Iwin_ij) * BCE(logits_i, yhat_i)
        + sum_{i,j} gamma_ij * (1-Iwin_ij) * BCE(logits_i, y_demo_j)

    Shared X across all i and j.
    """
    device = logits_rollouts.device
    logits_rollouts = logits_rollouts.to(device).float()  # [R,N]
    yhat_rollouts = yhat_rollouts.to(device).float()      # [R,N]
    y_demo = y_demo.to(device).float()                    # [D,N]
    gamma = gamma.to(device).float()                      # [R,D]
    I = indicator_win.to(device).float()                  # [R,D]
    if criterion.ndim == 0:
        criterion = criterion.repeat(S.shape[0])
    if normalize_gamma:
        gamma = gamma / (gamma.sum() + eps)

    R, N = logits_rollouts.shape
    D = y_demo.shape[0]

    # --- term 1: rollout wins -> fit yhat_i under logits_i ---
    # per-rollout BCE: [R]
    bce_roll = F.binary_cross_entropy_with_logits(
        logits_rollouts, yhat_rollouts, reduction="none"
    ).sum(dim=1)  # [R]

    # New advantage like term
    w_win_ij = gamma * I   # [R, D]
    l_adv = torch.clamp(criterion[:, None] -  S, 0)      #[R,D]
    loss_i =  (l_adv * w_win_ij).sum(axis=1) * bce_roll
    term1 = (loss_i).sum()

    # --- term 2: rollout loses to demo -> fit y_demo_j under logits_i ---
    # BCE for every (i,j): compute BCE(logits_i, y_demo_j) -> [R,D]
    # Expand:
    #   logits: [R,1,N]
    #   y_demo: [1,D,N]
    logits_ = logits_rollouts[:, None, :]   # [R,1,N]
    ydemo_  = y_demo[None, :, :]            # [1,D,N]
    bce_pair = F.binary_cross_entropy_with_logits(
        logits_.expand(R, D, N),
        ydemo_.expand(R, D, N),
        reduction="none",
        )  # [R,D, N]

    w_lose_ij = (gamma * (1.0 - I))             # [R,D]
    # Add a hinge term as well (sij - S_bar)+ for demo increase and (s_bar-sij)+ for the rollout term
    # New advantage like term
    l_advj = torch.clamp(S- criterion[:, None], 0)      #[R,D]
    loss_j =  (l_advj * w_lose_ij * bce_pair.sum(dim=2)).sum(axis=1)
    term2 = (loss_j).sum()
    
    # Total loss
    loss = term_weights[0] * term1 + term_weights[1] * term2
    info = {
        "w_win_sum": float(w_win_ij.sum().detach().cpu()),
        "w_lose_sum": float(w_lose_ij.sum().detach().cpu()),
        "indicator_mean": float(I.mean().detach().cpu()),
        "term1": float(term1.detach().cpu()),
        "term2": float(term2.detach().cpu()),
        "term1_activations": I.sum().item(), 
        "term2_activations": (1-I).sum().item(), 
        'S_mean': S.mean().detach().cpu().item(),
        "norm_paired_subdom":  ((gamma * S).sum()/gamma.sum()).detach().cpu().item(),
        "paired_subdom":  ((gamma * S).sum()).detach().cpu().item(),
        'demo_logprobs': bce_pair.detach(),
    }
    return SubdomLossOut(loss=loss, info=info)

def behavior_guided_subdominant_spring_loss(
    S,                                # [R,D]  rollout-demo subdominance matrix
    S_rev,                                # [R,D]  rollout-demo subdominance matrix
    logits_rollouts: torch.Tensor,    # [R,N] logits for shared X under each sampled theta_i
    yhat_rollouts: torch.Tensor,      # [R,N] pseudo labels for each rollout i
    y_demo: torch.Tensor,             # [D,N] demo labels for each demo j
    gamma: torch.Tensor,              # [R,D]
    indicator_win: torch.Tensor,      # [R,D] 1 if S_ij <= Srev_ji else 0
    criterion: torch.Tensor,
    eps: float = 1e-12,
    normalize_gamma: bool = False,
    term_weights: list = [1., 1.],
    bc_lambda: float = 0.1,
    **kwargs,
) -> SubdomLossOut:

    loss_out = subdominant_weighted_logloss_shared_X_multi_rollout(
        S,
        S_rev,
        logits_rollouts,  # [R,N]
        yhat_rollouts,    # [R,N]
        y_demo,           # [D,N]
        gamma,            # [R,D]
        indicator_win,    # [R,D]
        criterion,        # [R]
        eps = eps,
        normalize_gamma = normalize_gamma,
        term_weights = term_weights,
    )
    import ipdb;ipdb.set_trace()
    bc_loss, bc_info = ensemble_bc_loss(
        logits_policies=logits_rollouts,
        y_demo=y_demo,
        demo_weights=None,
    )
    loss = loss_out.loss + bc_lambda * bc_loss
    loss_out.info['bc_term'] = bc_loss.detach().cpu().item()
    return SubdomLossOut(loss = loss, info = loss_out.info)


# ---------------------------------------------------------------------------------

def subdominance_loss_from_features(
    rollout_feats,              # [R, K]
    demo_feats,                 # [D, K]
    mode: Mode = "absolute",
    agg: Agg = "mean",
    alpha=None,
    beta=None,
    reduction: Literal["mean", "sum", "none"] = "mean",
) -> Dict[str, object]:
    """
    High-level entry: compute S matrix, aggregate per rollout, then reduce to a scalar loss.

    Returns dict:
      {
        "S": pairwise matrix [R,D] (same backend as inputs),
        "per_rollout": [R], aggregated per rollout,
        "loss": scalar (or vector if reduction='none')
      }
    """
    S = compute_subdominance_matrix(rollout_feats, demo_feats, mode=mode, alpha=alpha, beta=beta)
    per_rollout = aggregate_subdominance(S, agg=agg)
    #  import ipdb;ipdb.set_trace()
    # Reduction to scalar loss (default: mean over rollouts)
    if _is_tensor(per_rollout):
        if reduction == "mean":
            loss = per_rollout.mean()
        elif reduction == "sum":
            loss = per_rollout.sum()
        elif reduction == "none":
            loss = per_rollout
        else:
            raise ValueError(f"Unknown reduction: {reduction}")
    else:
        if reduction == "mean":
            loss = float(per_rollout.mean())
        elif reduction == "sum":
            loss = float(per_rollout.sum())
        elif reduction == "none":
            loss = per_rollout
        else:
            raise ValueError(f"Unknown reduction: {reduction}")

    return {"S": S, "per_rollout": per_rollout, "loss": loss}



def _is_tensor(x):
    return _has_torch and isinstance(x, torch.Tensor)


def _to_backend(x, like):
    """Convert x to backend/dtype/device of like (torch or numpy)."""
    import numpy as np, torch

    if torch.is_tensor(like):                         # Torch backend
        if x is None: return None
        if torch.is_tensor(x): return x.to(like.device, like.dtype)
        x = np.asarray(x)
        return torch.as_tensor(x, device=like.device, dtype=like.dtype)

    # NumPy backend
    if x is None: return None
    if torch.is_tensor(x): x = x.detach().cpu().numpy()
    x = np.asarray(x)
    tgt = like.dtype if hasattr(like, "dtype") else np.float32
    return x.astype(tgt) if x.dtype != tgt else x

def _broadcast_params(alpha, beta, K, like):
    """
    Universal version of parameter broadcasting.
    Ensures alpha, beta:
      - match the backend of 'like' (numpy or torch)
      - lie on the same device as 'like' if torch
      - have dtype matching 'like'
      - have shape [1, K] for correct broadcasting
      - default to ones if None
      - scalars expanded to vectors

    Args:
        alpha: None | scalar | array-like | tensor
        beta:  None | scalar | array-like | tensor
        K: feature dimension
        like: reference tensor/array determining backend, device, dtype

    Returns:
        (alpha, beta) each shaped [1, K]
    """

    """Expand alpha/beta to match backend of 'like' with shape [1,K]."""

    def _prep(p):
        if p is None:
            return torch.ones(K, device=like.device, dtype=like.dtype) if torch.is_tensor(like) else np.ones(K, like.dtype if hasattr(like,"dtype") else np.float32)
        if np.isscalar(p) or (torch.is_tensor(p) and p.ndim==0):
            if torch.is_tensor(like): return torch.ones(K, device=like.device, dtype=like.dtype)*p
            arr = np.ones(K, dtype=np.asarray(p).dtype)*float(p); return arr
        p = _to_backend(p, like)
        p = p.reshape(-1)
        #  import ipdb;ipdb.set_trace()
        if p.size()[0] != K: raise ValueError(f"alpha/beta must have {K} elements, got {p.size()}")
        return p

    a = _prep(alpha); b = _prep(beta)
    if torch.is_tensor(a): return a.view(1,K), b.view(1,K)
    return a.reshape(1,K), b.reshape(1,K)

# -------------------------------------------------------------------------------------

def build_indicator_from_subdom(
    S: torch.Tensor,
    S_ref: torch.Tensor,
    *,
    margin: float = 0.0,
) -> torch.Tensor:
    """
    Indicator I_{ij} = 1 if S_ij + margin < S_ref_ji else 0.
    Shapes:
      S:      [R, D]
      S_ref:  [R, D]
    Returns:
      I: float tensor [R, D] in {0,1}
    """
    return (S + margin < S_ref).to(dtype=S.dtype)

def compute_subdominance_matrix(
    rollout_feats,              # [R, K]
    demo_feats,                 # [D, K]
    mode: str = "absolute",
    alpha=None,                 # scalar, [K], or [R,K]
    beta=None,                  # scalar, [K], or [R,K] (optional)
    eps: float = 1e-12,
    feat_reduce: Literal['sum', 'mean', 'max'] = 'sum',
):
    """
    Compute pairwise subdominance S[r, d] between rollout r and demo d for K features.

    Accepts:
      - alpha: scalar, (K,), or (R,K) where alpha[r,:] applies to rollout r
      - beta : scalar, (K,), or (R,K) similarly (if you want per-rollout offsets), if None it equals 0.

    absolute:  S = ReLU( alpha * (f_r - f_d) + beta ) summed over K
    relative:  S = ReLU( alpha * ((f_r / f_d) - 1) + beta ) summed over K

    Inputs may be torch tensors or numpy arrays; output matches the backend of rollout_feats.
    """
    like = rollout_feats
    rf = _to_backend(rollout_feats, like)   # [R,K]
    df = _to_backend(demo_feats, like)      # [D,K]
    beta = 0 if beta is None else beta
    R, K = rf.shape
    D, Kd = df.shape
    if Kd != K:
        raise ValueError(f"Feature dimensions must match, got K={K} vs Kd={Kd}.")

    is_torch = _is_tensor(rf)

    def _as_backend(x):
        return _to_backend(x, rf) if x is not None else None

    def _broadcast_param(p, name):
        # default
        if p is None:
            if is_torch:
                import torch
                p = torch.ones(K, device=rf.device, dtype=rf.dtype)
            else:
                p = np.ones(K, dtype=rf.dtype if hasattr(rf, "dtype") else np.float32)
        p = _as_backend(p)

        # scalar -> (1,1,K)
        if (is_torch and p.ndim == 0) or ((not is_torch) and np.isscalar(p)) or( (not is_torch) and np.ndim(p) == 0):
            return p.reshape(1, 1, 1) * (rf.new_ones((1, 1, K)) if is_torch else np.ones((1, 1, K), dtype=p.dtype))

        # (K,) -> (1,1,K)
        if p.ndim == 1:
            if p.shape[0] != K:
                raise ValueError(f"{name} must be scalar, (K,), or (R,K). Got {p.shape}.")
            return p.reshape(1, 1, K)

        # (R,K) -> (R,1,K)
        if p.ndim == 2:
            if p.shape != (R, K):
                raise ValueError(f"{name} must be scalar, (K,), or (R,K)={(R,K)}. Got {p.shape}.")
            return p.reshape(R, 1, K)

        raise ValueError(f"{name} must be scalar, (K,), or (R,K). Got ndim={p.ndim}.")

    alpha3 = _broadcast_param(alpha, "alpha")  # (1,1,K) or (R,1,K)
    beta3  = _broadcast_param(beta,  "beta")   # (1,1,K) or (R,1,K)
    # Expand features for pairwise ops -> [R, D, K]
    rf3 = rf[:, None, :]   # [R,1,K]
    df3 = df[None, :, :]   # [1,D,K]

    if mode == "absolute":
        core = alpha3 * (rf3 - df3) + beta3
    elif mode == "relative":
        if is_torch:
            import torch
            denom = torch.clamp(df3, min=eps)
        else:
            denom = np.clip(df3, eps, None)
        core = alpha3 * ((rf3 / denom) - 1.0) + beta3
    else:
        raise ValueError(f"Unknown mode: {mode}")
    #  import ipdb;ipdb.set_trace()
    if is_torch:
        import torch
        relu_core = torch.relu(core)

        if feat_reduce == "sum":
            return relu_core.sum(dim=-1)
        elif feat_reduce == "mean":
            return relu_core.mean(dim=-1)
        elif feat_reduce == "max":
            return relu_core.max(dim=-1).values
        elif feat_reduce == "l2":
            return torch.norm(relu_core, dim=-1)   # torch
    # or np.linalg.norm(..., axis=-1)
        else:
            raise ValueError(f"Unknown feat_reduce: {feat_reduce}")

    else:
        relu_core = np.maximum(core, 0.0)

        if feat_reduce == "sum":
            return relu_core.sum(axis=-1)
        elif feat_reduce == "mean":
            return relu_core.mean(axis=-1)
        elif feat_reduce == "max":
            return relu_core.max(axis=-1)
        else:
            raise ValueError(f"Unknown feat_reduce: {feat_reduce}")
# -------------------------------------------------------------------------------------

def compute_subdominance_matrix_simple(
    rollout_feats,              # [R, K]
    demo_feats,                 # [D, K]
    mode: Mode = "absolute",
    alpha=None,                 # [K] or scalar; default ones (vector of 1s)
    beta=None,                  # [K] or scalar; default ones (vector of 1s)
    eps: float = 1e-12,
):
    """
    Compute pairwise subdominance S[r, d] between rollout r and demo d for K features.

    absolute:  S = ReLU( alpha * (f_r - f_d) + beta ) reduced over K (sum over features)
    relative:  S = ReLU( alpha * ((f_r / f_d) - 1) + beta ) reduced over K

    Inputs may be torch tensors or numpy arrays; output matches the backend of inputs.
    """
    # Choose backend based on rollout_feats
    like = rollout_feats
    R, K = rollout_feats.shape
    D, Kd = demo_feats.shape
    if Kd != K:
        raise ValueError(f"Feature dimensions must match, got K={K} vs Kd={Kd}.")

    # Align backends
    rf = _to_backend(rollout_feats, like)           # [R, K]
    df = _to_backend(demo_feats, like)              # [D, K]
    alpha, beta = _broadcast_params(alpha, beta, K, rf)

    # Expand for pairwise ops -> [R, D, K]
    if _is_tensor(rf):
        rf3 = rf[:, None, :]                      # [R,1,K]
        df3 = df[None, :, :]                      # [1,D,K]
        if mode == "absolute":
            diff = rf3 - df3
            core = alpha * diff + beta
        elif mode == "relative":
            denom = torch.clamp(df3, min=eps)
            rel = (rf3 / denom) - 1.0
            core = alpha * rel + beta
        else:
            raise ValueError(f"Unknown mode: {mode}")
        S = torch.relu(core).sum(dim=-1)          # reduce over features -> [R, D]
        return S
    else:
        # numpy
        rf3 = rf[:, None, :]                      # [R,1,K]
        df3 = df[None, :, :]                      # [1,D,K]
        if mode == "absolute":
            diff = rf3 - df3
            core = alpha * diff + beta
        elif mode == "relative":
            denom = np.clip(df3, eps, None)
            rel = (rf3 / denom) - 1.0
            core = alpha * rel + beta
        else:
            raise ValueError(f"Unknown mode: {mode}")
        S = np.maximum(core, 0.0).sum(axis=-1)    # [R, D]
        return S

# -------------------------------------------------------------------------------------
def compute_subdominance_matrix_grouped(
    rollout_feats,              # list of (r_m, K) OR [R,K]
    demo_feats,                 # [D, K]
    mode: str = "absolute",
    alpha=None,
    beta=None,
    eps: float = 1e-12,
    return_group_info: bool = False,
):
    """
    Like compute_subdominance_matrix, but accepts rollout_feats as either:
      - a single matrix/tensor [R, K]
      - a list/tuple of matrices/tensors [(r_0,K), (r_1,K), ...]

    If a list is given, all rollout groups are concatenated in order, so rows
    from the same group stay contiguous in the final rollout matrix.

    Returns:
      S                  if return_group_info=False
      (S, row_groups)    if return_group_info=True

    where row_groups is e.g. [[0,1,2],[3,4],...]
    """

    # already a single matrix
    if not isinstance(rollout_feats, (list, tuple)):
        S = compute_subdominance_matrix(
            rollout_feats=rollout_feats,
            demo_feats=demo_feats,
            mode=mode,
            alpha=alpha,
            beta=beta,
            eps=eps,
        )
        if return_group_info:
            R = rollout_feats.shape[0]
            return S, [list(range(R))]
        return S

    if len(rollout_feats) == 0:
        raise ValueError("rollout_feats list is empty.")

    first = rollout_feats[0]
    is_torch = _is_tensor(first)

    # concat while preserving grouping
    row_groups = []
    pieces = []
    start = 0
    K = None

    for grp in rollout_feats:
        grp_b = _to_backend(grp, first)
        if grp_b.ndim != 2:
            raise ValueError("Each rollout group must be 2D with shape (r_m, K).")

        r_m, k_m = grp_b.shape
        if K is None:
            K = k_m
        elif k_m != K:
            raise ValueError(f"All rollout groups must have same K. Got {K} and {k_m}.")

        pieces.append(grp_b)
        row_groups.append(list(range(start, start + r_m)))
        start += r_m

    if is_torch:
        import torch
        rollout_mat = torch.cat(pieces, dim=0)
    else:
        rollout_mat = np.concatenate(pieces, axis=0)

    S = compute_subdominance_matrix(
        rollout_feats=rollout_mat,
        demo_feats=demo_feats,
        mode=mode,
        alpha=alpha,
        beta=beta,
        eps=eps,
    )

    if return_group_info:
        return S, row_groups
    return S
# -------------------------------------------------------------------------------------

def aggregate_subdominance(
    S,                          # [R, D] matrix
    agg: Agg = "mean"
):
    """
    Aggregate subdominance across demos per rollout.
    Returns vector s_agg with shape [R].
    """
    if _is_tensor(S):
        if agg == "sum":
            return S.sum(dim=1)
        elif agg == "mean":
            return S.mean(dim=1)
        elif agg == "max":
            return S.max(dim=1).values
        else:
            raise ValueError(f"Unknown aggregation: {agg}")
    else:
        if agg == "sum":
            return S.sum(axis=1)
        elif agg == "mean":
            return S.mean(axis=1)
        elif agg == "max":
            return S.max(axis=1)
        else:
            raise ValueError(f"Unknown aggregation: {agg}")

# ---------------------------------------------------------------------------------

def compute_alpha(
    rollouts,
    demos,
    beta,
    *,
    mode: str = "absolute",       # "absolute" | "relative"
    reduce: str = "none",         # "none" | "mean" | "median"
    means_mode: str = "identity", # passed to compute_sorted_demo_means if demos not already sorted
    alpha_max: float = 10.0,
    eps: float = 1e-12,
    sorted_check_eps: float = 0.0,
):
    """
    Compute per-rollout alpha.

    Accepts EITHER:
      - demos as already-sorted demo means (D,K) with each column nondecreasing, OR
      - raw/unsorted demos (D,K) (or (N,K) if means_mode="cumulative").

    If demos is not nondecreasing per column, we compute sorted demo means via:
        demos_sorted = compute_sorted_demo_means(demos, means_mode=means_mode)

    For each rollout r and feature k:
      fr = rollouts[r,k]
      fd = first demos_sorted[:,k] with fd > fr
      alpha[r,k] = min(alpha_max, beta_k / denom)
        denom = fd - fr                          (mode="absolute")
        denom = 1 - fd/(fr + 1e-6)               (mode="relative")
      If no fd or denom <= eps -> alpha_max.

    Returns:
      reduce="none"   -> (R,K)
      reduce="mean"   -> (K,)
      reduce="median" -> (K,)

    Inputs can be numpy or torch; output matches backend/dtype/device of rollouts.
    """
    like = rollouts
    r = _to_backend(rollouts, np.empty((), np.float64))
    d = _to_backend(demos,    np.empty((), np.float64))
    if r.ndim != 2 or d.ndim != 2 or r.shape[1] != d.shape[1]:
        raise ValueError("rollouts and demos must be 2D with same K.")

    # infer if already sorted per column
    if not bool(np.all(np.diff(d, axis=0) >= -sorted_check_eps)):
        d = compute_sorted_demo_means(d, means_mode=means_mode)

    R, K = r.shape
    D = d.shape[0]

    b = np.full(K, float(beta)) if np.isscalar(beta) else np.asarray(beta, float).reshape(-1)
    if b.size != K:
        raise ValueError(f"beta must be scalar or length-K (K={K}). Got {b.size}.")

    rel = (mode.lower() == "relative")
    if not rel and mode.lower() != "absolute":
        raise ValueError("mode must be 'absolute' or 'relative'.")

    a = np.full((R, K), float(alpha_max), dtype=float)
    for k in range(K):
        fr = r[:, k]
        idx = np.searchsorted(d[:, k], fr, side="right")
        ok = idx < D
        if not np.any(ok): 
            continue
        fd = d[np.clip(idx, 0, D - 1), k]
        denom = (1.0 - fd / (fr + 1e-6)) if rel else (fd - fr)
        good = ok & (denom > eps)
        if np.any(good):
            a[good, k] = np.minimum(alpha_max, b[k] / denom[good])
        #  import ipdb;ipdb.set_trace()
    red = reduce.lower()
    out = a if red == "none" else (np.nanmean(a, 0) if red == "mean" else np.nanmedian(a, 0) if red == "median" else None)
    if out is None:
        raise ValueError("reduce must be 'none', 'mean', or 'median'.")

    return _to_backend(out, like)

# ---------------------------------------------------------------------------------

def compute_beta(rollout_feats, demo_feats, mode: Mode = "absolute"):
    """
    Placeholder for closed-form beta based on rollout and demo features.
    Must return shape [K].
    For now, returns ones.
    """
    like = rollout_feats
    K = rollout_feats.shape[-1]
    return _to_backend(np.ones((K,), dtype=np.float32), like)

# ---------------------------------------------------------------------------------

def compute_sorted_demo_means(demos, *, means_mode: str = "identity"):
    """
    Returns demos_means_sorted: (D,K) sorted ascending per column.

    means_mode:
      - "identity": treat demos as already (D,K) means; just sort per column
      - "cumulative": interpret demos as (N,K) samples; build cumulative means
                      after sorting rows by ||row|| (magnitude):
                        m[t] = mean(rows_sorted[:t+1], axis=0)
                      then sort each column of m ascending (to satisfy searchsorted usage)
    """
    x = _to_backend(demos, np.empty((), np.float64))  # force numpy for processing
    if x.ndim != 2:
        raise ValueError("demos must be 2D (N,K) or (D,K).")

    means_mode = means_mode.lower()
    if means_mode == "identity":
        m = x
    elif means_mode == "cumulative":
        order = np.argsort(np.linalg.norm(x, axis=1))
        xs = x[order]
        m = np.cumsum(xs, axis=0) / (np.arange(xs.shape[0])[:, None] + 1.0)
    else:
        raise ValueError("means_mode must be 'identity' or 'cumulative'.")

    # sort each feature column ascending (required by np.searchsorted usage)
    return np.sort(m, axis=0)

# ---------------------------------------------------------------------------------


def compute_subdominance_loss(
    rollout_feats,
    demo_feats,
    mode: Mode = "absolute",
    agg: Agg = "mean",
    alpha=None,
    beta=None,
    reduction: Literal["mean", "sum", "none"] = "mean",
):
    """
    Thin wrapper used by the Learner to obtain a scalar loss
    while preserving detailed artifacts if needed.
    """
    out = subdominance_loss_from_features(
        rollout_feats=rollout_feats,
        demo_feats=demo_feats,
        mode=mode,
        agg=agg,
        alpha=alpha,
        beta=beta,
        reduction=reduction,
    )
    return out["loss"]

def compute_beat_rates(S, axis=1, ignore_diag=True, return_beat_matrix: bool = False):
    """
    Compute beat rates from a pairwise score matrix.

    Assumes:
        element i beats j if S[i,j] < S[j,i]

    Args
    ----
    S : (N,N) array-like
        Pairwise score matrix.
    axis : int
        Axis over which to compute rates.
        axis=1 -> row element beats column element (default).
    ignore_diag : bool
        Whether to ignore self-comparisons.

    Returns
    -------
    beat_rates : (N,) ndarray
        Fraction of opponents each element beats.
    beat_matrix : (N,N) ndarray
        Boolean matrix where beat_matrix[i,j] = True if i beats j.
    """
    S = np.asarray(S)

    if S.shape[0] != S.shape[1]:
        raise ValueError("S must be square for beat-rate computation.")

    # i beats j if S[i,j] < S[j,i]
    beat_matrix = S < S.T

    if ignore_diag:
        np.fill_diagonal(beat_matrix, False)
        denom = S.shape[0] - 1
    else:
        denom = S.shape[0]

    beat_rates = beat_matrix.sum(axis=axis) / max(1, denom)
    if return_beat_matrix:
        return beat_rates, beat_matrix
    return beat_rates


