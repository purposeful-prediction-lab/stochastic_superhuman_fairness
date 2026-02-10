from __future__ import annotations
from typing import Literal, Optional, Tuple, Dict, Union

try:
    import torch
    _has_torch = True
except Exception:
    _has_torch = False

import numpy as np


Mode = Literal["absolute", "relative"]
Agg  = Literal["sum", "mean", "max"]

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

def compute_subdominance_matrix(
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


# ---------- placeholders for parameter learning (alpha/beta) ----------
def compute_alpha_sorted_demos(
    rollouts, demos_sorted, beta, *,
    mode: str = "absolute",
    alpha_max: float = 100.0,
    eps: float = 1e-12,
    fr=None,
):
    like = rollouts
    r = _to_backend(rollouts, np.empty((), np.float64))
    d = _to_backend(demos_sorted, np.empty((), np.float64))
    if r.ndim != 2 or d.ndim != 2 or r.shape[1] != d.shape[1]:
        raise ValueError("rollouts and demos_sorted must be 2D with same K.")

    K, D = r.shape[1], d.shape[0]
    b = np.full(K, float(beta)) if np.isscalar(beta) else np.asarray(beta, float).reshape(-1)
    if b.size != K: raise ValueError(f"beta must be scalar or length-K (K={K}). Got {b.size}.")
    fr = np.nanmean(r, 0) if fr is None else np.asarray(fr, float).reshape(-1)
    if fr.size != K: raise ValueError(f"fr must be length-K (K={K}). Got {fr.size}.")
    rel = (mode.lower() == "relative")
    if not rel and mode.lower() != "absolute": raise ValueError("mode must be 'absolute' or 'relative'.")

    a = np.full(K, float(alpha_max))
    for k in range(K):
        i = np.searchsorted(d[:, k], fr[k], side="right")
        if i >= D: continue
        fd, frk = d[i, k], fr[k]
        denom = (1.0 - fd / (frk + 1e-6)) if rel else (fd - frk)
        if denom > eps: a[k] = min(alpha_max, b[k] / denom)

    return _to_backend(a, like)

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


def compute_alpha(
    rollouts: Union[np.ndarray, torch.Tensor],
    demos: Union[np.ndarray, torch.Tensor],
    beta,
    *,
    demos_are_sorted_means: bool | None = None,
    means_mode: str = "identity",
    mode: str = "absolute",          # passed through to compute_alpha_sorted_demos
    alpha_max: float = 100.0,
    eps: float = 1e-12,
    fr=None,
):
    """
    Wrapper:
      - If demos is already sorted mean demos: use it directly.
      - Else: compute + sort demo means, then call compute_alpha_sorted_demos.

    rollouts/demos can be numpy arrays or torch tensors.
    alpha returned matches backend/dtype/device of `rollouts` via _to_backend.
    """
    like = rollouts

    # decide whether demos is already "sorted mean demos"
    if demos_are_sorted_means is None:
        d_np = _to_backend(demos, np.empty((), np.float64))
        # heuristic: check monotone nondecreasing per column
        demos_are_sorted_means = bool(np.all(np.diff(d_np, axis=0) >= 0))

    demos_sorted = (
        _to_backend(demos, np.empty((), np.float64))
        if demos_are_sorted_means
        else compute_sorted_demo_means(demos, means_mode=means_mode)
    )

    # call the alpha core (it returns in backend of rollouts already)
    return compute_alpha_sorted_demos(
        rollouts,
        demos_sorted,
        beta,
        mode=mode,
        alpha_max=alpha_max,
        eps=eps,
        fr=fr,
    )

def compute_beta(rollout_feats, demo_feats, mode: Mode = "absolute"):
    """
    Placeholder for closed-form beta based on rollout and demo features.
    Must return shape [K].
    For now, returns ones.
    """
    like = rollout_feats
    K = rollout_feats.shape[-1]
    return _to_backend(np.ones((K,), dtype=np.float32), like)


# ---------- stochastic subdominance (OT-based, placeholder) ----------

def stochastic_subdominance_ot(
    rollout_feats,              # [R, K]
    demo_feats,                 # [D, K]
    mode: Mode = "absolute",
    alpha=None,
    beta=None,
    reg: float = 0.05,
    return_transport: bool = False,
) -> Dict[str, object]:
    """
    Placeholder for stochastic subdominance:
      1) Build pairwise subdominance cost matrix C = S (or transformation thereof)
      2) Solve an optimal transport (OT) problem producing a coupling P (R x D)
      3) Marginal over demos per rollout gives probabilities over rollouts or vice versa

    For now, we return a uniform coupling with the correct shapes and the base S matrix.
    Replace with a real OT solver (e.g., Sinkhorn) later.

    Returns dict:
      {
        "S": [R,D] base subdominance scores,
        "P": [R,D] transport plan (placeholder uniform),
        "obj": scalar objective proxy (e.g., <P, S>), placeholder,
      }
    """
    S = compute_subdominance_matrix(rollout_feats, demo_feats, mode=mode, alpha=alpha, beta=beta)

    if _is_tensor(S):
        R, D = S.shape
        P = torch.full_like(S, 1.0 / (R * D))
        obj = (P * S).sum()
    else:
        R, D = S.shape
        P = np.full_like(S, 1.0 / (R * D))
        obj = float((P * S).sum())

    out = {"S": S, "P": P, "obj": obj}
    if return_transport:
        out["transport"] = P
    return out


# ---------- convenience: Learner hook (optional) ----------

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
