from __future__ import annotations
import numpy as np
import torch
from typing import List, Dict, Literal, Union

TensorLike = Union[np.ndarray, torch.Tensor]

def _is_tensor(x): return isinstance(x, torch.Tensor)

def _ensure_2d(x):
    if x.ndim == 1:
        return x[:, None]
    return x

def phi_features(
    X: TensorLike,
    y: TensorLike,
    y_domain: Literal["01", "pm1"] = "01",  # kept for API compatibility; ignored
    add_bias: bool = False,
) -> TensorLike:
    """
    Compute φ(X, y) = y * X with y ∈ {0,1}.
    If add_bias=True, append a bias column equal to y.
    Backend (torch / numpy) matches X.
    """
    if _is_tensor(X):
        device, dtype = X.device, X.dtype
        y_t = torch.as_tensor(y, dtype=dtype, device=device).flatten()
        # ensure in [0,1] without remapping to ±1
        y_t = torch.clamp(y_t, 0.0, 1.0)
        phi = X * y_t[:, None]
        if add_bias:
            phi = torch.cat([phi, y_t[:, None]], dim=1)
        return phi
    else:
        Xn = np.asarray(X, dtype=np.float32)
        yn = np.asarray(y, dtype=np.float32).flatten()
        yn = np.clip(yn, 0.0, 1.0)
        phi = Xn * yn[:, None]
        if add_bias:
            phi = np.concatenate([phi, yn[:, None]], axis=1)
        return phi

def phi_mean(X: TensorLike, y: TensorLike, **kwargs) -> TensorLike:
    """Return mean φ(X, y) over the batch (shape [D(+1)])."""
    phi = phi_features(X, y, **kwargs)
    if _is_tensor(phi):
        return phi.mean(dim=0)
    return phi.mean(axis=0)


def phi_mean_per_demo(
    demos: List[Dict[str, TensorLike]],
    **kwargs,
) -> TensorLike:
    """
    Compute mean φ(X, y) for each demo in a list.
    Returns shape [N_demos, D(+1)].
    """
    phis = [phi_mean(d["X"], d["y"], **kwargs) for d in demos]
    if _is_tensor(phis[0]):
        return torch.stack(phis, dim=0)
    return np.stack(phis, axis=0)

def exp_phi(
    demos: List[Dict[str, TensorLike]],
    **kwargs,
) -> TensorLike:
    """
    Compute mean φ(X, y) for each demo in a list.
    Returns shape [N_demos, D(+1)].
    """
    phis = [phi_mean(d["X"], d["y"], **kwargs) for d in demos]
    if _is_tensor(phis[0]):
        return torch.stack(phis, dim=0)
    return np.stack(phis, axis=0)
