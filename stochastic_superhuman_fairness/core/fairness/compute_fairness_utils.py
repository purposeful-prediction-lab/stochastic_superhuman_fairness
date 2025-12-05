# stochastic_superhuman_fairness/core/fairness/metrics.py
import torch
import numpy as np
from importlib import import_module
from stochastic_superhuman_fairness.core.fairness.fairness_metrics import METRIC_REGISTRY

def compute_fairness_features(
    X: torch.Tensor,
    y_true: torch.Tensor,
    y_pred: torch.Tensor,
    A: torch.Tensor,
    metrics: list[str] | None = None,
    threshold: float = 0.5,
) -> torch.Tensor:
    """
    Compute fairness feature vector based on metrics in METRIC_REGISTRY.
    Converts inputs to numpy and applies registered functions.

    Parameters
    ----------
    X : torch.Tensor [N, D]
        Features (currently unused, reserved for future metrics).
    y_true : torch.Tensor [N]
        Ground truth labels (0/1).
    y_pred : torch.Tensor [N]
        Predicted probabilities (0–1).
    A : torch.Tensor [N] or [N,K]
        Protected attributes (binary or multi-column, reduced to first column).
    metrics : list of str, optional
        List of metric names to compute (keys from METRIC_REGISTRY).
        Defaults to ["D.DP", "D.EqOdds", "D.PRP", "D.Err"].
    threshold : float
        Threshold for converting probabilities to binary predictions.

    Returns
    -------
    fairness_feats : torch.Tensor [len(metrics)]
        Fairness feature vector as torch tensor.
    """
    if metrics is None:
        metrics = ["D.DP", "D.EqOdds", "D.PRP", "D.Err"]

    # --- Tensor → NumPy conversion ---
    y_true_np = y_true.detach().cpu().numpy().astype(int)
    y_pred_np = (y_pred.detach().cpu().numpy() > threshold).astype(int)
    A_np = A.detach().cpu().numpy()
    if A_np.ndim > 1:
        A_np = A_np[:, 0]  # take first protected attribute if multi-column

    feats = []
    for name in metrics:
        if name not in METRIC_REGISTRY:
            raise ValueError(f"Unknown fairness metric: {name}")
        fn = METRIC_REGISTRY[name]
        val = fn(y_true_np, y_pred_np, A_np)
        feats.append(val)

    feats = np.array(feats, dtype=np.float32)
    return torch.as_tensor(feats, dtype=torch.float32, device=y_pred.device)


def compute_fairness_features_torch(X, y_true, y_pred, A, metrics):
    """
    Differentiable fairness features in torch.

    Args:
        X: [N, F] torch.tensor (unused directly except for size)
        y_true: [N] float tensor
        y_pred: [N] float tensor in [0,1]
        A: [N] group IDs (0/1)
        metrics: list of strings like ["D.DP", "D.EqOdds", ...]

    Returns:
        torch.tensor [K] — one scalar per fairness metric.
        Requires grad wrt y_pred.
    """
    eps = 1e-6
    feats = []

    # group masks
    g0 = (A == 0)
    g1 = (A == 1)

    # predictions thresholded
    y_bin = (y_pred > 0.5).float()

    for m in metrics:
        if m == "D.DP":
            # demographic parity: P(y_hat=1 | A=1) − P(y_hat=1 | A=0)
            p1 = y_bin[g1].float().mean() if g1.any() else torch.tensor(0.0, device=X.device)
            p0 = y_bin[g0].float().mean() if g0.any() else torch.tensor(0.0, device=X.device)
            feats.append(p1 - p0)

        elif m == "D.Err":
            # error rate difference
            err1 = (y_bin[g1] != y_true[g1]).float().mean() if g1.any() else torch.tensor(0.0, device=X.device)
            err0 = (y_bin[g0] != y_true[g0]).float().mean() if g0.any() else torch.tensor(0.0, device=X.device)
            feats.append(err1 - err0)

        elif m == "D.EqOpp":
            # equal opportunity: TPR(A=1) - TPR(A=0)
            pos1 = (y_true[g1] == 1)
            pos0 = (y_true[g0] == 1)
            tpr1 = (y_bin[g1][pos1]).float().mean() if pos1.any() else torch.tensor(0.0, device=X.device)
            tpr0 = (y_bin[g0][pos0]).float().mean() if pos0.any() else torch.tensor(0.0, device=X.device)
            feats.append(tpr1 - tpr0)

        elif m == "D.EqOdds":
            # eq odds = |TPR diff| + |FPR diff|
            pos1 = (y_true[g1] == 1)
            neg1 = (y_true[g1] == 0)
            pos0 = (y_true[g0] == 1)
            neg0 = (y_true[g0] == 0)

            tpr1 = (y_bin[g1][pos1]).float().mean() if pos1.any() else torch.tensor(0.0, device=X.device)
            fpr1 = (y_bin[g1][neg1]).float().mean() if neg1.any() else torch.tensor(0.0, device=X.device)
            tpr0 = (y_bin[g0][pos0]).float().mean() if pos0.any() else torch.tensor(0.0, device=X.device)
            fpr0 = (y_bin[g0][neg0]).float().mean() if neg0.any() else torch.tensor(0.0, device=X.device)

            feats.append((tpr1 - tpr0).abs() + (fpr1 - fpr0).abs())

        elif m == "D.PRP":
            # predictive rate parity: PPV(A=1) - PPV(A=0)
            pred1 = y_bin[g1]
            pred0 = y_bin[g0]
            t1 = y_true[g1]
            t0 = y_true[g0]

            ppv1 = (t1[pred1 == 1]).float().mean() if (pred1 == 1).any() else torch.tensor(0.0, device=X.device)
            ppv0 = (t0[pred0 == 1]).float().mean() if (pred0 == 1).any() else torch.tensor(0.0, device=X.device)

            feats.append(ppv1 - ppv0)

        else:
            raise ValueError(f"Unknown fairness metric '{m}'")

    return torch.stack(feats)
