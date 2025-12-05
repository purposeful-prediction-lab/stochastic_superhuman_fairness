import numpy as np
import torch
# ---------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------
def _group_values(y, a, group_value):
    """Return subset of y for a specific protected group value."""
    return y[a == group_value]

def _safe_mean(arr):
    return np.mean(arr) if len(arr) > 0 else 0.0

# ---------------------------------------------------------------------
# Disparity Metrics
# ---------------------------------------------------------------------
def demographic_parity(y_pred, a):
    """
    |P(ŷ=1|A=0) - P(ŷ=1|A=1)|
    If no model predictions exist, y_pred can be ground truth y.
    """
    p0 = _safe_mean(y_pred[a == 0])
    p1 = _safe_mean(y_pred[a == 1])
    return abs(p0 - p1)

def equalized_odds(y_true, y_pred, a):
    """
    Average difference in TPR and FPR between groups.
    |TPR0 - TPR1| + |FPR0 - FPR1| / 2
    """
    def rates(y_t, y_p, mask):
        pos = y_t == 1
        neg = y_t == 0
        TPR = np.sum(y_p[pos & mask]) / max(np.sum(pos & mask), 1)
        FPR = np.sum(y_p[neg & mask]) / max(np.sum(neg & mask), 1)
        return TPR, FPR

    TPR0, FPR0 = rates(y_true, y_pred, a == 0)
    TPR1, FPR1 = rates(y_true, y_pred, a == 1)
    return 0.5 * (abs(TPR0 - TPR1) + abs(FPR0 - FPR1))

def predictive_rate_parity(y_true, y_pred, a):
    """
    |P(Y=1|A=0, ŷ=1) - P(Y=1|A=1, ŷ=1)|
    """
    def precision(y_t, y_p, mask):
        pred_pos = y_p == 1
        return np.sum(y_t[pred_pos & mask]) / max(np.sum(pred_pos & mask), 1)
    prec0 = precision(y_true, y_pred, a == 0)
    prec1 = precision(y_true, y_pred, a == 1)
    return abs(prec0 - prec1)

def prediction_error_disparity(y_true, a):
    """
    Label-imbalance proxy for prediction error disparity.
    Interprets P(Y=1|A) as inverse of error rate.
    """
    err0 = 1 - _safe_mean(y_true[a == 0])
    err1 = 1 - _safe_mean(y_true[a == 1])
    return abs(err0 - err1)

# ============================================================
# TORCH VERSIONS (fully differentiable)
# ============================================================


def _torch_safe_mean(x):
    return x.float().mean() if x.numel() > 0 else torch.tensor(0.0, device=x.device)


def demographic_parity_torch(y_pred, a):
    g0 = (a == 0)
    g1 = (a == 1)
    p0 = _torch_safe_mean(y_pred[g0.squeeze()])
    p1 = _torch_safe_mean(y_pred[g1.squeeze()])
    return (p0 - p1).abs()


def equalized_odds_torch(y_true, y_pred, a):
    y_bin = (y_pred > 0.5).float()
    def rates(y_t, y_p, mask):
        mask = mask.squeeze(-1)
        pos = (y_t == 1) & mask
        neg = (y_t == 0) & mask
        TPR = _torch_safe_mean(y_p[pos])
        FPR = _torch_safe_mean(y_p[neg])
        return TPR, FPR

    TPR0, FPR0 = rates(y_true, y_bin, a == 0)
    TPR1, FPR1 = rates(y_true, y_bin, a == 1)
    return ((TPR0 - TPR1).abs() + (FPR0 - FPR1).abs()) * 0.5


def predictive_rate_parity_torch(y_true, y_pred, a):
    y_bin = (y_pred > 0.5).float()

    def precision(y_t, y_p, mask):
        mask = mask.squeeze(-1)
        pred_pos = (y_p == 1) & mask
        return _torch_safe_mean(y_t[pred_pos])

    prec0 = precision(y_true, y_bin, a == 0)
    prec1 = precision(y_true, y_bin, a == 1)
    return (prec0 - prec1).abs()


def prediction_error_disparity_torch(y_true, a):
    err0 = 1 - _torch_safe_mean(y_true[a == 0])
    err1 = 1 - _torch_safe_mean(y_true[a == 1])
    return (err0 - err1).abs()

# ============================================================
# UNIFIED BACKEND-AWARE REGISTRY
# ============================================================

def _is_torch(x):
    return torch.is_tensor(x)


FAIRNESS_REGISTRY = {
    "D.DP":      (demographic_parity, demographic_parity_torch),
    "D.EqOdds":  (equalized_odds,     equalized_odds_torch),
    "D.PRP":     (predictive_rate_parity, predictive_rate_parity_torch),
    "D.Err":     (prediction_error_disparity, prediction_error_disparity_torch),
}


def compute_fairness_features(y_true, y_pred, a, 
    metrics = ["D.DP", "D.EqOdds", "D.PRP", "D.Err"],
    X = None,
    ):
    """
    Automatically picks numpy OR torch implementation.
    X : is unused, reserved  for any future metrics
    Returns a vector of fairness metric values.
    """
    feats = []

    use_torch = _is_torch(y_pred)
    a = a.squeeze(-1)
    #  import ipdb;ipdb.set_trace()
    for m in metrics:
        f_np, f_torch = FAIRNESS_REGISTRY[m]
        fargs = (y_true, y_pred, a) if (m == 'D.PRP') or ('D.EqOdds' == m) else (y_pred, a)
        if use_torch:
            feats.append(f_torch(*fargs))
        else:
            feats.append(f_np(*fargs))

    # stack in correct backend
    if use_torch:
        return torch.stack(feats)
    else:
        return np.array(feats, dtype=np.float32)
# ---------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------
METRIC_REGISTRY = {
    "D.DP": lambda y_true, y_pred, a: demographic_parity(y_pred, a),
    "D.EqOdds": equalized_odds,
    "D.PRP": predictive_rate_parity,
    "D.Err": lambda y_true, y_pred, a: prediction_error_disparity(y_true, a),
}
