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

# Other Metrics ----------------------------------------------
def zero_one_loss(y_true, y_pred, *args, decision_threshold: float = 0.5):
    """
    Mean zero-one loss.
    Accepts logits, probabilities, or 0/1 labels.
    Works for numpy or torch.
    """

    # ---- TORCH ----
    if torch.is_tensor(y_true):
        if torch.is_floating_point(y_pred):
            # logits if outside [0,1]
            if y_pred.min() < 0 or y_pred.max() > 1:
                y_pred = torch.sigmoid(y_pred)
            y_pred = (y_pred >= decision_threshold).float()
        return float((y_true != y_pred).float().mean().item())

    # ---- NUMPY ----
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    if np.issubdtype(y_pred.dtype, np.floating):
        if y_pred.min() < 0 or y_pred.max() > 1:
            y_pred = 1 / (1 + np.exp(-y_pred))  # sigmoid
        y_pred = (y_pred >= decision_threshold).astype(y_true.dtype)

    return float(np.mean(y_true != y_pred))
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

# Other Metrics ----------------------------------------------
def zero_one_loss_torch(y_true, y_pred):
    """
    Mean zero-one loss (misclassification rate).

    y_true: tensor of shape [N] or [N,1]
    y_pred: tensor of same shape (either probs or binary)
    Returns: scalar tensor
    """
    # ensure binary predictions
    if y_pred.dtype.is_floating_point:
        y_pred = (y_pred > 0.5).float()

    y_true = y_true.float()

    return (y_true != y_pred).float().mean()

# ---------------------------------------------------------------------
# Custom Cost functions
# ---------------------------------------------------------------------
def compute_directional_cost(R_feats, D_feats, n_dir=20):
    """
    Computes directional discrepancies before the OT step.
    R_feats: [R,K]
    D_feats: [D,K]
    """
    R, K = R_feats.shape
    D = D_feats.shape[0]

    out = {"feature_combinations": {}}

    # -----------------------------------------
    # (A) All-feature random directions
    # -----------------------------------------
    diffs = []
    for _ in range(n_dir):
        v = np.random.randn(K)
        v /= np.linalg.norm(v) + 1e-12
        r_proj = R_feats @ v
        d_proj = D_feats @ v
        diffs.append(np.abs(r_proj.mean() - d_proj.mean()))
    out["all_features_directional_cost"] = float(np.mean(diffs))

    # -----------------------------------------
    # (B) 2D feature combinations
    # -----------------------------------------
    for i in range(K):
        for j in range(i+1, K):
            name = f"f{i}_f{j}"
            diffs = []
            for _ in range(n_dir):
                v = np.random.randn(2)
                v /= np.linalg.norm(v) + 1e-12
                r_proj = R_feats[:, [i, j]] @ v
                d_proj = D_feats[:, [i, j]] @ v
                diffs.append(np.abs(r_proj.mean() - d_proj.mean()))
            out["feature_combinations"][name] = float(np.mean(diffs))

    return out


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
    "L.ZeroOne": (zero_one_loss, zero_one_loss_torch),
}


def compute_fairness_features(y_true, y_pred, a, 
    metrics = ["D.DP", "D.EqOdds", "D.PRP", "D.Err"],
    X = None,
    ):
    """
    Automatically picks numpy OR torch implementation.
    X : Observations features. Currently unused, reserved  for any future metrics
    Returns a vector of fairness metric values.
    """
    feats = []

    use_torch = _is_torch(y_pred)
    a = a.squeeze(-1)
    #  import ipdb;ipdb.set_trace()
    for m in metrics:
        f_np, f_torch = FAIRNESS_REGISTRY[m]
        if (m == 'D.PRP') or ('D.EqOdds' == m):
            fargs = (y_true, y_pred, a)  
        elif m == 'L.ZeroOne':
            fargs = (y_true, y_pred)  
        else:
            fargs = (y_pred, a)

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
    "L.ZeroOne": zero_one_loss,
}
