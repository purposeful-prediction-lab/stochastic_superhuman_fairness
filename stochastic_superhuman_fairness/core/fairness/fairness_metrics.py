import numpy as np
import torch
from typing import Union
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
def _to_bool01(y: np.ndarray) -> np.ndarray:
    """Convert labels to bool {0,1}."""
    y = np.asarray(y).reshape(-1)
    if np.issubdtype(y.dtype, np.floating):
        return y > 0.5
    return y > 0

def _to_bool_a(a: np.ndarray) -> np.ndarray:
    """Convert protected attribute to bool group indicator (a==1)."""
    a = np.asarray(a).reshape(-1)
    if np.issubdtype(a.dtype, np.floating):
        return a > 0.5
    return a > 0

def _rate_with_smoothing(num: float, den: float, alpha: float) -> float:
    """Laplace smoothing: (num + alpha) / (den + 2*alpha) for binary outcomes."""
    return (num + alpha) / (den + 2.0 * alpha)

def demographic_parity(y_pred, a, thresh: float = 0.5) -> float:
    """
    |P(Ŷ=1|A=0) - P(Ŷ=1|A=1)|
    """
    y_pred = np.asarray(y_pred).reshape(-1)
    a1 = _to_bool_a(a)
    y_hat = y_pred > thresh

    # If a group is empty (rare), return 0 gap by convention.
    n0 = (~a1).sum()
    n1 = (a1).sum()
    if n0 == 0 or n1 == 0:
        return 0.0

    p0 = y_hat[~a1].mean()
    p1 = y_hat[a1].mean()
    return float(abs(p0 - p1))

def equalized_odds(y_true, y_pred, a, thresh: float = 0.5, alpha: float = 1.0) -> float:
    """
    0.5*(|TPR0-TPR1| + |FPR0-FPR1|) with Laplace smoothing on each rate.
    """
    y_pred = np.asarray(y_pred).reshape(-1)
    y_true = np.asarray(y_true).reshape(-1)
    a1 = _to_bool_a(a)

    y_hat = y_pred > thresh
    y = _to_bool01(y_true)

    def tpr_fpr(group_mask: np.ndarray):
        # TPR = P(Ŷ=1 | Y=1, A=g)
        y_pos = group_mask & y
        tp = float(np.sum(y_hat & y_pos))
        pos = float(np.sum(y_pos))
        tpr = _rate_with_smoothing(tp, pos, alpha)

        # FPR = P(Ŷ=1 | Y=0, A=g)
        y_neg = group_mask & (~y)
        fp = float(np.sum(y_hat & y_neg))
        neg = float(np.sum(y_neg))
        fpr = _rate_with_smoothing(fp, neg, alpha)

        return tpr, fpr

    tpr0, fpr0 = tpr_fpr(~a1)
    tpr1, fpr1 = tpr_fpr(a1)
    eo = 0.5 * (abs(tpr0 - tpr1) + abs(fpr0 - fpr1))
    return float(eo)

def predictive_rate_parity(y_true, y_pred, a, thresh: float = 0.5, alpha: float = 1.0) -> float:
    """
    |PPV0 - PPV1| where PPV = P(Y=1 | Ŷ=1, A=g), smoothed.
    PPV = (TP + alpha) / (PredPos + 2*alpha)
    """
    y_pred = np.asarray(y_pred).reshape(-1)
    y_true = np.asarray(y_true).reshape(-1)
    a1 = _to_bool_a(a)

    y_hat = y_pred > thresh
    y = _to_bool01(y_true)

    def ppv(group_mask: np.ndarray):
        pred_pos = group_mask & y_hat
        tp = float(np.sum(pred_pos & y))
        pp = float(np.sum(pred_pos))
        return _rate_with_smoothing(tp, pp, alpha)

    ppv0 = ppv(~a1)
    ppv1 = ppv(a1)
    return float(abs(ppv0 - ppv1))

def prediction_error_disparity(y_true, y_pred, a, thresh: float = 0.5, alpha: float = 1.0) -> float:
    """
    |Err0 - Err1| where Err = P(Ŷ != Y | A=g), smoothed as a Bernoulli rate.
    Err = (errors + alpha) / (n_group + 2*alpha)
    """
    y_pred = np.asarray(y_pred).reshape(-1)
    y_true = np.asarray(y_true).reshape(-1)
    a1 = _to_bool_a(a)

    y_hat = y_pred > thresh
    y = _to_bool01(y_true)
    err = (y_hat != y)

    def err_rate(group_mask: np.ndarray):
        e = float(np.sum(err[group_mask]))
        n = float(np.sum(group_mask))
        return _rate_with_smoothing(e, n, alpha)

    e0 = err_rate(~a1)
    e1 = err_rate(a1)
    return float(abs(e0 - e1))

#==========
#  def demographic_parity(y_pred, a):
#      """
#      |P(ŷ=1|A=0) - P(ŷ=1|A=1)|
#      If no model predictions exist, y_pred can be ground truth y.
#      """
#      p0 = _safe_mean(y_pred[a == 0])
#      p1 = _safe_mean(y_pred[a == 1])
#      return abs(p0 - p1)
#
#  def equalized_odds(y_true, y_pred, a):
#      """
#      Average difference in TPR and FPR between groups.
#      |TPR0 - TPR1| + |FPR0 - FPR1| / 2
#      """
#      def rates(y_t, y_p, mask):
#          pos = y_t == 1
#          neg = y_t == 0
#          TPR = np.sum(y_p[pos & mask]) / max(np.sum(pos & mask), 1)
#          FPR = np.sum(y_p[neg & mask]) / max(np.sum(neg & mask), 1)
#          return TPR, FPR
#
#      TPR0, FPR0 = rates(y_true, y_pred, a == 0)
#      TPR1, FPR1 = rates(y_true, y_pred, a == 1)
#      return 0.5 * (abs(TPR0 - TPR1) + abs(FPR0 - FPR1))
#
#  def predictive_rate_parity(y_true, y_pred, a):
#      """
#      |P(Y=1|A=0, ŷ=1) - P(Y=1|A=1, ŷ=1)|
#      """
#      def precision(y_t, y_p, mask):
#          pred_pos = y_p == 1
#          return np.sum(y_t[pred_pos & mask]) / max(np.sum(pred_pos & mask), 1)
#      prec0 = precision(y_true, y_pred, a == 0)
#      prec1 = precision(y_true, y_pred, a == 1)
#      return abs(prec0 - prec1)
#
#  def prediction_error_disparity(y_true, y_pred, a):
#      """
#      Difference in total error rate across groups.
#      """
#      err0 = _safe_mean((y_pred != y_true)[a == 0])
#      err1 = _safe_mean((y_pred != y_true)[a == 1])
#      return abs(err0 - err1)
#
#  # Other Metrics ----------------------------------------------
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
def _to_bool01_torch(y: torch.Tensor) -> torch.Tensor:
    """Convert labels to bool {0,1}."""
    y = y.view(-1)
    if y.is_floating_point():
        return (y > 0.5)
    return (y > 0)

def _to_bool_a_torch(a: torch.Tensor) -> torch.Tensor:
    """Convert protected attribute to bool group indicator (a==1)."""
    try:
        return (a.view(-1) > 0)
    except:
        import ipdb;ipdb.set_trace()

def _rate_with_smoothing_torch(num: torch.Tensor, den: torch.Tensor, alpha: float) -> torch.Tensor:
    """Laplace smoothing: (num + alpha) / (den + 2*alpha) for binary outcomes."""
    return (num + alpha) / (den + 2.0 * alpha)

def demographic_parity_torch(y_pred, a, thresh: float = 0.5) -> torch.Tensor:
    """
    |P(Ŷ=1|A=0) - P(Ŷ=1|A=1)|  (no smoothing needed, but you can add if desired)
    """
    y_pred = y_pred.view(-1)
    a1 = _to_bool_a_torch(a)
    y_hat = (y_pred > thresh)

    p0 = y_hat[~a1].float().mean()
    p1 = y_hat[a1].float().mean()
    return torch.abs(p0 - p1)

def equalized_odds_torch(y_true, y_pred, a, thresh: float = 0.5, alpha: float = 1.0) -> torch.Tensor:
    """
    0.5*(|TPR0-TPR1| + |FPR0-FPR1|) with Laplace smoothing on each rate.
    """
    y_pred = y_pred.view(-1)
    y_true = y_true.view(-1)
    a1 = _to_bool_a_torch(a)

    y_hat = (y_pred > thresh)
    y = _to_bool01_torch(y_true)

    def tpr_fpr(group_mask: torch.Tensor):
        # TPR = P(Ŷ=1 | Y=1, A=g)
        y_pos = group_mask & y
        tp = (y_hat & y_pos).sum().float()
        pos = y_pos.sum().float()
        tpr = _rate_with_smoothing_torch(tp, pos, alpha)

        # FPR = P(Ŷ=1 | Y=0, A=g)
        y_neg = group_mask & (~y)
        fp = (y_hat & y_neg).sum().float()
        neg = y_neg.sum().float()
        fpr = _rate_with_smoothing_torch(fp, neg, alpha)
        return tpr, fpr

    tpr0, fpr0 = tpr_fpr(~a1)
    tpr1, fpr1 = tpr_fpr(a1)
    return 0.5 * (torch.abs(tpr0 - tpr1) + torch.abs(fpr0 - fpr1))

def predictive_rate_parity_torch(y_true, y_pred, a, thresh: float = 0.5, alpha: float = 1.0) -> torch.Tensor:
    """
    |PPV0 - PPV1| where PPV = P(Y=1 | Ŷ=1, A=g), smoothed.
    PPV = (TP + alpha) / (PredPos + 2*alpha)
    """
    y_pred = y_pred.view(-1)
    y_true = y_true.view(-1)
    a1 = _to_bool_a_torch(a)

    y_hat = (y_pred > thresh)
    y = _to_bool01_torch(y_true)

    def ppv(group_mask: torch.Tensor):
        pred_pos = group_mask & y_hat
        tp = (pred_pos & y).sum().float()
        pp = pred_pos.sum().float()
        return _rate_with_smoothing_torch(tp, pp, alpha)

    ppv0 = ppv(~a1)
    ppv1 = ppv(a1)
    return torch.abs(ppv0 - ppv1)

def prediction_error_disparity_torch(y_true, y_pred, a, thresh: float = 0.5, alpha: float = 1.0) -> torch.Tensor:
    """
    |Err0 - Err1| where Err = P(Ŷ != Y | A=g), smoothed as a Bernoulli rate.
    Err = (errors + alpha) / (n_group + 2*alpha)
    """
    y_pred = y_pred.view(-1)
    y_true = y_true.view(-1)
    a1 = _to_bool_a_torch(a)

    y_hat = (y_pred > thresh)
    y = _to_bool01_torch(y_true)
    err = (y_hat != y)

    def err_rate(group_mask: torch.Tensor):
        e = err[group_mask].sum().float()
        n = group_mask.sum().float()
        return _rate_with_smoothing_torch(e, n, alpha)

    e0 = err_rate(~a1)
    e1 = err_rate(a1)
    return torch.abs(e0 - e1)
#=========

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

def compute_fairness_features(
    y_true, y_pred, a,
    metrics=["D.DP", "D.EqOdds", "D.PRP", "D.Err"],
    X=None,
    weights: Union[list, np.ndarray, torch.Tensor] = None,
):
    feats = []
    use_torch = _is_torch(y_pred)
    a = a.squeeze(-1) if len(a.shape) > 1 else a.squeeze()

    for m in metrics:
        f_np, f_torch = FAIRNESS_REGISTRY[m]
        if (m == "D.PRP") or (m == "D.EqOdds") or (m == "D.Err"):
            fargs = (y_true, y_pred, a)
        elif m == "L.ZeroOne":
            fargs = (y_true, y_pred)
        else:
            fargs = (y_pred, a)

        feats.append(f_torch(*fargs) if use_torch else f_np(*fargs))

    # stack in correct backend
    if use_torch:
        feats_vec = torch.stack(feats)
    else:
        feats_vec = np.array(feats, dtype=np.float32)

    # apply weights (same backend) if provided
    if weights is not None:
        if len(weights) != len(metrics):
            raise ValueError(f"weights must have length {len(metrics)} (got {len(weights)})")

        if use_torch:
            w = weights if torch.is_tensor(weights) else torch.tensor(weights, dtype=feats_vec.dtype, device=feats_vec.device)
            w = w.to(dtype=feats_vec.dtype, device=feats_vec.device)
            feats_vec = feats_vec * w
        else:
            w = weights.detach().cpu().numpy() if torch.is_tensor(weights) else np.asarray(weights, dtype=feats_vec.dtype)
            feats_vec = feats_vec * w

    return feats_vec

# ---------------------------------------------------------------------------------

def compute_fairness_features_batched(
    y_true,
    y_pred_batch,   # [R, N] or [P, M, N]
    a,
    metrics=("D.DP", "D.EqOdds", "D.PRP", "D.Err"),
    X=None,
    weights=None,
):
    """
    Batched version of compute_fairness_features.

    Args:
        y_true: [N]
        y_pred_batch: [R, N] or any shape [..., N]
        a: [N]
    Returns:
        feats: [R, K] or [..., K]
    """
    if not torch.is_tensor(y_pred_batch):
        raise TypeError("compute_fairness_features_batched currently expects torch tensors")

    original_shape = y_pred_batch.shape[:-1]   # e.g. [R] or [P, M]
    N = y_pred_batch.shape[-1]
    y_pred_flat = y_pred_batch.reshape(-1, N)  # [B, N]

    def _single(y_pred_single):
        return compute_fairness_features(
            y_true=y_true,
            y_pred=y_pred_single,
            a=a,
            metrics=metrics,
            X=X,
            weights=weights,
        )  # [K]

    # vmap over the rollout dimension
    feats_flat = torch.vmap(_single)(y_pred_flat)   # [B, K]

    K = feats_flat.shape[-1]
    feats = feats_flat.reshape(*original_shape, K)
    return feats
# ---------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------
METRIC_REGISTRY = {
    "D.DP": lambda y_true, y_pred, a: demographic_parity(y_pred, a),
    "D.EqOdds": equalized_odds,
    "D.PRP": predictive_rate_parity,
    "D.Err": lambda y_true, y_pred, a: prediction_error_disparity(y_true, y_pred, a),
    "L.ZeroOne": zero_one_loss,
}

#=============================================================================================================
# Batched VErsions
#=============================================================================================================

def _to_bool01_torch_batched(y: torch.Tensor) -> torch.Tensor:
    y = y.reshape(-1)
    if y.is_floating_point():
        return y > 0.5
    return y > 0


def _to_bool_a_torch_batched(a: torch.Tensor) -> torch.Tensor:
    return a.reshape(-1) > 0


def _rate_with_smoothing_torch(num: torch.Tensor, den: torch.Tensor, alpha: float) -> torch.Tensor:
    return (num + alpha) / (den + 2.0 * alpha)


def demographic_parity_torch_batched(
    y_pred_batch: torch.Tensor,   # [B, N]
    a: torch.Tensor,              # [N]
    thresh: float = 0.5,
) -> torch.Tensor:
    a1 = _to_bool_a_torch_batched(a)                  # [N]
    y_hat = y_pred_batch > thresh                     # [B, N]

    g0 = (~a1).unsqueeze(0)                           # [1, N]
    g1 = a1.unsqueeze(0)                              # [1, N]

    n0 = g0.sum(dim=1).clamp_min(1).float()          # [1]
    n1 = g1.sum(dim=1).clamp_min(1).float()          # [1]

    p0 = (y_hat & g0).sum(dim=1).float() / n0        # [B]
    p1 = (y_hat & g1).sum(dim=1).float() / n1        # [B]
    return (p0 - p1).abs()


def equalized_odds_torch_batched(
    y_true: torch.Tensor,         # [N]
    y_pred_batch: torch.Tensor,   # [B, N]
    a: torch.Tensor,              # [N]
    thresh: float = 0.5,
    alpha: float = 1.0,
) -> torch.Tensor:
    y = _to_bool01_torch_batched(y_true).unsqueeze(0)    # [1, N]
    a1 = _to_bool_a_torch_batched(a).unsqueeze(0)        # [1, N]
    y_hat = y_pred_batch > thresh                        # [B, N]

    g0 = ~a1
    g1 = a1
    y_pos = y
    y_neg = ~y

    tp0 = (y_hat & g0 & y_pos).sum(dim=1).float()
    pos0 = (g0 & y_pos).sum(dim=1).float()
    fp0 = (y_hat & g0 & y_neg).sum(dim=1).float()
    neg0 = (g0 & y_neg).sum(dim=1).float()

    tp1 = (y_hat & g1 & y_pos).sum(dim=1).float()
    pos1 = (g1 & y_pos).sum(dim=1).float()
    fp1 = (y_hat & g1 & y_neg).sum(dim=1).float()
    neg1 = (g1 & y_neg).sum(dim=1).float()

    tpr0 = _rate_with_smoothing_torch(tp0, pos0, alpha)
    fpr0 = _rate_with_smoothing_torch(fp0, neg0, alpha)
    tpr1 = _rate_with_smoothing_torch(tp1, pos1, alpha)
    fpr1 = _rate_with_smoothing_torch(fp1, neg1, alpha)

    return 0.5 * ((tpr0 - tpr1).abs() + (fpr0 - fpr1).abs())


def predictive_rate_parity_torch_batched(
    y_true: torch.Tensor,         # [N]
    y_pred_batch: torch.Tensor,   # [B, N]
    a: torch.Tensor,              # [N]
    thresh: float = 0.5,
    alpha: float = 1.0,
) -> torch.Tensor:
    y = _to_bool01_torch_batched(y_true).unsqueeze(0)    # [1, N]
    a1 = _to_bool_a_torch_batched(a).unsqueeze(0)        # [1, N]
    y_hat = y_pred_batch > thresh                        # [B, N]

    g0 = ~a1
    g1 = a1

    pred_pos0 = y_hat & g0
    pred_pos1 = y_hat & g1

    tp0 = (pred_pos0 & y).sum(dim=1).float()
    pp0 = pred_pos0.sum(dim=1).float()
    tp1 = (pred_pos1 & y).sum(dim=1).float()
    pp1 = pred_pos1.sum(dim=1).float()

    ppv0 = _rate_with_smoothing_torch(tp0, pp0, alpha)
    ppv1 = _rate_with_smoothing_torch(tp1, pp1, alpha)
    return (ppv0 - ppv1).abs()


def prediction_error_disparity_torch_batched(
    y_true: torch.Tensor,         # [N]
    y_pred_batch: torch.Tensor,   # [B, N]
    a: torch.Tensor,              # [N]
    thresh: float = 0.5,
    alpha: float = 1.0,
) -> torch.Tensor:
    y = _to_bool01_torch_batched(y_true).unsqueeze(0)    # [1, N]
    a1 = _to_bool_a_torch_batched(a).unsqueeze(0)        # [1, N]
    y_hat = y_pred_batch > thresh                        # [B, N]
    err = y_hat != y                                     # [B, N]

    g0 = ~a1
    g1 = a1

    e0 = (err & g0).sum(dim=1).float()
    n0 = g0.sum(dim=1).float()
    e1 = (err & g1).sum(dim=1).float()
    n1 = g1.sum(dim=1).float()

    r0 = _rate_with_smoothing_torch(e0, n0, alpha)
    r1 = _rate_with_smoothing_torch(e1, n1, alpha)
    return (r0 - r1).abs()


def zero_one_loss_torch_batched(
    y_true: torch.Tensor,         # [N]
    y_pred_batch: torch.Tensor,   # [B, N]
) -> torch.Tensor:
    y_true = y_true.reshape(1, -1).float()
    if y_pred_batch.dtype.is_floating_point:
        y_pred_batch = (y_pred_batch > 0.5).float()
    return (y_pred_batch != y_true).float().mean(dim=1)


def compute_fairness_features_batched_fast(
    y_true,
    y_pred_batch,   # [B, N] or [P, M, N]
    a,
    metrics=("D.DP", "D.EqOdds", "D.PRP", "D.Err"),
    X=None,
    weights: Union[list, np.ndarray, torch.Tensor] = None,
):
    if not torch.is_tensor(y_pred_batch):
        raise TypeError("compute_fairness_features_batched_fast expects torch tensors")

    original_shape = y_pred_batch.shape[:-1]
    N = y_pred_batch.shape[-1]
    y_pred_flat = y_pred_batch.reshape(-1, N)  # [B, N]

    feats = []
    for m in metrics:
        if m == "D.DP":
            feats.append(demographic_parity_torch_batched(y_pred_flat, a))
        elif m == "D.EqOdds":
            feats.append(equalized_odds_torch_batched(y_true, y_pred_flat, a))
        elif m == "D.PRP":
            feats.append(predictive_rate_parity_torch_batched(y_true, y_pred_flat, a))
        elif m == "D.Err":
            feats.append(prediction_error_disparity_torch_batched(y_true, y_pred_flat, a))
        elif m == "L.ZeroOne":
            feats.append(zero_one_loss_torch_batched(y_true, y_pred_flat))
        else:
            raise ValueError(f"Unsupported metric: {m}")

    feats_flat = torch.stack(feats, dim=1)  # [B, K]

    if weights is not None:
        if len(weights) != len(metrics):
            raise ValueError(f"weights must have length {len(metrics)} (got {len(weights)})")
        if not torch.is_tensor(weights):
            weights = torch.tensor(weights, dtype=feats_flat.dtype, device=feats_flat.device)
        else:
            weights = weights.to(dtype=feats_flat.dtype, device=feats_flat.device)
        feats_flat = feats_flat * weights.unsqueeze(0)

    K = feats_flat.shape[-1]
    return feats_flat.reshape(*original_shape, K)
