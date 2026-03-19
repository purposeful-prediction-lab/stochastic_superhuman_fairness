from dataclasses import dataclass
import torch
import torch.nn.functional as F

@dataclass
class SubdomLossOut:
    loss: torch.Tensor
    info: dict

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
    ).mean(dim=1)  # [R]

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
    ).mean(dim=2)  # [R,D]

    w_lose_ij = gamma * (1.0 - I)           # [R,D]
    term2 = (w_lose_ij * bce_pair).sum()

    loss = term1 + term2
    #  loss = term2
    info = {
        "w_win_sum": float(w_win_i.sum().detach().cpu()),
        "w_lose_sum": float(w_lose_ij.sum().detach().cpu()),
        "indicator_mean": float(I.mean().detach().cpu()),
        "term1": float(term1.detach().cpu()),
        "term2": float(term2.detach().cpu()),
    }
    return SubdomLossOut(loss=loss, info=info)

def subdominant_weighted_logloss_shared_X_multi_rollout(
    *,
    S,                                # [R,D]  rollout-demo subdominance matrix
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
    ).mean(dim=1)  # [R]

    w_win_i = (gamma * I * S).sum(dim=1)  # [R]
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
    ).mean(dim=2)  # [R,D]

    w_lose_ij = gamma * (1.0 - I) * S          # [R,D]
    term2 = (w_lose_ij * bce_pair).sum()

    #  loss = term1 + term2
    loss = term2
    info = {
        "w_win_sum": float(w_win_i.sum().detach().cpu()),
        "w_lose_sum": float(w_lose_ij.sum().detach().cpu()),
        "indicator_mean": float(I.mean().detach().cpu()),
        "term1": float(term1.detach().cpu()),
        "term2": float(term2.detach().cpu()),
    }
    return SubdomLossOut(loss=loss, info=info)
