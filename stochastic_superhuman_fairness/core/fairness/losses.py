import torch
import torch.nn.functional as F

def ensemble_bc_loss(logits_policies, y_demo, demo_weights=None, policy_weights=None):
    """
    logits_policies: [P, N]
    y_demo:          [D, N]
    demo_weights:    [D] optional
    policy_weights:  [P] optional
    """
    P, N = logits_policies.shape
    D = y_demo.shape[0]

    logits = logits_policies[:, None, :].expand(P, D, N)  # [P,D,N]
    targets = y_demo[None, :, :].expand(P, D, N)          # [P,D,N]

    bce = F.binary_cross_entropy_with_logits(
        logits,
        targets,
        reduction="none",
    ).mean(dim=-1)  # [P,D]

    if demo_weights is None:
        demo_weights = torch.ones(D, device=logits_policies.device) / D
    else:
        demo_weights = demo_weights / demo_weights.sum().clamp_min(1e-12)

    if policy_weights is None:
        policy_weights = torch.ones(P, device=logits_policies.device) / P
    else:
        policy_weights = policy_weights / policy_weights.sum().clamp_min(1e-12)

    loss_per_policy = (bce * demo_weights[None, :]).sum(dim=1)  # [P]
    loss = (loss_per_policy * policy_weights).sum()

    return loss, {
        "bce_policy_demo": bce,
        "loss_per_policy": loss_per_policy,
        "demo_weights": demo_weights,
        "policy_weights": policy_weights,
    }
