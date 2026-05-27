import numpy, torch

def smooth_gamma(gamma, prev_gamma=None, ema=0.9):
    if ema < 0. or ema > 1.:
        raise ValueError(f"Gamma smoothing ema vlas must be in [0,1] but is {ema}, instead.")
    if prev_gamma is None:
        return gamma
    ema = 0. if ema is None else ema
    try:
        return ema * prev_gamma + (1.0 - ema) * gamma
    except:
        import ipdb;ipdb.set_trace()



def compute_subdom_policy_weights_torch(
    S,
    gamma,
    criterion,
    *,
    soft=True,
    temp=0.05,
    normalize=True,
    eps=1e-12,
):
    """
    S:        [R, D]
    gamma:    [R, D]
    criterion:[D] or [1, D]

    Returns rollout/demo weights for:
      term1 = -sum_i rollout_weights[i] * log p(rollout_i)
      term2 = -sum_j demo_weights[j]    * log p(demo_j)
    """
    gamma = to_backend(gamma, S)
    criterion = to_backend(criterion, S).reshape(1, -1)

    if S.shape != gamma.shape:
        raise ValueError(f"S and gamma shape mismatch: {S.shape} vs {gamma.shape}")

    if criterion.shape[1] != S.shape[1]:
        raise ValueError(f"criterion must have length D={S.shape[1]}")

    if soft:
        good = torch.sigmoid((criterion - S) / temp)
    else:
        good = (S <= criterion).to(S.dtype)

    bad = 1.0 - good

    rollout_weights = (gamma * good).sum(dim=1)  # [R]
    demo_weights = (gamma * bad).sum(dim=0)      # [D]

    if normalize:
        rollout_weights = rollout_weights / (rollout_weights.sum() + eps)
        demo_weights = demo_weights / (demo_weights.sum() + eps)

    return {
        "good_score": good,
        "bad_score": bad,
        "rollout_weights": rollout_weights,
        "demo_weights": demo_weights,
        "good_mass": (gamma * good).sum(),
        "bad_mass": (gamma * bad).sum(),
    }
