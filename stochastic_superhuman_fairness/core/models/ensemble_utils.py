import torch
import torch.nn as nn
from collections.abc import Iterable

def mix_policies_(
    model: nn.Module,
    good_idxs,
    bad_idxs,
    alpha: float = 0.2,
    mode: str = "one_to_many",   # "one_to_one", "one_to_many", "mean_to_many"
):
    """
    In-place permanent convex mixing of policy parameters.

    Update:
        bad <- (1 - alpha) * bad + alpha * good

    Assumes:
        model.policies is an indexable collection of same-structure modules.
    """
    if not hasattr(model, "policies"):
        raise AttributeError("model must have `policies`")

    if isinstance(good_idxs, int):
        good_idxs = [good_idxs]
    if isinstance(bad_idxs, int):
        bad_idxs = [bad_idxs]

    if not (0.0 <= alpha <= 1.0):
        raise ValueError(f"alpha must be in [0,1], got {alpha}")

    policies = model.policies
    n = len(policies)

    for i in [*good_idxs, *bad_idxs]:
        if not (0 <= i < n):
            raise IndexError(f"policy index {i} out of range for {n} policies")

    def _mix_modules_(src: nn.Module, dst: nn.Module, alpha: float):
        src_named = dict(src.named_parameters())
        with torch.no_grad():
            for name, p_dst in dst.named_parameters():
                p_src = src_named[name]
                if p_dst.shape != p_src.shape:
                    raise ValueError(
                        f"shape mismatch for {name}: {tuple(p_dst.shape)} vs {tuple(p_src.shape)}"
                    )
                # true in-place write into the destination parameter
                p_dst.mul_(1.0 - alpha).add_(p_src, alpha=alpha)

    if mode == "one_to_one":
        if len(good_idxs) != len(bad_idxs):
            raise ValueError("one_to_one requires len(good_idxs) == len(bad_idxs)")
        for g, b in zip(good_idxs, bad_idxs):
            _mix_modules_(policies[g], policies[b], alpha)

    elif mode == "one_to_many":
        src = policies[good_idxs[0]]
        for b in bad_idxs:
            _mix_modules_(src, policies[b], alpha)

    elif mode == "mean_to_many":
        srcs = [policies[i] for i in good_idxs]
        src_named_list = [dict(m.named_parameters()) for m in srcs]

        with torch.no_grad():
            for b in bad_idxs:
                for name, p_dst in policies[b].named_parameters():
                    src_group = [sd[name] for sd in src_named_list]
                    for p_src in src_group:
                        if p_src.shape != p_dst.shape:
                            raise ValueError(
                                f"shape mismatch for {name}: {tuple(p_dst.shape)} vs {tuple(p_src.shape)}"
                            )
                    mean_src = torch.stack(src_group, dim=0).mean(dim=0)
                    p_dst.mul_(1.0 - alpha).add_(mean_src, alpha=alpha)
    else:
        raise ValueError(f"unknown mode: {mode}")

# Optimizer related
# Params checker
def reset_optimizer_state_(optimizer, params):
    param_ids = {id(p) for p in params}

    for group in optimizer.param_groups:
        for p in group["params"]:
            if id(p) in param_ids:
                optimizer.state.pop(p, None)   # safest
def params_changed(
    model,
    fn,
    fn_args,
    atol=1e-8,
    rtol=1e-5,
    return_details=False,
    **fn_kwargs,
):
    """
    Runs fn(*args, **kwargs) and checks if any policy parameters changed.

    Returns:
        changed (bool) OR (changed, details dict)
    """

    # snapshot BEFORE (clone to avoid aliasing)
    before = {
        (i, name): p.detach().clone()
        for i, policy in enumerate(model.policies)
        for name, p in policy.named_parameters()
    }

    # run update function
    fn(*fn_args, **fn_kwargs)

    # compare AFTER
    changed = False
    details = {}

    for i, policy in enumerate(model.policies):
        for name, p in policy.named_parameters():
            key = (i, name)
            p_before = before[key]

            same = torch.allclose(p_before, p, atol=atol, rtol=rtol)

            if not same:
                changed = True
                if return_details:
                    diff = (p - p_before).abs().max().item()
                    details[key] = diff

    if return_details:
        return changed, details
    return changed
