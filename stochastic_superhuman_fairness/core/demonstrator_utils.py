import torch
from pathlib import Path

class AgreementStatsMixin:
    """
    Mixin for computing + saving/loading label agreement stats.
    Assumes `labels` are provided externally or via self.
    """

    # ---------- core compute ----------
    def compute_label_agreement(
        self,
        labels: torch.Tensor,
        *,
        return_disagreement_per_sample: bool = False,
        exclude_self: bool = True,
    ):
        if labels.ndim != 2:
            raise ValueError(f"labels must be [M, N], got {tuple(labels.shape)}")

        M, N = labels.shape
        if M < 2:
            return {
                "pairwise_agreement": 1,
                "mean_agreement": 1,
                "std_agreement": 0,
                "median_agreement": 1,
            }
            #  raise ValueError("Need at least 2 models")

        agree = labels[:, None, :] == labels[None, :, :]
        pairwise = agree.float().mean(dim=-1)

        if exclude_self:
            mask = ~torch.eye(M, dtype=torch.bool, device=labels.device)
            vals = pairwise[mask]
        else:
            vals = pairwise.reshape(-1)

        out = {
            "pairwise_agreement": pairwise,
            "mean_agreement": vals.mean(),
            "std_agreement": vals.std(unbiased=False),
            "median_agreement": vals.median(),
        }

        if return_disagreement_per_sample:
            denom = M - 1 if exclude_self else M
            disagree = (~agree).float().sum(dim=1) / denom
            out["disagreement_per_sample"] = disagree  # [M, N]

        return out

    # ---------- saving ----------
    def save_agreement_stats(self, stats: dict, path: str):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        stats_cpu = {
            k: (v.detach().cpu() if torch.is_tensor(v) else v)
            for k, v in stats.items()
        }

        torch.save(stats_cpu, path)

    # ---------- loading ----------
    def load_agreement_stats(self, path: str, device=None):
        stats = torch.load(path, map_location="cpu")

        if device is not None:
            for k, v in stats.items():
                if torch.is_tensor(v):
                    stats[k] = v.to(device)

        return stats


# Diagnostic Sets

