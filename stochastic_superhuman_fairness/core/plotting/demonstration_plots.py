import numpy as np
import torch
import matplotlib.pyplot as plt


def _to_numpy(x):
    if torch.is_tensor(x):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def plot_label_agreement_stats(
    agreement_stats,
    *,
    axes=None,  # None | (ax_stats, ax_heatmap) | dict with {"stats": ax, "heatmap": ax}
    split="train",  # "train" | "eval"
    title=None,
    figsize=(11, 4),
    heatmap_key="sorted_pairwise_agreement",
    fallback_heatmap_key="pairwise_agreement",
    heatmap_norm="global",  # "global" | "row" | None
    cmap="viridis",
    show_colorbar=True,
):
    """
    Plot label agreement summary stats + sorted pairwise agreement heatmap.

    agreement_stats format:
        {
            "train": {
                "pairwise_agreement": [M, M],
                "sorted_pairwise_agreement": [M, M], optional,
                "mean_agreement": scalar,
                "std_agreement": scalar,
                "median_agreement": scalar,
            },
            "eval": {...}
        }

    Returns:
        fig, (ax_stats, ax_heatmap)
    """
    if split not in agreement_stats:
        raise KeyError(f"split={split!r} not found. Available: {list(agreement_stats.keys())}")

    stats = agreement_stats[split]

    if axes is None:
        fig, (ax_stats, ax_heatmap) = plt.subplots(1, 2, figsize=figsize)
    elif isinstance(axes, dict):
        ax_stats = axes["stats"]
        ax_heatmap = axes["heatmap"]
        fig = ax_stats.figure
    else:
        ax_stats, ax_heatmap = axes
        fig = ax_stats.figure

    mean = float(_to_numpy(stats["mean_agreement"]))
    median = float(_to_numpy(stats["median_agreement"]))
    std = float(_to_numpy(stats["std_agreement"]))

    # ---- summary graph
    xs = np.arange(3)
    ys = np.array([mean, median, std])
    labels = ["mean", "median", "std"]

    ax_stats.bar(xs, ys)
    ax_stats.set_xticks(xs)
    ax_stats.set_xticklabels(labels)
    ax_stats.set_ylim(0, max(1.0, ys.max() * 1.1))
    ax_stats.set_ylabel("Agreement")
    ax_stats.set_title(f"{split} agreement stats")

    for x, y in zip(xs, ys):
        ax_stats.text(x, y, f"{y:.3f}", ha="center", va="bottom")

    # ---- heatmap
    if heatmap_key in stats:
        A = _to_numpy(stats[heatmap_key]).astype(float)
    else:
        A = _to_numpy(stats[fallback_heatmap_key]).astype(float)

    if heatmap_norm == "row":
        row_min = A.min(axis=1, keepdims=True)
        row_max = A.max(axis=1, keepdims=True)
        denom = np.maximum(row_max - row_min, 1e-12)
        A_plot = (A - row_min) / denom
        vmin, vmax = 0.0, 1.0
        cbar_label = "Row-normalized agreement"

    elif heatmap_norm == "global":
        A_plot = A
        vmin, vmax = A.min(), A.max()
        cbar_label = "Agreement"

    elif heatmap_norm is None:
        A_plot = A
        vmin, vmax = None, None
        cbar_label = "Agreement"

    else:
        raise ValueError("heatmap_norm must be 'global', 'row', or None")

    im = ax_heatmap.imshow(A_plot, aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax)
    ax_heatmap.set_title(f"{split} sorted pairwise agreement")
    ax_heatmap.set_xlabel("Model")
    ax_heatmap.set_ylabel("Model")

    if show_colorbar:
        cbar = fig.colorbar(im, ax=ax_heatmap)
        cbar.set_label(cbar_label)

    if title is not None:
        fig.suptitle(title)

    fig.tight_layout()
    return fig, (ax_stats, ax_heatmap)
