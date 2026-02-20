import numpy as np
import matplotlib.pyplot as plt
import math
import torch

from stochastic_superhuman_fairness.core.plotting.plot_utils import (
    compute_feature_means,
    annotate_mean_with_guides,
    inv_alpha_for_dims,
    annotate_inverse_alpha_arrows,
)

def plot_rollouts_vs_demos(
    rollouts: np.ndarray,
    demos: np.ndarray,
    feature_names=None,
    alpha=None,
    beta=None,
    *,
    pairs=None,
    rollout_label="rollouts",
    demo_label="demos",
    title=None,
    s_rollouts=10,
    s_demos=25,
    marker_rollouts="o",
    marker_demos="x",
    return_artists: bool = False,
):
    """
    Plot rollouts (R,K) and demos (D,K) as scatter plots.

    Conventions per subplot (i,j):
      - x = feature j
      - y = feature i

    New behavior:
      - compute rollout feature means once per call
      - plot mean point (mean[j], mean[i]) + solid guide lines to (xmin, ymin)
      - plot inverse alpha lines:
          vline at x = 1/alpha[j]
          hline at y = 1/alpha[i]
      - draw dotted <-> arrows from mean to those inverse-alpha locations
        and label them as 1/alpha_{feature_name}

    Modes:
      - pairs=None:
          If K==2: single plot.
          If K>2: upper-triangular matrix (fi vs fj for j>i).
      - pairs=[(i,j), ...]:
          Plot only specified feature pairs (fi vs fj).

    If return_artists=True, returns a third value:
      artists: list of dicts, each like:
        {"pair": (i,j), "rollouts": PathCollection, "demos": PathCollection}
    """
    rollouts = np.asarray(rollouts)
    demos = np.asarray(demos)

    if rollouts.ndim != 2 or demos.ndim != 2:
        raise ValueError("rollouts and demos must be 2D arrays.")
    if rollouts.shape[1] != demos.shape[1]:
        raise ValueError("rollouts and demos must have same feature dimension.")

    K = rollouts.shape[1]
    if K < 2:
        raise ValueError("K must be >= 2.")

    if feature_names is None:
        feature_names = [f"f{i}" for i in range(K)]
    if len(feature_names) != K:
        raise ValueError("feature_names must have length K.")

    def _broadcast(v):
        if v is None:
            return None
        if np.isscalar(v):
            return np.full(K, float(v))
        v = np.asarray(v, dtype=float).reshape(-1)
        if v.size != K:
            raise ValueError(f"alpha/beta must be scalar or length-K (K={K}). Got {v.size}.")
        return v

    alpha_v = _broadcast(alpha)
    beta_v = _broadcast(beta)

    # Compute rollout mean once per call
    rollout_mean = compute_feature_means(rollouts)  # [K]

    artists = [] if return_artists else None

    def _scatter(ax, i, j):
        # rollouts: semi-transparent
        sc_r = ax.scatter(
            rollouts[:, j], rollouts[:, i],
            s=s_rollouts,
            marker=marker_rollouts,
            label=rollout_label,
            alpha=0.35,
            zorder=1,
        )

        # demos: opaque
        sc_d = ax.scatter(
            demos[:, j], demos[:, i],
            s=s_demos,
            marker=marker_demos,
            label=demo_label,
            alpha=1.0,
            zorder=2,
        )

        # IMPORTANT: autoscale must consider scatter collections
        ax.autoscale(enable=True, axis="both", tight=False)

        mean_x = float(rollout_mean[j])
        mean_y = float(rollout_mean[i])

        # Draw inverse-alpha lines/arrows/labels first (this also expands limits)
        if alpha_v is not None:
            inv_x, inv_y = inv_alpha_for_dims(alpha_v, j, i)  # x=j, y=i
            annotate_inverse_alpha_arrows(
                ax,
                mean_x=mean_x,
                mean_y=mean_y,
                inv_alpha_x=inv_x,
                inv_alpha_y=inv_y,
                x_name=feature_names[j],
                y_name=feature_names[i],
            )

        # Now draw mean dot + guides to the UPDATED xmax/ymax
        annotate_mean_with_guides(
            ax,
            mean_x,
            mean_y,
            s=70,
            marker="o",
            lw=1.6,
            zorder=8,   # keep it above everything
        )

        # Labels
        ax.set_xlabel(feature_names[j])
        ax.set_ylabel(feature_names[i])

        return sc_r, sc_d
    # -------------------------
    # Explicit pairs mode
    # -------------------------
    if pairs is not None:
        pairs = [(int(i), int(j)) for i, j in pairs]
        for (i, j) in pairs:
            if not (0 <= i < K and 0 <= j < K):
                raise IndexError(f"Invalid feature indices ({i}, {j}) for K={K}")

        n = len(pairs)
        ncols = int(math.ceil(math.sqrt(n)))
        nrows = int(math.ceil(n / ncols))

        fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 4 * nrows))
        axes = np.atleast_1d(axes).ravel()

        for ax, (i, j) in zip(axes, pairs):
            sc_r, sc_d = _scatter(ax, i, j)
            if return_artists:
                artists.append({"pair": (i, j), "rollouts": sc_r, "demos": sc_d})

        for ax in axes[len(pairs):]:
            ax.axis("off")

        # legend on first axis
        axes[0].legend()
        if title:
            fig.suptitle(title)
        fig.tight_layout()

        return (fig, axes, artists) if return_artists else (fig, axes)

    # -------------------------
    # Default behavior
    # -------------------------
    if K == 2:
        fig, ax = plt.subplots(1, 1, figsize=(6, 5))
        sc_r, sc_d = _scatter(ax, 0, 1)
        ax.legend()
        if title:
            ax.set_title(title)
        fig.tight_layout()
        if return_artists:
            artists.append({"pair": (0, 1), "rollouts": sc_r, "demos": sc_d})
            return fig, ax, artists
        return fig, ax

    # Upper-triangular matrix for K>2
    n = K - 1
    fig, axes = plt.subplots(n, n, figsize=(3.2 * n, 3.2 * n), squeeze=False)

    for i in range(n):
        for j in range(n):
            ax = axes[i, j]
            x_feat = j + 1
            y_feat = i
            if x_feat <= y_feat:
                ax.axis("off")
                continue
            sc_r, sc_d = _scatter(ax, y_feat, x_feat)
            if return_artists:
                artists.append({"pair": (y_feat, x_feat), "rollouts": sc_r, "demos": sc_d})

    # legend on first active axis
    for i in range(n):
        for j in range(n):
            if axes[i, j].has_data():
                axes[i, j].legend()
                break
        else:
            continue
        break

    if title:
        fig.suptitle(title)
    fig.tight_layout()

    return (fig, axes, artists) if return_artists else (fig, axes)



def plot_zero_one_vs_features_subplots(
    rb,                          # RolloutBatch-like
    demos,                       # list[dict] with keys incl. 'fairness_feats', 'zero_one_loss'
    *,
    feature_names=None,          # list[str] length K
    alpha=None,                  # None, scalar, (K,), (K+1,), or (R,K) (we reduce to per-feature mean)
    title=None,
    rollout_label="rollouts",
    demo_label="demos",
    s_rollouts=12,
    s_demos=28,
    marker_rollouts="o",
    marker_demos="x",
    figsize_per_ax=(4.0, 3.5),
    return_artists: bool = False,
):
    """
    Make a single figure with K subplots. Subplot k shows:
        x = rollout/demo feature k
        y = zero_one_loss

    Rollouts come from:
        x_r = rb.feats[:,k]
        y_r = rb.zero_one_losses (length R)

    Demos come from list of dicts:
        x_d = demo['fairness_feats'][k]
        y_d = demo['zero_one_loss']

    Annotations (like plot_rollouts_vs_demos):
      - bold rollout-mean dot + solid guides to right/up
      - inverse-alpha origin at (1/alpha_x, 1/alpha_y) with rays right/up
      - dotted <-> arrows (offset) from mean to inverse-alpha point + labels

    Alpha handling:
      - alpha is scalar -> broadcast to K (x only; no y alpha unless K+1 provided)
      - alpha is (K,) -> x alpha for each subplot; no y alpha
      - alpha is (K+1,) -> last entry is alpha_loss (used as y alpha for all subplots)
      - alpha is (R,K) -> reduced to per-feature mean for x alpha; y alpha from (K+1) not applicable

    Returns:
      (fig, axes) or (fig, axes, artists)
    """
    # --- pull rollouts to numpy ---
    feats = rb.feats
    Xr = feats.detach().cpu().numpy() if torch.is_tensor(feats) else np.asarray(feats)
    y_r = rb.zero_one_losses
    y_r = y_r.detach().cpu().numpy() if torch.is_tensor(y_r) else np.asarray(y_r, dtype=float)

    R, K = Xr.shape
    if y_r.shape[0] != R:
        raise ValueError(f"rb.zero_one_losses must have length R={R}. Got {y_r.shape[0]}.")

    if feature_names is None:
        feature_names = [f"f{i}" for i in range(K)]
    if len(feature_names) != K:
        raise ValueError(f"feature_names must have length K={K}.")

    # --- demos to numpy arrays (DxK) and (D,) ---
    Xd_list, yd_list = [], []
    for d in demos:
        ff = d.get("fairness_feats", None)
        zl = d.get("zero_one_loss", None)
        if ff is None or zl is None:
            continue
        ff = ff.detach().cpu().numpy() if torch.is_tensor(ff) else np.asarray(ff, dtype=float)
        ff = ff.reshape(-1)
        if ff.size != K:
            raise ValueError(f"demo fairness_feats has size {ff.size}, expected K={K}.")
        Xd_list.append(ff)
        yd_list.append(float(zl))

    Xd = np.stack(Xd_list, axis=0) if len(Xd_list) else np.zeros((0, K), dtype=float)
    y_d = np.asarray(yd_list, dtype=float) if len(yd_list) else np.zeros((0,), dtype=float)

    # --- alpha parsing ---
    alpha_x = None
    alpha_y = None  # for zero_one_loss axis

    if alpha is not None:
        a = alpha
        if torch.is_tensor(a):
            a = a.detach().cpu().numpy()
        a = np.asarray(a, dtype=float)

        if a.ndim == 0:  # scalar
            alpha_x = np.full(K, float(a))
        elif a.ndim == 1:
            if a.size == K:
                alpha_x = a
            elif a.size == K + 1:
                alpha_x = a[:K]
                alpha_y = float(a[-1])
            else:
                raise ValueError(f"alpha 1D must be size K={K} or K+1={K+1}. Got {a.size}.")
        elif a.ndim == 2:
            if a.shape != (R, K):
                raise ValueError(f"alpha 2D must be shape (R,K)={(R,K)}. Got {a.shape}.")
            alpha_x = np.nanmean(a, axis=0)  # reduce to per-feature for plotting
        else:
            raise ValueError("alpha must be scalar, (K,), (K+1,), or (R,K).")

    # --- layout ---
    ncols = int(math.ceil(math.sqrt(K)))
    nrows = int(math.ceil(K / ncols))
    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(figsize_per_ax[0] * ncols, figsize_per_ax[1] * nrows),
        squeeze=False
    )
    axes_flat = axes.ravel()

    artists = [] if return_artists else None

    for k in range(K):
        ax = axes_flat[k]

        # scatter: rollouts + demos
        sc_r = ax.scatter(
            Xr[:, k], y_r,
            s=s_rollouts,
            marker=marker_rollouts,
            alpha=0.35,
            label=rollout_label,
            zorder=1,
        )
        sc_d = ax.scatter(
            Xd[:, k], y_d,
            s=s_demos,
            marker=marker_demos,
            alpha=1.0,
            label=demo_label,
            zorder=2,
        )

        # ensure limits include scatters
        ax.autoscale(enable=True, axis="both", tight=False)

        # rollout mean point
        mean_x = float(np.nanmean(Xr[:, k]))
        mean_y = float(np.nanmean(y_r))

        # inverse-alpha annotations (full if alpha_y provided, else x-only)
        if alpha_x is not None:
            inv_x = 1.0 / max(1e-12, float(alpha_x[k]))

            if alpha_y is not None:
                inv_y = 1.0 / max(1e-12, float(alpha_y))
                annotate_inverse_alpha_arrows(
                    ax,
                    mean_x=mean_x,
                    mean_y=mean_y,
                    inv_alpha_x=inv_x,
                    inv_alpha_y=inv_y,
                    x_name=feature_names[k],
                    y_name="zero_one",
                )
            else:
                # x-only version: vertical ray at inv_x and offset <-> arrow from mean_x to inv_x
                # (keeps “same style” but avoids inventing alpha for loss)
                xmin, xmax = ax.get_xlim()
                ymin, ymax = ax.get_ylim()
                ax.set_xlim(min(xmin, inv_x), max(xmax, inv_x))
                xmin, xmax = ax.get_xlim()
                x_span = max(1e-12, xmax - xmin)
                y_span = max(1e-12, ymax - ymin)
                dy = 0.03 * y_span

                # ray to the right at y=mean_y (visual "margin")
                ax.plot([inv_x, xmax], [mean_y, mean_y], lw=1.4, zorder=6)

                # arrow (offset from mean guides)
                ax.annotate(
                    "",
                    xy=(inv_x, mean_y + dy),
                    xytext=(mean_x, mean_y + dy),
                    arrowprops=dict(arrowstyle="<->", linestyle=":", lw=1.4),
                    zorder=7,
                    clip_on=False,
                )
                ax.annotate(
                    rf"$1/\alpha_{{{feature_names[k]}}}$",
                    xy=((mean_x + inv_x) / 2, mean_y + dy),
                    xytext=(6, 6),
                    textcoords="offset points",
                    ha="left",
                    va="bottom",
                    fontsize=9,
                    zorder=8,
                    clip_on=False,
                )

        # mean dot + solid guides to right/up (uses final limits)
        annotate_mean_with_guides(
            ax,
            mean_x,
            mean_y,
            s=70,
            marker="o",
            lw=1.6,
            zorder=9,
        )

        ax.set_xlabel(feature_names[k])
        ax.set_ylabel("zero_one_loss")

        if return_artists:
            artists.append({"k": k, "rollouts": sc_r, "demos": sc_d})

    # turn off unused axes
    for ax in axes_flat[K:]:
        ax.axis("off")

    # legend + title
    for ax in axes_flat:
        if ax.has_data():
            ax.legend()
            break
    if title is None:
        title = f"Zero-one loss vs fairness features (batch mean={getattr(rb, 'batch_zero_one_loss', float(np.nanmean(y_r))):.4f})"
    fig.suptitle(title)
    fig.tight_layout()

    return (fig, axes, artists) if return_artists else (fig, axes)
