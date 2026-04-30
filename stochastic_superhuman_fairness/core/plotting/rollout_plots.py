import numpy as np
import matplotlib.pyplot as plt
import math
import torch
from typing import Union

from matplotlib import cm, colors as mcolors
from matplotlib.lines import Line2D
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from matplotlib.gridspec import GridSpecFromSubplotSpec

from stochastic_superhuman_fairness.core.plotting.plotting_palettes import (
        MODE_PALETTE_100_PAPERSAFE, BASELINE_PALETTE, MODE_PALETTE_100_MAXVAR,
        )
from stochastic_superhuman_fairness.core.plotting.plot_utils import (
    compute_feature_means,
    annotate_mean_with_guides,
    inv_alpha_for_dims,
    annotate_inverse_alpha_arrows,
    cycle_palette_colors,
    _plot_mixed_text_points,
    _plot_text_points,
    _validate_demo_labels,
    _validate_rollout_mode_labels,
    _apply_limits_from_data,
    _to_np,
)
#  Subgroup block plot helpers
#  Plotting subplot groups in figure subplot axes
def make_inner_axes(parent_ax, n_subplots, figsize_per_ax=(4.0, 3.5)):
    """
    Replace one parent axis with a grid of internal subplots.
    Returns flat list of axes.
    """
    fig = parent_ax.figure
    subspec = parent_ax.get_subplotspec()
    parent_ax.remove()

    ncols = int(math.ceil(math.sqrt(n_subplots)))
    nrows = int(math.ceil(n_subplots / ncols))

    gs = GridSpecFromSubplotSpec(
        nrows, ncols,
        subplot_spec=subspec,
        wspace=0.3,
        hspace=0.3,
    )

    axes = [fig.add_subplot(gs[i, j]) for i in range(nrows) for j in range(ncols)]
    return np.array(axes, dtype=object)

def set_block_title(fig, axes, title):
    xs = [ax.get_position().x0 for ax in axes]
    xe = [ax.get_position().x1 for ax in axes]
    ys = [ax.get_position().y1 for ax in axes]

    x_center = (min(xs) + max(xe)) / 2
    y_top = max(ys)

    fig.text(
        x_center, y_top + 0.02,
        title,
        ha="center",
        va="bottom",
        fontsize=12,
        fontweight="bold",
    )
#--------------------------------------------------------------

def normalize_rollout_modes(rollout_feats):
    """
    Returns a list of mode arrays, each shape (r_m, K).
    Accepts:
      - (R,K)
      - (M,R,K)
      - list of (r_m,K)
    """
    if isinstance(rollout_feats, (list, tuple)):
        modes = []
        for Xm in rollout_feats:
            X = _to_np(Xm)
            if X.ndim != 2:
                raise ValueError("Each mode must be (r_m, K).")
            modes.append(X)
        if len(modes) == 0:
            raise ValueError("Empty rollout mode list.")
        K = modes[0].shape[1]
        if any(m.shape[1] != K for m in modes):
            raise ValueError("All modes must share the same K.")
        return modes

    X = _to_np(rollout_feats)
    if X.ndim == 2:
        return [X]
    if X.ndim == 3:
        return [X[m] for m in range(X.shape[0])]
    raise ValueError("rollout_feats must be (R,K), (M,R,K), or list of (r_m,K).")


def draw_origin_axes(
    ax,
    *,
    origin=(0.0, 0.0),
    lw=1.5,
    linestyle="-",
    color="black",
    zorder=3,
):
    x0, y0 = origin
    _, xmax = ax.get_xlim()
    _, ymax = ax.get_ylim()
    ax.plot([x0, xmax], [y0, y0], linestyle=linestyle, linewidth=lw, color=color, zorder=zorder)
    ax.plot([x0, x0], [y0, ymax], linestyle=linestyle, linewidth=lw, color=color, zorder=zorder)

def _as_demos(demo_feats, K):
    D = _to_np(demo_feats)
    if D.ndim != 2 or D.shape[1] != K:
        raise ValueError(f"demo_feats must be (D,K) with K={K}. Got {D.shape}.")
    return D


def _parse_baselines(baselines, K):
    """
    baselines: None or {name: array(K,)}
    """
    out = []
    if baselines is None:
        return out
    if not isinstance(baselines, dict):
        raise TypeError("baselines must be a dict[name -> array(K,)].")
    for name, v in baselines.items():
        b = _to_np(v).reshape(-1)
        if b.size != K:
            raise ValueError(f"Baseline '{name}' must have length K={K}. Got {b.size}.")
        out.append((name, b))
    return out


def _mean_xy(X, i, j):
    return float(np.mean(X[:, j])), float(np.mean(X[:, i]))


def _scatter(ax, x, y, *, label=None, s=20, marker="o", alpha=0.5, color=None, zorder=1):
    return ax.scatter(x, y, s=s, marker=marker, alpha=alpha, label=label, color=color, zorder=zorder)

def set_axis_with_offset(ax, offset=None):
    """
    If offset is not None (e.g. -0.05),
    forces axes to start at that value.
    """
    if offset is None:
        return

    xmin, xmax = ax.get_xlim()
    ymin, ymax = ax.get_ylim()

    ax.set_xlim(left=offset)
    ax.set_ylim(bottom=offset)

def draw_origin_diagonal(ax, *, lw=1.5, linestyle="-", color="black", zorder=3):
    """
    Draws a diagonal line starting at (0,0) extending to current upper limits.
    """
    xmin, xmax = ax.get_xlim()
    ymin, ymax = ax.get_ylim()

    # extend to max visible range
    upper = min(xmax, ymax)

    ax.plot(
        [0, upper],
        [0, upper],
        linestyle=linestyle,
        linewidth=lw,
        color=color,
        zorder=zorder,
    )

def infer_modes_and_dim(X):
    """
    Infers number of modes M and feature dimension K.

    Accepts:
        - Tensor/ndarray (R, K)
        - Tensor/ndarray (M, R, K)
        - list/tuple of arrays/tensors each (r_m, K)

    Returns:
        M: int
        K: int
    """

    # Case 1: list of modes
    if isinstance(X, (list, tuple)):
        if len(X) == 0:
            raise ValueError("Empty mode list.")

        first = X[0]
        first_np = first.detach().cpu().numpy() if torch.is_tensor(first) else np.asarray(first)

        if first_np.ndim != 2:
            raise ValueError("Each mode must be 2D (r_m, K).")

        K = first_np.shape[1]

        # sanity check consistency
        for Xm in X:
            Xm_np = Xm.detach().cpu().numpy() if torch.is_tensor(Xm) else np.asarray(Xm)
            if Xm_np.ndim != 2 or Xm_np.shape[1] != K:
                raise ValueError("All modes must have same feature dimension K.")

        return len(X), K

    # Case 2: tensor/array
    X_np = X.detach().cpu().numpy() if torch.is_tensor(X) else np.asarray(X)

    if X_np.ndim == 2:
        _, K = X_np.shape
        return 1, K

    if X_np.ndim == 3:
        M, _, K = X_np.shape
        return M, K

    raise ValueError("Input must be (R,K), (M,R,K), or list of (r_m,K).")

def aggregate_feature_mean(X):
    """
    Computes aggregate mean over ALL rollouts.

    Accepts:
        - Tensor/ndarray (R, K)
        - Tensor/ndarray (M, R, K)
        - list/tuple of (r_m, K)

    Returns:
        Tensor (K,) if input is torch
        ndarray (K,) if input is numpy/list
    """

    # ---- list of modes ----
    if isinstance(X, (list, tuple)):
        if len(X) == 0:
            raise ValueError("Empty rollout list.")

        is_torch = torch.is_tensor(X[0])

        if is_torch:
            X_all = torch.cat(X, dim=0)   # concat along rollout axis
            return X_all.mean(dim=0)
        else:
            X_all = np.concatenate([np.asarray(x) for x in X], axis=0)
            return X_all.mean(axis=0)

    # ---- tensor ----
    if torch.is_tensor(X):
        if X.ndim == 2:      # (R,K)
            return X.mean(dim=0)
        if X.ndim == 3:      # (M,R,K)
            return X.reshape(-1, X.shape[-1]).mean(dim=0)
        raise ValueError("Tensor must be (R,K) or (M,R,K).")

    # ---- numpy ----
    X_np = np.asarray(X)
    if X_np.ndim == 2:       # (R,K)
        return X_np.mean(axis=0)
    if X_np.ndim == 3:       # (M,R,K)
        return X_np.reshape(-1, X_np.shape[-1]).mean(axis=0)

    raise ValueError("Input must be (R,K), (M,R,K), or list of (r_m,K).")

def aggregate_mean_xy_over_modes(modes, x_idx, y_idx):
    '''Efficient aggregate means if internal dims are arras or tensors'''
    xs = np.concatenate([m[:, x_idx] for m in modes], axis=0)
    ys = np.concatenate([m[:, y_idx] for m in modes], axis=0)
    return float(xs.mean()), float(ys.mean())


# =====================================================================================================================
# Plotting Functions
# =====================================================================================================================
def plot_rollouts_vs_demos_pairs(
    rollout_feats,          # (R,K) or (M,R,K)
    demo_feats,             # (D,K)
    *,
    feature_names=None,     # list[str] length K
    pairs=None,             # list[(i,j)] where subplot is y=feat[i], x=feat[j]
    baselines=None,         # {name: array(K,)} optional baseline points
    title="Rollouts vs Demos",
    mode_colors : list = None, 
    s_rollouts=10,
    s_demos=18,
    s_means=40,
    s_baselines=90,
    alpha_rollouts=0.25,
    alpha_demos=0.25,
    alpha_baselines=1.0,
    figsize_per_ax=(4.0, 3.6),
):
    """
    Subplots of feature i vs feature j.
    Plots: rollouts, demos, mean rollouts, mean demos, baselines, and mode-means (if rollout_feats is (M,R,K)).
    """

    Xm = normalize_rollout_modes(rollout_feats)  # (M,R,K)
    all_means = aggregate_feature_mean(rollout_feats)

    M, K = infer_modes_and_dim(Xm)
    D = _as_demos(demo_feats, K)
    baselines_list = _parse_baselines(baselines, K)

    if feature_names is None:
        feature_names = [f"f{k}" for k in range(K)]
    if len(feature_names) != K:
        raise ValueError(f"feature_names must have length K={K}.")

    if pairs is None:
        # default: plot all pairs with y=feat[i] vs x=feat[j] for i>j
        pairs = [(i, j) for i in range(1, K) for j in range(0, i)]

    n = len(pairs)
    ncols = int(math.ceil(math.sqrt(n)))
    nrows = int(math.ceil(n / ncols))

    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(figsize_per_ax[0] * ncols, figsize_per_ax[1] * nrows),
        squeeze=False,
    )
    axes = axes.ravel()

    # mode colors (distinct, paper-friendly-ish; avoid red/blue/orange)
    mode_colors = cycle_palette_colors(M, mode_palette_MAXVAR) if mode_colors is None else mode_colors
    for ax, (i, j) in zip(axes, pairs):
        # rollouts cloud
        # --- rollouts by mode ---
        for m in range(M):
            X_mode = Xm[m]   # (R,K)

            _scatter(
                ax,
                X_mode[:, j],
                X_mode[:, i],
                label="rollouts" if m == 0 else None,   # legend once
                s=s_rollouts,
                marker="o",
                alpha=alpha_rollouts,
                color=mode_colors[m % len(mode_colors)],
                zorder=1,
            )
        
        # demos cloud
        _scatter(
            ax,
            D[:, j], D[:, i],
            label="demos",
            s=s_demos,
            marker="x",
            alpha=alpha_demos,
            zorder=2,
        )

        # mode means (if M>1)
        if M > 1:
            for m in range(M):
                mx, my = _mean_xy(Xm[m], i, j)
                _scatter(
                    ax, mx, my,
                    #  label=f"mode_{m}_mean",
                    s=s_means/1.5,
                    #  marker="D",
                    alpha=1.0,
                    color=mode_colors[m % len(mode_colors)],
                    zorder=5,
                )

        # overall means
        mx_r, my_r = all_means[j], all_means[i]
        mx_d, my_d = _mean_xy(D, i, j)

        _scatter(ax, mx_r, my_r, label="mean_rollouts", s=s_means, marker="*", color="red", alpha=1.0, zorder=5)
        _scatter(ax, mx_d, my_d, label="mean_demos",    s=s_means, marker="*", color="cyan", alpha=1.0, zorder=5)
        # Draw red vertical lines for aggregate mean of demos
        draw_origin_axes(ax, origin= (mx_r, my_r), color = 'red')

        # baselines
        for name, b in baselines_list:
            _scatter(
                ax,
                b[j], b[i],
                label=name,
                s=s_baselines,
                marker=f"${name[0].upper()}$",
                alpha=alpha_baselines,
                zorder=6,
            )

        ax.set_xlabel(feature_names[j])
        ax.set_ylabel(feature_names[i])
        ax.grid(True, alpha=0.2)
        ax.relim()
        ax.autoscale()

        set_axis_with_offset(ax, offset=-0.05)
        draw_origin_axes(ax)

    # turn off unused axes
    for ax in axes[len(pairs):]:
        ax.axis("off")

    # one legend (first axis)
    for ax in axes:
        if ax.has_data():
            ax.legend()
            break

    fig.suptitle(title)
    fig.tight_layout()
    return fig, axes
# -----------------------------------------------------------------------------------------------
def plot_zero_one_vs_features(
    rollout_feats,                  # (R,K) or (M,R,K) or list of (r_m,K); last dim is zero_one
    demo_feats,                     # (D,K) where last entry is zero_one
    *,
    ax = None,
    feature_names=None,             # length K-1
    baselines=None,                 # {name: array(K,)} optional, last entry is zero_one
    title="Zero-one vs Features",
    start_offset=None,              # e.g. -0.05
    mode_colors: list = None,
    plot_demos_as_text=False,
    plot_rollouts_as_text=False,
    demo_text_labels=None,          # e.g. demo ranks
    demo_colors = None,
    rollout_text_labels=None,       # e.g. [[1,2,3], [4,5,6], ...] or custom strings
    fontsize_demos=8,
    fontsize_rollouts=8,
    s_rollouts=10,
    s_demos=18,
    s_means=40,
    s_baselines=90,
    alpha_rollouts=0.25,
    alpha_demos=0.25,
    alpha_baselines=1.0,
    figsize_per_ax=(4.0, 3.5),
):
    """
    Subplots: x = feature_k, y = zero_one (last entry).
    Plots: rollouts (colored by mode), demos, mean_rollouts (red), mean_demos (cyan), baselines (optional).
    """

    modes = normalize_rollout_modes(rollout_feats)     # list[(r_m,K)]
    M = len(modes)
    K = modes[0].shape[1]
    K_feat = K - 1
    y_idx = K_feat

    D = _to_np(demo_feats)
    if D.ndim != 2 or D.shape[1] != K:
        raise ValueError(f"demo_feats must be (D,K) with K={K}. Got {D.shape}.")

    if feature_names is None:
        feature_names = [f"f{i}" for i in range(K_feat)]

    baselines_list = []
    if baselines is not None:
        if not isinstance(baselines, dict):
            raise TypeError("baselines must be a dict[name -> array(K,)].")
        for name, v in baselines.items():
            b = _to_np(v).reshape(-1)
            if b.size != K:
                raise ValueError(f"Baseline '{name}' must have length K={K}. Got {b.size}.")
            baselines_list.append((name, b))

    n = K_feat
    ncols = int(math.ceil(math.sqrt(n)))
    nrows = int(math.ceil(n / ncols))

    if ax is None:
        fig, axes = plt.subplots(
            nrows, ncols,
            figsize=(figsize_per_ax[0] * ncols, figsize_per_ax[1] * nrows),
            squeeze=False,
        )
        axes = axes.ravel()
    else:
        fig = ax.figure

        # if a single parent axis was passed, subdivide it
        if hasattr(ax, "get_subplotspec"):
            axes = make_inner_axes(ax, K_feat, figsize_per_ax=figsize_per_ax)
        else:
            axes = np.asarray(ax).ravel()    

    # paper-friendly, avoid red/blue/orange (use your palette; red reserved for rollout mean)
    mode_colors = cycle_palette_colors(M, mode_palette_MAXVAR) if mode_colors is None else mode_colors

    # validate labels
    demo_text_labels = _validate_demo_labels(demo_text_labels, len(D))
    rollout_text_labels = _validate_rollout_mode_labels(rollout_text_labels, modes)

    for k in range(K_feat):
        ax = axes[k]

        # ---------------- rollouts ----------------
        for m, Xm in enumerate(modes):
            color_m = mode_colors[m % len(mode_colors)]

            if plot_rollouts_as_text:
                labels_m = rollout_text_labels[m] if rollout_text_labels is not None else range(len(Xm))
                _plot_text_points(
                    ax,
                    Xm[:, k], Xm[:, y_idx],
                    labels_m,
                    color=color_m,
                    alpha=alpha_rollouts,
                    fontsize=fontsize_rollouts,
                    zorder=1,
                )
            else:
                ax.scatter(
                    Xm[:, k], Xm[:, y_idx],
                    s=s_rollouts,
                    alpha=alpha_rollouts,
                    marker="o",
                    color=color_m,
                    label="rollouts" if m == 0 else None,
                    zorder=1,
                )

            # keep mode mean as marker
            mx = float(Xm[:, k].mean())
            my = float(Xm[:, y_idx].mean())
            ax.scatter(
                mx, my,
                s=s_means / 1.5,
                marker="D",
                alpha=1.0,
                color=color_m,
                zorder=4,
            )

        # ---------------- demos ----------------
        if plot_demos_as_text:
            labels_d = demo_text_labels if demo_text_labels is not None else range(len(D))
            #  labels_d = demo_text_labels if demo_text_labels is not None else [None] * len(D)
            _plot_mixed_text_points(
                ax,
                D[:, k], D[:, y_idx],
                labels_d,
                colors = demo_colors,
                alpha_text=alpha_demos,
                alpha_marker=alpha_demos,
                fontsize=fontsize_demos,
                marker="x",
                s=s_demos,
                zorder_text=3,
                zorder_marker=2,
            )
            #  _plot_text_points(
            #      ax,
            #      D[:, k], D[:, y_idx],
            #      labels_d,
            #      color="black",
            #      alpha=alpha_demos,
            #      fontsize=fontsize_demos,
            #      zorder=2,
            #  )
        else:
            ax.scatter(
                D[:, k], D[:, y_idx],
                s=s_demos,
                alpha=alpha_demos,
                marker="x",
                label="demos",
                zorder=2,
            )

        # overall means
        mx_r, my_r = aggregate_mean_xy_over_modes(modes, k, y_idx)
        mx_d, my_d = float(D[:, k].mean()), float(D[:, y_idx].mean())
        ax.scatter(mx_r, my_r, s=s_means, marker="*", color="red",  label="mean_rollouts", zorder=6)
        ax.scatter(mx_d, my_d, s=s_means, marker="*", color="cyan", label="mean_demos",    zorder=6)

        # baselines
        for name, b in baselines_list:
            ax.scatter(
                b[k], b[y_idx],
                s=s_baselines,
                alpha=alpha_baselines,
                marker=f"${name[0].upper()}$",
                label=name,
                zorder=7,
            )

        ax.set_xlabel(feature_names[k])
        ax.set_ylabel("zero_one_loss")
        ax.grid(True, alpha=0.2)

        # limits + optional offset, then origin axes
        all_x = np.concatenate([D[:, k]] + [Xm[:, k] for Xm in modes])
        all_y = np.concatenate([D[:, y_idx]] + [Xm[:, y_idx] for Xm in modes])

        pad_x = 0.05 * (all_x.max() - all_x.min() + 1e-12)
        pad_y = 0.05 * (all_y.max() - all_y.min() + 1e-12)

        ax.set_xlim(all_x.min() - pad_x, all_x.max() + pad_x)
        ax.set_ylim(-0.05, all_y.max() + pad_y)
        #  ax.relim()
        #  ax.autoscale_view()
        set_axis_with_offset(ax, start_offset)
        draw_origin_axes(ax, origin=(0.0, 0.0))

    for ax in axes[K_feat:]:
        ax.axis("off")

    # one legend (first axis with data)
    for ax in axes:
        if ax.has_data():
            ax.legend()
            break
    if axes is not None and len(axes) > 0:
            axes[0].set_title(title, loc="center", fontsize=12, fontweight="bold")
    #  if ax is not None:
    #      set_block_title(fig, axes, title)
    #  fig.suptitle(title)
    fig.tight_layout()
    return fig, axes
# -----------------------------------------------------------------------------------------------
def _annotate_sidebar_direction(ax, *, best_to_worst=True, x=1.02, top_label = 'best', bot_label = 'worst'):
    """
    Annotate meaning of top->bottom next to the subplot.
    x is in ax.transAxes coordinates, so >1 puts it to the right of the axis.
    """

    # text
    ax.text(
        x, 1.02, top_label,
        transform=ax.transAxes,
        ha="left", va="bottom",
        fontsize=5.5,
    )
    ax.text(
        x, -0.02, bot_label,
        transform=ax.transAxes,
        ha="left", va="top",
        fontsize=5.5,
    )

    # arrow from top to bottom
    ax.annotate(
        "",
        xy=(x + 0.04, 0.01),
        xytext=(x + 0.04, 0.98),
        xycoords=ax.transAxes,
        textcoords=ax.transAxes,
        arrowprops=dict(arrowstyle="->", lw=0.8),
        annotation_clip=False,
    )
def _resolve_optional_labels(labels, n, name="labels"):
    if labels is None:
        return [None] * n
    if len(labels) != n:
        raise ValueError(f"{name} must have length {n}, got {len(labels)}.")
    return list(labels)


def _normalize_gamma_to_mode_list(gamma, modes, D):
    """
    Returns gamma_by_mode = list of arrays, each (r_m, D).
    Accepts:
      - gamma shape (sum_r_m, D)
      - list/tuple of arrays each (r_m, D)
    """
    if isinstance(gamma, (list, tuple)):
        gamma_by_mode = [np.asarray(gm) for gm in gamma]
        if len(gamma_by_mode) != len(modes):
            raise ValueError(
                f"gamma list must have one matrix per mode: expected {len(modes)}, got {len(gamma_by_mode)}."
            )
        for m, (gm, Xm) in enumerate(zip(gamma_by_mode, modes)):
            if gm.shape != (len(Xm), D):
                raise ValueError(
                    f"gamma[{m}] must have shape {(len(Xm), D)}, got {gm.shape}."
                )
        return gamma_by_mode

    G = np.asarray(gamma)
    total_r = sum(len(Xm) for Xm in modes)
    if G.shape != (total_r, D):
        raise ValueError(f"gamma must have shape {(total_r, D)}, got {G.shape}.")

    gamma_by_mode = []
    start = 0
    for Xm in modes:
        r = len(Xm)
        gamma_by_mode.append(G[start:start + r])
        start += r
    return gamma_by_mode

def _add_demo_coupling_sidebar(ax, coupl, cmap_obj, norm):
        import numpy as np
        from mpl_toolkits.axes_grid1.inset_locator import inset_axes

        cax = inset_axes(
            ax,
            width="3%",
            height="100%",
            loc="lower left",
            bbox_to_anchor=(1.01, 0.0, 1, 1),
            bbox_transform=ax.transAxes,
            borderpad=0,
        )

        img = np.asarray(coupl).reshape(-1, 1)
        cax.imshow(img, aspect="auto", cmap=cmap_obj, norm=norm, origin="upper")

        # --- ticks: min / mid / max ---
        n = len(coupl)
        idx_min = int(np.argmin(coupl))
        idx_max = int(np.argmax(coupl))
        idx_mid = n // 2

        val_min = float(coupl[idx_min])
        val_max = float(coupl[idx_max])
        val_mid = float(np.median(coupl))  # better than center index

        cax.set_yticks([idx_max, idx_mid, idx_min])
        cax.set_yticklabels([
            f"{val_max:.2g}",
            f"{val_mid:.2g}",
            f"{val_min:.2g}",
        ], fontsize=7)

        cax.set_xticks([])

        # subtle styling
        for spine in cax.spines.values():
            spine.set_visible(True)
            spine.set_linewidth(0.5)

        return cax

def plot_zero_one_vs_features_mode_coupling(
    rollout_feats,
    demo_feats,
    gamma,
    *,
    ax=None,                           # None, single axis if K_feat==1, or list/array of K_feat axes
    feature_names=None,
    title="Zero-one vs Features by Mode Coupling",
    start_offset=None,
    mode_colors=None,
    cmap="viridis",
    coupling_norm="global",
    rollout_text_labels=None,          # list per mode: [[..., None, ...], ...]
    demo_text_labels=None,             # length D, entries can be None
    annotate_sidebar=True,             # True or [top_label, bot_label]
    fontsize_rollouts=8,
    fontsize_demos=8,
    s_rollouts=16,
    s_demos=22,
    s_means=40,
    alpha_rollouts=0.35,
    alpha_demos=0.95,
    figsize_per_ax=(4.1, 3.6),
    default_demo_color="#1f77b4",
    default_rollout_marker="o",
    default_demo_marker="x",
):
    """
    Correct behavior:
      - zero_one is always feats[:, -1]
      - one subplot per feature_k for k in [0, ..., K-2]
      - each mode gets its own group of K-1 plots

    If ax is None:
      creates an (M x K_feat) grid

    If ax is provided:
      - if M == 1:
          ax may be:
            * a single matplotlib axis if K_feat == 1
            * a list/array of K_feat axes
      - otherwise:
          ax should be a 2D array-like of shape (M, K_feat)
    """

    modes = normalize_rollout_modes(rollout_feats)
    M = len(modes)
    K = modes[0].shape[1]
    K_feat = K - 1
    y_idx = K - 1

    Df = _to_np(demo_feats)
    if Df.ndim != 2 or Df.shape[1] != K:
        raise ValueError(f"demo_feats must have shape (D, {K}), got {Df.shape}")
    D = len(Df)

    if feature_names is None:
        feature_names = [f"f{i}" for i in range(K_feat)]

    if mode_colors is None:
        mode_colors = [f"C{i}" for i in range(M)]
    mode_colors = cycle_palette_colors(M, mode_palette_MAXVAR) if mode_colors is None else mode_colors

    rollout_text_labels = (
        [[None] * len(Xm) for Xm in modes]
        if rollout_text_labels is None else rollout_text_labels
    )
    if len(rollout_text_labels) != M:
        raise ValueError(f"rollout_text_labels must have {M} lists")
    for m, (labs, Xm) in enumerate(zip(rollout_text_labels, modes)):
        if len(labs) != len(Xm):
            raise ValueError(
                f"rollout_text_labels[{m}] must have length {len(Xm)}, got {len(labs)}"
            )

    demo_text_labels = _resolve_optional_labels(demo_text_labels, D, "demo_text_labels")
    gamma_by_mode = _normalize_gamma_to_mode_list(gamma, modes, D)
    demo_coupling_means = np.stack([gm.mean(axis=0) for gm in gamma_by_mode], axis=0)

    if coupling_norm not in {"per_mode", "global"}:
        raise ValueError("coupling_norm must be 'per_mode' or 'global'")

    cmap_obj = cm.get_cmap(cmap)

    if coupling_norm == "global":
        all_gamma_vals = np.concatenate([gm.ravel() for gm in gamma_by_mode])
        gvmin = float(all_gamma_vals.min())
        gvmax = float(all_gamma_vals.max())
        if gvmax <= gvmin:
            gvmax = gvmin + 1e-12
        global_norm = mcolors.Normalize(vmin=gvmin, vmax=gvmax)

    # -------- layout / axes normalization --------
    created_fig = False

    if ax is None:
        fig, axes = plt.subplots(
            M, K_feat,
            figsize=(figsize_per_ax[0] * K_feat * 1.12, figsize_per_ax[1] * M),
            squeeze=False,
        )
        created_fig = True
    else:
        fig = plt.gcf()

        if M == 1:
            if K_feat == 1:
                if isinstance(ax, np.ndarray):
                    axes = np.asarray(ax, dtype=object).reshape(1, 1)
                elif isinstance(ax, (list, tuple)):
                    if len(ax) != 1:
                        raise ValueError("For M=1, K_feat=1, ax must be a single axis or length-1 list.")
                    axes = np.asarray(ax, dtype=object).reshape(1, 1)
                else:
                    axes = np.asarray([[ax]], dtype=object)
            else:
                if not isinstance(ax, (list, tuple, np.ndarray)):
                    raise ValueError(f"For M=1 and K_feat={K_feat}, ax must be a list/array of {K_feat} axes.")
                axes = np.asarray(ax, dtype=object).reshape(1, K_feat)
        else:
            if not isinstance(ax, (list, tuple, np.ndarray)):
                raise ValueError(f"For M={M}, ax must be a 2D array-like of shape ({M}, {K_feat}).")
            axes = np.asarray(ax, dtype=object).reshape(M, K_feat)

        if axes.shape != (M, K_feat):
            raise ValueError(f"ax must have shape ({M}, {K_feat}), got {axes.shape}")

    # legend handles
    demo_mean_handle = Line2D(
        [0], [0],
        marker="*",
        linestyle="None",
        markersize=max(4, np.sqrt(s_means)),
        markerfacecolor="cyan",
        markeredgecolor="cyan",
        label="Demo mean",
    )

    out_axes = []

    # -------- plotting --------
    for m, Xm in enumerate(modes):
        coupl = demo_coupling_means[m]

        if coupling_norm == "global":
            norm = global_norm
        else:
            vmin = float(coupl.min())
            vmax = float(coupl.max())
            if vmax <= vmin:
                vmax = vmin + 1e-12
            norm = mcolors.Normalize(vmin=vmin, vmax=vmax)

        demo_colors = [cmap_obj(norm(v)) for v in coupl]
        rcolor = mode_colors[m]

        rollout_mean_handle = Line2D(
            [0], [0],
            marker="D",
            linestyle="None",
            markersize=max(4, np.sqrt(s_means)),
            markerfacecolor=rcolor,
            markeredgecolor=rcolor,
            label="Mode mean",
        )

        for k in range(K_feat):
            axk = axes[m, k]

            # rollouts: x = feature_k, y = zero_one
            _plot_mixed_text_points(
                axk,
                Xm[:, k], Xm[:, y_idx],
                rollout_text_labels[m],
                colors=[rcolor] * len(Xm),
                default_color=rcolor,
                alpha_text=alpha_rollouts,
                alpha_marker=alpha_rollouts,
                fontsize=fontsize_rollouts,
                marker=default_rollout_marker,
                s=s_rollouts,
                zorder_text=5,
                zorder_marker=2,
            )

            # demos: x = feature_k, y = zero_one
            _plot_mixed_text_points(
                axk,
                Df[:, k], Df[:, y_idx],
                demo_text_labels,
                colors=demo_colors,
                default_color=default_demo_color,
                alpha_text=alpha_demos,
                alpha_marker=alpha_demos,
                fontsize=fontsize_demos,
                marker=default_demo_marker,
                s=s_demos,
                zorder_text=6,
                zorder_marker=3,
            )

            # means for this feature_k vs zero_one
            rx = float(Xm[:, k].mean())
            ry = float(Xm[:, y_idx].mean())
            dx = float(Df[:, k].mean())
            dy = float(Df[:, y_idx].mean())

            axk.scatter(rx, ry, s=s_means, marker="D", color=rcolor, zorder=8)
            axk.scatter(dx, dy, s=s_means, marker="*", color="cyan", zorder=8)

            # titles
            #  axk.set_title(
            #      feature_names[k],
            #      loc="center",
            #      color=rcolor,
            #      fontweight="bold",
            #      fontsize=11,
            #  )
            #  axk.text(
            #      0.5, 0.992,
            #      f"\nmean γ ∈ [{coupl.min():.3g}, {coupl.max():.3g}]",
            #      transform=axk.transAxes,
            #      ha="center",
            #      va="bottom",
            #      fontsize=6,
            #      color=rcolor,
            #  )

            axk.set_xlabel(feature_names[k])
            if k == 0:
                axk.set_ylabel("zero_one")

            _apply_limits_from_data(
                axk,
                x_arrays=[Xm[:, k], Df[:, k]],
                y_arrays=[Xm[:, y_idx], Df[:, y_idx]],
                pad_frac=0.05,
                start_offset=start_offset,
            )

            axk.grid(True, alpha=0.2)
            axk.axhline(0.0, color="0.7", lw=0.8, zorder=0)
            axk.axvline(0.0, color="0.7", lw=0.8, zorder=0)

            _add_demo_coupling_sidebar(axk, coupl, cmap_obj, norm)

            if annotate_sidebar is True or isinstance(annotate_sidebar, list):
                if annotate_sidebar is True:
                    top_label, bot_label = "best", "worst"
                else:
                    top_label, bot_label = annotate_sidebar[0], annotate_sidebar[1]
                _annotate_sidebar_direction(axk, top_label=top_label, bot_label=bot_label)

            axk.legend(
                handles=[rollout_mean_handle, demo_mean_handle],
                loc="best",
                fontsize=8,
                frameon=True,
            )

            out_axes.append(axk)

    # optional row labels if we created the figure
    if created_fig and M > 1:
        for m in range(M):
            left = axes[m, 0].get_position().x0
            right = axes[m, -1].get_position().x1
            top = max(axes[m, k].get_position().y1 for k in range(K_feat))
            fig.text(
                0.5 * (left + right),
                top + 0.01,
                f"Mode {m}",
                ha="center",
                va="bottom",
                fontsize=12,
                fontweight="bold",
                color=mode_colors[m],
            )

    if created_fig and title is not None:
        fig.suptitle(title, y=0.995)
        fig.tight_layout()

    return fig, axes


def plot_zero_one_vs_features_demo_logprobs_from_log(
    rollout_feats,
    demo_feats,
    logs,
    *,
    ax=None,                         # None or array-like of shape (M, K-1)
    feature_names=None,
    logprob_key="demo_logprobs",     # inside log["train/l_terms"]
    title="Zero-one vs Features colored by demo logprob",
    mode_colors=None,
    cmap="viridis",
    logprob_norm="global",         # "per_mode" or "global"
    rollout_text_labels=None,        # list per mode
    demo_text_labels=None,           # len D
    fontsize_rollouts=8,
    fontsize_demos=8,
    s_rollouts=14,
    s_means=60,
    alpha_rollouts=0.35,
    alpha_demos=0.95,
    figsize_per_ax=(4.4, 3.6),
    default_rollout_marker="o",
    default_demo_marker="x",
    size_bin_min=20,
    size_bin_max=120,
    n_size_bins=10,
):
    """
    For each mode m and feature k in [0, ..., K-2], plot:
        x = feature_k
        y = zero_one = feats[:, -1]

    Demos are colored and sized by the demo logprob under that mode.

    Expected log format:
        logs[i]["train/l_terms"][logprob_key] -> shape (M, D)
    We use the LAST available entry.

    Returns
    -------
    fig, axes, demo_logprobs_norm
        axes has shape (M, K-1)
        demo_logprobs_norm has shape (M, D)
    """

    # ---------- extract last demo_logprobs ----------
    last_demo_logprobs = None
    for log in logs:
        ltd = log.get("train/l_terms", {})
        if logprob_key in ltd:
            last_demo_logprobs = ltd[logprob_key]

    if last_demo_logprobs is None:
        raise ValueError(f"Could not find train/l_terms['{logprob_key}'] in logs.")

    demo_logprobs = _to_np(last_demo_logprobs)
    if demo_logprobs.ndim == 1:
        demo_logprobs = demo_logprobs[None, :]
    if demo_logprobs.ndim != 2:
        raise ValueError(f"{logprob_key} must be 2D (M, D), got shape {demo_logprobs.shape}")

    # ---------- normalize inputs ----------
    modes = normalize_rollout_modes(rollout_feats)
    M = len(modes)
    K = modes[0].shape[1]
    K_feat = K - 1
    y_idx = K - 1

    Df = _to_np(demo_feats)
    if Df.ndim != 2 or Df.shape[1] != K:
        raise ValueError(f"demo_feats must have shape (D, {K}), got {Df.shape}")
    D = len(Df)

    if demo_logprobs.shape != (M, D):
        raise ValueError(
            f"{logprob_key} must have shape ({M}, {D}) to match modes x demos, "
            f"got {demo_logprobs.shape}"
        )

    if feature_names is None:
        feature_names = [f"f{i}" for i in range(K_feat)]


    mode_colors = cycle_palette_colors(M, mode_palette_MAXVAR) if mode_colors is None else mode_colors
    if len(mode_colors) <= M:
        raise ValueError(f"mode_colors must have length at least {M}")

    rollout_text_labels = (
        [[None] * len(Xm) for Xm in modes]
        if rollout_text_labels is None else rollout_text_labels
    )
    if len(rollout_text_labels) != M:
        raise ValueError(f"rollout_text_labels must have {M} lists")
    for m, (labs, Xm) in enumerate(zip(rollout_text_labels, modes)):
        if len(labs) != len(Xm):
            raise ValueError(
                f"rollout_text_labels[{m}] must have length {len(Xm)}, got {len(labs)}"
            )

    demo_text_labels = _resolve_optional_labels(demo_text_labels, D, "demo_text_labels")

    # ---------- normalize logprobs to [0, 1] ----------
    if logprob_norm == "global":
        #  lo = float(demo_logprobs.min())
        #  hi = float(demo_logprobs.max())
        #  if hi <= lo:
        #      hi = lo + 1e-12
        #  demo_logprobs_norm = (demo_logprobs - lo) / (hi - lo)
        #  lo, hi = np.percentile(demo_logprobs, [2, 98])
        #  p_clipped = np.clip(demo_logprobs, lo, hi)
        #  demo_logprobs_norm = (p_clipped - lo) / (hi - lo)
        demo_logprobs_norm = demo_logprobs / demo_logprobs.max()
        norm = mcolors.Normalize(vmin=demo_logprobs_norm.min(), vmax=demo_logprobs_norm.max())
    elif logprob_norm == "per_mode":
        lo = demo_logprobs.min(axis=1, keepdims=True)
        hi = demo_logprobs.max(axis=1, keepdims=True)
        hi = np.where(hi <= lo, lo + 1e-12, hi)
        demo_logprobs_norm = (demo_logprobs - lo) / (hi - lo)
        norm = None
    else:
        raise ValueError("logprob_norm must be 'per_mode' or 'global'")

    cmap_obj = cm.get_cmap(cmap)

    # ---------- 10 size bins ----------
    bin_edges = np.linspace(0.0, 1.0, n_size_bins + 1)
    size_levels = np.linspace(size_bin_min, size_bin_max, n_size_bins)

    def _sizes_from_probs(p):
        # p in [0, 1]
        idx = np.digitize(p, bin_edges[1:-1], right=False)   # 0..n_size_bins-1
        return size_levels[idx]

    # ---------- axes ----------
    created_fig = False

    if ax is None:
        fig, axes = plt.subplots(
            M, K_feat,
            figsize=(figsize_per_ax[0] * K_feat, figsize_per_ax[1] * M),
            squeeze=False,
        )
        created_fig = True
    else:
        fig = plt.gcf()
        axes = np.asarray(ax, dtype=object)
        if M == 1 and K_feat == 1 and axes.ndim == 0:
            axes = axes.reshape(1, 1)
        elif M == 1:
            axes = axes.reshape(1, K_feat)
        else:
            axes = axes.reshape(M, K_feat)

        if axes.shape != (M, K_feat):
            raise ValueError(f"ax must have shape ({M}, {K_feat}), got {axes.shape}")

    out_axes = []
    demo_mean_handle = Line2D(
        [0], [0],
        marker="*",
        linestyle="None",
        markersize=max(6, np.sqrt(s_means)),
        markerfacecolor="cyan",
        markeredgecolor="cyan",
        label="Demo mean",
    )
    # SOrt demo by confidence to plot brither colors on top
    #  order = np.argsort(demo_logprobs_norm)   # low -> high
    #
    #  x_ord = [order]
    #  y_ord = y[order]
    #  c_ord = [norm[i] for i in order]
    #  s_ord = demo_sizes[order]
    #  lab_ord = [demo_text_labels[i] for i in order]
    # ---------- plotting ----------

    import ipdb;ipdb.set_trace()
    for m, Xm in enumerate(modes):
        #  import ipdb;ipdb.set_trace()
        rcolor = mode_colors[m]
        p_raw = demo_logprobs[m]          # (D,)
        p_norm = demo_logprobs_norm[m]    # (D,)
        mean_lp = float(p_raw.mean())
        std_lp = float(p_raw.std())

        if logprob_norm == "global":
            #  demo_colors = [cmap_obj(norm(v)) for v in p_raw]
            demo_colors = [cmap_obj(v) for v in p_norm]
        else:
            demo_colors = [cmap_obj(v) for v in p_norm]

        demo_sizes = _sizes_from_probs(p_norm)
        rollout_mean_handle = Line2D(
            [0], [0],
            marker="D",
            linestyle="None",
            markersize=max(6, np.sqrt(s_means)),
            markerfacecolor=rcolor,
            markeredgecolor=rcolor,
            label=f"Mode {m} mean \nDemo norm logprobs μ={p_raw.mean():.2g}, σ={p_raw.std():.2g}",
        )
        for k in range(K_feat):
            axk = axes[m, k]

            # rollouts for this mode
            _plot_mixed_text_points(
                axk,
                Xm[:, k], Xm[:, k],
                rollout_text_labels[m],
                colors=[rcolor] * len(Xm),
                default_color=rcolor,
                alpha_text=alpha_rollouts,
                alpha_marker=alpha_rollouts,
                fontsize=fontsize_rollouts,
                marker=default_rollout_marker,
                s=s_rollouts,
                zorder_text=4,
                zorder_marker=2,
            )
            
            # demos: do size-aware mixed plotting manually
            #  for xd, yd, lab, cd, sd in zip(x_ord, y_ord, lab_ord, c_ord, s_ord):
            for xd, yd, lab, cd, sd in zip(Df[:, k], Df[:, y_idx], demo_text_labels, demo_colors, demo_sizes):
                xd = float(xd)
                yd = float(yd)
                if lab is None:
                    axk.scatter(
                        [xd], [yd],
                        color=cd,
                        alpha=alpha_demos,
                        marker=default_demo_marker,
                        #  s=float(sd),
                        zorder=3,
                    )
                else:
                    axk.annotate(
                        str(lab),
                        (xd, yd),
                        xytext=(2, 2),
                        textcoords="offset points",
                        ha="left",
                        va="bottom",
                        fontsize=fontsize_demos,
                        color=cd,
                        alpha=alpha_demos,
                        clip_on=True,
                        zorder=5,
                    )

            # means
            rx = float(Xm[:, k].mean())
            ry = float(Xm[:, y_idx].mean())
            dx = float(Df[:, k].mean())
            dy = float(Df[:, y_idx].mean())

            axk.scatter(rx, ry, s=s_means, marker="D", color=rcolor, zorder=7)
            axk.scatter(dx, dy, s=s_means, marker="*", color="cyan", zorder=7)

            axk.set_title(
                feature_names[k],
                loc="center",
                color=rcolor,
                fontweight="bold",
                fontsize=11,
            )
            axk.text(
                0.5, 1.00,
                f"logprob ∈ [{p_raw.min():.3g}, {p_raw.max():.3g}]",
                transform=axk.transAxes,
                ha="center",
                va="bottom",
                fontsize=7,
                color=rcolor,
            )

            axk.set_xlabel(feature_names[k])
            if k == 0:
                axk.set_ylabel("zero_one")

            _apply_limits_from_data(
                axk,
                x_arrays=[Xm[:, k], Df[:, k]],
                y_arrays=[Xm[:, y_idx], Df[:, y_idx]],
                pad_frac=0.05,
                start_offset=start_offset if "start_offset" in plot_zero_one_vs_features_demo_logprobs.__code__.co_varnames else None,
            )

            axk.grid(True, alpha=0.2)
            axk.axhline(0.0, color="0.7", lw=0.8, zorder=0)
            axk.axvline(0.0, color="0.7", lw=0.8, zorder=0)

            axk.legend(
                handles=[rollout_mean_handle, demo_mean_handle],
                loc="best",
                fontsize=8,
                frameon=True,
            )

            out_axes.append(axk)

    if created_fig and title is not None:
        fig.suptitle(title, y=0.995)
        fig.tight_layout()
    # ---- size legend (10 bins) ----
    size_levels = np.linspace(size_bin_min, size_bin_max, n_size_bins)
    bin_centers = np.linspace(0.05, 0.95, n_size_bins)

    # pick 3 representative bins to avoid clutter
    idxs = [0, n_size_bins // 2, n_size_bins - 1]

    size_handles = []
    size_labels = []

    for i in idxs:
        size_handles.append(
            plt.scatter([], [], s=size_levels[i], color="gray", alpha=0.6)
        )
        size_labels.append(f"{bin_centers[i]:.1f}")

    # attach to first axis only (cleanest)
    #  axes.flat[0].legend(
    #      size_handles,
    #      size_labels,
    #      title="norm logprob",
    #      loc="upper right",
    #      fontsize=7,
    #      title_fontsize=8,
    #      frameon=True,
    #  )
    # ---- shared colorbar ----
    from matplotlib.cm import ScalarMappable

    if logprob_norm == "global":
        sm = ScalarMappable(norm=norm, cmap=cmap_obj)
    else:
        # normalized [0,1]
        sm = ScalarMappable(norm=mcolors.Normalize(vmin=0.0, vmax=1.0), cmap=cmap_obj)

    sm.set_array([])

    cbar = fig.colorbar(
        sm,
        ax=axes.ravel().tolist(),
        fraction=0.015,
        pad=0.02,
    )

    cbar.set_label("demo logprob", fontsize=9)
    cbar.ax.tick_params(labelsize=8)
    return fig, axes

def plot_zero_one_vs_features_demo_logprobs(
    rollout_feats,
    demo_feats,
    demo_logprobs,
    *,
    ax=None,
    feature_names=None,
    title="Zero-one vs Features colored by demo logprob",
    mode_colors=None,
    cmap="viridis",
    logprob_norm="global",  # "per_mode" or "global"
    rollout_text_labels=None,
    demo_text_labels=None,
    fontsize_rollouts=8,
    fontsize_demos=8,
    s_rollouts=14,
    s_means=60,
    alpha_rollouts=0.35,
    alpha_demos=0.95,
    figsize_per_ax=(4.4, 3.6),
    default_rollout_marker="o",
    default_demo_marker="x",
    size_bin_min=20,
    size_bin_max=120,
    n_size_bins=10,
):
    """
    For each mode m and feature k in [0, ..., K-2], plot:
        x = feature_k
        y = zero_one = feats[:, -1]

    Demos are colored and sized by demo_logprobs under that mode.

    Args:
        rollout_feats: rollout feature arrays, either one array or list per mode
        demo_feats: demo feature array, shape (D, K)
        demo_logprobs: array-like, shape (M, D) or (D,)

    Returns:
        fig, axes, demo_logprobs_norm
    """

    # ---------- normalize inputs ----------
    modes = normalize_rollout_modes(rollout_feats)
    M = len(modes)
    K = modes[0].shape[1]
    K_feat = K - 1
    y_idx = K - 1

    Df = _to_np(demo_feats)
    if Df.ndim != 2 or Df.shape[1] != K:
        raise ValueError(f"demo_feats must have shape (D, {K}), got {Df.shape}")

    D = len(Df)

    demo_logprobs = _to_np(demo_logprobs)
    if demo_logprobs.ndim == 1:
        demo_logprobs = demo_logprobs[None, :]

    if demo_logprobs.ndim != 2:
        raise ValueError(
            f"demo_logprobs must be 2D with shape (M, D), got {demo_logprobs.shape}"
        )

    if demo_logprobs.shape != (M, D):
        raise ValueError(
            f"demo_logprobs must have shape ({M}, {D}) to match modes x demos, "
            f"got {demo_logprobs.shape}"
        )

    if feature_names is None:
        feature_names = [f"f{i}" for i in range(K_feat)]

    mode_colors = (
        cycle_palette_colors(M, mode_palette_MAXVAR)
        if mode_colors is None
        else mode_colors
    )

    if len(mode_colors) < M:
        raise ValueError(f"mode_colors must have length at least {M}")

    rollout_text_labels = (
        [[None] * len(Xm) for Xm in modes]
        if rollout_text_labels is None
        else rollout_text_labels
    )

    if len(rollout_text_labels) != M:
        raise ValueError(f"rollout_text_labels must have {M} lists")

    for m, (labs, Xm) in enumerate(zip(rollout_text_labels, modes)):
        if len(labs) != len(Xm):
            raise ValueError(
                f"rollout_text_labels[{m}] must have length {len(Xm)}, got {len(labs)}"
            )

    demo_text_labels = _resolve_optional_labels(
        demo_text_labels, D, "demo_text_labels"
    )

    # ---------- normalize logprobs to [0, 1] ----------
    if logprob_norm == "global":
        max_lp = demo_logprobs.max()
        if max_lp == 0:
            max_lp = 1e-12

        demo_logprobs_norm = demo_logprobs / max_lp
        norm = mcolors.Normalize(
            vmin=demo_logprobs_norm.min(),
            vmax=demo_logprobs_norm.max(),
        )

    elif logprob_norm == "per_mode":
        lo = demo_logprobs.min(axis=1, keepdims=True)
        hi = demo_logprobs.max(axis=1, keepdims=True)
        hi = np.where(hi <= lo, lo + 1e-12, hi)

        demo_logprobs_norm = (demo_logprobs - lo) / (hi - lo)
        norm = None

    else:
        raise ValueError("logprob_norm must be 'per_mode' or 'global'")

    cmap_obj = cm.get_cmap(cmap)

    # ---------- size bins ----------
    bin_edges = np.linspace(0.0, 1.0, n_size_bins + 1)
    size_levels = np.linspace(size_bin_min, size_bin_max, n_size_bins)

    def _sizes_from_probs(p):
        idx = np.digitize(p, bin_edges[1:-1], right=False)
        return size_levels[idx]

    # ---------- axes ----------
    created_fig = False

    if ax is None:
        fig, axes = plt.subplots(
            M,
            K_feat,
            figsize=(figsize_per_ax[0] * K_feat, figsize_per_ax[1] * M),
            squeeze=False,
        )
        created_fig = True
    else:
        axes = np.asarray(ax, dtype=object)

        if M == 1 and K_feat == 1 and axes.ndim == 0:
            axes = axes.reshape(1, 1)
        elif M == 1:
            axes = axes.reshape(1, K_feat)
        else:
            axes = axes.reshape(M, K_feat)

        if axes.shape != (M, K_feat):
            raise ValueError(f"ax must have shape ({M}, {K_feat}), got {axes.shape}")

        fig = axes.flat[0].figure

    demo_mean_handle = Line2D(
        [0], [0],
        marker="*",
        linestyle="None",
        markersize=max(6, np.sqrt(s_means)),
        markerfacecolor="cyan",
        markeredgecolor="cyan",
        label="Demo mean",
    )

    # ---------- plotting ----------
    for m, Xm in enumerate(modes):
        rcolor = mode_colors[m]
        p_raw = demo_logprobs[m]
        p_norm = demo_logprobs_norm[m]

        if logprob_norm == "global":
            demo_colors = [cmap_obj(v) for v in p_norm]
        else:
            demo_colors = [cmap_obj(v) for v in p_norm]

        demo_sizes = _sizes_from_probs(p_norm)

        rollout_mean_handle = Line2D(
            [0], [0],
            marker="D",
            linestyle="None",
            markersize=max(6, np.sqrt(s_means)),
            markerfacecolor=rcolor,
            markeredgecolor=rcolor,
            label=(
                f"Mode {m} mean\n"
                f"Demo norm logprobs μ={p_raw.mean():.2g}, σ={p_raw.std():.2g}"
            ),
        )

        for k in range(K_feat):
            axk = axes[m, k]

            # rollouts for this mode
            _plot_mixed_text_points(
                axk,
                Xm[:, k],
                Xm[:, y_idx],
                rollout_text_labels[m],
                colors=[rcolor] * len(Xm),
                default_color=rcolor,
                alpha_text=alpha_rollouts,
                alpha_marker=alpha_rollouts,
                fontsize=fontsize_rollouts,
                marker=default_rollout_marker,
                s=s_rollouts,
                zorder_text=4,
                zorder_marker=2,
            )

            # demos
            for xd, yd, lab, cd, sd in zip(
                Df[:, k],
                Df[:, y_idx],
                demo_text_labels,
                demo_colors,
                demo_sizes,
            ):
                xd = float(xd)
                yd = float(yd)

                if lab is None:
                    axk.scatter(
                        [xd], [yd],
                        color=cd,
                        alpha=alpha_demos,
                        marker=default_demo_marker,
                        s=float(sd),
                        zorder=3,
                    )
                else:
                    axk.annotate(
                        str(lab),
                        (xd, yd),
                        xytext=(2, 2),
                        textcoords="offset points",
                        ha="left",
                        va="bottom",
                        fontsize=fontsize_demos,
                        color=cd,
                        alpha=alpha_demos,
                        clip_on=True,
                        zorder=5,
                    )

            # means
            rx = float(Xm[:, k].mean())
            ry = float(Xm[:, y_idx].mean())
            dx = float(Df[:, k].mean())
            dy = float(Df[:, y_idx].mean())

            axk.scatter(rx, ry, s=s_means, marker="D", color=rcolor, zorder=7)
            axk.scatter(dx, dy, s=s_means, marker="*", color="cyan", zorder=7)

            axk.set_title(
                feature_names[k],
                loc="center",
                color=rcolor,
                fontweight="bold",
                fontsize=11,
            )

            axk.text(
                0.5,
                1.00,
                f"logprob ∈ [{p_raw.min():.3g}, {p_raw.max():.3g}]",
                transform=axk.transAxes,
                ha="center",
                va="bottom",
                fontsize=7,
                color=rcolor,
            )

            axk.set_xlabel(feature_names[k])
            if k == 0:
                axk.set_ylabel("zero_one")

            _apply_limits_from_data(
                axk,
                x_arrays=[Xm[:, k], Df[:, k]],
                y_arrays=[Xm[:, y_idx], Df[:, y_idx]],
                pad_frac=0.05,
            )

            axk.grid(True, alpha=0.2)
            axk.axhline(0.0, color="0.7", lw=0.8, zorder=0)
            axk.axvline(0.0, color="0.7", lw=0.8, zorder=0)

            axk.legend(
                handles=[rollout_mean_handle, demo_mean_handle],
                loc="best",
                fontsize=8,
                frameon=True,
            )

    if created_fig and title is not None:
        fig.suptitle(title, y=0.995)
        fig.tight_layout()

    # ---------- shared colorbar ----------
    from matplotlib.cm import ScalarMappable

    if logprob_norm == "global":
        sm = ScalarMappable(norm=norm, cmap=cmap_obj)
    else:
        sm = ScalarMappable(
            norm=mcolors.Normalize(vmin=0.0, vmax=1.0),
            cmap=cmap_obj,
        )

    sm.set_array([])

    cbar = fig.colorbar(
        sm,
        ax=axes.ravel().tolist(),
        fraction=0.015,
        pad=0.02,
    )

    cbar.set_label("demo logprob", fontsize=9)
    cbar.ax.tick_params(labelsize=8)

    return fig, axes
