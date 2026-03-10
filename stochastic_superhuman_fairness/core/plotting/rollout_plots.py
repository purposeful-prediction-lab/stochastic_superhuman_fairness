import numpy as np
import matplotlib.pyplot as plt
import math
import torch

from stochastic_superhuman_fairness.core.plotting.plotting_palettes import (
        MODE_PALETTE_100_PAPERSAFE, BASELINE_PALETTE, MODE_PALETTE_100_MAXVAR,
        )
from stochastic_superhuman_fairness.core.plotting.plot_utils import (
    compute_feature_means,
    annotate_mean_with_guides,
    inv_alpha_for_dims,
    annotate_inverse_alpha_arrows,
    cycle_palette_colors,
)
def _to_np(x):
    if torch.is_tensor(x):
        return x.detach().cpu().numpy()
    return np.asarray(x)

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

    #  import ipdb;ipdb.set_trace()
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
    feature_names=None,             # length K-1
    baselines=None,                 # {name: array(K,)} optional, last entry is zero_one
    title="Zero-one vs Features",
    start_offset=None,              # e.g. -0.05
    mode_colors: list = None,
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

    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(figsize_per_ax[0] * ncols, figsize_per_ax[1] * nrows),
        squeeze=False,
    )
    axes = axes.ravel()

    # paper-friendly, avoid red/blue/orange (use your palette; red reserved for rollout mean)
    mode_colors = cycle_palette_colors(M, mode_palette_MAXVAR) if mode_colors is None else mode_colors

    for k in range(K_feat):
        ax = axes[k]

        # rollouts per mode (same color as mode mean)
        for m, Xm in enumerate(modes):
            ax.scatter(
                Xm[:, k], Xm[:, y_idx],
                s=s_rollouts,
                alpha=alpha_rollouts,
                marker="o",
                color=mode_colors[m % len(mode_colors)],
                label="rollouts" if m == 0 else None,
                zorder=1,
            )
            # mode mean (same color)
            mx = float(Xm[:, k].mean())
            my = float(Xm[:, y_idx].mean())
            ax.scatter(
                mx, my,
                s=s_means/1.5,
                marker="D",
                alpha=1.0,
                color=mode_colors[m % len(mode_colors)],
                #  label=f"mode_{m}_mean" if len(modes) > 1 else None,
                zorder=4,
            )

        # demos
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
        ax.relim()
        ax.autoscale()
        set_axis_with_offset(ax, start_offset)
        draw_origin_axes(ax, origin=(0.0, 0.0))

    for ax in axes[K_feat:]:
        ax.axis("off")

    # one legend (first axis with data)
    for ax in axes:
        if ax.has_data():
            ax.legend()
            break

    fig.suptitle(title)
    fig.tight_layout()
    return fig, axes
# -----------------------------------------------------------------------------------------------

def plot_zero_one_vs_features_subplots(
    rb,
    demos,
    *,
    feature_names=None,
    alpha=None,
    baseline_fairness=None,
    title=None,
    rollout_label="rollouts",
    demo_label="demos",
    s_rollouts=12,
    s_demos=28,
    s_baselines=90,
    marker_rollouts="o",
    marker_demos="x",
    marker_baselines="X",
    baseline_alpha=1.0,
    figsize_per_ax=(4.0, 3.5),
    return_artists: bool = False,
):
    def _to_np(x):
        if torch.is_tensor(x):
            return x.detach().cpu().numpy()
        return np.asarray(x)

    # ----- rollouts -----
    Rall = _to_np(rb.feats)
    R, K = Rall.shape
    K_feat = K - 1

    Xr = Rall[:, :K_feat]
    yr = Rall[:, K_feat]
    # Compute rollout mean once per call
    #  rollout_mean = compute_feature_means(rollouts)  # [K]
    #  demo_mean    = compute_feature_means(demos)     # [K]   # NEW
    # ----- alphas ------
    alpha_x = None   # (K_feat,)
    alpha_y = None   # scalar for zero-one

    if alpha is not None:
        a = alpha.detach().cpu().numpy() if torch.is_tensor(alpha) else np.asarray(alpha, dtype=float)

        if a.ndim == 0:  # scalar -> repeat for all features + zero-one
            a = np.full((K_feat + 1,), float(a), dtype=float)
        else:
            a = a.reshape(-1)

    alpha_x = a[:K_feat]
    alpha_y = float(a[K_feat])
    # ----- demos -----
    Xd_list, yd_list = [], []
    for d in demos:
        ff = _to_np(d["fairness_feats"]).reshape(-1)
        Xd_list.append(ff[:K_feat])
        yd_list.append(float(ff[K_feat]))

    Xd = np.stack(Xd_list) if len(Xd_list) else np.zeros((0, K_feat))
    yd = np.asarray(yd_list) if len(yd_list) else np.zeros((0,))

    # ----- baselines -----
    baselines = []
    if baseline_fairness is not None:
        for name, v in baseline_fairness.items():
            b = _to_np(v).reshape(-1)
            baselines.append((name, b[:K_feat], float(b[K_feat])))

    # ----- feature names -----
    if feature_names is None:
        feature_names = [f"f{i}" for i in range(K_feat)]

    roll_alpha = 0.35 if alpha is None else float(np.asarray(alpha).reshape(()))

    # ----- layout -----
    ncols = int(math.ceil(math.sqrt(K_feat)))
    nrows = int(math.ceil(K_feat / ncols))
    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(figsize_per_ax[0] * ncols, figsize_per_ax[1] * nrows),
        squeeze=False,
    )
    axes_flat = axes.ravel()

    artists = [] if return_artists else None

    for k in range(K_feat):
        ax = axes_flat[k]

        sc_d = ax.scatter(Xd[:, k], yd,
                          s=s_demos, marker=marker_demos,
                          alpha=1.0, label=demo_label, zorder=2)
        sc_r = ax.scatter(Xr[:, k], yr,
                          s=s_rollouts, marker=marker_rollouts,
                          alpha=roll_alpha, label=rollout_label, zorder=1)

        # baselines
        sc_b_list = []
        
        c = 0
        for name, bf, bz in baselines:
            marker_char = f"${name[0].upper()}$"

            sc_b = ax.scatter(
                bf[k],
                bz,
                s=s_baselines,
                marker=marker_char,
                alpha=baseline_alpha,
                label=name,
                zorder=4,
                color = BASELINE_PALETTE[c]
            )
            c += 1
            sc_b_list.append(sc_b)
        # ----- means -----
        mx_r = float(np.mean(Xr[:, k]))
        my_r = float(np.mean(yr))

        if len(Xd) > 0:
            mx_d = float(np.mean(Xd[:, k]))
            my_d = float(np.mean(yd))
        else:
            mx_d, my_d = None, None

        # rollout mean (RED)
        mean_r = ax.scatter(mx_r, my_r,
                            s=90, color="red", marker="o",
                            label="mean_rollouts", zorder=5)

        # demo mean (CYAN)
        if mx_d is not None:
            mean_d = ax.scatter(mx_d, my_d,
                                s=90, color="cyan", marker="o",
                                label="mean_demos", zorder=5)
        else:
            mean_d = None

        # ----- guide lines from rollout mean -----
        ax.relim()
        ax.autoscale()

        ax.set_xlim(left=-0.05)
        ax.set_ylim(bottom=-0.05)

        xmin, xmax = ax.get_xlim()
        ymin, ymax = ax.get_ylim()

        ax.plot([mx_r, xmax], [my_r, my_r], lw=1.5, zorder=3, color = 'red')
        ax.plot([mx_r, mx_r], [my_r, ymax], lw=1.5, zorder=3, color = 'red')
        # Origin solid black lines
        ax.hlines(0, 0, xmax, color = 'black')
        ax.vlines(0, 0, ymax, color = 'black')

        # --- alpha guide lines: x = 1/alpha_k, y = 1/alpha_zeroone ---
        if alpha_x is not None:
            x_alpha = 1.0 / max(1e-12, float(alpha_x[k]))
            ax.axvline(x_alpha, linestyle="--", linewidth=1.5, color="red", zorder=3)

            # double dashed arrow from rollout-mean vertical (x=mx_r) to x_alpha line
            xmin, xmax = ax.get_xlim()
            ymin, ymax = ax.get_ylim()
            y_span = max(1e-12, ymax - ymin)
            dy = 0.04 * y_span
            y_arrow = my_r + dy

            ax.annotate(
                "",
                xy=(x_alpha, y_arrow),
                xytext=(mx_r, y_arrow),
                arrowprops=dict(arrowstyle="<->", linestyle="--", color="lightgray", lw=1.4),
                zorder=6,
                clip_on=False,
            )
            ax.annotate(
                f"alpha_{feature_names[k]}",
                xy=((mx_r + x_alpha) / 2, y_arrow),
                xytext=(0, 6),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=9,
                color="gray",
                zorder=7,
                clip_on=False,
            )

            # y = 1/alpha_zeroone (same across subplots)
            y_alpha = 1.0 / max(1e-12, float(alpha_y))
            ax.axhline(y_alpha, linestyle="--", linewidth=1.5, color="red", zorder=3)
        ax.set_xlabel(feature_names[k])
        ax.set_ylabel("zero_one_loss")
        ax.grid(True, alpha=0.2)

        if return_artists:
            artists.append({
                "k": k,
                "rollouts": sc_r,
                "demos": sc_d,
                "baselines": sc_b_list,
                "mean_rollouts": mean_r,
                "mean_demos": mean_d,
            })

    for ax in axes_flat[K_feat:]:
        ax.axis("off")

    for ax in axes_flat:
        if ax.has_data():
            ax.legend()
            break

    if title is None:
        title = "Zero-one loss vs features"

    fig.suptitle(title)
    fig.tight_layout()

    return (fig, axes, artists) if return_artists else (fig, axes)
# -----------------------------------------------------------------------------------------------

