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
def _to_np(x):
    if torch.is_tensor(x):
        return x.detach().cpu().numpy()
    return np.asarray(x)
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
    baseline_fairness: dict = None,   
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
    # Compute rollout mean once per call
    rollout_mean = compute_feature_means(rollouts)  # [K]
    demo_mean    = compute_feature_means(demos)     # [K]   # NEW
    if feature_names is None:
        feature_names = [f"f{i}" for i in range(K)]
    if len(feature_names) != K:
        raise ValueError("feature_names must have length K.")
    # ----------------------
    baseline_data = []
    if baseline_fairness is not None:
        if not isinstance(baseline_fairness, dict):
            raise TypeError("baseline_fairness must be a dict[label -> array-like]")
        for name, arr in baseline_fairness.items():
            B = np.asarray(arr, dtype=float)
            if B.ndim == 1:
                if B.size != K:
                    raise ValueError(f"Baseline '{name}' must have size K={K}. Got {B.size}.")
                B = B[None, :]
            elif B.ndim == 2:
                if B.shape[1] != K:
                    raise ValueError(f"Baseline '{name}' must have shape (N,K) with K={K}. Got {B.shape}.")
            else:
                raise ValueError(f"Baseline '{name}' must be (K,) or (N,K). Got ndim={B.ndim}.")
            baseline_data.append((name, B))

    # ---------------------

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
         # demos: opaque
        sc_d = ax.scatter(
            demos[:, j], demos[:, i],
            s=s_demos,
            marker=marker_demos,
            label=demo_label,
            alpha=0.45,
            zorder=2,
        )
        # rollouts: semi-transparent
        sc_r = ax.scatter(
            rollouts[:, j], rollouts[:, i],
            s=s_rollouts,
            marker=marker_rollouts,
            label=rollout_label,
            alpha=0.75,
            zorder=1,
            color = 'orange'
        )

       
        baseline_artists = []
        for name, B in baseline_data:
            sc_b = ax.scatter(
                B[:, j], B[:, i],
                s=s_demos,              # or new kwarg if you want
                marker="^",
                alpha=1.0,
                label=name,
                zorder=3,
            )
            baseline_artists.append(sc_b)
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
        # NEW: demo mean point (no guides)
        demo_mean_x = float(demo_mean[j])
        demo_mean_y = float(demo_mean[i])
        sc_md = ax.scatter(
            demo_mean_x, demo_mean_y,
            s=70,
            marker="o",
            color="cyan",
            label="mean_demos",
            zorder=9,
        )

        # label rollout mean explicitly (so it appears in legend)
        ax.scatter(
            mean_x, mean_y,
            s=70,
            marker="o",
            color="red",
            label="mean_rollouts",
            zorder=9,
        )
        # Labels
        ax.set_xlabel(feature_names[j])
        ax.set_ylabel(feature_names[i])

        #  return sc_r, sc_d
        return sc_r, sc_d, baseline_artists, sc_md 
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
            #  sc_r, sc_d = _scatter(ax, i, j)
            sc_r, sc_d, sc_bs, sc_md = _scatter(ax, i, j)
            if return_artists:
                #  artists.append({"pair": (i, j), "rollouts": sc_r, "demos": sc_d})
                artists.append({"pair": (i, j), "rollouts": sc_r, "demos": sc_d, "baselines": sc_bs, 'mean_demos': sc_md})

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
        #  sc_r, sc_d = _scatter(ax, 0, 1)
        sc_r, sc_d, sc_bs, sc_md = _scatter(ax, 0, 1)
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
            #  sc_r, sc_d, sc_bs = _scatter(ax, y_feat, x_feat)
            sc_r, sc_d, sc_bs, sc_md = _scatter(ax, y_feat, x_feat)
            if return_artists:
                artists.append({"pair": (i, j), "rollouts": sc_r, "demos": sc_d, "baselines": sc_bs, 'mean_demos': sc_md})
                #  artists.append({"pair": (y_feat, x_feat), "rollouts": sc_r, "demos": sc_d})

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
        baseline_palette = [
            "#66a61e",  # green
            'black'  ,  # black
            "#e7298a",  # magenta
            "#7570b3",  # muted purple
            "#a6761d",  # brown
            "#1b9e77",  # deep teal
            "#666666",  # dark gray
            "#8c6bb1",  # soft violet
            "#2b8cbe",  # steel blue (NOT bright blue)
        ]
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
                color = baseline_palette[c]
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

        ax.set_xlim(left=0)
        ax.set_ylim(bottom=0)

        xmin, xmax = ax.get_xlim()
        ymin, ymax = ax.get_ylim()

        ax.plot([mx_r, xmax], [my_r, my_r], lw=1.5, zorder=3, color = 'red')
        ax.plot([mx_r, mx_r], [my_r, ymax], lw=1.5, zorder=3, color = 'red')

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
