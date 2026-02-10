import numpy as np
import matplotlib.pyplot as plt
import math


from stochastic_superhuman_fairness.core.plotting.plot_utils import (
    compute_feature_means,
    annotate_mean_with_guides,
    inv_alpha_for_dims,
    annotate_inverse_alpha_arrows,
)


def plot_rollouts_vs_demos_old(
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

    Modes:
      - pairs=None:
          If K==2: single plot.
          If K>2: upper-triangular matrix (fi vs fj for j>i).
      - pairs=[(i,j), ...]:
          Plot only the specified feature pairs (fi vs fj).

    Optional margins:
      alpha: scalar or length-K. Vertical line at x=alpha[j] in subplot (i,j).
      beta:  scalar or length-K. Horizontal line at y=beta[i] in subplot (i,j).

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

    artists = [] if return_artists else None

    def _scatter(ax, i, j):
        # y = fi, x = fj
        sc_r = ax.scatter(
            rollouts[:, j], rollouts[:, i],
            s=s_rollouts, marker=marker_rollouts, label=rollout_label
        )
        sc_d = ax.scatter(
            demos[:, j], demos[:, i],
            s=s_demos, marker=marker_demos, label=demo_label
        )

        if alpha_v is not None:
            ax.axvline(alpha_v[j])
        if beta_v is not None:
            ax.axhline(beta_v[i])

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
