import numpy as np
import matplotlib.pyplot as plt
from stochastic_superhuman_fairness.core.utils_io import extract_log_key
from matplotlib.colors import ListedColormap, BoundaryNorm
# ==========================================================================================
# Extraction Helpers
# ==========================================================================================
def extract_gamma_series(
    logs,
    *,
    key="gamma_matrix",
):
    gammas, epochs = [], []

    for i, log in enumerate(logs):
        if key not in log:
            continue

        gamma = np.asarray(log[key], dtype=float)

        if gamma.ndim != 2:
            raise ValueError(f"Gamma at log {i} must be 2D, got {gamma.shape}")

        gammas.append(gamma)
        epochs.append(log.get("epoch", i))

    if not gammas:
        raise ValueError(f"No gamma matrices found under key '{key}'")

    shapes = {g.shape for g in gammas}
    if len(shapes) != 1:
        raise ValueError(f"Gamma matrices have inconsistent shapes: {shapes}")

    return np.stack(gammas, axis=0), np.asarray(epochs)

# ----------------------------------------

def extract_S_diagnostics_series(logs, *, key="S_diagnostics"):
    epochs = []
    vals = {}

    for i, log in enumerate(logs):
        if key not in log:
            continue

        d = log[key]
        epochs.append(log.get("epoch", i))

        for k, v in d.items():
            if np.isscalar(v):
                vals.setdefault(k, []).append(v)

    if not epochs:
        raise ValueError(f"No logs found with key '{key}'")

    epochs = np.asarray(epochs)

    vals = {
        k: np.asarray(v, dtype=float)
        for k, v in vals.items()
        if len(v) == len(epochs)
    }

    return epochs, vals

# ----------------------------------------

def extract_logged_gamma_stability(
    logs,
    *,
    diagnostics_key = 'gamma_diagnostics',
    keys=(
        "objective",
        "abs_delta",
        "rel_delta",
        "top1_flip_rate",
        "support_size",
    ),
):
    epochs = []
    vals = {k: [] for k in keys}

    for i, log in enumerate(logs):
        if diagnostics_key not in log:
        #  if not all(k in log[] for k in keys):
            continue

        epochs.append(log.get("epoch", i))

        for k in keys:
            vals[k].append(log[diagnostics_key][k])
    #  import ipdb;ipdb.set_trace()
    if not epochs:
        raise ValueError(f"No logs found containing all keys: {keys}")

    epochs = np.asarray(epochs)
    #  import ipdb;ipdb.set_trace()
    vals = {k: np.asarray(v, dtype=float) for k, v in vals.items()}

    return epochs, vals

# ----------------------------------------

def plot_gamma_snapshots(
    gammas,
    epochs=None,
    *,
    n_snapshots=6,
    ax=None,
    cmap="viridis",
    title="Coupling matrix snapshots",
):
    T, R, D = gammas.shape
    epochs = np.arange(T) if epochs is None else epochs

    idxs = np.linspace(0, T - 1, min(n_snapshots, T), dtype=int)

    if ax is None:
        fig, axes = plt.subplots(1, len(idxs), figsize=(4 * len(idxs), 4), squeeze=False)
        axes = axes.ravel()
    else:
        axes = np.asarray(ax).ravel()
        fig = axes[0].figure

    vmin, vmax = gammas.min(), gammas.max()

    for a, idx in zip(axes, idxs):
        im = a.imshow(gammas[idx], aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax)
        a.set_title(f"epoch {epochs[idx]}")
        a.set_xlabel("demos")
        a.set_ylabel("rollouts")

    fig.colorbar(im, ax=axes.tolist(), fraction=0.02, pad=0.02)
    fig.suptitle(title)
    fig.tight_layout()
    return fig, axes

# ----------------------------------------

def plot_demo_coverage_over_time(
    gammas,
    epochs=None,
    *,
    ax=None,
    cmap="viridis",
    title="Demo coverage over time",
):
    epochs = np.arange(len(gammas)) if epochs is None else epochs

    # shape: T, D
    demo_mass = gammas.sum(axis=1)

    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 4))
    else:
        fig = ax.figure

    im = ax.imshow(
        demo_mass.T,
        aspect="auto",
        origin="lower",
        cmap=cmap,
        extent=[epochs[0], epochs[-1], 0, demo_mass.shape[1] - 1],
    )

    ax.set_title(title)
    ax.set_xlabel("epoch")
    ax.set_ylabel("demo index")
    fig.colorbar(im, ax=ax, label="column mass")

    return fig, ax


# ----------------------------------------

def plot_rollout_activity_over_time(
    gammas,
    epochs=None,
    *,
    ax=None,
    cmap="viridis",
    title="Rollout activity over time",
):
    epochs = np.arange(len(gammas)) if epochs is None else epochs

    # shape: T, R
    rollout_mass = gammas.sum(axis=2)

    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 4))
    else:
        fig = ax.figure

    im = ax.imshow(
        rollout_mass.T,
        aspect="auto",
        origin="lower",
        cmap=cmap,
        extent=[epochs[0], epochs[-1], 0, rollout_mass.shape[1] - 1],
    )

    ax.set_title(title)
    ax.set_xlabel("epoch")
    ax.set_ylabel("rollout index")
    fig.colorbar(im, ax=ax, label="row mass")

    return fig, ax

# ----------------------------------------

def plot_top1_rollout_per_demo(
    gammas,
    epochs=None,
    *,
    ax=None,
    n_buckets: int | None = None,
    palette: list[str] | None = None,
    title="Top-1 rollout per demo over time",
):
    gammas = np.asarray(gammas)
    T, R, D = gammas.shape
    epochs = np.arange(T) if epochs is None else np.asarray(epochs)

    top1 = np.argmax(gammas, axis=1)  # [T, D]

    if n_buckets is None:
        cmap, norm, label = "turbo", None, "Top rollout"
    else:
        if not (1 <= n_buckets <= R):
            raise ValueError(f"n_buckets must be in [1, {R}]")

        top1 = (top1 * n_buckets) // R

        if palette is not None:
            if len(palette) < n_buckets:
                raise ValueError(
                    f"palette has {len(palette)} colors but n_buckets={n_buckets}"
                )
            cmap = ListedColormap(palette[:n_buckets])
        else:
            base = plt.cm.tab10 if n_buckets <= 10 else plt.cm.tab20 if n_buckets <= 20 else plt.cm.turbo
            cmap = ListedColormap(base(np.linspace(0, 1, n_buckets)))

        norm = BoundaryNorm(np.arange(-0.5, n_buckets + 0.5), n_buckets)
        label = "Rollout bucket"

    fig, ax = plt.subplots(figsize=(8, 4)) if ax is None else (ax.figure, ax)

    im = ax.imshow(
        top1.T,
        aspect="auto",
        origin="lower",
        interpolation="nearest",
        cmap=cmap,
        norm=norm,
    )

    ax.set(title=title, xlabel="Epoch", ylabel="Demo")

    cbar = fig.colorbar(im, ax=ax, label=label)
    if n_buckets is not None:
        cbar.set_ticks(np.arange(n_buckets))

    return fig, ax

#--------------------------------------------------------
def plot_top1_demo_per_rollout(
    gammas,
    epochs=None,
    *,
    ax=None,
    n_buckets=None,
    palette=None,
    color_order="cold_to_warm",
    title="Top-1 demo per rollout over time",
):
    gammas = np.asarray(gammas)

    if gammas.ndim != 3:
        raise ValueError(f"Expected [T,R,D], got {gammas.shape}")

    T, R, D = gammas.shape
    epochs = np.arange(T) if epochs is None else np.asarray(epochs)

    if len(epochs) != T:
        raise ValueError(f"epochs must have length {T}")

    if color_order not in ("cold_to_warm", "warm_to_cold"):
        raise ValueError("color_order must be 'cold_to_warm' or 'warm_to_cold'")

    vals = np.argmax(gammas, axis=2)  # [T,R]

    if n_buckets is None:
        n_levels = D
        colors = plt.cm.turbo(np.linspace(0, 1, max(D, 2)))
        label = "Best demo"
    else:
        if not 1 <= n_buckets <= D:
            raise ValueError(f"n_buckets must be in [1,{D}]")

        vals = np.minimum((vals * n_buckets) // D, n_buckets - 1)
        n_levels = n_buckets
        colors = (
            palette[:n_buckets]
            if palette is not None
            else plt.cm.coolwarm(np.linspace(0, 1, n_buckets))
        )
        label = "Demo bucket"

    if color_order == "warm_to_cold":
        colors = colors[::-1]

    cmap = ListedColormap(colors)
    norm = BoundaryNorm(np.arange(-0.5, n_levels + 0.5), n_levels)

    fig, ax = plt.subplots(figsize=(8, 4)) if ax is None else (ax.figure, ax)

    im = ax.imshow(
        vals.T,
        aspect="auto",
        origin="lower",
        interpolation="nearest",
        cmap=cmap,
        norm=norm,
        extent=[epochs[0], epochs[-1], -0.5, R - 0.5],
    )

    ax.set(title=title, xlabel="Epoch", ylabel="Rollout")

    cbar = fig.colorbar(im, ax=ax, label=label)

    if n_buckets is not None:
        edges = np.linspace(0, D, n_buckets + 1, dtype=int)
        cbar.set_ticks(np.arange(n_buckets))
        cbar.set_ticklabels([
            f"{edges[i]}–{edges[i + 1] - 1}"
            for i in range(n_buckets)
        ])

    return fig, ax

#--------------------------------------------------------

def plot_demo_entropy_over_time(
    gammas,
    epochs=None,
    *,
    ax=None,
    cmap="magma",
    eps=1e-12,
    title="Demo matching entropy over time",
):
    epochs = np.arange(len(gammas)) if epochs is None else epochs

    # normalize each column over rollouts
    col_mass = gammas.sum(axis=1, keepdims=True)
    p = gammas / np.maximum(col_mass, eps)

    # entropy shape: T, D
    entropy = -(p * np.log(np.maximum(p, eps))).sum(axis=1)

    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 4))
    else:
        fig = ax.figure

    im = ax.imshow(
        entropy.T,
        aspect="auto",
        origin="lower",
        cmap=cmap,
        extent=[epochs[0], epochs[-1], 0, entropy.shape[1] - 1],
    )

    ax.set_title(title)
    ax.set_xlabel("epoch")
    ax.set_ylabel("demo index")
    fig.colorbar(im, ax=ax, label="entropy")

    return fig, ax

# ----------------------------------------

def plot_gamma_delta_norm(
    gammas,
    epochs=None,
    *,
    ax=None,
    title="Coupling change over time",
):
    epochs = np.arange(len(gammas)) if epochs is None else epochs

    delta = np.diff(gammas, axis=0)
    delta_norm = np.linalg.norm(delta.reshape(len(delta), -1), axis=1)

    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 3))
    else:
        fig = ax.figure

    ax.plot(epochs[1:], delta_norm)
    ax.set_title(title)
    ax.set_xlabel("epoch")
    ax.set_ylabel(r"$||\gamma_t - \gamma_{t-1}||_F$")
    ax.grid(True, alpha=0.3)

    return fig, ax

# ----------------------------------------

def plot_topk_sparsified_gamma(
    gammas,
    epochs=None,
    *,
    k=3,
    t=-1,
    ax=None,
    cmap="viridis",
    title="Top-k sparsified coupling",
):
    epochs = np.arange(len(gammas)) if epochs is None else epochs

    G = gammas[t].copy()
    R, D = G.shape

    sparse = np.zeros_like(G)

    for j in range(D):
        idx = np.argsort(G[:, j])[-k:]
        sparse[idx, j] = G[idx, j]

    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 5))
    else:
        fig = ax.figure

    im = ax.imshow(sparse, aspect="auto", cmap=cmap)
    ax.set_title(f"{title}, epoch {epochs[t]}")
    ax.set_xlabel("demos")
    ax.set_ylabel("rollouts")
    fig.colorbar(im, ax=ax, label="gamma")

    return fig, ax

# ----------------------------------------

def plot_clustered_gamma(
    gammas,
    epochs=None,
    *,
    t=-1,
    ax=None,
    cmap="viridis",
    title="Clustered coupling matrix",
):
    from scipy.cluster.hierarchy import linkage, leaves_list

    epochs = np.arange(len(gammas)) if epochs is None else epochs
    G = gammas[t]

    row_order = leaves_list(linkage(G, method="average"))
    col_order = leaves_list(linkage(G.T, method="average"))

    Gc = G[row_order][:, col_order]

    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 5))
    else:
        fig = ax.figure

    im = ax.imshow(Gc, aspect="auto", cmap=cmap)
    ax.set_title(f"{title}, epoch {epochs[t]}")
    ax.set_xlabel("clustered demos")
    ax.set_ylabel("clustered rollouts")
    fig.colorbar(im, ax=ax, label="gamma")

    return fig, ax, row_order, col_order

# ============================================================================================================
# Main combinining function 
# ============================================================================================================
def plot_gamma_diagnostics_dashboard(
    logs,
    *,
    gamma_keys='gamma_matrix',
    n_snapshots=4,
    topk=1,
    palette = None,
    figsize=(18, 14),
):
    gammas, epochs = extract_gamma_series(logs, key=gamma_keys)
    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(6, 2)

    ax_demo = fig.add_subplot(gs[0, 0])
    ax_rollout = fig.add_subplot(gs[0, 1])
    ax_top1 = fig.add_subplot(gs[1, 0])
    ax_entropy = fig.add_subplot(gs[1, 1])
    ax_top1_d = fig.add_subplot(gs[2, 0])
    ax_topk = fig.add_subplot(gs[2, 1])
    ax_cluster = fig.add_subplot(gs[3, 0])
    ax_snapshot = fig.add_subplot(gs[3, 1])
    ax_stability_1 = fig.add_subplot(gs[4, 0])
    ax_stability_2 = fig.add_subplot(gs[4, 1])
    ax_Sgap = fig.add_subplot(gs[5, 0])
    ax_delta = fig.add_subplot(gs[5, 1])

    plot_demo_coverage_over_time(gammas, epochs, ax=ax_demo)
    plot_rollout_activity_over_time(gammas, epochs, ax=ax_rollout)
    plot_top1_rollout_per_demo(gammas, epochs, ax=ax_top1, n_buckets = extract_log_key(logs, 'train/P', extract_num=1)[0], palette = palette)
    plot_top1_demo_per_rollout(gammas, epochs, ax=ax_top1_d, n_buckets = None, palette = palette, color_order='warm_to_cold')
    plot_demo_entropy_over_time(gammas, epochs, ax=ax_entropy)
    plot_gamma_delta_norm(gammas, epochs, ax=ax_delta)
    plot_topk_sparsified_gamma(gammas, epochs, k=topk, ax=ax_topk)
    plot_logged_gamma_stability_compact(
    logs,
    ax_stability_1,
    ax_stability_2,
    )
    plot_S_gap_diagnostics_compact(logs, ax=ax_Sgap)
    try:
        plot_clustered_gamma(gammas, epochs, ax=ax_cluster)
    except Exception as e:
        ax_cluster.text(0.5, 0.5, f"Cluster plot failed:\n{e}", ha="center", va="center")
        ax_cluster.set_axis_off()

    # one final raw gamma snapshot
    im = ax_snapshot.imshow(gammas[-1], aspect="auto", cmap="viridis")
    ax_snapshot.set_title(f"Raw gamma, epoch {epochs[-1]}")
    ax_snapshot.set_xlabel("demos")
    ax_snapshot.set_ylabel("rollouts")
    fig.colorbar(im, ax=ax_snapshot, fraction=0.046, pad=0.04)

    fig.suptitle("Gamma / Coupling Diagnostics", fontsize=16)
    fig.tight_layout()

    return fig, {
        "gammas": gammas,
        "epochs": epochs,
        "axes": {
            "demo_coverage": ax_demo,
            "rollout_activity": ax_rollout,
            "top1": ax_top1,
            "entropy": ax_entropy,
            "delta_norm": ax_delta,
            "topk": ax_topk,
            "clustered": ax_cluster,
            "snapshot": ax_snapshot,
        },
    }

# ----------------------------------------------------------------------------------------

def _as_1d_metric(v, reduce="mean")->np.ndarray:
    v = np.asarray(v, dtype=float)

    if v.ndim == 1:
        return v

    if v.ndim == 2:
        if reduce == "mean":
            return v.mean(axis=1)
        elif reduce == "max":
            return v.max(axis=1)
        elif reduce == "norm":
            return np.linalg.norm(v, axis=1)
        else:
            raise ValueError(f"Unknown reduce={reduce}")

    return v.reshape(v.shape[0], -1).mean(axis=1)

def plot_logged_gamma_stability(
    logs,
    *,
    keys=(
        "objective",
        "abs_delta",
        "rel_delta",
        "top1_flip_rate",
        "support_size",
    ),
    ax=None,
    title="Gamma stability metrics",
    figsize=(9, 9),
):
    epochs, vals = extract_logged_gamma_stability(logs, keys=keys)

    n = len(keys)

    if ax is None:
        fig, axes = plt.subplots(n, 1, figsize=figsize, sharex=True)
    else:
        axes = np.asarray(ax).ravel()
        fig = axes[0].figure
        if len(axes) < n:
            raise ValueError(f"Need at least {n} axes, got {len(axes)}")

    for a, k in zip(axes, keys):
        a.plot(epochs, vals[k])
        a.set_ylabel(k)
        a.grid(True, alpha=0.3)

    axes[0].set_title(title)
    axes[-1].set_xlabel("epoch")

    fig.tight_layout()

    return fig, axes, vals

def plot_logged_gamma_stability_compact(
    logs,
    ax1,
    ax2,
    *,
    keys=("objective", "abs_delta", "rel_delta", "top1_flip_rate", "support_size"),
):
    epochs, vals = extract_logged_gamma_stability(logs, keys=keys)

    vals = {k: _as_1d_metric(v, reduce = 'mean') for k, v in vals.items()}

    # -------- instability axis --------
    for k in ("rel_delta", "abs_delta", "top1_flip_rate"):
        if k not in vals:
            continue

        y = vals[k]

        # delta-like metrics may be T-1
        if len(y) == len(epochs) - 1:
            x = epochs[1:]
        elif len(y) == len(epochs):
            x = epochs
        else:
            raise ValueError(
                f"{k} has incompatible length {len(y)} for epochs length {len(epochs)}"
            )

        ax1.plot(x, y, label=f"mean {k} gamma")

    ax1.set_title("Gamma instability")
    ax1.set_xlabel("epoch")
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    # -------- structure axis --------
    for k in ("support_size", "objective"):
        if k not in vals:
            continue

        y = vals[k]

        if len(y) == len(epochs) - 1:
            x = epochs[1:]
        elif len(y) == len(epochs):
            x = epochs
        else:
            raise ValueError(
                f"{k} has incompatible length {len(y)} for epochs length {len(epochs)}"
            )
        label = f" Support size (γ > 1e-9)" if k == 'support_size' else "Objective"
        ax2.plot(x, y, label=k)

    ax2.set_title("Gamma structure / objective")
    ax2.set_xlabel("epoch")
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    return ax1, ax2, vals



def plot_S_gap_diagnostics_compact(
    logs,
    *,
    key="S_diagnostics",
    ax=None,
    title="S cost-gap diagnostics",
    figsize=(10, 4),
):
    epochs, vals = extract_S_diagnostics_series(logs, key=key)

    if ax is None:
        fig, ax1 = plt.subplots(figsize=figsize)
    else:
        ax1 = ax
        fig = ax1.figure

    gap_keys = [k for k in ("mean_gap", "median_gap", "gap_over_best") if k in vals]
    tie_keys = [
        k for k in (
            "frac_near_tie_1e_4",
            "frac_near_tie_1e_3",
            "frac_near_tie_1e_2",
        )
        if k in vals
    ]

    for k in gap_keys:
        ax1.plot(epochs, vals[k], label=k)

    ax1.set_title(title)
    ax1.set_xlabel("epoch")
    ax1.set_ylabel("gap")
    ax1.grid(True, alpha=0.3)

    ax2 = None
    if tie_keys:
        ax2 = ax1.twinx()
        for k in tie_keys:
            ax2.plot(epochs, vals[k], linestyle="--", label=k)
        ax2.set_ylabel("near-tie fraction")

    lines1, labels1 = ax1.get_legend_handles_labels()
    if ax2 is not None:
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, fontsize=8, loc="best")
    else:
        ax1.legend(fontsize=8, loc="best")

    return fig, (ax1, ax2), vals
