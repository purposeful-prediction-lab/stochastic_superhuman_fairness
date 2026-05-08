import numpy as np
import matplotlib.pyplot as plt
import math
import torch
from stochastic_superhuman_fairness.core.plotting.plot_utils import  (
        _add_row_group_colors_to_heatmap, _to_np,
        normalize_rollout_modes,
        )


def plot_subdominance_heatmap(
    S,
    *,
    ax = None,
    rollout_labels=None,
    demo_labels=None,
    title="Subdominance heatmap",
    figsize=(8, 6),
    normalize="none",
    clip_percentile=None,
    annotate=False,
    fmt=".2f",
    show_colorbar=True,
    row_groups=None,          
    group_colors=None,        
    group_strip_width=2.18,   
):
    """
    Plot subdominance matrix S[r,d] as heatmap.

    Parameters
    ----------
    normalize:
        "none"   -> raw values
        "log"    -> log1p(S)
        "zscore" -> (S - mean)/std
        "minmax" -> rescale to [0,1]
    clip_percentile:
        If not None, clip values to given percentile (robust visualization).
    annotate:
        If True and matrix small (< 50x50 recommended), annotate cell values.
    """

    A = S.detach().cpu().numpy() if torch.is_tensor(S) else np.asarray(S)
    if A.ndim != 2:
        raise ValueError(f"S must be 2D (R,D). Got shape {A.shape}.")
    R, D = A.shape

    # ---- optional clipping ----
    if clip_percentile is not None:
        vmax = np.percentile(A, clip_percentile)
        A = np.clip(A, None, vmax)

    # ---- normalization ----
    if normalize == "log":
        A_plot = np.log1p(A)
    elif normalize == "zscore":
        A_plot = (A - A.mean()) / (A.std() + 1e-12)
    elif normalize == "minmax":
        mn, mx = A.min(), A.max()
        A_plot = (A - mn) / (mx - mn + 1e-12)
    elif normalize == "none":
        A_plot = A
    else:
        raise ValueError("normalize must be 'none', 'log', 'zscore', or 'minmax'.")

    # ---- plotting ----
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=figsize)
    else:
        ax = ax
        fig = ax.figure


    im = ax.imshow(A_plot, aspect="auto")

    ax.set_title(title)
    ax.set_xlabel("demos")
    ax.set_ylabel("rollouts")

    # ---- optional axis labels ----
    if demo_labels is not None and D <= 50:
        ax.set_xticks(np.arange(D))
        ax.set_xticklabels([str(x) for x in demo_labels], rotation=90)
    if rollout_labels is not None and R <= 50:
        ax.set_yticks(np.arange(R))
        ax.set_yticklabels([str(x) for x in rollout_labels])

    _add_row_group_colors_to_heatmap(
            ax,
            R,
            row_groups=row_groups,
            group_colors=group_colors,
            strip_width=group_strip_width,
        )
    # ---- annotation ----
    if annotate and R * D <= 2500:  # avoid insane slowdowns
        for i in range(R):
            for j in range(D):
                ax.text(
                    j, i,
                    format(A[i, j], fmt),
                    ha="center",
                    va="center",
                    fontsize=7,
                    color="black"
                )

    if show_colorbar:
        fig.colorbar(im, ax=ax)

    fig.tight_layout()
    return fig, ax

def plot_ot_solution_heatmaps(
    sol: dict,
    *,
    ax = None,
    mode: str = "replicate",
    title="Optimal transport (heatmaps)",
    gamma_normalize="none",
    gamma_clip_percentile=None,
    gamma_temperature: float = 1.0,
    figsize=(14, 4),
    show_colorbar=True,
    row_groups=None,          # NEW
    group_colors=None,        # NEW
    group_strip_width=2.18,   # NEW
):
    """
    Plot OT solution using heatmaps.

    sol must contain:
      - gamma_np: (R,D)
      - dual_rows_np: (R,)
      - dual_cols_np: (D,)

    mode:
      - "replicate": show u replicated to (R,D) and v replicated to (R,D)
      - "sum": show (u+v) as a single (R,D) heatmap instead of separate u/v heatmaps

    Returns (fig, axes).
    """
    def _np(x):
        if torch.is_tensor(x):
            return x.detach().cpu().numpy()
        return np.asarray(x)

    G = _np(sol["gamma_np"]).astype(float) * gamma_temperature
    u = _np(sol["dual_rows_np"]).reshape(-1).astype(float)
    v = _np(sol["dual_cols_np"]).reshape(-1).astype(float)

    if G.ndim != 2:
        raise ValueError(f"gamma_np must be 2D. Got {G.shape}.")
    R, D = G.shape
    if u.size != R: raise ValueError(f"dual_rows_np must have length R={R}. Got {u.size}.")
    if v.size != D: raise ValueError(f"dual_cols_np must have length D={D}. Got {v.size}.")

    # gamma preprocessing
    Gp = G.copy()
    if gamma_clip_percentile is not None:
        vmax = float(np.percentile(Gp, gamma_clip_percentile))
        Gp = np.clip(Gp, None, vmax)

    gn = gamma_normalize.lower()
    if gn == "log":
        Gp = np.log1p(Gp)
    elif gn == "minmax":
        mn, mx = float(Gp.min()), float(Gp.max())
        Gp = (Gp - mn) / (mx - mn + 1e-12)
    elif gn != "none":
        raise ValueError("gamma_normalize must be 'none', 'log', or 'minmax'.")

    mode = mode.lower()
    if mode not in {"replicate", "sum"}:
        raise ValueError("mode must be 'replicate' or 'sum'.")
    if ax is None:
        fig, axes = plt.subplots(1, 1, figsize=figsize)
    else:
        axes = ax
        fig = ax.figure

    #  fig, axes = plt.subplots(1, 3, figsize=figsize)

    # 1) gamma
    im0 = axes.imshow(Gp, aspect="auto")
    _add_row_group_colors_to_heatmap(
        axes,
        R,
        row_groups=row_groups,
        group_colors=group_colors,
        strip_width=group_strip_width,
    )
    axes.set_title(r"$\gamma$")
    axes.set_xlabel("demos")
    axes.set_ylabel("rollouts")
    if show_colorbar: fig.colorbar(im0, ax=axes, fraction=0.046, pad=0.04)

    fig.suptitle(title)
    fig.tight_layout()
    return fig, axes

def plot_optimal_transport_solution(
    sol: dict,
    *,
    rollout_labels=None,     # length R (optional)
    demo_labels=None,        # length D (optional)
    title="Optimal transport solution",
    gamma_normalize="none",  # "none" | "log" | "minmax"
    gamma_clip_percentile=None,
    annotate_gamma=False,
    gamma_fmt=".2f",
    figsize=(14, 4),
    show_colorbar=True,
):
    """
    Plot OT solution returned as:
      {
        "gamma_np": (R,D),
        "dual_rows_np": (R,),
        "dual_cols_np": (D,)
      }

    Recommended visualization:
      - gamma: heatmap (transport plan)
      - dual_rows / dual_cols: line plots (potentials)

    Parameters
    ----------
    gamma_normalize:
      "none"  -> raw gamma
      "log"   -> log1p(gamma)
      "minmax"-> rescale to [0,1]
    gamma_clip_percentile:
      clip gamma at this percentile (e.g., 99) for visibility
    annotate_gamma:
      annotate gamma values if small enough
    """
    # --- fetch + coerce to numpy ---
    gamma = sol.get("gamma_np", None)
    u = sol.get("dual_rows_np", None)
    v = sol.get("dual_cols_np", None)
    if gamma is None or u is None or v is None:
        raise ValueError("sol must contain keys: 'gamma_np', 'dual_rows_np', 'dual_cols_np'.")

    def _np(x):
        if torch.is_tensor(x):
            return x.detach().cpu().numpy()
        return np.asarray(x)

    G = _np(gamma)
    u = _np(u).reshape(-1)
    v = _np(v).reshape(-1)

    if G.ndim != 2:
        raise ValueError(f"gamma_np must be 2D (R,D). Got shape {G.shape}.")
    R, D = G.shape
    if u.size != R:
        raise ValueError(f"dual_rows_np must have length R={R}. Got {u.size}.")
    if v.size != D:
        raise ValueError(f"dual_cols_np must have length D={D}. Got {v.size}.")

    # --- optional clip ---
    G_plot = G.astype(float)
    if gamma_clip_percentile is not None:
        vmax = float(np.percentile(G_plot, gamma_clip_percentile))
        G_plot = np.clip(G_plot, None, vmax)

    # --- optional normalization ---
    gn = gamma_normalize.lower()
    if gn == "log":
        G_plot = np.log1p(G_plot)
    elif gn == "minmax":
        mn, mx = float(G_plot.min()), float(G_plot.max())
        G_plot = (G_plot - mn) / (mx - mn + 1e-12)
    elif gn == "none":
        pass
    else:
        raise ValueError("gamma_normalize must be 'none', 'log', or 'minmax'.")

    # --- figure ---
    fig, axes = plt.subplots(1, 3, figsize=figsize, gridspec_kw={"width_ratios": [1.3, 1.0, 1.0]})
    axG, axU, axV = axes

    # 1) gamma heatmap
    im = axG.imshow(G_plot, aspect="auto")
    axG.set_title(r"$\gamma$ (transport plan)")
    axG.set_xlabel("demos")
    axG.set_ylabel("rollouts")

    if demo_labels is not None and D <= 50:
        axG.set_xticks(np.arange(D))
        axG.set_xticklabels([str(x) for x in demo_labels], rotation=90)
    if rollout_labels is not None and R <= 50:
        axG.set_yticks(np.arange(R))
        axG.set_yticklabels([str(x) for x in rollout_labels])

    if annotate_gamma and R * D <= 2500:
        for i in range(R):
            for j in range(D):
                axG.text(j, i, format(G[i, j], gamma_fmt), ha="center", va="center", fontsize=7)

    if show_colorbar:
        fig.colorbar(im, ax=axG, fraction=0.046, pad=0.04)

    # 2) dual rows
    axU.plot(np.arange(R), u)
    axU.set_title("dual_rows")
    axU.set_xlabel("rollout index")
    axU.set_ylabel("value")
    if rollout_labels is not None and R <= 50:
        axU.set_xticks(np.arange(R))
        axU.set_xticklabels([str(x) for x in rollout_labels], rotation=90)

    # 3) dual cols
    axV.plot(np.arange(D), v)
    axV.set_title("dual_cols")
    axV.set_xlabel("demo index")
    axV.set_ylabel("value")
    if demo_labels is not None and D <= 50:
        axV.set_xticks(np.arange(D))
        axV.set_xticklabels([str(x) for x in demo_labels], rotation=90)

    fig.suptitle(title)
    fig.tight_layout()
    return fig, axes

def plot_indicator_matrix(
    M,
    *,
    ax = None,
    row_labels=None,
    col_labels=None,
    title="Indicator Matrix",
    figsize=(8, 6),
    cmap="Greys",
    show_colorbar=False,
    annotate=False,
    row_groups=None,
    group_colors=None,
    group_strip_width=0.18,
):
    """
    Plot a 0/1 indicator matrix with optional row-group color strip.

    row_groups example:
        [[0,1,2,3], [4,5], [6,7,8]]
    """
    M = np.asarray(M)
    if M.ndim != 2:
        raise ValueError("M must be 2D.")
    if not np.all(np.isin(M, [0, 1])):
        raise ValueError("M must contain only 0s and 1s.")

    R, C = M.shape

    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=figsize)
    else:
        ax = ax
        fig = ax.figure

    im = ax.imshow(M, cmap=cmap, aspect="auto", vmin=0, vmax=1)

    _add_row_group_colors_to_heatmap(
        ax,
        R,
        row_groups=row_groups,
        group_colors=group_colors,
        strip_width=group_strip_width,
    )

    if row_labels is not None:
        if len(row_labels) != R:
            raise ValueError(f"row_labels must have length {R}.")
        ax.set_yticks(np.arange(R))
        ax.set_yticklabels(row_labels)

    if col_labels is not None:
        if len(col_labels) != C:
            raise ValueError(f"col_labels must have length {C}.")
        ax.set_xticks(np.arange(C))
        ax.set_xticklabels(col_labels, rotation=45, ha="right")

    if annotate:
        for i in range(R):
            for j in range(C):
                ax.text(j, i, str(int(M[i, j])), ha="center", va="center", fontsize=8)

    ax.set_title(title)
    ax.set_xlabel("Columns")
    ax.set_ylabel("Rows")

    if show_colorbar:
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig.tight_layout()
    return fig, ax

def save_heatmap(
    matrix,
    filename="heatmap.png",
    xlabel="Columns",
    ylabel="Rows",
    title="Heatmap",
    cmap="viridis",
    normalize=False,
):
    if torch.is_tensor(matrix):
        M = matrix.detach().cpu().clone()
    else:
        M = np.array(matrix, copy=True)

    # Optional min–max normalization
    if normalize:
        m_min = M.min()
        m_max = M.max()
        if m_max > m_min:
            M = (M - m_min) / (m_max - m_min)

    plt.figure()
    if (M < 0 ).any():
        print('Negative values for input M @ save_heatmap function!')
    im = plt.imshow(M, cmap=cmap, aspect="auto")
    plt.colorbar(im)

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)

    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close()


def plot_dominance_counts(
    logs,
    *,
    ax=None,                 # <-- NEW
    title="Dominance counts per epoch",
    figsize=(9, 4),
    per_mode=None,
    x_timesteps=None,
    palette=None,
    annotate_per_mode: bool = False,
    bar_width=10.35,
    bar_alpha=0.45,
):
    """
    Same as before, but can plot into a provided axis.
    """

    def _get_nested_mixed(d, path, default=None):
        if not isinstance(d, dict):
            return default
        if path in d:
            return d[path]

        parts = path.split("/")
        cur = d
        i = 0
        while i < len(parts):
            if not isinstance(cur, dict):
                return default

            found = False
            for j in range(len(parts), i, -1):
                key = "/".join(parts[i:j])
                if key in cur:
                    cur = cur[key]
                    i = j
                    found = True
                    break

            if not found:
                return default

        return cur

    dom_rollouts, dom_demos, epochs = [], [], []

    for i, log in enumerate(logs):
        ltd = log.get("train/l_terms", {})
        if "dominant_rollouts" not in ltd or "dominant_demos" not in ltd:
            continue
        epochs.append(i)
        dom_rollouts.append(ltd["dominant_rollouts"])
        dom_demos.append(ltd["dominant_demos"])

    if not epochs:
        raise ValueError("No log entries with dominant counts found.")

    dom_rollouts = np.asarray(dom_rollouts, dtype=float)
    dom_demos = np.asarray(dom_demos, dtype=float)
    epochs = np.asarray(epochs, dtype=int)

    # ---------- axis handling ----------
    created_fig = False
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=figsize)
        created_fig = True
    else:
        fig = ax.figure

    # ---------- main lines ----------
    ax.plot(epochs, dom_demos, label="dominant_demos")
    ax.plot(epochs, dom_rollouts, label="dominant_rollouts")

    # ---------- per-mode bars ----------
    if per_mode is not None:
        if isinstance(per_mode, str):
            per_mode_vals = [_get_nested_mixed(log, per_mode, None) for log in logs]
        else:
            per_mode_vals = per_mode

        if len(per_mode_vals) != len(logs):
            raise ValueError("per_mode must align with logs.")

        if x_timesteps is None:
            x_timesteps = epochs.tolist()

        first_valid = next((v for v in per_mode_vals if v is not None), None)
        if first_valid is None:
            raise ValueError("per_mode provided but no valid entries.")

        n_modes = len(first_valid)
        if palette is None:
            cmap = plt.cm.get_cmap("tab10", n_modes)
            palette = [cmap(i) for i in range(n_modes)]

        if len(palette) < n_modes:
            raise ValueError(f"palette must have at least {n_modes} colors.")

        for x in x_timesteps:
            if x < 0 or x >= len(per_mode_vals) or per_mode_vals[x] is None:
                continue

            vals = np.asarray(per_mode_vals[x], dtype=float)

            if vals.shape[0] != n_modes:
                raise ValueError(
                    f"Inconsistent per_mode size at timestep {x}: "
                    f"expected {n_modes}, got {vals.size}."
                )

            bottom = 0.0

            # reverse so mode 0 is top
            for mode_idx in reversed(range(n_modes)):
                h = sum(vals[mode_idx])

                ax.bar(
                    x,
                    h,
                    width=bar_width,
                    bottom=bottom,
                    color=palette[mode_idx],
                    alpha=bar_alpha,
                    align="center",
                    edgecolor="none",
                    zorder=0,
                )

                if annotate_per_mode:
                    ax.text(
                        x,
                        bottom + h / 2,
                        f"{int(h)}",
                        ha="center",
                        va="center",
                        fontsize=5,
                        color="black",
                        zorder=5,
                    )

                bottom += h

    # ---------- labels ----------
    ax.set_xlabel("epoch")
    ax.set_ylabel("count")
    ax.set_title(title)
    ax.legend()

    if per_mode is not None:
        ax.text(
            0.01, 0.98,
            "Stacked bars: per-mode selected rollouts",
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=9,
            bbox=dict(
                boxstyle="round,pad=0.25",
                facecolor="white",
                alpha=0.8,
                edgecolor="none",
            ),
        )

    if created_fig:
        fig.tight_layout()

    return fig, ax

def plot_paired_subdominance_curve(
    logs,
    *,
    ax = None,
    title="Paired Subdominance Curve",
    figsize=(9, 4),
):
    """
    Plot, over epochs:
      - S_mean
      - norm_paired_subdom
      - ratio = norm_paired_subdom / S_mean   (on right y-axis)

    Expects each log entry to contain:
      log["S_mean"]
      log["norm_paired_subdom"]

    Returns
    -------
    fig, (ax_left, ax_right)
    """
    epochs, s_mean, norm_paired, ratio = [], [], [], []

    pair_key = "paired_subdom" # was norm_paired_subdom before 560
    for i, log in enumerate(logs):
        ltd = log.get("train/l_terms", {})
        if "S_mean" not in ltd or pair_key not in ltd:
            continue
        s = float(ltd["S_mean"])
        n = float(ltd[pair_key])
        epochs.append(i)
        s_mean.append(s)
        norm_paired.append(n)
        ratio.append(n / s if abs(s) > 1e-12 else np.nan)
    if not epochs:
        raise ValueError(f"No log entries with 'S_mean' and {pair_key} were found.")

    epochs = np.asarray(epochs)
    s_mean = np.asarray(s_mean)
    norm_paired = np.asarray(norm_paired)
    ratio = np.asarray(ratio)
    if ax is None:
        fig, ax_left = plt.subplots(1, 1, figsize=figsize)
    else:
        ax_left = ax
        fig = ax.figure

    ax_right = ax_left.twinx()
    l1 = ax_left.plot(epochs, s_mean, color = 'red', label="S_mean")
    l2 = ax_left.plot(epochs, norm_paired, color = 'navy', label=f"pair_key/ OT loss")
    l3 = ax_right.plot(epochs, ratio, color = 'violet', linestyle="--", label=f"{pair_key} / S_mean")

    ax_left.set_xlabel("epoch")
    ax_left.set_ylabel("value")
    ax_right.set_ylabel("ratio")
    ax_left.tick_params(axis="y", labelcolor='blue')
    ax_right.tick_params(axis="y", labelcolor='red')


    ax_left.set_title(title)

    lines = l1 + l2 + l3
    labels = [ln.get_label() for ln in lines]
    ax_left.legend(lines, labels)

    fig.tight_layout()
    return fig, (ax_left, ax_right)

#----------------------------------------------------------------------------------------------------


def _format_cfg(obj, indent=0):
    sp = "  " * indent

    if isinstance(obj, dict):
        lines = []
        for k, v in obj.items():
            if isinstance(v, (dict, list, tuple)):
                lines.append(f"{sp}{k}:")
                lines.append(_format_cfg(v, indent + 1))
            else:
                lines.append(f"{sp}{k}: {v}")
        return "\n".join(lines)

    if isinstance(obj, (list, tuple)):
        lines = []
        for v in obj:
            if isinstance(v, (dict, list, tuple)):
                lines.append(f"{sp}-")
                lines.append(_format_cfg(v, indent + 1))
            else:
                lines.append(f"{sp}- {v}")
        return "\n".join(lines)

    return f"{sp}{obj}"


def plot_cfg_string(cfg, *, ax=None, title="Configuration", fontsize=10, line_height=1.35,
                    width=12, margin_top=0.04, margin_left=0.02):
    """
    Create an empty plot and draw the config string on it.

    If ax is None, creates a figure whose height is estimated from the
    number of text lines so the whole string fits.
    """
    txt = _format_cfg(cfg)
    lines = txt.splitlines()
    n_lines = max(1, len(lines) + 2)  # + title space

    if ax is None:
        # rough height estimate in inches
        height = max(4, 0.22 * n_lines * (fontsize / 10) * line_height)
        fig, ax = plt.subplots(figsize=(width, height))
    else:
        fig = ax.figure

    ax.clear()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    if title:
        ax.set_title(title, fontsize=fontsize + 2, pad=12)

    ax.text(
        margin_left,
        1 - margin_top,
        txt,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=fontsize,
        family="monospace",
        color="black",
        linespacing=line_height,
        clip_on=False,
    )

    fig.tight_layout()
    return fig, ax


def plot_policy_probs(
    logs,
    *,
    ax=None,                 # <-- NEW
    title="Policy probabilities per epoch",
    figsize=(9, 4),
    palette=None,
    annotate_final: bool = True,
    linewidth=2.0,
    alpha=0.9,
    marker=None,
):
    """
    Plot policy probabilities over epochs.

    If ax is None -> creates figure
    else -> plots into provided axis
    """

    epochs = []
    probs = []

    for i, log in enumerate(logs):
        p = log.get("train/policy_probs", None)
        if p is None:
            continue

        p = np.asarray(p, dtype=float)

        if p.ndim != 1:
            raise ValueError(f"Expected policy_probs to be 1D, got shape {p.shape} at log {i}")

        epochs.append(i)
        probs.append(p)

    if len(probs) == 0:
        raise ValueError("No entries found for log['train/policy_probs'].")

    #  import ipdb;ipdb.set_trace()
    probs = np.stack(probs, axis=0)  # (T, P)
    epochs = np.asarray(epochs)

    T, P = probs.shape

    if palette is None:
        palette = [f"C{i}" for i in range(P)]

    # ---------- axis handling ----------
    created_fig = False
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
        created_fig = True
    else:
        fig = ax.figure

    # ---------- plotting ----------
    for p_idx in range(P):
        ax.plot(
            epochs,
            probs[:, p_idx],
            label=f"policy {p_idx}",
            color=palette[p_idx],
            linewidth=linewidth,
            alpha=alpha,
            marker=marker,
        )

        if annotate_final:
            ax.text(
                epochs[-1],
                probs[-1, p_idx],
                f"{probs[-1, p_idx]:.3f}",
                color=palette[p_idx],
                fontsize=8,
                va="center",
                ha="left",
            )

    ax.set_title(title)
    ax.set_xlabel("epoch")
    ax.set_ylabel("policy probability")
    ax.grid(True, alpha=0.25)

    # avoid duplicate legends if reusing axis
    handles, labels = ax.get_legend_handles_labels()
    if len(handles) > 0:
        ax.legend(fontsize=8, frameon=True)

    ymin = min(0.0, float(probs.min()))
    ymax = max(1.0, float(probs.max()))
    ax.set_ylim(ymin, ymax * 1.05)

    if created_fig:
        fig.tight_layout()

    return fig, ax
