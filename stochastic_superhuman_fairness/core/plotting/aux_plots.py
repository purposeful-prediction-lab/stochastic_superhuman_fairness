import numpy as np
import matplotlib.pyplot as plt
import torch
from stochastic_superhuman_fairness.core.plotting.plot_utils import  _add_row_group_colors_to_heatmap

def plot_subdominance_heatmap(
    S,
    *,
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
    fig, ax = plt.subplots(1, 1, figsize=figsize)
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

    fig, axes = plt.subplots(1, 3, figsize=figsize)

    # 1) gamma
    im0 = axes[0].imshow(Gp, aspect="auto")
    _add_row_group_colors_to_heatmap(
        axes[0],
        R,
        row_groups=row_groups,
        group_colors=group_colors,
        strip_width=group_strip_width,
    )
    axes[0].set_title(r"$\gamma$")
    axes[0].set_xlabel("demos")
    axes[0].set_ylabel("rollouts")
    if show_colorbar: fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)

    if mode == "replicate":
        U = np.repeat(u[:, None], D, axis=1)   # (R,D)
        V = np.repeat(v[None, :], R, axis=0)   # (R,D)

        im1 = axes[1].imshow(U, aspect="auto")
        _add_row_group_colors_to_heatmap(
            axes[1], R,
            row_groups=row_groups,
            group_colors=group_colors,
            strip_width=group_strip_width,
        ) 
        axes[1].set_title("dual_rows (replicated)")
        axes[1].set_xlabel("demos")
        axes[1].set_ylabel("rollouts")
        if show_colorbar: fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

        im2 = axes[2].imshow(V, aspect="auto")
        _add_row_group_colors_to_heatmap(
            axes[2], R,
            row_groups=row_groups,
            group_colors=group_colors,
            strip_width=group_strip_width,
        )
        axes[2].set_title("dual_cols (replicated)")
        axes[2].set_xlabel("demos")
        axes[2].set_ylabel("rollouts")
        if show_colorbar: fig.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)

    else:  # mode == "sum"
        UV = u[:, None] + v[None, :]           # (R,D)
        im1 = axes[1].imshow(UV, aspect="auto")
        _add_row_group_colors_to_heatmap(
            axes[1], R,
            row_groups=row_groups,
            group_colors=group_colors,
            strip_width=group_strip_width,
        )
        axes[1].set_title("dual potential (u + v)")
        axes[1].set_xlabel("demos")
        axes[1].set_ylabel("rollouts")
        if show_colorbar: fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

        # keep 3 panels: show also separate sign structure via centered version
        UVc = UV - UV.mean()
        im2 = axes[2].imshow(UVc, aspect="auto")
        _add_row_group_colors_to_heatmap(
            axes[2], R,
            row_groups=row_groups,
            group_colors=group_colors,
            strip_width=group_strip_width,
        )
        axes[2].set_title("(u + v) centered")
        axes[2].set_xlabel("demos")
        axes[2].set_ylabel("rollouts")
        if show_colorbar: fig.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)

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

    fig, ax = plt.subplots(figsize=figsize)
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
