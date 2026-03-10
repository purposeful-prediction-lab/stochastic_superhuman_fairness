from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt
import math
import mplcursors
from typing import Dict, Any, Optional, Tuple
import matplotlib.axes
import matplotlib.patches as patches
from stochastic_superhuman_fairness.core.plotting.plotting_palettes import BASIC_PALETTE, cycle_palette_colors

def add_hover_tooltips(artists, *, rollout_labels=None, demo_labels=None):
    """
    Adds interactive hover tooltips to scatter artists produced by plot_rollouts_vs_demos.

    Args:
      artists: list of dicts like:
          {"pair": (i,j), "rollouts": PathCollection, "demos": PathCollection}
      rollout_labels: list[str] length R (optional). If None, uses index.
      demo_labels: list[str] length D (optional). If None, uses index.

    Returns:
      A list of mplcursors instances (keep them alive).
    """

    cursors = []

    def _attach(scatter, labels, kind, pair):
        cursor = mplcursors.cursor(scatter, hover=True)

        @cursor.connect("add")
        def _on_add(sel):
            idx = int(sel.index)
            label = labels[idx] if labels is not None and idx < len(labels) else f"{kind}[{idx}]"
            x, y = sel.target
            sel.annotation.set_text(f"{label}\n(pair={pair})\nx={x:.4g}, y={y:.4g}")

        cursors.append(cursor)

    for item in artists:
        pair = item.get("pair", None)
        _attach(item["rollouts"], rollout_labels, "rollout", pair)
        _attach(item["demos"], demo_labels, "demo", pair)

    return cursors

# -------------------------
# 1) Compute means (separate helper)
# -------------------------
def compute_feature_means(feats: np.ndarray) -> np.ndarray:
    """
    feats: [N,K] array of rollout features
    returns: mean per feature [K]
    """
    feats = np.asarray(feats, dtype=float)
    if feats.ndim != 2:
        raise ValueError(f"feats must be [N,K], got {feats.shape}")
    return np.nanmean(feats, axis=0)


# -------------------------
# 2) Mean point + solid guide lines
# -------------------------
def annotate_mean_with_guides(
    ax: "matplotlib.axes.Axes",
    mean_x: float,
    mean_y: float,
    *,
    s: float = 70,                 # bold dot size
    marker: str = "o",
    lw: float = 1.6,
    guide_lw: float | None = None, # optionally separate linewidth
    zorder: int = 5,
):
    """
    Plot a bold mean point at (mean_x, mean_y) and draw solid guide lines
    from the mean to the *right* (xmax) and *up* (ymax), using current bounds.
    """
    if guide_lw is None:
        guide_lw = lw

    # Use current limits as the "baseline"
    xmin, xmax = ax.get_xlim()
    ymin, ymax = ax.get_ylim()

    # Bold mean point
    mean_artist = ax.scatter(
        [mean_x], [mean_y],
        s=s, marker=marker,
        zorder=zorder,
    )

    # Solid guides: right to xmax, up to ymax
    h_guide = ax.plot(
        [mean_x, xmax], [mean_y, mean_y],
        linestyle="-", linewidth=guide_lw,
        zorder=zorder - 1,
        color = 'red',
    )[0]
    v_guide = ax.plot(
        [mean_x, mean_x], [mean_y, ymax],
        linestyle="-", linewidth=guide_lw,
        zorder=zorder - 1,
        color = 'red',
    )[0]

    return mean_artist, v_guide, h_guide
# -------------------------
# 3) Dotted double arrows from guide lines -> 1/alpha locations + annotation
# -------------------------
def annotate_inverse_alpha_arrows(
    ax: "matplotlib.axes.Axes",
    *,
    mean_x: float,
    mean_y: float,
    inv_alpha_x: float,
    inv_alpha_y: float,
    x_name: str,
    y_name: str,
    pad_frac: float = 0.06,
    offset_frac: float = 0.03,
    lw: float = 1.4,
    arrow_ls: str = "-",
    text_fs: int = 9,
    zorder: int = 6,
) -> Dict[str, Any]:
    """
    Draw inverse-alpha origin at (inv_alpha_x, inv_alpha_y), with solid rays
    extending right/up. Draw offset dotted <-> arrows from mean to inv-alpha
    and label them as 1/alpha_{feature}.
    """

    # ---------- Expand limits to include mean + inv-alpha ----------
    xmin, xmax = ax.get_xlim()
    ymin, ymax = ax.get_ylim()

    x_vals = np.array([xmin, xmax, mean_x, inv_alpha_x])
    y_vals = np.array([ymin, ymax, mean_y, inv_alpha_y])

    x_lo, x_hi = np.nanmin(x_vals), np.nanmax(x_vals)
    y_lo, y_hi = np.nanmin(y_vals), np.nanmax(y_vals)

    x_span = max(1e-12, x_hi - x_lo)
    y_span = max(1e-12, y_hi - y_lo)

    ax.set_xlim(x_lo - pad_frac * x_span, x_hi + pad_frac * x_span)
    ax.set_ylim(y_lo - pad_frac * y_span, y_hi + pad_frac * y_span)

    xmin, xmax = ax.get_xlim()
    ymin, ymax = ax.get_ylim()

    # ---------- Alpha rays (origin = inv-alpha point) ----------
    alpha_h = ax.plot(
        [inv_alpha_x, xmax], [inv_alpha_y, inv_alpha_y],
        lw=lw, zorder=zorder, color = 'red',
        linestyle = 'dashed',
    )[0]
    alpha_v = ax.plot(
        [inv_alpha_x, inv_alpha_x], [inv_alpha_y, ymax],
        lw=lw, zorder=zorder, color = 'red',
        linestyle = 'dashed',
    )[0]

    # ---------- Offset dotted <-> arrows ----------
    dx = offset_frac * (xmax - xmin)
    dy = offset_frac * (ymax - ymin)

    arrow = dict(arrowstyle="<->", linestyle=arrow_ls, lw=lw)

    x_arrow = ax.annotate(
        "", xy=(inv_alpha_x, mean_y + dy), xytext=(mean_x, mean_y + dy),
        arrowprops=arrow, zorder=zorder + 1, clip_on=False
    )
    y_arrow = ax.annotate(
        "", xy=(mean_x + dx, inv_alpha_y), xytext=(mean_x + dx, mean_y),
        arrowprops=arrow, zorder=zorder + 1, clip_on=False
    )

    # ---------- Labels (up/right, margin-like) ----------
    x_label = ax.annotate(
        rf"$1/\alpha_{{{x_name}}}$",
        xy=((mean_x + inv_alpha_x) / 2, mean_y + dy),
        xytext=(6, 6), textcoords="offset points",
        ha="left", va="bottom",
        fontsize=text_fs, zorder=zorder + 2, clip_on=False
    )
    y_label = ax.annotate(
        rf"$1/\alpha_{{{y_name}}}$",
        xy=(mean_x + dx, (mean_y + inv_alpha_y) / 2),
        xytext=(6, 6), textcoords="offset points",
        ha="left", va="bottom",
        fontsize=text_fs, zorder=zorder + 2, clip_on=False
    )

    return {
        "alpha_h": alpha_h,
        "alpha_v": alpha_v,
        "x_arrow": x_arrow,
        "y_arrow": y_arrow,
        "x_label": x_label,
        "y_label": y_label,
    }
# -------------------------
# 4) Tiny adapter: get 1/alpha for selected dims
# -------------------------
def inv_alpha_for_dims(alpha: np.ndarray, ix: int, iy: int, eps=1e-12):
    a = np.asarray(alpha, dtype=float).reshape(-1)
    return (1.0 / max(a[ix], eps), 1.0 / max(a[iy], eps))


# -------------------------
# Example integration inside a scatter plot loop
# -------------------------
def add_mean_and_alpha_annotations(
    ax,
    rollout_feats_2d: np.ndarray,  # [N,K]
    ix: int,
    iy: int,
    feature_names: list[str],
    alpha: np.ndarray | None = None,
):
    means = compute_feature_means(rollout_feats_2d)  # [K]
    mean_x, mean_y = float(means[ix]), float(means[iy])

    annotate_mean_with_guides(ax, mean_x, mean_y)

    if alpha is not None:
        inv_x, inv_y = inv_alpha_for_dims(alpha, ix, iy)
        annotate_inverse_alpha_arrows(
            ax,
            mean_x=mean_x,
            mean_y=mean_y,
            inv_alpha_x=inv_x,
            inv_alpha_y=inv_y,
            x_name=feature_names[ix],
            y_name=feature_names[iy],
        )


def namespace_keywords_to_string(
    cfg,
    keywords,
    ignore_keywords=None,
    max_depth=None,
):
    """
    Extract selected keys from a nested NamespaceDict/dict and return
    an indented multiline string.

    Args
        cfg : NamespaceDict or dict
        keywords : list of keys to include
        ignore_keywords : list of keys whose subtrees should be skipped
        max_depth : maximum recursion depth (None = unlimited)
    """

    keywords = set(keywords)
    ignore_keywords = set(ignore_keywords or [])

    def is_mapping(x):
        return hasattr(x, "items") or isinstance(x, dict)

    def get_items(x):
        if isinstance(x, dict):
            return x.items()
        if hasattr(x, "items"):
            return x.items()
        return []

    def format_value(v):
        if isinstance(v, float):
            return f"{v:.4g}"
        return str(v)

    def recurse(obj, indent=0):
        if max_depth is not None and indent > max_depth:
            return []

        lines = []

        for k, v in get_items(obj):

            if k in ignore_keywords:
                continue

            if is_mapping(v):
                child_lines = recurse(v, indent + 1)

                if k in keywords or child_lines:
                    lines.append("   " * indent + f"{k}:")
                    lines.extend(child_lines)

            else:
                if k in keywords:
                    lines.append("   " * indent + f"{k}: {format_value(v)}")

        return lines

    return "\n".join(recurse(cfg))


def add_cfg_text_to_figure(
    fig,
    cfg,
    keywords,
    *,
    ignore_keywords=None,
    max_depth=None,
    x=0.01,
    y=0.5,
    fontsize=10,
    rotation=0,
    va="center",
    ha="left",
):
    """
    Add selected config text to the side of a matplotlib figure.
    """

    text = namespace_keywords_to_string(
        cfg,
        keywords,
        ignore_keywords=ignore_keywords,
        max_depth=max_depth,
    )

    fig.text(
        x,
        y,
        text,
        fontsize=fontsize,
        va=va,
        ha=ha,
        family="monospace",
        rotation=rotation,
    )

    return text

def _normalize_row_groups(row_groups, R):
    """
    row_groups: list of lists/arrays of row indices, e.g. [[1,2,3],[4,5]]
    Returns:
        row_to_group: length-R array, -1 for unassigned rows
        n_groups: int
    """
    if row_groups is None:
        return None, 0

    row_to_group = np.full(R, -1, dtype=int)

    for g, rows in enumerate(row_groups):
        rows = np.asarray(rows, dtype=int).reshape(-1)
        for r in rows:
            if r < 0 or r >= R:
                raise ValueError(f"Row index {r} out of bounds for R={R}.")
            if row_to_group[r] != -1:
                raise ValueError(f"Row {r} appears in more than one group.")
            row_to_group[r] = g

    return row_to_group, len(row_groups)


def _add_row_group_colors_to_heatmap(
    ax,
    R,
    row_groups = None,
    group_colors: list = None,
    strip_width: float = 0.18,
):
    """
    Draw a colored strip to the left of the heatmap, one color per row group.
    Assumes imshow row centers are at y = 0,1,...,R-1 and cell height = 1.
    """
    row_to_group, n_groups = _normalize_row_groups(row_groups, R)
    if row_to_group is None:
        return
    colors = cycle_palette_colors(n_groups, BASIC_PALETTE) if group_colors is None else group_colors
    if len(colors) < n_groups:
        raise ValueError(f"Need at least {n_groups} group_colors, got {len(colors)}.")

    # Put strip just left of the heatmap
    x0 = -0.5 - strip_width

    for r in range(R):
        g = row_to_group[r]
        if g == -1:
            continue
        rect = patches.Rectangle(
            (x0, r - 0.5),
            strip_width,
            1.0,
            facecolor=colors[g],
            edgecolor="none",
            clip_on=False,
            zorder=5,
        )
        ax.add_patch(rect)
