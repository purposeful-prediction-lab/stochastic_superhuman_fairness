import colorsys
import numpy as np

BASELINE_PALETTE = [
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
MODE_PALETTE_100_MAXVAR = [
    "#ff0000", "#00ff00", "#ff00ff", "#ffff00",
    "#00ff7f", "#ff7f00", "#7f00ff", "#ff007f",
    "#7fff00", "#ffbf00", "#bf00ff", "#ff0055",
    "#55ff00", "#ff9900", "#9900ff", "#ff0033",
    "#33ff00", "#ff6600", "#6600ff", "#ff0011",
    "#11ff00", "#ff4400", "#4400ff", "#ff2200",
    "#22ff00", "#ff8800", "#8800ff", "#ff0066",
    "#66ff00", "#ffaa00", "#aa00ff", "#ff0044",
    "#44ff00", "#ffcc00", "#cc00ff", "#ff0022",
    "#22ff66", "#ff7700", "#7700ff", "#ff0099",
    "#99ff00", "#ffbb00", "#bb00ff", "#ff0055",
    "#55ff33", "#ff5500", "#5500ff", "#ff00bb",
    "#bbff00", "#ffdd00", "#dd00ff", "#ff0077",
    "#77ff00", "#ff9900", "#9900dd", "#ff0044",
    "#44ff55", "#ff6600", "#6600dd", "#ff0022",
    "#22ff88", "#ff8800", "#8800dd", "#ff0099",
    "#99ff33", "#ffaa00", "#aa00dd", "#ff0055",
    "#55ff66", "#ff7700", "#7700dd", "#ff0033",
    "#33ff99", "#ffbb00", "#bb00dd", "#ff0077",
    "#77ff44", "#ffcc00", "#cc00dd", "#ff0044",
    "#44ffaa", "#ff5500", "#5500dd", "#ff0022",
    "#22ffbb", "#ff9900", "#9900cc", "#ff0066",
    "#66ff55", "#ffaa00", "#aa00cc", "#ff0033",
    "#33ffcc", "#ff7700", "#7700cc", "#ff0099",
    "#99ff66", "#ffbb00", "#bb00cc", "#ff0055",
]
MODE_PALETTE_100_PAPERSAFE = [
    "#1b9e77", "#66a61e", "#4d9221", "#a6761d",
    "#8c564b", "#e7298a", "#d95f02", "#b2182b",
    "#7f0000", "#6a3d9a", "#b15928", "#fb9a99",
    "#cab2d6", "#fdbf6f", "#ff7f00", "#e31a1c",
    "#a6d854", "#ffd92f", "#c51b7d", "#7a0177",
    "#8c510a", "#bf812d", "#dfc27d", "#a6611a",
    "#80cdc1", "#018571", "#c2a5cf", "#762a83",
    "#e08214", "#f46d43", "#d53e4f", "#9e0142",
    "#f1b6da", "#de77ae", "#c994c7", "#dd1c77",
    "#980043", "#c7eae5", "#5ab4ac", "#01665e",
    "#ccebc5", "#7fc97f", "#4daf4a", "#2ca25f",
    "#238b45", "#005824", "#fe9929", "#ec7014",
    "#cc4c02", "#993404", "#fee391", "#fec44f",
    "#fe9929", "#d95f0e", "#993404", "#8c2d04",
    "#ffffcc", "#ffeda0", "#fed976", "#feb24c",
    "#fd8d3c", "#f03b20", "#bd0026", "#800026",
    "#e5f5e0", "#c7e9c0", "#a1d99b", "#74c476",
    "#41ab5d", "#238b45", "#006d2c", "#00441b",
    "#fff5f0", "#fee0d2", "#fcbba1", "#fc9272",
    "#fb6a4a", "#ef3b2c", "#cb181d", "#99000d",
    "#fff7ec", "#fee8c8", "#fdd49e", "#fdbb84",
    "#fc8d59", "#ef6548", "#d7301f", "#990000",
    "#f7f4f9", "#e7e1ef", "#d4b9da", "#c994c7",
    "#df65b0", "#e7298a", "#ce1256", "#91003f",
    "#f7fcf5", "#e5f5e0", "#c7e9c0", "#a1d99b",
]
BASIC_PALETTE = [
        "#1b9e77",
        "#d95f02", "#7570b3", "#e7298a", "#66a61e",
        "#a6761d", "#666666", "#b2182b", "#6a3d9a", "#4d9221"
    ]
#================================================================


def generate_paper_safe_mode_palette(
    n=100,
    include_blue=True,
):
    """
    Generate up to n distinct publication-friendly colors.

    The first ~20 colors are hand-picked to maximize distinguishability.
    Remaining colors are generated automatically.

    Parameters
    ----------
    n : int
        Number of colors.
    include_blue : bool
        If False, removes blue/cyan colors from both the curated and
        generated portions.
    """

    # -------- curated colors --------
    palette = [
        "#1f77b4",  # blue
        "#ff7f0e",  # orange
        "#2ca02c",  # green
        "#d62728",  # red
        "#9467bd",  # purple
        "#8c564b",  # brown
        "#e377c2",  # pink
        "#bcbd22",  # olive
        "#17becf",  # cyan
        "#7f7f7f",  # gray

        "#aec7e8",  # light blue
        "#ffbb78",  # light orange
        "#98df8a",  # light green
        "#ff9896",  # light red
        "#c5b0d5",  # light purple
        "#c49c94",  # light brown
        "#f7b6d2",  # light pink
        "#dbdb8d",  # light olive
        "#9edae5",  # light cyan
        "#c7c7c7",  # light gray
    ]

    if not include_blue:
        palette = [
            c for i, c in enumerate(palette)
            if i not in (0, 8, 10, 18)   # blue/cyan entries
        ]

    if len(palette) >= n:
        return palette[:n]

    # -------- generate the remainder --------

    hues = np.linspace(0, 1, 5 * n)

    for h in hues:

        if not include_blue:
            # skip cyan-blue range
            if 0.48 <= h <= 0.72:
                continue

        s = 0.60
        v = 0.82

        r, g, b = colorsys.hsv_to_rgb(h, s, v)
        color = "#{:02x}{:02x}{:02x}".format(
            int(255 * r),
            int(255 * g),
            int(255 * b),
        )

        if color not in palette:
            palette.append(color)

        if len(palette) >= n:
            break

    return palette[:n]

def cycle_palette_colors(n, palette: list)-> list:
    if n <= len(palette):
        return palette[:n]
    return [palette[i % len(palette)] for i in range(n)]


