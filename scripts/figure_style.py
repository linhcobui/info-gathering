"""House figure style (docs/FIGURES.md): the ten colours, Palatino, quiet axes.

    from figure_style import apply, HIGHLIGHT, INTERESTING, OBVIOUS, INK, MUTED, FAINT
    apply()
"""
import matplotlib.pyplot as plt
from matplotlib import font_manager

HIGHLIGHT = "#ff9500"                                                          # our method / the point of the figure
INTERESTING = ["#0048a6", "#0091ff", "#740cad", "#0d948f", "#850f67", "#cdb3ff"]  # baselines worth attention, in order
OBVIOUS = ["#000000", "#6b6b6b", "#b3b3b3"]                                    # expected baselines / references
INK, MUTED, FAINT = "#000000", "#6b6b6b", "#b3b3b3"                            # text, secondary text, unimportant
FONT = ["Palatino Linotype", "P052", "TeX Gyre Pagella", "DejaVu Sans"]       # DejaVu only for missing glyphs


def apply(font_size=9):
    installed = {f.name for f in font_manager.fontManager.ttflist}
    plt.rcParams.update({
        "font.family": [f for f in FONT if f in installed],     # missing names make matplotlib warn on every text"font.size": font_size, "mathtext.fontset": "custom", "mathtext.rm": "P052",
        "mathtext.it": "P052:italic", "mathtext.bf": "P052:bold",
        "text.color": INK, "axes.labelcolor": INK, "axes.titlecolor": INK, "axes.edgecolor": MUTED,
        "xtick.color": MUTED, "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": False, "grid.color": FAINT, "grid.linewidth": 0.5, "grid.alpha": 0.6,
        "axes.prop_cycle": plt.cycler(color=[HIGHLIGHT] + INTERESTING),
        "legend.frameon": False, "pdf.fonttype": 42, "ps.fonttype": 42, "savefig.dpi": 150,
    })
