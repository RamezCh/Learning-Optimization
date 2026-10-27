"""
plot_style.py
-------------
Shared matplotlib settings for all project figures.

`text.parse_math = False` makes "$" render literally, so labels like "$40-$70" are not
turned into italic math ("40-70" with the dollar signs swallowed).
"""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

COLORS = {
    "ours": "#059669",
    "rival": "#64748b",
    "accent": "#2563eb",
    "warn": "#dc2626",
    "amber": "#d97706",
    "dark": "#1e293b",
    "grid": "#e2e8f0",
}


def apply_style() -> None:
    plt.rcParams.update({
        "text.parse_math": False,
        "figure.facecolor": "#ffffff",
        "axes.facecolor": "#ffffff",
        "axes.grid": True,
        "grid.color": COLORS["grid"],
        "grid.linestyle": "--",
        "axes.titleweight": "bold",
        "axes.titlesize": 12,
        "axes.labelweight": "bold",
        "legend.frameon": True,
        "legend.fontsize": 9,
        "savefig.dpi": 200,
        "savefig.bbox": "tight",
    })
