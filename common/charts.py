"""Shared matplotlib style for the demo charts."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

JEV = "#6C4DF6"     # JEV purple
BASE = "#9AA5B1"    # baselines grey
ALT = "#12B886"     # secondary green
WARN = "#F76707"    # orange


def style():
    plt.rcParams.update({"figure.dpi": 150, "savefig.dpi": 150, "font.size": 10, "axes.spines.top": False,
                         "axes.spines.right": False, "axes.grid": True, "grid.alpha": 0.25, "axes.titleweight": "bold",
                         "axes.titlesize": 12, "legend.frameon": False})
    return plt
