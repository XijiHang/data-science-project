"""Create readable SVG figures from the calculated benchmark results."""
from pathlib import Path
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE = Path(__file__).resolve().parent
COLORS = ["#747c83", "#bd4b30", "#17746b"]
LABELS = ["Persistence", "Boosting · MAE loss", "Boosting · MSE loss"]


def make_figures():
    results = json.loads((BASE / "metrics.json").read_text(encoding="utf-8"))
    models = list(results["models"])
    folder = BASE
    folder.mkdir(exist_ok=True)
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 13,
        "text.color": "#192a2a", "axes.labelcolor": "#192a2a",
        "xtick.color": "#64716f", "ytick.color": "#192a2a",
        "svg.fonttype": "none", "svg.hashsalt": "pm25-portfolio",
    })
    specs = [
        ("error-comparison", "Lower average error, larger errors on high days",
         [("mae_all", "All test station-days", "Mean absolute error (µg/m³)", 140),
          ("mae_high", "High-concentration station-days", "Mean absolute error (µg/m³)", 140)],
         "Lower is better. Both panels use the same scale."),
        ("alert-tradeoff", "More precise alerts can still miss more high days",
         [("recall_high_pct", "Recall · high cases caught", "Recall (%)", 100),
          ("precision_high_pct", "Precision · alerts that are correct", "Precision (%)", 100)],
         "Higher is better. Recall and precision answer different questions."),
    ]
    for filename, title, panels, note in specs:
        fig, axes = plt.subplots(1, 2, figsize=(13.4, 5.1), sharey=True)
        fig.patch.set_facecolor("#ffffff")
        fig.subplots_adjust(left=.20, right=.96, top=.70, bottom=.25, wspace=.20)
        fig.text(.04, .92, title, fontsize=19, weight="bold")
        fig.text(.04, .85, note, fontsize=12, color="#64716f")
        for ax, (key, subtitle, unit, maximum) in zip(axes, panels):
            ax.axhspan(.64, 1.36, color="#faf0eb", zorder=0)
            for y, name, color in zip([2, 1, 0], models, COLORS):
                row = results["models"][name]
                value = row[key]
                lo, hi = row[key + "_ci"]
                ax.errorbar(value, y, xerr=[[value-lo], [hi-value]], fmt="o",
                            color=color, markersize=8, capsize=4, elinewidth=1.7, zorder=3)
                ax.annotate(f"{value:.1f}", (value, y), xytext=(0, 12),
                            textcoords="offset points", ha="center", fontsize=12, weight="bold", color=color)
            ax.set(xlim=(0, maximum), ylim=(-.5, 2.55), xlabel=unit,
                   yticks=[2, 1, 0], yticklabels=LABELS)
            ax.set_xticks([0, 40, 80, 120] if maximum == 140 else [0, 25, 50, 75, 100])
            ax.set_title(subtitle, loc="left", fontsize=12, pad=15, weight="bold")
            ax.grid(axis="x", color="#e4e9e6", linewidth=.8)
            ax.set_axisbelow(True)
            ax.spines[["top", "right", "left"]].set_visible(False)
            ax.spines["bottom"].set_color("#bac6c0")
            ax.tick_params(axis="y", length=0, pad=15)
        fig.text(.04, .105, "4,920 test station-days · 459 high cases · Cutoff: training P90 = 168.4 µg/m³", fontsize=11)
        fig.text(.04, .05, "Whiskers: 95% paired seven-day block-bootstrap intervals, conditional on fitted models. Source: UCI / Chen (2017).", fontsize=10, color="#64716f")
        fig.savefig(folder / f"{filename}.svg", metadata={"Date": None})
        plt.close(fig)
    print("Saved error-comparison.svg and alert-tradeoff.svg", flush=True)


if __name__ == "__main__":
    make_figures()
