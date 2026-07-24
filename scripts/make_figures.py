"""
Generates paper/figures/r2_vs_horizon.png from reports/results.csv: the
headline figure showing out-of-sample R^2 (with 95% block-bootstrap CI) decay
across prediction horizons, for both OFI feature variants.

Kept separate from run_ofi_study.py so figures can be regenerated from an
existing results.csv without re-running the (much slower) walk-forward study.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

# Categorical slots 1 (blue) and 2 (aqua) from the project's validated
# palette -- fixed assignment, not cycled, so "ofi_best" is always blue and
# "ofi_multilevel" is always aqua across every figure in the paper.
COLORS = {"ofi_best": "#2a78d6", "ofi_multilevel": "#1baf7a"}
LABELS = {"ofi_best": "Best-level OFI", "ofi_multilevel": "Depth-weighted 5-level OFI"}
INK = "#0b0b0b"
SECONDARY_INK = "#52514e"
MUTED = "#898781"
GRIDLINE = "#e1e0d9"
SURFACE = "#fcfcfb"


def plot_r2_vs_horizon(results: pd.DataFrame, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 4.5), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)

    for feature in ["ofi_best", "ofi_multilevel"]:
        sub = results[results["feature"] == feature].sort_values("horizon_s")
        color = COLORS[feature]
        ax.plot(
            sub["horizon_s"], sub["oos_r2_pct"],
            color=color, linewidth=2, marker="o", markersize=8,
            markerfacecolor=color, markeredgecolor=SURFACE, markeredgewidth=2,
            zorder=3,
        )
        ax.fill_between(
            sub["horizon_s"], sub["oos_r2_ci_low_pct"], sub["oos_r2_ci_high_pct"],
            color=color, alpha=0.10, zorder=1,
        )
        # Direct label at the line's right (longest-horizon) end.
        last = sub.iloc[-1]
        ax.annotate(
            LABELS[feature],
            xy=(last["horizon_s"], last["oos_r2_pct"]),
            xytext=(8, 0), textcoords="offset points",
            va="center", ha="left", color=INK, fontsize=10,
        )

    ax.axhline(0, color=MUTED, linewidth=1, zorder=0)
    ax.set_xticks(sorted(results["horizon_s"].unique()))
    ax.set_xticklabels([f"{int(h)}s" for h in sorted(results["horizon_s"].unique())])
    ax.set_xlabel("Prediction horizon", color=SECONDARY_INK, fontsize=10)
    ax.set_ylabel("Out-of-sample R² (%)", color=SECONDARY_INK, fontsize=10)
    ax.set_title(
        "Out-of-sample R² by horizon (95% block-bootstrap CI)",
        color=INK, fontsize=12, loc="left", pad=12,
    )
    ax.grid(axis="y", color=GRIDLINE, linewidth=1)
    ax.set_axisbelow(True)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    for spine in ["left", "bottom"]:
        ax.spines[spine].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelsize=9)
    # Give the right-edge direct labels room so they don't clip the axes.
    ax.set_xlim(right=results["horizon_s"].max() * 1.6)

    fig.tight_layout()
    fig.savefig(out_path, dpi=200, facecolor=SURFACE)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate the paper's figures.")
    parser.add_argument("--results", type=Path, default=Path("reports/results.csv"))
    parser.add_argument("--out-dir", type=Path, default=Path("paper/figures"))
    args = parser.parse_args()

    results = pd.read_csv(args.results)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    out_path = args.out_dir / "r2_vs_horizon.png"
    plot_r2_vs_horizon(results, out_path)
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
