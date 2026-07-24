"""
Stage 5 headline figure: out-of-sample R^2 vs. prediction horizon, one line per
feature set, with block-bootstrap 95% CI bands. This is "the money plot" for the
paper -- the R^2 decay curve is the single figure that most directly tests the
hypothesis (predictive power exists at short horizons and decays toward zero as
the horizon grows).

Consumes reports/results.csv exactly as written by scripts/run_ofi_study.py, so
it's a separate script rather than folded into that one: re-running the study is
expensive (walks the full dataset through every fold), re-plotting its output is
not, and keeping them separate means a plot styling tweak never requires
re-running the actual analysis.

Run via: python -m scripts.plot_results --results reports/results.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # no display available when run headless / from a script
import matplotlib.pyplot as plt
import pandas as pd


def plot_r2_decay(results: pd.DataFrame, out_path: Path) -> None:
    """One line per `feature`, x=horizon_s, y=oos_r2_pct, shaded band=bootstrap
    CI. A horizontal zero line makes it immediate to see where predictability
    (by this metric, against the stated training-mean benchmark -- see
    modeling.py) crosses into "no better than the naive baseline."
    """
    required = {
        "feature",
        "horizon_s",
        "oos_r2_pct",
        "oos_r2_ci_low_pct",
        "oos_r2_ci_high_pct",
    }
    missing = required - set(results.columns)
    if missing:
        raise ValueError(f"results is missing required column(s): {missing}")

    fig, ax = plt.subplots(figsize=(7, 4.5))
    for feature, group in results.groupby("feature"):
        group = group.sort_values("horizon_s")
        ax.plot(group["horizon_s"], group["oos_r2_pct"], marker="o", label=feature)
        ax.fill_between(
            group["horizon_s"],
            group["oos_r2_ci_low_pct"],
            group["oos_r2_ci_high_pct"],
            alpha=0.2,
        )

    ax.axhline(0.0, color="black", linewidth=0.8, linestyle="--")
    ax.set_xlabel("prediction horizon (s)")
    ax.set_ylabel("out-of-sample R² (%, vs. training-mean baseline)")
    ax.set_title("OFI predictive power vs. horizon")
    ax.legend()
    fig.tight_layout()

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot the R^2-vs-horizon decay curve from a run_ofi_study.py results CSV."
    )
    parser.add_argument("--results", type=Path, default=Path("reports/results.csv"))
    parser.add_argument("--out", type=Path, default=Path("paper/figures/r2_decay.png"))
    args = parser.parse_args()

    results = pd.read_csv(args.results)
    plot_r2_decay(results, args.out)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
