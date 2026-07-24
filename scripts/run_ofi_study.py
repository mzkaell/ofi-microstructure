"""
End-to-end Stage 2-4 run: builds the OFI feature table from reconstructed
book states, then for each (feature set, horizon) pair runs purged/embargoed
walk-forward out-of-sample R^2, a block-bootstrap CI on it, and a full-sample
Newey-West significance test -- and prints one summary table.

This is a research-runner script, not library code (hence living in scripts/
rather than src/): it wires together src/features.py, src/modeling.py, and
src/evaluation.py with one particular, documented choice of window size,
horizon list, fold count, and bootstrap/HAC parameters. Re-run with different
CLI args to explore other choices; the underlying modules are what's tested.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.evaluation import block_bootstrap_r2_ci, newey_west_significance, run_walk_forward
from src.features import build_feature_table


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the full OFI predictability study.")
    parser.add_argument("--processed", type=Path, default=Path("data/processed/book"))
    parser.add_argument(
        "--raw",
        type=Path,
        default=Path("data/raw"),
        help="raw capture dir, used only to build the trade-imbalance baseline",
    )
    parser.add_argument("--out", type=Path, default=Path("reports/results.csv"))
    parser.add_argument("--window-seconds", type=float, default=1.0)
    parser.add_argument(
        "--horizons-seconds", type=float, nargs="+", default=[1.0, 2.0, 5.0, 10.0]
    )
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--min-train-fraction", type=float, default=0.5)
    parser.add_argument("--n-boot", type=int, default=500)
    args = parser.parse_args()

    print(f"Building feature table (window={args.window_seconds}s) ...")
    df = build_feature_table(
        args.processed, args.window_seconds, args.horizons_seconds, raw_dir=args.raw
    )
    print(f"{len(df)} windows, {df.index[0]}..{df.index[-1]}")

    # Each entry maps a feature label to a function of horizon -> feature
    # column(s). ofi_best/ofi_multilevel/trade_imbalance are the same column
    # at every horizon; the AR(1) baseline is horizon-specific by
    # construction (trail_ret_{h}s is only a fair comparison against
    # fwd_ret_{h}s of the *same* h), hence the callable instead of a fixed list.
    feature_specs = [
        ("ofi_best", lambda h: ["ofi_best"]),
        ("ofi_multilevel", lambda h: ["ofi_multilevel"]),
        ("trade_imbalance", lambda h: ["trade_imbalance"]),
        ("ar1", lambda h: [f"trail_ret_{h:g}s"]),
    ]

    rows = []
    for feature_label, feature_cols_fn in feature_specs:
        for h in args.horizons_seconds:
            feature_cols = feature_cols_fn(h)
            if feature_cols[0] not in df.columns:
                print(
                    f"  skipping {feature_label:>15s} h={h:>4g}s: column "
                    f"'{feature_cols[0]}' not in feature table"
                )
                continue
            target_col = f"fwd_ret_{h:g}s"
            bins_per_horizon = int(round(h / args.window_seconds))

            # purge = exactly the trailing training rows whose h-bin-ahead
            # label reaches into the test set (bins_per_horizon - 1 of them);
            # embargo = an equal-sized extra buffer against residual serial
            # correlation beyond pure label overlap. See
            # evaluation.walk_forward_splits's docstring for the mechanics.
            oos = run_walk_forward(
                df,
                feature_cols,
                target_col,
                args.n_splits,
                args.min_train_fraction,
                purge=bins_per_horizon - 1,
                embargo=bins_per_horizon,
            )
            r2 = block_bootstrap_r2_ci(
                oos["y_true"],
                oos["y_pred_model"],
                oos["y_pred_baseline"],
                block_size=max(20, 5 * bins_per_horizon),
                n_boot=args.n_boot,
            )
            nw = newey_west_significance(
                df[feature_cols],
                df[target_col],
                feature_cols,
                maxlags=max(2 * bins_per_horizon, 1),
            )
            coef_col = feature_cols[0]
            rows.append(
                {
                    "feature": feature_label,
                    "horizon_s": h,
                    "n_oos": len(oos.dropna()),
                    "oos_r2_pct": 100 * r2.point_estimate,
                    "oos_r2_ci_low_pct": 100 * r2.ci_low,
                    "oos_r2_ci_high_pct": 100 * r2.ci_high,
                    "nw_coef": nw.params[coef_col],
                    "nw_tstat": nw.tvalues[coef_col],
                    "nw_pvalue": nw.pvalues[coef_col],
                }
            )
            print(
                f"  {feature_label:>15s} h={h:>4g}s  "
                f"OOS R^2={100*r2.point_estimate:+.4f}% "
                f"[{100*r2.ci_low:+.4f}%, {100*r2.ci_high:+.4f}%]  "
                f"NW t={nw.tvalues[coef_col]:+.2f} p={nw.pvalues[coef_col]:.4g}"
            )
            if r2.point_estimate > 0.10:  # 10% OOS R^2 on sub-10s returns is not credible
                print(
                    f"    *** WARNING: {100*r2.point_estimate:.2f}% OOS R^2 is "
                    "suspiciously high for short-horizon return prediction -- "
                    "check for leakage before trusting this number. ***"
                )

    results = pd.DataFrame(rows)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(args.out, index=False)
    print(f"\nwrote {args.out}")
    print(results.to_string(index=False))


if __name__ == "__main__":
    main()
