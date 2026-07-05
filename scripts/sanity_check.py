"""
Stage 1 data-quality sanity checks: run this after (or during, on partial data)
the live capture to verify the reconstructed book behaves like a real order book,
and to get a feel for how much usable data we actually have.

Run via: python -m scripts.sanity_check --raw data/raw --processed data/processed/book

Deliberately a plain script, not a notebook: it needs to run unattended against
however many days have accumulated so far, and its output (a JSON summary + a
couple of PNGs under reports/sanity/) is meant to be regenerated repeatedly as
more data comes in, not hand-edited.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # no display available when run headless / from a script
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def load_processed(processed_dir: Path) -> pd.DataFrame:
    files = sorted(Path(processed_dir).glob("*.parquet"))
    if not files:
        raise FileNotFoundError(
            f"no processed book-state files under {processed_dir} -- run "
            "`python -m src.orderbook --raw <raw> --out <processed>` first"
        )
    return pd.concat(pd.read_parquet(f) for f in files).sort_index()


def check_spread_positivity(df: pd.DataFrame) -> dict:
    """Ask > bid should ALWAYS hold. Any violation means a reconstruction bug,
    not a market phenomenon -- a crossed/locked top-of-book should never come
    out of a correctly maintained L2 book replay."""
    violations = df[df["spread"] <= 0]
    return {
        "n_rows": len(df),
        "n_violations": len(violations),
        "violation_timestamps": [str(t) for t in violations.index[:10]],
    }


def spread_stats_bps(df: pd.DataFrame) -> dict:
    spread_bps = (df["spread"] / df["mid_price"]) * 1e4
    return {
        "mean_bps": float(spread_bps.mean()),
        "median_bps": float(spread_bps.median()),
        "p95_bps": float(spread_bps.quantile(0.95)),
        "max_bps": float(spread_bps.max()),
    }


def message_rate_and_resync_stats(df: pd.DataFrame) -> dict:
    span_s = (df.index[-1] - df.index[0]).total_seconds()
    n_resync = int(df["resync"].sum())
    return {
        "span_seconds": span_s,
        "n_events": len(df),
        "events_per_second": len(df) / span_s if span_s > 0 else float("nan"),
        "n_resync_points": n_resync,
        "resync_rate_per_hour": (
            n_resync / (span_s / 3600) if span_s > 0 else float("nan")
        ),
    }


def collector_log_summary(raw_dir: Path) -> dict:
    """Independent cross-check using acquisition.py's own live log, separate
    from what reconstruction derives from the raw events -- if the two
    disagree substantially on reconnect/gap counts, that's itself worth
    investigating rather than a nuisance to reconcile."""
    log_path = Path(raw_dir) / "collector.log"
    if not log_path.exists():
        return {"note": "no collector.log found"}
    lines = log_path.read_text().splitlines()
    return {
        "reconnects": sum("reconnected" in line for line in lines),
        "snapshot_failures": sum(
            "snapshot" in line and "failed" in line for line in lines
        ),
        "loop_crashes": sum("crashed" in line for line in lines),
    }


def latency_stats_ms(raw_dir: Path, sample_files: int = 3) -> dict:
    """Compares Binance's exchange event time (E) against our local receipt
    time (_recv_time_ms) directly from raw depth JSONL. Pure monitoring
    signal (network + processing latency) that never feeds into features, so
    it's computed straight from raw data rather than through the reconstructed
    table. Samples a few files rather than the whole capture -- this only
    needs to catch gross clock/latency problems, not be exhaustive."""
    depth_dir = Path(raw_dir) / "depth_events"
    files = sorted(depth_dir.glob("depth_*.jsonl"))[-sample_files:]
    deltas = []
    for path in files:
        with open(path) as f:
            for line in f:
                try:
                    rec = json.loads(line)
                    deltas.append(rec["_recv_time_ms"] - rec["E"])
                except (json.JSONDecodeError, KeyError):
                    continue
    if not deltas:
        return {"note": "no depth events found to sample"}
    arr = np.array(deltas)
    return {
        "n_sampled": len(arr),
        "median_ms": float(np.median(arr)),
        "p95_ms": float(np.percentile(arr, 95)),
        "max_ms": float(arr.max()),
        # Negative would mean receipt before exchange time -- clock skew
        # between this machine and Binance, worth knowing about either way.
        "negative_count": int((arr < 0).sum()),
    }


def plot_mid_price(df: pd.DataFrame, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(df.index, df["mid_price"], linewidth=0.6)
    ax.set_title("Mid price -- full capture window")
    ax.set_ylabel("mid price")
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(out_dir / "mid_price_full.png", dpi=150)
    plt.close(fig)

    zoom_start = df.index[-1] - pd.Timedelta(minutes=5)
    zoomed = df[df.index >= zoom_start]
    if len(zoomed) > 1:
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(zoomed.index, zoomed["mid_price"], linewidth=0.8)
        ax.set_title("Mid price -- last 5 minutes")
        ax.set_ylabel("mid price")
        fig.autofmt_xdate()
        fig.tight_layout()
        fig.savefig(out_dir / "mid_price_zoom_5min.png", dpi=150)
        plt.close(fig)


def run_sanity_checks(raw_dir: Path, processed_dir: Path, report_dir: Path) -> dict:
    df = load_processed(processed_dir)

    report = {
        "spread_positivity": check_spread_positivity(df),
        "spread_stats_bps": spread_stats_bps(df),
        "message_rate_and_resync": message_rate_and_resync_stats(df),
        "collector_log": collector_log_summary(raw_dir),
        "latency_ms": latency_stats_ms(raw_dir),
    }

    plot_mid_price(df, report_dir)

    report_dir.mkdir(parents=True, exist_ok=True)
    (report_dir / "summary.json").write_text(json.dumps(report, indent=2, default=str))

    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Stage 1 data-quality sanity checks.")
    parser.add_argument("--raw", default="data/raw")
    parser.add_argument("--processed", default="data/processed/book")
    parser.add_argument("--report", default="reports/sanity")
    args = parser.parse_args()

    report = run_sanity_checks(Path(args.raw), Path(args.processed), Path(args.report))
    print(json.dumps(report, indent=2, default=str))

    if report["spread_positivity"]["n_violations"] > 0:
        print(
            f"\n*** WARNING: {report['spread_positivity']['n_violations']} rows with "
            "spread <= 0 -- this indicates a reconstruction bug, investigate before "
            "trusting any downstream features. ***"
        )


if __name__ == "__main__":
    main()
