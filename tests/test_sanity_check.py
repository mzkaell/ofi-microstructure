"""
Tests for scripts/sanity_check.py's pure-logic pieces, using small synthetic
DataFrames/files rather than a real capture.
"""

import json

import pandas as pd

from scripts.sanity_check import (
    check_spread_positivity,
    collector_log_summary,
    latency_stats_ms,
    message_rate_and_resync_stats,
    plot_mid_price,
    spread_stats_bps,
)


def _make_df(n=5, spread=10.0, mid=100.0, resync_at=(0,)):
    idx = pd.date_range("2026-01-01", periods=n, freq="100ms", tz="UTC")
    return pd.DataFrame(
        {
            "mid_price": [mid] * n,
            "spread": [spread] * n,
            "resync": [i in resync_at for i in range(n)],
        },
        index=idx,
    )


def test_check_spread_positivity_flags_violations():
    df = _make_df(n=3, spread=10.0)
    df.iloc[1, df.columns.get_loc("spread")] = -1.0  # inject a violation

    result = check_spread_positivity(df)

    assert result["n_rows"] == 3
    assert result["n_violations"] == 1


def test_check_spread_positivity_clean_data():
    df = _make_df(n=5, spread=10.0)
    result = check_spread_positivity(df)
    assert result["n_violations"] == 0


def test_spread_stats_bps_computes_relative_to_mid():
    df = _make_df(n=4, spread=10.0, mid=100.0)  # 10/100 * 1e4 = 1000 bps
    stats = spread_stats_bps(df)
    assert stats["mean_bps"] == 1000.0
    assert stats["median_bps"] == 1000.0


def test_message_rate_and_resync_stats():
    df = _make_df(n=11, resync_at=(0, 5))  # 100ms spacing -> 1.0s span, 10 intervals
    stats = message_rate_and_resync_stats(df)
    assert stats["n_events"] == 11
    assert stats["n_resync_points"] == 2
    assert stats["span_seconds"] == 1.0


def test_collector_log_summary_counts_lines(tmp_path):
    (tmp_path / "collector.log").write_text(
        "2026-01-01T00:00:00 collector starting for BTCUSDT\n"
        "2026-01-01T00:00:01 [depth] reconnected to ws\n"
        "2026-01-01T00:00:02 [snapshot] fetch failed: timeout\n"
        "2026-01-01T00:00:03 [trade] loop crashed: err\n"
    )
    summary = collector_log_summary(tmp_path)
    assert summary["reconnects"] == 1
    assert summary["snapshot_failures"] == 1
    assert summary["loop_crashes"] == 1


def test_collector_log_summary_missing_file(tmp_path):
    summary = collector_log_summary(tmp_path)
    assert "note" in summary


def test_latency_stats_ms_computes_recv_minus_exchange_time(tmp_path):
    depth_dir = tmp_path / "depth_events"
    depth_dir.mkdir()
    lines = [
        json.dumps({"E": 1000, "_recv_time_ms": 1050}),
        json.dumps({"E": 2000, "_recv_time_ms": 2030}),
    ]
    (depth_dir / "depth_2026-01-01_00.jsonl").write_text("\n".join(lines) + "\n")

    stats = latency_stats_ms(tmp_path)

    assert stats["n_sampled"] == 2
    assert stats["median_ms"] == 40.0
    assert stats["negative_count"] == 0


def test_plot_mid_price_writes_png_files(tmp_path):
    df = _make_df(n=20)
    out_dir = tmp_path / "figs"
    plot_mid_price(df, out_dir)
    assert (out_dir / "mid_price_full.png").exists()
