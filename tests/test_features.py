"""
Tests for src/features.py: the OFI formula itself, resync exclusion, window
aggregation, and the no-lookahead contract on forward-return targets. All
synthetic/hand-computed -- no dependency on real captured data.
"""

import json

import numpy as np
import pandas as pd
import pytest

from src.features import (
    aggregate_trade_imbalance,
    aggregate_windows,
    compute_event_ofi,
    compute_targets,
    compute_trailing_returns,
    load_raw_trades,
)


def _book_row(t, bid_p, bid_q, ask_p, ask_q, resync=False, levels=1):
    """One book-state row with `levels` levels populated (rest NaN), matching
    reconstruct_book_states' column schema."""
    row = {"resync": resync}
    for i in range(1, 6):
        if i <= levels:
            row[f"bid_price_{i}"] = bid_p - (i - 1)
            row[f"bid_size_{i}"] = bid_q
            row[f"ask_price_{i}"] = ask_p + (i - 1)
            row[f"ask_size_{i}"] = ask_q
        else:
            row[f"bid_price_{i}"] = np.nan
            row[f"bid_size_{i}"] = np.nan
            row[f"ask_price_{i}"] = np.nan
            row[f"ask_size_{i}"] = np.nan
    row["mid_price"] = (bid_p + ask_p) / 2
    row["spread"] = ask_p - bid_p
    row["_t"] = t
    return row


def _make_df(rows):
    df = pd.DataFrame(rows)
    df.index = pd.to_datetime(df.pop("_t"), unit="s", utc=True)
    return df


def test_level1_ofi_matches_hand_computed_formula():
    """Four consecutive transitions, one per branch of the CKS2014 formula:
    bid price up, ask price down (both bullish+bearish mix), then bid flat
    with size change, then ask flat with size change -- each checked against
    the hand-derived expected value."""
    rows = [
        _book_row(0, bid_p=100.0, bid_q=1.0, ask_p=101.0, ask_q=1.0),
        # bid up (99->100? no: 100->101), ask unchanged in price, size same
        _book_row(1, bid_p=101.0, bid_q=2.0, ask_p=101.0, ask_q=1.0),
        # bid flat, size 2.0->3.0 (+1 contribution); ask flat, size 1.0->1.0 (0)
        _book_row(2, bid_p=101.0, bid_q=3.0, ask_p=101.0, ask_q=1.0),
        # bid flat (size unchanged, 0); ask price down 101->100.5 -> +ask_q(=2.0) subtracted
        _book_row(3, bid_p=101.0, bid_q=3.0, ask_p=100.5, ask_q=2.0),
    ]
    df = compute_event_ofi(_make_df(rows), levels=1)

    # row 0: no predecessor -> NaN
    assert np.isnan(df["ofi_best"].iloc[0])
    # row 1: bid price up -> bid_delta = curr bid_q = 2.0; ask flat -> ask_delta = 1.0-1.0=0
    assert df["ofi_best"].iloc[1] == pytest.approx(2.0 - 0.0)
    # row 2: bid flat -> bid_delta = 3.0-2.0=1.0; ask flat -> ask_delta = 1.0-1.0=0
    assert df["ofi_best"].iloc[2] == pytest.approx(1.0 - 0.0)
    # row 3: bid flat -> bid_delta = 3.0-3.0=0; ask price down -> ask_delta = curr ask_q = 2.0
    assert df["ofi_best"].iloc[3] == pytest.approx(0.0 - 2.0)


def test_resync_row_ofi_is_nan_not_a_stale_delta():
    rows = [
        _book_row(0, bid_p=100.0, bid_q=1.0, ask_p=101.0, ask_q=1.0),
        _book_row(1, bid_p=150.0, bid_q=1.0, ask_p=151.0, ask_q=1.0, resync=True),
        _book_row(2, bid_p=150.5, bid_q=2.0, ask_p=151.0, ask_q=1.0),
    ]
    df = compute_event_ofi(_make_df(rows), levels=1)

    assert np.isnan(df["ofi_best"].iloc[1]), "resync row must not diff against pre-gap state"
    # row 2 is a normal event relative to row 1 (the resync row), unaffected
    assert not np.isnan(df["ofi_best"].iloc[2])


def test_multilevel_ofi_treats_missing_level_as_no_contribution():
    """Row 1 only has 1 real level; row 2 has 2. Level 2's contribution must
    be excluded (not NaN-poison the whole multilevel sum) since it's
    undefined on row 1."""
    rows = [
        _book_row(0, bid_p=100.0, bid_q=1.0, ask_p=101.0, ask_q=1.0, levels=2),
        _book_row(1, bid_p=101.0, bid_q=2.0, ask_p=101.0, ask_q=1.0, levels=1),
    ]
    df = compute_event_ofi(_make_df(rows), levels=2)

    # level 1: bid up -> bid_delta=2.0, ask flat -> ask_delta=0 => e1 = 2.0
    # level 2: undefined on row 1 (NaN price) -> excluded, weight 1/2 unused
    assert df["ofi_multilevel"].iloc[1] == pytest.approx(2.0 * 1.0)
    assert not np.isnan(df["ofi_multilevel"].iloc[1])


def test_aggregate_windows_sums_ofi_and_fills_quiet_bins():
    rows = [
        _book_row(0, 100.0, 1.0, 101.0, 1.0),
        _book_row(1, 100.0, 2.0, 101.0, 1.0),  # in window [0,2)
        # window [2,4): no events at all
        _book_row(4, 100.0, 1.0, 101.0, 3.0),  # in window [4,6)
    ]
    df = compute_event_ofi(_make_df(rows), levels=1)
    win = aggregate_windows(df, window_seconds=2.0)

    assert len(win) == 3  # windows starting at t=0, 2, 4
    assert win["n_events"].tolist() == [2, 0, 1]
    # quiet window [2,4) must carry OFI=0, not NaN
    assert win["ofi_best"].iloc[1] == 0.0
    assert win["mid_price_filled"].iloc[1] == True  # noqa: E712
    # its mid_price is forward-filled from the last real window
    assert win["mid_price"].iloc[1] == win["mid_price"].iloc[0]


def test_targets_use_only_future_windows_and_trailing_rows_are_nan():
    idx = pd.date_range("2026-01-01", periods=5, freq="1s")
    win = pd.DataFrame(
        {"ofi_best": [0.0] * 5, "mid_price": [100.0, 101.0, 102.0, 103.0, 104.0]},
        index=idx,
    )
    out = compute_targets(win, window_seconds=1.0, horizons_seconds=[1.0, 2.0])

    expected_1s = np.log(win["mid_price"].shift(-1)) - np.log(win["mid_price"])
    assert out["fwd_ret_1s"].iloc[:4].tolist() == pytest.approx(
        expected_1s.iloc[:4].tolist()
    )
    assert np.isnan(out["fwd_ret_1s"].iloc[-1]), "no future data -> NaN, not leaked/filled"
    assert np.isnan(out["fwd_ret_2s"].iloc[-2:]).all()


def test_horizon_must_be_integer_multiple_of_window():
    idx = pd.date_range("2026-01-01", periods=3, freq="1s")
    win = pd.DataFrame({"mid_price": [100.0, 101.0, 102.0]}, index=idx)
    with pytest.raises(ValueError):
        compute_targets(win, window_seconds=2.0, horizons_seconds=[3.0])


def test_trailing_returns_look_backward_and_leading_rows_are_nan():
    """Mirror image of compute_targets: trail_ret_1s[t] should equal
    fwd_ret_1s computed from t-1, and the *leading* (not trailing) rows are
    the ones with no data yet."""
    idx = pd.date_range("2026-01-01", periods=5, freq="1s")
    win = pd.DataFrame(
        {"mid_price": [100.0, 101.0, 102.0, 103.0, 104.0]}, index=idx
    )
    out = compute_trailing_returns(win, window_seconds=1.0, horizons_seconds=[1.0])

    assert np.isnan(out["trail_ret_1s"].iloc[0]), "no past data yet -> NaN"
    expected = np.log(win["mid_price"]) - np.log(win["mid_price"].shift(1))
    assert out["trail_ret_1s"].iloc[1:].tolist() == pytest.approx(
        expected.iloc[1:].tolist()
    )


def test_trailing_return_horizon_must_be_integer_multiple_of_window():
    idx = pd.date_range("2026-01-01", periods=3, freq="1s")
    win = pd.DataFrame({"mid_price": [100.0, 101.0, 102.0]}, index=idx)
    with pytest.raises(ValueError):
        compute_trailing_returns(win, window_seconds=2.0, horizons_seconds=[3.0])


def test_load_raw_trades_signs_by_buyer_maker_flag(tmp_path):
    """m=True -> buyer was the passive maker -> seller-initiated -> negative.
    m=False -> buyer was the aggressor -> buyer-initiated -> positive."""
    trades_dir = tmp_path / "trades"
    trades_dir.mkdir()
    lines = [
        json.dumps({"E": 2000, "q": "1.5", "m": False}),  # buyer-initiated: +1.5
        json.dumps({"E": 1000, "q": "0.5", "m": True}),  # seller-initiated: -0.5
        "{not valid json",  # torn last line, must be skipped not raise
    ]
    (trades_dir / "trade_2026-01-01_00.jsonl").write_text("\n".join(lines) + "\n")

    df = load_raw_trades(tmp_path)

    assert list(df["signed_qty"]) == [-0.5, 1.5]  # sorted by event time
    assert df.index.is_monotonic_increasing


def test_load_raw_trades_empty_when_no_files(tmp_path):
    (tmp_path / "trades").mkdir()
    df = load_raw_trades(tmp_path)
    assert df.empty
    assert "signed_qty" in df.columns


def test_aggregate_trade_imbalance_sums_into_windows_and_fills_quiet_bins():
    idx = pd.date_range("2026-01-01", periods=2, freq="1s", tz="UTC")  # 2 windows of 1s
    trades = pd.DataFrame(
        {"signed_qty": [1.0, 2.0, -0.5]},
        index=pd.to_datetime(
            ["2026-01-01 00:00:00.100", "2026-01-01 00:00:00.900", "2026-01-01 00:00:02.100"],
            utc=True,
        ),
    )
    # trades fall in windows [00:00:00,00:00:01) and [00:00:02,00:00:03) --
    # the second window isn't in `idx` at all, so it must be dropped, and the
    # untouched window [00:00:01,00:00:02) must come back as 0.0, not NaN.
    result = aggregate_trade_imbalance(trades, window_seconds=1.0, full_index=idx)

    assert result.loc[idx[0]] == pytest.approx(3.0)
    assert result.loc[idx[1]] == 0.0
    assert len(result) == 2


def test_aggregate_trade_imbalance_empty_trades_returns_all_zero():
    idx = pd.date_range("2026-01-01", periods=3, freq="1s")
    empty = pd.DataFrame({"signed_qty": pd.Series(dtype=float)})
    result = aggregate_trade_imbalance(empty, window_seconds=1.0, full_index=idx)
    assert (result == 0.0).all()
    assert len(result) == 3


def test_aggregate_trade_imbalance_raises_on_tz_mismatch():
    """A tz-naive/aware mismatch must raise, not silently reindex to all-zero
    (which would be indistinguishable from a genuinely quiet market)."""
    naive_idx = pd.date_range("2026-01-01", periods=2, freq="1s")
    aware_trades = pd.DataFrame(
        {"signed_qty": [1.0]},
        index=pd.to_datetime(["2026-01-01 00:00:00.500"], utc=True),
    )
    with pytest.raises(ValueError):
        aggregate_trade_imbalance(aware_trades, window_seconds=1.0, full_index=naive_idx)
