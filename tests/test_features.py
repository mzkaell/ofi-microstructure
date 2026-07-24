"""
Tests for src/features.py: the OFI formula itself, resync exclusion, window
aggregation, and the no-lookahead contract on forward-return targets. All
synthetic/hand-computed -- no dependency on real captured data.
"""

import numpy as np
import pandas as pd
import pytest

from src.features import aggregate_windows, compute_event_ofi, compute_targets


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
        {
            "ofi_best": [0.0] * 5,
            "mid_price": [100.0, 101.0, 102.0, 103.0, 104.0],
            "mid_price_filled": [False] * 5,
        },
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


def test_targets_spanning_a_gap_are_nan_not_fabricated():
    """A multi-window data-collection gap forward-fills a constant mid_price
    (aggregate_windows' mid_price_filled=True). A target must be NaN if
    EITHER its origin or destination window falls in that gap -- otherwise a
    long gap manufactures thousands of fake exactly-zero returns paired with
    exactly-zero OFI, which is fabricated data, not a real observation of
    "the market didn't move"."""
    idx = pd.date_range("2026-01-01", periods=6, freq="1s")
    win = pd.DataFrame(
        {
            "ofi_best": [0.1, 0.1, 0.0, 0.0, 0.1, 0.1],
            "mid_price": [100.0, 100.5, 100.5, 100.5, 100.5, 101.0],
            #                real   real  FILLED FILLED  real   real
            "mid_price_filled": [False, False, True, True, False, False],
        },
        index=idx,
    )
    out = compute_targets(win, window_seconds=1.0, horizons_seconds=[1.0])

    # row 0 -> row 1: both real, valid target.
    assert not np.isnan(out["fwd_ret_1s"].iloc[0])
    # row 1 -> row 2: destination (row 2) is filled -> NaN, even though row 1 itself is real.
    assert np.isnan(out["fwd_ret_1s"].iloc[1])
    # rows 2, 3: origin itself filled -> NaN regardless of destination.
    assert np.isnan(out["fwd_ret_1s"].iloc[2])
    assert np.isnan(out["fwd_ret_1s"].iloc[3])
    # row 4 -> row 5: both real again, valid target.
    assert not np.isnan(out["fwd_ret_1s"].iloc[4])
