"""
Tests for the offline batch reconstruction pieces of src/orderbook.py: parsing
raw JSONL/snapshot files off disk, the forward-only snapshot sequencer, and
`reconstruct_book_states` end-to-end (using small in-memory event/snapshot
lists so the test doesn't depend on any real captured data).
"""

import json

import pandas as pd
import pytest

from src.orderbook import (
    DepthEvent,
    Snapshot,
    load_raw_depth_events,
    load_raw_snapshots,
    make_snapshot_sequencer,
    parse_depth_event,
    parse_snapshot,
    reconstruct_book_states,
)


def test_parse_depth_event_uses_binance_field_names():
    raw = {
        "e": "depthUpdate",
        "E": 1000,
        "s": "BTCUSDT",
        "U": 10,
        "u": 12,
        "pu": 9,
        "b": [["100.5", "1.2"]],
        "a": [["100.6", "0.0"]],
    }
    event = parse_depth_event(raw)
    assert event.event_time_ms == 1000
    assert event.first_update_id == 10
    assert event.final_update_id == 12
    assert event.prev_final_update_id == 9
    assert event.bids == [(100.5, 1.2)]
    assert event.asks == [(100.6, 0.0)]


def test_parse_snapshot():
    raw = {"lastUpdateId": 42, "bids": [["100.0", "1.0"]], "asks": [["101.0", "2.0"]]}
    snap = parse_snapshot(raw)
    assert snap.last_update_id == 42
    assert snap.bids == [(100.0, 1.0)]
    assert snap.asks == [(101.0, 2.0)]


def test_load_raw_depth_events_sorts_across_files_and_skips_malformed(tmp_path):
    depth_dir = tmp_path / "depth_events"
    depth_dir.mkdir()
    # Deliberately write the later hour's file first and put events out of
    # order within a file, to verify the loader sorts by (event_time, u).
    (depth_dir / "depth_2026-01-01_01.jsonl").write_text(
        json.dumps(
            {"E": 2000, "U": 20, "u": 20, "pu": 19, "b": [], "a": []}
        ) + "\n"
        + "{not valid json\n"  # simulates a torn last line from an abrupt kill
    )
    (depth_dir / "depth_2026-01-01_00.jsonl").write_text(
        json.dumps({"E": 1000, "U": 10, "u": 10, "pu": 9, "b": [], "a": []}) + "\n"
    )

    events = load_raw_depth_events(tmp_path)

    assert [e.event_time_ms for e in events] == [1000, 2000]


def test_load_raw_snapshots_parses_filename_timestamp_and_sorts(tmp_path):
    snap_dir = tmp_path / "snapshots"
    snap_dir.mkdir()
    (snap_dir / "snapshot_2026-01-01T01-00-00.json").write_text(
        json.dumps({"lastUpdateId": 200, "bids": [], "asks": []})
    )
    (snap_dir / "snapshot_2026-01-01T00-00-00.json").write_text(
        json.dumps({"lastUpdateId": 100, "bids": [], "asks": []})
    )

    snaps = load_raw_snapshots(tmp_path)

    assert [s.last_update_id for _, s in snaps] == [100, 200]
    assert snaps[0][0] < snaps[1][0]  # capture_time_ms is increasing


def test_snapshot_sequencer_is_forward_only_and_raises_when_exhausted():
    snaps = [
        (1, Snapshot(last_update_id=1, bids=[], asks=[])),
        (2, Snapshot(last_update_id=2, bids=[], asks=[])),
    ]
    fetch = make_snapshot_sequencer(snaps)
    assert fetch().last_update_id == 1
    assert fetch().last_update_id == 2
    with pytest.raises(RuntimeError):
        fetch()


def test_reconstruct_book_states_end_to_end(tmp_path):
    snapshot = Snapshot(last_update_id=100, bids=[(99.0, 1.0)], asks=[(101.0, 1.0)])
    events = [
        DepthEvent(1_000, 101, 101, 100, bids=[(99.0, 2.0)], asks=[]),
        DepthEvent(1_100, 102, 102, 101, bids=[], asks=[(100.5, 1.0)]),
    ]
    out_dir = tmp_path / "processed"

    df = reconstruct_book_states(
        raw_dir=tmp_path,  # unused: events/snapshots override below
        out_dir=out_dir,
        depth_levels=5,
        events=events,
        snapshots=[(0, snapshot)],
    )

    assert list(df["resync"]) == [True, False]
    assert df["bid_price_1"].tolist() == [99.0, 99.0]
    assert df["ask_price_1"].tolist() == [101.0, 100.5]
    assert df["mid_price"].tolist() == [100.0, 99.75]

    # One partition per UTC calendar day should have been written to disk, and
    # it should be byte-for-byte re-readable via pandas/pyarrow.
    written = list(out_dir.glob("*.parquet"))
    assert len(written) == 1
    reloaded = pd.read_parquet(written[0])
    assert reloaded["bid_price_1"].tolist() == [99.0, 99.0]


def test_reconstruct_book_states_raises_without_snapshots(tmp_path):
    (tmp_path / "depth_events").mkdir()
    (tmp_path / "snapshots").mkdir()  # exists but empty
    with pytest.raises(ValueError):
        reconstruct_book_states(tmp_path, tmp_path / "out")
