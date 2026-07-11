"""
Tests for src/orderbook.py.

These are the tests that matter most in the whole repo: if book reconstruction is
wrong, every OFI number downstream is silently wrong too, with no obvious symptom.
Every scenario here is a synthetic, hand-constructed sequence -- no network, no
real market data -- specifically so the resync/gap-detection logic can be verified
against known-correct expected states.
"""

from src.orderbook import DepthEvent, LocalOrderBook, Snapshot, resync_and_replay


def snapshot_fetcher(*snapshots: Snapshot):
    """Returns a zero-arg callable that yields the given snapshots in order, one
    per call, and records how many times it was called (via `.calls`)."""
    remaining = list(snapshots)
    state = {"calls": 0}

    def _fetch() -> Snapshot:
        state["calls"] += 1
        return remaining.pop(0)

    _fetch.calls = state
    return _fetch


def test_clean_sequence_reconstructs_correctly():
    """No gaps: snapshot + three valid chained events should produce the exact
    expected book state, and the snapshot should only be fetched once. Only the
    very first applied event should be flagged as a resync point."""
    snap = Snapshot(
        last_update_id=100,
        bids=[(99.0, 1.0), (98.0, 2.0)],
        asks=[(101.0, 1.0), (102.0, 2.0)],
    )
    events = [
        DepthEvent(1, 101, 101, 100, bids=[(99.0, 1.5)], asks=[]),
        DepthEvent(2, 102, 102, 101, bids=[(100.0, 2.0)], asks=[]),
        DepthEvent(3, 103, 103, 102, bids=[], asks=[(101.0, 0.0)]),  # remove level
    ]
    fetch = snapshot_fetcher(snap)

    book = None
    resync_flags = []
    for book, _, is_resync in resync_and_replay(iter(events), fetch):
        resync_flags.append(is_resync)

    assert fetch.calls["calls"] == 1
    assert resync_flags == [True, False, False]
    assert book.best_bid == (100.0, 2.0)
    assert book.best_ask == (102.0, 2.0)  # 101 level was removed
    assert book.bids == {100.0: 2.0, 99.0: 1.5, 98.0: 2.0}
    assert book.asks == {102.0: 2.0}
    assert book.last_update_id == 103
    assert book.spread == 2.0
    assert book.mid_price == 101.0


def test_bridging_event_pu_mismatch_is_not_treated_as_a_gap():
    """The first event applied after a (re)sync can legitimately have a `pu`
    that does NOT equal the snapshot's lastUpdateId -- e.g. here the event
    bundles multiple update IDs (U=99, u=105) so lastUpdateId+1=101 falls
    strictly inside its range, and its `pu` (50) refers to whatever real event
    preceded it in the live stream, which has nothing to do with the snapshot.
    Per Binance's docs this must still be applied (validity comes from the
    U<=lastUpdateId+1<=u check), NOT rejected as a sequence gap."""
    snap = Snapshot(last_update_id=100, bids=[(99.0, 1.0)], asks=[(101.0, 1.0)])
    events = [
        DepthEvent(1, 99, 105, 50, bids=[(99.0, 2.0)], asks=[]),
        DepthEvent(2, 106, 106, 105, bids=[(99.0, 3.0)], asks=[]),  # chains normally
    ]
    fetch = snapshot_fetcher(snap)

    book = None
    resync_flags = []
    for book, _, is_resync in resync_and_replay(iter(events), fetch):
        resync_flags.append(is_resync)

    assert resync_flags == [True, False]
    assert book.bids == {99.0: 3.0}
    assert book.last_update_id == 106


def test_gap_triggers_resync_not_silent_patch():
    """A `pu` mismatch mid-stream must be caught and trigger a full resync from a
    fresh snapshot -- it must NOT be silently applied as if nothing happened."""
    snap1 = Snapshot(
        last_update_id=100,
        bids=[(99.0, 1.0)],
        asks=[(101.0, 1.0)],
    )
    # After snap1, snap2 represents the true exchange state once events up to
    # update 103 have happened for real -- but our local replica never saw update
    # 103 correctly because e3 below claims a bogus `pu`.
    snap2 = Snapshot(
        last_update_id=103,
        bids=[(100.0, 5.0)],
        asks=[(103.0, 5.0)],
    )
    events = [
        DepthEvent(1, 101, 101, 100, bids=[], asks=[]),  # valid, chains to snap1
        DepthEvent(2, 102, 102, 101, bids=[], asks=[]),  # valid
        DepthEvent(3, 103, 103, 999, bids=[], asks=[]),  # BOGUS pu -> gap
        DepthEvent(4, 104, 104, 103, bids=[], asks=[(104.0, 1.0)]),  # bridges snap2
    ]
    fetch = snapshot_fetcher(snap1, snap2)
    resync_messages = []

    book = None
    resync_flags = []
    for book, _, is_resync in resync_and_replay(
        iter(events), fetch, on_resync=resync_messages.append
    ):
        resync_flags.append(is_resync)

    assert fetch.calls["calls"] == 2, "must have refetched a snapshot after the gap"
    assert len(resync_messages) == 1
    assert "gap" in resync_messages[0]
    # Event 4 is the new resync point; events 1-2 were the original sync point.
    assert resync_flags == [True, False, True]
    # Final state must come from snap2 + event 4, NOT from patching event 3 onto snap1.
    assert book.best_bid == (100.0, 5.0)
    assert book.asks == {103.0: 5.0, 104.0: 1.0}
    assert book.last_update_id == 104


def test_gap_is_caught_via_U_when_pu_is_always_absent():
    """Some feeds (e.g. Binance.US) never populate `pu` on any event, not just the
    post-resync one. Gap detection must then fall back to checking `U` continuity
    -- if it silently no-ops instead (treating "pu is None" as "nothing to check"),
    a dropped message goes undetected and the book is left permanently corrupted
    with no resync ever triggered."""
    snap1 = Snapshot(last_update_id=100, bids=[(99.0, 1.0)], asks=[(101.0, 1.0)])
    snap2 = Snapshot(last_update_id=103, bids=[(100.0, 5.0)], asks=[(103.0, 5.0)])
    events = [
        DepthEvent(1, 101, 101, None, bids=[], asks=[]),  # valid, chains to snap1
        DepthEvent(2, 102, 102, None, bids=[], asks=[]),  # valid
        DepthEvent(3, 999, 103, None, bids=[], asks=[]),  # BOGUS U -> gap
        DepthEvent(4, 104, 104, None, bids=[], asks=[(104.0, 1.0)]),  # bridges snap2
    ]
    fetch = snapshot_fetcher(snap1, snap2)
    resync_messages = []

    book = None
    resync_flags = []
    for book, _, is_resync in resync_and_replay(
        iter(events), fetch, on_resync=resync_messages.append
    ):
        resync_flags.append(is_resync)

    assert fetch.calls["calls"] == 2, "must have refetched a snapshot after the gap"
    assert len(resync_messages) == 1
    assert "gap" in resync_messages[0]
    assert resync_flags == [True, False, True]
    assert book.best_bid == (100.0, 5.0)
    assert book.asks == {103.0: 5.0, 104.0: 1.0}
    assert book.last_update_id == 104
    assert book.spread == 3.0  # would be negative if event 3 had been misapplied


def test_stale_duplicate_event_is_dropped_not_applied():
    """An old/duplicate event (u <= snapshot.lastUpdateId) arriving in the buffer
    must be discarded, not applied -- applying it would double-count an update
    already reflected in the snapshot."""
    snap = Snapshot(last_update_id=100, bids=[(99.0, 1.0)], asks=[(101.0, 1.0)])
    events = [
        # Stale: already covered by the snapshot (u=95 <= lastUpdateId=100).
        # If this were wrongly applied it would set the bid at 50.0 to 999.
        DepthEvent(1, 90, 95, 89, bids=[(50.0, 999.0)], asks=[]),
        DepthEvent(2, 101, 101, 100, bids=[(99.0, 2.0)], asks=[]),
    ]
    fetch = snapshot_fetcher(snap)

    book = None
    for book, _, _ in resync_and_replay(iter(events), fetch):
        pass

    assert 50.0 not in book.bids, "stale event must not have been applied"
    assert book.bids == {99.0: 2.0}
    assert book.last_update_id == 101


def test_stale_snapshot_relative_to_buffer_triggers_refetch():
    """If even the oldest buffered event starts after lastUpdateId+1, waiting for
    more events can never close that hole -- the code must refetch a fresher
    snapshot instead of buffering forever."""
    snap1 = Snapshot(last_update_id=100, bids=[(1.0, 1.0)], asks=[(2.0, 1.0)])
    snap2 = Snapshot(last_update_id=110, bids=[(10.0, 1.0)], asks=[(11.0, 1.0)])
    events = [
        # U=105 > lastUpdateId(100)+1 => snap1 is stale relative to this event.
        DepthEvent(1, 105, 106, 104, bids=[], asks=[]),
        # This event bridges snap2 (U=111 <= 111 <= u=111).
        DepthEvent(2, 111, 111, 110, bids=[(12.0, 3.0)], asks=[]),
    ]
    fetch = snapshot_fetcher(snap1, snap2)
    resync_messages = []

    book = None
    for book, _, _ in resync_and_replay(
        iter(events), fetch, on_resync=resync_messages.append
    ):
        pass

    assert fetch.calls["calls"] == 2, "must refetch once snap1 is found stale"
    assert any("stale" in m for m in resync_messages)
    assert book.last_update_id == 111
    assert book.bids == {10.0: 1.0, 12.0: 3.0}


def test_all_buffered_events_after_resync_are_yielded_individually():
    """If several events accumulate in the buffer before the bridging condition
    is met, each must still be applied and yielded one at a time (so the
    processed table keeps per-event granularity), not collapsed into a single
    row for the whole batch."""
    snap = Snapshot(last_update_id=100, bids=[(1.0, 1.0)], asks=[(2.0, 1.0)])
    events = [
        DepthEvent(1, 101, 101, 100, bids=[(1.0, 2.0)], asks=[]),
        DepthEvent(2, 102, 102, 101, bids=[(1.0, 3.0)], asks=[]),
        DepthEvent(3, 103, 103, 102, bids=[(1.0, 4.0)], asks=[]),
    ]
    fetch = snapshot_fetcher(snap)

    seen = []
    for book, event, is_resync in resync_and_replay(iter(events), fetch):
        seen.append((event.final_update_id, is_resync, dict(book.bids)))

    assert [s[0] for s in seen] == [101, 102, 103]
    assert [s[1] for s in seen] == [True, False, False]
    # Each row's book state should reflect only events applied up to that point.
    assert seen[0][2] == {1.0: 2.0}
    assert seen[1][2] == {1.0: 3.0}
    assert seen[2][2] == {1.0: 4.0}


def test_top_n_and_prune_bound_book_size():
    """`top_n` returns levels sorted correctly (best bid first descending, best
    ask first ascending), and `_prune` keeps the tracked book bounded."""
    book = LocalOrderBook(depth_levels=3)
    book.apply_snapshot(
        Snapshot(
            last_update_id=1,
            bids=[(100.0, 1.0), (99.0, 1.0), (98.0, 1.0), (97.0, 1.0)],
            asks=[(101.0, 1.0), (102.0, 1.0), (103.0, 1.0), (104.0, 1.0)],
        )
    )
    assert len(book.bids) == 3  # 97.0 pruned (furthest from touch)
    assert len(book.asks) == 3  # 104.0 pruned
    top = book.top_n(2)
    assert top["bid_prices"] == [100.0, 99.0]
    assert top["ask_prices"] == [101.0, 102.0]
