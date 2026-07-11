"""
Local L2 order book reconstruction from Binance's `<symbol>@depth` diff stream.

Why this file is the foundation of the whole study: OFI (Cont, Kukanov & Stoikov,
2014, "The Price Impact of Order Book Events") is defined as the event-by-event
*delta* between consecutive best-bid/ask states. If the book we replay ever applies
an update out of sequence -- because we silently skipped one after a dropped
WebSocket message or a REST-snapshot race -- the "delta" we compute is the difference
between two non-adjacent states. That doesn't crash anything or look unusual; it just
quietly injects noise/bias into every OFI number downstream. So every place this
module can detect a break in the update sequence, it raises a `SequenceGapError`
instead of guessing, and the caller must resync from a fresh REST snapshot rather
than patch over the gap.

Reconstruction procedure implemented here follows Binance's documented algorithm for
keeping a local order book in sync with the diff-depth stream:
https://developers.binance.com/docs/binance-spot-api-docs/web-socket-streams
("How to manage a local order book correctly"):

    1. Open the diff stream and buffer every event received.
    2. Fetch a REST depth snapshot (gives `lastUpdateId` + full bid/ask levels).
    3. Discard any buffered event whose `u` (final update ID) <= lastUpdateId --
       it's already reflected in the snapshot.
    4. The first event applied must satisfy `U <= lastUpdateId + 1 <= u` -- i.e. its
       range straddles the point right after the snapshot. Apply the snapshot, then
       that event (its own `pu` is irrelevant here -- it refers to whatever real
       event preceded it in the live stream, not to the snapshot), then every
       subsequent buffered event in order, checked normally from here on.
    5. Every event after that must have `pu == (last applied event's u)`. Any
       mismatch is a gap: discard the current book state, fetch a fresh snapshot,
       and restart from step 3.

This module also provides `reconstruct_book_states`, the offline batch job that
replays a raw capture (from acquisition.py) through `resync_and_replay` to produce
the processed top-N-level book-state table Stage 2 builds features from. It is
always rebuildable from data/raw/ -- if a reconstruction bug is found, we fix this
file and rerun the batch job, we never re-collect.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Iterable, Iterator, Optional

import pandas as pd


class SequenceGapError(Exception):
    """Raised when a diff-depth event does not chain onto the previously applied
    event (its `pu` does not match the last applied event's `u`). This signals the
    local book is no longer known-good; the caller must resync from a fresh
    snapshot rather than continue applying events."""


@dataclass
class DepthEvent:
    """One message from the `<symbol>@depth` diff stream, using Binance's field
    names (E/U/u/pu) directly so this class is easy to cross-check against the raw
    JSON while debugging."""

    event_time_ms: int  # 'E' -- exchange event time; the authoritative clock for
    # all downstream windowing (never local receipt time -- see acquisition.py).
    first_update_id: int  # 'U'
    final_update_id: int  # 'u'
    prev_final_update_id: Optional[int]  # 'pu'; may be absent for the very first
    # event applied right after a snapshot (no "previous" to check), and on some
    # feeds -- e.g. Binance.US -- is absent on every event, in which case
    # apply_event falls back to checking 'U' continuity instead.
    bids: list[tuple[float, float]]  # [(price, qty), ...]; qty == 0 means "remove
    # this price level" per the Binance diff-stream spec, not "size is zero".
    asks: list[tuple[float, float]]


@dataclass
class Snapshot:
    """A REST `GET /api/v3/depth` response."""

    last_update_id: int
    bids: list[tuple[float, float]]
    asks: list[tuple[float, float]]


class LocalOrderBook:
    """Maintains bid/ask price -> size maps and exposes best-bid/ask, mid-price,
    spread, and top-N levels. Contains no network code and no knowledge of
    "resyncing" -- that orchestration lives in `resync_and_replay` below, so this
    class can be unit-tested with plain synthetic data.
    """

    def __init__(self, depth_levels: int = 50):
        # We only ever need the top 5 levels for OFI features. Keeping ~50 gives a
        # large safety margin against a level "falling out of view" before an
        # OFI-relevant event depletes everything above it, while still bounding
        # memory growth over a multi-day run (the diff stream can reference price
        # levels arbitrarily far from the touch, which would otherwise accumulate
        # forever). Levels beyond this band essentially never reach the top 5
        # within the 1-10s horizons this study cares about.
        self._depth_levels = depth_levels
        self.bids: dict[float, float] = {}
        self.asks: dict[float, float] = {}
        self.last_update_id: Optional[int] = None
        self._initialized = False

    @property
    def initialized(self) -> bool:
        return self._initialized

    def apply_snapshot(self, snapshot: Snapshot) -> None:
        self.bids = {p: q for p, q in snapshot.bids if q > 0}
        self.asks = {p: q for p, q in snapshot.asks if q > 0}
        self.last_update_id = snapshot.last_update_id
        self._initialized = True
        self._prune()

    def apply_event(self, event: DepthEvent, check_gap: bool = True) -> None:
        """Apply one diff event. Raises SequenceGapError if it doesn't chain onto
        the last applied event -- the caller must catch this and resync.

        `check_gap=False` is for exactly one situation: the first event applied
        right after a (re)sync. Per Binance's docs, that event's validity is
        established by its [U, u] range straddling lastUpdateId+1 -- its `pu`
        refers to whatever real event preceded it in the live stream, which has
        nothing to do with the snapshot's lastUpdateId, so the normal pu-chain
        check does not apply to it. Every event after that one is a genuine
        continuation and must be checked normally.
        """
        if not self._initialized:
            raise RuntimeError("apply_snapshot must be called before apply_event")

        if check_gap:
            if event.prev_final_update_id is not None:
                if event.prev_final_update_id != self.last_update_id:
                    raise SequenceGapError(
                        f"event.pu={event.prev_final_update_id} != "
                        f"last_update_id={self.last_update_id}"
                    )
            else:
                # Some feeds (e.g. Binance.US, as opposed to Binance.com) never
                # populate `pu` on any event -- not just the post-resync one this
                # class otherwise expects it to be absent on. Falling back to a
                # no-op here would disable gap detection for the entire stream:
                # a dropped/reordered message would silently leave stale price
                # levels on one side of the book while the other side keeps
                # moving, eventually crossing the spread. Binance's pre-`pu`
                # algorithm covers exactly this case: the new event's `U` must
                # pick up immediately after the last applied event's `u`.
                if event.first_update_id != self.last_update_id + 1:
                    raise SequenceGapError(
                        f"event.U={event.first_update_id} != "
                        f"last_update_id + 1={self.last_update_id + 1}"
                    )

        for price, qty in event.bids:
            if qty == 0:
                self.bids.pop(price, None)
            else:
                self.bids[price] = qty
        for price, qty in event.asks:
            if qty == 0:
                self.asks.pop(price, None)
            else:
                self.asks[price] = qty

        self.last_update_id = event.final_update_id
        self._prune()

    def _prune(self) -> None:
        """Drop price levels beyond `depth_levels` from the touch, on each side."""
        if len(self.bids) > self._depth_levels:
            keep = sorted(self.bids, reverse=True)[: self._depth_levels]
            self.bids = {p: self.bids[p] for p in keep}
        if len(self.asks) > self._depth_levels:
            keep = sorted(self.asks)[: self._depth_levels]
            self.asks = {p: self.asks[p] for p in keep}

    def top_n(self, n: int = 5) -> dict:
        bid_prices = sorted(self.bids, reverse=True)[:n]
        ask_prices = sorted(self.asks)[:n]
        return {
            "bid_prices": bid_prices,
            "bid_sizes": [self.bids[p] for p in bid_prices],
            "ask_prices": ask_prices,
            "ask_sizes": [self.asks[p] for p in ask_prices],
        }

    @property
    def best_bid(self) -> Optional[tuple[float, float]]:
        if not self.bids:
            return None
        p = max(self.bids)
        return p, self.bids[p]

    @property
    def best_ask(self) -> Optional[tuple[float, float]]:
        if not self.asks:
            return None
        p = min(self.asks)
        return p, self.asks[p]

    @property
    def mid_price(self) -> Optional[float]:
        bb, ba = self.best_bid, self.best_ask
        if bb is None or ba is None:
            return None
        return (bb[0] + ba[0]) / 2

    @property
    def spread(self) -> Optional[float]:
        bb, ba = self.best_bid, self.best_ask
        if bb is None or ba is None:
            return None
        return ba[0] - bb[0]


def resync_and_replay(
    events: Iterator[DepthEvent],
    fetch_snapshot: Callable[[], Snapshot],
    on_resync: Optional[Callable[[str], None]] = None,
) -> Iterator[tuple[LocalOrderBook, DepthEvent]]:
    """Pure, network-free implementation of Binance's buffer/snapshot/resync
    procedure (see module docstring). Takes an iterator of already-received
    DepthEvents and a callable that returns a fresh Snapshot on demand -- in
    production (acquisition.py) `events` is fed by a live WebSocket buffer and
    `fetch_snapshot` makes a REST call; in tests, both are canned synthetic data.
    This separation is what makes the resync/gap-detection logic unit-testable
    without a network connection.

    Yields (book, event, is_resync_point) once per successfully applied event,
    including the first one right after each (re)sync. `is_resync_point` is
    True only for that first post-(re)sync event -- its delta against "the
    snapshot" is a coarse jump, not a real preceding order-book event, so it
    has no well-defined event-to-event OFI contribution and Stage 2 must
    exclude it (and only it) from per-event OFI terms.
    """
    on_resync = on_resync or (lambda msg: None)
    book = LocalOrderBook()
    buffer: list[DepthEvent] = []
    snapshot: Optional[Snapshot] = None
    synced = False

    for event in events:
        if synced:
            try:
                book.apply_event(event)
            except SequenceGapError as exc:
                on_resync(f"sequence gap ({exc}); resyncing from a fresh snapshot")
                synced = False
                buffer = [event]
                snapshot = None
            else:
                yield book, event, False
            continue

        # --- not yet synced: buffer this event and try to establish sync ---
        buffer.append(event)
        if snapshot is None:
            snapshot = fetch_snapshot()

        # Step 3: discard events already reflected in the snapshot.
        buffer = [e for e in buffer if e.final_update_id > snapshot.last_update_id]
        if not buffer:
            continue

        oldest = buffer[0]
        if oldest.first_update_id > snapshot.last_update_id + 1:
            # The snapshot is stale relative to even the oldest surviving buffered
            # event: there's a hole between the two that more buffering can never
            # close (waiting only accumulates events *after* the hole). Get a
            # fresher snapshot rather than looping forever.
            on_resync("snapshot stale relative to buffer; refetching")
            snapshot = fetch_snapshot()
            continue

        newest = buffer[-1]
        if newest.final_update_id < snapshot.last_update_id + 1:
            # Haven't buffered far enough past the snapshot yet -- normal, keep
            # waiting for more events.
            continue

        # Step 4: find the first buffered event whose [U, u] range contains
        # lastUpdateId + 1. That event is special (see apply_event's
        # `check_gap` docstring) -- apply it with the pu-chain check disabled.
        # Every event after it in the buffer is a genuine continuation of the
        # live stream and must chain normally, so it's applied (and yielded)
        # one at a time rather than in one silent batch.
        for i, e in enumerate(buffer):
            if e.first_update_id <= snapshot.last_update_id + 1 <= e.final_update_id:
                book.apply_snapshot(snapshot)
                book.apply_event(e, check_gap=False)
                yield book, e, True

                tail = buffer[i + 1 :]
                for ti, tail_event in enumerate(tail):
                    try:
                        book.apply_event(tail_event)
                    except SequenceGapError as exc:
                        # A second gap landed immediately after the first
                        # resync -- rare, but must be handled rather than
                        # crashing a multi-day/multi-file batch replay.
                        on_resync(
                            f"sequence gap ({exc}) immediately after resync; "
                            "resyncing again"
                        )
                        buffer = tail[ti:]
                        snapshot = None
                        synced = False
                        break
                    else:
                        yield book, tail_event, False
                else:
                    buffer = []
                    snapshot = None
                    synced = True
                break


# ---------------------------------------------------------------------------
# Offline batch reconstruction: raw capture (acquisition.py) -> processed
# top-N-level book-state Parquet table (Stage 2 input).
# ---------------------------------------------------------------------------


def parse_depth_event(raw: dict) -> DepthEvent:
    """Parses one raw depth-diff JSON record (Binance's own field names) into a
    DepthEvent. `pu` is absent on some historical/alternate feeds, hence `.get`."""
    return DepthEvent(
        event_time_ms=raw["E"],
        first_update_id=raw["U"],
        final_update_id=raw["u"],
        prev_final_update_id=raw.get("pu"),
        bids=[(float(p), float(q)) for p, q in raw["b"]],
        asks=[(float(p), float(q)) for p, q in raw["a"]],
    )


def parse_snapshot(raw: dict) -> Snapshot:
    return Snapshot(
        last_update_id=raw["lastUpdateId"],
        bids=[(float(p), float(q)) for p, q in raw["bids"]],
        asks=[(float(p), float(q)) for p, q in raw["asks"]],
    )


def load_raw_depth_events(raw_dir: Path) -> list[DepthEvent]:
    """Loads and parses every depth-event JSONL file under raw_dir/depth_events,
    sorted by (event_time, final_update_id). Unparseable lines are skipped -- a
    torn last line from an abrupt process kill is expected, by design, given
    acquisition.py's append-only JSONL writer, and is safely dropped here rather
    than raising."""
    events = []
    depth_dir = Path(raw_dir) / "depth_events"
    for path in sorted(depth_dir.glob("depth_*.jsonl")):
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    events.append(parse_depth_event(json.loads(line)))
                except (json.JSONDecodeError, KeyError):
                    continue
    events.sort(key=lambda e: (e.event_time_ms, e.final_update_id))
    return events


def load_raw_snapshots(raw_dir: Path) -> list[tuple[int, Snapshot]]:
    """Loads every snapshot dump under raw_dir/snapshots, returning
    (capture_time_ms, Snapshot) pairs sorted by capture time. capture_time_ms is
    parsed from the filename -- the wall-clock moment acquisition.py fetched it
    -- rather than derived from lastUpdateId, because resync_and_replay needs to
    hand out snapshots in the chronological order they were actually captured."""
    snaps = []
    snap_dir = Path(raw_dir) / "snapshots"
    for path in sorted(snap_dir.glob("snapshot_*.json")):
        ts_str = path.stem[len("snapshot_") :]
        try:
            dt = datetime.strptime(ts_str, "%Y-%m-%dT%H-%M-%S").replace(
                tzinfo=timezone.utc
            )
            raw = json.loads(path.read_text())
            snaps.append((int(dt.timestamp() * 1000), parse_snapshot(raw)))
        except (ValueError, json.JSONDecodeError, KeyError):
            continue
    snaps.sort(key=lambda pair: pair[0])
    return snaps


def make_snapshot_sequencer(
    snapshots: list[tuple[int, Snapshot]]
) -> Callable[[], Snapshot]:
    """Returns a zero-arg callable compatible with resync_and_replay's
    `fetch_snapshot` parameter, backed by pre-captured snapshots instead of a
    live REST call. Each call advances a forward-only pointer to the next
    snapshot in chronological order.

    Forward-only is deliberate, not just simplest: resync_and_replay only ever
    calls fetch_snapshot() to bridge a gap that occurred *after* wherever replay
    currently stands, so the correct next snapshot is always later in time than
    the previous one handed out -- there is never a reason to rewind.
    """
    idx = {"i": 0}

    def _fetch() -> Snapshot:
        if idx["i"] >= len(snapshots):
            raise RuntimeError(
                "ran out of captured snapshots to resync from -- the raw capture "
                "likely ended (or had a long enough disconnect) with no further "
                "snapshot available to bridge the remaining events"
            )
        _, snap = snapshots[idx["i"]]
        idx["i"] += 1
        return snap

    return _fetch


def reconstruct_book_states(
    raw_dir: Path,
    out_dir: Path,
    depth_levels: int = 5,
    events: Optional[Iterable[DepthEvent]] = None,
    snapshots: Optional[list[tuple[int, Snapshot]]] = None,
) -> pd.DataFrame:
    """Replays a raw capture through resync_and_replay and writes one row per
    successfully applied event to `out_dir`, partitioned into one Parquet file
    per UTC calendar day (of exchange event time). Columns: bid/ask price and
    size for each of the top `depth_levels` levels, mid_price, spread, and a
    `resync` flag (see resync_and_replay's docstring) marking rows with no
    well-defined event-to-event OFI delta -- Stage 2 must exclude these when
    computing per-event OFI terms, but they're kept in the table (not dropped
    here) since mid_price/spread are still perfectly valid for those rows.

    `events`/`snapshots` are accepted as explicit overrides purely so this
    function is testable against small synthetic inputs without touching disk;
    normal usage just passes `raw_dir`.
    """
    if events is None:
        events = load_raw_depth_events(raw_dir)
    if snapshots is None:
        snapshots = load_raw_snapshots(raw_dir)
    if not snapshots:
        raise ValueError(f"no snapshots found under {raw_dir}/snapshots")

    fetch_snapshot = make_snapshot_sequencer(snapshots)
    resync_messages: list[str] = []

    rows = []
    for book, event, is_resync in resync_and_replay(
        iter(events), fetch_snapshot, on_resync=resync_messages.append
    ):
        top = book.top_n(depth_levels)
        row = {"event_time_ms": event.event_time_ms, "resync": is_resync}
        for i in range(depth_levels):
            row[f"bid_price_{i + 1}"] = (
                top["bid_prices"][i] if i < len(top["bid_prices"]) else float("nan")
            )
            row[f"bid_size_{i + 1}"] = (
                top["bid_sizes"][i] if i < len(top["bid_sizes"]) else float("nan")
            )
            row[f"ask_price_{i + 1}"] = (
                top["ask_prices"][i] if i < len(top["ask_prices"]) else float("nan")
            )
            row[f"ask_size_{i + 1}"] = (
                top["ask_sizes"][i] if i < len(top["ask_sizes"]) else float("nan")
            )
        row["mid_price"] = book.mid_price
        row["spread"] = book.spread
        rows.append(row)

    df = pd.DataFrame(rows)
    df["event_time"] = pd.to_datetime(df["event_time_ms"], unit="ms", utc=True)
    df = df.set_index("event_time").sort_index()

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for date, day_df in df.groupby(df.index.date):
        day_df.to_parquet(out_dir / f"{date}.parquet")

    n_resync = int(df["resync"].sum())
    print(
        f"reconstructed {len(df)} book states, {df.index[0]}..{df.index[-1]}, "
        f"{n_resync} resync point(s), {len(resync_messages)} resync event(s) logged"
    )
    return df


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="Replay a raw Binance capture into a processed book-state table."
    )
    parser.add_argument("--raw", default="data/raw", help="raw capture directory")
    parser.add_argument(
        "--out", default="data/processed/book", help="output directory for Parquet"
    )
    parser.add_argument("--depth-levels", type=int, default=5)
    args = parser.parse_args()
    reconstruct_book_states(Path(args.raw), Path(args.out), args.depth_levels)


if __name__ == "__main__":
    main()
