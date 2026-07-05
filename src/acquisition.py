"""
Live data collector: persists Binance's raw `<symbol>@depth@100ms` diff stream,
`<symbol>@trade` stream, and periodic REST order-book snapshots to disk.

Deliberate scope: this module does NOT reconstruct the order book or detect
sequence gaps. It only captures the wire data as faithfully as possible. Gap
detection and book reconstruction happen later, offline, via
`orderbook.reconstruct_book_states`, reusing the exact same tested
`resync_and_replay` logic from orderbook.py. Two reasons for this split:

  1. `resync_and_replay`'s resync step needs to fetch a snapshot to bridge a gap.
     Binance's REST endpoint only ever returns the *current* book -- it cannot
     answer "what did the book look like right after the gap that happened three
     days ago." So gap-bridging snapshots must already exist on disk, captured
     periodically *during* the live run, for the offline pass to splice in. This
     collector's job is exactly to make sure those snapshots exist; it is not
     required to reconstruct anything itself.
  2. Keeping the live collector dead simple (append raw bytes, don't compute
     anything) minimizes the amount of code that can break during an unattended
     multi-day run. If a reconstruction bug is later found, we replay raw data
     through a fixed orderbook.py -- we never have to re-collect.

Binance's own reference for the reconstruction procedure this raw data feeds into:
https://developers.binance.com/docs/binance-spot-api-docs/web-socket-streams
("How to manage a local order book correctly")
"""

from __future__ import annotations

import argparse
import asyncio
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import AsyncIterator, Callable, Optional

import requests
import websockets

# Binance.com blocks all US IPs (HTTP 451) regardless of account status -- this
# is a network-level geo-block, not something fixable in code. Binance.US is a
# close API mirror (same REST/WS message shapes) built for US users, so we keep
# both venues available and only swap base URLs; none of the reconstruction
# logic in orderbook.py needs to know which venue the data came from.
VENUES = {
    "binance": {
        "rest_base": "https://api.binance.com",
        "ws_base": "wss://stream.binance.com:9443/ws",
    },
    "binance-us": {
        "rest_base": "https://api.binance.us",
        "ws_base": "wss://stream.binance.us:9443/ws",
    },
}


def fetch_snapshot_raw(symbol: str, rest_base: str, limit: int = 1000) -> dict:
    """One REST call to a Binance-family order-book snapshot endpoint. Returns
    the raw JSON dict unmodified (not our `orderbook.Snapshot` type) so the
    caller can persist the exact wire response -- this raw dump is what offline
    reconstruction will later parse and feed to `resync_and_replay` as a
    gap-bridging snapshot."""
    resp = requests.get(
        f"{rest_base}/api/v3/depth",
        params={"symbol": symbol.upper(), "limit": limit},
        timeout=10,
    )
    resp.raise_for_status()
    return resp.json()


async def _stream_raw_messages(
    url: str, on_reconnect: Callable[[str], None]
) -> AsyncIterator[dict]:
    """Yields parsed JSON messages from a single Binance raw WS stream, using
    `websockets`' built-in auto-reconnect iterator (see the `websockets` docs,
    "Design > Automatic reconnection"): `async for ws in websockets.connect(url)`
    transparently reconnects with backoff whenever the connection drops.

    A reconnect means messages may have been missed while disconnected. We don't
    try to detect or fix that here (see module docstring) -- we just log it, so
    an unattended multi-day run gives some live visibility, while the *actual*
    gap detection happens offline from the `pu` sequence numbers in the captured
    data itself, which is authoritative regardless of what this log says.
    """
    first_connection = True
    async for ws in websockets.connect(url, ping_interval=20, ping_timeout=20):
        if not first_connection:
            on_reconnect(f"reconnected to {url}")
        first_connection = False
        try:
            async for message in ws:
                yield json.loads(message)
        except websockets.ConnectionClosed:
            continue  # the outer `async for ws in ...` will reconnect for us


def stream_depth_events(
    symbol: str, ws_base: str, on_reconnect: Callable[[str], None]
) -> AsyncIterator[dict]:
    url = f"{ws_base}/{symbol.lower()}@depth@100ms"
    return _stream_raw_messages(url, on_reconnect)


def stream_trade_events(
    symbol: str, ws_base: str, on_reconnect: Callable[[str], None]
) -> AsyncIterator[dict]:
    url = f"{ws_base}/{symbol.lower()}@trade"
    return _stream_raw_messages(url, on_reconnect)


class HourlyRotatingWriter:
    """Append-only JSONL writer, rotating to a new file whenever the UTC hour
    (of local receipt time) changes. Hourly rotation bounds file size and means
    a crash mid-run only risks losing the current partial hour, not the whole
    capture. Line-buffered so each record is flushed promptly -- a line that's
    torn by an abrupt kill is simply unparseable and gets skipped when the raw
    data is read back, per JSONL's natural crash-safety.
    """

    def __init__(self, out_dir: Path, name: str):
        self.out_dir = Path(out_dir)
        self.name = name
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self._current_hour: Optional[str] = None
        self._fh = None

    def write(self, record: dict) -> None:
        hour_str = datetime.fromtimestamp(
            record["_recv_time_ms"] / 1000, tz=timezone.utc
        ).strftime("%Y-%m-%d_%H")
        if hour_str != self._current_hour:
            if self._fh is not None:
                self._fh.close()
            self._current_hour = hour_str
            path = self.out_dir / f"{self.name}_{hour_str}.jsonl"
            self._fh = open(path, "a", buffering=1)
        self._fh.write(json.dumps(record) + "\n")

    def close(self) -> None:
        if self._fh is not None:
            self._fh.close()
            self._fh = None


async def run_collector(
    symbol: str,
    out_dir: Path,
    venue: str = "binance-us",
    snapshot_interval_s: float = 900.0,
) -> None:
    """Orchestrates three concurrent loops for the duration of the run:
      - depth diff stream -> data/raw/depth_events/
      - trade stream      -> data/raw/trades/
      - REST snapshot every `snapshot_interval_s` -> data/raw/snapshots/

    Snapshot cadence (default 15 min) is deliberately much more frequent than a
    live book would need for its own "defensive resync" -- because there is no
    live book here (see module docstring), the snapshots exist purely so
    offline reconstruction always has a recent anchor to splice in after any
    gap. The tighter the cadence, the less data has to be discarded around a
    gap (time between the gap and the next available snapshot is unreconstructable
    and must be dropped, never guessed at).
    """
    rest_base = VENUES[venue]["rest_base"]
    ws_base = VENUES[venue]["ws_base"]
    out_dir = Path(out_dir)
    depth_writer = HourlyRotatingWriter(out_dir / "depth_events", "depth")
    trade_writer = HourlyRotatingWriter(out_dir / "trades", "trade")
    snapshot_dir = out_dir / "snapshots"
    snapshot_dir.mkdir(parents=True, exist_ok=True)
    collector_log = out_dir / "collector.log"

    def log_event(msg: str) -> None:
        line = f"{datetime.now(timezone.utc).isoformat()} {msg}"
        print(line, flush=True)
        with open(collector_log, "a") as f:
            f.write(line + "\n")

    async def snapshot_loop() -> None:
        while True:
            try:
                raw = await asyncio.to_thread(fetch_snapshot_raw, symbol, rest_base)
                ts = datetime.now(timezone.utc)
                path = snapshot_dir / f"snapshot_{ts.strftime('%Y-%m-%dT%H-%M-%S')}.json"
                path.write_text(json.dumps(raw))
                log_event(f"[snapshot] saved lastUpdateId={raw['lastUpdateId']}")
            except Exception as exc:  # noqa: BLE001 -- log and keep collecting
                log_event(f"[snapshot] fetch failed: {exc!r}")
            await asyncio.sleep(snapshot_interval_s)

    async def depth_loop() -> None:
        # `_stream_raw_messages` already reconnects forever on a *clean*
        # disconnect. This outer try/except is a second line of defense against
        # anything unexpected (e.g. a malformed message) -- over a multi-day
        # unattended run, one uncaught exception in a single task would
        # otherwise take down `asyncio.gather` and silently kill the whole
        # collector, including the trade stream and snapshot loop.
        count = 0
        while True:
            try:
                async for msg in stream_depth_events(
                    symbol, ws_base, on_reconnect=lambda m: log_event(f"[depth] {m}")
                ):
                    msg["_recv_time_ms"] = int(time.time() * 1000)
                    depth_writer.write(msg)
                    count += 1
                    if count % 50_000 == 0:
                        log_event(f"[depth] {count} events captured")
            except Exception as exc:  # noqa: BLE001
                log_event(f"[depth] loop crashed: {exc!r}; restarting in 5s")
                await asyncio.sleep(5)

    async def trade_loop() -> None:
        count = 0
        while True:
            try:
                async for msg in stream_trade_events(
                    symbol, ws_base, on_reconnect=lambda m: log_event(f"[trade] {m}")
                ):
                    msg["_recv_time_ms"] = int(time.time() * 1000)
                    trade_writer.write(msg)
                    count += 1
                    if count % 10_000 == 0:
                        log_event(f"[trade] {count} events captured")
            except Exception as exc:  # noqa: BLE001
                log_event(f"[trade] loop crashed: {exc!r}; restarting in 5s")
                await asyncio.sleep(5)

    log_event(f"collector starting for {symbol}, writing to {out_dir}")
    try:
        await asyncio.gather(snapshot_loop(), depth_loop(), trade_loop())
    finally:
        depth_writer.close()
        trade_writer.close()
        log_event("collector stopped")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Collect Binance L2 depth + trade data for OFI research."
    )
    parser.add_argument("--symbol", default="BTCUSDT")
    parser.add_argument("--out", default="data/raw")
    parser.add_argument(
        "--venue",
        choices=list(VENUES.keys()),
        default="binance-us",
        help="which Binance-family API to hit (default: binance-us, since binance.com blocks US IPs)",
    )
    parser.add_argument(
        "--snapshot-interval",
        type=float,
        default=900.0,
        help="seconds between REST order-book snapshot dumps (default: 900 = 15 min)",
    )
    args = parser.parse_args()

    try:
        asyncio.run(
            run_collector(
                args.symbol, Path(args.out), args.venue, args.snapshot_interval
            )
        )
    except KeyboardInterrupt:
        print("\nStopping collector (Ctrl+C received)...")


if __name__ == "__main__":
    main()
