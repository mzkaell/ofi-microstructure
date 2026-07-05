"""
Tests for the parts of src/acquisition.py that don't require a live network
connection -- the rotating writer's file-management logic. The WebSocket/REST
functions are exercised manually against the real Binance endpoints (see the
project verification steps), not here, since mocking them wouldn't tell us
anything about whether the real integration works.
"""

import json

from src.acquisition import HourlyRotatingWriter


def _ms(dt_str: str) -> int:
    """Helper: 'YYYY-MM-DD HH:MM:SS' (UTC) -> epoch milliseconds."""
    from datetime import datetime, timezone

    dt = datetime.strptime(dt_str, "%Y-%m-%d %H:%M:%S").replace(tzinfo=timezone.utc)
    return int(dt.timestamp() * 1000)


def test_writes_within_same_hour_go_to_one_file(tmp_path):
    writer = HourlyRotatingWriter(tmp_path, "depth")
    writer.write({"_recv_time_ms": _ms("2026-07-04 10:00:01"), "v": 1})
    writer.write({"_recv_time_ms": _ms("2026-07-04 10:59:59"), "v": 2})
    writer.close()

    files = list(tmp_path.glob("depth_*.jsonl"))
    assert len(files) == 1
    lines = files[0].read_text().strip().split("\n")
    assert len(lines) == 2
    assert json.loads(lines[0])["v"] == 1
    assert json.loads(lines[1])["v"] == 2


def test_rotates_to_new_file_on_hour_change(tmp_path):
    writer = HourlyRotatingWriter(tmp_path, "depth")
    writer.write({"_recv_time_ms": _ms("2026-07-04 10:59:59"), "v": 1})
    writer.write({"_recv_time_ms": _ms("2026-07-04 11:00:00"), "v": 2})
    writer.close()

    files = sorted(tmp_path.glob("depth_*.jsonl"))
    assert len(files) == 2
    assert "10" in files[0].name
    assert "11" in files[1].name


def test_close_is_idempotent(tmp_path):
    writer = HourlyRotatingWriter(tmp_path, "trade")
    writer.write({"_recv_time_ms": _ms("2026-07-04 00:00:00"), "v": 1})
    writer.close()
    writer.close()  # must not raise
