"""
Stage 2: order-flow imbalance (OFI) feature computation and forward-return
targets, built directly on the event-level book-state table produced by
src/orderbook.py (one row per successfully applied diff event, top-N levels).

OFI definition (Cont, Kukanov & Stoikov 2014, "The Price Impact of Order Book
Events", eq. 1-2), applied per level and per successive event n:

    bid side:  price up   -> +q^B_n            (a better bid appeared)
               price flat -> +(q^B_n - q^B_{n-1})  (size added/removed at the touch)
               price down -> -q^B_{n-1}          (the old best bid was pulled)

    ask side is the mirror image with the outcome sign flipped, because an
    ask *strengthening* (price down -- a more aggressive seller) is bearish
    while a bid strengthening is bullish:

               price up   -> -q^A_{n-1}
               price flat -> +(q^A_n - q^A_{n-1})
               price down -> +q^A_n

    level OFI_n = bid_delta_n - ask_delta_n

`ofi_best` is this formula applied at the top of book (level 1) exactly as in
the paper. `ofi_multilevel` is a depth-weighted extension across the top 5
levels (not part of the original paper, which only defines level 1): level i
is weighted 1/i, so the touch dominates but deeper levels still contribute --
deeper levels are noisier/less informative about imminent price impact, hence
the decay rather than an equal-weighted sum or dropping them outright. This
weighting choice is a design decision, not a literature-standard constant --
call it out as such if asked.

Two invariants downstream code depends on, both load-bearing for avoiding
lookahead bias (see CLAUDE.md):

  1. A row's OFI is a delta against the *previous row*, matched by depth rank
     (level i vs level i), not by price identity. Resync rows (see
     orderbook.reconstruct_book_states) have no real predecessor -- their
     delta would be against a stale pre-gap state -- so their OFI is forced
     to NaN rather than computed.
  2. Window aggregation and target computation never use information from
     after the window they're attached to: a window's OFI sums only events
     whose timestamp falls inside it, and its forward-return target is
     computed from mid-price at the window's end vs. a strictly later
     window's end. Nothing between "now" and "now + horizon" leaks into the
     feature.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def load_book_states(processed_dir: Path) -> pd.DataFrame:
    """Loads every day-partitioned Parquet file written by
    reconstruct_book_states and concatenates them into one time-sorted table."""
    processed_dir = Path(processed_dir)
    files = sorted(processed_dir.glob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"no book-state parquet files under {processed_dir}")
    df = pd.concat([pd.read_parquet(f) for f in files])
    return df.sort_index()


def compute_event_ofi(book_df: pd.DataFrame, levels: int = 5) -> pd.DataFrame:
    """Adds `ofi_best` (level-1 OFI) and `ofi_multilevel` (1/i-weighted sum
    across the top `levels` levels) columns, comparing each row to the row
    immediately before it (by depth rank, per the module docstring).

    A level's contribution is NaN wherever that level isn't defined on either
    the current or previous row (e.g. early in the capture, or a thin book
    with fewer than `levels` real levels on one side) -- it's excluded from
    the multilevel sum (treated as "no observed event there", i.e. 0) rather
    than poisoning the whole row's feature with NaN just because a deep,
    largely irrelevant level was momentarily absent.
    """
    df = book_df.copy()
    multilevel = pd.Series(0.0, index=df.index)
    any_level_defined = pd.Series(False, index=df.index)

    for i in range(1, levels + 1):
        bid_p, bid_q = df[f"bid_price_{i}"], df[f"bid_size_{i}"]
        ask_p, ask_q = df[f"ask_price_{i}"], df[f"ask_size_{i}"]
        prev_bid_p, prev_bid_q = bid_p.shift(1), bid_q.shift(1)
        prev_ask_p, prev_ask_q = ask_p.shift(1), ask_q.shift(1)

        level_defined = (
            bid_p.notna() & prev_bid_p.notna() & ask_p.notna() & prev_ask_p.notna()
        )

        bid_delta = np.select(
            [bid_p > prev_bid_p, bid_p == prev_bid_p],
            [bid_q, bid_q - prev_bid_q],
            default=-prev_bid_q,
        )
        ask_delta = np.select(
            [ask_p > prev_ask_p, ask_p == prev_ask_p],
            [-prev_ask_q, ask_q - prev_ask_q],
            default=ask_q,
        )
        # Comparisons above are False (not an error) when a price is NaN, so
        # the arithmetic can produce garbage for undefined levels -- masked
        # out by level_defined immediately below before it touches anything.
        level_ofi = pd.Series(bid_delta - ask_delta, index=df.index).where(
            level_defined
        )

        if i == 1:
            df["ofi_best"] = level_ofi
        weight = 1.0 / i
        multilevel = multilevel + level_ofi.fillna(0.0) * weight
        any_level_defined = any_level_defined | level_defined

    df["ofi_multilevel"] = multilevel.where(any_level_defined)

    # Resync rows' "previous row" is unrelated pre-gap state -- see module
    # docstring invariant (1). Overrides whatever the shift-based formula
    # above computed for them, including a spuriously-defined level_defined.
    df.loc[df["resync"], ["ofi_best", "ofi_multilevel"]] = np.nan
    return df


def aggregate_windows(event_df: pd.DataFrame, window_seconds: float) -> pd.DataFrame:
    """Sums event-level OFI into fixed, non-overlapping (tumbling) time
    windows of width `window_seconds`, per CLAUDE.md's "event-based,
    aggregated over fixed windows" spec.

    Windows with zero events (possible but rare given ~4 events/sec observed
    in this capture) get OFI=0 (no flow observed -- correct, not missing) and
    their mid_price forward-filled from the last real observation, flagged in
    `mid_price_filled` so staleness there is visible rather than silent.
    """
    freq = pd.Timedelta(seconds=window_seconds)
    bin_start = event_df.index.floor(freq)
    grouped = event_df.groupby(bin_start)

    out = pd.DataFrame(
        {
            "ofi_best": grouped["ofi_best"].sum(),
            "ofi_multilevel": grouped["ofi_multilevel"].sum(),
            "mid_price": grouped["mid_price"].last(),
            "n_events": grouped.size(),
        }
    )

    full_index = pd.date_range(out.index.min(), out.index.max(), freq=freq)
    out = out.reindex(full_index)
    out["mid_price_filled"] = out["mid_price"].isna()
    out["mid_price"] = out["mid_price"].ffill()
    out["ofi_best"] = out["ofi_best"].fillna(0.0)
    out["ofi_multilevel"] = out["ofi_multilevel"].fillna(0.0)
    out["n_events"] = out["n_events"].fillna(0).astype(int)
    out.index.name = "window_start"
    return out


def compute_targets(
    window_df: pd.DataFrame, window_seconds: float, horizons_seconds: list[float]
) -> pd.DataFrame:
    """Adds one forward log-return column per horizon:
    fwd_ret_{h}s = log(mid_price[t + h]) - log(mid_price[t])

    No-lookahead by construction: a window's OFI covers events up to and
    including its own end, and this target only ever reaches for mid_price at
    the *current* window and a strictly later one -- never anything from
    between them that wasn't already summarized into an earlier window's OFI.
    The trailing `horizon / window_seconds` rows of each column are NaN (no
    future data exists yet to compute them) rather than filled -- callers
    must drop NaNs per-horizon, not impute.

    Horizons must be positive integer multiples of window_seconds so each one
    lands exactly on a window boundary instead of interpolating.
    """
    out = window_df.copy()
    for h in horizons_seconds:
        shift = h / window_seconds
        if shift != round(shift) or shift < 1:
            raise ValueError(
                f"horizon {h}s must be a positive integer multiple of "
                f"window_seconds={window_seconds}s"
            )
        future_mid = out["mid_price"].shift(-int(round(shift)))
        out[f"fwd_ret_{h:g}s"] = np.log(future_mid) - np.log(out["mid_price"])
    return out


def build_feature_table(
    processed_dir: Path,
    window_seconds: float,
    horizons_seconds: list[float],
    depth_levels: int = 5,
) -> pd.DataFrame:
    book = load_book_states(processed_dir)
    events = compute_event_ofi(book, levels=depth_levels)
    windows = aggregate_windows(events, window_seconds)
    return compute_targets(windows, window_seconds, horizons_seconds)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Stage 2: compute OFI features and forward-return targets "
        "from reconstructed book states."
    )
    parser.add_argument("--processed", type=Path, default=Path("data/processed/book"))
    parser.add_argument("--out", type=Path, default=Path("data/processed/features"))
    parser.add_argument("--window-seconds", type=float, default=1.0)
    parser.add_argument(
        "--horizons-seconds", type=float, nargs="+", default=[1.0, 2.0, 5.0, 10.0]
    )
    parser.add_argument("--depth-levels", type=int, default=5)
    args = parser.parse_args()

    df = build_feature_table(
        args.processed, args.window_seconds, args.horizons_seconds, args.depth_levels
    )
    args.out.mkdir(parents=True, exist_ok=True)
    out_path = args.out / f"features_{int(args.window_seconds * 1000)}ms.parquet"
    df.to_parquet(out_path)
    print(f"wrote {len(df)} rows ({df.index[0]}..{df.index[-1]}) to {out_path}")


if __name__ == "__main__":
    main()
