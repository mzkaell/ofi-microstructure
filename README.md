# Order Flow Imbalance & Short-Horizon Return Predictability

A rigorous, hypothesis-driven research project testing whether **order flow imbalance (OFI)** —
the net signed volume arriving at the best bid/ask — predicts short-horizon (1-10s) mid-price
returns in BTC/USDT. This is a research artifact, not a trading bot: the goal is honest,
statistically defensible measurement of a small effect, not a headline number.

> **Hypothesis.** OFI has statistically significant predictive power at sub-10s horizons
> (out-of-sample R² > 0, robust to Newey-West HAC standard errors), with predictive power
> decaying as the horizon increases. A small R² (fractions of a percent to a few percent) is
> the expected, *correct* outcome — real microstructure edges are tiny, and a suspiciously large
> result is treated as a bug/leakage signal, not a win (see [Status](#status--roadmap)).

## Methodology grounding

OFI is defined per **Cont, Kukanov & Stoikov (2014)**, *"The Price Impact of Order Book
Events"*: event-based, computed from the signed change in size/price at the best bid and ask on
every book update, aggregated over fixed time windows. This project implements the level-1
formula exactly as specified in the paper, plus a depth-weighted 5-level extension (design
choice, not from the paper — see [`src/features.py`](src/features.py)).

## Pipeline

```
Stage 1  Acquisition & book reconstruction   src/acquisition.py, src/orderbook.py
Stage 2  OFI features & forward targets      src/features.py
Stage 3  Regression models                   src/modeling.py
Stage 4  Walk-forward validation & inference src/evaluation.py
Stage 5  Results & write-up                  paper/  (not started yet)
```

**1. Acquisition & book reconstruction.** A resilient asyncio collector
([`acquisition.py`](src/acquisition.py)) persists Binance's raw `depth@100ms` diff stream,
`trade` stream, and periodic REST snapshots to disk, untouched — it does no reconstruction
itself. Reconstruction ([`orderbook.py`](src/orderbook.py)) replays that raw capture offline
through `LocalOrderBook` + `resync_and_replay`, implementing Binance's documented
buffer/snapshot/sequence-ID algorithm for keeping a local book in sync, including gap detection
that forces a resync from a fresh snapshot rather than silently continuing on corrupted state.
The raw layer is immutable and the processed layer is always rebuildable from it — a
reconstruction bug means re-running this step, never re-collecting.

**2. Feature engineering.** [`features.py`](src/features.py) computes event-level OFI
(level-1 and depth-weighted multilevel), aggregates it into fixed non-overlapping time windows,
and computes forward log-mid-price-return targets at each horizon. Every lookahead-bias
boundary is explicit and tested: resync rows get `NaN` OFI (no real predecessor to delta
against), and a window's target only ever reaches forward in time, never backward into
information already summarized elsewhere.

**3. Modeling.** [`modeling.py`](src/modeling.py) fits OLS (OFI → forward return) and defines
the benchmark forecast: a **training-mean baseline**, not "predict zero" — out-of-sample R² is
only meaningful relative to a stated benchmark (Campbell & Thompson 2008), and predicting zero
silently assumes the true unconditional return is exactly zero.

**4. Validation.** [`evaluation.py`](src/evaluation.py) runs strict **walk-forward**
validation — chronological, expanding-window, never shuffled — computes out-of-sample R² against
the training-mean benchmark, **Newey-West HAC** significance tests (necessary because both OFI
and short-horizon returns are autocorrelated, which understates ordinary OLS standard errors),
and **moving block bootstrap** confidence intervals on R² (a plain i.i.d. bootstrap would
understate sampling variability for serially-dependent data).

[`scripts/run_ofi_study.py`](scripts/run_ofi_study.py) wires all of this into one end-to-end run
and writes a results table.

## Status & Roadmap

**Built and tested (42 passing unit tests):**
- [x] Live collector with automatic reconnect/backoff and crash-safe append-only storage
- [x] Order-book reconstruction with gap detection and resync (see [Data Integrity](#data-integrity-what-went-wrong-and-how-it-was-caught) below)
- [x] Event-level and depth-weighted OFI, fixed-window aggregation, forward-return targets
- [x] OLS regression vs. a training-mean baseline
- [x] Walk-forward out-of-sample R², Newey-West HAC significance, block-bootstrap CIs

**Not yet built:**
- [ ] AR(1)-on-returns and signed-trade-flow baselines (to show OFI adds information beyond
      naive momentum/mean-reversion and beyond simple trade flow)
- [ ] Ridge regression over the multi-level OFI feature set
- [ ] Purge/embargo windows at walk-forward fold boundaries (Lopez de Prado-style), so no
      feature/target pair straddles a train/test split
- [ ] Newey-West-robust Wald test for multi-level vs. single-level OFI
- [ ] The full multi-day (5-10 day) capture and a completed run of the pipeline end-to-end
- [ ] The written research paper (`paper/`)

**Results are intentionally not reported here yet.** A README claiming numbers before the full
pipeline has actually been run on the full dataset is exactly the kind of thing this project's
own standards (see [`CLAUDE.md`](CLAUDE.md)) exist to prevent. Once Stage 4 runs end-to-end on
the complete capture, this section will report the R²-vs-horizon table with bootstrap CIs.

## Data Integrity: what went wrong (and how it was caught)

Two real data-quality issues surfaced during this project, both worth documenting explicitly
rather than glossing over:

1. **Binance.com geo-blocks all US IPs** (HTTP 451) at the network level, independent of
   account status. The collector supports switching venues (`--venue binance-us`), which mirrors
   binance.com's API closely enough that no reconstruction logic needed to change.
2. **Gap detection was silently disabled on Binance.US.** Binance's local-book algorithm
   normally validates sequence continuity via each event's `pu` (previous update ID) field. The
   original implementation treated a missing `pu` as "nothing to check" — correct for the one
   event where that's actually expected (the first event applied after a snapshot), but
   Binance.US never populates `pu` on *any* event, so every dropped or reordered message went
   undetected across the entire feed. This let stale price levels linger on one side of the book
   while the other kept moving, **crossing the spread on ~11% of reconstructed rows** in the
   affected capture. Fixed by falling back to Binance's classic `U`-continuity check when `pu` is
   absent (see the `test_gap_is_caught_via_U_when_pu_is_always_absent` test), and caught in the
   first place by the spread-positivity sanity check this project runs on every capture — exactly
   the kind of invariant check the [`scripts/sanity_check.py`](scripts/sanity_check.py) module
   exists to catch.

## Repository structure

```
src/
  acquisition.py     WebSocket collector + REST snapshot fetch, CLI entrypoint
  orderbook.py        LocalOrderBook state machine, resync/replay, batch reconstruction
  features.py          OFI computation, window aggregation, forward-return targets
  modeling.py          OLS fit, training-mean baseline
  evaluation.py        walk-forward splits, OOS R², Newey-West, block bootstrap
scripts/
  sanity_check.py       Stage 1 data-quality report (spreads, gaps, latency, plots)
  run_ofi_study.py       end-to-end Stage 2-4 run, writes reports/results.csv
tests/                 42 tests, one file per src/ module
data/
  raw/                 immutable capture (gitignored — regenerate via acquisition.py)
  processed/           reconstructed book states + features (gitignored — rebuildable from raw)
notebooks/             exploration
paper/                 write-up (not started)
reports/sanity/        sanity-check output (plots + summary.json)
CLAUDE.md              project standards/spec this codebase is held to
```

## Setup

```bash
git clone https://github.com/mzkaell/ofi-microstructure.git
cd ofi-microstructure
python -m venv .venv
.venv\Scripts\activate        # Windows; use `source .venv/bin/activate` on macOS/Linux
pip install -r requirements.txt
pytest tests/ -v               # all 42 should pass
```

## Usage

```bash
# 1. Collect raw data (run for several days; survives disconnects/reboots)
python -m src.acquisition --out data/raw --venue binance-us

# 2. Reconstruct the order book from the raw capture (rerun anytime; deterministic)
python -m src.orderbook --raw data/raw --out data/processed/book

# 3. Check data quality (spreads, gaps, message rate, latency, mid-price plots)
python -m scripts.sanity_check

# 4. Compute OFI features and forward-return targets
python -m src.features --processed data/processed/book --out data/processed/features

# 5. Run the full walk-forward study (OOS R², Newey-West, bootstrap CIs)
python -m scripts.run_ofi_study --processed data/processed/book --out reports/results.csv
```

## Tech stack

Python · pandas · NumPy · statsmodels · SciPy · matplotlib · asyncio · websockets ·
Parquet/PyArrow · pytest · Binance WebSocket/REST APIs

## References

- Cont, R., Kukanov, A., & Stoikov, S. (2014). *The Price Impact of Order Book Events.*
  Journal of Financial Econometrics.
- Campbell, J. Y., & Thompson, S. B. (2008). *Predicting Excess Stock Returns Out of Sample.*
  Review of Financial Studies.
- Newey, W. K., & West, K. D. (1987). *A Simple, Positive Semi-Definite, Heteroskedasticity and
  Autocorrelation Consistent Covariance Matrix.* Econometrica.
- Künsch, H. R. (1989). *The Jackknife and the Bootstrap for General Stationary Observations.*
  Annals of Statistics.
