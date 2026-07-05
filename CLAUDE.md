# Project: Order Flow Imbalance & Short-Horizon Return Predictability

## Goal
Rigorous quantitative research artifact (SSRN-style paper + clean code) testing
whether order flow imbalance (OFI) predicts short-horizon BTC/USDT mid-price returns.
This is research, NOT a trading bot. Rigor > headline results.

## Hypothesis
OFI has statistically significant predictive power at sub-10s horizons
(out-of-sample R² > 0, robust to Newey-West HAC SEs), with R² decaying
monotonically as horizon increases.

## Methodology grounding
OFI defined per Cont, Kukanov & Stoikov (2014). Event-based, aggregated over
fixed windows. Include depth-weighted top-5-level variant.

## Non-negotiable standards
- Walk-forward validation ONLY. Never shuffle time-series data.
- Flag any potential lookahead bias / data leakage explicitly.
- Small honest R² (fractions of a % to a few %) is expected and correct.
  Warn if any result looks suspiciously high — usually a bug or leakage.
- Use proper methods: Newey-West for autocorrelated errors, block bootstrap
  for time-series CIs. Explain every statistical choice.
- Comment code so every line is interview-defensible.

## Stack
Python. pandas, numpy, statsmodels, scipy, matplotlib. Avoid heavy frameworks.

## Structure
- src/acquisition.py   — data download / collection
- src/orderbook.py     — L2 book reconstruction from diff stream
- src/features.py      — OFI computation, targets
- src/modeling.py      — regressions, baselines
- src/evaluation.py    — walk-forward, bootstrap, metrics
- notebooks/           — exploration
- paper/               — the writeup

## Commands
- `python -m src.acquisition`  — pull data
- `pytest`                     — run tests

## Workflow
- Build stage by stage. Verify each stage before moving on.
- Write a test or sanity check for each module before proceeding.