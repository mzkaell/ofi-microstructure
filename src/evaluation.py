"""
Stage 4: walk-forward validation, out-of-sample R^2, Newey-West HAC
significance testing, and block-bootstrap confidence intervals.

This module is where CLAUDE.md's non-negotiable standards actually get
enforced in code, so each one gets a paragraph:

Walk-forward only, never shuffled: `walk_forward_splits` yields purely
chronological (train, test) index pairs with an *expanding* training window
-- every test fold is predicted using only data strictly earlier in time.
There is no k-fold/shuffled CV anywhere in this module; adding one would
leak future order-flow/price information into training and silently invent
predictability that isn't real.

Out-of-sample R^2 needs a stated benchmark: `oos_r2` computes the Campbell &
Thompson (2008) R^2 = 1 - SSE(model)/SSE(benchmark), where the benchmark is
the training-mean baseline from modeling.py, not zero. See modeling.py's
docstring for why.

Newey-West HAC: `newey_west_significance` fits OLS on the full sample with a
heteroskedasticity-and-autocorrelation-consistent covariance matrix (Newey &
West 1987). This is necessary, not optional, because both OFI and short-
horizon returns are autocorrelated (order flow clusters; horizons longer than
the sampling window make consecutive targets overlap), which understates
ordinary OLS standard errors and would overstate significance. `maxlags`
should be set to at least the horizon-to-window ratio (e.g. horizon=10s over
a 1s window needs maxlags>=10) so the overlap-induced autocorrelation is
actually covered.

Purge/embargo: `walk_forward_splits` also trims the trailing edge of each
fold's training set (Lopez de Prado, *Advances in Financial Machine
Learning*, ch. 7) so that no training row's forward-looking label reaches
into that fold's test period, plus an additional embargo buffer against
residual serial correlation. See its docstring for the exact mechanics and
why only one boundary needs trimming in a pure forward-chaining design.

Block bootstrap: `block_bootstrap_r2_ci` resamples the out-of-sample
(y_true, y_pred, y_pred_baseline) triples in contiguous blocks (not
single points) so the resampled series preserves the original's short-range
serial dependence -- a standard i.i.d. bootstrap would understate the true
sampling variability of R^2 for autocorrelated time series (Kunsch 1989
moving block bootstrap).

Small honest R^2 (fractions of a % to a low single-digit %) is the expected,
correct outcome here -- CLAUDE.md flags suspiciously high results as a bug/
leakage signal, not a win.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import statsmodels.api as sm

from src.modeling import fit_ols, predict_historical_mean


def walk_forward_splits(
    n_obs: int,
    n_splits: int,
    min_train_fraction: float = 0.5,
    purge: int = 0,
    embargo: int = 0,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Yields (train_idx, test_idx) *positional* index arrays for `n_splits`
    chronological, non-overlapping, contiguous test folds covering the tail
    (1 - min_train_fraction) of the sample. Each fold's training window
    expands to include everything strictly before that fold's test period --
    fold k+1's training data is a superset of fold k's, plus fold k's test
    period, exactly mimicking how a researcher re-fits as more history
    becomes available. No fold's train_idx ever contains an index >= any of
    its own test_idx.

    `purge` and `embargo` (Lopez de Prado, *Advances in Financial Machine
    Learning*, ch. 7) trim the trailing edge of each fold's training set,
    immediately before test_start:

    - `purge`: the number of trailing training rows to drop because their
      *label* (a forward-looking target `h` bins ahead) reaches into the
      test period even though the row itself sits before it. A training row
      at position `test_start - 1` with an h-bin-ahead target technically
      has its label computed from data inside the test window -- fitting on
      it leaks test-period price information into training. Callers should
      set `purge = horizon_bins - 1` (the exact number of trailing rows
      whose target overlaps test_start) for whichever horizon they're
      evaluating.
    - `embargo`: an additional buffer of trailing training rows dropped
      beyond the purge, as defense-in-depth against residual short-range
      serial correlation between training and test observations that label
      purging alone doesn't address (e.g. a feature's own autocorrelation
      structure, not just its label's literal overlap).

    Both only ever trim the *start-of-test* boundary here, not a
    post-test boundary: unlike k-fold CV, this is a pure forward-chaining
    walk-forward where a fold's training data is always strictly earlier in
    time than its test data, so there is no "training data after test" for
    an embargo on the far side to protect against -- when fold k's test
    period is absorbed into fold k+1's training set, the *same* purge/embargo
    trim at fold k+1's own test boundary already covers it.
    """
    if n_splits < 1:
        raise ValueError("n_splits must be >= 1")
    if purge < 0 or embargo < 0:
        raise ValueError("purge and embargo must be >= 0")
    min_train = int(n_obs * min_train_fraction)
    remaining = n_obs - min_train
    fold_size = remaining // n_splits
    if fold_size < 1:
        raise ValueError(
            f"not enough observations ({n_obs}) for {n_splits} folds with "
            f"min_train_fraction={min_train_fraction}"
        )
    trim = purge + embargo
    splits = []
    for k in range(n_splits):
        test_start = min_train + k * fold_size
        test_end = n_obs if k == n_splits - 1 else test_start + fold_size
        train_end = max(0, test_start - trim)
        if train_end < 1:
            raise ValueError(
                f"purge+embargo={trim} leaves no training data before fold "
                f"{k}'s test_start={test_start}"
            )
        splits.append((np.arange(0, train_end), np.arange(test_start, test_end)))
    return splits


def run_walk_forward(
    df: pd.DataFrame,
    feature_cols: list[str],
    target_col: str,
    n_splits: int,
    min_train_fraction: float = 0.5,
    purge: int = 0,
    embargo: int = 0,
) -> pd.DataFrame:
    """Runs the full walk-forward procedure and returns one pooled DataFrame
    of out-of-sample rows (across all folds, concatenated in time order) with
    columns y_true, y_pred_model, y_pred_baseline. `df` must already be
    sorted by time (window_start), which build_feature_table guarantees.
    `purge`/`embargo` are forwarded to walk_forward_splits -- see its
    docstring for what each protects against.
    """
    assert df.index.is_monotonic_increasing, "df must be time-sorted"
    X, y = df[feature_cols], df[target_col]
    oos_chunks = []
    for train_idx, test_idx in walk_forward_splits(
        len(df), n_splits, min_train_fraction, purge=purge, embargo=embargo
    ):
        X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
        X_test, y_test = X.iloc[test_idx], y.iloc[test_idx]

        model = fit_ols(X_train, y_train, feature_cols)
        y_pred_model = model.predict(X_test)
        y_pred_baseline = predict_historical_mean(y_train, X_test.index)

        oos_chunks.append(
            pd.DataFrame(
                {
                    "y_true": y_test,
                    "y_pred_model": y_pred_model,
                    "y_pred_baseline": y_pred_baseline,
                }
            )
        )
    return pd.concat(oos_chunks)


def oos_r2(y_true: pd.Series, y_pred: pd.Series, y_pred_baseline: pd.Series) -> float:
    """Campbell & Thompson (2008) out-of-sample R^2 against a stated
    benchmark forecast, computed on rows where all three are non-NaN."""
    panel = pd.DataFrame(
        {"y_true": y_true, "y_pred": y_pred, "y_pred_baseline": y_pred_baseline}
    ).dropna()
    sse_model = ((panel["y_true"] - panel["y_pred"]) ** 2).sum()
    sse_baseline = ((panel["y_true"] - panel["y_pred_baseline"]) ** 2).sum()
    if sse_baseline == 0:
        raise ValueError("benchmark SSE is 0 -- cannot compute R^2 against it")
    return 1.0 - sse_model / sse_baseline


def newey_west_significance(
    X: pd.DataFrame, y: pd.Series, feature_cols: list[str], maxlags: int
):
    """Fits OLS of y on feature_cols + intercept over the FULL sample with a
    Newey-West HAC covariance matrix, for testing whether the in-sample
    relationship between OFI and returns is statistically significant once
    autocorrelation is accounted for. This is a separate question from
    predictive out-of-sample R^2 (run_walk_forward/oos_r2) -- a coefficient
    can be significant in-sample yet still fail to generalize out-of-sample,
    or vice versa with a small sample; report both, don't conflate them.

    Returns the fitted statsmodels results object -- .params, .bse,
    .tvalues, .pvalues are all already HAC-adjusted.
    """
    panel = pd.concat([X[feature_cols], y.rename("__y__")], axis=1).dropna()
    X_ = sm.add_constant(panel[feature_cols], has_constant="add")
    return sm.OLS(panel["__y__"], X_).fit(
        cov_type="HAC", cov_kwds={"maxlags": maxlags}
    )


@dataclass
class BootstrapResult:
    point_estimate: float
    ci_low: float
    ci_high: float
    n_boot: int
    block_size: int


def block_bootstrap_r2_ci(
    y_true: pd.Series,
    y_pred: pd.Series,
    y_pred_baseline: pd.Series,
    block_size: int,
    n_boot: int = 1000,
    ci: float = 0.95,
    seed: int = 0,
) -> BootstrapResult:
    """Moving block bootstrap (Kunsch 1989) confidence interval for the
    out-of-sample R^2. Resamples contiguous blocks of `block_size`
    consecutive observations (with replacement, preserving the (y_true,
    y_pred, y_pred_baseline) pairing within each block) rather than single
    points, so short-range serial correlation in the OOS residuals is
    preserved in each bootstrap replicate rather than averaged away.
    """
    panel = pd.DataFrame(
        {"y_true": y_true, "y_pred": y_pred, "y_pred_baseline": y_pred_baseline}
    ).dropna()
    n = len(panel)
    if block_size < 1 or block_size > n:
        raise ValueError(f"block_size must be in [1, {n}], got {block_size}")
    point = oos_r2(panel["y_true"], panel["y_pred"], panel["y_pred_baseline"])

    rng = np.random.default_rng(seed)
    n_blocks = int(np.ceil(n / block_size))
    max_start = n - block_size
    r2_samples = np.empty(n_boot)
    values = panel.to_numpy()
    for b in range(n_boot):
        starts = rng.integers(0, max_start + 1, size=n_blocks)
        idx = np.concatenate([np.arange(s, s + block_size) for s in starts])[:n]
        resampled = values[idx]
        sse_model = np.sum((resampled[:, 0] - resampled[:, 1]) ** 2)
        sse_baseline = np.sum((resampled[:, 0] - resampled[:, 2]) ** 2)
        r2_samples[b] = 1.0 - sse_model / sse_baseline if sse_baseline != 0 else np.nan

    alpha = 1.0 - ci
    lo, hi = np.nanpercentile(r2_samples, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return BootstrapResult(
        point_estimate=point, ci_low=lo, ci_high=hi, n_boot=n_boot, block_size=block_size
    )
