"""Tests for src/evaluation.py: fold chronology, out-of-sample R^2, Newey-West
significance, and block-bootstrap CIs, all against small synthetic panels."""

import numpy as np
import pandas as pd
import pytest

from src.evaluation import (
    block_bootstrap_r2_ci,
    newey_west_significance,
    oos_r2,
    run_walk_forward,
    walk_forward_splits,
)
from src.modeling import fit_ols


def test_walk_forward_splits_are_chronological_expanding_and_cover_the_tail():
    splits = walk_forward_splits(n_obs=100, n_splits=5, min_train_fraction=0.5)

    assert len(splits) == 5
    all_test_idx = []
    prev_train_size = 0
    for train_idx, test_idx in splits:
        # No future leakage: every train index precedes every test index.
        assert train_idx.max(initial=-1) < test_idx.min()
        # Expanding window: each fold's train set is a strict superset of the last.
        assert len(train_idx) >= prev_train_size
        prev_train_size = len(train_idx)
        all_test_idx.append(test_idx)

    all_test_idx = np.concatenate(all_test_idx)
    assert sorted(all_test_idx.tolist()) == list(range(50, 100))  # no gaps/overlaps
    assert len(all_test_idx) == len(set(all_test_idx.tolist()))  # no duplicates


def test_walk_forward_splits_raises_when_too_few_observations_for_folds():
    with pytest.raises(ValueError):
        walk_forward_splits(n_obs=10, n_splits=20, min_train_fraction=0.5)


def test_purge_and_embargo_trim_the_trailing_training_edge():
    (train_idx, test_idx), = walk_forward_splits(
        n_obs=100, n_splits=1, min_train_fraction=0.5, purge=3
    )
    assert test_idx.tolist() == list(range(50, 100))
    assert train_idx.max() == 50 - 3 - 1  # last 3 rows before test_start dropped

    (train_idx2, _), = walk_forward_splits(
        n_obs=100, n_splits=1, min_train_fraction=0.5, purge=3, embargo=4
    )
    assert train_idx2.max() == 50 - 3 - 4 - 1  # purge + embargo both trimmed


def test_purge_embargo_preserve_expanding_window_across_folds():
    splits = walk_forward_splits(n_obs=100, n_splits=5, min_train_fraction=0.5, purge=2, embargo=1)
    prev_train_size = 0
    for train_idx, test_idx in splits:
        assert train_idx.max(initial=-1) <= test_idx.min() - 1 - 2 - 1  # gap of purge+embargo=3
        assert len(train_idx) >= prev_train_size
        prev_train_size = len(train_idx)


def test_purge_embargo_raises_when_it_would_remove_all_training_data():
    with pytest.raises(ValueError):
        walk_forward_splits(n_obs=100, n_splits=1, min_train_fraction=0.5, purge=49, embargo=1)


def test_run_walk_forward_recovers_noiseless_relationship_out_of_sample():
    idx = pd.date_range("2026-01-01", periods=100, freq="1s")
    ofi = np.linspace(-1, 1, 100)
    df = pd.DataFrame({"ofi": ofi, "ret": 0.5 * ofi}, index=idx)

    oos = run_walk_forward(df, ["ofi"], "ret", n_splits=2, min_train_fraction=0.5)

    assert len(oos) == 50  # tail half, since min_train_fraction=0.5
    assert oos["y_pred_model"].tolist() == pytest.approx(oos["y_true"].tolist(), abs=1e-6)
    r2 = oos_r2(oos["y_true"], oos["y_pred_model"], oos["y_pred_baseline"])
    assert r2 == pytest.approx(1.0, abs=1e-6)


def test_run_walk_forward_purge_excludes_boundary_row_with_test_overlapping_label():
    """Row 19 (the last pre-test row, since test_start=20) is poisoned with a
    label that only makes sense if it had 'seen' test-period data -- standing
    in for a real forward-looking target whose horizon reached past
    test_start. purge=1 must exclude exactly that row from the fit, verified
    by matching predictions against a model fit on the manually-trimmed
    slice; without purge, the poisoned row corrupts the fit instead."""
    n = 40
    idx = pd.date_range("2026-01-01", periods=n, freq="1s")
    ofi = np.linspace(-1, 1, n)
    poisoned = pd.DataFrame({"ofi": ofi, "ret": 0.1 * ofi}, index=idx)
    poisoned.iloc[19, poisoned.columns.get_loc("ret")] = 999.0

    expected_model = fit_ols(poisoned.iloc[:19][["ofi"]], poisoned.iloc[:19]["ret"], ["ofi"])
    expected_pred = expected_model.predict(poisoned.iloc[20:][["ofi"]])

    oos_purged = run_walk_forward(poisoned, ["ofi"], "ret", n_splits=1, min_train_fraction=0.5, purge=1)
    assert oos_purged["y_pred_model"].tolist() == pytest.approx(expected_pred.tolist())

    oos_unpurged = run_walk_forward(poisoned, ["ofi"], "ret", n_splits=1, min_train_fraction=0.5)
    assert oos_unpurged["y_pred_model"].tolist() != pytest.approx(expected_pred.tolist())


def test_oos_r2_is_zero_when_model_equals_baseline_and_one_when_perfect():
    y_true = pd.Series([1.0, 2.0, 3.0, 4.0])
    baseline = pd.Series([2.5, 2.5, 2.5, 2.5])

    assert oos_r2(y_true, baseline, baseline) == pytest.approx(0.0)
    assert oos_r2(y_true, y_true, baseline) == pytest.approx(1.0)


def test_newey_west_significance_recovers_coefficient_with_hac_se():
    x = np.arange(50, dtype=float)
    # Tiny deterministic wiggle, not pure noiseless collinearity -- avoids a
    # degenerate (all-zero-residual) HAC covariance matrix.
    y = 2.0 + 3.0 * x + 0.01 * np.sin(x)
    X = pd.DataFrame({"ofi": x})
    results = newey_west_significance(X, pd.Series(y), ["ofi"], maxlags=5)

    assert results.params["ofi"] == pytest.approx(3.0, abs=0.01)
    assert results.pvalues["ofi"] < 0.01


def test_block_bootstrap_ci_is_tight_and_correct_for_deterministic_cases():
    y_true = pd.Series(np.linspace(0.0, 10.0, 200))
    perfect_pred = y_true.copy()
    baseline = pd.Series(np.full(200, y_true.mean()))

    perfect = block_bootstrap_r2_ci(y_true, perfect_pred, baseline, block_size=10, n_boot=200)
    assert perfect.point_estimate == pytest.approx(1.0)
    assert perfect.ci_low == pytest.approx(1.0, abs=1e-9)
    assert perfect.ci_high == pytest.approx(1.0, abs=1e-9)

    tied = block_bootstrap_r2_ci(y_true, baseline, baseline, block_size=10, n_boot=200)
    assert tied.point_estimate == pytest.approx(0.0)
    assert tied.ci_low == pytest.approx(0.0, abs=1e-9)
    assert tied.ci_high == pytest.approx(0.0, abs=1e-9)
