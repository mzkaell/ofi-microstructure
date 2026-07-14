"""Tests for src/modeling.py against small synthetic panels."""

import numpy as np
import pandas as pd
import pytest

from src.modeling import fit_ols, predict_historical_mean


def test_fit_ols_recovers_known_linear_relationship():
    rng = np.arange(20, dtype=float)
    X = pd.DataFrame({"ofi": rng})
    y = pd.Series(2.0 + 3.0 * rng, name="target")  # noiseless: y = 2 + 3*ofi

    model = fit_ols(X, y, feature_cols=["ofi"])
    pred = model.predict(X)

    assert model.results.params["const"] == pytest.approx(2.0, abs=1e-8)
    assert model.results.params["ofi"] == pytest.approx(3.0, abs=1e-8)
    assert pred.tolist() == pytest.approx(y.tolist(), abs=1e-6)


def test_fit_ols_drops_nan_rows_from_either_side():
    X = pd.DataFrame({"ofi": [1.0, np.nan, 3.0, 4.0]})
    y = pd.Series([10.0, 20.0, np.nan, 40.0])

    model = fit_ols(X, y, feature_cols=["ofi"])

    # Only rows 0 and 3 have both a valid feature and target -- a line
    # through exactly those two points is fully determined; check it doesn't
    # error and produces the exact fit through them.
    pred = model.predict(pd.DataFrame({"ofi": [1.0, 4.0]}))
    assert pred.tolist() == pytest.approx([10.0, 40.0], abs=1e-6)


def test_fit_ols_raises_when_no_rows_survive_dropna():
    X = pd.DataFrame({"ofi": [np.nan, np.nan]})
    y = pd.Series([1.0, 2.0])
    with pytest.raises(ValueError):
        fit_ols(X, y, feature_cols=["ofi"])


def test_historical_mean_baseline_uses_train_mean_not_test_mean():
    y_train = pd.Series([1.0, 2.0, 3.0])  # mean = 2.0
    test_index = pd.RangeIndex(5)

    pred = predict_historical_mean(y_train, test_index)

    assert (pred == 2.0).all()
    assert list(pred.index) == list(test_index)
