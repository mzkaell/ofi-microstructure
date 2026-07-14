"""
Stage 3: regression models and the naive baseline used to benchmark them.

Deliberately thin: this module only fits/predicts on a single given
train/test split. Walk-forward fold looping, Newey-West inference, and
out-of-sample R^2 all live in src/evaluation.py, which calls into this module
once per fold -- keeping the split logic out of here means fit/predict are
trivially testable against small synthetic panels without touching real data
or CV machinery.

Why a "historical mean" baseline, not "predict zero": out-of-sample R^2 for
return prediction is only meaningful relative to a stated benchmark (Campbell
& Thompson 2008; Welch & Goyal 2008). Predicting 0 silently assumes the
unconditional mean return is exactly 0, which inflates R^2 by whatever the
true (small, nonzero) drift is. The historical-mean baseline instead predicts
each test point with the *training-set* mean target -- a real, if weak,
forecast -- so OFI's R^2 has to earn its keep over "knowing the training
sample's average return," not over a strawman.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
import statsmodels.api as sm


@dataclass
class FittedModel:
    results: sm.regression.linear_model.RegressionResultsWrapper
    feature_cols: list[str]

    def predict(self, X: pd.DataFrame) -> pd.Series:
        X_ = sm.add_constant(X[self.feature_cols], has_constant="add")
        return pd.Series(self.results.predict(X_), index=X.index, name="y_pred")


def fit_ols(X: pd.DataFrame, y: pd.Series, feature_cols: list[str]) -> FittedModel:
    """Plain (non-robust) OLS of y on feature_cols + intercept, dropping any
    row with a NaN in either the features or the target. Ordinary SEs are
    fine here because this fit is only used for point predictions; robust
    (Newey-West) inference on the coefficients is a separate, explicit step
    in evaluation.py, not silently baked into every fold's fit."""
    panel = pd.concat([X[feature_cols], y.rename("__y__")], axis=1).dropna()
    if panel.empty:
        raise ValueError("no non-NaN rows to fit on after dropping NaNs")
    X_ = sm.add_constant(panel[feature_cols], has_constant="add")
    results = sm.OLS(panel["__y__"], X_).fit()
    return FittedModel(results=results, feature_cols=feature_cols)


def predict_historical_mean(y_train: pd.Series, index: pd.Index) -> pd.Series:
    """The benchmark forecast: every test-period prediction is the
    *training*-period target mean (never the test period's own mean, which
    would leak future information into the "naive" forecast it's supposed to
    be beaten by)."""
    mean = y_train.dropna().mean()
    return pd.Series(mean, index=index, name="y_pred")
