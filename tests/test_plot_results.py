"""Tests for scripts/plot_results.py using a small synthetic results table."""

import pandas as pd
import pytest

from scripts.plot_results import plot_r2_decay


def _make_results():
    return pd.DataFrame(
        {
            "feature": ["ofi_best"] * 4 + ["ofi_multilevel"] * 4,
            "horizon_s": [1, 2, 5, 10] * 2,
            "oos_r2_pct": [1.5, 1.0, 0.3, 0.0, 1.8, 1.2, 0.4, 0.05],
            "oos_r2_ci_low_pct": [0.5, 0.2, -0.5, -0.8, 0.6, 0.3, -0.4, -0.7],
            "oos_r2_ci_high_pct": [2.5, 1.8, 1.1, 0.8, 3.0, 2.1, 1.2, 0.9],
        }
    )


def test_plot_r2_decay_writes_file(tmp_path):
    out_path = tmp_path / "r2_decay.png"
    plot_r2_decay(_make_results(), out_path)
    assert out_path.exists()
    assert out_path.stat().st_size > 0


def test_plot_r2_decay_raises_on_missing_columns():
    bad = pd.DataFrame({"feature": ["ofi_best"], "horizon_s": [1]})
    with pytest.raises(ValueError):
        plot_r2_decay(bad, "unused.png")
