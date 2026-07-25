"""Tests for src/metrics.py."""

import math
import pytest
from src.metrics import rmse, nasa_score


def test_rmse_perfect_prediction_is_zero():
    assert rmse([10, 20, 30], [10, 20, 30]) == 0.0


def test_rmse_known_value():
    assert rmse([1, 2, 3], [1, 2, 4]) == pytest.approx(1 / math.sqrt(3))


def test_rmse_is_symmetric_in_sign():
    assert rmse([10], [15]) == rmse([10], [5])


def test_rmse_accepts_lists_series_and_arrays():
    import numpy as np
    import pandas as pd
    a, b = [1, 2, 3], [1, 2, 4]
    expected = rmse(a, b)
    assert rmse(np.array(a), np.array(b)) == expected
    assert rmse(pd.Series(a), pd.Series(b)) == expected


def test_nasa_perfect_prediction_is_zero():
    assert nasa_score([100, 100, 100], [100, 100, 100]) == 0.0


def test_nasa_late_prediction_known_value():
    assert nasa_score([100], [130]) == pytest.approx(math.exp(30 / 10) - 1)


def test_nasa_early_prediction_known_value():
    assert nasa_score([100], [70]) == pytest.approx(math.exp(30 / 13) - 1)


def test_nasa_penalises_late_more_than_early():
    assert nasa_score([100], [130]) > nasa_score([100], [70])


def test_nasa_sums_across_predictions():
    single = nasa_score([100], [130])
    doubled = nasa_score([100, 100], [130, 130])
    assert doubled == pytest.approx(2 * single)
