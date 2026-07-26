"""Tests for src/models.py."""

import numpy as np
import pytest
from src.models import MeanRULBaseline


def test_baseline_predicts_training_mean():
    m = MeanRULBaseline().fit(X=None, y=[10, 20, 30])
    preds = m.predict(np.zeros((5, 3)))  # 5 rows, 3 features (ignored)
    assert np.allclose(preds, 20.0)


def test_baseline_returns_one_prediction_per_row():
    m = MeanRULBaseline().fit(X=None, y=[100, 100, 100, 100])
    preds = m.predict(np.zeros((7, 2)))
    assert len(preds) == 7


def test_baseline_ignores_features():
    """Same y, different X -> same predictions."""
    m1 = MeanRULBaseline().fit(X=np.zeros((3, 2)), y=[10, 20, 30])
    m2 = MeanRULBaseline().fit(X=np.ones((3, 2)) * 999, y=[10, 20, 30])
    assert m1.mean_ == m2.mean_


def test_baseline_raises_if_predict_before_fit():
    m = MeanRULBaseline()
    with pytest.raises(RuntimeError, match="fit"):
        m.predict(np.zeros((3, 2)))


def test_baseline_rmse_equals_target_std():
    """The mean predictor's RMSE on its own training set = std of y.

    This is the mathematical property that makes the baseline useful:
    any model with RMSE < std(y) has learned something.
    """
    from src.metrics import rmse
    y = np.array([10, 20, 30, 40, 50, 60], dtype=float)
    m = MeanRULBaseline().fit(X=None, y=y)
    preds = m.predict(np.zeros((len(y), 1)))
    assert rmse(y, preds) == pytest.approx(y.std())
