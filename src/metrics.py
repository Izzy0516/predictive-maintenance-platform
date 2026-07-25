"""Regression metrics for Remaining Useful Life (RUL) prediction."""

import numpy as np


def rmse(y_true, y_pred) -> float:
    """Root Mean Squared Error."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    errors = y_true - y_pred
    return float(np.sqrt((errors ** 2).mean()))


def nasa_score(y_true, y_pred) -> float:
    """NASA C-MAPSS asymmetric scoring function."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    d = y_pred - y_true
    scores = np.where(d < 0, np.exp(-d / 13) - 1, np.exp(d / 10) - 1)
    return float(scores.sum())
