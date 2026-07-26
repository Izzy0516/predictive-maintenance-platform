"""Predictive models for Remaining Useful Life.

Every model here follows the sklearn estimator convention: fit(X, y), predict(X).
This lets us swap models without changing the surrounding code and gives us
compatibility with sklearn's cross-validation and pipeline tooling.
"""

import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin


class MeanRULBaseline(BaseEstimator, RegressorMixin):
    """Predicts the mean RUL of the training set, regardless of input.

    This is the dumbest useful baseline for RUL regression. Any real model
    must beat this to demonstrate it has learned something from the sensors.
    Its RMSE equals the standard deviation of the training target.
    """

    def fit(self, X, y):
        self.mean_ = float(np.asarray(y, dtype=float).mean())
        return self

    def predict(self, X):
        if not hasattr(self, "mean_"):
            raise RuntimeError("Call fit before predict")
        n = len(X)
        return np.full(n, self.mean_)
