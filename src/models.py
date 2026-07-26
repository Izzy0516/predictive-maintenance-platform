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

from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def make_ridge_pipeline(alpha: float = 1.0) -> Pipeline:
    """Standard-scaled Ridge regression pipeline.

    StandardScaler ensures all features contribute equally to the L2 penalty.
    Features here range from ~8 (s15) to ~9000 (s9); without scaling, Ridge
    would disproportionately shrink coefficients on small-valued features
    because their per-unit contribution to the penalty is larger.

    The Pipeline wrapper is critical: it re-fits the scaler on each CV
    fold's training data only, then applies it to that fold's test data.
    Fitting the scaler on the combined data would leak test-set statistics
    into the training preprocessing — a subtle but real form of leakage.

    Args:
        alpha: Regularisation strength. Higher -> more shrinkage toward 0.

    Returns:
        A sklearn Pipeline with .fit(X, y) and .predict(X) methods.
    """
    return Pipeline([
        ("scaler", StandardScaler()),
        ("ridge", Ridge(alpha=alpha, random_state=42)),
    ])

from sklearn.ensemble import RandomForestRegressor


def make_random_forest(n_estimators: int = 100, max_depth: int | None = None) -> RandomForestRegressor:
    """Random Forest regressor for RUL prediction.

    Trees are invariant to feature scale, so no StandardScaler needed.
    n_jobs=-1 uses all CPU cores for parallel tree building.

    Args:
        n_estimators: Number of trees. More trees -> lower variance, slower.
        max_depth: Cap tree depth. None means grow until leaves are pure.

    Returns:
        A fitted-when-you-call-.fit sklearn RandomForestRegressor.
    """
    return RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=42,
        n_jobs=-1,
    )