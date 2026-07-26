"""Train/test splitting for engine-lifecycle data.

The C-MAPSS dataset contains ~200 consecutive cycles per engine. Rows within
one engine are highly correlated, so a naive row-level split leaks information
across train and test (cycle 87 and cycle 88 of the same engine are nearly
identical). All splits in this module respect engine boundaries: an engine
appears entirely in train or entirely in test, never both.
"""

from typing import Iterator
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold


def grouped_kfold_engines(
    df: pd.DataFrame,
    n_splits: int = 5,
) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    """Yield (train_indices, test_indices) for engine-grouped K-fold CV.

    Uses sklearn's GroupKFold under the hood, keyed on engine_id. Guarantees
    no engine appears in both sides of any split.

    Args:
        df: DataFrame containing an 'engine_id' column.
        n_splits: Number of folds. Default 5 -> ~20 test engines per fold on FD001.

    Yields:
        (train_idx, test_idx) as numpy arrays of row positions.
    """
    if "engine_id" not in df.columns:
        raise ValueError("df must contain an 'engine_id' column")
    if df["engine_id"].nunique() < n_splits:
        raise ValueError(
            f"Cannot make {n_splits} folds from {df['engine_id'].nunique()} engines"
        )

    gkf = GroupKFold(n_splits=n_splits)
    # X is a placeholder — GroupKFold ignores its content, only needs the length.
    X_placeholder = np.zeros(len(df))
    yield from gkf.split(X_placeholder, groups=df["engine_id"].to_numpy())


def train_test_split_by_engine(
    df: pd.DataFrame,
    test_fraction: float = 0.2,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Deterministic single train/test split by engine.

    Randomly assigns entire engines to train or test. Useful when you want
    one clean split instead of K folds (e.g. for a final held-out evaluation
    or a quick sanity check).

    Args:
        df: DataFrame with an 'engine_id' column.
        test_fraction: Fraction of *engines* (not rows) that go to test.
        random_state: Seed for reproducibility.

    Returns:
        (train_df, test_df) — each a copy, indexes reset.
    """
    if "engine_id" not in df.columns:
        raise ValueError("df must contain an 'engine_id' column")

    rng = np.random.default_rng(random_state)
    engines = df["engine_id"].unique()
    n_test = max(1, int(round(len(engines) * test_fraction)))
    test_engines = set(rng.choice(engines, size=n_test, replace=False))

    is_test = df["engine_id"].isin(test_engines)
    train_df = df[~is_test].reset_index(drop=True)
    test_df = df[is_test].reset_index(drop=True)
    return train_df, test_df
