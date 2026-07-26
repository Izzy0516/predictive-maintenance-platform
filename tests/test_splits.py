"""Tests for src/splits.py."""

import numpy as np
import pandas as pd
import pytest
from src.splits import grouped_kfold_engines, train_test_split_by_engine


@pytest.fixture
def toy_df():
    """20 engines, 10 rows each = 200 rows total."""
    rows = []
    for engine_id in range(1, 21):
        for cycle in range(1, 11):
            rows.append({"engine_id": engine_id, "cycle": cycle, "s2": float(cycle)})
    return pd.DataFrame(rows)


# ---------- grouped_kfold_engines ----------

def test_kfold_yields_correct_number_of_folds(toy_df):
    folds = list(grouped_kfold_engines(toy_df, n_splits=5))
    assert len(folds) == 5


def test_kfold_no_engine_in_both_train_and_test(toy_df):
    """The one invariant this module exists to guarantee."""
    for train_idx, test_idx in grouped_kfold_engines(toy_df, n_splits=5):
        train_engines = set(toy_df.iloc[train_idx]["engine_id"])
        test_engines = set(toy_df.iloc[test_idx]["engine_id"])
        assert train_engines.isdisjoint(test_engines)


def test_kfold_every_row_appears_in_exactly_one_test_fold(toy_df):
    """Union of test folds should cover the full dataset exactly once."""
    all_test_indices = np.concatenate(
        [test_idx for _, test_idx in grouped_kfold_engines(toy_df, n_splits=5)]
    )
    assert sorted(all_test_indices) == list(range(len(toy_df)))


def test_kfold_raises_if_engine_id_missing():
    df = pd.DataFrame({"cycle": [1, 2, 3]})
    with pytest.raises(ValueError, match="engine_id"):
        list(grouped_kfold_engines(df))


def test_kfold_raises_if_too_few_engines(toy_df):
    small = toy_df[toy_df.engine_id <= 3]
    with pytest.raises(ValueError, match="Cannot make"):
        list(grouped_kfold_engines(small, n_splits=5))


# ---------- train_test_split_by_engine ----------

def test_single_split_no_engine_leakage(toy_df):
    train, test = train_test_split_by_engine(toy_df, test_fraction=0.2)
    assert set(train["engine_id"]).isdisjoint(set(test["engine_id"]))


def test_single_split_test_fraction_of_engines(toy_df):
    train, test = train_test_split_by_engine(toy_df, test_fraction=0.2)
    # 20 engines, 20% -> 4 test engines, 16 train engines
    assert test["engine_id"].nunique() == 4
    assert train["engine_id"].nunique() == 16


def test_single_split_is_deterministic(toy_df):
    t1, _ = train_test_split_by_engine(toy_df, random_state=42)
    t2, _ = train_test_split_by_engine(toy_df, random_state=42)
    pd.testing.assert_frame_equal(t1, t2)


def test_single_split_all_rows_of_test_engines_go_to_test(toy_df):
    """If engine 5 is in test, ALL 10 of its rows are in test, not a subset."""
    _, test = train_test_split_by_engine(toy_df, test_fraction=0.2)
    for engine_id in test["engine_id"].unique():
        rows_in_test = (test["engine_id"] == engine_id).sum()
        rows_in_original = (toy_df["engine_id"] == engine_id).sum()
        assert rows_in_test == rows_in_original
