"""Tests for src/data.py."""

import numpy as np
import pandas as pd
import pytest
from src.data import add_rul, add_rolling_features, SENSOR_COLUMNS


@pytest.fixture
def toy_df():
    """A tiny two-engine DataFrame we can reason about by hand."""
    rows = []
    for engine_id, n_cycles in [(1, 10), (2, 6)]:
        for cycle in range(1, n_cycles + 1):
            row = {"engine_id": engine_id, "cycle": cycle}
            for s in SENSOR_COLUMNS:
                row[s] = float(cycle)  # trivially increasing per cycle
            rows.append(row)
    return pd.DataFrame(rows)


# ---------- add_rul ----------

def test_rul_last_cycle_of_each_engine_is_zero(toy_df):
    out = add_rul(toy_df)
    last_rows = out.groupby("engine_id").tail(1)
    assert (last_rows["RUL"] == 0).all()


def test_rul_decreases_by_one_each_cycle(toy_df):
    out = add_rul(toy_df)
    engine_1 = out[out.engine_id == 1].sort_values("cycle")
    diffs = engine_1["RUL"].diff().dropna()
    assert (diffs == -1).all()


def test_rul_clipping_caps_healthy_region(toy_df):
    out = add_rul(toy_df, cap=3)
    assert out["RUL_clipped"].max() == 3
    assert (out["RUL_clipped"] <= out["RUL"]).all()


# ---------- add_rolling_features ----------

def test_rolling_features_no_nans(toy_df):
    """Every row should have a rolling mean and std; no NaNs from short windows."""
    out = add_rolling_features(toy_df, window=5)
    rolling_cols = [c for c in out.columns if "rolling" in c]
    assert out[rolling_cols].isna().sum().sum() == 0


def test_rolling_features_do_not_bleed_across_engines(toy_df):
    """Engine 2's cycle 1 rolling mean must equal its own s2 value,
    not include any of engine 1's data."""
    out = add_rolling_features(toy_df, window=5)
    engine_2_cycle_1 = out[(out.engine_id == 2) & (out.cycle == 1)]
    # s2 at engine 2 cycle 1 is 1.0 (from the fixture); its rolling mean should be 1.0.
    assert engine_2_cycle_1["s2_rolling_mean"].iloc[0] == 1.0


def test_rolling_std_is_zero_at_first_cycle(toy_df):
    """A single observation has undefined std; we fill with 0.0."""
    out = add_rolling_features(toy_df, window=5)
    first_cycles = out[out.cycle == 1]
    assert (first_cycles[[f"{s}_rolling_std" for s in SENSOR_COLUMNS]] == 0.0).all().all()


def test_rolling_mean_of_first_cycle_equals_that_cycles_value(toy_df):
    """min_periods=1 means the first row's rolling mean = the row's own value."""
    out = add_rolling_features(toy_df, window=5)
    for engine_id in [1, 2]:
        first = out[(out.engine_id == engine_id) & (out.cycle == 1)]
        assert first["s2_rolling_mean"].iloc[0] == first["s2"].iloc[0]
