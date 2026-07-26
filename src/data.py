"""Data loading and feature engineering for the C-MAPSS FD001 dataset."""

from pathlib import Path
import pandas as pd

# Original 26 columns in the raw file (24 real + 2 phantom trailing empties).
_RAW_COLUMNS = [
    "engine_id", "cycle",
    "setting_1", "setting_2", "setting_3",
    "s1", "s2", "s3", "s4", "s5", "s6", "s7", "s8", "s9", "s10",
    "s11", "s12", "s13", "s14", "s15", "s16", "s17", "s18", "s19", "s20", "s21",
]

# Constant sensors and settings identified in EDA (zero variance -> no info).
_DROP_COLUMNS = ["setting_1", "setting_2", "setting_3",
                 "s1", "s5", "s6", "s10", "s16", "s18", "s19"]

# The 14 informative sensors kept after cleaning.
SENSOR_COLUMNS = ["s2", "s3", "s4", "s7", "s8", "s9",
                  "s11", "s12", "s13", "s14", "s15", "s17", "s20", "s21"]


def load_raw(path: str | Path) -> pd.DataFrame:
    """Load a raw C-MAPSS FD001 training file into a DataFrame.

    The raw file is whitespace-separated with two trailing empty columns.
    We name all 26 columns then drop the constant sensors and settings.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Raw data not found at {path}. Download the C-MAPSS dataset "
            "from https://data.nasa.gov/dataset/c-mapss-aircraft-engine-simulator-data "
            "and place train_FD001.txt in data/raw/."
        )
    df = pd.read_csv(path, sep=r"\s+", header=None, names=_RAW_COLUMNS, engine="python")
    return df.drop(columns=_DROP_COLUMNS)


def add_rul(df: pd.DataFrame, cap: int = 125) -> pd.DataFrame:
    """Compute Remaining Useful Life for each row.

    RUL = max_cycle_for_engine - current_cycle
    RUL_clipped applies a piecewise-linear degradation assumption:
    engines are considered "healthy" until they have `cap` cycles left,
    at which point RUL begins its linear countdown to zero.
    """
    df = df.copy()
    max_cycle = df.groupby("engine_id")["cycle"].transform("max")
    df["RUL"] = max_cycle - df["cycle"]
    df["RUL_clipped"] = df["RUL"].clip(upper=cap)
    return df


def add_rolling_features(df: pd.DataFrame, window: int = 5) -> pd.DataFrame:
    """Add per-engine rolling mean and std for each sensor.

    Uses min_periods=1 so the earliest cycles of each engine receive
    partial-window values instead of NaN. Rolling std of a single point
    is undefined -> filled with 0.0 (no observed variation yet).
    Grouping by engine_id prevents the window from bleeding across engines.
    """
    df = df.copy()
    grouped = df.groupby("engine_id")[SENSOR_COLUMNS]
    means = grouped.rolling(window=window, min_periods=1).mean().reset_index(level=0, drop=True)
    stds = grouped.rolling(window=window, min_periods=1).std().reset_index(level=0, drop=True).fillna(0.0)
    df[[f"{s}_rolling_mean" for s in SENSOR_COLUMNS]] = means
    df[[f"{s}_rolling_std" for s in SENSOR_COLUMNS]] = stds
    return df


def load_and_engineer(
    raw_path: str | Path = "data/raw/train_FD001.txt",
    rul_cap: int = 125,
    rolling_window: int = 5,
) -> pd.DataFrame:
    """Full pipeline: raw file -> cleaned + RUL + rolling features.

    This is the single entry point the rest of the code should call.
    """
    df = load_raw(raw_path)
    df = add_rul(df, cap=rul_cap)
    df = add_rolling_features(df, window=rolling_window)
    return df

def load_test_and_rul(
    test_path: str | Path = "data/raw/test_FD001.txt",
    rul_path: str | Path = "data/raw/RUL_FD001.txt",
    rul_cap: int = 125,
    rolling_window: int = 5,
) -> tuple[pd.DataFrame, pd.Series]:
    """Load the held-out FD001 test set and its true RUL labels.

    The test file contains sensor history for 100 engines, stopped some
    number of cycles before failure. RUL_FD001.txt contains one number
    per engine: the true RUL at the last recorded cycle. Only that last
    cycle per engine is graded.

    Returns:
        - X_test: feature matrix for the last cycle of each engine (100 rows)
        - y_test: true RUL for each engine, clipped at `rul_cap` for
          consistency with the training-target convention
    """
    test_path = Path(test_path)
    rul_path = Path(rul_path)
    for p in (test_path, rul_path):
        if not p.exists():
            raise FileNotFoundError(f"Required file missing: {p}")

    # Load and clean the test data with the same pipeline as train.
    df = load_raw(test_path)
    df = add_rolling_features(df, window=rolling_window)

    # Keep only the last recorded cycle of each engine — that's what's graded.
    last_cycles = df.groupby("engine_id").tail(1).reset_index(drop=True)

    # True RUL labels (one per engine, in engine_id order).
    true_rul = pd.read_csv(rul_path, header=None, names=["RUL"])["RUL"]
    true_rul_clipped = true_rul.clip(upper=rul_cap)

    return last_cycles, true_rul_clipped
