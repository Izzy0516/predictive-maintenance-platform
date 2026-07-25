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
