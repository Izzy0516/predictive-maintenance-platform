"""Train the production model on the full FD001 training set and save it.

Run this once (or whenever the data or model changes):
    python -m src.train

The saved model is loaded by the API at request time — training never happens
inside the request path.
"""

from pathlib import Path
import joblib

from src.data import load_and_engineer
from src.models import make_gradient_boosting


MODEL_PATH = Path("models/rul_model.joblib")
FEATURE_LIST_PATH = Path("models/feature_columns.joblib")

NON_FEATURE_COLS = ["engine_id", "cycle", "RUL", "RUL_clipped"]


def main() -> None:
    """Train the chosen model on all of FD001 train and persist to disk."""
    print("Loading and engineering training data...")
    df = load_and_engineer()

    feature_cols = [c for c in df.columns if c not in NON_FEATURE_COLS]
    X = df[feature_cols]
    y = df["RUL_clipped"]

    print(f"Training on {len(X)} rows, {len(feature_cols)} features, "
          f"{df['engine_id'].nunique()} engines...")
    model = make_gradient_boosting()
    model.fit(X, y)
    print("Training complete.")

    MODEL_PATH.parent.mkdir(exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    joblib.dump(feature_cols, FEATURE_LIST_PATH)
    print(f"Saved model to {MODEL_PATH}")
    print(f"Saved feature list to {FEATURE_LIST_PATH}")


if __name__ == "__main__":
    main()
