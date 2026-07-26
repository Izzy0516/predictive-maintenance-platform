"""FastAPI service for RUL prediction.

Loads the trained model at startup, exposes a single /predict endpoint that
accepts sensor readings and returns a predicted Remaining Useful Life.

Run locally:
    uvicorn src.api:app --reload

Then POST to http://localhost:8000/predict
"""

from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, AsyncIterator

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field


MODEL_PATH = Path("models/rul_model.joblib")
FEATURE_LIST_PATH = Path("models/feature_columns.joblib")


# Loaded once at startup, reused for every request.
_model: Any = None
_feature_columns: list[str] = []


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Load model and feature list into memory at startup, clear at shutdown.

    The lifespan pattern replaces the deprecated @app.on_event('startup') hook.
    Everything before `yield` runs at startup; everything after runs at shutdown.
    """
    global _model, _feature_columns
    if not MODEL_PATH.exists():
        raise RuntimeError(
            f"Model artifact not found at {MODEL_PATH}. "
            "Run `python -m src.train` first."
        )
    _model = joblib.load(MODEL_PATH)
    _feature_columns = joblib.load(FEATURE_LIST_PATH)
    yield
    _model = None
    _feature_columns = []


app = FastAPI(
    title="Predictive Maintenance API",
    description="Predicts Remaining Useful Life (RUL) for jet engines given sensor readings.",
    version="0.1.0",
    lifespan=lifespan,
)


class PredictionRequest(BaseModel):
    """One row of feature values, keyed by feature name.

    Callers must supply values for every column the model was trained on.
    The service checks this at request time so missing features fail loudly.
    """
    features: dict[str, float] = Field(
        ...,
        description="Map of feature_name -> value. Must include all training features.",
        examples=[{"s2": 642.15, "s3": 1591.82, "s4": 1408.7}],
    )


class PredictionResponse(BaseModel):
    predicted_rul: float = Field(..., description="Predicted remaining useful life, in cycles.")
    recommendation: str = Field(..., description="Human-readable maintenance recommendation.")


def _recommend(rul: float) -> str:
    """Translate a numeric RUL into a maintenance action."""
    if rul < 20:
        return "immediate_inspection"
    if rul < 60:
        return "schedule_inspection"
    return "healthy"


@app.get("/health")
def health() -> dict[str, str]:
    """Liveness check. Returns 200 if the app is up and the model is loaded."""
    return {"status": "ok", "model_loaded": str(_model is not None)}


@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest) -> PredictionResponse:
    """Predict RUL for a single engine reading."""
    missing = [c for c in _feature_columns if c not in request.features]
    if missing:
        raise HTTPException(
            status_code=400,
            detail=f"Missing required features: {missing[:5]}{' ...' if len(missing) > 5 else ''}",
        )

    # Build a single-row DataFrame in the exact column order the model saw at fit time.
    X = pd.DataFrame([request.features])[_feature_columns]
    pred = float(_model.predict(X)[0])

    return PredictionResponse(
        predicted_rul=round(pred, 2),
        recommendation=_recommend(pred),
    )
