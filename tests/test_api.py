"""Tests for the FastAPI service in src/api.py.

Uses TestClient inside a `with` block, which correctly triggers the
lifespan startup and shutdown events. Without the `with`, the model
would never be loaded and every request would hit a NoneType error.
"""

import joblib
import pytest
from fastapi.testclient import TestClient

from src.api import app, MODEL_PATH, FEATURE_LIST_PATH


@pytest.fixture(scope="module")
def client() -> TestClient:
    """Boot the app once for all tests in this module, firing lifespan events."""
    if not MODEL_PATH.exists():
        pytest.skip("Trained model not found; run `python -m src.train` first")
    with TestClient(app) as c:
        yield c


@pytest.fixture(scope="module")
def sample_features() -> dict[str, float]:
    """A minimal valid feature dict — every feature the model expects, set to 0."""
    feature_cols = joblib.load(FEATURE_LIST_PATH)
    return {c: 0.0 for c in feature_cols}


def test_health_endpoint_returns_ok(client):
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["model_loaded"] == "True"


def test_predict_returns_expected_shape(client, sample_features):
    response = client.post("/predict", json={"features": sample_features})
    assert response.status_code == 200
    body = response.json()
    assert "predicted_rul" in body
    assert "recommendation" in body
    assert isinstance(body["predicted_rul"], (int, float))
    assert body["recommendation"] in {"immediate_inspection", "schedule_inspection", "healthy"}


def test_predict_rejects_missing_features(client):
    """Missing feature -> 400 with a helpful message."""
    response = client.post("/predict", json={"features": {"s2": 100.0}})
    assert response.status_code == 400
    assert "Missing required features" in response.json()["detail"]


def test_predict_rejects_malformed_body(client):
    """Missing the top-level 'features' key -> 422 from pydantic."""
    response = client.post("/predict", json={"nonsense": 1})
    assert response.status_code == 422


def test_recommendation_matches_prediction_thresholds(client, sample_features):
    """The recommendation logic should be consistent with the predicted RUL."""
    response = client.post("/predict", json={"features": sample_features})
    body = response.json()
    rul = body["predicted_rul"]
    rec = body["recommendation"]

    if rul < 20:
        assert rec == "immediate_inspection"
    elif rul < 60:
        assert rec == "schedule_inspection"
    else:
        assert rec == "healthy"
