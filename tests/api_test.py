# tests/test_api.py

from fastapi.testclient import TestClient
from core.microservice import app
from unittest.mock import patch


client = TestClient(app)


def test_predict_success():
    response = client.post("/predict", json={"features": [1] * 9})

    assert response.status_code == 200
    data = response.json()

    assert "prediction" in data
    assert isinstance(data["prediction"], (int, type(None)))


def test_predict_invalid_length():
    response = client.post("/predict", json={"features": [1, 2]})

    assert response.status_code == 422


def test_predict_invalid_type():
    response = client.post("/predict", json={"features": ["a"] * 9})

    assert response.status_code == 422

@patch("core.microservice.model_service.predict")
def test_predict_model_service_value_error(mock_predict):
    mock_predict.side_effect = ValueError("Invalid feature length")
    response = client.post("/predict", json={"features": [1] * 9})

    assert response.status_code == 400
    assert response.json() == {"detail": "Invalid feature length"}
