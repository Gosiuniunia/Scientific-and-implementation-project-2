# tests/test_api.py

from fastapi.testclient import TestClient
from core.microservice import app


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
