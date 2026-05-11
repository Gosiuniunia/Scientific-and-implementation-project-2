import numpy as np
import pytest
from unittest.mock import MagicMock
from core.microservice import AIModelService


@pytest.fixture
def mock_service(monkeypatch):
    mock_pipeline = MagicMock()

    mock_scaler = MagicMock()
    mock_svc = MagicMock()

    mock_pipeline.named_steps = {"scaler": mock_scaler, "svc": mock_svc}

    mock_scaler.transform.return_value = np.array([[1, 2, 3]])

    mock_svc.classes_ = np.array([0, 1, 2, 3])

    mock_svc.decision_function.return_value = np.array(
        [[0.5, -0.5, 0.6, -0.6, 0.2, -0.2]]
    )

    monkeypatch.setattr("joblib.load", lambda _: mock_pipeline)

    return AIModelService("fake_path")


def test_predict_valid(mock_service):
    features = [1] * 9

    result = mock_service.predict(features)

    assert isinstance(result, (int, type(None)))


def test_predict_invalid_length(mock_service):
    with pytest.raises(ValueError):
        mock_service.predict([1, 2, 3])


def test_count_votes_winner(mock_service):
    votes = np.array([[3, 1, 0, 0]])

    result = mock_service.count_votes(votes)

    assert result == 0


def test_count_votes_tie(mock_service):
    votes = np.array([[2, 2, 0, 0]])

    result = mock_service.count_votes(votes)

    assert result is None


def test_predict_with_voting_shape(mock_service):
    X = np.array([[1, 2, 3]])
    decisions = np.array([[0.5, -0.5, 0.6, -0.6, 0.2, -0.2]])

    votes = mock_service.predict_with_voting(X, decisions)

    assert votes.shape == (1, 4)
