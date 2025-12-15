import numpy as np
from unittest.mock import MagicMock, patch
from itertools import combinations

from PCoA_prediction import(    
     count_votes,
     predict_with_voting,
     map_class,
     predict_class
)

def test_count_votes_no_tie():
    votes_0 = np.array([[2, 1, 0, 0]])
    votes_2 = np.array([[1, 1, 3, 2]])
    assert count_votes(votes_0) == 0
    assert count_votes(votes_2) == 2

def test_count_votes_tie():
    votes = np.array([[1, 3, 3, 0]])
    assert count_votes(votes) is None


def test_predict_with_voting():
    X_scaled = np.zeros((1, 9))
    print(X_scaled.shape, X_scaled)
    decisions = np.array([[1, -1, 0.3, -0.1, 0.5, -0.5]])
    classes = np.array([0, 1, 2, 3])
    pairs = list(combinations(classes, 2))

    votes = predict_with_voting(X_scaled, decisions, classes, pairs, threshold=0.2)

    assert votes.shape == (1, 4)
    assert votes.sum() == 5
    
def test_map_class():
    assert map_class(0) == "autumn"
    assert map_class(1) == "spring"
    assert map_class(2) == "summer"
    assert map_class(3) == "winter"
    assert map_class(None) is None


@patch("joblib.load")
def test_predict_class_integration(mock_load):
    mock_scaler = MagicMock()
    mock_scaler.transform.return_value = np.zeros((1, 9))

    mock_svc = MagicMock()
    mock_svc.classes_ = np.array([0, 1, 2, 3])
    mock_svc.decision_function.return_value = np.array([
        [1, 1, 1, -1, -1, -1]
    ])

    mock_pipeline = MagicMock()
    mock_pipeline.named_steps = {
        "scaler": mock_scaler,
        "svc": mock_svc
    }

    mock_load.return_value = mock_pipeline

    features = [1,2,3,4,5,6,7,8,9]
    result = predict_class("fake_path.pkl", features)

    assert result in {"autumn", "spring", "summer", "winter"}