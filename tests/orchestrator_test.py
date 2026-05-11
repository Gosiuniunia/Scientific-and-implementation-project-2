import pytest
from unittest.mock import patch, MagicMock
import requests

from core.microservice_orchestrator import AIServiceOrchestrator


def test_map_result_to_color_type():
    orchestrator = AIServiceOrchestrator("http://fake")

    assert orchestrator.map_result_to_color_type(0) == "autumn"
    assert orchestrator.map_result_to_color_type(1) == "spring"
    assert orchestrator.map_result_to_color_type(2) == "summer"
    assert orchestrator.map_result_to_color_type(3) == "winter"
    assert orchestrator.map_result_to_color_type(999) == "unknown"


@patch("core.microservice_orchestrator.requests.post")
def test_get_prediction_success(mock_post):
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {"prediction": 2}
    mock_post.return_value = mock_response

    orchestrator = AIServiceOrchestrator("http://fake-url")

    result = orchestrator.get_prediction_from_ai_service([1] * 9)

    assert result == "summer"


@patch("core.microservice_orchestrator.requests.post")
def test_prediction_none(mock_post):
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {"prediction": None}
    mock_post.return_value = mock_response

    orchestrator = AIServiceOrchestrator("http://fake-url")

    result = orchestrator.get_prediction_from_ai_service([1] * 9)

    assert result == "none"


@patch("core.microservice_orchestrator.requests.post")
def test_api_error(mock_post):
    mock_response = MagicMock()
    mock_response.status_code = 500
    mock_response.text = "Internal error"
    mock_post.return_value = mock_response

    orchestrator = AIServiceOrchestrator("http://fake-url")

    with pytest.raises(Exception):
        orchestrator.get_prediction_from_ai_service([1] * 9)


@patch("core.microservice_orchestrator.requests.post")
def test_request_payload(mock_post):
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {"prediction": 1}
    mock_post.return_value = mock_response

    orchestrator = AIServiceOrchestrator("http://fake-url")

    features = [1] * 9
    orchestrator.get_prediction_from_ai_service(features)

    mock_post.assert_called_once_with(
        "http://fake-url/predict", json={"features": features}, timeout=5
    )


@patch("core.microservice_orchestrator.requests.post")
def test_timeout(mock_post):
    mock_post.side_effect = requests.exceptions.Timeout

    orchestrator = AIServiceOrchestrator("http://fake-url")

    with pytest.raises(Exception):
        orchestrator.get_prediction_from_ai_service([1] * 9)
