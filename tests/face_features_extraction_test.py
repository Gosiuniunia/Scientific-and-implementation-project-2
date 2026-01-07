import sys
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

import pytest
import numpy as np
from unittest.mock import MagicMock, patch
from core.utils.face_features_extraction import (
    get_face_landmarks,
    extract_iris_colour,
    extract_skin_colour,
    extract_hair_colour,
    extract_lab_values_from_photo,
    extract_face_features
)


@pytest.fixture
def dummy_img():
    """Returns a dummy RGB image for testing."""
    return np.zeros((100, 100, 3), dtype=np.uint8)


@pytest.fixture
def dummy_landmarks():
    """Returns dummy face landmarks for testing."""
    lm = MagicMock()
    lm.x = 0.5
    lm.y = 0.5
    return [[lm] * 6]


def test_extract_skin_colour(dummy_img, dummy_landmarks):
    """
    Tests extract_skin_colour.
    Verifies that the returned LAB values match mocked output.
    """
    with patch("core.utils.face_features_extraction.get_lab_colour") as mock_lab:
        mock_lab.return_value = np.array([50, 0, 0])
        result = extract_skin_colour(dummy_img, dummy_landmarks)
        assert isinstance(result, np.ndarray)
        assert np.array_equal(result, np.array([50, 0, 0]))


def test_extract_iris_colour(dummy_img):
    """
    Tests extract_iris_colour with mocked dependencies.
    Verifies that the returned LAB values match mocked output.
    """
    lm = MagicMock()
    lm.x = 0.5
    lm.y = 0.5
    face_landmarks = [[lm] * 500]

    with patch("core.utils.face_features_extraction.crop_img") as mock_crop, \
         patch("core.utils.face_features_extraction.apply_kmeans") as mock_kmeans, \
         patch("core.utils.face_features_extraction.get_color_between_points") as mock_color_between, \
         patch("core.utils.face_features_extraction.get_lab_colour") as mock_lab:

        mock_crop.return_value = (np.zeros((10, 10, 3)), (0, 0))
        mock_kmeans.return_value = np.zeros((10, 10, 3))
        mock_color_between.return_value = np.array([10, 10, 10])
        mock_lab.return_value = np.array([20, 0, 0])

        result = extract_iris_colour(dummy_img, face_landmarks)
        assert isinstance(result, np.ndarray)
        assert np.array_equal(result, np.array([20, 0, 0]))


def test_extract_hair_colour(dummy_img):
    """
    Tests extract_hair_colour with mocked dependencies.
    Verifies that the returned LAB values match mocked output.
    """
    lm = MagicMock()
    lm.x = 0.5
    lm.y = 0.5
    face_landmarks = [[lm] * 400]

    with patch("core.utils.face_features_extraction.crop_img") as mock_crop, \
         patch("core.utils.face_features_extraction.apply_kmeans") as mock_kmeans, \
         patch("core.utils.face_features_extraction.get_color_between_points") as mock_color_between, \
         patch("core.utils.face_features_extraction.get_lab_colour") as mock_lab:

        mock_crop.return_value = (np.zeros((10, 10, 3)), (0, 0))
        mock_kmeans.return_value = np.zeros((10, 10, 3))
        mock_color_between.return_value = np.array([15, 15, 15])
        mock_lab.return_value = np.array([30, 0, 0])

        result = extract_hair_colour(dummy_img, face_landmarks)
        assert isinstance(result, np.ndarray)
        assert np.array_equal(result, np.array([30, 0, 0]))


def test_extract_lab_values_from_photo(dummy_img):
    """
    Tests extract_lab_values_from_photo.
    Verifies that all LAB features (iris, skin, hair) are combined correctly.
    """
    with patch("core.utils.face_features_extraction.white_balance") as mock_wb, \
         patch("core.utils.face_features_extraction.get_face_landmarks") as mock_landmarks, \
         patch("core.utils.face_features_extraction.extract_iris_colour") as mock_iris, \
         patch("core.utils.face_features_extraction.extract_skin_colour") as mock_skin, \
         patch("core.utils.face_features_extraction.extract_hair_colour") as mock_hair:

        mock_wb.return_value = dummy_img / 255
        mock_landmarks.return_value = ["face"]
        mock_iris.return_value = np.array([1, 2, 3])
        mock_skin.return_value = np.array([4, 5, 6])
        mock_hair.return_value = np.array([7, 8, 9])

        result = extract_lab_values_from_photo(dummy_img, "FL", "opts")
        assert isinstance(result, list)
        assert result == [1, 2, 3, 4, 5, 6, 7, 8, 9]


def test_extract_face_features(dummy_img):
    """
    Tests extract_face_features end-to-end.
    Verifies that features returned match the mocked extraction output.
    """
    with patch("core.utils.face_features_extraction.init_face_landmark") as mock_init, \
         patch("core.utils.face_features_extraction.extract_lab_values_from_photo") as mock_extract:

        mock_init.return_value = ("FL", "opts")
        mock_extract.return_value = [1, 2, 3]

        result = extract_face_features(dummy_img, "model.task")
        assert isinstance(result, list)
        assert result == [1, 2, 3]
