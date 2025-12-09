import pytest
import numpy as np
from unittest.mock import patch

from face_features_extraction import (
    extract_iris_colour,
    extract_skin_colour,
    extract_hair_colour,
    extract_lab_values_from_photo,
)


@pytest.fixture
def dummy_img():
    return np.ones((100, 100, 3), dtype=np.uint8) * 120


@pytest.fixture
def dummy_landmarks():
    class LM:
        def __init__(self, x, y):
            self.x = x
            self.y = y
    return [[LM(2, 2) for _ in range(500)]]

def test_extract_iris_colour(dummy_img, dummy_landmarks):
    with patch("face_features_extraction.crop_img", return_value=(np.ones((10,10,3), dtype=np.uint8), (0,0))), \
         patch("face_features_extraction.apply_kmeans", return_value=np.ones((10,10,3), dtype=np.uint8)), \
         patch("face_features_extraction.get_color_between_points", return_value=np.array([10,20,30])), \
         patch("face_features_extraction.get_lab_colour", return_value=np.array([1,2,3])):
        out = extract_iris_colour(dummy_img, dummy_landmarks)
        assert out.tolist() == [1,2,3]

def test_extract_skin_colour(dummy_img, dummy_landmarks):
    with patch("face_features_extraction.get_lab_colour", return_value=np.array([9,9,9])):
        out = extract_skin_colour(dummy_img, dummy_landmarks)
        assert out.tolist() == [9,9,9]

def test_extract_hair_colour(dummy_img, dummy_landmarks):
    with patch("face_features_extraction.crop_img", return_value=(np.ones((10,10,3), dtype=np.uint8), (0,0))), \
         patch("face_features_extraction.apply_kmeans", return_value=np.ones((10,10,3), dtype=np.uint8)), \
         patch("face_features_extraction.get_color_between_points", return_value=np.array([5,6,7])), \
         patch("face_features_extraction.get_lab_colour", return_value=np.array([3,3,3])):
        out = extract_hair_colour(dummy_img, dummy_landmarks)
        assert out.tolist() == [3,3,3]


def test_extract_lab_values_from_photo(dummy_img, dummy_landmarks):
    with patch("face_features_extraction.white_balance", return_value=dummy_img.astype(np.float32)), \
         patch("face_features_extraction.get_face_landmarks", return_value=dummy_landmarks), \
         patch("face_features_extraction.extract_iris_colour", return_value=np.array([1,2,3])), \
         patch("face_features_extraction.extract_skin_colour", return_value=np.array([4,5,6])), \
         patch("face_features_extraction.extract_hair_colour", return_value=np.array([7,8,9])):
        out = extract_lab_values_from_photo(dummy_img, None, None)
        assert out == [1,2,3,4,5,6,7,8,9]