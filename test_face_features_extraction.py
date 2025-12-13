
from unittest.mock import MagicMock, patch
import numpy as np

from face_features_extraction import (
    get_face_landmarks,
    extract_iris_colour,
    extract_skin_colour,
    extract_hair_colour,
    extract_lab_values_from_photo,
    extract_face_features
)

@patch("face_features_extraction.get_lab_colour")
def test_extract_skin_colour(mock_get_lab):
    img = np.zeros((100,100,3), dtype=np.uint8)

    landmark = MagicMock()
    landmark.x = 0.5
    landmark.y = 0.5

    face_landmarks = [[landmark, landmark, landmark, landmark, landmark, landmark]]

    mock_get_lab.return_value = np.array([50, 0, 0])

    result = extract_skin_colour(img, face_landmarks)

    assert np.array_equal(result, np.array([50, 0, 0]))

@patch("face_features_extraction.crop_img")
@patch("face_features_extraction.apply_kmeans")
@patch("face_features_extraction.get_color_between_points")
@patch("face_features_extraction.get_lab_colour")
def test_extract_iris_colour(
    mock_lab, mock_color_between, mock_kmeans, mock_crop
):
    img = np.zeros((100,100,3), dtype=np.uint8)

    lm = MagicMock()
    lm.x = 0.5
    lm.y = 0.5

    face = [lm] * 500
    face_landmarks = [face]

    mock_crop.return_value = (np.zeros((10,10,3)), (0,0))
    mock_kmeans.return_value = np.zeros((10,10,3))
    mock_color_between.return_value = np.array([10,10,10])
    mock_lab.return_value = np.array([20,0,0])

    result = extract_iris_colour(img, face_landmarks)

    assert np.array_equal(result, np.array([20,0,0]))

@patch("face_features_extraction.crop_img")
@patch("face_features_extraction.apply_kmeans")
@patch("face_features_extraction.get_color_between_points")
@patch("face_features_extraction.get_lab_colour")
def test_extract_hair_colour(
    mock_lab, mock_color_between, mock_kmeans, mock_crop
):
    img = np.zeros((100,100,3), dtype=np.uint8)

    lm = MagicMock()
    lm.x = 0.5
    lm.y = 0.5

    face = [lm] * 400
    face_landmarks = [face]

    mock_crop.return_value = (np.zeros((10,10,3)), (0,0))
    mock_kmeans.return_value = np.zeros((10,10,3))
    mock_color_between.return_value = np.array([15,15,15])
    mock_lab.return_value = np.array([30,0,0])

    result = extract_hair_colour(img, face_landmarks)

    assert np.array_equal(result, np.array([30,0,0]))

@patch("face_features_extraction.white_balance")
@patch("face_features_extraction.get_face_landmarks")
@patch("face_features_extraction.extract_iris_colour")
@patch("face_features_extraction.extract_skin_colour")
@patch("face_features_extraction.extract_hair_colour")
def test_extract_lab_values_from_photo(
    mock_hair, mock_skin, mock_iris, mock_landmarks, mock_wb
):
    img = np.zeros((100,100,3), dtype=np.uint8)

    mock_wb.return_value = img / 255
    mock_landmarks.return_value = ["face"]

    mock_iris.return_value = np.array([1,2,3])
    mock_skin.return_value = np.array([4,5,6])
    mock_hair.return_value = np.array([7,8,9])

    result = extract_lab_values_from_photo(img, "FL", "opts")

    assert result == [1,2,3,4,5,6,7,8,9]

@patch("face_features_extraction.init_face_landmark")
@patch("face_features_extraction.extract_lab_values_from_photo")
def test_extract_face_features(mock_extract, mock_init):
    mock_init.return_value = ("FL", "opts")
    mock_extract.return_value = [1,2,3]

    img = np.zeros((10,10,3))
    result = extract_face_features(img, "model.task")

    assert result == [1,2,3]
