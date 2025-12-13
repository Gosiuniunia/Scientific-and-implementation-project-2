from unittest.mock import MagicMock, patch
import numpy as np

from utils.color_utils import (
    crop_img,
    apply_kmeans,
    get_lab_colour,
    get_color_between_points
)

def test_crop_img_returns_image_and_origin():
    img = np.zeros((100,100,3), dtype=np.uint8)

    lm = MagicMock()
    lm.x = 0.5
    lm.y = 0.5

    landmarks = [lm] * 10
    indices = [0, 1, 2]

    cropped, origin = crop_img(img, landmarks, indices)

    assert cropped.ndim == 3
    assert isinstance(origin, tuple)
    assert len(origin) == 2

@patch("utils.color_utils.cv2.kmeans")
def test_apply_kmeans(mock_kmeans):
    img = np.zeros((4,4,3), dtype=np.uint8)

    labels = np.zeros((16,1), dtype=np.int32)
    centers = np.array([[10,20,30]], dtype=np.uint8)

    mock_kmeans.return_value = (None, labels, centers)

    result = apply_kmeans(img, k=1)

    assert result.shape == img.shape
    assert (result == [10,20,30]).all()

@patch("utils.color_utils.cv2.cvtColor")
def test_get_lab_colour(mock_cvt):
    mock_cvt.return_value = np.array([[[50, 0, 0]]])

    bgr = np.array([[10,20,30], [10,20,30]])
    result = get_lab_colour(bgr)

    mock_cvt.assert_called_once()
    assert np.array_equal(result, np.array([50,0,0]))

def test_get_color_between_points_center():
    segmented_img = np.zeros((10,10,3), dtype=np.uint8)
    segmented_img[5,5] = [100,150,200]

    p1 = (4,4)
    p2 = (6,6)
    origin = (0,0)

    color = get_color_between_points(p1, p2, origin, segmented_img)

    assert np.array_equal(color, [100,150,200])

def test_get_color_between_points_clipped():
    segmented_img = np.zeros((5,5,3), dtype=np.uint8)
    segmented_img[4,4] = [1,2,3]

    p1 = (100,100)
    p2 = (200,200)
    origin = (0,0)

    color = get_color_between_points(p1, p2, origin, segmented_img)

    assert np.array_equal(color, [1,2,3])

def test_get_color_between_points_clipped():
    segmented_img = np.zeros((5,5,3), dtype=np.uint8)
    segmented_img[4,4] = [1,2,3]

    p1 = (100,100)
    p2 = (200,200)
    origin = (0,0)

    color = get_color_between_points(p1, p2, origin, segmented_img)

    assert np.array_equal(color, [1,2,3])

