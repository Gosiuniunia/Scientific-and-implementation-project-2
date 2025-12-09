import pytest
import numpy as np

from utils.color_utils import (
    crop_img,
    apply_kmeans,
    get_lab_colour,
    get_color_between_points
)

def test_crop_img_basic():
    img = np.ones((100, 100, 3), dtype=np.uint8)
    class LM:
        def __init__(self, x, y):
            self.x = x
            self.y = y

    landmarks = [
        LM(0.2, 0.1),
        LM(0.4, 0.1),
        LM(0.2, 0.2),
        LM(0.4, 0.2),
    ]
    indices = [0, 1, 2, 3]

    cropped, origin = crop_img(img, landmarks, indices)

    assert origin == (20, 10)
    assert cropped.shape[0] == 21
    assert cropped.shape[1] == 21

def test_apply_kmeans():
    img = np.zeros((20, 20, 3), dtype=np.uint8)
    img[:10] = [0, 0, 255]
    img[10:] = [0, 255, 0]

    segmented = apply_kmeans(img, k=2)
    assert segmented.shape == img.shape
    assert isinstance(segmented, np.ndarray)

def test_get_lab_colour():
    bgr_list = [
        np.array([10, 20, 30], dtype=np.uint8),
        np.array([20, 30, 40], dtype=np.uint8),
    ]

    out = get_lab_colour(bgr_list)
    assert out.shape == (3,)
    assert out.dtype == np.uint8

def test_get_color_between_points_midpoint_inside():
    seg = np.zeros((10, 10, 3), dtype=np.uint8)
    seg[5, 5] = [100, 150, 200]

    p1 = (6, 6)
    p2 = (4, 4)
    origin = (0, 0)

    color = get_color_between_points(p1, p2, origin, seg)

    assert color.tolist() == [100, 150, 200]


def test_get_color_between_points_clamped():
    seg = np.zeros((5, 5, 3), dtype=np.uint8)
    seg[0, 0] = [10, 20, 30]

    p1 = (-100, -100)
    p2 = (-50, -50)
    origin = (0, 0)

    color = get_color_between_points(p1, p2, origin, seg)

    assert color.tolist() == [10, 20, 30]