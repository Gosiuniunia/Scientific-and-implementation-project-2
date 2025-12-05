"""
This module provides helper functions for color extraction and processing from facial regions.
It supports white balancing, image cropping based on facial landmarks, color segmentation using K-Means,
and color space conversion to LAB formats.


"""

import numpy as np
import cv2
from utils.white_balancing.classes import WBsRGB as wb_srgb

def white_balance(img):
  """
  Performs white-balancing of the image
    Ref: Afifi, Mahmoud, et al. "When color constancy goes wrong: Correcting improperly white-balanced images." 
    Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. 2019.

  Args:
        img (np.ndarray): Input image in BGR format.

  Returns:
        wb_img (np.ndarray): White-balanced image in RGB format.
  """
  wbModel = wb_srgb.WBsRGB()
  wb_img = wbModel.correctImage(img)
  return wb_img


def crop_img(img, landmarks, indices):
    """
    Crops a region of the image based on facial landmarks.

    Args:
        img (np.ndarray): Input image in BGR format.
        landmarks (List[NormalizedLandmark]): List of facial landmarks.
        indices (List[int]): Indices of the landmarks to define the region.

    Returns:
        Tuple[np.ndarray, Tuple[int, int]]: 
            - Cropped region of the image (np.ndarray).
            - Origin (x, y) of the cropped region relative to the original image.
    """
    h, w, _ = img.shape
    points = np.array(
        [[int(landmarks[i].x * w), int(landmarks[i].y * h)] for i in indices]
    )
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(mask, [points], 255)
    masked_img = cv2.bitwise_and(img, img, mask=mask)
    x, y, w_box, h_box = cv2.boundingRect(points)
    x = max(x, 0)
    y = max(y, 0)
    w_box = min(w_box, img.shape[1] - x)
    h_box = min(h_box, img.shape[0] - y)
    cropped_img = masked_img[y : y + h_box, x : x + w_box]
    return cropped_img, (x, y)


def apply_kmeans(img, k=5):
    """
    Applies K-Means clustering to segment colors in the image.

    Args:
        img (np.ndarray): Input image in BGR format.
        k (int, optional): Number of clusters. Defaults to 4.

    Returns:
        Tuple[np.ndarray, np.ndarray]:
            - Cluster centers (BGR colors) as np.ndarray of shape (k, 3).
            - Segmented image with colors replaced by their cluster center.
    """
    img_data = img.reshape((-1, 3))
    img_data = np.float32(img_data)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
    _, labels, centers = cv2.kmeans(
        img_data, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS
    )
    centers = np.uint8(centers)
    segmented_img = centers[labels.flatten()]
    segmented_img = segmented_img.reshape(img.shape)
    return segmented_img


def get_lab_colour(bgr_array):
    """
    Converts a list of BGR colors to average LAB colour representations.

    Args:
        bgr_array (List[np.ndarray] or np.ndarray): List or array of BGR colors.

    Returns:
        np.ndarray: Concatenated LAB average values (length 3).
    """
    avg_bgr = np.mean(bgr_array, axis=0).astype(np.uint8)
    avg_bgr_reshaped = avg_bgr.reshape((1, 1, 3))
    avg_lab = cv2.cvtColor(avg_bgr_reshaped, cv2.COLOR_BGR2Lab)[0, 0]
    return np.concatenate([avg_lab])


def get_color_between_points(p1, p2, crop_origin, segmented_img):
    """
    Gets the color from the image at the midpoint between two points: p1 and p2.

    Args:
        p1 (Tuple[float, float]): First point (x, y).
        p2 (Tuple[float, float]): Second point (x, y).
        crop_origin (Tuple[int, int]): Origin (x, y) of the crop in original image.
        segmented_img (np.ndarray): Segmented image (from KMeans).

    Returns:
        np.ndarray: BGR color at the midpoint between p1 and p2.
    """
    cx = int((p1[0] + p2[0]) / 2) - crop_origin[0]
    cy = int((p1[1] + p2[1]) / 2) - crop_origin[1]

    h, w = segmented_img.shape[:2]
    cx = np.clip(cx, 0, w - 1)
    cy = np.clip(cy, 0, h - 1)

    return segmented_img[cy, cx]
