"""
This script extracts color features from facial regions (iris, skin, eyebrows) using MediaPipe Face Landmarker.
It processes loaded image and computes LAB color values.

"""


import mediapipe as mp
import cv2
import numpy as np
import pandas as pd
from utils.color_utils import white_balance, crop_img, apply_kmeans, get_lab_colour, get_color_between_points
import os


def init_face_landmark(model_path):
    """
    Initializes the MediaPipe FaceLandmarker model for facial landmark detection.

    Args:
        model_path (str): Path to the '.task' model file.

    Returns:
        tuple: A tuple containing the FaceLandmarker class and its configuration options (FaceLandmarkerOptions).
    """
    BaseOptions = mp.tasks.BaseOptions
    FaceLandmarker = mp.tasks.vision.FaceLandmarker
    FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
    VisionRunningMode = mp.tasks.vision.RunningMode

    options = FaceLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=model_path),
        running_mode=VisionRunningMode.IMAGE,
    )
    return FaceLandmarker, options


def get_face_landmarks(FaceLandmarker, options, img_rgb):
    """
    Detects facial landmarks from an RGB image.

    Args:
        FaceLandmarker: MediaPipe FaceLandmarker class.
        options: Configuration options for the landmark model.
        img_rgb (np.ndarray): The input image in RGB format.

    Returns:
        list: A list of detected facial landmarks.
    """
    with FaceLandmarker.create_from_options(options) as landmarker:
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)
        result = landmarker.detect(mp_image)
        return result.face_landmarks


def extract_iris_colour(img, face_landmarks):
    """
    Extracts the iris color from the image using facial landmarks.

    Args:
        img (np.ndarray): The original image in BGR format.
        face_landmarks (list): List of facial landmarks.

    Returns:
        np.ndarray: A LAB color vector representing the iris color.
    """
    right_iris_indices = [374, 476, 475, 474]
    left_iris_indices = [469, 145, 471, 159]
    pupil_indices = [468, 473]
    for face_landmarks in face_landmarks:
        left_iris_img, left_origin = crop_img(img, face_landmarks, left_iris_indices)
        right_iris_img, right_origin = crop_img(img, face_landmarks, right_iris_indices)
        segmented_img_li = apply_kmeans(left_iris_img)
        segmented_img_ri = apply_kmeans(right_iris_img)
        left_iris_colour = get_color_between_points(
            (
                face_landmarks[469].x * img.shape[1],
                face_landmarks[469].y * img.shape[0],
            ),
            (
                face_landmarks[145].x * img.shape[1],
                face_landmarks[145].y * img.shape[0],
            ),
            left_origin,
            segmented_img_li,
        )
        right_iris_colour = get_color_between_points(
            (
                face_landmarks[374].x * img.shape[1],
                face_landmarks[374].y * img.shape[0],
            ),
            (
                face_landmarks[476].x * img.shape[1],
                face_landmarks[476].y * img.shape[0],
            ),
            right_origin,
            segmented_img_ri,
        )

        iris_colour = get_lab_colour([right_iris_colour, left_iris_colour])
        return iris_colour


def extract_skin_colour(img, face_landmarks):
    """
    Extracts skin color by sampling predefined facial landmarks.

    Args:
        img (np.ndarray): The original image in BGR format.
        face_landmarks (list): List of facial landmarks.

    Returns:
        np.ndarray: A LAB color vector representing the skin tone.
    """
    skin_extraction_landmarks = [195, 5]
    skin_colours = []
    for face in face_landmarks:
        for idx in skin_extraction_landmarks:
            if idx < len(face):
                landmark = face[idx]
                x = int(landmark.x * img.shape[1])
                y = int(landmark.y * img.shape[0])
                if 0 <= x < img.shape[1] and 0 <= y < img.shape[0]:
                    color = img[y, x]
                    skin_colours.append(color)
    skin_colour = get_lab_colour(skin_colours)

    return skin_colour


def extract_hair_colour(img, face_landmarks):
    """
    Extracts eyebrow (hair) color using facial landmarks.

    Args:
        img (np.ndarray): The original image in BGR format.
        face_landmarks (list): List of facial landmarks.

    Returns:
        np.ndarray: A LAB color vector representing eyebrow color.
    """
    left_eyebrow = [70, 63, 105, 66, 107, 55, 65, 52, 53]
    right_eyebrow = [336, 296, 334, 293, 276, 283, 282, 295, 285]
    for face_landmarks in face_landmarks:
        left_eyebrow_img, left_origin = crop_img(img, face_landmarks, left_eyebrow)
        right_eyebrow_img, right_origin = crop_img(img, face_landmarks, right_eyebrow)
        segmented_img_le = apply_kmeans(left_eyebrow_img)
        segmented_img_re = apply_kmeans(right_eyebrow_img)
        left_eyebrow_colour = get_color_between_points(
            (
                face_landmarks[105].x * img.shape[1],
                face_landmarks[105].y * img.shape[0],
            ),
            (face_landmarks[65].x * img.shape[1], face_landmarks[65].y * img.shape[0]),
            left_origin,
            segmented_img_le,
        )

        right_eyebrow_colour = get_color_between_points(
            (
                face_landmarks[334].x * img.shape[1],
                face_landmarks[334].y * img.shape[0],
            ),
            (
                face_landmarks[295].x * img.shape[1],
                face_landmarks[295].y * img.shape[0],
            ),
            right_origin,
            segmented_img_re,
        )
    eyebrow_colour = get_lab_colour([left_eyebrow_colour, right_eyebrow_colour])
    return eyebrow_colour


def extract_lab_values_from_photo(img, FaceLandmarker, options):
    """
    Loads an image, detects facial landmarks, and extracts iris, skin, and eyebrow colors.

    Args:
        img (str): loaded image.
        FaceLandmarker: MediaPipe FaceLandmarker class.
        options: Configuration options for the landmark model.

    Returns:
        list: A flattened list of LAB color features from iris, skin, and eyebrow.
    """
    balanced_img = white_balance(img)
    img = (balanced_img * 255).astype(np.uint8)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    face_landmarks = get_face_landmarks(FaceLandmarker, options, img_rgb)

    iris_colour = extract_iris_colour(img, face_landmarks)
    skin_colour = extract_skin_colour(img, face_landmarks)
    eyebrow_colour = extract_hair_colour(img, face_landmarks)
    extracted_values = np.concatenate(
        [iris_colour, skin_colour, eyebrow_colour]
    ).tolist()
    return extracted_values


def extract_face_features(image, model_path):
    """ 
    Extracts color features in LAB color spaces from uploaded images.

    Args:
        image (str): Image given as a 
        model_path (str): Path to the face landmarker model.

    Saves:
         - CSV file at output_csv_path containing extracted features and labels.
    """
    FaceLandmarker, options = init_face_landmark(model_path)

    extracted_values = extract_lab_values_from_photo(image, FaceLandmarker, options)
    return extracted_values


# image = 'img.png'
# img = cv2.imread(image)
# print(extract_face_features(img, 'C:/studia/P_nw/face_landmarker.task'))