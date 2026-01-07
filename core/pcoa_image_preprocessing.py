import numpy as np
import gradio as gr
import os
from PIL import Image
from core.utils.face_features_extraction import extract_face_features, get_number_of_faces

class PCOAImageProcessor:
    """
    Class representing preprocessing module for PCOA task.
    Provides methods to validate and extract face features for color analysis.
    """

    def __init__(self, image: np.ndarray):
        """
        Initializes the PCOAImageProcessor with an image.
        Args:
            image (np.ndarray): Input image as a NumPy array.
        Fields:
            _original_image (np.ndarray): Original uploaded image.
            _processed_image (np.ndarray): Preprocessed image after feature extraction.
            _original_image_path (str): Path to the original image file.
            _processed_image_path (str): Path to the processed image file.
            _is_preprocessed (gr.State): State indicating if the image has been preprocessed.
            _validation_message (str): Message regarding image validation status.
            model_path (str): Path to the pre-trained model for feature extraction.
            landmarker_path (str): Path to the face landmarker task file.
        """
        self._original_image = image.copy() if image is not None else None
        self._processed_image = None
        self._original_image_path = "" 
        self._processed_image_path = ""
        self._is_preprocessed = gr.State(False)
        self._validation_message = ""
        self.model_path = "svc.pkl"
        self.landmarker_path = 'C:/studia/P_nw/face_landmarker.task'

    def get_image(self) -> np.ndarray:
        """
        Gets the original image.
        Returns:
            np.ndarray: Original image as a NumPy array.
        """
        return self._original_image
    
    def set_image(self, image: np.ndarray):
        """
        Sets the original image.
        Args:
            image (np.ndarray): Input image as a NumPy array.
        """
        self._original_image = image.copy() if image is not None else None

    def get_processed_image(self) -> np.ndarray:
        """
        Gets the processed image.
        Returns:
            np.ndarray: Processed image as a NumPy array.
        """
        return self._processed_image
    
    def set_processed_image(self, image: np.ndarray):
        """
        Sets the processed image.
        Args:
            image (np.ndarray): Processed image as a NumPy array."""
        self._processed_image = image.copy() if image is not None else None

    def validate_image(self, file_path: str) -> tuple[bool, str, np.ndarray]:
        """
        Function to validate the uploaded image file.
        1. Checks if a file is uploaded.
        2. Checks file extension (only JPG and PNG allowed).
        3. Checks if the file is not corrupted and can be opened as an image.
        4. Converts the image to a NumPy array.

        Args:
            file_path (str): Path to the uploaded image file.
        Returns:
            tuple: (is_valid (bool), message (str), image_numpy (np.ndarray or None))
        1. is_valid: True if the image is valid, False otherwise.
        2. message: Validation message.
        3. image_numpy: NumPy array of the image if valid, None otherwise.
        """

        # Check if file got uploaded
        if not file_path:
            return False, "No image uploaded.", None

        # File extension check
        _, ext = os.path.splitext(file_path)
        print(f"File extension: {ext}")
        allowed_extensions = {'.jpg', '.jpeg', '.png'}
        
        if ext.lower() not in allowed_extensions:
            return False, f"Unsupported image format ({ext}). Please use JPG or PNG.", None

        # Trial of opening the file to check for corruption
        try:
            with Image.open(file_path) as img:
                img.verify()
            with Image.open(file_path) as img:
                # Converting image to RGB scale
                img = img.convert("RGB") 
                image_numpy = np.array(img)
                # return True, "Image validated successfully.", image_numpy
        except Exception as e:
            return False, f"File validaton error: {str(e)}", None
        
        # Checking number of faces present on the uploaded photo
        print("trial of faces detection check")
        try:
            with Image.open(file_path) as img:
                img = img.convert("RGB")
                image_numpy = np.array(img)
                no_faces = get_number_of_faces(image_numpy, self.landmarker_path)
                print(f"Number of faces spotted: {no_faces}")
                if no_faces == 1:
                    return True, "Image validated successfully.", image_numpy
                elif no_faces == 0:
                    return False, "No face detected. Please make sure that you uploaded an image with one face present.", None
                elif no_faces > 1:
                    return False, "Detected multiple faces. Please upload a picture of one person only.", None
        except Exception as e:
            return False, f"File validation error: {str(e)}", None
        
    def preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """
        Applies face feature extraction and white balancing to the uploaded image.
        Args:
            image (np.ndarray): Input image as a NumPy array.
        Returns:
            np.ndarray: Preprocessed image as a NumPy array.
        1. If face feature extraction fails, returns None.
        2. If successful, stores the processed image and updates the preprocessing state.
        """
        try:
            features = extract_face_features(image, self.landmarker_path)
        except Exception as e:
            print(f"Error during face feature extraction: {e}")
            return None
        if features is None:
            return None
        self._processed_image = features
        self._is_preprocessed = True
        return self._processed_image
        
