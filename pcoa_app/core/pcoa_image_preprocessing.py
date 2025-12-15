import numpy as np
import gradio as gr
import io

class PCOAImageProcessor:
    def __init__(self, image: np.ndarray):
        self._original_image = image.copy() if image is not None else None
        self._processed_image = None
        self._original_image_path = "" 
        self._processed_image_path = ""

        self._width = image.shape[1] if image is not None and len(image.shape) >= 2 else 0
        self._height = image.shape[0] if image is not None and len(image.shape) >= 1 else 0
        self._channels = image.shape[2] if image is not None and len(image.shape) == 3 else 0

        self._is_validated = gr.State(False)
        self._is_preprocessed = gr.State(False)
        self._validation_message = ""

    def get_image(self) -> np.ndarray:
        return self._original_image

    def validate_image(self, image: np.ndarray) -> tuple[bool, str, np.ndarray]:
        """Validate the uploaded image"""
        if image is None:
            self._is_validated.set(False)
            self._validation_message = "No image uploaded. Please upload an image."
            return False, self._validation_message, None
        
        if len(image.shape) < 2:
            self._is_validated.set(False)
            self._validation_message = "Invalid image format. Please upload a valid image."
            return False, self._validation_message, None
        
        # check if image file type is .jpg or .png
        if self._original_image_path.lower().endswith(('.jpg', '.jpeg', '.png')) is False:
            self._is_validated.set(False)
            self._validation_message = "Unsupported image format. Please upload a .jpg or .png file."
            return False, self._validation_message, None
        
        self._is_validated.set(True)
        self._validation_message = "Image validated successfully."
        return True, self._validation_message, image
    
    def preprocess_image(self, image: np.ndarray) -> np.ndarray:
        # Here put the preprocessing logic, whitebalancing ect
        return image
    
    @staticmethod
    def preprocess_image(image: np.ndarray) -> tuple[np.ndarray, str]:
        """
        Complete pipeline for processing image from Gradio interface
        Returns: (processed_image, status_message)
        """
        if image is None:
            return None, "Please upload an image"
        
        img_processor = PCOAImageProcessor(image)
        is_valid, message, validated_image = img_processor.validate_image(image)
        
        if not is_valid:
            return None, message
        
        try:
            processed_image = img_processor.preprocess_image(validated_image)
            return processed_image, f"Image processed successfully. Shape: {processed_image.shape}"
        except Exception as e:
            return None, f"Error during preprocessing: {str(e)}"