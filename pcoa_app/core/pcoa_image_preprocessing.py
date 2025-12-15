import numpy as np
import gradio as gr
import io
from utils.enums import *

class PCOAImageProcessor:
    def __init__(self, image: np.ndarray):
        self._original_image = image.copy() if image is not None else None
        self._processed_image = None
        self._original_image_path = "" 
        self._processed_image_path = ""

        self._width = image.shape[1] if image is not None and len(image.shape) >= 2 else 0
        self._height = image.shape[0] if image is not None and len(image.shape) >= 1 else 0
        self._channels = image.shape[2] if image is not None and len(image.shape) == 3 else 0

        self._is_validated = False
        self._is_preprocessed = False
        self._validation_message = ""

    def get_image(self) -> np.ndarray:
        return self._original_image

    def validate_image(self, image: np.ndarray) -> tuple[bool, str, np.ndarray]:
        """
        Validates image provided by user:
        - checks if image is not None
        - checks if image is ...     
        """
        if image is None:
            return False, "Error: No image uploaded", None
        
        try:
            if isinstance(image, np.ndarray):
                if len(image.shape) not in [2, 3]:
                    return False, "Error: Invalid image dimensions", None
                if len(image.shape) == 3 and image.shape[2] not in [3, 4]:
                    return False, "Error: Invalid number of channels", None
                return True, "Image validated successfully", image
                
            else:
                return False, "Error: Expected numpy array from Gradio", None
                
        except Exception as e:
            return False, f"Error: Invalid image format - {str(e)}", None
    
    def preprocess_image(self, image: np.ndarray) -> np.ndarray:
        # Here put the preprocessing logic, whitebalancing ect
        return image
    
    @staticmethod
    def process_gradio_image(image: np.ndarray) -> tuple[np.ndarray, str]:
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