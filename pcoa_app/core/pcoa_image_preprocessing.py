import numpy as np
import gradio as gr
from PIL import Image, ImageDraw
import io
from utils.enums import *

class PCOAImage:
    def __init__(self, image: np.ndarray):
        self.image = image

    def get_image(self) -> np.ndarray:
        return self.image

    def validate_image(self, image: np.ndarray) -> bool:
        if image is None:
            return "Error: No image uploaded"
        try:
            # Convert to PIL Image if needed
            if isinstance(image, np.ndarray):
                img = Image.fromarray(image)
            else:
                img = image 
            # Validate format
            if img.format not in ['JPEG', 'PNG', None]:
                img_byte_arr = io.BytesIO()
                img.save(img_byte_arr, format='PNG')
                img = Image.open(img_byte_arr)
            return img
        except Exception as e:
            return f"Error: Invalid image format - {str(e)}"
    
    def preprocess_image(self, image: np.ndarray) -> np.ndarray:
        # Here put the preprocessing logic, whitebalancing ect
        return image