import numpy as np
import gradio as gr
import io
import cv2
import os
from PIL import Image
from utils.color_utils import white_balance

class PCOAImageProcessor:
    def __init__(self, image: np.ndarray):
        self._original_image = image.copy() if image is not None else None
        self._processed_image = None
        self._original_image_path = "" 
        self._processed_image_path = ""

        self._width = image.shape[1] if image is not None and len(image.shape) >= 2 else 0
        self._height = image.shape[0] if image is not None and len(image.shape) >= 1 else 0
        self._channels = image.shape[2] if image is not None and len(image.shape) == 3 else 0

        self._is_preprocessed = gr.State(False)
        self._validation_message = ""

    def get_image(self) -> np.ndarray:
        return self._original_image
    
    def set_image(self, image: np.ndarray):
        self._original_image = image.copy() if image is not None else None
        self._width = image.shape[1] if image is not None and len(image.shape) >= 2 else 0
        self._height = image.shape[0] if image is not None and len(image.shape) >= 1 else 0
        self._channels = image.shape[2] if image is not None and len(image.shape) == 3 else 0
    

    def validate_image(self, file_path: str) -> tuple[bool, str, np.ndarray]:
        print(f"Validating image at path: {file_path}")

        if not file_path:
            return False, "No image uploaded.", None

        # 2. Sprawdź rozszerzenie pliku
        # Pobieramy rozszerzenie i zamieniamy na małe litery
        _, ext = os.path.splitext(file_path)
        print(f"File extension: {ext}")
        allowed_extensions = {'.jpg', '.jpeg', '.png'}
        
        if ext.lower() not in allowed_extensions:
            return False, f"Unsupported format ({ext}). Please use JPG or PNG.", None

        # 3. Sprawdź czy plik nie jest uszkodzony i wczytaj do NumPy
        try:
            # Otwieramy przez PIL, żeby sprawdzić nagłówki
            with Image.open(file_path) as img:
                img.verify() # Sprawdza spójność pliku (czy to naprawdę obraz)
                
            # Jeśli verify() przeszło, musimy otworzyć go ponownie do odczytu danych
            # (verify "zamyka" plik i przesuwa wskaźnik)
            with Image.open(file_path) as img:
                # Konwersja do RGB (bo PNG może mieć RGBA, a JPG RGB)
                img = img.convert("RGB") 
                image_numpy = np.array(img)
                return True, "Image validated successfully.", image_numpy

        except Exception as e:
            return False, f"File validaton error: {str(e)}", None
    
    def preprocess_image(self, image: np.ndarray) -> np.ndarray:
        white_balanced_img = white_balance(image)
        return white_balanced_img
    
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