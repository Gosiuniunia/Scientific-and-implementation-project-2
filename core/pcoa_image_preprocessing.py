import numpy as np
import gradio as gr
import io
import cv2
import os
from PIL import Image
from face_features_extraction import extract_face_features

class PCOAImageProcessor:
    def __init__(self, image: np.ndarray):
        self._original_image = image.copy() if image is not None else None
        self._processed_image = None
        self._original_image_path = "" 
        self._processed_image_path = ""
        self._is_preprocessed = gr.State(False)
        self._validation_message = ""

        self.model_path = "svc.pkl"

    def get_image(self) -> np.ndarray:
        return self._original_image
    
    def set_image(self, image: np.ndarray):
        self._original_image = image.copy() if image is not None else None

    def get_processed_image(self) -> np.ndarray:
        return self._processed_image
    
    def set_processed_image(self, image: np.ndarray):
        self._processed_image = image.copy() if image is not None else None

    def validate_image(self, file_path: str) -> tuple[bool, str, np.ndarray]:
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
        """Applies face feature extraction and white balancing to the image."""
        try:
            features = extract_face_features(image, "face_landmarker.task")
        except Exception as e:
            print(f"Error during face feature extraction: {e}")
            return None
        if features is None:
            return None
        self._processed_image = features
        self._is_preprocessed = True
        return self._processed_image
        
