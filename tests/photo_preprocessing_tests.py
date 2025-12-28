import sys
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

import pytest
from core.pcoa_image_preprocessing import PCOAImageProcessor
import numpy as np
from PIL import Image
from unittest.mock import patch


class TestPCOAImageProcessor: 
    @pytest.fixture
    def processor(self):
        """
        Creates a new PCOAImageProcessor instance with a dummy initial image with each test run.
        """
        dummy_img = np.zeros((100, 100, 3), dtype=np.uint8)
        return PCOAImageProcessor(dummy_img)

    @pytest.fixture
    def valid_image_file(self, tmp_path):
        """
        Returns a path to valid test image in .png format
        """
        return "test_images/valid_test_image.png"

    def test_initialization(self):
        """
        Checks if initialization of PCOAImageProcessor object is correct.
        Checks:
        - if getter returns any value for both processed and unprocessed image
        - if getter doesn't change image shape
        """
        img = np.zeros((50, 50, 3), dtype=np.uint8)
        image_processor = PCOAImageProcessor(img)
        
        assert image_processor.get_image() is not None
        assert image_processor.get_image().shape == (50, 50, 3)
        assert image_processor.get_processed_image() is None

    def test_validate_no_file_path(self, processor):
        """
        Checks behaviour when validation is run on no uploaded file
        Checks:
        - if is_valid is set to False
        - if message contains phrase about no image uploaded
        """
        is_valid, msg, img = processor.validate_image("")
        assert is_valid is False
        assert "No image uploaded" in msg

    def test_validate_wrong_extension(self, processor):
        """
        Checks status and error message in situation where file with wring extension is loaded
        """
        is_valid, msg, img = processor.validate_image("test_images/Selena-Gomez-2010.webp")
        assert is_valid is False
        assert "Unsupported image format" in msg

    def test_validate_corrupted_file(self, processor, tmp_path):
        """
        Checks status and error message when a corrupted file is provided to the application.
        Provided file is a file with supported image extension but unmatching text content. 
        """
        
        is_valid, msg, img = processor.validate_image("test_images/corrupted_file.jpg")
        assert is_valid is False
        assert "File validaton error" in msg

    @patch("face_features_extraction.get_number_of_faces")
    def test_validate_success_one_face(self, mock_count_faces, processor):
        """
        Tests the path with valid file provided, with only one face present.
        """
        mock_count_faces.return_value = 1
        is_valid, msg, img_array = processor.validate_image("tests/test_images/valid_test_image.png")
        assert is_valid is True
        assert "validated successfully" in msg
        assert isinstance(img_array, np.ndarray)

    @patch("face_features_extraction.get_number_of_faces")
    def test_validate_failure_zero_faces(self, mock_count_faces, processor):
        """
        Test a failure path where image without a face is provided
        """
        mock_count_faces.return_value = 0
        is_valid, msg, img_array = processor.validate_image("tests/test_images/cabinet.jpg")
        assert is_valid is False
        assert "No face detected" in msg

    @patch("face_features_extraction.extract_face_features")
    def test_preprocess_success(self, mock_extract, processor):
        """
        Tests that extracted features are stored correctly in image processor object.
        """
        correct_img_path = "tests/test_images/valid_test_image.png"
        with Image.open(correct_img_path) as img:
            img = img.convert("RGB")
            image_numpy = np.array(img)
        
        dummy_features = image_numpy
        mock_extract.return_value = dummy_features
        
        result = processor.preprocess_image(image_numpy)
        
        assert result is not None
        assert processor.get_processed_image() is not None

    @patch("face_features_extraction.extract_face_features")
    def test_preprocess_failure_exception(self, mock_extract, processor):
        """
        Tests handling of situation when features extraction process fails.
        Passing a cabinet image as a test one.
        """
        mock_extract.side_effect = Exception("Model failed")
        with Image.open("tests/test_images/cabinet.jpg") as img:
                img = img.convert("RGB")
                image_numpy = np.array(img)
        result = processor.preprocess_image(image_numpy)
        assert result is None
        assert processor.get_processed_image() is None