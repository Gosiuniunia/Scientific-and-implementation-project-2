import sys
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

from unittest.mock import patch
from core.pcoa_ai_model import ColorAnalysisModel

class TestColorAnalysisModel:
    @patch("core.pcoa_ai_model.predict_class")
    def test_predict(self, mock_predict_func):
        """
        Checks result correctness returned by external predict_class function present in PCOA_prediction.py file.
        """
        fake_features = [67, 129, 129, 209, 135, 141, 82, 140, 142]
        mock_predict_func.return_value = "spring"

        model = ColorAnalysisModel()
        result = model.predict(fake_features)
        assert result == "spring"
        mock_predict_func.assert_called_once_with("svc.pkl", fake_features)