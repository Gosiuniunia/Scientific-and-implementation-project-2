from core.pcoa_app import PCOAApp
from core.pcoa_ai_model import ColorAnalysisModel
from core.pcoa_image_preprocessing import PCOAImageProcessor
from core.pcoa_result_visualisation import ResultVisualizer

"""
Script launching the Personal Color Analysis application.
"""

if __name__ == "__main__":
    ai_model = ColorAnalysisModel()
    image_processor = PCOAImageProcessor(None)
    result_visualiser = ResultVisualizer()
    app = PCOAApp(ai_model=ai_model, image_processor=image_processor, result_visualiser=result_visualiser)
    app._launch()