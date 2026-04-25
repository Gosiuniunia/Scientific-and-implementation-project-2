from core.pcoa_app import PCOAApp
from core.pcoa_ai_model import ColorAnalysisModel
from core.pcoa_image_preprocessing import PCOAImageProcessor
from core.pcoa_result_visualisation import ResultVisualizer
from core.microservice_orchestrator import AIServiceOrchestrator

"""
Script launching the Personal Color Analysis application.
"""

AI_SERVICE_URL = "http://127.0.0.1:8000"

if __name__ == "__main__":
    # ai_model = ColorAnalysisModel()
    image_processor = PCOAImageProcessor(None)
    result_visualiser = ResultVisualizer()
    ai_model_orchestrator = AIServiceOrchestrator(AI_SERVICE_URL)
    app = PCOAApp(
        image_processor=image_processor,
        result_visualiser=result_visualiser,
        ai_model_orchestrator=ai_model_orchestrator,
    )
    app._launch()
