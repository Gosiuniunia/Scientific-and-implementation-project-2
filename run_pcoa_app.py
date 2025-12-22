from core.pcoa_app import PCOAApp
from core.pcoa_ai_model import ColorAnalysisModel
from core.pcoa_image_preprocessing import PCOAImageProcessor

if __name__ == "__main__":
    ai_model = ColorAnalysisModel()
    image_processor = PCOAImageProcessor(None)
    app = PCOAApp(ai_model=ai_model, image_processor=image_processor)
    app._launch()