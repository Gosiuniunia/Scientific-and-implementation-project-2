from utils.enums import *
from core.pcoa_app import PCOAApp, build_ui
from core.pcoa_ai_model import ColorAnalysisModel

if __name__ == "__main__":
    app = PCOAApp()
    ui = build_ui(app)
    model = ColorAnalysisModel()
    ui.launch()