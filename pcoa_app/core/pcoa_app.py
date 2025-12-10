import numpy as np
import gradio as gr
from PIL import Image, ImageDraw
from utils.enums import *
from core.pcoa_image_preprocessing import PCOAImage


class PCOAApp:
    def __init__(self):
        self.photo_uploaded = PhotoUploadStatus.NOT_UPLOADED
        self.photo_validated = PhotoValidationStatus.NOT_VALIDATED
        self.prediction_done = PredictionStatus.NOT_DONE
        self.predicted_type = None
        self.current_image = None

    def predict_color_type(self, image: PCOAImage) -> ColorType:
        # Example placeholder model
        return ColorType.SPRING
    
    def show_uploaded_image(self, image: np.ndarray):
        """Simply returns the uploaded image unchanged."""
        if image is None:
            return None
        
        self.photo_uploaded = PhotoUploadStatus.UPLOADED
        return image
    
    def run_prediction(self, image: np.ndarray):
        """Full prediction logic."""
        if image is None:
            return "Please upload an image first.", None, None, None
        
        # Convert numpy array → PCOAImage
        pcoa = PCOAImage(image)
        
        result = self.predict_color_type(pcoa)
        self.prediction_done = PredictionStatus.DONE
        self.predicted_types = result
        
        # Dummy example palette
        color1, color2, color3 = "#aabbcc", "#ccddaa", "#ffee88"
        
        return (
            f"Your Personal Color Type is: **{result.name}**",
            color1,
            color2,
            color3
        )


def build_ui(app: PCOAApp):
    with gr.Blocks() as demo:

        state = gr.State(app)

        gr.Markdown("# 🎨 Personal Color Analysis Systemsss")

        with gr.Row():
            with gr.Column():
                
                img_input = gr.Image(
                    label="Upload Your Photo",
                    type="numpy"
                )
                
                # OUTPUT image preview
                img_preview = gr.Image(
                    label="Your Image",
                    interactive=False
                )
                
                # Bind image preview
                img_input.change(
                    fn=lambda img, app: app.show_uploaded_image(img),
                    inputs=[img_input, state],
                    outputs=img_preview
                )

                analyze_button = gr.Button("Analyze Color")

            with gr.Column():
                result_text = gr.Markdown()
                color1 = gr.ColorPicker(label="Dominant Color 1")
                color2 = gr.ColorPicker(label="Dominant Color 2")
                color3 = gr.ColorPicker(label="Dominant Color 3")
        
        analyze_button.click(
            fn=lambda img, app: app.run_prediction(img),
            inputs=[img_input, state],
            outputs=[result_text, color1, color2, color3]
        )

    return demo

    



