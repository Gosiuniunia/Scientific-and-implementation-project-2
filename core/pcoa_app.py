import numpy as np
import gradio as gr
import time
from core.pcoa_image_preprocessing import PCOAImageProcessor

# added so Iphone HEIC images don't crash the app
from pillow_heif import register_heif_opener
register_heif_opener()

class PCOAApp:
    def __init__(self, ai_model, image_processor):
        self.gdpr_accepted = gr.State(False)
        self.ai_model = ai_model
        self.image_processor = image_processor
        self.build_ui()

    def build_ui(self):
        with gr.Blocks() as self.demo:
            gr.Markdown("# 🎨 Personal Color Analysis System")
            # GDPR Modal
            self.gdpr_modal, self.accept_btn, self.decline_btn, self.gdpr_message = (
                self.build_gdpr_modal()
            )
            # Main App (hidden initially)
            with gr.Group(visible=False) as self.main_app:

                # Create a horizontal layout with two columns
                with gr.Row():
                    # Left column: image upload section
                    right_margin_css = """
                        .right-margin {
                            margin-right: 50px;
                        } """
                    with gr.Column(elem_classes=right_margin_css, scale=1):
                        (
                            self.img_input,
                            self.status_message,
                            self.analyze_button,
                            self.submit_image_button,
                            self.progress_bar,
                        ) = self.build_photo_upload_section()

                    # Right column: results section (hidden initially)
                    with gr.Column(scale=1):
                        with gr.Group(visible=False) as self.results_section:
                            self.result_message = gr.Markdown("", visible=False)
                            self.description = gr.Markdown("", visible=False)
                            self.jewelerly_recommendation = gr.Markdown("", visible=False)
                            self.palette_html_output = gr.HTML(label="Interactive Palette")

                # Show analyze button when image is uploaded
                self.img_input.change(
                    fn=self.on_image_uploaded,
                    inputs=[self.img_input],
                    outputs=[
                        self.analyze_button,
                        self.submit_image_button,
                        self.status_message,
                    ],
                )

            self.submit_image_button.click(
                fn=self.on_image_submitted,
                inputs=[self.img_input],
                outputs=[
                    self.status_message,
                    self.analyze_button,
                    self.submit_image_button],
            )

            self.analyze_button.click(
                fn=self.run_prediction,
                inputs=[self.img_input],
                outputs=[
                    self.status_message,   # left column: status
                    self.description,  # right column: description
                    self.jewelerly_recommendation,  # right column: jewerly recommendation
                    self.palette_html_output,
                    self.analyze_button,   # left column: re-enable button
                    self.result_message,    # right column: prediction text
                    self.results_section
                ],
            )
            
            # GDPR buttons
            self.accept_btn.click(
                fn=self.accept_gdpr,
                inputs=[],
                outputs=[self.gdpr_modal, self.main_app, self.gdpr_accepted]
            )

            self.decline_btn.click(
                fn=self.decline_gdpr,
                inputs=[],
                outputs=[self.gdpr_message]
            )

        return self.demo
    
    def on_image_uploaded(self, img):
        if img is None:
            return (
                gr.update(visible=False),  # analyze_button
                gr.update(visible=True), # submit button
                gr.update(value="### Please upload an image of one person or take a photo", visible=True),
            )

        return (
            gr.update(visible=False),   # analyze_button
            gr.update(visible=True),  # submit button
            gr.update(value="### Image uploaded. \n ### Click Submit Image button to validate it", visible=True),
        )
    
    def on_image_submitted(self, img):
        if img is None:
            return (
                gr.update(value="### ❌ Upload an image first", visible=True),
                gr.update(visible=False),  # analyze_button
                gr.update(visible=True),   # submit button
            )
        else:
            # validate 
            is_valid, message, numpy_image = self.image_processor.validate_image(img)
            if not is_valid:
                return (
                    gr.update(value=f"### ❌ {message}", visible=True),
                    gr.update(visible=False),  # analyze_button
                    gr.update(visible=True),   # submit button
                )
            # if image is valid, set it in the processor
            self.image_processor.set_image(numpy_image)

            return (
                gr.update(value="### ✅ Image was valid and got submitted successfully! \n ### Click on the Analyze button to start analysis.", visible=True),
                gr.update(visible=True),   # analyze_button
                gr.update(visible=False),  # submit button
            )

    def build_gdpr_modal(self):
        with gr.Group(visible=True) as gdpr_modal:
            gr.Markdown("## 🔒 Privacy & Data Protection Notice")
            gr.Markdown("""
            **Personal Color Analysis System – GDPR Compliance**

            By using this application, you acknowledge and agree to the following:

            **Data Processing:**
            - Photos are processed locally for color analysis only
            - Images are not permanently stored
            - No third-party data sharing
            - Data is cleared when the session ends
            - You consent to image processing for analysis

            **Your Rights:**
            - Stop using the service at any time
            - Request deletion of your data
            - Right to data portability

            **Contact:** your-email@domain.com
            """)

            gdpr_message = gr.Markdown("", visible=False)

            with gr.Row():
                decline_btn = gr.Button("❌ Decline", variant="secondary", size="lg")
                accept_btn = gr.Button("✅ Accept & Continue", variant="primary", size="lg")

        return gdpr_modal, accept_btn, decline_btn, gdpr_message

    # Accept button logic
    def accept_gdpr(self):
            return (
                gr.update(visible=False),  # hide GDPR modal
                gr.update(visible=True),   # show main app
                True                       # set GDPR accepted state
            )

    # Decline button logic
    def decline_gdpr(self):
        return gr.update(
            value="❌ You must accept the privacy policy to use this app.",
            visible=True
        )
    
    def build_photo_upload_section(self):
        # Status message
        status_message = gr.Markdown(
            value="### Please upload an image in .jpg or .png format.",
            visible=True
        )
        # Image upload section
        img_input = gr.Image(
            label="Input photo",
            # needed for file validation step
            type='filepath',
            height=400,
            # app requirement - user can upload image or take photo with webcam
            sources=['upload', 'webcam']
        )

        # Image submission button
        submit_image_button = gr.Button(
            "Submit Image",
            variant="secondary",
            visible=False,
            size="lg"
        )
        
        # Analyze button (initially hidden)
        analyze_button = gr.Button(
            "🔍 Analyze",
            variant="primary",
            visible=False,
            size="lg"
        )
        
        # Progress indicator
        progress_bar = gr.Progress()

        return img_input, status_message, analyze_button, submit_image_button, progress_bar
    
    def run_prediction(self, image: np.ndarray, progress=gr.Progress()):
        """
        Run prediction pipeline and update UI elements accordingly
        Args:
            image (np.ndarray): Input image as a NumPy array.
            progress (gr.Progress): Gradio progress bar for process updates.
        Returns:
            tuple: Updated UI elements.
        """
        def handle_error(stage, error):
            print(f"!!! ERROR at [{stage}]: {error}")
            return (
                gr.update(value=f"❌ Failed at {stage}: {str(error)}", visible=True), # Status
                "", "",                    # Description, Jewelry
                "",                        # HTML Palette
                gr.update(interactive=True), # Button
                gr.update(visible=False),    # Result Message
                gr.update(visible=False)     # Result Container
            )

        # Image upload
        try:
            progress(0.1, desc="Starting analysis...")
            time.sleep(0.5)
            raw_img = self.image_processor.get_image()
            if raw_img is None:
                raise ValueError("No image data found in processor.")
        except Exception as e:
            return handle_error("Image Loading", e)

        # Image preprocessing (feature extraction)
        try:
            progress(0.3, desc="Preprocessing image...")
            time.sleep(0.5)
            preprocessed_image = self.image_processor.preprocess_image(raw_img)
            self.image_processor.set_processed_image(preprocessed_image)
            print("--- Preprocessing: SUCCESS ---")
        except Exception as e:
            return handle_error("Preprocessing", e)

        # Predicting the seasonal type
        try:
            progress(0.6, desc="Performing prediction...")
            time.sleep(0.5)
            current_img = self.image_processor.get_processed_image()
            prediction_results = self.ai_model.predict(current_img)
            
            if not prediction_results:
                raise ValueError("Couldn't give result for given image. Please upload different one.")
            
            print(f"--- Prediction: SUCCESS (Result: {prediction_results}) ---")
        except Exception as e:
            return handle_error("AI Prediction", e)

        # Getting color pallete and descriptions based on predicted color type
        try:
            color_palette = self.ai_model.get_palette_info(prediction_results)
            if not color_palette or len(color_palette) < 3:
                color_palette = ["#808080", "#A0A0A0", "#C0C0C0", "#D0D0D0", "#E0E0E0", "#F0F0F0", "#B0B0B0", "#909090"]
                
            description = self.ai_model.get_description(prediction_results)
            jewelry = self.ai_model.get_jewelry_recommendation(prediction_results)
            full_palette_html = self._generate_palette_html(prediction_results)
            print(f"--- Recommendations Retrieval: SUCCESS ---")
        except Exception as e:
            return handle_error("Data Retrieval", e)
        
        # build prediction results string with emoji for each season
        season_emojis = {
            "spring": "🌸",
            "summer": "☀️",
            "autumn": "🍂",
            "winter": "❄️"
        }
        emoji = season_emojis.get(prediction_results.lower(), "")
        prediction_results_string = f"{prediction_results.capitalize()} {emoji}"

        # Showing recommendations in the UI
        try:
            progress(1.0, desc="Analysis complete!")
            time.sleep(0.5)
            return (
                gr.update(value="### ✅ Image analyzed successfully!", visible=True),
                gr.update(value=f"### 📝 Description\n{description} <br><br>\n", visible=True), 
                gr.update(value=f"### 💍 Jewelry recommendations\n{jewelry} <br>\n ### 🎨 Recommended color palette with color codes: <br><br>\n", visible=True),
                full_palette_html,
                gr.update(interactive=True),
                gr.update(
                    value=f"### 🎨 Your seasonal color type:<br>\n <h2 style='text-align: center;'> {prediction_results_string} </h2><br><br>",
                    visible=True
                ),
                gr.update(visible=True)
            )
        except Exception as e:
            return handle_error("UI Update", e)
        
    def _generate_palette_html(self, season):
        html_colors = ""
        try:
            colors = self.ai_model.get_palette_info(season)
        except Exception as e:
            print(f"Error retrieving color palette: {e}")
            return "<div style='color:red'>Season not found</div>"
        for c in colors:
                    html_colors += f"""
                    <div style="display:flex; flex-direction:column; align-items:center; justify-content:flex-start;">
                        <div style="
                            background-color: {c}; 
                            width: 100%; 
                            padding-bottom: 100%; 
                            border-radius: 12px; 
                            box-shadow: 0 4px 8px rgba(0,0,0,0.3);
                            transition: transform 0.3s cubic-bezier(0.25, 0.8, 0.25, 1);
                            cursor: pointer;
                            "
                            onmouseover="this.style.transform='scale(1.15)'; this.style.zIndex='10';"
                            onmouseout="this.style.transform='scale(1.0)'; this.style.zIndex='1';">
                        </div>
                        <div style="
                            margin-top: 10px; 
                            ">
                            {c}
                        </div>
                    </div>
                    """
        return f"""
        <div style="
            display: grid; 
            grid-template-columns: repeat(4, 1fr); 
            gap: 15px; 
            width: 100%; 
            max-width: 400px;
            margin: 0 auto;   
        ">
            {html_colors}
        </div>
        """
    
    def _launch(self):
        self.demo.launch()