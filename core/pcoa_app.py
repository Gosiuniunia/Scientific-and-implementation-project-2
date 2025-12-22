import numpy as np
import gradio as gr
from core.pcoa_image_preprocessing import PCOAImageProcessor

# added so Iphone HEIC images don't crash the app
from pillow_heif import register_heif_opener
register_heif_opener()

class PCOAApp:
    def __init__(self, ai_model, image_processor):
        self.gdpr_accepted = gr.State(False)
        self.prediction_done = gr.State(False)
        self.predicted_type = gr.State("")
        self.ai_model = ai_model
        self.image_processor = image_processor
        self.build_ui()

    def build_ui(self):
        with gr.Blocks() as self.demo:
            # GDPR Modal
            self.gdpr_modal, self.accept_btn, self.decline_btn, self.gdpr_message = (
                self.build_gdpr_modal()
            )
            # Main App (hidden initially)
            with gr.Group(visible=False) as self.main_app:
                # Create a horizontal layout with two columns
                with gr.Row():
                    # Left column: image upload section
                    with gr.Column(scale=1):
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
                            self.primary_color = gr.ColorPicker(label="Primary Color", interactive=False)
                            self.secondary_color = gr.ColorPicker(label="Secondary Color", interactive=False)
                            self.accent_color = gr.ColorPicker(label="Accent Color", interactive=False)
                            self.recommendations = gr.Markdown("", visible=False)

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
                    self.primary_color,    # right column: color pickers
                    self.secondary_color,
                    self.accent_color,
                    self.recommendations,  # right column: recommendations
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
                gr.update(value="Please upload an image to begin analysis", visible=True),
            )

        return (
            gr.update(visible=False),   # analyze_button
            gr.update(visible=True),  # submit button
            gr.update(value="Image uploaded. Click Submit to validate", visible=True),
        )
    
    def on_image_submitted(self, img):
        if img is None:
            return (
                gr.update(value="❌ Please upload an image first", visible=True),
                gr.update(visible=False),  # analyze_button
                gr.update(visible=True),   # submit button
            )
        
        else:
            # validate 
            print(f"Submitting image for validation: {img}")
            print(f"Type of img in pcoa app: {type(img)}")
            is_valid, message, numpy_image = self.image_processor.validate_image(img)
            if not is_valid:
                return (
                    gr.update(value=f"❌ {message}", visible=True),
                    gr.update(visible=False),  # analyze_button
                    gr.update(visible=True),   # submit button
                )
            # if image is valued, set it in the processor
            self.image_processor.set_image(numpy_image)

            return (
                gr.update(value="✅ Image was valid and got submitted successfully! You can now analyze your colors.", visible=True),
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
        gr.Markdown("# 🎨 Personal Color Analysis System")
        gr.Markdown("Upload your photo to discover your personal color palette!")
        # Image upload section
        img_input = gr.Image(
            label="📸 Upload Your Photo",
            type='filepath',
            # type="numpy",
            height=400,
            # app requirement - user can upload image or take photo with webcam
            sources=['upload', 'webcam']
        )
        
        # Status message
        status_message = gr.Markdown(
            value="Please upload an image in *.jpg* or *.png* format to begin analysis.",
            visible=True
        )

        # Image submission button
        submit_image_button = gr.Button(
            "📤 Submit Image",
            variant="secondary",
            visible=False,
            size="lg"
        )
        
        
        # Analyze button (initially hidden)
        analyze_button = gr.Button(
            "🔍 Analyze My Colors",
            variant="primary",
            visible=False,
            size="lg"
        )
        
        # Progress indicator
        progress_bar = gr.Progress()

        return img_input, status_message, analyze_button, submit_image_button, progress_bar
    
    def run_prediction(self, image: np.ndarray, progress=gr.Progress()):
        """Run prediction pipeline and update UI elements accordingly"""
        try:

            # Mock progress
            progress(0.1, desc="Starting analysis...")

            # Run white-balancing
            progress(0.3, desc="Preprocessing image...")
            preprocessed_image = self.image_processor.preprocess_image(
                self.image_processor.get_image()
            )
            self.image_processor.set_image(preprocessed_image)


            result = self.ai_model.predict(self.image_processor.get_image())[0]

            # Safely get palette
            color_palette = self.ai_model.get_palette_info(result)
            if not color_palette or len(color_palette) < 3:
                color_palette = ["#808080", "#A0A0A0", "#C0C0C0"]  # fallback

            recommendations = self.ai_model.get_description(result)

            # Update state properly
            self.prediction_done.value = True
            self.predicted_type.value = result

            progress(1.0, desc="Analysis complete!")

            return (
                gr.update(
                    value="✅ Image analyzed successfully!",
                    visible=True
                ),                  # status_message (left column)
                color_palette[0],    # primary color
                color_palette[1],    # secondary color
                color_palette[2],    # accent color
                recommendations,     # recommendations
                gr.update(interactive=True),  # re-enable button
                gr.update(
                    value=f"🎨 **Your Personal Color Type: {result}**\n\n{recommendations}",
                    visible=True
                ),  
                gr.update(visible=True)                   # result_message (right column)
            )

        except Exception as e:
            return (
                gr.update(
                    value=f"❌ Analysis failed: {str(e)}",
                    visible=True
                ),
                None, None, None,
                "",
                gr.update(interactive=True),
                gr.update(visible=False),
                gr.update(visible=False)
            )

    
    def build_prediction_result_section(self):
        gr.Markdown("## 🎨 Your Color Analysis Results")

        result_message = gr.Markdown("", visible=False)
        primary_color = gr.ColorPicker(label="Primary Color", interactive=False)
        secondary_color = gr.ColorPicker(label="Secondary Color", interactive=False)
        accent_color = gr.ColorPicker(label="Accent Color", interactive=False)

        recommendations = gr.Markdown("", visible=False)

        return (
            result_message,
            primary_color,
            secondary_color,
            accent_color,
            recommendations,
        )

    def _launch(self):
        self.demo.launch()
    



