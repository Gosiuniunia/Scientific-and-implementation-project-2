import numpy as np
import gradio as gr
from utils.enums import *
from core.pcoa_image_preprocessing import PCOAImageProcessor


class PCOAApp:
    def __init__(self, ai_model, image_processor):
        self.gdpr_accepted = gr.State(False)
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

                # 1️⃣ Build result section FIRST
                (
                    self.result_message,
                    self.primary_color,
                    self.secondary_color,
                    self.accent_color,
                    self.recommendations,
                ) = self.build_prediction_result_section()

                # 2️⃣ Build image upload section
                (
                    self.img_input,
                    self.status_message,
                    self.img_preview,
                    self.analyze_button,
                    self.progress_bar,
                ) = self.build_photo_upload_section()

                # 3️⃣ NOW wire button (components exist!)
                # ✅ SHOW analyze button when image is uploaded
                self.img_input.change(
                    fn=self.on_image_uploaded,
                    inputs=[self.img_input],
                    outputs=[
                        self.analyze_button,
                        self.status_message,
                        self.img_preview,
                    ],
                )
                self.analyze_button.click(
                    fn=self.run_prediction,
                    inputs=[self.img_input],
                    outputs=[
                        self.status_message,
                        self.primary_color,
                        self.secondary_color,
                        self.accent_color,
                        self.recommendations,
                        self.analyze_button,
                    ],
                )
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
                gr.update(value="Please upload an image to begin analysis", visible=True),
                gr.update(visible=False),  # img_preview
            )

        return (
            gr.update(visible=True),   # analyze_button
            gr.update(value="Image uploaded. Ready to analyze!", visible=True),
            gr.update(value=img, visible=True),  # img_preview
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
            type="numpy",
            height=400
        )
        
        # Status message
        status_message = gr.Markdown(
            value="Please upload an image to begin analysis",
            visible=True
        )
        
        # Image preview
        img_preview = gr.Image(
            label="✨ Processed Image",
            interactive=False,
            visible=False,
            height=300
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

        return img_input, status_message, img_preview, analyze_button, progress_bar
    
    def run_prediction(self, image: np.ndarray, progress=gr.Progress()):
        """Full prediction logic with state management and progress tracking."""
        print("Running prediction...")
        if not self.image_processor:
            print("No valid image to process.")
            return (
                "❌ Please upload and validate an image first.",
                gr.update(visible=False),  # results section
                None, None, None,  # color pickers
                "",  # recommendations
                gr.update(interactive=True)  # re-enable button
            )
        
        try:
            print("Starting analysis...")
            # Disable button during processing
            progress(0.1, desc="Starting analysis...")
            
            # Preprocess image if needed
            progress(0.3, desc="Processing image...")
            self.image_processor.preprocess_image(image)
            
            # Run color analysis
            progress(0.6, desc="Analyzing colors...")
            result = self.ai_model.predict(self.image_processor.get_image())

            # Update state
            progress(0.8, desc="Generating recommendations...")
            self.prediction_done = PredictionStatus.DONE
            self.predicted_type = result
            
            # Get color palette and recommendations
            color_palette = self.get_season_colors(result)
            recommendations = self.get_style_recommendations(result)
            
            progress(1.0, desc="Analysis complete!")
            
            return (
                f"🎨 **Your Personal Color Type: {result}**\n\n{self.get_season_description(result)}",
                gr.update(visible=True),  # show results section
                color_palette[0],  # primary color
                color_palette[1],  # secondary color
                color_palette[2],  # accent color
                recommendations,   # style recommendations
                gr.update(interactive=True)  # re-enable button
            )
            
        except Exception as e:
            print(f"Error during prediction: {str(e)}")
            return (
                f"❌ Analysis failed: {str(e)}",
                gr.update(visible=False),
                None, None, None, "",
                gr.update(interactive=True)  # re-enable button
            )
    
    def get_season_colors(self, season: ColorType) -> list:
        """Get representative colors for each season"""
        season_palettes = {
            ColorType.SPRING: ["#FF6B6B", "#4ECDC4", "#45B7D1"],  # Warm, bright colors
            ColorType.SUMMER: ["#96CEB4", "#FFEAA7", "#DDA0DD"],  # Cool, soft colors  
            ColorType.AUTUMN: ["#D63031", "#E17055", "#FDCB6E"],  # Warm, muted colors
            ColorType.WINTER: ["#2D3436", "#0984E3", "#E84393"]   # Cool, clear colors
        }
        return season_palettes.get(season, ["#808080", "#A0A0A0", "#C0C0C0"])
    
    def get_season_description(self, season: ColorType) -> str:
        """Get detailed description for each season"""
        descriptions = {
            ColorType.SPRING: "You have warm undertones with bright, clear coloring. Spring types look best in warm, vibrant colors that complement their natural radiance.",
            ColorType.SUMMER: "You have cool undertones with soft, muted coloring. Summer types shine in cool, gentle colors that enhance their natural elegance.",
            ColorType.AUTUMN: "You have warm undertones with rich, deep coloring. Autumn types look stunning in warm, earthy colors that match their natural depth.",
            ColorType.WINTER: "You have cool undertones with high contrast coloring. Winter types excel in cool, bold colors that complement their striking features."
        }
        return descriptions.get(season, "Your unique coloring has been analyzed.")
    
    def get_style_recommendations(self, season: ColorType) -> str:
        """Get style recommendations for each season"""
        recommendations = {
            ColorType.SPRING: "**Best Colors:** Coral, peach, golden yellow, bright green, clear blue\n**Avoid:** Black, pure white, dark colors\n**Metals:** Gold jewelry works best",
            ColorType.SUMMER: "**Best Colors:** Soft pink, lavender, powder blue, sage green, soft gray\n**Avoid:** Orange, bright yellow, warm colors\n**Metals:** Silver jewelry is ideal",
            ColorType.AUTUMN: "**Best Colors:** Rust, olive green, golden brown, deep orange, warm red\n**Avoid:** Pink, icy colors, cool tones\n**Metals:** Gold and copper jewelry",
            ColorType.WINTER: "**Best Colors:** True red, royal blue, emerald green, black, pure white\n**Avoid:** Orange, golden yellow, warm colors\n**Metals:** Silver and platinum jewelry"
        }
        return recommendations.get(season, "Consult with a color analyst for personalized recommendations.")
    
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
    



