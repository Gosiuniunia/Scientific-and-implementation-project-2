import numpy as np
import gradio as gr
from utils.enums import *
from core.pcoa_image_preprocessing import PCOAImageProcessor


class PCOAApp:
    def __init__(self, ai_model=None, image_processor=None):
        self.photo_uploaded = PhotoUploadStatus.NOT_UPLOADED
        self.photo_validated = PhotoValidationStatus.NOT_VALIDATED
        self.photo_preprocessed = PhotoPreprocessingStatus.NOT_VALIDATED
        self.prediction_done = PredictionStatus.NOT_DONE
        self.recommendation_generated = RecommendationStatus.NOT_GENERATED
        self.terms_accepted = TermsAccepted.NOT_ACCEPTED

        self.predicted_type = None
        self.current_image = None
        self.ai_model = ai_model
        self.image_processor = image_processor
    
    def show_uploaded_image(self, image: np.ndarray):
        if image is None:
            return None
        return image
    
    def get_ui_state(self) -> dict:
        """Get current UI state for conditional rendering"""
        return {
            'show_analyze_button': self.photo_validated == PhotoValidationStatus.VALIDATED,
            'show_results': self.prediction_done == PredictionStatus.DONE,
            'show_upload_status': self.photo_uploaded == PhotoUploadStatus.UPLOADED,
            'enable_interactions': self.photo_uploaded == PhotoUploadStatus.UPLOADED
        }

    
    def show_uploaded_image(self, image: np.ndarray):
        """
        Process and validate uploaded image with state updates.
        """
        if image is None:
            self.photo_uploaded = PhotoUploadStatus.NOT_UPLOADED
            self.photo_validated = PhotoValidationStatus.NOT_VALIDATED
            self.prediction_done = PredictionStatus.NOT_DONE
            self.processor = None
            
            # Return UI updates: image, status, button visibility, results visibility
            return None, "No image uploaded", gr.update(visible=False), gr.update(visible=False)
        
        # Create processor and validate
        self.processor = PCOAImageProcessor(image)
        is_valid, message, processed_image = self.processor.validate_image(image)
        
        if is_valid:
            self.photo_uploaded = PhotoUploadStatus.UPLOADED
            self.photo_validated = PhotoValidationStatus.VALIDATED
            self.current_image = processed_image
            
            # Return: processed image, status message, show analyze button, hide results
            return (
                processed_image, 
                f"✅ {message}", 
                gr.update(visible=True, interactive=True),  # analyze button
                gr.update(visible=False)  # results section
            )
        else:
            self.photo_uploaded = PhotoUploadStatus.NOT_UPLOADED
            self.photo_validated = PhotoValidationStatus.NOT_VALIDATED
            
            # Return: no image, error message, hide analyze button, hide results
            return (
                None, 
                f"❌ {message}", 
                gr.update(visible=False), 
                gr.update(visible=False)
            )
    
    def run_prediction(self, image: np.ndarray, progress=gr.Progress()):
        """Full prediction logic with state management and progress tracking."""
        print("Running prediction...")
        if not self.processor or not self.photo_validated:
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
            self.processor.preprocess_image(image)
            
            # Run color analysis
            progress(0.6, desc="Analyzing colors...")
            result = self.ai_model.predict(self.processor.get_image())
            
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


    def build_ui(self):
        """Build and return the Gradio interface"""
        with gr.Blocks(theme=gr.themes.Soft()) as demo:
            
            state = gr.State(self)

            # GDPR Disclaimer Modal (simpler approach)
            with gr.Group(visible=True) as gdpr_modal:
                gr.Markdown("## 🔒 Privacy & Data Protection Notice")
                gr.Markdown("""
                **Personal Color Analysis System - GDPR Compliance**
                
                By using this application, you acknowledge and agree to the following:
                
                **Data Processing:**
                - Your uploaded photos are processed locally for color analysis purposes only
                - Images are temporarily stored in memory during analysis and are not saved permanently
                - Your images and analysis results are not shared with third parties
                - All data is cleared when you close the application
                - You consent to the processing of your image data for color analysis
                
                **Your Rights:**
                - You can stop using the service at any time
                - You can request deletion of your data (contact us if needed)
                - You have the right to data portability
                
                **Contact:** For privacy concerns, contact [your-email@domain.com]
                """)
                
                gdpr_message = gr.Markdown("", visible=False)
                
                with gr.Row():
                    decline_btn = gr.Button("❌ Decline", variant="secondary", size="lg")
                    accept_btn = gr.Button("✅ Accept & Continue", variant="primary", size="lg")

                # if decline button clicked, close app
                decline_btn.click(
                    fn=lambda: exit(),
                    inputs=[],
                    outputs=[])


            gr.Markdown("# 🎨 Personal Color Analysis System")
            gr.Markdown("Upload your photo to discover your personal color palette!")

            with gr.Row():
                with gr.Column(scale=1):
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

                with gr.Column(scale=1):
                    # Results section (initially hidden)
                    with gr.Group(visible=False) as results_section:
                        gr.Markdown("## 🎨 Your Color Analysis Results")
                        
                        result_text = gr.Markdown()
                        
                        gr.Markdown("### Your Recommended Colors:")
                        with gr.Row():
                            color1 = gr.ColorPicker(label="Primary Color", interactive=False)
                            color2 = gr.ColorPicker(label="Secondary Color", interactive=False)
                            color3 = gr.ColorPicker(label="Accent Color", interactive=False)
                        
                        # Additional recommendations
                        gr.Markdown("### Style Recommendations")
                        recommendations = gr.Markdown()
            
            # Event handlers with state management
            img_input.change(
                fn=lambda img, app: app.show_uploaded_image(img),
                inputs=[img_input, state],
                outputs=[img_preview, status_message, analyze_button, results_section]
            )

            analyze_button.click(
                fn=lambda img, app: app.run_prediction(img),
                inputs=[img_input, state],
                outputs=[result_text, results_section, color1, color2, color3, recommendations, analyze_button]
            )

        return demo
    



