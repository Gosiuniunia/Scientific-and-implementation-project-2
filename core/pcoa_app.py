import numpy as np
import gradio as gr
import time

# added so Iphone HEIC images don't crash the app
from pillow_heif import register_heif_opener

register_heif_opener()


class PCOAApp:
    """
    Class representing both Gradio UI and backend logic app components.
    Provides methods to handle app usage flow with relevant UI and backend responses.
    """

    def __init__(self, image_processor, result_visualiser, ai_model_orchestrator):
        """
        Initializes the PCOAApp object with an PCOAImageProcessor and ColorAnalysisModel instances.
        Args:
            gdpr_accepted: gradio State: stores information if GDPR disclaimer got accepted to control app components display state
            image_processor: PCOAImageProcessor: object of PCOAImageProcessor class, representing image preprocessing module
            ai_model_orchestrator: AIServiceOrchestrator: object of AIServiceOrchestrator class, representing the link to AI microservice and prediction retrieval module
        """
        self.gdpr_accepted = gr.State(False)
        self.image_processor = image_processor
        self.result_visualiser = result_visualiser
        self.ai_model_orchestrator = ai_model_orchestrator
        self.build_ui()

    def build_ui(self):
        """
        Function building both UI and backend components of the app.
        Returns:
            self.demo: The constructed Gradio demo instance ready to be launched
        """
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
                    # added for the left margin
                    with gr.Column(scale=1, min_width=1):
                        pass
                    # Left column: image upload section
                    with gr.Column(scale=20):
                        (
                            self.img_input,
                            self.status_message,
                            self.analyze_button,
                            self.submit_image_button,
                            self.progress_bar,
                            self.reset_button,  # Added for reset functionality
                        ) = self.build_photo_upload_section()

                    # added to create margin between left and right side
                    with gr.Column(scale=1, min_width=1):
                        pass

                    # Right column: results section (hidden initially)
                    with gr.Column(scale=20):
                        with gr.Group(visible=False) as self.results_section:
                            self.result_message = gr.Markdown("", visible=False)
                            self.description = gr.Markdown("", visible=False)
                            self.jewelerly_recommendation = gr.Markdown(
                                "", visible=False
                            )
                            self.palette_html_output = gr.HTML(
                                label="Interactive Palette"
                            )

                    # added for the right margin
                    with gr.Column(scale=1, min_width=1):
                        pass

                # Show analyze button when image is uploaded
                self.img_input.change(
                    fn=self.on_image_uploaded,
                    inputs=[self.img_input],
                    outputs=[
                        self.analyze_button,
                        self.submit_image_button,
                        self.status_message,
                        self.reset_button,  # Update reset button visibility
                    ],
                )
                # Handle click of submit image button; trigger of validation
                self.submit_image_button.click(
                    fn=self.on_image_submitted,
                    inputs=[self.img_input],
                    outputs=[
                        self.status_message,
                        self.analyze_button,
                        self.submit_image_button,
                    ],
                )

                # Handle click of the analyze image button; trigger of prediction
                self.analyze_button.click(
                    fn=self.on_run_prediction,
                    inputs=[],
                    outputs=[
                        self.status_message,  # left column: status
                        self.description,  # right column: description
                        self.jewelerly_recommendation,  # right column: jewerly recommendation
                        self.palette_html_output,
                        self.analyze_button,  # left column: re-enable button
                        self.result_message,  # right column: prediction text
                        self.results_section,
                        self.reset_button,  # Show reset button after analysis
                    ],
                )

                # Handle click of the reset button
                self.reset_button.click(
                    fn=self.on_reset,
                    inputs=[],
                    outputs=[
                        self.img_input,
                        self.status_message,
                        self.analyze_button,
                        self.submit_image_button,
                        self.results_section,
                        self.reset_button,
                        self.result_message,
                        self.description,
                        self.jewelerly_recommendation,
                        self.palette_html_output,
                    ],
                )

            # GDPR buttons
            self.accept_btn.click(
                fn=self.on_accept_gdpr,
                inputs=[],
                outputs=[self.gdpr_modal, self.main_app, self.gdpr_accepted],
            )

            self.decline_btn.click(
                fn=self.on_decline_gdpr, inputs=[], outputs=[self.gdpr_message]
            )

        return self.demo

    def on_image_uploaded(self, img):
        """
        Function handling visuals related to image upload action.
        Checks if image got uploaded and updates visibility of Analyse button and Submit Image buttons.
        Returns message to user about process status.
        Returns:
        tuple: A 4-element tuple containing Gradio updates:
            'Analyze' button visibility status
            'Submit' button visibility status
            str: Image upload action status message
            'Reset' button visibility status
        """
        if img is None:
            return (
                gr.update(visible=False),  # analyze_button
                gr.update(visible=True),  # submit button
                gr.update(
                    value="### Please upload an image of one person or take a photo",
                    visible=True,
                ),
                gr.update(visible=False),  # reset button
            )

        return (
            gr.update(visible=False),  # analyze_button
            gr.update(visible=True),  # submit button
            gr.update(
                value="### Image uploaded. \n ### Click Submit Image button to validate it",
                visible=True,
            ),
            gr.update(visible=False),  # reset button
        )

    def on_image_submitted(self, img):
        """
        Function handling visuals related to image submission action.
        Checks if image is valid.
        Returns message to user about process status.
        Returns:
        tuple: A 3-element tuple containing Gradio updates:
            str: Image upload action status message
            'Analyze' button visibility status
            'Submit' button visibility status
        """
        if img is None:
            return (
                gr.update(value="### ❌ Upload an image first", visible=True),
                gr.update(visible=False),  # analyze_button
                gr.update(visible=True),  # submit button
            )
        else:
            # validate
            is_valid, message, numpy_image = self.image_processor.validate_image(img)
            if not is_valid:
                return (
                    gr.update(value=f"### ❌ {message}", visible=True),
                    gr.update(visible=False),  # analyze_button
                    gr.update(visible=True),  # submit button
                )
            # if image is valid, set it in the processor
            self.image_processor.set_image(numpy_image)

            return (
                gr.update(
                    value="### ✅ Image was valid and got submitted successfully! \n ### Click on the Analyze button to start analysis.",
                    visible=True,
                ),
                gr.update(visible=True),  # analyze_button
                gr.update(visible=False),  # submit button
            )

    def on_reset(self):
        """
        Function to reset the app to its initial state after GDPR acceptance.
        Clears image input and hides results.
        """
        return (
            None,
            gr.update(
                value="### Please upload an image in .jpg or .png format.", visible=True
            ),  # status_message
            gr.update(visible=False),  # analyze_button
            gr.update(visible=False),  # submit_image_button
            gr.update(visible=False),  # results_section
            gr.update(visible=False),  # reset_button
            gr.update(value="", visible=False),  # result_message
            gr.update(value="", visible=False),  # description
            gr.update(value="", visible=False),  # jewelry_recommendation
            gr.update(value=""),  # palette_html_output
        )

    def build_gdpr_modal(self):
        """
        Function building the visuals for initial GDPR disclaimer window.
        Returns:
            tuple: A 4-element tuple containing the UI components:
                            1. gradio.Group: The main container for the modal (to toggle visibility).
                            2. gradio.Button: The 'Accept' button.
                            3. gradio.Button: The 'Decline' button.
                            4. gradio.Markdown: The component for displaying status messages.
        """
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
                accept_btn = gr.Button(
                    "✅ Accept & Continue", variant="primary", size="lg"
                )

        return gdpr_modal, accept_btn, decline_btn, gdpr_message

    # Accept button logic
    def on_accept_gdpr(self):
        """
        Function handling state of visuals related to GDPR statement acceptance action.
        Returns:
            tuple: A 3-element tuple containing:
                1. dict: Update to hide the GDPR modal.
                2. dict: Update to reveal the main application container.
                3. bool: The new state value (True) for the gdpr_accepted flag.
        """
        return (
            gr.update(visible=False),  # hide GDPR modal
            gr.update(visible=True),  # show main app
            True,  # set GDPR accepted state
        )

    # Decline button logic
    def on_decline_gdpr(self):
        """
        Function handling state of visuals related to GDPR statement acceptance action.
        Returns:
            tuple: A 2-element tuple containing:
                dict: A Gradio update for the status message component that:
                1. Sets the error text indicating acceptance is required.
                2. Makes the message visible.
        """
        return gr.update(
            value="❌ You must accept the privacy policy to use this app.", visible=True
        )

    def build_photo_upload_section(self):
        """
        Function building visuals placed in photo upload section of ther app (left pane).
        Sets proper visuals visibility statuses and descriptions.

        Returns:
            tuple: A 6-element tuple containing the UI components:
                            1. gradio.Image: The input component for file upload and webcam.
                            2. gradio.Markdown: The status message component.
                            3. gradio.Button: The 'Analyze' button (initially hidden).
                            4. gradio.Button: The 'Submit Image' button (initially hidden).
                            5. gradio.Progress: The progress bar instance for tracking analysis.
                            6. gradio.Button: The 'Reset' button (initially hidden).
        """
        # Status message
        status_message = gr.Markdown(
            value="### Please upload an image in .jpg or .png format.", visible=True
        )
        # Image upload section
        img_input = gr.Image(
            label="Input photo",
            # needed for file validation step
            type="filepath",
            height=400,
            # app requirement - user can upload image or take photo with webcam
            sources=["upload", "webcam"],
        )

        # Image submission button
        submit_image_button = gr.Button(
            "Submit Image", variant="secondary", visible=False, size="lg"
        )

        # Analyze button (initially hidden)
        analyze_button = gr.Button(
            "🔍 Analyze", variant="primary", visible=False, size="lg"
        )

        # Reset button (initially hidden)
        reset_button = gr.Button(
            "🔄 Reset & New Analysis", variant="primary", visible=False, size="lg"
        )

        # Progress indicator
        progress_bar = gr.Progress()

        return (
            img_input,
            status_message,
            analyze_button,
            submit_image_button,
            progress_bar,
            reset_button,
        )

    def on_run_prediction(self, progress=gr.Progress()):
        """
        Runs prediction pipeline and updates UI elements accordingly
        Args:
            progress (gr.Progress): Gradio progress bar for process updates.
        Returns:
            tuple: A 8-element tuple containing updates for the results interface:
                            1. dict: Update with the process status message (Success).
                            2. dict: Update with the season description text.
                            3. dict: Update with the jewelry and palette description text.
                            4. str:  The raw HTML string with the color palette visualization.
                            5. dict: Update for Analyze button visibility.
                            6. dict: Update for the seasonal color type header/result.
                            7. dict: Update to reveal the results container.
                            8. dict: Update to reveal the Reset button.
        """

        def handle_error(stage, error):
            """
            Helper function for troubleshooting purpose to see at which action app fails.
            """
            print(f"!!! ERROR at [{stage}]: {error}")
            return (
                gr.update(
                    value=f"❌ Failed at {stage}: {str(error)}", visible=True
                ),  # Status
                "",
                "",  # Description, Jewelry
                "",  # HTML Palette
                gr.update(interactive=True),  # Button
                gr.update(visible=False),  # Result Message
                gr.update(visible=False),  # Result Container
                gr.update(visible=False),  # Reset Button
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

            # linking to ai microservice
            prediction_results = (
                self.ai_model_orchestrator.get_prediction_from_ai_service(current_img)
            )

            if not prediction_results:
                raise ValueError(
                    "Couldn't give result for given image. Please upload different one."
                )

            print(f"--- Prediction: SUCCESS (Result: {prediction_results}) ---")
        except Exception as e:
            return handle_error("AI Prediction", e)

        # Getting color pallete and descriptions based on predicted color type
        try:
            color_palette = self.result_visualiser.get_palette_info(prediction_results)
            if not color_palette or len(color_palette) < 3:
                color_palette = [
                    "#808080",
                    "#A0A0A0",
                    "#C0C0C0",
                    "#D0D0D0",
                    "#E0E0E0",
                    "#F0F0F0",
                    "#B0B0B0",
                    "#909090",
                ]

            description = self.result_visualiser.get_description(prediction_results)
            jewelry = self.result_visualiser.get_jewelry_recommendation(
                prediction_results
            )
            full_palette_html = self._generate_palette_html(prediction_results)
            print(f"--- Recommendations Retrieval: SUCCESS ---")
        except Exception as e:
            return handle_error("Data Retrieval", e)

        # build prediction results string with emoji for each season
        season_emojis = {"spring": "🌸", "summer": "☀️", "autumn": "🍂", "winter": "❄️"}
        emoji = season_emojis.get(prediction_results.lower(), "")
        prediction_results_string = f"{prediction_results.capitalize()} {emoji}"

        # Showing recommendations in the UI
        try:
            progress(1.0, desc="Analysis complete!")
            time.sleep(0.5)
            if prediction_results.lower() == "none":
                return (
                    gr.update(
                        value="### ❓ Unsure prediction", visible=True
                    ),  # markdown 1
                    gr.update(
                        value=f"### 📝 Description\n{description} <br><br>\n",
                        visible=True,
                    ),  # markdown 2
                    gr.update(visible=True),
                    gr.update(value="", visible=False),  # html
                    gr.update(visible=False),  # Hide analyze button
                    gr.update(value="", visible=False),  # markdown 6
                    gr.update(visible=True),  # group
                    gr.update(visible=True),  # Show reset button
                )
            else:
                return (
                    gr.update(
                        value="### ✅ Image analyzed successfully!", visible=True
                    ),
                    gr.update(
                        value=f"### 📝 Description\n{description} <br><br>\n",
                        visible=True,
                    ),
                    gr.update(
                        value=f"### 💍 Jewelry recommendations\n{jewelry} <br>\n ### 🎨 Recommended color palette with color codes: <br><br>\n",
                        visible=True,
                    ),
                    full_palette_html,
                    gr.update(visible=False),  # Hide analyze button
                    gr.update(
                        value=f"### 🎨 Your seasonal color type:<br>\n <h2 style='text-align: center;'> {prediction_results_string} </h2><br>",
                        visible=True,
                    ),
                    gr.update(visible=True),
                    gr.update(visible=True),  # Show reset button
                )
        except Exception as e:
            return handle_error("UI Update", e)

    def _generate_palette_html(self, season):
        """
        Generates color pallete tiles corresponding to passed season value.
        Args:
            season: str:  representing name of seasonal color type
        Returns:
            str: HTML and CSS style enabling color palette tiles display
        """
        html_colors = ""
        try:
            # colors = self.ai_model.get_palette_info(season)
            colors = self.result_visualiser.get_palette_info(season)
        except Exception as e:
            print(f"Error retrieving color palette: {e}")
            return "<div style='color:red'>Season not found</div>"
        for c in colors:
            html_colors += f"""
                    <div style="display:flex; flex-direction:column; align-items:center; justify-content:flex-start;">
                        <div style="
                            background-color: {c}; 
                            width: 80px;
                            height: 80px;
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
        """
        Function launches the underlying Gradio web server.
        """
        self.demo.launch()
