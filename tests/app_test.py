import sys
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

import pytest
from unittest.mock import patch, MagicMock
from core.pcoa_app import PCOAApp

class TestPCOAApp:
    @pytest.fixture
    def mock_dependencies(self):
        """
        Creates mock versions of the 3 main subsystems: AI module, image preprocessing module and results visualisation.
        """
        ai_mock = MagicMock()
        proc_mock = MagicMock()
        vis_mock = MagicMock()
        return ai_mock, proc_mock, vis_mock

    @pytest.fixture
    def app_with_mocks(self):
        """
        Creates an app instance where all dependencies 
        (and Gradio itself) are completely mocked.
        """
        ai_mock = MagicMock()
        proc_mock = MagicMock()
        vis_mock = MagicMock()
        
        with patch("core.pcoa_app.gr") as mock_gradio:
            app = PCOAApp(ai_mock, proc_mock, vis_mock)
            app.demo = MagicMock()
            
            return app, ai_mock, proc_mock, vis_mock
        
    
    @patch("core.pcoa_app.gr")
    def test_initialization(self, mock_gradio, mock_dependencies):
        """
        Tests app launch actions:
            - instance initialisation with subsystems objects
            - setting the app work definition state - gdpr acceptance
            - UI build function call
        """
        ai, proc, vis = mock_dependencies

        app = PCOAApp(ai, proc, vis)
        assert app.ai_model == ai
        assert app.image_processor == proc
        assert app.result_visualiser == vis

        assert app.gdpr_accepted is not None
        assert len(mock_gradio.mock_calls) > 0

    # tests for build_gdpr_modal
    @patch("core.pcoa_app.gr")
    def test_build_gdpr_modal(self, mock_gr, app_with_mocks):
        """
        Tests if the GDPR modal is built with the correct structure.
        Checks:
        - if number of returned parameters equals to 4
        - if created visual has proper elements (Group, markdown text, button)
        """
        app, _, _, _= app_with_mocks
        mock_group = MagicMock()
        mock_gr.Group.return_value.__enter__.return_value = mock_group
    
        mock_row = MagicMock()
        mock_gr.Row.return_value.__enter__.return_value = mock_row

        result = app.build_gdpr_modal()
        
        assert len(result) == 4
        mock_gr.Group.assert_called()       
        mock_gr.Markdown.assert_called()    
        mock_gr.Button.assert_called()

    @patch("core.pcoa_app.gr")
    def test_on_accept_gdpr(self, mock_gr, app_with_mocks):
        """
        Tests if accepting GDPR hides the modal, shows the app, and sets the state flag to True.
        Checks:
        - number of returned parameters
        - values of modals visibility variables
        - number of update method calls and their results
        """
        app, _, _, _ = app_with_mocks

        result = app.on_accept_gdpr()

        assert len(result) == 3
        assert result[2] is True

        assert mock_gr.update.call_count == 2
        args1, kwargs1 = mock_gr.update.call_args_list[0]
        assert kwargs1['visible'] is False

        args2, kwargs2 = mock_gr.update.call_args_list[1]
        assert kwargs2['visible'] is True

    @patch("core.pcoa_app.gr")
    def test_on_decline_gdpr(self, mock_gr, app_with_mocks):
        """
        Tests if declining GDPR shows proper message to the user
        Checks:
        - number of update method calls and their results
        - content of message displayed to user
        """
        app, _, _, _= app_with_mocks

        result = app.on_decline_gdpr()
        mock_gr.update.assert_called_once()
        
        args, kwargs = mock_gr.update.call_args
        assert "must accept" in kwargs['value'] 
        assert kwargs['visible'] is True

    #test for build_ui
    @patch("core.pcoa_app.gr")
    @patch("core.pcoa_app.PCOAApp.build_photo_upload_section")
    @patch("core.pcoa_app.PCOAApp.build_gdpr_modal")
    def test_build_ui(self, mock_gdpr_builder, mock_photo_builder, mock_gr, app_with_mocks):
        """
        Tests the structure of whole UI building function
        Checks:
        - if buttons trigger correct functions
        - if proper components such as Blocks are created
        """  
        app, _, _, _, = app_with_mocks

        mock_modal = MagicMock(name="modal")
        mock_accept = MagicMock(name="accept_btn")
        mock_decline = MagicMock(name="decline_btn")
        mock_gdpr_msg = MagicMock(name="gdpr_msg")
        mock_gdpr_builder.return_value = (mock_modal, mock_accept, mock_decline, mock_gdpr_msg)

        mock_img_input = MagicMock(name="img_input")
        mock_status = MagicMock(name="status")
        mock_analyze = MagicMock(name="analyze")
        mock_submit = MagicMock(name="submit")
        mock_progress = MagicMock(name="progress")
        mock_reset = MagicMock(name="reset")
        mock_photo_builder.return_value = (mock_img_input, mock_status, mock_analyze, mock_submit, mock_progress, mock_reset)

        mock_main_app_group = MagicMock(name="main_app_group")
        mock_gr.Group.return_value.__enter__.return_value = mock_main_app_group

        app.build_ui()

        mock_gr.Blocks.assert_called()

        mock_img_input.change.assert_called_once()

        args, kwargs = mock_img_input.change.call_args
        assert kwargs['fn'] == app.on_image_uploaded

        mock_submit.click.assert_called_once()
        args, kwargs = mock_submit.click.call_args
        assert kwargs['fn'] == app.on_image_submitted

        mock_analyze.click.assert_called_once()
        args, kwargs = mock_analyze.click.call_args
        assert kwargs['fn'] == app.on_run_prediction
        
        mock_accept.click.assert_called_once()
        args, kwargs = mock_accept.click.call_args
        assert kwargs['fn'] == app.on_accept_gdpr

        assert mock_main_app_group in kwargs['outputs']

    # test for build_photo_upload_section
    @patch("core.pcoa_app.gr")
    def test_build_photo_upload_section(self, mock_gr, app_with_mocks):
        """
        Tests if the GDPR modal is built with the correct structure.
        Checks:
        - if number of returned parameters equals to 5
        - if created visual has proper elements (Group, markdown text, button)
        """
        app, _, _, _, = app_with_mocks

        result = app.build_photo_upload_section()
        
        assert len(result) == 6 
        mock_gr.Markdown.assert_called()    
        mock_gr.Button.assert_called()
        mock_gr.Image.assert_called()
        mock_gr.Progress.assert_called()

    @patch("core.pcoa_app.gr")
    def test_image_uploaded_none(self, mock_gr, app_with_mocks):
        """
        Tests case when user doesn't upload the image.
        Checks:
        - number of parameters returned
        - components visibility status
        - message to user
        """
        app, _ , _, _= app_with_mocks
        
        result = app.on_image_uploaded(None)
        assert len(result) == 4
        
        # Analyze button - hidden
        args1, kwargs1 = mock_gr.update.call_args_list[0]
        assert kwargs1['visible'] is False
        
        # Submit button - visible
        args2, kwargs2 = mock_gr.update.call_args_list[1]
        assert kwargs2['visible'] is True
        
        # Status message
        args3, kwargs3 = mock_gr.update.call_args_list[2]
        assert "Please upload" in kwargs3['value']


    @patch("core.pcoa_app.gr")
    def test_image_uploaded_success(self, mock_gr, app_with_mocks):
        """
        Tests case when user successfully uploads the image.
        Checks:
        - number of parameters returned
        - message to user
        """
        app, _, _, _ = app_with_mocks
        
        result = app.on_image_uploaded("fake_image_file.jpg")
        
        assert len(result) == 4
        
        # Analyze Button should be hidden
        args1, kwargs1 = mock_gr.update.call_args_list[0]
        assert kwargs1['visible'] is False
        
        # Submit Button should be visible
        args2, kwargs2 = mock_gr.update.call_args_list[1]
        assert kwargs2['visible'] is True
        
        # Status Message check
        args3, kwargs3 = mock_gr.update.call_args_list[2]
        assert "Click Submit" in kwargs3['value']


    @patch("core.pcoa_app.gr")
    def test_image_submitted_none(self, mock_gr, app_with_mocks):
        """
        Test a case where user submits an image without uploading it.
        Checks:
        - number of returned values
        - visuals visibility statuses
        """
        app, _, _, _= app_with_mocks
        
        result = app.on_image_submitted(None)
        assert len(result) == 3
        
        args, kwargs = mock_gr.update.call_args_list[0]
        assert "Upload an image first" in kwargs['value']
        assert kwargs['visible'] is True

    @patch("core.pcoa_app.gr")
    def test_image_submitted_invalid(self, mock_gr, app_with_mocks):
        """
        Test a case where user submits an invalid image.
        Checks:
        - number of returned values
        - status messages
        - if there was no trial of setting a corrupted file in processor
        """
        app, _, proc, _ = app_with_mocks
        
        proc.validate_image.return_value = (False, "Corrupted file", None)
        result = app.on_image_submitted("bad_file.txt")
        
        assert len(result) == 3
    
        args, kwargs = mock_gr.update.call_args_list[0]
        assert "Corrupted file" in kwargs['value']

        proc.set_image.assert_not_called()

    @patch("core.pcoa_app.gr")
    def test_image_submitted_success(self, mock_gr, app_with_mocks):
        """
        Test a case where user submits valid image.
        Checks:
        - number of returned values
        - status messages
        """
        app, _, proc, _ = app_with_mocks
        
        fake_numpy_img = MagicMock()
        proc.validate_image.return_value = (True, "OK", fake_numpy_img)
    
        result = app.on_image_submitted("good_file.jpg")
        
        success_update_call = mock_gr.update.call_args_list[0]
        assert "submitted successfully" in success_update_call.kwargs['value']
    
        proc.set_image.assert_called_once_with(fake_numpy_img)

    # tests for on_run_prediction
    @patch("core.pcoa_app.gr") 
    @patch("core.pcoa_app.time.sleep")
    def test_on_run_prediction_success(self, mock_sleep, mock_gr, app_with_mocks):
        """
        Test the case when prediction was successful.
        Checks:
            - if flow of getting image, image preprocessing and running prediction is preserved
            - messages shown to user
            - components visibilities
        """
        app, _, proc, _ = app_with_mocks 
        
        proc.get_image.return_value = "raw_image_data"
        # features giving spring as result
        proc.preprocess_image.return_value = [67, 129, 129, 209, 135, 141, 82, 140, 142]
        proc.get_processed_image.return_value = [67, 129, 129, 209, 135, 141, 82, 140, 142]

        app.ai_model.predict.return_value = "spring"
        app.result_visualiser.get_palette_info.return_value = ["#639E3F", "#DF5485"]
        app.result_visualiser.get_description.return_value = "Your beauty is bright"
        app.result_visualiser.get_jewelry_recommendation.return_value = "pastel"
        with patch.object(app, '_generate_palette_html', return_value="<div>Palette</div>") as mock_html_gen:
            mock_progress = MagicMock()
            result = app.on_run_prediction(progress=mock_progress)

        proc.get_image.assert_called()
        proc.preprocess_image.assert_called_with("raw_image_data")
        app.ai_model.predict.assert_called()
        
        proc.get_image.assert_called()
        
        assert len(result) == 10
        
        args, kwargs = mock_gr.update.call_args_list[-3]
        
        assert kwargs['visible'] is True
        args_first, kwargs_first = mock_gr.update.call_args_list[0]
        assert "analyzed successfully" in kwargs_first['value']


    @patch("core.pcoa_app.gr")
    @patch("core.pcoa_app.time.sleep")
    def test_on_run_prediction_failure(self, mock_sleep, mock_gr, app_with_mocks):
        """
        Test the case when prediction fails.
        Checks:
        - number of parameters returned
        - if function execution stopped before generating color palette
        """
        app, _, proc, _ = app_with_mocks
        
        proc.get_image.return_value = "data"
        app.ai_model.predict.side_effect = Exception("AI Model Offline")

        mock_progress = MagicMock()
        result = app.on_run_prediction(progress=mock_progress)

        assert len(result) == 10
        
        found_error = False
        for call in mock_gr.update.call_args_list:
            if 'value' in call.kwargs and "AI Model Offline" in call.kwargs['value']:
                found_error = True
                break
        assert found_error, "Did not find the Exception message in UI updates"
        app.result_visualiser.get_palette_info.assert_not_called()

    # tests for _generate_palette_html function
    def test_generate_palette_html_success(self, app_with_mocks):
        """
        Tests generation of color palette tiles in case fetching the colors was successful.
        """
        app, _, _, vis_mock = app_with_mocks
        fake_colors = ["#639E3F", "#DF5485", "#DD9A29", "#215380", "#EBDDCC", "#FDAA63", "#008EAA", "#963CBD"]
        vis_mock.get_palette_info.return_value = fake_colors
        
        html_output = app._generate_palette_html("spring")
        
        assert "#639E3F" in html_output
        assert "#DF5485" in html_output
        assert "display: grid" in html_output
        vis_mock.get_palette_info.assert_called_once_with("spring")

    def test_generate_palette_html_failure(self, app_with_mocks):
        """
        Tests generation of color palette tiles in case fetching the colors failed.
        """
        app, _, _, vis_mock = app_with_mocks
        vis_mock.get_palette_info.side_effect = Exception("Palette details missing")
        html_output = app._generate_palette_html("unknown_season")
        assert "Season not found" in html_output
        assert "color:red" in html_output

    # test for _launch method
    def test_launch(self, app_with_mocks):
        app, _, _, _= app_with_mocks
        app._launch()
        app.demo.launch.assert_called_once()