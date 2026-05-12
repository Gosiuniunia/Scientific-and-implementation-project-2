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
            app = PCOAApp(proc_mock, vis_mock, ai_mock)
            app.demo = MagicMock()

            return app, proc_mock, vis_mock, ai_mock

    @patch("core.pcoa_app.gr")
    def test_initialization(self, mock_gradio, mock_dependencies):
        """
        Tests app launch actions:
            - instance initialisation with subsystems objects
            - setting the app work definition state - gdpr acceptance
            - UI build function call
        """
        ai_model_orchestrator, proc, vis = mock_dependencies

        app = PCOAApp(
            image_processor=proc,
            result_visualiser=vis,
            ai_model_orchestrator=ai_model_orchestrator,
        )
        assert app.ai_model_orchestrator == ai_model_orchestrator
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
        app, _, _, _ = app_with_mocks
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
        assert kwargs1["visible"] is False

        args2, kwargs2 = mock_gr.update.call_args_list[1]
        assert kwargs2["visible"] is True

    @patch("core.pcoa_app.gr")
    def test_on_decline_gdpr(self, mock_gr, app_with_mocks):
        """
        Tests if declining GDPR shows proper message to the user
        Checks:
        - number of update method calls and their results
        - content of message displayed to user
        """
        app, _, _, _ = app_with_mocks

        result = app.on_decline_gdpr()
        mock_gr.update.assert_called_once()

        args, kwargs = mock_gr.update.call_args
        assert "must accept" in kwargs["value"]
        assert kwargs["visible"] is True

    # test for build_ui
    @patch("core.pcoa_app.gr")
    @patch("core.pcoa_app.PCOAApp.build_photo_upload_section")
    @patch("core.pcoa_app.PCOAApp.build_gdpr_modal")
    def test_build_ui(
        self, mock_gdpr_builder, mock_photo_builder, mock_gr, app_with_mocks
    ):
        """
        Tests the structure of whole UI building function
        Checks:
        - if buttons trigger correct functions
        - if proper components such as Blocks are created
        """
        (
            app,
            _,
            _,
            _,
        ) = app_with_mocks

        mock_modal = MagicMock(name="modal")
        mock_accept = MagicMock(name="accept_btn")
        mock_decline = MagicMock(name="decline_btn")
        mock_gdpr_msg = MagicMock(name="gdpr_msg")
        mock_gdpr_builder.return_value = (
            mock_modal,
            mock_accept,
            mock_decline,
            mock_gdpr_msg,
        )

        mock_img_input = MagicMock(name="img_input")
        mock_status = MagicMock(name="status")
        mock_analyze = MagicMock(name="analyze")
        mock_submit = MagicMock(name="submit")
        mock_progress = MagicMock(name="progress")
        mock_reset = MagicMock(name="reset")
        mock_photo_builder.return_value = (
            mock_img_input,
            mock_status,
            mock_analyze,
            mock_submit,
            mock_progress,
            mock_reset,
        )

        mock_main_app_group = MagicMock(name="main_app_group")
        mock_gr.Group.return_value.__enter__.return_value = mock_main_app_group

        app.build_ui()

        mock_gr.Blocks.assert_called()

        mock_img_input.change.assert_called_once()

        args, kwargs = mock_img_input.change.call_args
        assert kwargs["fn"] == app.on_image_uploaded

        mock_submit.click.assert_called_once()
        args, kwargs = mock_submit.click.call_args
        assert kwargs["fn"] == app.on_image_submitted

        mock_analyze.click.assert_called_once()
        args, kwargs = mock_analyze.click.call_args
        assert kwargs["fn"] == app.on_run_prediction

        mock_accept.click.assert_called_once()
        args, kwargs = mock_accept.click.call_args
        assert kwargs["fn"] == app.on_accept_gdpr

        assert mock_main_app_group in kwargs["outputs"]

    # test for build_photo_upload_section
    @patch("core.pcoa_app.gr")
    def test_build_photo_upload_section(self, mock_gr, app_with_mocks):
        """
        Tests if the GDPR modal is built with the correct structure.
        Checks:
        - if number of returned parameters equals to 5
        - if created visual has proper elements (Group, markdown text, button)
        """
        (
            app,
            _,
            _,
            _,
        ) = app_with_mocks

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
        app, _, _, _ = app_with_mocks

        result = app.on_image_uploaded(None)
        assert len(result) == 4

        # Analyze button - hidden
        args1, kwargs1 = mock_gr.update.call_args_list[0]
        assert kwargs1["visible"] is False

        # Submit button - visible
        args2, kwargs2 = mock_gr.update.call_args_list[1]
        assert kwargs2["visible"] is True

        # Status message
        args3, kwargs3 = mock_gr.update.call_args_list[2]
        assert "Please upload" in kwargs3["value"]

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
        assert kwargs1["visible"] is False

        # Submit Button should be visible
        args2, kwargs2 = mock_gr.update.call_args_list[1]
        assert kwargs2["visible"] is True

        # Status Message check
        args3, kwargs3 = mock_gr.update.call_args_list[2]
        assert "Click Submit" in kwargs3["value"]

    @patch("core.pcoa_app.gr")
    def test_image_submitted_none(self, mock_gr, app_with_mocks):
        """
        Test a case where user submits an image without uploading it.
        Checks:
        - number of returned values
        - visuals visibility statuses
        """
        app, _, _, _ = app_with_mocks

        result = app.on_image_submitted(None)
        assert len(result) == 3

        args, kwargs = mock_gr.update.call_args_list[0]
        assert "Upload an image first" in kwargs["value"]
        assert kwargs["visible"] is True

    @patch("core.pcoa_app.gr")
    def test_image_submitted_invalid(self, mock_gr, app_with_mocks):
        """
        Test a case where user submits an invalid image.
        Checks:
        - number of returned values
        - status messages
        - if there was no trial of setting a corrupted file in processor
        """
        app, proc, _, _ = app_with_mocks

        proc.validate_image.return_value = (False, "Corrupted file", None)
        result = app.on_image_submitted("bad_file.txt")

        assert len(result) == 3

        args, kwargs = mock_gr.update.call_args_list[0]
        assert "Corrupted file" in kwargs["value"]

        proc.set_image.assert_not_called()

    @patch("core.pcoa_app.gr")
    def test_image_submitted_success(self, mock_gr, app_with_mocks):
        """
        Test a case where user submits valid image.
        Checks:
        - number of returned values
        - status messages
        """
        app, proc, _, _ = app_with_mocks

        fake_numpy_img = MagicMock()
        proc.validate_image.return_value = (True, "OK", fake_numpy_img)

        result = app.on_image_submitted("good_file.jpg")

        success_update_call = mock_gr.update.call_args_list[0]
        assert "submitted successfully" in success_update_call.kwargs["value"]

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
        app, proc, _, _ = app_with_mocks

        proc.get_image.return_value = "raw_image_data"
        # features giving spring as result
        proc.preprocess_image.return_value = [67, 129, 129, 209, 135, 141, 82, 140, 142]
        proc.get_processed_image.return_value = [
            67,
            129,
            129,
            209,
            135,
            141,
            82,
            140,
            142,
        ]

        app.ai_model_orchestrator.get_prediction_from_ai_service.return_value = "spring"
        app.result_visualiser.get_palette_info.return_value = ["#639E3F", "#DF5485"]
        app.result_visualiser.get_description.return_value = "Your beauty is bright"
        app.result_visualiser.get_jewelry_recommendation.return_value = "pastel"
        with patch.object(
            app, "_generate_palette_html", return_value="<div>Palette</div>"
        ) as mock_html_gen:
            mock_progress = MagicMock()
            result = app.on_run_prediction(progress=mock_progress)

        proc.get_image.assert_called()
        proc.preprocess_image.assert_called_with("raw_image_data")
        app.ai_model_orchestrator.get_prediction_from_ai_service.assert_called()

        proc.get_image.assert_called()
        
        assert len(result) == 10
        
        args, kwargs = mock_gr.update.call_args_list[-3]
        
        assert kwargs['visible'] is True
        args_first, kwargs_first = mock_gr.update.call_args_list[0]
        assert "analyzed successfully" in kwargs_first['value']

        assert kwargs["visible"] is True
        args_first, kwargs_first = mock_gr.update.call_args_list[0]
        assert "analyzed successfully" in kwargs_first["value"]

    @patch("core.pcoa_app.gr")
    @patch("core.pcoa_app.time.sleep")
    def test_on_run_prediction_failure(self, mock_sleep, mock_gr, app_with_mocks):
        """
        Test the case when prediction fails.
        Checks:
        - number of parameters returned
        - if function execution stopped before generating color palette
        """
        app, proc, _, _ = app_with_mocks

        proc.get_image.return_value = "data"
        app.ai_model_orchestrator.get_prediction_from_ai_service.side_effect = (
            Exception("AI Model Offline")
        )

        mock_progress = MagicMock()
        result = app.on_run_prediction(progress=mock_progress)

        assert len(result) == 10
        
        found_error = False
        for call in mock_gr.update.call_args_list:
            if "value" in call.kwargs and "AI Model Offline" in call.kwargs["value"]:
                found_error = True
                break
        assert found_error, "Did not find the Exception message in UI updates"
        app.result_visualiser.get_palette_info.assert_not_called()

    # tests for _generate_palette_html function
    def test_generate_palette_html_success(self, app_with_mocks):
        """
        Tests generation of color palette tiles in case fetching the colors was successful.
        """
        app, _, vis_mock, _ = app_with_mocks
        fake_colors = [
            "#639E3F",
            "#DF5485",
            "#DD9A29",
            "#215380",
            "#EBDDCC",
            "#FDAA63",
            "#008EAA",
            "#963CBD",
        ]
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
        app, _, vis_mock, _ = app_with_mocks
        vis_mock.get_palette_info.side_effect = Exception("Palette details missing")
        html_output = app._generate_palette_html("unknown_season")
        assert "Season not found" in html_output
        assert "color:red" in html_output

    # test for _launch method
    def test_launch(self, app_with_mocks):
        app, _, _, _ = app_with_mocks
        app._launch()
        app.demo.launch.assert_called_once()

    @patch("core.pcoa_app.register_user")
    @patch("core.pcoa_app.gr")
    def test_do_register_success(self, mock_gr, mock_register, app_with_mocks):
        """Test successful user registration."""
        app, _, _, _ = app_with_mocks
        mock_register.return_value = (True, "Account created")
        
        result = app.do_register("new_user", "password123")
        
        mock_register.assert_called_once_with("new_user", "password123")
        mock_gr.update.assert_called_with(value="Account created")

    def test_show_login_visibility(self, app_with_mocks):
        """Test if show_login correctly sets visibility flags for Gradio components."""
        app, _, _, _ = app_with_mocks
        
        result = app.show_login()
        assert len(result) == 2
        assert result[0]['visible'] is True
        assert result[1]['visible'] is False

    @patch("core.pcoa_app.login_user")
    @patch("core.pcoa_app.get_user_analyses")
    @patch("core.pcoa_app.gr")
    def test_do_login_success(self, mock_gr, mock_history, mock_login, app_with_mocks):
        """Test successful login and user history loading."""
        app, _, _, _ = app_with_mocks
        mock_login.return_value = (True, "")
        mock_history.return_value = [("1", "Test", "spring", "2024-01-01")]
        
        # Trigger login
        result = app.do_login("user1", "pass1")
        
        assert app.current_user == "user1"
        assert len(app.history) == 1
        # Verify UI updates for user panel visibility and greeting
        mock_gr.update.assert_any_call(value="## 👋 user1")
        mock_gr.update.assert_any_call(visible=True) # for user_panel and logout_btn

    @patch("core.pcoa_app.gr")
    def test_logout(self, mock_gr, app_with_mocks):
        """Test logout process and application state reset."""
        app, _, _, _ = app_with_mocks
        app.current_user = "active_user"
        
        app.logout()
        
        assert app.current_user is None
        assert app.history == []
        # Verify login button visibility reset
        mock_gr.update.assert_any_call(visible=True) # login_btn

    @patch("core.pcoa_app.time.sleep")
    def test_prediction_result_formatting(self, mock_sleep, app_with_mocks):
        """Verify that analysis results contain the correct emoji and season name."""
        app, _, proc, vis = app_with_mocks
        proc.get_image.return_value = "img"
        app.ai_model_orchestrator.get_prediction_from_ai_service.return_value = "autumn"
        vis.get_description.return_value = "Desc"
        vis.get_jewelry_recommendation.return_value = "Gold"
        
        with patch("core.pcoa_app.gr") as mock_gr:
            mock_gr.update.side_effect = lambda **kwargs: MagicMock(kwargs=kwargs)
            
            result = app.on_run_prediction(progress=MagicMock())
            
            prediction_html = result[5].kwargs['value']
            assert "Autumn" in prediction_html
            assert "🍂" in prediction_html

    @patch("core.pcoa_app.save_color_analysis")
    @patch("core.pcoa_app.get_user_analyses")
    def test_save_result_logic(self, mock_get, mock_save, app_with_mocks):
        """Test database save call logic."""
        app, _, _, _ = app_with_mocks
        app.current_user = "user1"
        app.last_prediction = "spring"
        
        app.save_result("My analysis")
        
        mock_save.assert_called_once_with("user1", "spring", "My analysis")
        mock_get.assert_called()

    @patch("core.pcoa_app.gr")
    @patch("core.pcoa_app.time.sleep")
    def test_save_panel_visibility_unauthorized(self, mock_sleep, mock_gr, app_with_mocks):
        """Verify that the save panel is HIDDEN for unauthorized users."""
        app, _, proc, _ = app_with_mocks
        app.current_user = None 
        
        mock_gr.update.side_effect = lambda **kwargs: kwargs
        
        proc.get_image.return_value = "img"
        app.ai_model_orchestrator.get_prediction_from_ai_service.return_value = "winter"
        
        result = app.on_run_prediction(progress=MagicMock())

        assert result[8]['visible'] is False
        assert result[9]['visible'] is False

    @patch("core.pcoa_app.gr")
    @patch("core.pcoa_app.time.sleep")
    def test_save_panel_visibility_authorized(self, mock_sleep, mock_gr, app_with_mocks):
        """Verify that the save panel is VISIBLE for authorized users."""
        app, _, proc, _ = app_with_mocks
        app.current_user = "logged_in_user"
        
        mock_gr.update.side_effect = lambda **kwargs: kwargs
        
        proc.get_image.return_value = "img"
        app.ai_model_orchestrator.get_prediction_from_ai_service.return_value = "winter"
        
        result = app.on_run_prediction(progress=MagicMock())
        
        assert result[8]['visible'] is True
        assert result[9]['visible'] is True

    def test_save_result_no_user_or_prediction(self, app_with_mocks):
        """Test save_result when no user is logged in or no prediction exists."""
        app, _, _, _ = app_with_mocks
        
        # Scenario 1: No user logged in
        app.current_user = None
        hist, update = app.save_result("Test")
        assert update['value'] == ""
        
        # Scenario 2: last_prediction is "none"
        app.current_user = "user1"
        app.last_prediction = "none"
        hist, update = app.save_result("Test")
        assert "No saved analyses" in hist

    @patch("core.pcoa_app.register_user")
    def test_on_register_empty_fields(self, mock_reg, app_with_mocks):
        """Test registration attempts with empty fields."""
        app, _, _, _ = app_with_mocks
        # Simulate empty inputs
        result = app.do_register("", "")
        
        # Verify error message return
        assert result['value'] == "❌ Fill all fields"
        mock_reg.assert_not_called()

    @patch("core.pcoa_app.login_user")
    def test_do_login_failure_scenarios(self, mock_login, app_with_mocks):
        """Test various login failure scenarios."""
        app, _, _, _ = app_with_mocks
        
        # Test empty fields (status at index 0)
        result = app.do_login("", "")
        assert result[0]['value'] == "❌ Username and password required"
        
        # Test invalid credentials
        mock_login.return_value = (False, "Invalid credentials")
        result = app.do_login("user", "wrong")
        
        assert result[0]['value'] == "Invalid credentials"
        assert result[1]['visible'] is True

    def test_format_history_empty(self, app_with_mocks):
        """Test history formatting when history is empty."""
        app, _, _, _ = app_with_mocks
        app.history = []
        output = app.format_history([])
        assert "### 📝 No saved analyses\nPerform your first one and save the result." in output

    def test_on_reset_state(self, app_with_mocks):
        """Test the application state reset functionality."""
        app, _, _, _ = app_with_mocks
        
        results = app.on_reset()
        assert len(results) == 12
        
        assert results[0] is None  # Image input cleared
        assert "Please upload an image" in results[1]['value']  # Status message
        assert results[2]['visible'] is False  # analyze_button
        assert results[3]['visible'] is False  # submit_image_button
        assert results[4]['visible'] is False  # results_section
        assert results[5]['visible'] is False  # reset_button
        
        assert results[6]['value'] == ""
        assert results[9]['value'] == ""
        
        assert results[10]['visible'] is False
        assert results[11]['visible'] is False

    def test_on_run_prediction_image_loading_failure(self, app_with_mocks):
        """Test handling of image loading failures (get_image returns None)."""
        app, _, proc_mock, _ = app_with_mocks

        app.image_processor = proc_mock
        
        proc_mock.configure_mock(**{'get_image.return_value': None})
        mock_progress = MagicMock()
    
        result = app.on_run_prediction(progress=mock_progress)
        
        status_update = result[0]
        if isinstance(status_update, dict):
            assert "No image data found in processor." in status_update.get('value', '')
        else:
            assert "No image data found in processor." in str(status_update)

    def test_on_run_prediction_preprocessing_exception(self, app_with_mocks):
        """Test coverage for 'except Exception' in the Preprocessing section."""
        app, _, proc_mock, _ = app_with_mocks

        app.image_processor = proc_mock

        proc_mock.configure_mock(**{'get_image.return_value': MagicMock()})
        proc_mock.configure_mock(**{'preprocess_image.side_effect': Exception("Preprocessing failed")})
        
        result = app.on_run_prediction(progress=MagicMock())
        found_error = any("Failed at Preprocessing" in str(r) for r in result if isinstance(r, (str, dict)))
        assert found_error is True

    def test_on_run_prediction_unsure_result_none(self, app_with_mocks):
        """Test coverage for unsure prediction logic (results.lower() == 'none')."""
        app, proc_mock, vis_mock, ai_mock = app_with_mocks
        
        proc_mock.configure_mock(**{'get_image.return_value': MagicMock()})
        ai_mock.configure_mock(**{'get_prediction_from_ai_service.return_value': "none"})
        vis_mock.configure_mock(**{'get_description.return_value': "Unknown type description"})
        
        result = app.on_run_prediction(progress=MagicMock())
        found_unsure = any("Unsure prediction" in str(r) for r in result if isinstance(r, (str, dict)))
        assert found_unsure is True
        assert result[8]['visible'] is False
        assert result[9]['visible'] is False

    def test_on_run_prediction_no_results_value_error(self, app_with_mocks):
        """Test coverage for ValueError when no prediction results are returned."""
        app, _, proc_mock, ai_mock = app_with_mocks
        
        proc_mock.configure_mock(**{'get_image.return_value': MagicMock()})
        proc_mock.configure_mock(**{'preprocess_image.return_value': MagicMock()})
        ai_mock.configure_mock(**{'get_prediction_from_ai_service.return_value': None})
        
        result = app.on_run_prediction(progress=MagicMock())
        found_error = any("Failed at AI Prediction" in str(r) for r in result if isinstance(r, (str, dict)))
        assert found_error is True

    def test_on_run_prediction_data_retrieval_exception(self, app_with_mocks):
        """Test coverage for exceptions in the Data Retrieval section."""
        app, ai_mock, proc_mock, vis_mock  = app_with_mocks
        
        app.result_visualiser = vis_mock
        app.ai_model_orchestrator = ai_mock
        app.image_processor = proc_mock

        proc_mock.configure_mock(**{'get_image.return_value': MagicMock()})
        ai_mock.configure_mock(**{'get_prediction_from_ai_service.return_value': "Spring"})
        vis_mock.configure_mock(**{'get_palette_info.side_effect': Exception("Database error")})

        result = app.on_run_prediction(progress=MagicMock())
        
        found_error = any("Failed at Data Retrieval" in str(r) for r in result if isinstance(r, (str, dict)))
        assert found_error is True

    def test_on_run_prediction_ui_update_exception(self, app_with_mocks):
        """Test coverage for exceptions in the UI Update section."""
