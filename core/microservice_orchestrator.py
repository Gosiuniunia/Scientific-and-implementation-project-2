import requests


class AIServiceOrchestrator:
    """
    AIServiceOrchestrator is a class that serves as an intermediary between the application and the AI service.
    It is responsible for sending requests to the AI service, handling responses, and mapping the prediction

    """

    def __init__(self, ai_url: str):
        self.ai_url = ai_url

    def map_result_to_color_type(self, prediction: int) -> str:
        """
        Maps the prediction result in numerical frm to a color type given as a string.
        Args:
            prediction: The prediction result from the AI service.
        Returns:
            The corresponding color type.
        """
        color_types = {0: "autumn", 1: "spring", 2: "summer", 3: "winter"}
        return color_types.get(prediction, "unknown")

    def get_prediction_from_ai_service(self, features: list[int]) -> str:
        """
        Sends a request to the AI service with the given features and retrieves the prediction.
        Args:
            features: A list of integers representing the features for prediction.
        Returns:
            The color type predicted by the AI service.
        """
        response = requests.post(
            f"{self.ai_url}/predict", json={"features": features}, timeout=5
        )

        if response.status_code != 200:
            raise Exception(f"AI service error: {response.text}")

        prediction = response.json()["prediction"]
        if prediction is None:
            return "none"

        return self.map_result_to_color_type(prediction)
