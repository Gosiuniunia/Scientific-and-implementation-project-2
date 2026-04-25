import requests


class AIServiceOrchestrator:
    def __init__(self, ai_url: str):
        self.ai_url = ai_url

    def map_result_to_color_type(self, prediction: int) -> str:
        color_types = {0: "autumn", 1: "spring", 2: "summer", 3: "winter"}
        return color_types.get(prediction, "unknown")

    def get_prediction_from_ai_service(self, features: list[int]) -> str:
        response = requests.post(
            f"{self.ai_url}/predict", json={"features": features}, timeout=5
        )

        if response.status_code != 200:
            raise Exception(f"AI service error: {response.text}")

        prediction = response.json()["prediction"]

        return self.map_result_to_color_type(prediction)
