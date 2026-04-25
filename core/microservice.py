import numpy as np
import joblib
from itertools import combinations
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Annotated

EXPECTED_NO_FEATURES = 9
MODEL_PATH = "models/svc.pkl"


class AIModelService:
    """
    AIModelService is a class that encapsulates the logic for loading a pre-trained machine learning model and making predictions based on input features.
    The class initializes by loading the model from a specified path and extracting the necessary components for prediction
    """

    def __init__(self, model_path: str):
        model = joblib.load(model_path)

        self.scaler = model.named_steps["scaler"]
        self.svc = model.named_steps["svc"]

        self.classes = self.svc.classes_
        self.pairs = list(combinations(self.classes, 2))
        self.class_to_index = {c: i for i, c in enumerate(self.classes)}

    def predict_with_voting(self, X_scaled, decisions, threshold=0.2):
        n_samples = X_scaled.shape[0]
        votes = np.zeros((n_samples, len(self.classes)), dtype=int)

        for idx, (c1, c2) in enumerate(self.pairs):
            i1 = self.class_to_index[c1]
            i2 = self.class_to_index[c2]

            for i in range(n_samples):
                val = decisions[i, idx]

                if abs(val) < threshold:
                    continue
                elif val > 0:
                    votes[i, i1] += 1
                else:
                    votes[i, i2] += 1

        return votes

    def count_votes(self, votes):
        v = votes[0]
        max_vote = np.max(v)

        if np.sum(v == max_vote) > 1:
            return None

        return np.argmax(v)

    def predict(self, features: list[int]) -> str:
        if len(features) != EXPECTED_NO_FEATURES:
            raise ValueError("Invalid feature length")

        # scale input
        X_scaled = self.scaler.transform([features])

        # decision function
        decisions = self.svc.decision_function(X_scaled)

        # voting
        votes = self.predict_with_voting(X_scaled, decisions)

        prediction = self.count_votes(votes)

        return prediction


app = FastAPI()
model_service = AIModelService(MODEL_PATH)


class PredictionRequest(BaseModel):
    """
    PredictionRequest is a Pydantic model that defines the structure of the request body for making predictions.
    It contains a single field, features, which is a list of integers representing the input features for the prediction.
    This model is used to validate the incoming request data and ensure that it conforms to the expected format before processing it in the prediction endpoint.
    """

    features: Annotated[
        list[int],
        Field(
            min_length=9,
            max_length=9,
            description="List of 9 numeric features extracted from the image",
        ),
    ]


class PredictionResponse(BaseModel):
    """
    PredictionResponse is a Pydantic model that defines the structure of the response body for the prediction endpoint.
    It contains a single field, prediction, which is an integer representing the predicted class label for the input features.
    This model is used to format the response data in a consistent way, making it easier for clients to parse and understand the prediction results returned by the API.
    """

    prediction: int


@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest):
    try:
        prediction = model_service.predict(request.features)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    return PredictionResponse(prediction=prediction)
