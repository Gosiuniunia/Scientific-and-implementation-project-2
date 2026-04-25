from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import numpy as np
import joblib

EXPECTED_NO_FEATURES = 9
MODEL_PATH = "models/svc.pkl"


class AIModelService:
    def __init__(self, model_path: str):
        self.model = joblib.load(model_path)

    def predict(self, features: list[int]) -> int:
        if len(features) != EXPECTED_NO_FEATURES:
            raise ValueError("Invalid feature length")

        x = np.array(features).reshape(1, -1)
        return int(self.model.predict(x)[0])


app = FastAPI()
model_service = AIModelService(MODEL_PATH)


class PredictionRequest(BaseModel):
    features: list[int]


class PredictionResponse(BaseModel):
    prediction: int


@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest):
    try:
        prediction = model_service.predict(request.features)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    return PredictionResponse(prediction=prediction)
