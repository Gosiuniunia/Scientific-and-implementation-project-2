from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import numpy as np
import joblib

# length of feature list 
EXPECTED_NO_FEATURES = 9

app = FastAPI()

# load the model
model = joblib.load("models/svc.pkl")

# request
class PredictionRequest(BaseModel):
    features: list[int]

# response
class PredictionResponse(BaseModel):
    prediction: int


@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest):
    features = np.array(request.features).reshape(1, -1)
    if len(request.features) != EXPECTED_NO_FEATURES:
        raise HTTPException(status_code=400, detail=f"Expected {EXPECTED_NO_FEATURES} features, got {len(request.features)}")
    prediction = model.predict(features)
    return PredictionResponse(prediction=prediction)

