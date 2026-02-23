from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
import joblib

app = FastAPI(title="Pose-Based Cognitive Risk Model")

model = joblib.load("risk_model.pkl")
scaler = joblib.load("scaler.pkl")

class InputData(BaseModel):
    reaction_time: float
    sequence_accuracy: float
    avg_joint_angle_error: float
    movement_smoothness: float
    instruction_delay: float
    error_repetition_count: int

@app.post("/predict")
def predict(data: InputData):

    features = np.array([[
        data.reaction_time,
        data.sequence_accuracy,
        data.avg_joint_angle_error,
        data.movement_smoothness,
        data.instruction_delay,
        data.error_repetition_count
    ]])

    scaled = scaler.transform(features)

    prediction = model.predict(scaled)[0]
    probability = model.predict_proba(scaled)[0][1]

    risk = "Low"
    if probability > 0.7:
        risk = "High"
    elif probability > 0.4:
        risk = "Medium"

    return {
        "prediction": int(prediction),
        "risk_probability": float(probability),
        "risk_level": risk
    }
