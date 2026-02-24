from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from movement_features import MovementAnalyzer
from pose_feature_extractor import extract_pose_features
import cv2
import numpy as np
from PIL import Image
import io
import joblib

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

try:
    model = joblib.load("risk_model.pkl")
    scaler = joblib.load("scaler.pkl")
except Exception as e:
    model = None
    scaler = None
    print(f"Warning: Could not load model/scaler file. {e}")

@app.post("/predict")
async def predict(
    file: UploadFile = File(...),
    instruction: str = Form("")
):
    contents = await file.read()
    image = Image.open(io.BytesIO(contents))
    frame = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

    # Pass instruction into feature extractor
    features = extract_pose_features(frame, instruction)
    if not features:
        return {"error": "No pose detected"}

    instruction_check = features.get("instruction_check", {})
    target_met = instruction_check.get("target_met", False)
    target_distance = instruction_check.get("target_distance", 1.0)

    analyzer = MovementAnalyzer()
    analyzer.update(features["left_elbow_angle"], features["right_elbow_angle"])
    computed_features = analyzer.compute_features()
    if not computed_features:
        return {"error": "Not enough data"}

    # ── If instruction check explicitly fails, return prediction=0 early ──
    if not target_met:
        return {
            "prediction": 0,
            "probability": max(0.1, 1.0 - target_distance),  # rough confidence
            "features": computed_features,
            "instruction_check": instruction_check,
            "reason": f"Pose check failed for: {instruction}"
        }

    # ── Run ML model for final prediction ──
    X = np.array([list(computed_features.values())])
    prediction = model.predict(X)[0]
    probability = model.predict_proba(X).max()

    return {
        "prediction": int(prediction),
        "probability": float(probability),
        "features": computed_features,
        "instruction_check": instruction_check,
    }