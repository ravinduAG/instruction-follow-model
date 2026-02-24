from fastapi import FastAPI, UploadFile, File
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

# Load your model
try:
    model = joblib.load("risk_model.pkl")
    scaler = joblib.load("scaler.pkl")  # Also load scaler
except Exception as e:
    model = None
    scaler = None
    print(f"Warning: Could not load model/scaler file. {e}")

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    # Read image
    contents = await file.read()
    image = Image.open(io.BytesIO(contents))
    frame = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    
    features = extract_pose_features(frame)
    if not features:
        return {"error": "No pose detected"}

    analyzer = MovementAnalyzer()
    analyzer.update(features["left_elbow_angle"], features["right_elbow_angle"])
    computed_features = analyzer.compute_features()
    if not computed_features:
        return {"error": "Not enough data"}

    # Predict
    X = np.array([list(computed_features.values())])
    prediction = model.predict(X)[0]
    probability = model.predict_proba(X).max()

    return {"prediction": int(prediction), "probability": float(probability), "features": computed_features}