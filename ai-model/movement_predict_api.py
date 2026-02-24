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

def sanitize(val):
    """Convert numpy types to native Python types for JSON serialization."""
    if isinstance(val, (np.floating, np.float32, np.float64)):
        return float(val)
    if isinstance(val, (np.integer, np.int32, np.int64)):
        return int(val)
    if isinstance(val, (np.bool_,)):
        return bool(val)
    return val

def sanitize_dict(d: dict) -> dict:
    return {k: sanitize(v) for k, v in d.items()}

@app.post("/predict")
async def predict(
    file: UploadFile = File(...),
    instruction: str = Form("")
):
    contents = await file.read()
    image = Image.open(io.BytesIO(contents))
    frame = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

    features = extract_pose_features(frame, instruction)
    if not features:
        return {
            "prediction": 0,
            "probability": 0.1,
            "error": "No pose detected",
            "instruction_check": {"target_met": False, "target_distance": 1.0, "instruction_type": "none"},
            "features": {}
        }

    instruction_check = features.get("instruction_check", {})
    target_met      = bool(instruction_check.get("target_met", False))
    target_distance = float(instruction_check.get("target_distance", 1.0))

    analyzer = MovementAnalyzer()
    analyzer.update(features["left_elbow_angle"], features["right_elbow_angle"])
    computed_features = analyzer.compute_features()
    if not computed_features:
        return {"error": "Not enough data"}

    computed_features = sanitize_dict(computed_features)

    # ── Use pose check as the primary decision ──
    if target_met:
        return {
            "prediction":        1,
            "probability":       float(max(0.75, 1.0 - target_distance)),
            "features":          computed_features,
            "instruction_check": instruction_check,
        }
    else:
        return {
            "prediction":        0,
            "probability":       float(max(0.1, 1.0 - target_distance)),
            "features":          computed_features,
            "instruction_check": instruction_check,
            "reason":            f"Pose check failed for: {instruction}"
        }