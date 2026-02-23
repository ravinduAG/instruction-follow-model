import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

pose = None

def _init_pose():
    global pose
    if pose is None:
        base_options = python.BaseOptions(model_asset_path='pose_landmarker_lite.task')
        options = vision.PoseLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.IMAGE
        )
        pose = vision.PoseLandmarker.create_from_options(options)

def calculate_angle(a, b, c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)

    radians = np.arccos(
        np.dot(a-b, c-b) /
        (np.linalg.norm(a-b) * np.linalg.norm(c-b) + 1e-6)
    )
    return np.degrees(radians)

def extract_pose_features(frame):
    _init_pose()
    
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)
    results = pose.detect(mp_image)

    if not results.pose_landmarks:
        return None

    landmarks = results.pose_landmarks[0]

    left_shoulder = [landmarks[11].x, landmarks[11].y]
    left_elbow = [landmarks[13].x, landmarks[13].y]
    left_wrist = [landmarks[15].x, landmarks[15].y]

    right_shoulder = [landmarks[12].x, landmarks[12].y]
    right_elbow = [landmarks[14].x, landmarks[14].y]
    right_wrist = [landmarks[16].x, landmarks[16].y]

    left_elbow_angle = calculate_angle(left_shoulder, left_elbow, left_wrist)
    right_elbow_angle = calculate_angle(right_shoulder, right_elbow, right_wrist)

    return {
        "left_elbow_angle": left_elbow_angle,
        "right_elbow_angle": right_elbow_angle
    }