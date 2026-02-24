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

def check_instruction(landmarks, instruction: str) -> dict:
    """Check if the pose matches the given instruction."""

    def dist(a, b):
        return float(np.sqrt((landmarks[a].x - landmarks[b].x)**2 + (landmarks[a].y - landmarks[b].y)**2))

    instruction_lower = instruction.lower()

    # ── Touch your nose ──────────────────────────────────────────────────────
    if "nose" in instruction_lower:
        left_dist  = dist(15, 0)   # LEFT_WRIST=15, NOSE=0
        right_dist = dist(16, 0)   # RIGHT_WRIST=16, NOSE=0
        min_dist   = min(left_dist, right_dist)
        return {
            "target_distance": float(min_dist),
            "target_met":      bool(min_dist < 0.15),
            "instruction_type": "touch_nose"
        }

    # ── Touch your head ──────────────────────────────────────────────────────
    elif "head" in instruction_lower:
        left_wrist  = landmarks[15]
        right_wrist = landmarks[16]
        left_ear    = landmarks[7]   # LEFT_EAR=7
        right_ear   = landmarks[8]   # RIGHT_EAR=8

        left_above  = bool(left_wrist.y  < left_ear.y  + 0.05)
        right_above = bool(right_wrist.y < right_ear.y + 0.05)
        best_dist   = min(dist(15, 7), dist(16, 8))
        return {
            "target_distance": float(best_dist),
            "target_met":      bool(left_above or right_above),
            "instruction_type": "touch_head"
        }

    # ── Wave your hand ───────────────────────────────────────────────────────
    elif "wave" in instruction_lower:
        left_wrist     = landmarks[15]
        right_wrist    = landmarks[16]
        left_shoulder  = landmarks[11]
        right_shoulder = landmarks[12]

        left_raised  = bool(left_wrist.y  < left_shoulder.y  - 0.05)
        right_raised = bool(right_wrist.y < right_shoulder.y - 0.05)
        height_diff  = min(
            left_shoulder.y  - left_wrist.y,
            right_shoulder.y - right_wrist.y
        )
        return {
            "target_distance": float(abs(height_diff)),
            "target_met":      bool(left_raised or right_raised),
            "instruction_type": "wave_hand"
        }

    # ── Clap your hands ──────────────────────────────────────────────────────
    elif "clap" in instruction_lower:
        wrist_dist = dist(15, 16)  # LEFT_WRIST=15, RIGHT_WRIST=16
        return {
            "target_distance": float(wrist_dist),
            "target_met":      bool(wrist_dist < 0.15),
            "instruction_type": "clap_hands"
        }

    # ── Unknown ──────────────────────────────────────────────────────────────
    else:
        return {
            "target_distance": 1.0,
            "target_met":      False,
            "instruction_type": "unknown"
        }


def extract_pose_features(frame, instruction: str = ""):
    _init_pose()

    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)
    results = pose.detect(mp_image)

    if not results.pose_landmarks:
        return None

    landmarks = results.pose_landmarks[0]

    left_shoulder = [landmarks[11].x, landmarks[11].y]
    left_elbow    = [landmarks[13].x, landmarks[13].y]
    left_wrist    = [landmarks[15].x, landmarks[15].y]

    right_shoulder = [landmarks[12].x, landmarks[12].y]
    right_elbow    = [landmarks[14].x, landmarks[14].y]
    right_wrist    = [landmarks[16].x, landmarks[16].y]

    left_elbow_angle  = calculate_angle(left_shoulder, left_elbow, left_wrist)
    right_elbow_angle = calculate_angle(right_shoulder, right_elbow, right_wrist)

    instruction_check = check_instruction(landmarks, instruction) if instruction else {
        "target_distance": 1.0,
        "target_met":      False,
        "instruction_type": "none"
    }

    return {
        "left_elbow_angle":  left_elbow_angle,
        "right_elbow_angle": right_elbow_angle,
        "instruction_check": instruction_check,
    }