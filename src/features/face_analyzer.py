from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# MediaPipe EAR indices (6 points per eye)
_LEFT_EYE_IDX = [33, 160, 158, 133, 153, 144]
_RIGHT_EYE_IDX = [362, 385, 387, 263, 373, 380]

# MAR indices: vertical = 13 (upper lip center) <-> 14 (lower lip center)
# horizontal = 61 (left mouth corner) <-> 291 (right mouth corner)
_MAR_VERTICAL = [13, 14]
_MAR_HORIZONTAL = [61, 291]

# Head pose PnP reference points (MediaPipe index -> 3D mm)
_PNP_IDX = [1, 152, 33, 263, 61, 291]
_PNP_3D = np.array(
    [
        [0.0, 0.0, 0.0],       # nose tip
        [0.0, -63.6, -12.5],   # chin
        [-43.3, 32.7, -26.0],  # left eye outer
        [43.3, 32.7, -26.0],   # right eye outer
        [-28.9, -28.9, -24.1], # left mouth
        [28.9, -28.9, -24.1],  # right mouth
    ],
    dtype=np.float32,
)


def _distance(p1: np.ndarray, p2: np.ndarray) -> float:
    return float(np.linalg.norm(p1 - p2))


def _ear(landmarks: np.ndarray, indices: list[int]) -> float:
    p = [landmarks[i] for i in indices]
    vert1 = _distance(p[1], p[5])
    vert2 = _distance(p[2], p[4])
    horiz = _distance(p[0], p[3])
    if horiz == 0:
        return 0.0
    return (vert1 + vert2) / (2.0 * horiz)


def _mar(landmarks: np.ndarray) -> float:
    vertical = _distance(landmarks[_MAR_VERTICAL[0]], landmarks[_MAR_VERTICAL[1]])
    horizontal = _distance(landmarks[_MAR_HORIZONTAL[0]], landmarks[_MAR_HORIZONTAL[1]])
    if horizontal == 0:
        return 0.0
    return vertical / horizontal


def _head_pose(landmarks: np.ndarray, image_w: int, image_h: int) -> tuple[float, float, float]:
    """Return (pitch, yaw, roll) in degrees using solvePnP."""
    image_points = np.array([landmarks[i] for i in _PNP_IDX], dtype=np.float32)
    focal_length = image_w
    center = (image_w / 2.0, image_h / 2.0)
    camera_matrix = np.array(
        [[focal_length, 0, center[0]], [0, focal_length, center[1]], [0, 0, 1]],
        dtype=np.float32,
    )
    dist_coeffs = np.zeros((4, 1), dtype=np.float32)
    success, rotation_vec, translation_vec = cv2.solvePnP(
        _PNP_3D, image_points, camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE
    )
    if not success:
        return 0.0, 0.0, 0.0
    rotation_mat, _ = cv2.Rodrigues(rotation_vec)
    # Decompose rotation matrix to pitch, yaw, roll
    pitch = float(np.degrees(np.arcsin(-rotation_mat[2, 0])))
    yaw = float(np.degrees(np.arctan2(rotation_mat[2, 1], rotation_mat[2, 2])))
    roll = float(np.degrees(np.arctan2(rotation_mat[1, 0], rotation_mat[0, 0])))
    return pitch, yaw, roll


def _model_path() -> Path:
    root = Path(__file__).resolve().parents[2]
    return root / "weights" / "face_landmarker.task"


class FaceAnalyzer:
    """Real MediaPipe Face Mesh extractor following CLAUDE.md geometry contract."""

    def __init__(self, ear_threshold: float, mar_threshold: float) -> None:
        self.ear_threshold = ear_threshold
        self.mar_threshold = mar_threshold

        from mediapipe.tasks.python.vision import FaceLandmarker, FaceLandmarkerOptions, RunningMode
        from mediapipe.tasks.python.core.base_options import BaseOptions

        model = _model_path()
        if not model.exists():
            raise FileNotFoundError(f"FaceLandmarker model not found: {model}")

        base_options = BaseOptions(model_asset_path=str(model))
        options = FaceLandmarkerOptions(
            base_options=base_options,
            running_mode=RunningMode.IMAGE,
            num_faces=1,
            min_face_detection_confidence=0.5,
            min_face_presence_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self._landmarker = FaceLandmarker.create_from_options(options)

    def analyze(self, face_roi: np.ndarray) -> Dict[str, float | bool | np.ndarray | None]:
        if face_roi.size == 0:
            return self._synthetic_analysis()

        # OpenCV is BGR, MediaPipe expects RGB
        rgb = cv2.cvtColor(face_roi, cv2.COLOR_BGR2RGB)
        h, w = rgb.shape[:2]
        import mediapipe as mp
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        
        try:
            result = self._landmarker.detect(mp_image)
        except Exception as e:
            logger.debug(f"MediaPipe detection failed: {e}, using synthetic fallback")
            return self._synthetic_analysis()

        if not result.face_landmarks:
            return self._synthetic_analysis()

        landmarks_norm = result.face_landmarks[0]
        landmarks = np.array([(lm.x * w, lm.y * h) for lm in landmarks_norm], dtype=np.float32)

        ear_left = _ear(landmarks, _LEFT_EYE_IDX)
        ear_right = _ear(landmarks, _RIGHT_EYE_IDX)
        mar = _mar(landmarks)
        pitch, yaw, roll = _head_pose(landmarks, w, h)

        avg_ear = (ear_left + ear_right) / 2.0

        return {
            "ear_left": float(ear_left),
            "ear_right": float(ear_right),
            "mar": float(mar),
            "pitch": float(pitch),
            "yaw": float(yaw),
            "roll": float(roll),
            "eye_closed": avg_ear < self.ear_threshold,
            "is_yawning": mar > self.mar_threshold,
            "landmarks_468": landmarks,
        }

    def _synthetic_analysis(self) -> Dict[str, float | bool | np.ndarray | None]:
        """Return neutral (non-triggering) values when MediaPipe cannot detect a face.

        Absence of face detection is not evidence of fatigue — returning a
        simulated-fatigue signal here causes systematic false positives whenever
        the camera angle or lighting falls outside MediaPipe's operating range.
        """
        neutral_ear = 0.35  # clearly above any closed-eye threshold
        return {
            "ear_left": neutral_ear,
            "ear_right": neutral_ear,
            "mar": 0.0,
            "pitch": 0.0,
            "yaw": 0.0,
            "roll": 0.0,
            "eye_closed": False,
            "is_yawning": False,
            "landmarks_468": None,
        }
