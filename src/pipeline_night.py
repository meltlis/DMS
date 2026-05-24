"""夜间/近红外摄像头版 DMS Pipeline。

使用 weights/best.pt（YOLO classify，drowsy/notdrowsy 二分类）
替代 MediaPipe EAR，适配近红外摄像头输入。

流程：
  IR帧 → CLAHE增强 → Haar人脸检测 → 裁剪ROI → best.pt分类
       → 滚动窗口PERCLOS等效计算 → 疲劳状态输出
"""
from __future__ import annotations

import collections
import threading
from pathlib import Path
from typing import Any, Dict

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
_IR_CLASSIFIER_PATH = ROOT / "weights" / "best.pt"
_HAAR_PATH = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"


def _clahe_enhance(frame: np.ndarray) -> np.ndarray:
    """CLAHE 对比度增强，适配近红外图像。"""
    if len(frame.shape) == 3:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    else:
        gray = frame.copy()
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    return cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)


class _RollingWindow:
    """固定时长滚动窗口，统计 drowsy 帧比例（PERCLOS 等效）。"""

    def __init__(self, window_seconds: float = 10.0, fps: int = 30) -> None:
        maxlen = int(window_seconds * fps)
        self._buf: collections.deque[float] = collections.deque(maxlen=maxlen)

    def push(self, drowsy_prob: float) -> None:
        self._buf.append(drowsy_prob)

    def perclos(self, threshold: float = 0.5) -> float:
        if not self._buf:
            return 0.0
        return sum(1 for p in self._buf if p >= threshold) / len(self._buf)

    def mean_prob(self) -> float:
        return float(np.mean(self._buf)) if self._buf else 0.0

    def reset(self) -> None:
        self._buf.clear()


class NightPipeline:
    """近红外/夜间驾驶员疲劳检测 pipeline。

    state 输出格式与 DMSPipeline 保持一致，方便 web/app.py 统一处理。
    """

    def __init__(
        self,
        window_seconds: float = 10.0,
        fps: int = 30,
        warning_perclos: float = 0.30,
        alert_perclos: float = 0.50,
        device: str = "cuda",
    ) -> None:
        self.warning_perclos = warning_perclos
        self.alert_perclos = alert_perclos
        self._window = _RollingWindow(window_seconds, fps)
        self._face_det = cv2.CascadeClassifier(_HAAR_PATH)
        self._lock = threading.Lock()

        from ultralytics import YOLO
        self._model = YOLO(str(_IR_CLASSIFIER_PATH))
        # classify 模型用 CPU 推理也很快，CUDA 加速可选
        self._device = device

    def reset(self) -> None:
        self._window.reset()

    def process(self, frame: np.ndarray, ts: float) -> Dict[str, Any]:
        enhanced = _clahe_enhance(frame)
        gray = cv2.cvtColor(enhanced, cv2.COLOR_BGR2GRAY)

        # 人脸检测
        faces = self._face_det.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=4, minSize=(60, 60)
        )

        face_bbox = None
        drowsy_prob = 0.0
        detected = False

        if len(faces) > 0:
            x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
            face_bbox = [int(x), int(y), int(w), int(h)]
            roi = enhanced[y:y+h, x:x+w]
            if roi.size > 0:
                result = self._model(roi, verbose=False, device=self._device)
                probs = result[0].probs
                # names: {0: 'drowsy', 1: 'notdrowsy'}
                drowsy_prob = float(probs.data[0])
                detected = True
        else:
            # 无人脸时用全帧兜底（DROZY 场景人脸占画面大）
            result = self._model(enhanced, verbose=False, device=self._device)
            probs = result[0].probs
            drowsy_prob = float(probs.data[0])
            detected = True

        self._window.push(drowsy_prob if detected else 0.0)

        perclos = self._window.perclos(threshold=0.5)
        mean_prob = self._window.mean_prob()

        # 疲劳等级判定
        if perclos >= self.alert_perclos:
            fatigue = "ALERT"
        elif perclos >= self.warning_perclos:
            fatigue = "WARNING"
        else:
            fatigue = "NORMAL"

        alerts = [fatigue] if fatigue != "NORMAL" else []

        return {
            "fatigue":     fatigue,
            "distraction": "NORMAL",  # 夜间版暂不检测注意力分散
            "danger":      "NORMAL",  # 夜间版暂不检测手机/安全带
            "alerts":      alerts,
            "track_id":    0,
            "debug": {
                "ear_left":          None,
                "ear_right":         None,
                "mar":               None,
                "pitch":             None,
                "yaw":               None,
                "roll":              None,
                "perclos":           round(perclos, 4),
                "nod_freq":          None,
                "yawn_count":        None,
                "gaze_away_duration": None,
                "continuous_closed": None,
                "lstm_score":        round(drowsy_prob, 4),
                "lstm_pred":         1 if drowsy_prob >= 0.5 else 0,
                "face_bbox":         face_bbox,
                "landmarks_global":  None,
                "all_bboxes":        {},
                "ir_drowsy_prob":    round(drowsy_prob, 4),
                "ir_perclos":        round(perclos, 4),
            },
        }
