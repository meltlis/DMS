"""DMS Web Monitor — FastAPI backend, port 8000."""
from __future__ import annotations

import asyncio
import base64
import json
import sys
import tempfile
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np
import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, UploadFile, File
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.pipeline import DMSPipeline, load_yaml

PORT = 8000
ROOT = Path(__file__).resolve().parents[1]
STATIC = Path(__file__).parent / "static"
STATIC.mkdir(exist_ok=True)
UPLOAD_DIR = Path(tempfile.gettempdir()) / "dms_uploads"
UPLOAD_DIR.mkdir(exist_ok=True)

app = FastAPI(title="DMS Monitor")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

_pipeline: DMSPipeline | None = None
_executor = ThreadPoolExecutor(max_workers=1)
_pipeline_lock = threading.Lock()


def _init_pipeline() -> DMSPipeline:
    global _pipeline
    if _pipeline is None:
        thresholds = load_yaml(ROOT / "configs" / "thresholds.yaml")
        runtime = load_yaml(ROOT / "configs" / "runtime.yaml")
        _pipeline = DMSPipeline(thresholds=thresholds, runtime=runtime)
    return _pipeline


def _do_process(frame: np.ndarray, ts: float) -> dict:
    with _pipeline_lock:
        return _init_pipeline().process(frame, ts)


def _do_reset() -> None:
    p = _init_pipeline()
    try:
        p.fatigue_fsm.reset()
        p.fsm.current_track_id = None
        p.fsm.reset_count = 0
        p.fsm.history = []
        p.temporal._windows.clear()
        p.temporal._closed_since.clear()
        p.danger._phone_first_ts = None
        p.danger._start_ts = None
        p.danger._seatbelt_seen = False
    except Exception as exc:
        print(f"[reset] {exc}")


async def _aprocess(frame: np.ndarray, ts: float) -> dict:
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(_executor, _do_process, frame, ts)


# ── frame annotation ──────────────────────────────────────────────────────────

_OCV = {
    "NORMAL":  (100, 210, 120),
    "WARNING": (60,  170, 240),
    "ALERT":   (60,  60,  220),
}


def _annotate(frame: np.ndarray, state: dict) -> np.ndarray:
    out = frame.copy()
    h, w = out.shape[:2]
    fatigue = state.get("fatigue", "NORMAL")
    color = _OCV.get(fatigue, _OCV["NORMAL"])
    debug = state.get("debug", {})

    bbox = debug.get("face_bbox")
    if bbox and all(v is not None for v in bbox):
        x, y, bw, bh = [int(v) for v in bbox]
        cv2.rectangle(out, (x, y), (x + bw, y + bh), color, 2)
        (tw, th), _ = cv2.getTextSize(fatigue, cv2.FONT_HERSHEY_SIMPLEX, 0.50, 1)
        cv2.rectangle(out, (x, y - th - 8), (x + tw + 8, y), color, -1)
        cv2.putText(out, fatigue, (x + 4, y - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.50, (255, 255, 255), 1)

    overlay = out.copy()
    cv2.rectangle(overlay, (0, 0), (w, 30), (12, 12, 12), -1)
    cv2.addWeighted(overlay, 0.65, out, 0.35, 0, out)

    d = debug
    parts: list[str] = []
    el, er = d.get("ear_left"), d.get("ear_right")
    if el is not None and er is not None:
        parts.append(f"EAR {(el + er) / 2:.2f}")
    if d.get("mar") is not None:
        parts.append(f"MAR {d['mar']:.2f}")
    if d.get("perclos") is not None:
        parts.append(f"PERCLOS {d['perclos']:.0%}")
    cv2.putText(out, "  |  ".join(parts), (8, 19),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, (200, 200, 200), 1)

    return out


def _enc(frame: np.ndarray, q: int = 55) -> str:
    _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, q])
    return base64.b64encode(buf.tobytes()).decode()


def _safe_f(v) -> float | None:
    try:
        return round(float(v), 4) if v is not None else None
    except Exception:
        return None


def _serialize(state: dict) -> dict:
    d = state.get("debug", {})
    return {
        "fatigue":           state.get("fatigue", "NORMAL"),
        "distraction":       state.get("distraction", "NORMAL"),
        "danger":            state.get("danger", "NORMAL"),
        "alerts":            state.get("alerts", []),
        "track_id":          state.get("track_id", -1),
        "ear_left":          _safe_f(d.get("ear_left")),
        "ear_right":         _safe_f(d.get("ear_right")),
        "mar":               _safe_f(d.get("mar")),
        "pitch":             _safe_f(d.get("pitch")),
        "yaw":               _safe_f(d.get("yaw")),
        "roll":              _safe_f(d.get("roll")),
        "perclos":           _safe_f(d.get("perclos")),
        "nod_freq":          _safe_f(d.get("nod_freq")),
        "yawn_count":        d.get("yawn_count"),
        "continuous_closed": _safe_f(d.get("continuous_closed")),
        "lstm_score":        _safe_f(d.get("lstm_score")),
        "lstm_pred":         d.get("lstm_pred"),
    }


# ── routes ────────────────────────────────────────────────────────────────────

@app.on_event("startup")
async def _startup() -> None:
    loop = asyncio.get_event_loop()
    await loop.run_in_executor(_executor, _init_pipeline)
    print(f"\n  DMS Monitor  →  http://localhost:{PORT}\n")


@app.get("/", response_class=HTMLResponse)
async def index() -> str:
    return (STATIC / "index.html").read_text(encoding="utf-8")


@app.post("/reset")
async def reset_pipeline():
    loop = asyncio.get_event_loop()
    await loop.run_in_executor(_executor, _do_reset)
    return {"ok": True}


@app.post("/analyze")
async def analyze_image(file: UploadFile = File(...)):
    data = await file.read()
    arr = np.frombuffer(data, np.uint8)
    frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if frame is None:
        return JSONResponse({"error": "Cannot decode image"}, status_code=400)
    state = await _aprocess(frame, time.time())
    annotated = _annotate(frame, state)
    return {"frame": _enc(annotated), "state": _serialize(state)}


@app.websocket("/ws")
async def ws_endpoint(ws: WebSocket) -> None:
    await ws.accept()
    print("[ws] client connected")
    ts = 0.0
    try:
        while True:
            raw = await ws.receive_bytes()
            arr = np.frombuffer(raw, np.uint8)
            frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            if frame is None:
                continue
            ts += 1.0 / 30.0
            import time
            t0 = time.time()
            try:
                state = await _aprocess(frame, ts)
                t1 = time.time()
                annotated = _annotate(frame, state)
                t2 = time.time()
                payload = json.dumps({
                    "frame": _enc(annotated),
                    "state": _serialize(state),
                })
                t3 = time.time()
                await ws.send_text(payload)
                t4 = time.time()
                total = (t4 - t0) * 1000
                if int(ts) <= 3 or total > 50:
                    print(f"[perf] frame ts={ts:.2f} total={total:.1f}ms "
                          f"(process={ (t1-t0)*1000:.1f} annotate={ (t2-t1)*1000:.1f} "
                          f"encode={ (t3-t2)*1000:.1f} send={ (t4-t3)*1000:.1f})")
            except Exception as e:
                print(f"[ws] PROCESS ERROR: {e}")
                import traceback
                traceback.print_exc()
    except WebSocketDisconnect:
        print("[ws] client disconnected")
    except Exception as exc:
        print(f"[ws] FATAL: {exc}")
        import traceback
        traceback.print_exc()


@app.post("/upload-video")
async def upload_video(file: UploadFile = File(...)) -> dict:
    uid = uuid.uuid4().hex[:10]
    suffix = Path(file.filename or "upload.avi").suffix or ".avi"
    dest = UPLOAD_DIR / f"{uid}{suffix}"
    data = await file.read()
    dest.write_bytes(data)
    return {"id": uid, "path": str(dest), "name": file.filename, "size": len(data)}


@app.websocket("/ws/file")
async def ws_file(ws: WebSocket) -> None:
    """Server-driven video: OpenCV reads the file, pushes annotated frames."""
    await ws.accept()
    video_path: str | None = None
    try:
        init = await asyncio.wait_for(ws.receive_json(), timeout=30)
        video_path = init.get("path", "")
        if not video_path or not Path(video_path).exists():
            await ws.send_json({"error": "文件不存在"})
            return

        _do_reset()

        loop = asyncio.get_event_loop()
        queue: asyncio.Queue = asyncio.Queue(maxsize=6)

        def _read_video() -> None:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                asyncio.run_coroutine_threadsafe(
                    queue.put({"error": "OpenCV 无法打开该视频，请检查编解码器"}), loop)
                return
            total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
            src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
            stride, ts, n = 2, 0.0, 0
            try:
                while True:
                    ok, frame = cap.read()
                    if not ok:
                        break
                    n += 1
                    ts += 1.0 / src_fps
                    if n % stride != 0:
                        continue
                    state = _do_process(frame, ts)
                    ann = _annotate(frame, state)
                    msg = {
                        "frame":    _enc(ann),
                        "state":    _serialize(state),
                        "progress": round(n / total, 3),
                    }
                    fut = asyncio.run_coroutine_threadsafe(queue.put(msg), loop)
                    fut.result(timeout=20)
            except Exception as exc:
                print(f"[video thread] {exc}")
            finally:
                cap.release()
                asyncio.run_coroutine_threadsafe(queue.put(None), loop)

        threading.Thread(target=_read_video, daemon=True).start()

        while True:
            item = await asyncio.wait_for(queue.get(), timeout=120)
            if item is None:
                await ws.send_json({"done": True})
                break
            if "error" in item:
                await ws.send_json(item)
                break
            await ws.send_text(json.dumps(item))

    except (WebSocketDisconnect, asyncio.TimeoutError):
        pass
    except Exception as exc:
        print(f"[ws/file] {exc}")
    finally:
        if video_path:
            try:
                Path(video_path).unlink(missing_ok=True)
            except Exception:
                pass


if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=PORT, log_level="warning")
