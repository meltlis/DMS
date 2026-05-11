import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np

# Ensure src is in python path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.pipeline import DMSPipeline, load_yaml


def draw_dashboard(frame: np.ndarray, state: dict) -> np.ndarray:
    """Draw a visual dashboard on the frame."""
    h, w = frame.shape[:2]
    
    # Create sidebar for dashboard
    sidebar_w = 300
    canvas = np.zeros((h, w + sidebar_w, 3), dtype=np.uint8)
    canvas[:, :w] = frame
    
    debug = state.get("debug", {})
    
    # Draw Bounding Boxes
    all_bboxes = debug.get("all_bboxes", {})
    face_bbox = debug.get("face_bbox")
    
    if face_bbox:
        fx, fy, fw, fh = face_bbox
        cv2.rectangle(canvas, (fx, fy), (fx + fw, fy + fh), (255, 0, 0), 2)
        cv2.putText(canvas, "Face", (fx, fy - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
        
    for cls_name, bboxes in all_bboxes.items():
        if cls_name == "face": continue
        color = (0, 0, 255) if cls_name == "phone" else (0, 165, 255) # Red for phone, orange for others
        for (bx, by, bx2, by2) in bboxes:
            cv2.rectangle(canvas, (bx, by), (bx2, by2), color, 2)
            cv2.putText(canvas, cls_name, (bx, by - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
    # Draw Landmarks
    landmarks = debug.get("landmarks_global")
    if landmarks is not None and len(landmarks) > 0:
        # Draw a subset of points to avoid clutter, e.g. eyes and mouth
        # Eye and mouth approximate indices in MediaPipe
        key_indices = [33, 160, 158, 133, 153, 144, 362, 385, 387, 263, 373, 380, 13, 14, 61, 291]
        for idx in key_indices:
            if idx < len(landmarks):
                lx, ly = int(landmarks[idx][0]), int(landmarks[idx][1])
                cv2.circle(canvas, (lx, ly), 1, (0, 255, 0), -1)

    # Sidebar parameters
    x_offset = w + 10
    y_offset = 30
    line_h = 25
    
    def put_text(text: str, color=(255, 255, 255), scale=0.6, thick=1):
        nonlocal y_offset
        cv2.putText(canvas, text, (x_offset, y_offset), cv2.FONT_HERSHEY_SIMPLEX, scale, color, thick)
        y_offset += line_h

    # Title
    put_text("DMS SYSTEM DEMO", (200, 200, 200), 0.8, 2)
    y_offset += 10
    
    # States
    def state_color(val):
        if val == "NORMAL": return (0, 255, 0)
        if val in ["WARNING", "PHONE", "NO_SEATBELT"]: return (0, 255, 255)
        return (0, 0, 255) # ALERT
        
    put_text(f"Fatigue:    {state['fatigue']}", state_color(state['fatigue']), 0.7, 2)
    put_text(f"Distract:   {state['distraction']}", state_color(state['distraction']), 0.7, 2)
    put_text(f"Danger:     {state['danger']}", state_color(state['danger']), 0.7, 2)
    y_offset += 10
    
    # Metrics
    put_text("--- Active Metrics ---", (150, 150, 150))
    earn = (debug.get("ear_left", 0) + debug.get("ear_right", 0)) / 2
    put_text(f"EAR (Eyes): {earn:.3f}")
    put_text(f"MAR (Mouth): {debug.get('mar', 0):.3f}")
    y_offset += 10
    
    put_text("--- Pose ---", (150, 150, 150))
    put_text(f"Pitch: {debug.get('pitch', 0):>6.1f} deg")
    put_text(f"Yaw:   {debug.get('yaw', 0):>6.1f} deg")
    put_text(f"Roll:  {debug.get('roll', 0):>6.1f} deg")
    y_offset += 10
    
    put_text("--- Temporal ---", (150, 150, 150))
    put_text(f"PERCLOS: {debug.get('perclos', 0):.1%}")
    put_text(f"Gaze Awy: {debug.get('gaze_away_duration', 0):.1f}s")
    put_text(f"Eye Cls:  {debug.get('continuous_closed', 0):.1f}s")
    
    if state.get("alerts"):
        y_offset += 10
        put_text("!!! ALERTS !!!", (0, 0, 255), 0.8, 2)
        for a in state["alerts"]:
            put_text(f"- {a}", (0, 0, 255), 0.7, 2)
            
    return canvas


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", "-i", type=str, required=True, help="Input video path")
    parser.add_argument("--output", "-o", type=str, default="demo_output.mp4", help="Output video path")
    parser.add_argument("--show", action="store_true", help="Display window during processing")
    parser.add_argument("--max-frames", type=int, default=600, help="Maximum number of frames to process")
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    thresholds = load_yaml(root / "configs" / "thresholds.yaml")
    runtime = load_yaml(root / "configs" / "runtime.yaml")

    pipeline = DMSPipeline(thresholds=thresholds, runtime=runtime)
    
    cap = cv2.VideoCapture(args.input)
    if not cap.isOpened():
        print(f"Error opening video: {args.input}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0: fps = 30
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # 300px sidebar
    out_width = width + 300
    
    writer = cv2.VideoWriter(
        args.output,
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (out_width, height)
    )

    print(f"Processing {args.input}...")
    print(f"Saving to {args.output} at {fps} FPS")

    frame_count = 0
    start_time = time.time()
    
    while frame_count < args.max_frames:
        ret, frame = cap.read()
        if not ret:
            break
            
        ts = time.time()
        state = pipeline.process(frame, ts)
        
        canvas = draw_dashboard(frame, state)
        writer.write(canvas)
        
        if args.show: # Normally won't work perfectly in VS Code terminal without an X server setup, but allowed
            cv2.imshow("DMS Demo", canvas)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
        frame_count += 1
        if frame_count % 30 == 0:
            print(f"Processed {frame_count} frames...")

    elapsed = time.time() - start_time
    print(f"Done! {frame_count} frames processed in {elapsed:.1f}s ({frame_count/max(elapsed,0.1):.1f} fps).")
    
    cap.release()
    writer.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()