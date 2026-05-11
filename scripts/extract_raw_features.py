import sys
import os
import cv2
import json
from pathlib import Path

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.pipeline import DMSPipeline, load_yaml

def main():
    root = Path(__file__).parent.parent
    thresholds = load_yaml(root / "configs/thresholds.yaml")
    runtime = load_yaml(root / "configs/runtime.yaml")
    pipeline = DMSPipeline(thresholds=thresholds, runtime=runtime)
    
    video_dir = root.parent / "dataset/DROZY/DROZY/videos_i8"
    
    results = {}
    total_videos = list(video_dir.glob("*.mp4"))
    
    for idx, vp in enumerate(total_videos):
        cap = cv2.VideoCapture(str(vp))
        frames_feats = []
        processed = 0
        while True:
            if processed >= 300: 
                break
            ok, frame = cap.read()
            if not ok: 
                break
            processed += 1
            if processed % 4 != 0: 
                continue
            
            ts = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
            if ts <= 0.001: 
                ts = processed / 30.0
            
            state = pipeline.process(frame, ts)
            raw = state["debug"].get("raw_feats", {})
            try:
                raw_ear = (raw.get("ear_left", 0.3) + raw.get("ear_right", 0.3)) / 2
                frames_feats.append({
                    "ts": ts, 
                    "ear": float(raw_ear), 
                    "mar": float(raw.get("mar", 0.0)), 
                    "yaw": float(raw.get("yaw", 0.0)), 
                    "pitch": float(raw.get("pitch", 0.0))
                })
            except Exception as e:
                frames_feats.append({
                    "ts": ts, 
                    "ear": 0.3, 
                    "mar": 0.0, 
                    "yaw": 0.0, 
                    "pitch": 0.0
                })
                
        results[vp.name] = frames_feats
        print(f"[{idx+1}/{len(total_videos)}] Extracted {vp.name}")

    reports_dir = root / "reports"
    reports_dir.mkdir(exist_ok=True)
    with open(reports_dir / "raw_features.json", "w") as f:
        json.dump(results, f)

if __name__ == "__main__":
    main()