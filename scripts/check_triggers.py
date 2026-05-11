import json
from src.temporal.aggregator import TemporalAggregator
from src.decision.rules import fatigue_level

def check_triggers():
    with open("reports/raw_features.json") as f:
        video_data = json.load(f)
    
    triggers = {"continuous_closed": 0, "perclos": 0, "yawn": 0, "nod": 0, "score": 0}
    
    for vp_name, feats in video_data.items():
        agg = TemporalAggregator(fps=30, window_seconds=3.0, yaw_threshold_deg=30.0)
        agg.pitch_threshold_deg = 15.0
        
        for f in feats:
            feat_dict = {
                "eye_closed": f["ear"] < 0.16,
                "is_yawning": f["mar"] > 0.6,
                "yaw": f["yaw"],
                "pitch": f["pitch"]
            }
            m = agg.update(feat_dict, track_id=1, ts=f["ts"])
            
            # evaluate components
            perclos = m["perclos"]
            continuous_closed = m.get("continuous_closed", 0.0)
            yawn_count = int(m.get("yawn_count", 0))
            nod_freq = float(m.get("nod_freq", 0.0))
            
            score = (0.5 * min(perclos / 0.40, 1.0) + 0.3 * min(nod_freq / 10.0, 1.0) + 0.2 * min(yawn_count / 3.0, 1.0))
            
            if continuous_closed >= 3.0: triggers["continuous_closed"] += 1
            elif score > 0.25: triggers["score"] += 1
            elif perclos > 0.15 or yawn_count >= 2: triggers["perclos"] += 1
            
    print(triggers)

if __name__ == "__main__":
    check_triggers()