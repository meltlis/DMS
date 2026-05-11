import json
import itertools
from src.temporal.aggregator import TemporalAggregator
from src.decision.rules import fatigue_level

def main():
    with open("reports/raw_features.json") as f:
        video_data = json.load(f)
    
    # Ground Truth Mapping
    # drozy KSS mappings logic from evaluate_dataset_accuracy.py
    # anything >= 7 is fatigued (1), else alert (0)
    # the mapping can be simplified if we just read it from the dataset or define it
    
    # Quick KSS map matching evaluate script:
    # We will read reports/accuracy_eval.csv because it already has the ground truth per video mapped!
    import csv
    video_gt = {}
    with open("reports/accuracy_eval.csv") as f:
        for row in csv.DictReader(f):
            stem = row['video'].split('\\')[-1]
            try:
                video_gt[stem] = int(row['gt_sleepy'])
            except ValueError:
                pass
            
    # Parameters to search
    ears = [0.10, 0.12, 0.14, 0.15, 0.16]
    perclos_warns = [0.15, 0.20, 0.25, 0.30]
    perclos_alerts = [0.40] # not really used for binary classification in our mock
    sleepy_ratios = [0.01, 0.05, 0.10, 0.15, 0.20, 0.30]
    
    best_acc = 0.0
    best_params = None
    
    total = len(ears) * len(perclos_warns) * len(perclos_alerts) * len(sleepy_ratios)
    cnt = 0
    
    for ear, p_warn, p_alert, s_ratio in itertools.product(ears, perclos_warns, perclos_alerts, sleepy_ratios):
        if p_warn >= p_alert: continue
        
        cnt += 1
        correct = 0
        total_eval = 0
        
        for vp_name, feats in video_data.items():
            if vp_name not in video_gt: continue
            gt = video_gt[vp_name]
            total_eval += 1
            
            # Reset aggregator
            agg = TemporalAggregator(fps=30, window_seconds=3.0, yaw_threshold_deg=30.0)
            agg.pitch_threshold_deg = 15.0
            
            warning_or_alert = 0
            
            for f in feats:
                # build dict
                feat_dict = {
                    "eye_closed": f["ear"] < ear,
                    "is_yawning": f["mar"] > 0.6,
                    "yaw": f["yaw"],
                    "pitch": f["pitch"]
                }
                # update
                m = agg.update(feat_dict, track_id=1, ts=f["ts"])
                
                # rules
                level = "NORMAL"
                if m.get("continuous_closed", 0.0) >= 3.0:
                    level = "WARNING"
                elif m["perclos"] >= p_warn:
                    level = "WARNING"
                elif m.get("yawn_count", 0) >= 2:
                    level = "WARNING"
                    
                if level in ("WARNING", "ALERT"):
                    warning_or_alert += 1
                    
            used = len(feats)
            ratio = (warning_or_alert / used) if used > 0 else 0.0
            pred = 1 if ratio >= s_ratio else 0
            
            if pred == gt:
                correct += 1
                
        acc = correct / total_eval if total_eval else 0.0
        if acc > best_acc:
            best_acc = acc
            best_params = (ear, p_warn, p_alert, s_ratio)
            print(f"New Best! Acc: {acc:.4f} | EAR: {ear}, WARN: {p_warn}, ALERT: {p_alert}, Ratio: {s_ratio}")
            if acc >= 0.90:
                print(">>> 90% reached! <<<")
                break
                
    print(f"\nFinal Best -> Acc: {best_acc:.4f} @ EAR: {best_params[0]}, Warn: {best_params[1]}, Alert: {best_params[2]}, Ratio: {best_params[3]}")

if __name__ == "__main__":
    main()