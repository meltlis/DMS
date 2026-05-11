"""Analyze what thresholds would improve YawDD accuracy to 90%+."""
import csv, pathlib

data = []
with open("reports/yawdd_eval_full.csv", encoding="utf-8") as f:
    for row in csv.DictReader(f):
        data.append(row)

normal  = [r for r in data if r["label"] == "Normal"]
yawning = [r for r in data if r["label"] == "Yawning"]

normal_ratios  = sorted(float(r["yawn_ratio"]) for r in normal)
yawning_ratios = sorted(float(r["yawn_ratio"]) for r in yawning)

print("Normal video yawn_ratio distribution:")
buckets = [0.0, 0.01, 0.02, 0.05, 0.10, 0.20, 1.01]
for lo, hi in zip(buckets, buckets[1:]):
    n = sum(1 for r in normal_ratios if lo <= r < hi)
    print(f"  [{lo:.2f}, {hi:.2f}): {n}")

print()
print("Yawning video yawn_ratio distribution:")
for lo, hi in zip(buckets, buckets[1:]):
    n = sum(1 for r in yawning_ratios if lo <= r < hi)
    print(f"  [{lo:.2f}, {hi:.2f}): {n}")

print()
print("Threshold sweep (yawn_ratio_threshold):")
print(f"  {'Threshold':>10}  {'Acc':>6}  {'Prec':>6}  {'Recall':>7}  {'TP':>4}  {'FP':>4}  {'TN':>4}  {'FN':>4}")
for thresh in [0.01, 0.02, 0.03, 0.05, 0.08, 0.10, 0.15, 0.20]:
    tp = sum(1 for r in yawning if float(r["yawn_ratio"]) >= thresh)
    fn = len(yawning) - tp
    fp = sum(1 for r in normal  if float(r["yawn_ratio"]) >= thresh)
    tn = len(normal)  - fp
    acc  = (tp + tn) / (len(yawning) + len(normal))
    prec = tp / (tp + fp) if (tp + fp) else 1.0
    rec  = tp / (tp + fn) if (tp + fn) else 0.0
    print(f"  {thresh:>10.2f}  {acc:>6.1%}  {prec:>6.1%}  {rec:>7.1%}  {tp:>4}  {fp:>4}  {tn:>4}  {fn:>4}")

# Also: how many FN have yawn_ratio exactly 0?
zero = sum(1 for r in yawning if float(r["yawn_ratio"]) == 0.0)
print(f"\nYawning videos with yawn_ratio=0.000 (MAR never triggered): {zero}")
print("These need MAR threshold lowering, not decision threshold tweaking.")

# Check warning_alert_ratio for FN cases
fn_cases = [r for r in yawning if r["predicted_yawning"] == "0"]
wa_ratios = [float(r["warning_alert_ratio"]) for r in fn_cases]
print(f"\nFN cases: warning_alert_ratio mean={sum(wa_ratios)/len(wa_ratios):.3f}  max={max(wa_ratios):.3f}")
print(f"  FN with warning_alert_ratio>0: {sum(1 for r in wa_ratios if r > 0)}")
