import csv, pathlib

data = []
with open("reports/accuracy_eval.csv", encoding="utf-8") as f:
    for row in csv.DictReader(f):
        data.append(row)

drozy = [r for r in data if r["dataset"] == "DROZY" and r["gt_sleepy"] != ""]
tp = [r for r in drozy if r["predicted_sleepy"] == "1" and r["gt_sleepy"] == "1"]
fp = [r for r in drozy if r["predicted_sleepy"] == "1" and r["gt_sleepy"] == "0"]
tn = [r for r in drozy if r["predicted_sleepy"] == "0" and r["gt_sleepy"] == "0"]
fn = [r for r in drozy if r["predicted_sleepy"] == "0" and r["gt_sleepy"] == "1"]

gt_sleepy = sum(1 for r in drozy if r["gt_sleepy"] == "1")
print(f"Total: {len(drozy)}  GT-sleepy: {gt_sleepy}  GT-awake: {len(drozy)-gt_sleepy}")
print(f"TP: {len(tp)}  FP: {len(fp)}  TN: {len(tn)}  FN: {len(fn)}")
prec = len(tp)/(len(tp)+len(fp)) if (len(tp)+len(fp)) else 0
rec  = len(tp)/(len(tp)+len(fn)) if (len(tp)+len(fn)) else 0
print(f"Precision: {prec:.2f}  Recall: {rec:.2f}")

ratios_awake  = [float(r["warning_or_alert_ratio"]) for r in drozy if r["gt_sleepy"] == "0"]
ratios_sleepy = [float(r["warning_or_alert_ratio"]) for r in drozy if r["gt_sleepy"] == "1"]
print(f"\nAwake  ratio mean={sum(ratios_awake)/len(ratios_awake):.3f}  max={max(ratios_awake):.3f}")
print(f"Sleepy ratio mean={sum(ratios_sleepy)/len(ratios_sleepy):.3f}  max={max(ratios_sleepy):.3f}")

print("\nFP (pred=sleepy, actually awake):")
for r in sorted(fp, key=lambda x: -float(x["warning_or_alert_ratio"])):
    print(f"  {pathlib.Path(r['video']).name}  ratio={r['warning_or_alert_ratio']}  kss={r['kss']}")

print("\nFN (pred=awake, actually sleepy):")
for r in sorted(fn, key=lambda x: float(x["warning_or_alert_ratio"])):
    print(f"  {pathlib.Path(r['video']).name}  ratio={r['warning_or_alert_ratio']}  kss={r['kss']}")
