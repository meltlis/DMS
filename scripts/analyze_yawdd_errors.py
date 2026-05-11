import csv, pathlib, collections

data = []
with open("reports/yawdd_eval_full.csv", encoding="utf-8") as f:
    for row in csv.DictReader(f):
        data.append(row)

yawning = [r for r in data if r["label"] == "Yawning"]
fn = [r for r in yawning if r["predicted_yawning"] == "0"]
tp = [r for r in yawning if r["predicted_yawning"] == "1"]

print(f"Yawning videos: {len(yawning)}  TP={len(tp)}  FN={len(fn)}")
print()

# FN breakdown by glasses type
def glasses_type(name):
    n = name.lower()
    if "sunglasses" in n: return "SunGlasses"
    if "noglasses" in n:  return "NoGlasses"
    if "glasses" in n:    return "Glasses"
    return "Unknown"

fn_glasses = collections.Counter(glasses_type(pathlib.Path(r["video"]).name) for r in fn)
tp_glasses = collections.Counter(glasses_type(pathlib.Path(r["video"]).name) for r in tp)
all_glasses = collections.Counter(glasses_type(pathlib.Path(r["video"]).name) for r in yawning)

print("Detection rate by glasses type:")
for g in ["NoGlasses", "Glasses", "SunGlasses"]:
    total = all_glasses[g]
    detected = tp_glasses[g]
    missed = fn_glasses[g]
    rate = detected/total if total else 0
    print(f"  {g:12s}: {detected}/{total} detected ({rate:.0%} recall)  FN={missed}")

print()
# FN yawn_ratio distribution
fn_ratios = [float(r["yawn_ratio"]) for r in fn]
print(f"FN yawn_ratio stats: mean={sum(fn_ratios)/len(fn_ratios):.4f}  max={max(fn_ratios):.4f}")
print(f"  ratio=0.000: {sum(1 for r in fn_ratios if r == 0.0)}")
print(f"  ratio 0.001-0.049: {sum(1 for r in fn_ratios if 0 < r < 0.05)}")

print()
# TP yawn_ratio stats
tp_ratios = [float(r["yawn_ratio"]) for r in tp]
print(f"TP yawn_ratio: mean={sum(tp_ratios)/len(tp_ratios):.3f}  min={min(tp_ratios):.3f}")

print()
# Subject-level: which subjects consistently fail?
subject_fn = collections.Counter()
subject_total = collections.Counter()
for r in yawning:
    name = pathlib.Path(r["video"]).name
    subj = name.split("-")[0]
    subject_total[subj] += 1
    if r["predicted_yawning"] == "0":
        subject_fn[subj] += 1

print("Subjects with all Yawning videos missed (consistent FN):")
all_miss = [(s, subject_fn[s], subject_total[s]) for s in subject_fn if subject_fn[s] == subject_total[s]]
for s, fn_c, tot in sorted(all_miss, key=lambda x: -x[1]):
    print(f"  Subject {s:3s}: missed {fn_c}/{tot}")

print(f"\nTotal subjects with ≥1 FN: {len(subject_fn)}")
print(f"Total subjects fully missed: {len(all_miss)}")
