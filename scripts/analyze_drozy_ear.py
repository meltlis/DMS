"""
分析 DROZY 近红外视频中的 EAR 分布，对比 KSS 困倦等级。

输出：
  - 各 KSS 等级的 EAR 均值/中位数/分布图
  - MediaPipe 在近红外下的人脸检测率
  - 与 RGB 阈值 0.16 的对比
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import cv2
import numpy as np
import json

DROZY_ROOT = ROOT.parents[0] / "dataset" / "DROZY" / "DROZY"
VIDEO_DIR  = DROZY_ROOT / "videos_i8"
KSS_FILE   = DROZY_ROOT / "KSS.txt"

STRIDE = 15  # 每隔15帧采样一次（约0.5s间隔）
MAX_FRAMES_PER_VIDEO = 300  # 最多采300帧/视频


def read_kss() -> dict[str, int]:
    """读取 KSS.txt，返回 {subject_session: kss_score}。
    KSS.txt 每行3个分数对应 subject 的 session1/2/3。
    行号从1开始对应 subject id。
    """
    kss_map: dict[str, int] = {}
    lines = KSS_FILE.read_text().strip().splitlines()
    for subj_idx, line in enumerate(lines, start=1):
        scores = line.split()
        for sess_idx, score in enumerate(scores, start=1):
            key = f"{subj_idx}-{sess_idx}"
            kss_map[key] = int(score)
    return kss_map



def _get_face_detector():
    """使用 OpenCV Haar cascade 做人脸检测（在近红外下也有效）。"""
    cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    return cv2.CascadeClassifier(cascade_path)


def process_video(video_path: Path, face_analyzer) -> dict:
    """处理单个视频，返回 EAR 采样列表和检测率。"""
    face_det = _get_face_detector()

    cap = cv2.VideoCapture(str(video_path))
    ears: list[float] = []
    total_frames = 0
    detected_frames = 0
    n = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        n += 1
        if n % STRIDE != 0:
            continue
        total_frames += 1
        if total_frames > MAX_FRAMES_PER_VIDEO:
            break

        # 近红外图增强对比度
        if len(frame.shape) == 3:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            gray = frame
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)

        # Haar cascade 人脸检测
        faces = face_det.detectMultiScale(
            enhanced, scaleFactor=1.1, minNeighbors=4, minSize=(60, 60)
        )

        if len(faces) == 0:
            # 尝试用整帧作为人脸 ROI（DROZY 视频人脸较大）
            roi_bgr = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)
        else:
            x, y, w_f, h_f = max(faces, key=lambda f: f[2] * f[3])
            roi_bgr = cv2.cvtColor(enhanced[y:y+h_f, x:x+w_f], cv2.COLOR_GRAY2BGR)

        result = face_analyzer.analyze(roi_bgr)
        # landmarks_468 is None in synthetic fallback, real ndarray only on true detection
        if result.get("landmarks_468") is not None:
            detected_frames += 1
            el = result.get("ear_left")
            er = result.get("ear_right")
            if el is not None and er is not None:
                ears.append((el + er) / 2.0)

    cap.release()

    detection_rate = detected_frames / total_frames if total_frames > 0 else 0.0
    return {
        "ears": ears,
        "total_frames": total_frames,
        "detected_frames": detected_frames,
        "detection_rate": detection_rate,
    }


def main():
    if not VIDEO_DIR.exists():
        print(f"[错误] 找不到视频目录: {VIDEO_DIR}")
        return

    kss_map = read_kss()
    videos = sorted(VIDEO_DIR.glob("*.mp4"))
    print(f"找到 {len(videos)} 个近红外视频")
    print(f"KSS 标签数: {len(kss_map)}")
    print(f"采样步长: 每 {STRIDE} 帧取1帧，最多 {MAX_FRAMES_PER_VIDEO} 帧/视频\n")

    from src.features.face_analyzer import FaceAnalyzer
    face_analyzer = FaceAnalyzer(ear_threshold=0.16, mar_threshold=0.50)

    # 按 KSS 分组
    kss_ears: dict[int, list[float]] = {}
    results = []

    for vp in videos:
        stem = vp.stem  # e.g. "1-2"
        kss = kss_map.get(stem)
        if kss is None:
            print(f"  [跳过] {stem} 没有 KSS 标签")
            continue

        print(f"  处理 {stem}.mp4  KSS={kss} ...", end=" ", flush=True)
        r = process_video(vp, face_analyzer)
        print(f"检测率={r['detection_rate']:.1%}  EAR样本={len(r['ears'])}"
              + (f"  均值={np.mean(r['ears']):.3f}" if r['ears'] else "  无EAR数据"))

        if r['ears']:
            kss_ears.setdefault(kss, []).extend(r['ears'])
        results.append({"video": stem, "kss": kss, **r})

    print("\n" + "=" * 60)
    print("EAR 分布按 KSS 等级汇总")
    print("=" * 60)
    print(f"{'KSS':>4}  {'状态':8}  {'样本数':>6}  {'均值':>6}  {'中位数':>6}  {'<0.16占比':>9}  {'<0.20占比':>9}")
    print("-" * 60)

    kss_labels = {
        1: "极清醒", 2: "很清醒", 3: "清醒", 4: "略清醒", 5: "一般",
        6: "有困意", 7: "困", 8: "很困", 9: "极困"
    }

    all_ears_by_alertness: dict[str, list[float]] = {"清醒(1-5)": [], "困倦(6-9)": []}

    for kss in sorted(kss_ears.keys()):
        ears = kss_ears[kss]
        arr = np.array(ears)
        label = kss_labels.get(kss, "?")
        below16 = (arr < 0.16).mean()
        below20 = (arr < 0.20).mean()
        print(f"  {kss:2d}  {label:8s}  {len(arr):6d}  {arr.mean():6.3f}  {np.median(arr):6.3f}  {below16:9.1%}  {below20:9.1%}")
        if kss <= 5:
            all_ears_by_alertness["清醒(1-5)"].extend(ears)
        else:
            all_ears_by_alertness["困倦(6-9)"].extend(ears)

    print("\n" + "=" * 60)
    print("清醒 vs 困倦 EAR 对比")
    print("=" * 60)
    for label, ears in all_ears_by_alertness.items():
        if not ears:
            print(f"  {label}: 无数据")
            continue
        arr = np.array(ears)
        print(f"  {label}: n={len(arr)}  均值={arr.mean():.3f}  中位={np.median(arr):.3f}  "
              f"std={arr.std():.3f}  <0.16={( arr<0.16).mean():.1%}  <0.20={(arr<0.20).mean():.1%}")

    # 给出阈值建议
    alert_ears = np.array(all_ears_by_alertness["清醒(1-5)"])
    drowsy_ears = np.array(all_ears_by_alertness["困倦(6-9)"])
    if len(alert_ears) > 0 and len(drowsy_ears) > 0:
        # 找最优分割点
        best_thresh, best_acc = 0.0, 0.0
        for t in np.arange(0.10, 0.35, 0.01):
            tp = (drowsy_ears < t).sum()
            tn = (alert_ears >= t).sum()
            acc = (tp + tn) / (len(drowsy_ears) + len(alert_ears))
            if acc > best_acc:
                best_acc, best_thresh = acc, t

        print(f"\n  建议 EAR 阈值（近红外）: {best_thresh:.2f}  (分类准确率 {best_acc:.1%})")
        print(f"  当前 RGB 阈值: 0.16")
        if best_thresh > 0.16:
            print(f"  → 红外下眼睛看起来更'睁开'，阈值应上调 +{best_thresh-0.16:.2f}")
        else:
            print(f"  → 红外下眼睛看起来更'闭合'，阈值应下调 {best_thresh-0.16:.2f}")

    # 总体检测率
    all_det = [r["detection_rate"] for r in results]
    print(f"\n  MediaPipe 平均检测率（近红外）: {np.mean(all_det):.1%}")
    print(f"  最低检测率: {np.min(all_det):.1%}   最高: {np.max(all_det):.1%}")

    # 保存结果
    out = ROOT / "runs" / "drozy_ear_analysis.json"
    out.parent.mkdir(exist_ok=True)
    save_data = {
        "kss_summary": {
            str(k): {
                "n": len(v),
                "mean": float(np.mean(v)),
                "median": float(np.median(v)),
                "std": float(np.std(v)),
                "below_0.16": float((np.array(v) < 0.16).mean()),
                "below_0.20": float((np.array(v) < 0.20).mean()),
            }
            for k, v in kss_ears.items()
        },
        "suggested_threshold": float(best_thresh) if len(alert_ears) > 0 and len(drowsy_ears) > 0 else None,
        "detection_rate_mean": float(np.mean(all_det)),
    }
    out.write_text(json.dumps(save_data, indent=2, ensure_ascii=False))
    print(f"\n  结果已保存: {out}")


if __name__ == "__main__":
    main()
