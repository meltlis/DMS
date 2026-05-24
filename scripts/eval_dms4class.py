"""验证 DMS 4 类检测模型精度，并可视化预测结果样本。"""
from __future__ import annotations

import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", type=str,
                        default=str(ROOT / "runs" / "detect" / "dms4class" / "weights" / "best.pt"))
    parser.add_argument("--data", type=str,
                        default=str(ROOT / "datasets" / "dms4class" / "data.yaml"))
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    from ultralytics import YOLO

    if not Path(args.weights).exists():
        print(f"[错误] 找不到权重文件: {args.weights}")
        print("请先训练：python scripts/train_dms4class.py")
        return

    model = YOLO(args.weights)
    metrics = model.val(
        data=args.data,
        imgsz=args.imgsz,
        device=args.device,
        plots=True,
    )

    print("\n每类 AP50：")
    names = ["face", "phone", "cigarette", "seatbelt"]
    for i, name in enumerate(names):
        ap = metrics.ap_class_index
        if i < len(metrics.box.ap50):
            print(f"  {name:12s}: {metrics.box.ap50[i]:.3f}")

    print(f"\n  mAP50    : {metrics.box.map50:.3f}")
    print(f"  mAP50-95 : {metrics.box.map:.3f}")


if __name__ == "__main__":
    main()
