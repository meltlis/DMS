from __future__ import annotations

import argparse
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description="Fine-tune YOLO for driver-hand objects.")
    parser.add_argument(
        "--data",
        default=str(ROOT / "datasets" / "driver_objects" / "data.yaml"),
        help="YOLO data.yaml with classes such as phone, bottle, cup, cigarette.",
    )
    parser.add_argument("--weights", default=str(ROOT / "weights" / "yolo11s.pt"))
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch", type=int, default=12)
    parser.add_argument("--imgsz", type=int, default=960)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--name", default="driver_objects")
    args = parser.parse_args()

    data_path = Path(args.data)
    if not data_path.exists():
        print(f"[ERROR] Missing dataset config: {data_path}")
        print("Expected YOLO layout:")
        print("  datasets/driver_objects/data.yaml")
        print("  datasets/driver_objects/train/images, train/labels")
        print("  datasets/driver_objects/val/images,   val/labels")
        print("Recommended labels: phone, bottle, cup, cigarette")
        return

    from ultralytics import YOLO

    model = YOLO(args.weights)
    results = model.train(
        data=str(data_path),
        epochs=args.epochs,
        batch=args.batch,
        imgsz=args.imgsz,
        device=args.device,
        project=str(ROOT / "runs" / "detect"),
        name=args.name,
        patience=20,
        save=True,
        plots=True,
        close_mosaic=15,
        hsv_h=0.015,
        hsv_s=0.60,
        hsv_v=0.45,
        degrees=4.0,
        translate=0.08,
        scale=0.35,
        mosaic=0.80,
        mixup=0.05,
    )

    best = ROOT / "runs" / "detect" / args.name / "weights" / "best.pt"
    print("\nTraining finished")
    print(f"Best weights: {best}")
    print(f"Metrics: {results.results_dict}")
    print("\nTo use it, set in configs/thresholds.yaml:")
    print(f"  aux_model_path: {best.relative_to(ROOT).as_posix()}")


if __name__ == "__main__":
    main()
