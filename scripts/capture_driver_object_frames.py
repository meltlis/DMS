from __future__ import annotations

import argparse
import time
from pathlib import Path

import cv2


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description="Capture webcam frames for driver-object annotation.")
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--label", default="phone", help="phone | bottle | cup | cigarette | negatives")
    parser.add_argument("--interval", type=float, default=0.35, help="Seconds between saved frames.")
    parser.add_argument("--max-frames", type=int, default=250)
    parser.add_argument(
        "--out",
        default=str(ROOT / "datasets" / "driver_objects_raw"),
        help="Raw image output folder. Annotate these images before training.",
    )
    args = parser.parse_args()

    out_dir = Path(args.out) / args.label / time.strftime("%Y%m%d_%H%M%S")
    out_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open camera index {args.camera}")

    print(f"Saving frames to: {out_dir}")
    print("Press q to stop.")

    saved = 0
    last_save = 0.0
    try:
        while saved < args.max_frames:
            ok, frame = cap.read()
            if not ok:
                break

            now = time.time()
            if now - last_save >= args.interval:
                path = out_dir / f"{args.label}_{saved:05d}.jpg"
                cv2.imwrite(str(path), frame)
                saved += 1
                last_save = now

            cv2.putText(
                frame,
                f"{args.label}: {saved}/{args.max_frames}",
                (12, 28),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2,
            )
            cv2.imshow("capture driver object frames", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()

    print(f"Captured {saved} frames.")
    print("Next: annotate bounding boxes and export YOLO format to datasets/driver_objects/.")


if __name__ == "__main__":
    main()
