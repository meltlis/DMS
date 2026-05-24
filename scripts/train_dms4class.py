"""
Fine-tune YOLOv8n → 4类 DMS 检测模型

类别：face / phone / cigarette / seatbelt
起点：yolov8n.pt（COCO 80类预训练权重，含 cell phone 类，迁移效果好）
输出：runs/detect/dms4class/weights/best.pt
"""
from __future__ import annotations

import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs",  type=int,   default=60)
    parser.add_argument("--batch",   type=int,   default=16)
    parser.add_argument("--imgsz",   type=int,   default=640)
    parser.add_argument("--device",  type=str,   default="cuda")
    parser.add_argument("--data",    type=str,
                        default=str(ROOT / "datasets" / "dms4class" / "data.yaml"))
    parser.add_argument("--weights", type=str,   default="yolov8n.pt",
                        help="起点权重，默认 COCO 预训练 yolov8n.pt")
    args = parser.parse_args()

    data_path = Path(args.data)
    if not data_path.exists():
        print(f"[错误] 找不到数据集配置: {data_path}")
        print("请先运行：python scripts/prepare_dms_dataset.py")
        return

    from ultralytics import YOLO

    print("=" * 60)
    print("DMS 4 类检测模型训练")
    print(f"  起点权重 : {args.weights}")
    print(f"  数据集   : {data_path}")
    print(f"  轮次     : {args.epochs}")
    print(f"  批大小   : {args.batch}")
    print(f"  图片尺寸 : {args.imgsz}")
    print(f"  设备     : {args.device}")
    print("=" * 60)

    model = YOLO(args.weights)

    results = model.train(
        data=str(data_path),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        project=str(ROOT / "runs" / "detect"),
        name="dms4class",
        patience=15,          # 15 轮无提升提前停止
        save=True,
        plots=True,
        # 学习率
        lr0=0.01,
        lrf=0.005,
        warmup_epochs=3,
        # 数据增强（车内场景加强颜色/亮度变换）
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.5,
        degrees=5.0,          # 轻微旋转
        translate=0.1,
        scale=0.4,
        mosaic=1.0,
        mixup=0.1,
        copy_paste=0.0,
        # 类别权重均衡（cigarette 样本少，提高权重）
        cls=0.5,
    )

    best = ROOT / "runs" / "detect" / "dms4class" / "weights" / "best.pt"
    metrics = results.results_dict

    print("\n" + "=" * 60)
    print("训练完成！")
    print(f"  最佳权重 : {best}")
    print(f"  mAP50    : {metrics.get('metrics/mAP50(B)', 'N/A'):.3f}")
    print(f"  mAP50-95 : {metrics.get('metrics/mAP50-95(B)', 'N/A'):.3f}")
    print("\n验证命令：")
    print(f"  python scripts/eval_dms4class.py")
    print("=" * 60)


if __name__ == "__main__":
    main()
