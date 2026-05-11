# Evaluation Starter Kit

This folder provides a minimum structure for evaluating the current DMS program.

## Layout

- `miniset/train/images`, `miniset/train/labels`
- `miniset/val/images`, `miniset/val/labels`
- `miniset/test/images`, `miniset/test/labels`
- `templates/annotation_guide.md`
- `templates/label_yolo_template.txt`
- `templates/data.yaml`

## Quick batch run

```powershell
cd dms-system
uv run python scripts/batch_eval.py --max-frames 300 --limit-per-dataset 3
```

Output CSV is written to `reports/batch_summary.csv`.
