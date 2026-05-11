# Annotation Guide (4 classes)

Target classes must follow this exact mapping:

- `0 face`
- `1 phone`
- `2 cigarette`
- `3 seatbelt`

## Label format

Use YOLO txt format per image (same stem name):

`class_id x_center y_center width height`

All coordinates are normalized to `[0,1]` by image width/height.

## Rules

1. One object per line.
2. Skip uncertain objects instead of forcing noisy boxes.
3. Keep class IDs continuous and fixed as above.
4. For seatbelt, annotate only visible belt region near driver torso.
5. Do not mix fatigue-state labels into this 4-class detection set.

## Split suggestion

- Train: 80%
- Val: 10%
- Test: 10%
