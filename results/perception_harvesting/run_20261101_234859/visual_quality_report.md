# Unified Perception Pipeline — Visual Quality Report

Generated: 2026-11-01T18:34:12Z
YOLO model: `C:\Users\Yash Bhardwaj\Desktop\vlm\models\yolov8n-seg.pt`

## Per-Sequence Summary

| Sequence | Frames | Tracks | Detections | Accepted | Rejected | Masks | Ego-motion |
| --- | --- | --- | --- | --- | --- | --- | --- |
| tum_xyz | 414 | 15 | 1570 | 75 | 1495 | 75 | YES |
| tum_halfsphere | 511 | 19 | 1748 | 95 | 1653 | 95 | YES |
| tum_static | 359 | 10 | 1711 | 50 | 1661 | 50 | YES |
| openloris | 500 | 3 | 92 | 15 | 77 | 15 | YES |
| bonn_crowd | 464 | 20 | 1373 | 100 | 1273 | 100 | YES |
| bonn_box | 295 | 6 | 355 | 30 | 325 | 30 | YES |
| d455_01 | 1022 | 36 | 1187 | 180 | 1007 | 180 | YES |
| d455_02 | 1057 | 68 | 3422 | 340 | 3082 | 340 | YES |

## Rejection Reason Distribution

| Rejection Prefix | Count |
| --- | --- |
| NOT_SELECTED_FOR_VLM | 10554 |
| TOO_SMALL | 16 |
| BLURRY | 3 |

## Notes

- `identity_fallback` ego-motion: 3D tracking still functional, but world-frame displacement is camera-frame displacement (uncompensated).
- All detections are from actual image frames. No hardcoded coordinates were used.
- Segmentation masks are polygon contours from YOLO-seg inference.