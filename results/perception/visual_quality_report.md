# Unified Perception Pipeline — Visual Quality Report

Generated: 2026-11-10T11:48:26Z
YOLO model: `C:\Users\Yash Bhardwaj\Desktop\vlm\models\yolov8s-seg.pt`

## Per-Sequence Summary

| Sequence | Frames | Tracks | Detections | Accepted | Rejected | Masks | Ego-motion |
| --- | --- | --- | --- | --- | --- | --- | --- |
| tum_xyz | 414 | 18 | 1809 | 90 | 1719 | 90 | YES |
| tum_halfsphere | 511 | 22 | 2068 | 110 | 1958 | 110 | YES |
| tum_static | 359 | 13 | 1895 | 65 | 1830 | 65 | YES |
| openloris | 500 | 6 | 139 | 30 | 109 | 30 | YES |
| bonn_crowd | 464 | 20 | 1417 | 100 | 1317 | 100 | YES |
| bonn_box | 295 | 11 | 373 | 55 | 318 | 55 | YES |
| d455_01 | 997 | 39 | 1847 | 189 | 1658 | 189 | YES |
| d455_02 | 1057 | 87 | 4162 | 424 | 3738 | 424 | YES |

## Perception Quality Audit

| Sequence | Detections | Duplicates (IoU>0.7) | Small Objs (<1024px) | Mask Frag Rate | Transient Rate | Track Dropout Rate |
| --- | --- | --- | --- | --- | --- | --- |
| tum_xyz | 1809 | 0 (0.0%) | 0 (0.0%) | 12.9% | 0.0% | 71.0% |
| tum_halfsphere | 2068 | 0 (0.0%) | 3 (0.1%) | 7.3% | 0.0% | 77.6% |
| tum_static | 1895 | 0 (0.0%) | 0 (0.0%) | 16.9% | 0.0% | 56.4% |
| openloris | 139 | 0 (0.0%) | 2 (1.4%) | 7.9% | 0.0% | 74.3% |
| bonn_crowd | 1417 | 1 (0.1%) | 0 (0.0%) | 0.2% | 0.0% | 76.0% |
| bonn_box | 373 | 0 (0.0%) | 0 (0.0%) | 0.0% | 0.0% | 60.1% |
| d455_01 | 1847 | 2 (0.1%) | 49 (2.6%) | 5.4% | 0.0% | 95.8% |
| d455_02 | 4162 | 0 (0.0%) | 74 (1.8%) | 4.3% | 0.0% | 91.3% |

### Aggregate Audit Summary

- **Total Raw Detections:** 13710
- **Duplicate Detection Rate (IoU >= 0.70):** 0.02% (3 instances)
- **Small Object Recall (< 1024 px²):** 0.93% (mean conf: 0.55)
- **Mask Fragmentation Rate:** 7.29% (1000 of 13710 masks)
- **Transient / Spurious Detection Rate:** 0.00%
- **Track Dropout / Missed Detection Rate:** 88.52%

## Rejection Reason Distribution

| Rejection Prefix | Count |
| --- | --- |
| NOT_SELECTED_FOR_VLM | 12584 |
| TOO_SMALL | 59 |
| BLURRY | 4 |

## Notes

- `identity_fallback` ego-motion: 3D tracking still functional, but world-frame displacement is camera-frame displacement (uncompensated).
- All detections are from actual image frames. No hardcoded coordinates were used.
- Segmentation masks are polygon contours from YOLO-seg inference.