# VLM Data Package (`vlm/data/`)

This directory contains the self-contained raw sequences, trajectory logs, sensor calibrations, ground-truth annotations, and observation caches required to run, test, and reproduce the Estimator-Aware VLM research pipeline.

## 1. Directory Structure

```text
data/
├── README.md                           # Directory inventory and sensor documentation
├── raw/                                # 8 canonical benchmark RGB-D sequences
│   ├── rgbd_dataset_freiburg3_walking_xyz/          # TUM dynamic benchmark (827 pairs, 2,884 GT poses)
│   ├── rgbd_dataset_freiburg3_walking_static/       # TUM static control (717 pairs, 2,484 GT poses)
│   ├── rgbd_dataset_freiburg3_walking_halfsphere/   # TUM moving-camera hemisphere arc (1,021 pairs, 3,582 GT poses)
│   ├── rgbd_bonn_crowd/                             # Bonn dense dynamic crowd (927 pairs, 933 GT poses)
│   ├── rgbd_bonn_moving_obstructing_box/            # Bonn human dragging box (589 pairs, 593 GT poses)
│   ├── OpenLORIS-Scene_corridor1-4/                 # OpenLORIS corridor (1,500 pairs, 2,515 GT poses)
│   ├── my_desk_recording_01/                        # RealSense D455 Session 1 (5,108 pairs, handheld)
│   └── my_desk_recording_02/                        # RealSense D455 Session 2 (5,283 pairs, handheld)
├── annotations/                        # Ground-truth entity annotations and mobility labels
│   └── classroom_evaluation_ground_truth.json
├── fixtures/                           # Deterministic test fixtures for unit and regression tests
│   └── geometry_samples.json
├── observations/                       # Unified multi-sequence observation cache
│   ├── object_observations.jsonl       # Multi-frame object detection instances
│   ├── object_tracks.jsonl             # Confirmed spatial-temporal object tracks
│   └── multiview_association_montage.jpg
├── processed/                          # Pre-processed sequence caches and metadata
│   └── tum_fr1_desk/                   # Static desk control observations (613 frames)
├── semantic_mapping_fused_desk_run/    # Fused multi-session reference map
├── semantic_mapping_my_desk_recording_01/  # Session 1 MapPoints.ply & visual features
└── semantic_mapping_my_desk_recording_02/  # Session 2 MapPoints.ply & visual features
```

## 2. Active Sequence Inventory

| Sequence ID | Dataset Family | Sensor | Frames | Reference Ground Truth | Dynamic Characteristics |
| :--- | :--- | :--- | :---: | :--- | :--- |
| `tum_xyz` | TUM RGB-D | Microsoft Kinect v1 | 827 | Vicon MoCap (< 1 mm) | 2 walking persons moving office chairs |
| `tum_static` | TUM RGB-D | Microsoft Kinect v1 | 717 | Vicon MoCap (< 1 mm) | 2 seated persons (stationary error floor) |
| `tum_halfsphere` | TUM RGB-D | Microsoft Kinect v1 | 1,021 | Vicon MoCap (< 1 mm) | Moving camera arc with walking persons |
| `bonn_crowd` | Bonn RGB-D | ASUS Xtion Pro Live | 927 | OptiTrack MoCap | Dense crowd walking across FOV |
| `bonn_box` | Bonn RGB-D | ASUS Xtion Pro Live | 589 | OptiTrack MoCap | Person dragging large obstructing box |
| `openloris` | OpenLORIS-Scene | Intel RealSense D435 | 1,500 | 3D LiDAR ICP (~10 mm) | Low-texture corridor with crossing pedestrians |
| `d455_01` | RealSense D455 | Intel RealSense D455 | 5,108 | None (No MoCap) | Handheld multi-session office traversal |
| `d455_02` | RealSense D455 | Intel RealSense D455 | 5,283 | None (No MoCap) | Relocated chairs, boxes, monitors |

## 3. Data Integrity & Governance

1. **Raw Immutability**: All files in `data/raw/` are strictly read-only. Automated pipeline tests enforce zero modification of input sequence data.
2. **Zero Fabrication**: Trajectory reference metrics for uninstrumented sequences (`my_desk_recording_01` and `02`) are recorded strictly as `null` rather than substituted with heuristic trajectory approximations.
