# Dataset Inventory, Sensor Contracts, and Data Provenance

## 1. Multi-Dataset Factual Inventory

The research framework is evaluated across eight distinct RGB-D sequences representing diverse robotic kinematics, environmental textures, and dynamic complexities. Every sequence is physically present in `data/raw/` and verified by ground-truth trajectory alignment or sensory contracts:

| Sequence ID | Dataset Family | Local Path (`data/raw/`) | Physical Sensor | RGB-D Pairs | Reference Poses | Reference Source | Dynamic Characteristics |
| :--- | :--- | :--- | :--- | :---: | :---: | :--- | :--- |
| `tum_xyz` | TUM RGB-D | `rgbd_dataset_freiburg3_walking_xyz` | Microsoft Kinect v1 | 827 | 2,884 | Vicon MoCap (< 1 mm) | 2 walking persons moving office chairs |
| `tum_static` | TUM RGB-D | `rgbd_dataset_freiburg3_walking_static` | Microsoft Kinect v1 | 717 | 2,484 | Vicon MoCap (< 1 mm) | 2 seated persons (stationary error floor) |
| `tum_halfsphere` | TUM RGB-D | `rgbd_dataset_freiburg3_walking_halfsphere` | Microsoft Kinect v1 | 1,021 | 3,582 | Vicon MoCap (< 1 mm) | Moving camera arc with walking persons |
| `bonn_crowd` | Bonn RGB-D | `rgbd_bonn_crowd` | ASUS Xtion Pro Live | 927 | 933 | OptiTrack MoCap | Dense crowd walking across FOV |
| `bonn_box` | Bonn RGB-D | `rgbd_bonn_moving_obstructing_box` | ASUS Xtion Pro Live | 589 | 593 | OptiTrack MoCap | Person dragging large obstructing box |
| `openloris` | OpenLORIS-Scene | `OpenLORIS-Scene_corridor1-4` | Intel RealSense D435 | 1,500 | 2,515 | 3D LiDAR ICP (~10 mm) | Low-texture corridor with crossing pedestrians |
| `d455_01` | RealSense D455 | `my_desk_recording_01` | Intel RealSense D455 | 5,108 | null | None (No MoCap) | Handheld multi-session office traversal |
| `d455_02` | RealSense D455 | `my_desk_recording_02` | Intel RealSense D455 | 5,283 | null | None (No MoCap) | Relocated chairs, boxes, monitors |

---

## 2. Sensor Models & Intrinsic Calibrations

Each dataset adapter implements the standard `DatasetAdapter` contract, providing calibrated intrinsic parameters and metric depth scaling:

### 2.1 Microsoft Kinect v1 (TUM RGB-D Sequences)
- **Sequences**: `tum_xyz`, `tum_static`, `tum_halfsphere`
- **Focal Length**: $f_x = 535.4$, $f_y = 539.2$
- **Principal Point**: $c_x = 320.1$, $c_y = 247.6$
- **Resolution**: $640 	imes 480$
- **Depth Encoding**: Infrared structured light pattern ($Z_{\text{metric}} = \frac{\text{raw\_val}}{5000.0}$, depth scale $0.0002\text{ m/unit}$).

### 2.2 ASUS Xtion Pro Live (Bonn RGB-D Sequences)
- **Sequences**: `bonn_crowd`, `bonn_box`
- **Focal Length**: $f_x = 542.8$, $f_y = 542.6$
- **Principal Point**: $c_x = 315.6$, $c_y = 237.8$
- **Resolution**: $640 	imes 480$
- **Depth Encoding**: Structured light infrared ($Z_{\text{metric}} = \frac{\text{raw\_val}}{1000.0}$, depth scale $0.0010\text{ m/unit}$).

### 2.3 Intel RealSense D435 (OpenLORIS-Scene Corridor)
- **Sequence**: `openloris` (`corridor1-4`)
- **Focal Length**: $f_x = 611.5$, $f_y = 611.5$
- **Principal Point**: $c_x = 433.2$, $c_y = 249.5$
- **Resolution**: $848 	imes 480$
- **Depth Encoding**: Active stereo infrared ($Z_{\text{metric}} = \frac{\text{raw\_val}}{1000.0}$, depth scale $0.0010\text{ m/unit}$).

### 2.4 Intel RealSense D455 (Handheld Office Traversal)
- **Sequences**: `d455_01`, `d455_02`
- **Focal Length**: $f_x = 383.9$, $f_y = 383.5$
- **Principal Point**: $c_x = 320.3$, $c_y = 244.6$
- **Resolution**: $640 	imes 480$
- **Depth Encoding**: Wide-baseline active stereo infrared ($Z_{\text{metric}} = \frac{\text{raw\_val}}{1000.0}$, depth scale $0.0010\text{ m/unit}$).

---

## 3. Benchmark Populations & Track Partition Split

To ensure rigorous statistical isolation and prevent data leakage, observations and tracks across the 8 sequences are organized into canonical populations and partitions:

```
┌────────────────────────────────────────────────────────┐
│ Canonical Pipeline Population (885 Observations, 177 Tracks) │
│ ┌────────────────────────────────────────────────────┐ │
│ │ Development Split: tum_xyz + bonn_crowd (30 Tracks)│ │
│ │ (Controlled VLM prompt evaluation, evidence fusion)    │ │
│ └────────────────────────────────────────────────────┘ │
│ ┌────────────────────────────────────────────────────┐ │
│ │ Validation & Held-Out Partitions (147 Tracks)      │ │
│ │ (Generalization check & final evaluation)          │ │
│ └────────────────────────────────────────────────────┘ │
└────────────────────────────────────────────────────────┘
```

1. **Full Observation Population ($N = 885$)**:
   - 885 causal observation records extracted from YOLOv8-Seg tracking (`results/perception_harvesting/run_20261101_234859/candidate_observations.jsonl`).
   - Grouped into 177 unique tracks identified by composite key `(sequence_id, track_id)`.
   - 131 dynamic tracks, 46 static tracks.

2. **Evaluation Split Partitions (`results/track_annotation/evaluation_split.json`)**:
   - **Development**: 30 tracks (`tum_xyz` + `bonn_crowd`) for controlled prompt optimization and threshold calibration.
   - **Validation**: Independent held-out sequences for policy generalizability checks.
   - **Held-Out**: Unseen benchmark sequences for unbiased evaluation.
   - **Perception-Only**: RealSense D455 captures without optical MoCap ground truth (perception and tracking metrics only).

3. **Multi-Session Desk Entity Census (Historical Reference, $N = 52$)**:
   - 52 physical objects from the multi-session desk environment:
     - `IMMOVABLE`: 3 instances (fixed architectural structures, cabinetry).
     - `MOVABLE`: 45 instances (chairs, monitors, laptops, keyboards, backpacks).
     - `MOBILE`: 4 instances (robotic platforms, drones).

### Materialized Asset Directory (`benchmarks/assets/multiview_samples/`)
Multi-view visual assets are frozen in canonical storage:
* **Instances**: 42 physical object instances with verified multi-view crops.
* **Total Image Assets**: 187 image files across 42 instance subdirectories (`inst_backpack_001` through `inst_whiteboard_001`).
* **Metadata**: Every instance folder contains `sample_meta.json` recording camera trajectory timestamps, 3D bounding box coordinates, and viewpoint angular diversity (minimum angular spread $\ge 12^\circ$).

---

## 4. Data Contracts & Serialization Schemas

### 4.1 Per-Frame Observation Record (`object_observations.jsonl`)
Every detected 2D instance projected into 3D metric space emits an `ObservationRecord`:

```json
{
  "obs_id": "seq1_frame_0142_det_03",
  "source_session": "semantic_mapping_my_desk_recording_01",
  "frame_index": 142,
  "timestamp": 1791788670.0385,
  "local_track_id": "chair_02",
  "global_entity_id": "inst_chair_002",
  "raw_class_name": "chair",
  "yoloe_class_name": "chair",
  "detector_confidence": 0.912,
  "camera_pose_twc": [1.205, -0.450, 0.892, 0.012, -0.045, 0.852, 0.521],
  "bbox_2d_xyxy": [184, 112, 420, 395],
  "centroid_3d_camera": [0.152, 0.084, 1.450],
  "centroid_3d_world": [1.412, -0.320, 0.745],
  "inlier_depth_ratio": 0.884,
  "depth_variance_m2": 0.0034
}
```

### 4.2 Multi-Frame Tracked Entity Record (`tracks_summary.json`)
Causal temporal filtering aggregates frame observations into persistent tracked entities:

```json
{
  "entity_id": "chair_01",
  "semantic_class": "chair",
  "lifespan_frames": 184,
  "observation_count": 142,
  "first_timestamp": 1791788665.139,
  "last_timestamp": 1791788672.481,
  "initial_centroid_world": [0.852, -1.240, 0.650],
  "current_centroid_world": [1.468, -1.215, 0.648],
  "world_displacement_m": 0.616,
  "displacement_uncertainty_m": 0.067,
  "observed_motion_state": "DISPLACED",
  "camera_ego_motion_during_track_m": 2.140,
  "inlier_point_ratio": 0.891
}
```

---

## 5. Cryptographic Immutability & Anti-Leakage Verification

To enforce scientific integrity:
1. **Raw Sequence Immutability**: All raw sensor data directories (`data/raw/*`) remain read-only. Scripts attempting in-place writes to raw datasets are rejected by automated tests (`tests/test_dataset_contracts.py`).
2. **Two-File Benchmark Isolation**:
   - `paper_benchmark_inputs_expanded.json` contains strictly sensory evidence, 2D/3D metrics, and image asset paths with **zero ground-truth labels**.
   - `paper_benchmark_ground_truth_expanded.json` resides in a cryptographically isolated schema consumed exclusively by final scoring routines.
3. **Leakage Prevention Regression Suite**:
   - `tests/test_benchmark_ontology_and_leakage.py` asserts that no inference prompt or feature adapter can inspect ground-truth labels.
   - `tests/test_data_traceability.py` verifies SHA-256 digests across all input manifests and generated artifacts.



---

## 7. Cryptographic Provenance Reconciliations

- **OpenLORIS `corridor1-4` Ground Truth**:
  * Path: `data/raw/OpenLORIS-Scene_corridor1-4/groundtruth.txt`
  * Verified SHA-256: `0c6125b1ae6b8753d5c0e05404c417af22d7e47b999d4b6178c9bb104291c683`
  * Role: Authoritative ground-truth reference for corridor tracking evaluations.

- **TUM `fr3_walking_halfsphere` Condition S3**:
  * Classification: `GEOMETRY_PLUS_DETECTOR_SEMANTIC` (YOLO person detector intersected with moving geometric tracks; not a live VLM call).
