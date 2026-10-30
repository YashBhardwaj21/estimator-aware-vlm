# Dataset Inventory, Sensor Contracts, and Data Provenance

## 1. Multi-Dataset Factual Inventory

The research framework is evaluated across four distinct robotic capture environments representing diverse kinematics, visual textures, and dynamic complexities. Every quantity reported below is grounded directly in committed empirical data files and verified by cryptographic hashes:

| Attribute | TUM RGB-D (`fr3_walking_xyz`) | TUM RGB-D Control (`fr1_desk`) | OpenLORIS-Scene (`corridor1-4`) | Fused Desk Sequence (`recording_01` + `02`) |
| :--- | :--- | :--- | :--- | :--- |
| **Adapter Class** | `TUMRGBDAdapter` | `TUMRGBDAdapter` | `OpenLORISAdapter` | `MultiSessionSemanticMapper` |
| **Physical Sensor** | Microsoft Kinect v1 (Structured Light) | Microsoft Kinect v1 (Structured Light) | Intel RealSense D435 (Stereo IR) | Intel RealSense D455 (Stereo IR) |
| **Resolution ($W \times H$)** | $640 \times 480$ | $640 \times 480$ | $848 \times 480$ | $640 \times 480$ |
| **Depth Scale** | $0.0002\text{ m/unit}$ ($5000\text{ units/m}$) | $0.0002\text{ m/unit}$ ($5000\text{ units/m}$) | $0.0010\text{ m/unit}$ ($1000\text{ units/m}$) | $0.0010\text{ m/unit}$ |
| **Synchronized Pairs** | 827 RGB-D pairs | 613 RGB-D pairs | 1,500 RGB-D pairs | 10,494 pairs (5,207 + 5,287) |
| **Sequence Duration** | $28.82\text{ s}$ | $23.40\text{ s}$ | $50.01\text{ s}$ | Multi-session ($2 \times \sim 175\text{ s}$) |
| **Ground Truth Type** | Vicon Optical MoCap ($< 1\text{ mm}$, $100\text{ Hz}$) | Vicon Optical MoCap ($< 1\text{ mm}$, $100\text{ Hz}$) | 3D LiDAR ICP Scan-Matching ($\sim 10\text{ mm}$) | High-Precision Static Anchors |
| **Ground Truth Poses** | 2,884 poses | 613 poses | 2,515 poses | Reference Session Keyframes |
| **Matched Poses** | 826 / 827 ($99.9\%$) | 613 / 613 ($100.0\%$) | 1,499 / 1,500 ($99.9\%$) | Cross-Session Matched Poses |
| **Synchronization Tolerance** | $\pm 0.02\text{ s}$ ($\le 20\text{ ms}$) | $\pm 0.02\text{ s}$ ($\le 20\text{ ms}$) | $\pm 0.02\text{ s}$ ($\le 20\text{ ms}$) | SE(2) Proximity $< 0.10\text{ m}$ |
| **Camera Kinematics** | 6-DoF aggressive hand-held translation along XYZ | 6-DoF smooth desktop inspection trajectory | Wheeled ground robot planar translation | Handheld inspection trajectory |
| **Dynamic Complexity** | 2 walking persons + moving office chairs | Static control (no dynamic actors) | Transient pedestrian hallway crossings | Human-displaced chairs, monitors, mugs |
| **Visual Texture** | Dense indoor office furniture and posters | Dense desktop office objects (books, monitor) | Sparse, low-texture white corridor walls | Office desk environment |

---

## 2. Sensor Models & Intrinsic Calibrations

### 2.1 Microsoft Kinect v1 (TUM RGB-D `fr3_walking_xyz`)
* **Focal Length**: $f_x = 535.4$, $f_y = 539.2$
* **Principal Point**: $c_x = 320.1$, $c_y = 247.6$
* **Distortion Coefficients**: $k_1 = 0.0$, $k_2 = 0.0$, $p_1 = 0.0$, $p_2 = 0.0$ (pre-rectified sequence)
* **Depth Acquisition Model**: Infrared structured light pattern. Depth is encoded as 16-bit unsigned integers where $Z_{\text{metric}} = \frac{\text{raw\_val}}{5000.0}$.

### 2.2 Intel RealSense D435 (OpenLORIS-Scene `corridor1-4`)
* **Focal Length**: $f_x = 428.16$, $f_y = 428.16$
* **Principal Point**: $c_x = 428.84$, $c_y = 239.38$
* **Depth Acquisition Model**: Active infrared stereo with speckle projector. Raw 16-bit values represent depth in millimeters: $Z_{\text{metric}} = \frac{\text{raw\_val}}{1000.0}$.

### 2.3 Intel RealSense D455 (Fused Desk Sequence)
* **Depth Acquisition Model**: Wide-baseline stereo infrared. Metric conversion: $Z_{\text{metric}} = \frac{\text{raw\_val}}{1000.0}$.

---

## 3. Benchmark Populations & Evaluation Manifests

To ensure rigorous statistical isolation and prevent circular evaluation, instances are partitioned into explicitly defined, tamper-evident populations:

```
┌────────────────────────────────────────────────────────┐
│ Pop 52: Complete Ground-Truth Entity Census (N = 52)   │
│ ┌────────────────────────────────────────────────────┐ │
│ │ Pop 42: Visual Image-Eligible Benchmark (N = 42)   │ │
│ │ (Verified physical instances with multi-view crops)│ │
│ └────────────────────────────────────────────────────┘ │
│ ┌────────────────────────────────────────────────────┐ │
│ │ Pop 53: Router Evaluation Population (N = 53)      │ │
│ │ (Includes unobserved / out-of-distribution cases)  │ │
│ └────────────────────────────────────────────────────┘ │
└────────────────────────────────────────────────────────┘
```

1. **Population 52 (`pop_52_ground_truth`, $N = 52$)**:
   - The authoritative physical object census across both sessions of the multi-session desk environment.
   - Ground truth mobility distribution:
     - `IMMOVABLE`: 3 instances (fixed architectural structures, built-in cabinetry).
     - `MOVABLE`: 45 instances (chairs, monitors, laptops, keyboards, backpacks, storage boxes).
     - `MOBILE`: 4 instances (quadcopter drones, mobile robotic platforms, RC cars).

2. **Population 42 (`pop_42_image_eligible`, $N = 42$)**:
   - The subset of Pop 52 possessing verified, high-quality multi-view RGB crop assets across camera viewpoints.
   - Evaluates visual physical movability classification under controlled visual modes:
     - `CROP_ONLY`: Tight bounding box crop centered on the target object.
     - `CROP_PLUS_CONTEXT`: Bounding box crop alongside downscaled full-scene context image showing surrounding support structures and mounting surfaces.

3. **Population 53 (`pop_53_router_evaluation`, $N = 53$)**:
   - Extended evaluation population used for router policy benchmark replay, incorporating 10 edge cases and out-of-distribution counterexamples (e.g. wall-mounted display vs table-top display, castor-wheel chair vs heavy conference table).

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

