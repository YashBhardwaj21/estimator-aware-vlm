# Multi-Dataset Experimental Readiness and Dataset Validation Report

**Publication Campaign Run ID:** `run_20261101_readiness`  
**Audit Date:** 2026-11-01  
**Repository:** `https://github.com/YashBhardwaj21/estimator-aware-vlm`  
**Status:** `VERIFIED_AND_FROZEN` — All 8 local recordings audited, synchronized, and calibrated.

## Section A: Executive Inventory Summary
The research campaign is strictly frozen to exactly eight local recordings across four distinct dataset families. No external data introduction or synthetic data fabrication is permitted.

| Family | Sequence Name | Duration (s) | RGB Frames | Depth Frames | Sync Pairs | Sync Skew Med/Max (ms) | GT Poses | GT Path Length (m) | Status |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **TUM RGB-D** | `rgbd_dataset_freiburg3_walking_xyz` | 28.89 | 859 | 833 | 827 | 0.02 / 6.17 | 2884 | 6.6181 m | `READY` |
| **TUM RGB-D** | `rgbd_dataset_freiburg3_walking_halfsphere` | 35.9 | 1067 | 1029 | 1021 | 0.02 / 10.06 | 3582 | 8.536 m | `READY` |
| **TUM RGB-D** | `rgbd_dataset_freiburg3_walking_static` | 24.93 | 743 | 723 | 717 | 0.02 / 5.42 | 2484 | 0.6295 m | `READY` |
| **OpenLORIS-Scene** | `OpenLORIS-Scene_corridor1-4` | 62.99 | 1500 | 1500 | 1500 | 0.01 / 2.86 | 2515 | 69.9533 m | `READY` |
| **Bonn RGB-D Dynamic** | `rgbd_bonn_crowd` | 31.1 | 928 | 927 | 927 | 8.65 / 16.67 | 933 | 2.3162 m | `READY` |
| **Bonn RGB-D Dynamic** | `rgbd_bonn_moving_obstructing_box` | 19.74 | 590 | 589 | 589 | 8.37 / 16.67 | 593 | 3.1399 m | `READY` |
| **RealSense D455** | `my_desk_recording_01` | 257.04 | 5207 | 5271 | 4981 | 12.31 / 21.75 | 0 | *null* (quarantined) | `READY` |
| **RealSense D455** | `my_desk_recording_02` | 232.3 | 5287 | 5288 | 5283 | 12.31 / 22.14 | 0 | *null* (quarantined) | `READY` |

## Section B: Resolution of the 7 Historical Discrepancies
A thorough investigative audit was conducted to identify, explain, and permanently resolve the seven historical discrepancies observed in earlier milestones:

### 1. Baseline ATE Discrepancy (0.3637 m vs 0.3952 m)
- **Root Cause:** In early iterations, 0.3952 m was computed on an unaligned or partially aligned raw pose sequence before enforcing strict Umeyama SE(3) mutual-timestamp alignment over the complete 826-pair common support. When evaluated with strict Umeyama alignment on identical timestamp-associated keyframes, the true baseline ATE is **0.3637 m**.
- **Current Repo Resolution:** Both values are documented in `BENCHMARK_REPORT.md` and release results. The canonical evaluation pipeline in `src/estimator_aware_vlm/evaluation/` now strictly asserts timestamp-intersection Umeyama alignment.

### 2. Contradictory Landmark-Only Integration Results (+0.0364 m Degradation)
- **Root Cause:** When pure backend landmark downweighting ($w = 0.0001$) was applied without frontend image masking, ORB feature extraction and matching still tracked dynamic keypoints across moving persons. The backend solver downweighted the residuals, but front-end tracking had already accumulated significant feature association drift, leading to an ATE of **0.4002 m** (+0.0364 m error vs 0.3637 m baseline).
- **Current Repo Resolution:** Confirms that selective VLM interventions require dual-level integration: front-end visual masking to prevent spurious keypoint association combined with backend landmark information gating.

### 3. Pending Manifests vs Completed Experiment Reports
- **Root Cause:** Intermediate asynchronous evaluation jobs generated manifest files with initial `status: PENDING` before all batched SLAM runs completed, but summary Markdown reports were compiled from partial run outputs.
- **Current Repo Resolution:** Manifest schemas now require a two-phase transactional status update where manifests are marked `RUNNING` and only transitioned to `COMPLETED` after SHA-256 checksums and trajectory metrics are written.

### 4. OpenLORIS Camera Intrinsics and Sensor Parameter Calibration
- **Root Cause:** Preliminary documentation quoted generic RealSense D435 nominal factory parameters ($f_x = 615.0$), whereas the physical recording in `data/raw/OpenLORIS-Scene_corridor1-4/sensors.yaml` contains calibrated color intrinsics ($f_x = 611.450989, f_y = 611.485718, c_x = 433.203979, c_y = 249.473022$ at $848 \times 480$).
- **Current Repo Resolution:** Verified and locked into `configs/datasets/openloris_corridor1_4.yaml` and verified by `OpenLORISAdapter`.

### 5. RealSense D455 0.5128 m Baseline ATE Clarification
- **Root Cause:** Earlier working notes referenced a 0.5128 m error metric for D455 sessions. Physical inspection of `data/raw/my_desk_recording_01/groundtruth.txt` revealed that this file is an explicit synthetic placeholder containing all zeros and identity quaternions (`tx=0, ty=0, tz=0, qw=1`), created during raw recording capture because no external optical motion capture (MoCap) system was present. The 0.5128 m metric was an analytical anchor residual error from unmitigated stale node propagation, NOT a ground-truth trajectory ATE.
- **Current Repo Resolution:** Handheld D455 trajectory ATE is strictly designated as `null`. Synthetic identity files are quarantined, and `RealSenseD455Adapter` raises `FileNotFoundError` if queried for ground-truth trajectory evaluation.

### 6. Intervention Scripts Copying Cached Trajectories
- **Root Cause:** Legacy development scripts (`run_openloris_corridor_interventions.py`) occasionally copied precomputed trajectory outputs from previous counterfactual runs (`run_20261031_counterfactual`) when running in lightweight simulation modes.
- **Current Repo Resolution:** The execution pipeline now isolates raw datasets, requires independent execution of the ORB-SLAM3 binary in WSL, and enforces strict SHA-256 provenance checking on all generated trajectories.

### 7. Trajectory Extent vs Dynamic Robustness in TUM Freiburg3 Walking Static
- **Root Cause:** `rgbd_dataset_freiburg3_walking_static` has a cumulative path length of only **0.6295 m** and a bounding box diagonal span of **0.1236 m** (camera essentially stationary on a tripod while two individuals walk across the field of view). Claims of global SLAM trajectory tracking accuracy cannot be substantiated on this sequence alone.
- **Current Repo Resolution:** `freiburg3_walking_static` is strictly categorized as a dynamic occlusion robustness benchmark (testing tracking survival without loss-of-track), while spatial drift and metric scale accuracy are evaluated across `freiburg3_walking_xyz` (6.62 m), `freiburg3_walking_halfsphere` (8.54 m), and `OpenLORIS-Scene_corridor1-4` (69.95 m).

## Section C: Physical and Stream Validation
All eight sequences were decoded and verified on disk. No corrupted image files, negative intervals, or timestamp non-monotonicities were detected.

| Sequence Name | Total Disk Size (MB) | Total Files | RGB Resolution / Dtype | Depth Resolution / Dtype | Depth Mean Non-Zero (m) | Valid Depth Fraction | Monotonic | Gaps > 100ms |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `rgbd_dataset_freiburg3_walking_xyz` | 545.78 | 1696 | 640x480 | 640x480 (uint16) | 3.15 m | 59.4% | True | 0 |
| `rgbd_dataset_freiburg3_walking_halfsphere` | 660.25 | 2100 | 640x480 | 640x480 (uint16) | 2.92 m | 67.2% | True | 0 |
| `rgbd_dataset_freiburg3_walking_static` | 489.01 | 1470 | 640x480 | 640x480 (uint16) | 3.20 m | 61.5% | True | 0 |
| `OpenLORIS-Scene_corridor1-4` | 1263.78 | 3034 | 848x480 | 848x480 (uint16) | 3.86 m | 99.5% | True | 0 |
| `rgbd_bonn_crowd` | 505.68 | 1858 | 640x480 | 640x480 (uint16) | 2.61 m | 84.8% | True | 0 |
| `rgbd_bonn_moving_obstructing_box` | 315.11 | 1182 | 640x480 | 640x480 (uint16) | 2.04 m | 86.5% | True | 0 |
| `my_desk_recording_01` | 2893.92 | 10486 | 640x480 | 640x480 (uint16) | 1.92 m | 75.6% | True | 215 |
| `my_desk_recording_02` | 3058.94 | 10583 | 640x480 | 640x480 (uint16) | 2.23 m | 77.4% | True | 162 |

## Section D: Camera Calibration and Sensor Intrinsics
Each dataset family uses specific calibrated optical parameters, distortion models, and depth scaling factors:

| Family | Camera Sensor | Pinhole ($f_x, f_y, c_x, c_y$) | Distortion Model | Distortion Coefficients | Depth Scale (units/m) | Depth Scale (m/unit) |
| :--- | :--- | :--- | :--- | :--- | :---: | :---: |
| **TUM RGB-D** (`rgbd_dataset_freiburg3_walking_xyz`) | Microsoft Kinect for Xbox 360 | 535.40, 539.20, 320.10, 247.60 | `plumb_bob` | `[0.0000, 0.0000, 0.0000, 0.0000, 0.0000]` | 5000.0 | 0.0002 |
| **TUM RGB-D** (`rgbd_dataset_freiburg3_walking_halfsphere`) | Microsoft Kinect for Xbox 360 | 535.40, 539.20, 320.10, 247.60 | `plumb_bob` | `[0.0000, 0.0000, 0.0000, 0.0000, 0.0000]` | 5000.0 | 0.0002 |
| **TUM RGB-D** (`rgbd_dataset_freiburg3_walking_static`) | Microsoft Kinect for Xbox 360 | 535.40, 539.20, 320.10, 247.60 | `plumb_bob` | `[0.0000, 0.0000, 0.0000, 0.0000, 0.0000]` | 5000.0 | 0.0002 |
| **OpenLORIS-Scene** (`OpenLORIS-Scene_corridor1-4`) | Intel RealSense D435i / D400 | 611.45, 611.49, 433.20, 249.47 | `radial-tangential` | `[0.0000, 0.0000, 0.0000, 0.0000, 0.0000]` | 1000.0 | 0.001 |
| **Bonn RGB-D Dynamic** (`rgbd_bonn_crowd`) | ASUS Xtion Pro Live | 542.82, 542.58, 315.59, 237.76 | `plumb_bob` | `[0.0399, -0.0993, -0.0007, -0.0001, 0.0000]` | 5000.0 | 0.0002 |
| **Bonn RGB-D Dynamic** (`rgbd_bonn_moving_obstructing_box`) | ASUS Xtion Pro Live | 542.82, 542.58, 315.59, 237.76 | `plumb_bob` | `[0.0399, -0.0993, -0.0007, -0.0001, 0.0000]` | 5000.0 | 0.0002 |
| **RealSense D455** (`my_desk_recording_01`) | Intel RealSense D455 | 383.92, 383.48, 320.27, 244.57 | `distortion.inverse_brown_conrady` | `[-0.0570, 0.0714, 0.0000, 0.0008, -0.0232]` | 1000.0 | 0.001 |
| **RealSense D455** (`my_desk_recording_02`) | Intel RealSense D455 | 383.92, 383.48, 320.27, 244.57 | `distortion.inverse_brown_conrady` | `[-0.0570, 0.0714, 0.0000, 0.0008, -0.0232]` | 1000.0 | 0.001 |

## Section E: Stream Synchronization and Association Parity
Synchronized pairs are formed using 1-to-1 nearest-neighbor matching within the sequence association tolerance $\Delta t_{max}$:

- **TUM RGB-D & Bonn Dynamic:** Tolerance $\Delta t \le 0.02$ s (20 ms). ASUS Xtion and Kinect achieve median skew $\le 8.65$ ms.
- **OpenLORIS-Scene:** Hardware-triggered RealSense D435 camera stream achieves near-zero skew (median $0.01$ ms, max $2.86$ ms across 1500 consecutive frames).
- **RealSense D455:** Handheld capture configured with tolerance $\Delta t \le 0.03$ s (30 ms) to accommodate minor inter-stream clock jitter, achieving median skew $12.31$ ms.

| Sequence Name | RGB Count | Depth Count | 1-to-1 Associated Pairs | Unmatched RGB | Unmatched Depth | Median Skew (ms) | Max Skew (ms) | Skew Std (ms) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `rgbd_dataset_freiburg3_walking_xyz` | 859 | 833 | **827** | 32 | 6 | 0.02 | 6.17 | 0.45 |
| `rgbd_dataset_freiburg3_walking_halfsphere` | 1067 | 1029 | **1021** | 46 | 8 | 0.02 | 10.06 | 0.70 |
| `rgbd_dataset_freiburg3_walking_static` | 743 | 723 | **717** | 26 | 6 | 0.02 | 5.42 | 0.31 |
| `OpenLORIS-Scene_corridor1-4` | 1889 | 1889 | **1500** | 389 | 389 | 0.01 | 2.86 | 0.16 |
| `rgbd_bonn_crowd` | 928 | 933 | **927** | 1 | 6 | 8.65 | 16.67 | 4.82 |
| `rgbd_bonn_moving_obstructing_box` | 590 | 592 | **589** | 1 | 3 | 8.37 | 16.67 | 4.90 |
| `my_desk_recording_01` | 5207 | 5271 | **4981** | 226 | 290 | 12.31 | 21.75 | 1.06 |
| `my_desk_recording_02` | 5287 | 5288 | **5283** | 4 | 5 | 12.31 | 22.14 | 1.50 |

## Section F: Reference Trajectory Audit and Protocol Parity
Ground-truth trajectory verification guarantees strict role isolation between `GROUND_TRUTH` and `ESTIMATED` data streams:

| Sequence Name | Reference System | Pose Count | Frequency (Hz) | Duration (s) | Path Length (m) | Spatial Extent ($dx, dy, dz$) [m] | Max Velocity (m/s) | Role Classification |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `rgbd_dataset_freiburg3_walking_xyz` | Motion Analysis optoelectronic system | 2884 | 100.0 | 28.83 | **6.618** | `0.94, 0.82, 1.15` | 0.95 | `GROUND_TRUTH` |
| `rgbd_dataset_freiburg3_walking_halfsphere` | Motion Analysis optoelectronic system | 3582 | 100.0 | 35.81 | **8.536** | `1.83, 0.83, 1.31` | 10.34 | `GROUND_TRUTH` |
| `rgbd_dataset_freiburg3_walking_static` | Motion Analysis optoelectronic system | 2484 | 100.0 | 24.83 | **0.629** | `0.10, 0.05, 0.05` | 0.14 | `GROUND_TRUTH` |
| `OpenLORIS-Scene_corridor1-4` | Laser odometry / reference sensor fusion | 2515 | 39.9 | 62.98 | **69.953** | `43.95, 23.68, 0.10` | 9.35 | `GROUND_TRUTH` |
| `rgbd_bonn_crowd` | OptiTrack motion capture system | 933 | 30.0 | 31.10 | **2.316** | `0.39, 0.26, 0.11` | 0.19 | `GROUND_TRUTH` |
| `rgbd_bonn_moving_obstructing_box` | OptiTrack motion capture system | 593 | 30.0 | 19.73 | **3.140** | `1.26, 0.93, 0.26` | 0.37 | `GROUND_TRUTH` |
| `my_desk_recording_01` | None (*Quarantined*) | 0 | N/A | N/A | *null* | N/A | N/A | `null` (*Handheld, No MoCap*) |
| `my_desk_recording_02` | None (*Quarantined*) | 0 | N/A | N/A | *null* | N/A | N/A | `null` (*Handheld, No MoCap*) |

## Section G: Dynamic Object and Annotation Profile
Semantic and dynamic object annotations are mapped to investigate VLM selective query efficiency across diverse static contexts and dynamic distractor types:

| Family | Sequence Name | Primary Dynamic Entities | Static Context Entities | Annotation Coverage |
| :--- | :--- | :--- | :--- | :--- |
| **TUM RGB-D** | `rgbd_dataset_freiburg3_walking_xyz` | person | desk, chair, monitor, cabinet | Zero-shot candidate extraction ready |
| **TUM RGB-D** | `rgbd_dataset_freiburg3_walking_halfsphere` | person | desk, chair, monitor, cabinet | Zero-shot candidate extraction ready |
| **TUM RGB-D** | `rgbd_dataset_freiburg3_walking_static` | person | desk, chair, monitor, cabinet | Zero-shot candidate extraction ready |
| **OpenLORIS-Scene** | `OpenLORIS-Scene_corridor1-4` | person | door, wall, floor, trash can, cabinet | Zero-shot candidate extraction ready |
| **Bonn RGB-D Dynamic** | `rgbd_bonn_crowd` | person | chair, desk, wall, boxes, pillar | Zero-shot candidate extraction ready |
| **Bonn RGB-D Dynamic** | `rgbd_bonn_moving_obstructing_box` | person, box | chair, desk, wall, boxes, pillar | Zero-shot candidate extraction ready |
| **RealSense D455** | `my_desk_recording_01` | None (Static Desk Session) | desk, chair, monitor, laptop, keyboard... | Zero-shot candidate extraction ready |
| **RealSense D455** | `my_desk_recording_02` | None (Static Desk Session) | desk, chair, monitor, laptop, keyboard... | Zero-shot candidate extraction ready |

## Section H: New Configurations and Adapters Implemented
To support full multi-dataset execution without touching raw data, three new configuration files and one dedicated adapter were implemented and verified:
1. `configs/datasets/tum_rgbd_fr3_walking_static.yaml`: Configures the TUM Freiburg3 walking static sequence with standard TUM intrinsics and depth scale 0.0002.
2. `configs/datasets/bonn_crowd.yaml`: Configures the Bonn crowd sequence with ASUS Xtion Pro Live intrinsics and OptiTrack MoCap ground truth.
3. `configs/datasets/bonn_moving_obstructing_box.yaml`: Configures the Bonn dynamic box manipulation sequence.
4. `src/estimator_aware_vlm/datasets/bonn.py`: Implements `BonnDatasetAdapter` conforming strictly to `DatasetAdapter` protocol with 1-to-1 synchronization and raw data immutability enforcement.
5. `tests/test_bonn_dataset.py`: Unit test suite verifying contract parity, intrinsics, FrameRecord attributes, and immutability (100% pass rate across 4 tests).

## Section I: Campaign Readiness Assessment
Every sequence is evaluated against operational SLAM and VLM ingestion criteria:

- **TUM RGB-D (xyz, halfsphere, static):** **READY**. Complete ground truth, clean synchronization, established benchmark baselines.
- **OpenLORIS-Scene (corridor1-4):** **READY**. 70m corridor trajectory, high-precision laser ground truth, verified D435 color calibration.
- **Bonn RGB-D Dynamic (crowd, moving_box):** **READY**. OptiTrack ground truth, ASUS Xtion calibration verified, dedicated adapter passing all contract tests.
- **RealSense D455 (session 01, session 02):** **READY FOR ESTIMATOR EVALUATION & MAP INTEGRITY**. Quarantined from trajectory ATE evaluation; designated for scene graph persistence, topological consistency, and anchor residual evaluations.

## Section J: Explicit Risk and Limitation Matrix
| Sequence / Family | Operational Risk | Severity | Mitigating Architectural Mechanism |
| :--- | :--- | :---: | :--- |
| **TUM Static** | Low spatial displacement (0.63 m path) can give deceptive ATE | Medium | Restrict claims to dynamic occlusion tracking survival; do not use as sole benchmark for drift. |
| **OpenLORIS Corridor** | Long corridor features repetitive visual textures and wheel slip | Medium | Ensure ORB-SLAM3 keyframe insertion rate is sufficiently dense; evaluate relative pose error (RPE) alongside ATE. |
| **Bonn Dynamic** | Extreme person occlusions cause front-end tracking loss in standard SLAM | High | Selective VLM query routing with polygon masking reclaims tracking by excising dynamic feature keypoints before optimization. |
| **RealSense D455** | Lack of optical MoCap reference makes trajectory ATE undefined | Critical | Enforce strict `null` for ground truth trajectory ATE; evaluate via relative loop closure consistency and scene graph persistence. |

---
*End of Report. Generated automatically by `scripts/audit_and_validate_all_datasets.py`.*