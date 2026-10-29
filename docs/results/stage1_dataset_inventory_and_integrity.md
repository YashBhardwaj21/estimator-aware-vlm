# Stage 1: Dataset Inventory, Calibration & Integrity Audit

**Date**: 2026-10-28  
**Audit Script**: [`scripts/check_dataset_integrity.py`](file:///c:/Users/Yash%20Bhardwaj/Desktop/vlm/scripts/check_dataset_integrity.py)  
**Machine-Readable Report**: [`outputs/reports/stage1_dataset_inventory.json`](file:///c:/Users/Yash%20Bhardwaj/Desktop/vlm/outputs/reports/stage1_dataset_inventory.json)  
**Test Suite**: [`tests/test_tum_adapter.py`](file:///c:/Users/Yash%20Bhardwaj/Desktop/vlm/tests/test_tum_adapter.py), [`tests/test_dataset_contracts.py`](file:///c:/Users/Yash%20Bhardwaj/Desktop/vlm/tests/test_dataset_contracts.py) (17/17 passed)

---

## 1. Executive Summary

Stage 1 establishes a dataset-agnostic, contract-enforced, and fail-safe input pipeline for RGB-D SLAM experimentation. It addresses five critical experimental validity requirements:
1. **Raw Data Immutability**: All downloaded sequence directories (`data/raw/*`) are strictly read-only. Stray generated files from previous runs (`associations.txt`, `associations_pass_b.*`, `masked_pass_b/`) were safely quarantined into `archive/stray_raw_artifacts/`. Programmatic guards now raise `PermissionError` if an export targets raw data directories.
2. **Dedicated OpenLORIS Adapter**: Built [`OpenLORISAdapter`](file:///c:/Users/Yash%20Bhardwaj/Desktop/vlm/src/estimator_aware_vlm/datasets/openloris.py) adhering to the canonical `DatasetAdapter` protocol, supporting RealSense D400 streams, intrinsics, and depth conversion ($0.001\text{ m/unit} = 1000\text{ units/m}$).
3. **Strict 1-to-1 Association**: Implemented [`associate_streams()`](file:///c:/Users/Yash%20Bhardwaj/Desktop/vlm/src/estimator_aware_vlm/datasets/base.py#L239) with greedy closest-unmatched pairing, timestamp deduplication, and float epsilon boundary handling.
4. **Continuous Ground-Truth Evaluation**: Added [`quaternion_slerp()`](file:///c:/Users/Yash%20Bhardwaj/Desktop/vlm/src/estimator_aware_vlm/datasets/base.py#L66) and [`Trajectory.interpolate_pose_at()`](file:///c:/Users/Yash%20Bhardwaj/Desktop/vlm/src/estimator_aware_vlm/datasets/base.py#L180), strictly preventing extrapolation outside ground-truth time spans.
5. **Ground-Truth Role Quarantine**: Poses from `groundtruth.txt` are tagged `TrajectoryRole.GROUND_TRUTH` and prohibited from entering estimator, tracking, or routing slots.

---

## 2. Factual Sequence Inventory

| Metric | TUM RGB-D (`fr3_walking_xyz`) | OpenLORIS-Scene (`corridor1-4`) |
| :--- | :--- | :--- |
| **Adapter Class** | `TUMRGBDAdapter` | `OpenLORISAdapter` |
| **Dataset Type** | `tum_rgbd` | `openloris` |
| **Sensor Hardware** | Microsoft Kinect v1 RGB-D | Intel RealSense D400 (D435 Color/Aligned Depth) |
| **Native Resolution** | $640 \times 480$ | $848 \times 480$ |
| **Calibrated Focal Length ($f_x, f_y$)** | $535.40, 539.20$ | $611.45, 611.49$ |
| **Principal Point ($c_x, c_y$)** | $320.10, 247.60$ | $433.20, 249.47$ |
| **Distortion Model / Coeffs** | Plumb bob / $[0, 0, 0, 0, 0]$ | Plumb bob / $[0, 0, 0, 0, 0]$ |
| **Depth Scale Factor** | $0.0002\text{ m/unit}$ ($5000\text{ units/m}$) | $0.0010\text{ m/unit}$ ($1000\text{ units/m}$) |
| **Synchronized Frames on Disk** | 827 | 1,500 |
| **Effective Frame Rate** | $28.7\text{ Hz}$ | $30.0\text{ Hz}$ |
| **RGB-D Time Span** | $1341846313.592 - 1341846342.416$ ($28.8\text{s}$) | $1560200172.200 - 1560200222.211$ ($50.0\text{s}$) |
| **Ground-Truth Provenance** | High-precision Vicon motion capture | Offline LiDAR-Odometry SLAM (global frame) |
| **Total Ground-Truth Poses** | 2,884 | 2,515 |
| **Ground-Truth Time Span** | $1341846313.638 - 1341846342.470$ | $1560200172.200 - 1560200235.179$ |
| **Matched RGB-D / GT Pairs** | $826 / 827$ ($99.9\%$) | $1,499 / 1,500$ ($99.9\%$) |
| **Trajectory Role Tag** | `TrajectoryRole.GROUND_TRUTH` | `TrajectoryRole.GROUND_TRUTH` |
| **Integrity Audit Status** | **PASSED (0 issues)** | **PASSED (0 issues)** |

> [!NOTE]
> On OpenLORIS `corridor1-4`, `color.txt` and `aligned_depth.txt` contain 1,889 timestamp entries, but the local disk contains exactly 1,500 PNG files due to rosbag extraction truncation. `OpenLORISAdapter` validates disk existence for every pair, returning exactly 1,500 valid frames without file read exceptions.

---

## 3. Data Integrity & Immutability Verification

### Immutability Audit
- **TUM Raw Directory** (`data/raw/rgbd_dataset_freiburg3_walking_xyz`):
  - Verified files: `rgb.txt`, `depth.txt`, `groundtruth.txt`, `accelerometer.txt`, `rgb/`, `depth/`.
  - Stray files removed: `associations.txt`, `associations_pass_b.txt`, `associations_pass_b.manifest.json`, `masked_pass_b/`.
  - Status: **Pristine / Read-Only**.
- **OpenLORIS Raw Directory** (`data/raw/OpenLORIS-Scene_corridor1-4`):
  - Verified files: `color.txt`, `aligned_depth.txt`, `groundtruth.txt`, `sensors.yaml`, `trans_matrix.yaml`, `d400_accelerometer.txt`, `d400_gyroscope.txt`, `color/`, `aligned_depth/`, `groundtruth/`.
  - Stray files removed: `associations.txt`, `associations_pass_b.txt`, `associations_pass_b.manifest.json`, `masked_pass_b/`.
  - Status: **Pristine / Read-Only**.

### Automated Tests ([`tests/test_dataset_contracts.py`](file:///c:/Users/Yash%20Bhardwaj/Desktop/vlm/tests/test_dataset_contracts.py))
- `test_associate_streams_empty_streams`: Passes.
- `test_associate_streams_duplicate_timestamps`: Passes.
- `test_associate_streams_strict_one_to_one`: Passes (no duplicate target usage).
- `test_associate_streams_tolerance_boundary`: Passes with float epsilon margin.
- `test_quaternion_slerp_properties`: Passes (interpolates rotations and verifies antipodal symmetry $\mathbf{q} \equiv -\mathbf{q}$).
- `test_trajectory_interpolation_no_extrapolation`: Passes (strictly rejects timestamps $< t_0$ or $> t_{\text{end}}$).
- `test_trajectory_interpolation_gap_rejection`: Passes (rejects gaps $> 1.5\text{s}$).
- `test_adapter_factory_dispatch`: Passes (`get_dataset_adapter` dispatches to `TUMRGBDAdapter` and `OpenLORISAdapter`).
- `test_adapter_canonical_frame_contract`: Passes (verifies frame contracts and `estimated_pose is None`).
- `test_raw_data_immutability_prohibition`: Passes (`export_associations` raises `PermissionError` when directed to raw folders).
- `test_raw_data_directory_is_clean`: Passes (verifies no stray files exist in raw dirs).

---

## 4. Scientific Scope & Methodological Boundaries

In compliance with the Master Implementation Prompt:
1. **Single-Session Limitation**: Both available sequences (`fr3_walking_xyz` and `corridor1-4`) are single-session runs.
   - `fr3_walking_xyz` evaluates dynamic-object robustness (two people walking while camera translates).
   - `corridor1-4` evaluates camera tracking in an indoor corridor.
2. **Untested Long-Term Persistence**: There are no repeated-session revisit or environmental reconfiguration sequences present in the local filesystem. Long-term landmark persistence and across-session map consistency are declared **NOT TESTED** on empirical data.
3. **No Ground-Truth Leakage**: Ground-truth poses are isolated under `TrajectoryRole.GROUND_TRUTH` and are used exclusively in downstream trajectory evaluation (ATE/RPE), never in online tracking or routing.
