# Baseline ORB-SLAM3 Campaign Execution and Verification Report

**Timestamp (UTC)**: 2026-11-08T16:28:00Z  
**Total Sequences**: 8  
**Control Condition**: Original, unmodified RGB-D streams, no masks, no VLM decisions, no semantic weighting.  

## Executive Summary Table

| Sequence ID | Sequence Name | Family | Poses | Coverage | ATE RMSE (m) | RPE Trans (m/frame) | RPE Rot (deg/frame) | Verification Status |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| `tum_xyz` | `rgbd_dataset_freiburg3_walking_xyz` | TUM RGB-D | 827 | 28.64% | 0.3957 | 0.0225 | 0.5461 | **VERIFIED_REUSED** |
| `tum_halfsphere` | `rgbd_dataset_freiburg3_walking_halfsphere` | TUM RGB-D | 1021 | 28.42% | 0.2923 | 0.0208 | 0.5259 | **VERIFIED_REUSED** |
| `tum_static` | `rgbd_dataset_freiburg3_walking_static` | TUM RGB-D | 717 | 28.76% | 0.0140 | 0.0127 | 0.2482 | **VERIFIED_REUSED** |
| `openloris` | `OpenLORIS-Scene_corridor1-4` | OpenLORIS | 1500 | 59.60% | 1.6394 | 0.0583 | 0.5140 | **VERIFIED_REUSED** |
| `bonn_crowd` | `rgbd_bonn_crowd` | Bonn Dynamic | 927 | 99.04% | 0.4993 | 0.0951 | 1.5341 | **VERIFIED_REUSED** |
| `bonn_box` | `rgbd_bonn_moving_obstructing_box` | Bonn Dynamic | 589 | 99.16% | 0.5565 | 0.0223 | 1.1654 | **VERIFIED_REUSED** |
| `d455_01` | `my_desk_recording_01` | RealSense D455 | 4980 | 100.00% | N/A (GT-free) | N/A | N/A | **VERIFIED_REUSED** |
| `d455_02` | `my_desk_recording_02` | RealSense D455 | 5283 | 100.00% | N/A (GT-free) | N/A | N/A | **VERIFIED_REUSED** |

## Artifact Integrity and Provenance

Each baseline sequence directory under `results/baselines/<sequence_name>/` contains:
- `CameraTrajectory.txt`: Full 6-DoF estimated camera trajectory (TUM format)
- `KeyFrameTrajectory.txt`: Keyframe camera trajectory
- `MapPoints.ply`: 3D map points point cloud export
- `associations.txt`: Exact RGB-D frame association pairings
- `camera_settings.yaml`: Exact sensor calibration parameters used for execution
- `orbslam3_execution.log`: Full stdout/stderr execution log from ORB-SLAM3 runtime
- `baseline_evaluation_metrics.json`: Evaluated ATE / RPE metrics (or GT-free pose record)
- `run_manifest.json`: Cryptographic run manifest with SHA-256 digests for all inputs and outputs

All eight baseline runs establish the fixed experimental control condition for all downstream evaluations.
