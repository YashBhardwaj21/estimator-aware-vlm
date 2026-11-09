# Baseline ORB-SLAM3 Campaign Execution and Verification Report

**Timestamp (UTC)**: 2026-11-08T19:09:16Z  
**Total Sequences**: 8  
**Control Condition**: Original, unmodified RGB-D streams, no masks, no VLM decisions, no semantic weighting.  

## Executive Summary Table

| Sequence ID | Sequence Name | Poses | Frame Tracking % | GT Sample Cov % | Est Match Rate % | ATE RMSE (m) | 1s RPE Trans (m) | 1s RPE Rot (deg) | Status |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `rgbd_dataset_freiburg3_walking_xyz` | `rgbd_dataset_freiburg3_walking_xyz` | 827 | 100.00% | 28.64% | 99.88% | 0.3957 | 0.2777 | 5.29 | **VERIFIED_REUSED** |
| `rgbd_dataset_freiburg3_walking_halfsphere` | `rgbd_dataset_freiburg3_walking_halfsphere` | 1021 | 100.00% | 28.42% | 99.71% | 0.2923 | 0.1590 | 2.66 | **VERIFIED_REUSED** |
| `rgbd_dataset_freiburg3_walking_static` | `rgbd_dataset_freiburg3_walking_static` | 717 | 100.00% | 28.76% | 99.58% | 0.0140 | 0.0185 | 0.35 | **VERIFIED_REUSED** |
| `OpenLORIS-Scene_corridor1-4` | `OpenLORIS-Scene_corridor1-4` | 1500 | 100.00% | 59.60% | 99.93% | 1.6394 | 1.5491 | 13.18 | **VERIFIED_REUSED** |
| `rgbd_bonn_crowd` | `rgbd_bonn_crowd` | 927 | 100.00% | 99.04% | 99.68% | 0.4993 | 0.5986 | 15.00 | **VERIFIED_REUSED** |
| `rgbd_bonn_moving_obstructing_box` | `rgbd_bonn_moving_obstructing_box` | 589 | 100.00% | 99.16% | 99.83% | 0.5565 | 0.3513 | 19.21 | **VERIFIED_REUSED** |
| `my_desk_recording_01` | `my_desk_recording_01` | 4980 | 100.00% | N/A | N/A | N/A (GT-free) | N/A | N/A | **VERIFIED_REUSED** |
| `my_desk_recording_02` | `my_desk_recording_02` | 5283 | 100.00% | N/A | N/A | N/A (GT-free) | N/A | N/A | **VERIFIED_REUSED** |

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
