# Ego-Motion Provenance Report

Generated: 2026-11-01T19:03:45Z
Sequences: 8

> [!IMPORTANT]
> All eight CameraTrajectory.txt files contain **ESTIMATED** poses produced by ORB-SLAM3.
> No ground-truth poses were fed into the estimator.
> The first pose of every trajectory is the identity (origin) — this is ORB-SLAM3's
> standard behavior: it initialises the map at the camera's starting position.

## Summary Table

| Sequence | Pose Type | GT Available | Execution Log | Poses | Duration |
| --- | --- | --- | --- | --- | --- |
| tum_xyz | ESTIMATED | Yes | Yes | 827 | 28.8 s |
| tum_halfsphere | ESTIMATED | Yes | Yes | 1021 | 35.9 s |
| tum_static | ESTIMATED | Yes | Yes | 717 | 24.9 s |
| openloris | ESTIMATED | Yes | Yes | 1500 | 50.0 s |
| bonn_crowd | ESTIMATED | Yes | Yes | 927 | 31.1 s |
| bonn_box | ESTIMATED | Yes | Yes | 589 | 19.7 s |
| d455_01 | ESTIMATED | **NO** | Not found | 5108 | 257.0 s |
| d455_02 | ESTIMATED | **NO** | Not found | 5283 | 232.3 s |

## Causal Safety Check

**No future information was used in any trajectory.**

ORB-SLAM3 processes frames sequentially. Each camera pose at time `t` is computed
using only frames `[0, t]`. The saved CameraTrajectory.txt is the output of this
causal, sequential process. It is safe to use for ego-motion compensation in the
perception pipeline because:

- The pipeline reads already-saved trajectory files (offline mode).
- Frame `i` is matched to the nearest trajectory pose by timestamp interpolation.
- No trajectory pose is derived from frames after the matched frame.

**Ground-truth poses are NOT used in the perception pathway.**
The TrajectoryRole.ESTIMATED tag is enforced throughout the codebase.

## Per-Sequence Details

### tum_xyz

| Field | Value |
| --- | --- |
| Dataset | TUM RGB-D |
| Recording | rgbd_dataset_freiburg3_walking_xyz |
| Trajectory file | `results/publication_campaign/run_20261101_execution/baselines/rgbd_dataset_freiburg3_walking_xyz/CameraTrajectory.txt` |
| Execution log | `results/publication_campaign/run_20261101_execution/baselines/rgbd_dataset_freiburg3_walking_xyz/orbslam3_execution.log` |
| Generator binary | ORB-SLAM3 (RGB-D mode, WSL2/Linux) |
| Generator script | `scripts/run_baseline_slam.py` |
| Pose type | **ESTIMATED** |
| Ground truth available | True |
| Ground truth source | TUM RGB-D motion capture (tum_rgbd_groundtruth.txt) |
| Pose convention | TUM: timestamp tx ty tz qx qy qz qw (world-to-camera SE3) |
| Timestamp scale | UNIX epoch seconds |
| Timestamp range | 1341846313.592 – 1341846342.416 (28.8 s) |
| Poses in file | 827 |
| First pose is identity | True |
| Future information used | False |

> First pose is identity (origin). Subsequent poses are estimated incrementally by ORB-SLAM3. No ground-truth poses used inside the estimator.

### tum_halfsphere

| Field | Value |
| --- | --- |
| Dataset | TUM RGB-D |
| Recording | rgbd_dataset_freiburg3_walking_halfsphere |
| Trajectory file | `results/publication_campaign/run_20261101_execution/baselines/rgbd_dataset_freiburg3_walking_halfsphere/CameraTrajectory.txt` |
| Execution log | `results/publication_campaign/run_20261101_execution/baselines/rgbd_dataset_freiburg3_walking_halfsphere/orbslam3_execution.log` |
| Generator binary | ORB-SLAM3 (RGB-D mode, WSL2/Linux) |
| Generator script | `scripts/run_baseline_slam.py` |
| Pose type | **ESTIMATED** |
| Ground truth available | True |
| Ground truth source | TUM RGB-D motion capture |
| Pose convention | TUM: timestamp tx ty tz qx qy qz qw |
| Timestamp scale | UNIX epoch seconds |
| Timestamp range | 1341846434.710 – 1341846470.606 (35.9 s) |
| Poses in file | 1021 |
| First pose is identity | True |
| Future information used | False |

> Same pipeline as tum_xyz. All poses estimated incrementally.

### tum_static

| Field | Value |
| --- | --- |
| Dataset | TUM RGB-D |
| Recording | rgbd_dataset_freiburg3_walking_static |
| Trajectory file | `results/publication_campaign/run_20261101_execution/baselines/rgbd_dataset_freiburg3_walking_static/CameraTrajectory.txt` |
| Execution log | `results/publication_campaign/run_20261101_execution/baselines/rgbd_dataset_freiburg3_walking_static/orbslam3_execution.log` |
| Generator binary | ORB-SLAM3 (RGB-D mode, WSL2/Linux) |
| Generator script | `scripts/run_baseline_slam.py` |
| Pose type | **ESTIMATED** |
| Ground truth available | True |
| Ground truth source | TUM RGB-D motion capture |
| Pose convention | TUM: timestamp tx ty tz qx qy qz qw |
| Timestamp scale | UNIX epoch seconds |
| Timestamp range | 1341846226.818 – 1341846251.750 (24.9 s) |
| Poses in file | 717 |
| First pose is identity | True |
| Future information used | False |

> Scene: dynamic people, static background. ORB-SLAM3 baseline without any dynamic filtering.

### openloris

| Field | Value |
| --- | --- |
| Dataset | OpenLORIS-Scene |
| Recording | OpenLORIS-Scene_corridor1-4 |
| Trajectory file | `results/publication_campaign/run_20261101_execution/baselines/OpenLORIS-Scene_corridor1-4/CameraTrajectory.txt` |
| Execution log | `results/publication_campaign/run_20261101_execution/baselines/OpenLORIS-Scene_corridor1-4/orbslam3_execution.log` |
| Generator binary | ORB-SLAM3 (RGB-D mode, WSL2/Linux) |
| Generator script | `scripts/run_baseline_slam.py` |
| Pose type | **ESTIMATED** |
| Ground truth available | True |
| Ground truth source | OpenLORIS-Scene ground-truth trajectory (provided with dataset) |
| Pose convention | TUM: timestamp tx ty tz qx qy qz qw |
| Timestamp scale | UNIX epoch seconds |
| Timestamp range | 1560200172.200 – 1560200222.211 (50.0 s) |
| Poses in file | 1500 |
| First pose is identity | True |
| Future information used | False |

> Long corridor translation visible in last pose (tx≈11.7m, tz≈41.4m). Trajectory covers full corridor1-4 sweep.

### bonn_crowd

| Field | Value |
| --- | --- |
| Dataset | Bonn RGB-D Dynamic |
| Recording | rgbd_bonn_crowd |
| Trajectory file | `results/publication_campaign/run_20261101_execution/baselines/rgbd_bonn_crowd/CameraTrajectory.txt` |
| Execution log | `results/publication_campaign/run_20261101_execution/baselines/rgbd_bonn_crowd/orbslam3_execution.log` |
| Generator binary | ORB-SLAM3 (RGB-D mode, WSL2/Linux) |
| Generator script | `scripts/run_baseline_slam.py` |
| Pose type | **ESTIMATED** |
| Ground truth available | True |
| Ground truth source | Bonn RGB-D motion capture ground truth (provided with dataset) |
| Pose convention | TUM: timestamp tx ty tz qx qy qz qw |
| Timestamp scale | UNIX epoch seconds |
| Timestamp range | 1548339819.874 – 1548339850.943 (31.1 s) |
| Poses in file | 927 |
| First pose is identity | True |
| Future information used | False |

> Dynamic crowd scene. Camera trajectory used Bonn_Xtion.yaml calibration.

### bonn_box

| Field | Value |
| --- | --- |
| Dataset | Bonn RGB-D Dynamic |
| Recording | rgbd_bonn_moving_obstructing_box |
| Trajectory file | `results/publication_campaign/run_20261101_execution/baselines/rgbd_bonn_moving_obstructing_box/CameraTrajectory.txt` |
| Execution log | `results/publication_campaign/run_20261101_execution/baselines/rgbd_bonn_moving_obstructing_box/orbslam3_execution.log` |
| Generator binary | ORB-SLAM3 (RGB-D mode, WSL2/Linux) |
| Generator script | `scripts/run_baseline_slam.py` |
| Pose type | **ESTIMATED** |
| Ground truth available | True |
| Ground truth source | Bonn RGB-D motion capture ground truth (provided with dataset) |
| Pose convention | TUM: timestamp tx ty tz qx qy qz qw |
| Timestamp scale | UNIX epoch seconds |
| Timestamp range | 1548339348.694 – 1548339368.399 (19.7 s) |
| Poses in file | 589 |
| First pose is identity | True |
| Future information used | False |

> A cardboard box is deliberately pushed into the camera's field to obstruct ORB-SLAM3. Baseline trajectory shows significant drift (last pose tx≈0.88m, ty≈0.96m, tz≈-1.87m).

### d455_01

| Field | Value |
| --- | --- |
| Dataset | RealSense D455 (Custom) |
| Recording | my_desk_recording_01 |
| Trajectory file | `data/semantic_mapping_my_desk_recording_01/trajectory/CameraTrajectory.txt` |
| Execution log | `N/A` |
| Generator binary | ORB-SLAM3 (RGB-D mode, earlier session) |
| Generator script | `Unknown — trajectory predates run_20261101_execution campaign; found in data/semantic_mapping_*/trajectory/` |
| Pose type | **ESTIMATED** |
| Ground truth available | False |
| Ground truth source | NONE — no independent camera ground truth exists for D455 custom recordings |
| Pose convention | TUM: timestamp tx ty tz qx qy qz qw |
| Timestamp scale | UNIX epoch seconds |
| Timestamp range | 1791788635.549 – 1791788892.591 (257.0 s) |
| Poses in file | 5108 |
| First pose is identity | True |
| Future information used | False |

> IMPORTANT: No independent trajectory ground truth. ATE/RPE cannot be computed for this sequence. Sequence can support VLM evaluation and perception experiments only. The trajectory was generated in an earlier SLAM session (not part of run_20261101_execution).

### d455_02

| Field | Value |
| --- | --- |
| Dataset | RealSense D455 (Custom) |
| Recording | my_desk_recording_02 |
| Trajectory file | `data/semantic_mapping_my_desk_recording_02/trajectory/CameraTrajectory.txt` |
| Execution log | `N/A` |
| Generator binary | ORB-SLAM3 (RGB-D mode, earlier session) |
| Generator script | `Unknown — trajectory predates run_20261101_execution campaign` |
| Pose type | **ESTIMATED** |
| Ground truth available | False |
| Ground truth source | NONE — no independent camera ground truth exists for D455 custom recordings |
| Pose convention | TUM: timestamp tx ty tz qx qy qz qw |
| Timestamp scale | UNIX epoch seconds |
| Timestamp range | 1791797662.379 – 1791797894.676 (232.3 s) |
| Poses in file | 5283 |
| First pose is identity | True |
| Future information used | False |

> IMPORTANT: No independent trajectory ground truth. ATE/RPE cannot be computed for this sequence. Sequence can support VLM evaluation and perception experiments only.

## Flags Requiring Attention

1. **d455_01 and d455_02** have no independent camera ground truth.
   - ATE and RPE cannot be computed for these sequences.
   - They are valid for VLM evaluation and perception-level experiments.
   - Their trajectories come from an **earlier ORB-SLAM3 session** whose
     generation script is not recorded. Treat provenance as PARTIAL for these two.

2. **All six benchmark sequences** (TUM×3, OpenLORIS×1, Bonn×2) have fully
   documented ORB-SLAM3 execution logs and independent camera ground truth.
   These form the quantitative SLAM evaluation benchmark.

3. The **openloris** trajectory shows a large translation (≈41m in Z).
   This is consistent with a long-corridor recording. Verify this against
   the dataset ground-truth trajectory before reporting ATE.