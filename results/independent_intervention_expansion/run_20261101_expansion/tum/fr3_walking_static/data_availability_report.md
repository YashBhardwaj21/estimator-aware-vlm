# Data Availability and Status Report: TUM fr3_walking_static

## 1. Sequence Identification
- **Identifier**: `rgbd_dataset_freiburg3_walking_static`
- **Benchmark Family**: TUM RGB-D Benchmark (Dynamic Objects Category)
- **Official Source**: Computer Vision Group, Technical University of Munich (TUM)
- **Status**: `BLOCKED_BY_MISSING_INPUT`

## 2. Sequence Characteristics
- **Expected Trajectory Duration**: ~25.2 seconds
- **Expected RGB-D Frame Count**: ~745 frames
- **Ground-Truth Path Length**: ~0.282 meters (nearly stationary tripod configuration)
- **Primary Dynamic Contamination**: Two individuals walking and interacting in front of a static office desk.

## 3. Provenance and Local Availability Audit
An inspection of the workspace `data/raw/` reveals:
- `data/raw/rgbd_dataset_freiburg3_walking_xyz` (Present, 859 frames)
- `data/raw/rgbd_dataset_freiburg3_walking_halfsphere` (Present, 1021 frames)
- `data/raw/rgbd_dataset_freiburg3_walking_static` (**Missing from local filesystem**)

In accordance with project constraints (zero data fabrication, fail-closed reproducibility, and adherence to local dataset boundary), `rgbd_dataset_freiburg3_walking_static` is formally classified as `BLOCKED_BY_MISSING_INPUT`.

## 4. Experimental Alignment and Substitution
The research roadmap required validating dynamic-object intervention on an additional TUM sequence. `rgbd_dataset_freiburg3_walking_halfsphere` is present locally, offering:
- 1021 synchronized RGB-D pairs (significantly longer than `static`)
- Moving 6-DoF half-sphere camera trajectory (testing ego-motion compensation far more rigorously than a stationary 0.28m tripod mount)
- Full high-frequency MoCap ground truth (3582 poses)
- Dual walking human actors (`person_01`, `person_02`)

All multi-sequence validation objectives for TUM dynamic environments are therefore executed on `freiburg3_walking_halfsphere`.

