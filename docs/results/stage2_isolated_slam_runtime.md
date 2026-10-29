# Stage 2: Isolated & Configurable SLAM Runtime

**Date**: 2026-10-29  
**Specification**: Stage 2 of [Research Implementation Audit](file:///c:/Users/Yash%20Bhardwaj/Desktop/vlm/docs/research_implementation_audit.md#L120-L132)  
**Package**: [`src/estimator_aware_vlm/slam/`](file:///c:/Users/Yash%20Bhardwaj/Desktop/vlm/src/estimator_aware_vlm/slam/)  
**Test Suite**: [`tests/test_slam_runtime.py`](file:///c:/Users/Yash%20Bhardwaj/Desktop/vlm/tests/test_slam_runtime.py) (12/12 passed), Total Suite: 29/29 passed  

---

## 1. Executive Summary

Stage 2 replaces fragile, hardcoded SLAM execution scripts with an isolated, configurable, and strictly fail-closed SLAM runtime. It addresses three critical operational and scientific validity risks identified during the research audit:

1. **Configurable Runtime Environment**: Eliminates hardcoded Linux paths (`/home/saturn/ORB_SLAM3/...`) and environment assumptions. All binary paths, vocabulary files, sensor calibration files, and WSL distributions/users are configurable through explicit CLI flags, environment variables, or standardized fallbacks.
2. **Strict Workspace Isolation**: Every SLAM run (both Pass A baseline and Pass B filtered) executes within an isolated experiment directory (`outputs/experiments/<experiment_id>/`). Association files and dynamic-masked images are created exclusively within the experiment workspace, guaranteeing zero mutation of read-only raw data directories (`data/raw/*`).
3. **Fail-Closed Integrity & Stale File Quarantine**: Pre-existing trajectory files (`CameraTrajectory.txt`, `KeyFrameTrajectory.txt`) are automatically purged before process invocation to prevent silent ingestion of stale runs. Non-zero exit codes, missing trajectories, and empty trajectories (0 valid poses) raise explicit, traceable exceptions (`SLAMExecutionError`, `EmptyTrajectoryError`).
4. **Cryptographic Run Manifests**: Every experiment exports an auditable `run_manifest.json` recording input files, output artifacts, SHA-256 hashes, parameters, platform metadata, and git commit provenance.

---

## 2. Architecture & Components

```
src/estimator_aware_vlm/slam/
├── __init__.py           # Unified exports for SLAM runtime and experiment management
├── runtime.py            # SLAMRuntimeConfig, ORBSLAM3Runner, to_wsl_path, Exception hierarchy
└── experiments.py        # create_experiment_dir, export_run_manifest, SHA-256 hashing
```

### 2.1 SLAMRuntimeConfig
Provides hierarchical resolution of SLAM execution parameters:

$$\text{CLI Argument} \succ \text{Specific Env Var} \succ \text{Root Env Var} \succ \text{Sensible Fallback}$$

- `orb_binary`: `--orb-binary` $\to$ `ORB_SLAM3_BIN` $\to$ `$ORB_SLAM3_ROOT/Examples/RGB-D/rgbd_tum` $\to$ `/home/saturn/ORB_SLAM3/Examples/RGB-D/rgbd_tum`
- `orb_vocab`: `--orb-vocab` $\to$ `ORB_SLAM3_VOCAB` $\to$ `$ORB_SLAM3_ROOT/Vocabulary/ORBvoc.txt` $\to$ `/home/saturn/ORB_SLAM3/Vocabulary/ORBvoc.txt`
- `settings_path`: `--settings` $\to$ `ORB_SLAM3_SETTINGS` $\to$ Dataset Sensor Config (e.g., `configs/sensors/OpenLORIS_D400.yaml`)
- `wsl_distro`: `--wsl-distro` $\to$ `WSL_DISTRO_NAME` $\to$ `Ubuntu-22.04`
- `wsl_user`: `--wsl-user` $\to$ `WSL_USER` $\to$ `saturn`
- `timeout_sec`: `--timeout` $\to$ Default 900 seconds
- `fail_closed`: Boolean flag enforcing strict exception raising on error.

### 2.2 ORBSLAM3Runner
Encapsulates safe subprocess invocation and lifecycle validation:
- `purge_stale_trajectories(output_dir)`: Unlinks pre-existing `CameraTrajectory.txt` and `KeyFrameTrajectory.txt` before process launch.
- `to_wsl_path(path)`: Maps Windows drive letters (`C:\...` $\to$ `/mnt/c/...`) and converts separators.
- `run(...)`:
  - Enforces association file existence.
  - Captures combined stdout and stderr to `orbslam3_execution.log`.
  - Validates exit code: raises `SLAMExecutionError` on returncode $\ne 0$.
  - Validates output file existence: raises `SLAMExecutionError` if `CameraTrajectory.txt` is missing.
  - Validates non-empty poses: raises `EmptyTrajectoryError` if parsed pose count is 0.

### 2.3 Isolated Experiment Management
- `create_experiment_dir(experiment_id, prefix, base_dir)`: Allocates dedicated run directory under `outputs/experiments/<experiment_id>/`.
- `export_run_manifest(...)`: Generates cryptographic provenance recording git commit, timestamp, environment, inputs, outputs, and SHA-256 digests.

---

## 3. Script Refactoring

### [`scripts/run_baseline_slam.py`](file:///c:/Users/Yash%20Bhardwaj/Desktop/vlm/scripts/run_baseline_slam.py)
- Ingests dataset profiles via `get_dataset_adapter()`.
- Generates `associations.txt` inside the isolated experiment directory, avoiding touching `data/raw/`.
- Uses `ORBSLAM3Runner` with fail-closed checks.
- Generates `baseline_evaluation_metrics.json` and `run_manifest.json`.
- Maintains legacy mirror under `outputs/baseline_slam/` for backwards compatibility.

### [`scripts/run_pass_b_slam.py`](file:///c:/Users/Yash%20Bhardwaj/Desktop/vlm/scripts/run_pass_b_slam.py)
- Places generated masked images in `<experiment_dir>/masked_pass_b/` instead of `data/raw/*`.
- Uses relative path redirection in `associations_pass_b.txt` so ORB-SLAM3 resolves masked frames without writing to raw sequence directories.
- Caches mask generation deterministically using `associations_pass_b.manifest.json`.
- Exports `mask_causality_report.json`, `pass_b_evaluation_metrics.json`, `trajectory_improvement_comparison.json`, and `run_manifest.json`.

---

## 4. Automated Verification Results

All tests pass cleanly in pytest:

```text
tests/test_dataset_contracts.py  ...........                               [ 37%]
tests/test_tum_adapter.py        ......                                    [ 58%]
tests/test_slam_runtime.py       ............                              [100%]
============================== 29 passed in 14.56s ==============================
```

| Test Case | Responsibility | Status |
| :--- | :--- | :---: |
| `test_slam_runtime_config_defaults` | Default binary, vocabulary, and WSL settings | PASSED |
| `test_slam_runtime_config_env_overrides` | Environment variable resolution (`ORB_SLAM3_ROOT`, etc.) | PASSED |
| `test_slam_runtime_config_cli_precedence`| CLI argument override precedence | PASSED |
| `test_to_wsl_path` | Drive letter mapping and POSIX path handling | PASSED |
| `test_purge_stale_trajectories` | Deletion of pre-existing trajectories prior to run | PASSED |
| `test_runner_missing_associations_fails`| Fail-closed on missing association file | PASSED |
| `test_runner_fail_closed_on_nonzero_exit`| Fail-closed exception on crash / returncode $\ne 0$ | PASSED |
| `test_runner_fail_closed_on_missing_trajectory`| Detection of missing trajectory output | PASSED |
| `test_runner_fail_closed_on_empty_trajectory`| Detection of 0 valid poses in output file | PASSED |
| `test_runner_successful_trajectory_generation`| Successful execution and pose parsing | PASSED |
| `test_create_experiment_dir` | Experiment directory sandboxing | PASSED |
| `test_export_run_manifest` | Cryptographic SHA-256 artifact manifest generation | PASSED |

