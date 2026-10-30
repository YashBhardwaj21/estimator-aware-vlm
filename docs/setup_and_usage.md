# Setup, CLI Workflows, and Operations Guide

## 1. System Requirements & Environment Setup

### 1.1 Prerequisites
* **Operating System**: Windows 11 with WSL2 (Ubuntu 22.04) or native Linux (Ubuntu 20.04/22.04).
* **Python**: Python 3.10 to Python 3.13 (Python 3.13.7 verified).
* **WSL2 Runtime**: Ubuntu 22.04 instance with compiled ORB-SLAM3 executable:
  `/home/saturn/ORB_SLAM3/Examples/RGB-D/rgbd_tum`
* **GPU Hardware**: NVIDIA GPU with CUDA 12.x for accelerated local VLM inference (optional; CPU execution supported via `llama.cpp`).

### 1.2 Python Environment Setup
From the repository root:

```powershell
# Create and activate virtual environment
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# Install core package in editable mode with all dependencies
pip install -e .
pip install -r requirements.txt
pip install -r requirements-mapping.txt
```

### 1.3 Automated Preflight Dependency Check
Run the automated preflight diagnostics tool to verify Python version, PyTorch CUDA availability, OpenCV, directory layout, and dataset integrity:

```powershell
python scripts/check_system_dependencies.py
```

Expected diagnostic output:
```text
================================================================================
ESTIMATOR-AWARE VLM: SYSTEM DEPENDENCY & PREFLIGHT CHECK
================================================================================
[PASS] Python Version: 3.13.x
[PASS] PyTorch Installation: 2.x
[PASS] OpenCV Version: 4.x
[PASS] Canonical Path Resolution: OK
[PASS] Model Weights & Checkpoints: OK
[PASS] Core Package Imports: estimator_aware_vlm, scene_graph: OK
================================================================================
```

---

## 2. Serving the Local Vision-Language Model

The framework uses **Qwen2.5-VL-3B-Instruct** served via `llama.cpp` (`llama-server`) exposing an OpenAI-compatible HTTP API on port 8089.

### 2.1 Launching llama-server

```bash
llama-server \
  -m models/Qwen2.5-VL-3B-Instruct-Q4_K_M.gguf \
  --mmproj models/mmproj-Qwen2.5-VL-3B-Instruct-Q8_0.gguf \
  --host 127.0.0.1 \
  --port 8089 \
  -c 4096 \
  -t 4 \
  --n-gpu-layers 33
```

### 2.2 Verifying Server Health
Verify the HTTP health endpoint:

```powershell
curl http://127.0.0.1:8089/health
# Expected JSON response: {"status": "ok"}
```

Test query execution via live VLM client test:
```powershell
python scripts/test_live_vlm_query.py
```

---

## 3. End-to-End Command-Line Pipeline Workflows

The research evaluation pipeline executes across four decoupled functional modules:

```
[Raw Sequence] ──> Pass A SLAM ──> Perception & Tracking ──> Routing & VLM ──> Pass B SLAM ──> Figures
```

### Step 1: Run Baseline SLAM (Pass A)
Executes ORB-SLAM3 in headless mode on raw, unmasked RGB-D frames:

```powershell
# Run baseline SLAM on primary TUM sequence
python scripts/run_baseline_slam.py --sequence rgbd_dataset_freiburg3_walking_xyz

# Run baseline SLAM on OpenLORIS corridor sequence
python scripts/run_baseline_slam.py --sequence OpenLORIS-Scene_corridor1-4
```
**Outputs**: Written to `outputs/baseline_slam/` (or `outputs/baseline_slam_<sequence>/`):
* `CameraTrajectory.txt`: Raw estimated camera trajectory in TUM format.
* `KeyFrameTrajectory.txt`: Keyframe trajectory.
* `MapPoints.ply`: 3D sparse SLAM map points.
* `baseline_evaluation_metrics.json`: Trajectory error metrics against ground truth.

### Step 2: Run Open-Vocabulary Perception & 3D Tracking
Executes instance segmentation (YOLOv8-Seg / YOLOE), 3D metric back-projection, and multi-frame causal tracking:

```powershell
python scripts/run_perception_pipeline.py --sequence rgbd_dataset_freiburg3_walking_xyz
```
**Outputs**: Written to `outputs/perception_tracks/`:
* `tracks_summary.json`: Detailed 3D entity trajectory statistics and persistence metrics.
* `perception_crops/`: Materialized per-instance multi-view visual crops and scene contexts.

### Step 3: Run Comparative Routing & Live VLM Inference
Evaluates candidate routing policies (B0 through B6) against candidate tracks:

```powershell
python scripts/run_routing_and_vlm_experiment.py \
  --sequence rgbd_dataset_freiburg3_walking_xyz \
  --server-url http://127.0.0.1:8089 \
  --vlm-cost 0.50
```
**Outputs**: Written to `outputs/vlm_inference/`:
* `policy_comparison.json`: Comprehensive cross-policy benchmark table with token accounting.
* `routing_decisions.json`: Calibrated B6 routing decisions specifying which entities are suppressed in tracking.

### Step 4: Run Causal SLAM Mitigation (Pass B)
Generates dynamic instance suppression masks (tight polygon contours) and re-runs ORB-SLAM3:

```powershell
python scripts/run_pass_b_slam.py \
  --sequence rgbd_dataset_freiburg3_walking_xyz \
  --mask-type polygon
```
**Outputs**: Written to `outputs/pass_b_slam/`:
* `CameraTrajectory.txt`: Filtered Pass B estimated camera trajectory.
* `MapPoints.ply`: Cleaned static SLAM map points.
* `mask_causality_report.json`: Cryptographic audit logging suppressed pixel counts and frame indices.
* `trajectory_improvement_comparison.json`: Paired before/after trajectory error deltas and percentage gains.

### Step 5: Generate Publication Figures
Renders publication-grade figures (Figures 2 through 10) directly from live empirical output files:

```powershell
python scripts/generate_paper_figures.py
```
**Outputs**: Written to `outputs/figures/`:
* `fig2_pangolin_3d_slam_visualization.png`: Side-by-side Pass A vs Pass B map point comparison.
* `fig3_trajectory_evaluation_plot.png`: 3D and top-down $XY$ trajectory error plots.
* `fig4_rgbd_dynamic_masking_example.png`: 4-panel RGB, Depth, Polygon Mask, and Masked Frame.
* `fig5_semantic_3d_scene_graph.png`: 3D metric scene graph with physical mobility states.
* `fig6_scene_map_before_after_comparison.png`: 2D occupancy grid showing ghost point elimination.
* `fig7_object_displacement_uncertainty.png`: Entity displacement time-series with $3\sigma$ decision bounds.
* `fig8_movability_confusion_matrix.png`: Mobility classification confusion matrix.
* `fig9_routing_policy_comparison.png`: Policy comparison (B0–B6) bar charts.
* `fig10_slam_performance_query_efficiency.png`: Trajectory metric drops and Pareto efficiency curve.

---

## 4. Master Benchmark Orchestrator (`run_paper_benchmark.py`)

For standardized, reproducible evaluation, the repository provides a master benchmark runner supporting three distinct protocols:

### 4.1 Policy Replay Protocol (Deterministic Offline)
Evaluates routing policies deterministically using immutable pre-computed visual model predictions:

```powershell
python scripts/run_paper_benchmark.py --protocol policy-replay
```

### 4.2 Visual Live Protocol
Evaluates visual physical movability classification on the 42 verified instances via live over-the-wire VLM queries:

```powershell
python scripts/run_paper_benchmark.py \
  --protocol visual-live \
  --server-url http://127.0.0.1:8089 \
  --view-mode CROP_PLUS_CONTEXT
```

### 4.3 Router Live Protocol
Executes end-to-end dynamic selective routing with live VLM queries invoked on-demand:

```powershell
python scripts/run_paper_benchmark.py \
  --protocol router-live \
  --server-url http://127.0.0.1:8089
```

All benchmark runs generate a cryptographic `run_manifest.json` recording git commit SHA, model parameters, and SHA-256 digests of all generated artifacts.

---

## 5. Automated Test Suite Execution

The repository maintains an automated pytest suite covering packaging, unit mathematics, and end-to-end data pipeline flow:

```powershell
# Run the entire test suite
pytest tests/ -v

# Run fast unit tests only (excluding live network tests)
pytest tests/ -m "not live_vlm" -q

# Run specific functional areas
pytest tests/test_packaging_and_paths.py            # Import hygiene and path resolution
pytest tests/test_trajectory_metrics.py              # ATE/RPE mathematics on common support
pytest tests/test_router_policies.py                 # Policy evaluation rules (B0-B6)
pytest tests/test_benchmark_manifest_integrity.py    # Population counts and leakage checks
pytest tests/test_scene_graph_integration.py         # Contextual scene graph maintenance
```

