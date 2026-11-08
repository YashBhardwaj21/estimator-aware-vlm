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

Test query execution against the OpenAI-compatible completion endpoint:
```powershell
curl http://127.0.0.1:8089/v1/chat/completions `
  -H "Content-Type: application/json" `
  -d '{"messages":[{"role":"user","content":"Is an office desk movable or immovable? Answer with one word."}],"max_tokens":10}'
```

---

## 3. End-to-End Command-Line Pipeline Workflows

The research evaluation pipeline executes across four decoupled functional modules:

```
[Raw Sequence] ──> Pass A SLAM ──> Perception & Tracking ──> Routing & VLM ──> Pass B SLAM ──> Figures
```

### 1. Run Baseline SLAM (Control Condition)
Executes ORB-SLAM3 in headless mode on raw, unmasked RGB-D frames:

```powershell
# Run baseline SLAM on single sequence
python scripts/run_baseline_slam.py --sequence tum_xyz

# Execute full baseline campaign across all eight benchmark sequences
python scripts/run_baseline_slam.py --all
```
**Outputs**: Written to `results/baselines/<sequence_name>/` and `results/baselines/baseline_campaign_summary.json`:
* `CameraTrajectory.txt`: Raw estimated camera trajectory in TUM format.
* `KeyFrameTrajectory.txt`: Keyframe trajectory.
* `MapPoints.ply`: 3D sparse SLAM map points.
* `associations.txt`: Exact RGB-D frame association pairings.
* `camera_settings.yaml`: Exact sensor calibration parameters.
* `orbslam3_execution.log`: Full stdout/stderr execution log.
* `baseline_evaluation_metrics.json`: Trajectory error metrics against ground truth.
* `run_manifest.json`: Cryptographic run manifest with SHA-256 digests.

### 2. Run Open-Vocabulary Perception & 3D Tracking
Executes instance segmentation (YOLOv8-Seg / YOLOE), 3D metric back-projection, and multi-frame causal tracking:

```powershell
python scripts/run_perception_pipeline.py --sequence rgbd_dataset_freiburg3_walking_xyz
```
**Outputs**: Written to `outputs/perception_tracks/`:
* `tracks_summary.json`: Detailed 3D entity trajectory statistics and persistence metrics.
* `perception_crops/`: Materialized per-instance multi-view visual crops and scene contexts.

### 3. Run Controlled VLM Evaluation
Evaluates candidate prompt configurations (V1 through V4) against development split tracks:

```powershell
python scripts/run_controlled_vlm_evaluation.py
```
**Outputs**: Written to `results/vlm_evaluation/run_20261101_193504/`:
* `inference_records.jsonl`: Full prompt, completion, latency, and token accounting for 120 evaluations.
* `evaluation_metrics.json`: Per-configuration balanced accuracy and precision metrics.
* `evaluation_summary.md`: Human-readable evaluation report.

### 4. Run Native Estimator Factorial Experiment
Executes the native ORB-SLAM3 factorial matrix (I0–I3) with live VLM queries and EDRR gating:

```powershell
python scripts/run_native_estimator_experiment.py
```
**Outputs**: Written to `results/native_estimator_integration/run_20261101_native/`:
* `native_paired_factorial_matrix.csv`: 4-mode factorial comparison table.
* `live_vlm_query_records.jsonl`: Audited live VLM interaction logs.
* `telemetry_I*/`: Complete native telemetry logs for each factorial mode.

### 5. Generate Canonical Publication Figures
Renders publication-grade figures directly from canonical empirical results:

```powershell
python scripts/generate_canonical_figures.py
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

## 4. Policy Benchmark, Evidence Fusion & Statistical Analysis

### 4.1 Run Query Selection Policy Benchmark
Evaluates P0–P7 query selection policies across 7 query budgets (1% to 100%):

```powershell
python scripts/run_query_selection_benchmark.py
```
**Outputs**: Written to `results/query_policy/`:
* `benchmark_summary.json`: Detailed recall, precision, and query count per policy and budget.
* `query_decisions.jsonl`: Per-observation routing decisions.
* `policy_comparison_table.md`: Camera-ready comparison table.

### 4.2 Run Decision & Evidence Fusion
Executes Bayesian log-odds fusion combining semantic priors, geometric evidence, and VLM classifications:

```powershell
python scripts/run_decision_evidence_fusion.py
```
**Outputs**: Written to `results/decision_fusion/`:
* `fused_interventions.jsonl`: 885 causal observation intervention decisions.
* `fusion_metrics.json`: Track-level and observation-level precision, recall, and F1 scores.
* `fusion_report.md`: Comprehensive fusion performance audit.

### 4.3 Run Statistical Significance & Ablation Study
Evaluates paired McNemar test against P0 baseline and runs four component ablations (A1–A4):

```powershell
python scripts/run_statistical_significance_and_ablation.py
```
**Outputs**: Written to `results/statistical_analysis/`:
* `statistical_significance_report.json`: McNemar contingency table ($b=24, c=0, p=0.0001$).
* `ablation_summary.json`: Sensitivity analysis for geometric, temporal, and risk terms.

---

## 5. Pipeline Data Preparation & Verification Workflows

### 5.1 Dataset Integrity Preflight
Verify all 8 dataset sequences, timestamps, and RGB-D pair alignments:

```powershell
python scripts/check_dataset_integrity.py
```

### 5.2 Perception Harvesting & Crop Extraction
Extract candidate tracks and visual crop assets across all sequences:

```powershell
python scripts/run_unified_perception_pipeline.py
python scripts/extract_candidate_crops.py
```
**Outputs**: Written to `results/perception_harvesting/run_20261101_234859/`.

### 5.3 Track Annotation & Evaluation Split
Build ground-truth annotation dataset and 4-partition split:

```powershell
python scripts/build_track_annotation_dataset.py
python scripts/build_ego_motion_provenance_and_split.py
```
**Outputs**: Written to `results/track_annotation/`.

### 5.4 Estimator Context Features
Build 36-dimensional causal feature vectors for all 885 observations:

```powershell
python scripts/build_estimator_context_features.py
```
**Outputs**: Written to `results/estimator_context/feature_vectors.jsonl`.

---

## 6. Automated Test Suite Execution

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

---

## 7. Native SLAM Telemetry & Factorial Integration Workflows

### 7.1 Execute Native C++ Dynamic Landmark Unit Tests
Verify the 10 core native landmark management and observation weighting tests in WSL:

```bash
bash native/build_and_run_tests.sh
```

### 7.2 Run Factorial SLAM & EDRR Benchmark
Ensure the live VLM server is running on port 8089, then execute:

```bash
python scripts/run_native_estimator_experiment.py
```

Outputs are exported directly to `results/native_estimator_integration/run_20261101_native/`:
- `native_paired_factorial_matrix.csv`: 4-mode factorial comparison table.
- `live_vlm_query_records.jsonl`: Audited live VLM interaction logs with token and latency metrics.
- `telemetry_I*/`: Complete native telemetry logs (`frame_feature_associations.jsonl`, `landmark_support.jsonl`, `optimizer_observations.jsonl`, `optimizer_summary.json`).
- `NATIVE_SLAM_INTEGRATION_REPORT.md`: Comprehensive factorial analysis report.
- `EDRR_GROUPED_VALIDATION_REPORT.md`: EDRR formulation and data sufficiency gate report.

### 7.3 Verify Release Integrity and Provenance Checksums
Generate and verify cryptographic SHA-256 digests across all canonical pipeline assets:

```bash
python scripts/generate_release_checksums.py
```
