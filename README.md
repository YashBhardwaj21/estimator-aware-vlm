# Estimator-Aware Vision-Language Model (VLM) for Dynamic 3D Scene Graphs


## Overview

The **Estimator-Aware Vision-Language Model (VLM)** repository provides selective multimodal reasoning for physical object affordances, state change identification, and landmark lifecycle management in changing indoor environments.

In dynamic indoor SLAM, objects can be stationary, physically relocated between sessions, or moving dynamically. Constantly querying large vision-language models for every detected object is computationally prohibitive and prone to hallucination on simple static structures. Conversely, relying solely on static semantic dictionaries fails when objects of the same class exhibit differing affordances (e.g., a wall-mounted monitor versus a portable tablet, or a stationary chair versus one moved across the room).

This repository implements **estimator-aware selective routing**:
1. **Lightweight Pre-Routing Evidence**: Inexpensive geometric consistency, temporal persistence, and semantic entropy extractors evaluate candidate objects from RGB-D frames and 2D bounding boxes without invoking foundation models.
2. **Estimator-Aware Utility Formulation**: When spatial displacement or affordance ambiguity is detected, an expected query utility $U(q)$ is computed against inference cost, factoring in estimator registration uncertainty.
3. **Selective VLM Escalation**: Only ambiguous objects with high expected information gain relative to the SLAM estimator are escalated to the Vision-Language Model (`Qwen2.5-VL-3B-Instruct`).
4. **Dynamic 3D Scene Graph Association**: Affordance and motion classifications update the persistent 3D scene graph, decoupling historical poses for moved objects and purging dynamic agents from the map.

---

## Repository Structure

```text
estimator-aware-vlm/
├── pyproject.toml              # Modern Python packaging configuration (PEP 517/621)
├── requirements.txt            # Core VLM research, benchmarks, and test dependencies
├── requirements-mapping.txt    # Additional dependencies for 3D semantic mapping
├── requirements-recording.txt  # Additional dependencies for RealSense D455 recording
├── README.md                   # Repository overview, setup, and quick start guide
├── benchmarks/                 # Versioned benchmark inputs, manifests, and release bundles
│   ├── assets/                 # Multi-view RGB crop assets (42 physical instances)
│   ├── release_results/        # Frozen reproducible benchmark release bundles
│   ├── model_registry.json     # Architecture-agnostic VLM backend configuration
│   └── BENCHMARK_REPORT.md     # Official benchmark report and uncertainty reconciliation
├── configs/                    # Sensor calibration and pipeline runtime configs
├── data/                       # Raw RGB-D sequences and processed observation datasets
├── docs/                       # Core documentation suite (6 consolidated guides)
├── mapping/                    # Semantic mapping, multi-session SE(2) fusion, occupancy grids
├── models/                     # Deep learning checkpoints (YOLOv8-Seg, YOLOE, Ultralytics)
├── outputs/                    # Experiment runs, trajectories, figure renders, and benchmarks
├── recording/                  # Synchronized RealSense D455 capture and rosbag conversion
├── scripts/                    # CLI execution, evaluation, and system preflight diagnostics
├── src/
│   ├── estimator_aware_vlm/    # Core package: routing policies, evidence adapters, models
│   └── scene_graph/            # Spatial perception engine (detection, geometry, tracking)
└── tests/                      # Automated pytest unit, regression, and integrity test suite
```

### Documentation Suite (`docs/`)

Comprehensive, self-contained documentation is organized into 6 reference documents:

| Document | Primary Focus & Contents |
| :--- | :--- |
| [`docs/system_architecture.md`](docs/system_architecture.md) | **System Architecture**: Modular pipeline breakdown, pinhole back-projection, SE(2) registration, spatial uncertainty gating, query utility formulation, and SLAM runtime isolation. |
| [`docs/setup_and_usage.md`](docs/setup_and_usage.md) | **Setup & Operations Guide**: Environment installation, preflight dependency checks, local `llama-server` configuration, and 5-step CLI pipeline workflows. |
| [`docs/dataset_inventory_and_provenance.md`](docs/dataset_inventory_and_provenance.md) | **Dataset Inventory & Provenance**: Sensor contracts for TUM RGB-D, OpenLORIS-Scene, and Multi-Session Desk; population definitions; observation schemas; and cryptographic immutability. |
| [`docs/benchmark_protocol_and_routing_policies.md`](docs/benchmark_protocol_and_routing_policies.md) | **Benchmark Protocols**: Dual-axis evaluation framework, formal definitions of policies B0–B6, empirical counterfactual utility calibration, and dual-cost latency accounting. |
| [`docs/empirical_benchmark_and_slam_results.md`](docs/empirical_benchmark_and_slam_results.md) | **Empirical SLAM Results**: Trajectory metric tables (ATE RMSE $0.4372\text{ m} \to 0.0167\text{ m}$, $-96.17\%$ error reduction, $p < 10^{-200}$), OpenLORIS results, geometric masking ablation, and figure catalog. |
| [`docs/experimental_analysis_and_limitations.md`](docs/experimental_analysis_and_limitations.md) | **Analysis & Limitations**: Validation of hypotheses H1–H5, physical case studies, claim-to-evidence matrix, and honest operational boundaries. |

---

## Setup & Installation

### Requirements
- **Python**: `>= 3.10`
- **Operating System**: Linux (Ubuntu 22.04 LTS recommended), WSL2, or Windows
- **VLM Inference Engine**: `llama-server` (from `llama.cpp`) or any OpenAI-compatible server hosting `Qwen2.5-VL-3B-Instruct`

### 1. Install Package & Dependencies
Clone the repository and install in editable mode:

```bash
git clone https://github.com/YashBhardwaj21/estimator-aware-vlm.git
cd estimator-aware-vlm

# Install core research dependencies and editable package
pip install -e .

# Or install optional mapping dependencies (Open3D, OpenCV, Ultralytics)
pip install -r requirements-mapping.txt

# Or install optional live recording dependencies (pyrealsense2, rosbags)
pip install -r requirements-recording.txt
```

### 2. Verify Environment Readiness
Run the built-in system preflight diagnostic tool:

```bash
python scripts/check_system_dependencies.py
```

### 3. Start Local VLM Server (Optional for Live Inference)
To run live visual inference with `Qwen2.5-VL-3B-Instruct`, start `llama-server`:

```bash
llama-server \
  -m /path/to/Qwen2.5-VL-3B-Instruct-Q4_K_M.gguf \
  --mmproj /path/to/mmproj-Qwen2.5-VL-3B-Instruct-Q8_0.gguf \
  --host 127.0.0.1 --port 8089 -c 4096 -t 4
```

Verify that the server endpoint is active:
```bash
curl http://127.0.0.1:8089/health
```

---

## Quick Start & Evaluation

All commands should be executed from the **repository root**:

### 1. Run Automated Test Suite
Execute the full pytest suite across packaging, path resolution, navigation decoupling, data pipelines, benchmark manifests, and router scoring metrics:

```bash
pytest tests/ -v
```

### 2. End-to-End Research Pipeline Execution
Execute the core experimental pipeline on the primary TUM RGB-D dynamic sequence:

```bash
# Baseline unmasked ORB-SLAM3 trajectory estimation
python scripts/run_baseline_slam.py --sequence rgbd_dataset_freiburg3_walking_xyz

# Open-vocabulary perception and 3D temporal tracking
python scripts/run_perception_pipeline.py --sequence rgbd_dataset_freiburg3_walking_xyz

# Comparative routing and selective VLM inference
python scripts/run_routing_and_vlm_experiment.py --sequence rgbd_dataset_freiburg3_walking_xyz --server-url http://127.0.0.1:8089

# Causal dynamic feature suppression SLAM
python scripts/run_pass_b_slam.py --sequence rgbd_dataset_freiburg3_walking_xyz --mask-type polygon
```
*For detailed CLI execution guides across OpenLORIS and multi-session runs, see [`docs/setup_and_usage.md`](docs/setup_and_usage.md).*

### 3. Run Standardized Paper Benchmark Suite
Run the master benchmark orchestrator across evaluation protocols:

```bash
# Deterministic Policy Replay (evaluates B0-B6 using verified predictions)
python scripts/run_paper_benchmark.py --protocol policy-replay

# Live Visual Benchmark (queries live VLM over HTTP)
python scripts/run_paper_benchmark.py --protocol visual-live --server-url http://127.0.0.1:8089 --view-mode CROP_PLUS_CONTEXT
```

---

## Benchmark Populations & Evaluation Protocol

| Population Identifier | Size ($N$) | Definition | Primary Use Case |
| :--- | :---: | :--- | :--- |
| `pop_42_image_eligible`| 42 | Physical instances with verified multi-view RGB crops (187 image files). | Zero-shot VLM physical movability classification & affordance scoring. |
| `pop_53_router_evaluation`| 53 | Full physical entity census (52 canonical objects + 1 structural whiteboard). | Estimator-aware router evaluation, cost-accuracy Pareto frontier. |
| `pop_15_balanced_router` | 15 | Balanced benchmark manifest (3 IMMOVABLE, 9 MOVABLE, 3 MOBILE). | Controlled policy suite evaluation separating affordance from motion. |
| `pop_10_tum_candidates` | 10 | Real candidate dynamic tracks on TUM `fr3_walking_xyz`. | Downstream SLAM feature starvation and trajectory evaluation. |

### Dual Metrics: End-to-End vs Selective Accuracy
The benchmark framework enforces strict separation between:
- **End-to-End Accuracy**: Accuracy across the entire evaluated population ($N$), penalizing bypassed entities, JSON parsing failures, or misclassifications (accuracy = 0).
- **Selective Accuracy**: Accuracy computed exclusively on the subset of entities selected and queried by the policy ($N_{\text{queried}}$), measuring pure visual model precision when escalated.

### Uncertainty Accounting & Analytical Bounds
- **Physical Survey Noise Floor**: $\sigma_{\text{survey}} = \pm 0.043\text{ m}$ (calibrated physical measurement uncertainty of the survey tools).
- **Consensus Estimator Registration**: $\theta = -15.558^\circ$, $t = [-0.1059, 0.0486]^\top\text{ m}$ with scale $s = 0.0955\text{ m}$ across 7 consensus anchors (inlier RMS residual $= 0.1198\text{ m}$).
- **First-Order Propagated Covariance**: $\sigma_\theta = 0.7363^\circ, \sigma_{t_x} = 57.49\text{ mm}, \sigma_{t_y} = 37.76\text{ mm}$.
- **Relocation Significance**: Verified relocations (`inst_chair_002`: $0.563\text{ m}$, `inst_chair_003`: $0.386\text{ m}$, `inst_storage_box_007`: $0.396\text{ m}$) significantly exceed their individual $3\sigma$ confidence bounds ($p < 10^{-8}$).

---

## Reproducible Release Bundles

Canonical verified benchmark results, ground truth manifests, and cryptographic checksums are published under [`benchmarks/release_results/v2_verified/`](benchmarks/release_results/v2_verified/). The release bundle includes:
- `release_checksums.json`: Cryptographic SHA-256 digests locking all 9 release assets.
- `VERIFIED_EMPIRICAL_AUDIT_REPORT.md`: Comprehensive evaluation report with empirical proofs, trajectory metrics, and baseline comparisons.
- `verified_empirical_metrics.json`: Complete machine-readable empirical benchmark metrics.
- `canonical_benchmark_manifest.json`: Authoritative physical entity census and population definitions.
- `router_evaluation_manifest.json`: Balanced 15-instance benchmark manifest.
- `paper_benchmark_inputs_expanded.json` & `paper_benchmark_ground_truth_expanded.json`: 42-instance inputs and isolated ground-truth labels.
- `outputs/figures/`: Publication-ready Figures 1 through 10 generated directly from verified execution data.

---

## License

This project is licensed under the Apache 2.0 License - see the [LICENSE](LICENSE) file for details.
