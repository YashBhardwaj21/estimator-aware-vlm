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
| `pop_52_ground_truth` | 52 | Physical objects with verified cross-session physical ground truth. | Geometric registration, displacement analysis, baseline comparison. |
| `pop_42_image_eligible`| 42 | High-confidence instances with clear multi-view crop assets (146 files). | Live VLM visual classification & affordance scoring. |
| `pop_53_router_evaluation`| 53 | 52 canonical objects + 1 structural whiteboard instance. | Estimator-aware router evaluation, cost-accuracy Pareto frontier. |

### Dual Metrics: End-to-End vs Selective Accuracy
The benchmark framework enforces strict separation between:
- **End-to-End Accuracy**: Accuracy across the entire evaluated population ($N$), penalizing bypassed entities, JSON parsing failures, or misclassifications (accuracy = 0).
- **Selective Accuracy**: Accuracy computed exclusively on the subset of entities selected and queried by the policy ($N_{\text{queried}}$), measuring pure visual model precision when escalated.

### Uncertainty Accounting
- **Physical Survey Noise Floor**: $\sigma_{\text{survey}} = \pm 0.043\text{ m}$ (calibrated physical measurement uncertainty of the ground-truth survey tools).
- **Estimator Registration Uncertainty**: $\sigma_{\text{reg}} = 0.1215\text{ m}$ (empirical inlier RMS residual error $= 0.1198\text{ m}$ of static anchor SE(2) cross-session alignment).
- **Relocation Significance Threshold**: Relocations are marked significant when measured displacement exceeds $3\sigma_{\text{combined}} = 0.3644\text{ m}$.

---

## Reproducible Release Bundles

Frozen benchmark results, canonical ground truth, and cryptographic manifests are published under [`benchmarks/release_results/v1_baseline/`](benchmarks/release_results/v1_baseline/). The bundle includes:
- `release_checksums.json`: SHA-256 digests for all benchmark manifests and reports.
- `BENCHMARK_REPORT.md`: Comprehensive evaluation report with methodology, error analysis, and metric tables.
- `benchmark_summary_metrics.json`: High-level summary metrics and agreement statistics.

---

## License

This project is licensed under the Apache 2.0 License - see the [LICENSE](LICENSE) file for details.
