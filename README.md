# Estimator-Aware Vision-Language Model (VLM) for Dynamic 3D Scene Graphs

[![CI Status](https://github.com/YashBhardwaj21/estimator-aware-vlm/actions/workflows/ci.yml/badge.svg)](https://github.com/YashBhardwaj21/estimator-aware-vlm/actions/workflows/ci.yml)
[![Python Version](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-Apache--2.0-green.svg)](LICENSE)

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
├── STRUCTURE.md                # Comprehensive component inventory and module breakdown
├── benchmarks/                 # Versioned benchmark inputs, manifests, and release bundles
│   ├── assets/                 # Multi-view RGB crop assets (42 physical instances)
│   ├── release_results/        # Frozen reproducible benchmark release bundles
│   ├── model_registry.json     # Architecture-agnostic VLM backend configuration
│   └── BENCHMARK_REPORT.md     # Official benchmark report and uncertainty reconciliation
├── configs/                    # Sensor calibration and pipeline runtime configs
├── mapping/                    # Semantic mapping, SE(2) multi-session fusion, occupancy grids
├── recording/                  # Synchronized RealSense D455 capture and rosbag conversion
├── scripts/                    # CLI execution, evaluation, and system preflight diagnostics
├── slam/                       # Visual SLAM trajectory tracking and evaluation tools
├── src/
│   ├── estimator_aware_vlm/    # Core package: routing policies, evidence adapters, models
│   └── scene_graph/            # Spatial perception engine (detection, geometry, tracking)
├── tests/                      # Automated pytest unit, regression, and integrity test suite
└── assignment/                 # Immutable historical benchmark and submission archive
```

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

### 2. Run Deterministic Policy Replay Benchmark
Evaluate baseline policies (B0–B5) and the proposed estimator-aware router (B6) against the 53-entity evaluation population using cached inferences:

```bash
python scripts/run_paper_benchmark.py --protocol policy-replay --task-mode class-conditioned
```
Outputs and cryptographic execution manifests are generated under `outputs/paper_benchmarks/<run_id>/`.

### 3. Run Live VLM Visual Evaluation
Run online multimodal inference against the 42-instance image-eligible benchmark population:

```bash
python scripts/run_paper_benchmark.py --protocol visual-live --server-url http://127.0.0.1:8089 --task-mode class-conditioned
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

Frozen benchmark results, canonical ground truth, and cryptographic manifests are archived under [`benchmarks/release_results/v1_baseline/`](benchmarks/release_results/v1_baseline/). The bundle includes:
- `release_checksums.json`: SHA-256 digests for all benchmark manifests and reports.
- `BENCHMARK_REPORT.md`: Comprehensive evaluation report with methodology, error analysis, and metric tables.
- `benchmark_summary_metrics.json`: High-level summary metrics and agreement statistics.

---

## License

This project is licensed under the Apache 2.0 License - see the [LICENSE](LICENSE) file for details.
