# Vision-Language Model (VLM) Subsystem for Dynamic 3D Scene Graphs

## Overview

The VLM subsystem provides selective multimodal reasoning for physical object affordances and landmark lifecycle management in changing indoor environments.

In dynamic indoor SLAM, objects can be stationary, physically relocated between sessions, or moving dynamically. Constantly querying large vision-language models for every detected object is computationally prohibitive and prone to hallucination on simple static structures. Conversely, relying solely on static semantic dictionaries fails when objects of the same class have different affordances (for example, a wall-mounted monitor versus a portable tablet, or a stationary chair versus one moved across the room).

This module solves that trade-off using **estimator-aware selective routing**:
1. Inexpensive pre-router extractors evaluate geometric consistency, temporal persistence, and semantic entropy from RGB-D frames and 2D bounding boxes without invoking foundation models.
2. When spatial displacement or affordance ambiguity is detected, an expected query utility $U(q)$ is computed against inference cost.
3. Only ambiguous objects with high relevance to the SLAM estimator are escalated to the Vision-Language Model (Qwen2.5-VL-3B).
4. The resulting affordance and motion classifications update the persistent 3D scene graph, decoupling historical poses for moved objects and purging dynamic agents from the map.

---

## Directory Structure

```
vlm/
├── README.md               # This file: overview, setup, and quick start
├── requirements.txt        # Python package dependencies
├── recording/              # Live sensor data capture, D455 diagnostics, and ROS bag replay
├── slam/                   # Visual SLAM trajectory tracking, evaluation, and estimation configs
├── mapping/                # 2D/3D semantic mapping, ground plane alignment, and multi-session fusion
├── navigation/             # Autonomous A* path planning, L298N chassis control, and web visualizer
├── configs/                # Sensor, runtime, estimation, and navigation configuration YAMLs
├── tools/                  # Evaluation benchmarks, diagnostics, and reports
├── docs/                   # Core technical documentation
│   ├── architecture.md     # System architecture, components, and data flow
│   ├── data.md             # Dataset files, manifests, and schemas
│   ├── implementation.md   # Source code modules, scripts, and inputs/outputs
│   ├── usage.md            # Execution commands and workflows
│   └── evaluation.md       # Evaluation protocol, metrics, and benchmark results
├── src/                    # Core library implementation
│   ├── scene_graph/        # Full spatial engine: data discovery, geometry, YOLOE, tracking, graph
│   ├── geometric_evidence.py
│   ├── temporal_evidence.py
│   ├── semantic_evidence.py
│   ├── stage5_adapter.py
│   ├── router_policy.py
│   ├── selective_router.py
│   ├── cross_session_association.py
│   ├── contextual_scene_graph_fusion.py
│   └── statistical_analysis.py
├── scripts/                # Execution, evaluation, and audit CLI scripts
├── benchmarks/             # Versioned benchmark inputs and ground truth manifests
├── data/                   # Self-contained sensor observations and crop assets (preserved)
├── outputs/                # Benchmark run logs and final report deliverables (preserved)
└── tests/                  # Automated pytest regression suite (232 tests)
```

---

## Setup

### Requirements
- **Python**: 3.10+ (tested on Python 3.10 and 3.13)
- **Platform**: Linux / WSL2 (Ubuntu 22.04 recommended) or Windows
- **VLM Inference Engine**: `llama.cpp` (`llama-server`) with `Qwen2.5-VL-3B-Instruct-GGUF`

### Installation
From the repository root, install Python dependencies:
```bash
pip install -r vlm/requirements.txt
```

### Local VLM Server (Qwen2.5-VL-3B)
The subsystem connects to an OpenAI-compatible server running locally via `llama-server`:
```bash
# Start llama-server on port 8089
llama-server \
  -m /path/to/Qwen2.5-VL-3B-Instruct-Q4_K_M.gguf \
  --mmproj /path/to/mmproj-Qwen2.5-VL-3B-Instruct-Q8_0.gguf \
  --host 127.0.0.1 --port 8089 -c 4096 -t 4
```

Verify that the server is responding:
```bash
curl http://127.0.0.1:8089/health
```

---

## Quick Start

### 1. Run Automated Test Suite
Verify that all unit, integration, and regression tests pass:
```bash
pytest vlm/tests/ -q
```
Expected output: `232 passed`.

### 2. Run Deterministic Policy Replay Benchmark
Evaluate all baseline policies (B0–B5) and the proposed estimator-aware router (B6) against the 53-entity benchmark dataset using pre-cached visual inferences:
```bash
python vlm/scripts/run_paper_benchmark.py --mode policy-replay
```
Outputs are written to `vlm/outputs/benchmark_runs/<run_id>/`.

### 3. Run Live VLM Visual Evaluation
Run live visual inference through the local Qwen server across the 42-instance visual dataset:
```bash
python vlm/scripts/run_paper_benchmark.py --mode visual-live --server-url http://127.0.0.1:8089
```

---

## Documentation Links

- [Architecture](docs/architecture.md): Component breakdown, coordinate frames, SE(2) registration, and utility formulation.
- [Data & Manifests](docs/data.md): Observation records, visual and router manifests, ground truth schemas, and isolation rules.
- [Implementation](docs/implementation.md): Source modules, CLI scripts, dependencies, inputs, and outputs.
- [Usage Guide](docs/usage.md): Step-by-step commands for running servers, benchmarks, ablations, and tests.
- [Evaluation & Results](docs/evaluation.md): Benchmark metrics, baseline policies, empirical results, and scope boundaries.
