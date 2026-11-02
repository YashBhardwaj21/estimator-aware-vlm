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
├── configs/                    # Sensor calibration and pipeline runtime configs
├── data/                       # Raw RGB-D sequences and processed observation datasets
├── docs/                       # Core documentation suite (6 consolidated guides)
├── mapping/                    # Semantic mapping, multi-session SE(2) fusion, occupancy grids
├── models/                     # Deep learning checkpoints (YOLOv8-Seg, YOLOE, Ultralytics)
├── outputs/                    # Experiment runs, trajectories, figure renders, and benchmarks
├── recording/                  # Synchronized RealSense D455 capture and rosbag conversion
├── results/                    # Empirical benchmark and evaluation outputs
│   ├── perception_harvesting/          # 3D bounding boxes and tracking (885 obs, 177 tracks)
│   ├── track_annotation/               # Verified ground-truth annotations and evaluation split
│   ├── vlm_evaluation/                 # Controlled multi-view evaluation (V1–V4, Baselines 0A & 0B)
│   ├── policy_vlm_inference/           # 168 live observation query records
│   ├── estimator_context/              # 885 causal estimator feature vectors
│   ├── query_policy/                   # Policy routing benchmark outputs (P0–P7)
│   ├── decision_fusion/                # Bayesian log-odds evidence fusion outputs
│   ├── native_estimator_integration/   # Native ORB-SLAM3 factorial evaluation (I0–I3)
│   ├── statistical_analysis/           # Paired McNemar hypothesis tests and ablation results
│   └── publication_campaign/           # CAMERA_READY_RESEARCH_MANUSCRIPT.md
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
| [`docs/setup_and_usage.md`](docs/setup_and_usage.md) | **Setup & Operations Guide**: Environment installation, preflight dependency checks, local `llama-server` configuration, and standard CLI pipeline workflows. |
| [`docs/dataset_inventory_and_provenance.md`](docs/dataset_inventory_and_provenance.md) | **Dataset Inventory & Provenance**: Sensor contracts for TUM RGB-D, OpenLORIS-Scene, and Multi-Session Desk; population definitions; observation schemas; and cryptographic immutability. |
| [`docs/benchmark_protocol_and_routing_policies.md`](docs/benchmark_protocol_and_routing_policies.md) | **Benchmark Protocols**: Dual-axis evaluation framework, formal definitions of policies B0–B6, empirical counterfactual utility calibration, and dual-cost latency accounting. |
| [`docs/empirical_benchmark_and_slam_results.md`](docs/empirical_benchmark_and_slam_results.md) | **Empirical SLAM Results**: Native ORB-SLAM3 factorial experiment (I0–I3), baseline ATE survey across 8 sequences, entity-attribution ablation (T0–T5), OpenLORIS corridor analysis (C0–C2), and EDRR data-sufficiency gate. |
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

# Controlled VLM evaluation across 4 prompt configurations
python scripts/run_controlled_vlm_evaluation.py

# Native ORB-SLAM3 factorial experiment (I0–I3) with live VLM and EDRR
python scripts/run_native_estimator_experiment.py
```
*For detailed CLI execution guides across OpenLORIS and multi-session runs, see [`docs/setup_and_usage.md`](docs/setup_and_usage.md).*

### 3. Run Query Selection Policy Benchmark
Evaluate P0–P7 routing policies across 7 query budgets:

```bash
# Requires causal feature vectors in results/estimator_context/
python scripts/run_query_selection_benchmark.py

# Bayesian log-odds evidence fusion across all 885 observations
python scripts/run_decision_evidence_fusion.py
```

### 4. Run Statistical Significance & Ablation Analysis
Evaluate McNemar significance test and component-ablation study:

```bash
# Requires fused interventions and query decisions
python scripts/run_statistical_significance_and_ablation.py
```
*Outputs written to `results/statistical_analysis/`.*

### Dataset Coverage & Pipeline Population Summary

| Sequence | Sensor | GT Poses | Dynamic Actors |
| :--- | :--- | :---: | :--- |
| TUM `fr3_walking_xyz` | Kinect v1 | 2,884 | 2 walking persons |
| TUM `fr3_walking_static` | Kinect v1 | 2,484 | seated persons |
| TUM `fr3_walking_halfsphere` | Kinect v1 | 3,582 | walking persons |
| Bonn `rgbd_bonn_crowd` | RealSense | — | crowd |
| Bonn `rgbd_bonn_balloon_tracking` | RealSense | — | dragging box |
| OpenLORIS `corridor1-4` | RealSense D435 | 2,515 | pedestrians |
| D455 `session_01` | RealSense D455 | No MoCap | — |
| D455 `session_02` | RealSense D455 | No MoCap | — |
| **Total** | | | **177 tracks (131 dynamic)** |

**Pipeline population**: 8 sequences → 885 causal observations → 177 tracks → 4-partition evaluation split.
Development split: `tum_xyz` + `bonn_crowd` (30 dev tracks). Validation, held-out, and perception-only partitions
defined in `results/track_annotation/evaluation_split.json`.

---

### Native Telemetry, EDRR, and Dual-Level Integration Summary

The native SLAM factorial experiment validates native ORB-SLAM3 telemetry export, the decision-theoretic Expected Downstream Regret Reduction (EDRR) formulation, sequence-level grouped cross-validation, and native ORB-SLAM3 landmark management on TUM `fr3_walking_xyz` with live `Qwen2.5-VL-3B-Instruct` inference:

| Factorial Mode | Condition | Front-End Mask | Landmark Weight ($\Omega_{ij}$) | Reactivation | ATE RMSE (m) | Delta ATE (m) | Relative Delta | Telemetry Artifacts |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **I0** | Baseline Control | None | $1.0 \cdot \Sigma^{-1}$ | Disabled | **0.3637 m** | 0.0 m | Baseline | `telemetry_I0_baseline_control/` |
| **I1** | Image Mask Only | Pass B Polygon | $1.0 \cdot \Sigma^{-1}$ | Disabled | **0.3729 m** | -0.0092 m | -2.53% | `telemetry_I1_image_mask_only/` |
| **I2** | Landmark Only | None | $0.0001 \cdot \Sigma^{-1}$ | Enabled | **0.4002 m** | -0.0364 m | -10.01% | `telemetry_I2_landmark_only_integration/` |
| **I3** | Dual Integration | Pass B Polygon | $0.0001 \cdot \Sigma^{-1}$ | Enabled | **0.3729 m** | -0.0092 m | -2.53% | `telemetry_I3_dual_integration/` |

- **Native C++ Tests**: 10/10 unit tests passing (`native/build_and_run_tests.sh`), verifying dynamic landmark lifecycle states (`ACTIVE`, `TEMPORARILY_EXCLUDED`, `REACTIVATION_PENDING`, `REACTIVATED`, `PERMANENTLY_BAD`), information matrix scaling, and reversible reactivation.
- **Data Sufficiency Gate**: EDRR grouped cross-validation is formally gated as `NOT_TRAINED_INSUFFICIENT_DATA` ($N = 4 < 30$ required).
- **Live VLM Serving**: 4 live VLM queries issued during I3 run; all returned `IMMOVABLE`. Full audited HTTP interaction logs with latency and token counts are preserved in `results/native_estimator_integration/run_20261101_native/live_vlm_query_records.jsonl`.

---

## Evaluation Protocol & Population Summary

| Partition | Sequences | Tracks | Observations | Role |
| :--- | :--- | :---: | :---: | :--- |
| Development | `tum_xyz` + `bonn_crowd` | 30 | — | Controlled VLM prompt evaluation, evidence fusion tuning |
| Validation | Held-out validation sequences | varies | — | Generalization audit |
| Held-Out | Unseen benchmark sequences | varies | — | Final model evaluation |
| Perception-Only | D455 sessions (no MoCap GT) | varies | — | Perception & tracking metrics only |
| **Total** | **8 sequences** | **177** | **885** | Full pipeline evaluation |

### Track Annotation Schema
Track annotations in `results/track_annotation/track_annotations.csv` define:
- `physical_mobility`: `DYNAMIC` or `STATIC` ground-truth label (sourced from MoCap reference or manual audit).
- `observed_motion`: Boolean flag indicating whether the track exhibited visible centroid displacement.
- `annotation_confidence`: `LOW`, `MEDIUM`, or `HIGH` reflecting ground-truth confidence.
- `sequence_id`, `track_id`: Composite unique key (neither alone is globally unique).

### Key Empirical Results
- **Estimator-Context Heuristic (P7)** @ 10% budget: 31.4% recall on truly moving tracks (54.4% recall on self-mobile tracks), 100.0% precision, 88 VLM queries.
- **Bayesian Evidence Fusion (P7)**: Dynamic track recall = 70.6%, masking precision = 83.7%, Track F1 = 0.766, static retention = 90.7%.
- **Statistical Significance (McNemar test)**: $b = 30, c = 0, p < 0.00001$, $\Delta\text{precision} = +34.1\%$ over budget-matched random routing P2.
- **Ablation A4 (No Estimator Displacement Risk)**: Precision drops from 100.0% to 93.2% without causal displacement risk gating.

---

## Canonical Outputs & Manuscript

The camera-ready research manuscript is located at [`results/publication_campaign/CAMERA_READY_RESEARCH_MANUSCRIPT.md`](results/publication_campaign/CAMERA_READY_RESEARCH_MANUSCRIPT.md).
All numeric claims are backed by canonical stage outputs and verified by the automated sync audit:

| Component | Canonical Output Path | Key Metrics & Role |
| :--- | :--- | :--- |
| Perception Harvesting | `results/perception_harvesting/run_20261101_234859/` | 885 observations, 177 tracks across 8 sequences |
| Track Annotation | `results/track_annotation/` | 177 track annotations, 4-partition evaluation split |
| Controlled VLM Evaluation | `results/vlm_evaluation/run_20261102_101448/` | 120 live inference records; V2 bal-acc = 0.606, V3 bal-acc = 0.576 |
| Estimator Context Extraction | `results/estimator_context/` | 885 causal feature vectors, evaluation_labels.jsonl |
| Policy Inference Records | `results/policy_vlm_inference/live_policy_vlm_records.jsonl` | 168 policy-selected live queries (100% hash verification) |
| Query Selection Benchmark | `results/query_policy/` | P0–P7 benchmark (P7@10%: 31.4% rec, 100% prec, 1,000 P2 draws) |
| Decision & Evidence Fusion | `results/decision_fusion/` | Bayesian log-odds fused interventions; P7 Track F1 = 0.766 |
| Native Factorial SLAM | `results/native_estimator_integration/run_20261101_native/` | I0–I3 factorial SLAM (I0 ATE = 0.3637 m, I1 = 0.3729 m) |
| Statistical Hypothesis Testing | `results/statistical_analysis/` | Paired exact McNemar tests ($p = 0.0107$), A1–A4 ablations |
| Research Manuscript | `results/publication_campaign/` | Camera-ready manuscript |

---

## License

This project is licensed under the Apache 2.0 License - see the [LICENSE](LICENSE) file for details.
