# Estimator-Aware VLM: System Architecture & Repository Structure

This document provides a comprehensive structural breakdown of the standalone **Estimator-Aware Vision-Language Model (VLM)** repository, detailing module boundaries, primary files, data contracts, and pipeline responsibilities.

---

## 1. Top-Level Directory Layout

```text
estimator-aware-vlm/
├── pyproject.toml              # Build system, dual-package definition, and pytest configuration
├── requirements.txt            # Core VLM research and evaluation dependencies
├── requirements-mapping.txt    # Semantic mapping and 3D point cloud dependencies
├── requirements-recording.txt  # RealSense sensor streaming and ROS bag extraction dependencies
├── README.md                   # Repository overview, setup, and quick-start guide
├── STRUCTURE.md                # Comprehensive module breakdown (this document)
├── benchmarks/                 # Ground truth datasets, evaluation manifests, and release results
├── configs/                    # Sensor calibration and pipeline runtime configuration YAMLs
├── mapping/                    # Semantic mapping, multi-session SE(2) fusion, and occupancy grid export
├── recording/                  # Synchronized RealSense D455 capture and rosbag replay/extraction
├── scripts/                    # Command-line benchmark execution and preflight diagnostics
├── slam/                       # Visual SLAM trajectory tracking and evaluation tools
├── src/                        # Core Python source packages
│   ├── estimator_aware_vlm/    # Routing policies, evidence adapters, models, common paths
│   └── scene_graph/            # Spatial perception engine (YOLOE, geometry, causal tracking)
├── tests/                      # Automated unit, integration, and regression test suite
└── assignment/                 # Immutable historical benchmark and submission archive
```

---

## 2. Core Python Packages (`src/`)

The repository installs two synchronized Python packages in editable mode via `pyproject.toml`:

### 2.1 `estimator_aware_vlm` (`src/estimator_aware_vlm/`)
Contains the core vision-language routing and evidence extraction research modules:

- **`common/paths.py`**: Centralized canonical path resolution (`get_repo_root()`, `get_raw_data_dir()`, `get_processed_data_dir()`, `get_mapping_outputs_dir()`, `get_benchmarks_dir()`, `resolve_path()`). Ensures code executes identically regardless of invocation working directory.
- **`selective_router.py`**: Implements the primary `SelectiveVLMDecisionEngine` that fuses geometric evidence, temporal persistence, semantic entropy, and estimator state into an expected utility score $U(q)$ against token inference cost.
- **`router_policy.py`**: Defines candidate evaluation policies:
  - `B0_AlwaysVLM`: Upper-bound baseline querying the VLM for every detected instance.
  - `B1_NeverVLM`: Static heuristic baseline using class lookup.
  - `B2_ConfidenceThreshold`: Standard confidence-thresholded escalation.
  - `B3_EntropyThreshold`: Semantic entropy-thresholded escalation.
  - `B4_GeometricDisplacement`: Geometry-only escalation.
  - `B5_TemporalInstability`: Temporal track instability escalation.
  - `B6_EstimatorAware`: Proposed full utility-driven router balancing accuracy and inference budget.
- **`geometric_evidence.py`**: Computes spatial displacement magnitude, bounding box overlap, and alignment confidence relative to estimator uncertainty.
- **`temporal_evidence.py`**: Tracks multi-frame persistence, detection frequency, and track fragmentation.
- **`semantic_evidence.py`**: Evaluates 2D detection confidence entropy and class distribution stability.
- **`multimodal_inference_engine.py`**: Architecture-agnostic client for OpenAI-compatible VLM servers (`llama-server`, vLLM, Ollama), handling prompt construction, payload formatting, error handling, and health verification.
- **`evidence_adapter.py`**: Bridges raw scene graph track observations into the benchmark scoring format.
- **`cross_session_association.py`**: Associates objects across disjoint mapping sessions using rigid SE(2) registration.
- **`contextual_scene_graph_fusion.py`**: Reconciles moved vs stationary objects, updates landmark persistence, and generates fused 3D scene graphs.
- **`statistical_analysis.py`**: Statistical hypothesis testing and confidence interval computation.

### 2.2 `scene_graph` (`src/scene_graph/`)
Provides upstream spatial perception and 3D geometry extraction:

- **`data/`**: Sensor packet synchronization, frame representations, and observation dataset loaders.
- **`geometry/`**: Pinhole back-projection, RANSAC ground plane leveling (`floor_alignment.py`), and 3D bounding box computation.
- **`perception/`**: Open-vocabulary 2D detection and segmentation wrappers (`YOLOEDetector`).
- **`tracking/`**: Multi-frame 3D object association and lifespan management (`causal_tracker.py`).
- **`mapping/`**: Global object association, metric occupancy grid rasterization, and spatial indexing.
- **`relations/`**: Spatial predicate evaluation (support, proximity, containment).
- **`graph/`**: Dynamic temporal 3D scene graph data structures (`temporal_graph.py`).

---

## 3. Pipelines & Workflows

### 3.1 Live Recording & Ingestion (`recording/`)
- `record_d455.py`: High-speed synchronized RealSense D455 RGB-D + IMU recording pipeline.
- `convert_ros2_bag_to_rgbd.py`: Ingests ROS 2 `.db3` bags and converts them into standardized PNG image sequences and timestamp metadata under `data/raw/`.
- `replay_rgbd_ros.py`: Replays recorded RGB-D sequences back to ROS 2 topics for hardware-in-the-loop validation.
- `check_d455_hardware.py` & `check_d455_imu.py`: USB 3.0 throughput and IMU gyroscope drift diagnostics.

### 3.2 Trajectory Tracking & SLAM (`slam/`)
- `trajectory_manager.py`: Parses TUM format trajectories (`CameraTrajectory.txt`), validates pose alignment, and performs nearest-neighbor timestamp matching ($\le 35\text{ ms}$).
- `evaluate_trajectory.py`: Computes Absolute Trajectory Error (ATE) and Relative Pose Error (RPE) against ground truth trajectories.
- `d455_orbslam3.yaml`: Camera calibration and feature tracking profiles for ORB-SLAM3.

### 3.3 Semantic Mapping & Multi-Session Fusion (`mapping/`)
- `run_offline_semantic_mapping.py`: Primary mapping pipeline. Ingests RGB-D sequences and SLAM camera trajectories, performs object detection and tracking, unprojects depth into 3D point clouds, fits ground planes, and outputs `object_observations.jsonl`.
- `fuse_multi_runs.py`: Solves inter-session SE(2) registration via consensus RANSAC, detects physical object displacements, and generates the fused 3D scene graph and neutral metric occupancy grid (`occupancy_grid.pgm` and `occupancy_grid.yaml`).
- `run_mapping_pipeline.py`: Automated multi-step pipeline runner orchestrating mapping across raw recordings.
- `generate_clean_floor_plan.py`: Renders publication-grade 2D architectural floor plans from fused scene graph bounding boxes.

### 3.4 Benchmarks & Evaluation (`benchmarks/`, `scripts/`)
- `scripts/run_paper_benchmark.py`: Unified benchmark CLI supporting `--protocol policy-replay`, `--protocol visual-live`, and `--protocol router-live`. Generates comprehensive performance tables and cryptographic execution manifests (`run_manifest.json`).
- `scripts/check_system_dependencies.py`: Automated preflight diagnostics tool for validating Python version, CUDA acceleration, directory structure, and package dependencies.
- `benchmarks/assets/multiview_samples/`: Materialized 42-instance multi-view RGB crop assets (146 files) used for reproducible visual evaluation.
- `benchmarks/model_registry.json`: Architecture-agnostic VLM backend configuration mapping model identifiers to endpoints and parameters.
- `benchmarks/BENCHMARK_REPORT.md`: Comprehensive evaluation report documenting methodology, error analyses, and uncertainty reconciliation.
- `benchmarks/release_results/v1_baseline/`: Frozen release bundle containing canonical benchmark outputs, metrics, and `release_checksums.json`.

---

## 4. Test Suite (`tests/`)

The test suite covers end-to-end functionality, import hygiene, and data integrity:

- `test_packaging_and_paths.py`: Asserts package import independence, canonical path resolution, and zero legacy `vlm` imports.
- `test_navigation_decoupling.py`: Asserts complete decoupling of autonomous navigation code and verifies mapping exports neutral occupancy grids.
- `test_data_pipeline_flow.py`: Verifies recording, conversion, and mapping directory structures and contracts.
- `test_benchmark_manifest_integrity.py`: Cryptographically verifies benchmark manifest schemas, population sizes, asset paths, and prevents label leakage.
- `test_router_metrics.py`: Verifies dual metric scoring rules, task-mode handling, and scoring penalties for malformed JSON or ID mismatches.

---

## 5. Historical Archive (`assignment/`)

The `assignment/` directory is an **immutable historical archive** containing the original course assignment submission and baseline snapshot. It is excluded from active test discovery (`pytest.ini_options` `norecursedirs`) and is never imported by the active `estimator_aware_vlm` package.
