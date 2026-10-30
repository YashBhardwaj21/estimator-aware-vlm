# System Architecture & Implementation

## 1. System Overview & Repository Layout

The **Estimator-Aware Vision-Language Model (VLM)** framework integrates multimodal perception with 3D spatial representations to protect visual SLAM state estimation in dynamic, multi-session environments. Rather than querying costly foundation models indiscriminately or relying on naive heuristic thresholds, the system formulates VLM querying as a utility-driven resource allocation problem: an observation is escalated to the VLM if and only if the expected downstream improvement in state estimation exceeds the latency and computational cost of foundation model inference.

### 1.1 Top-Level Directory Layout

```text
estimator-aware-vlm/
├── pyproject.toml              # Build system, dual-package definition, and pytest configuration
├── requirements.txt            # Core VLM research, benchmark, and evaluation dependencies
├── requirements-mapping.txt    # Semantic mapping and 3D point cloud dependencies
├── requirements-recording.txt  # RealSense sensor streaming and ROS bag extraction dependencies
├── README.md                   # Repository overview, setup, and quick-start guide
├── benchmarks/                 # Versioned manifests, ground truth, and multi-view crop assets
│   ├── assets/                 # Multi-view RGB crop assets (42 physical instances)
│   ├── release_results/        # Frozen reproducible benchmark release bundles
│   └── model_registry.json     # Architecture-agnostic VLM backend configuration
├── configs/                    # Sensor calibration and pipeline runtime configuration YAMLs
├── data/                       # Raw RGB-D sequences and processed observation datasets
├── docs/                       # Core documentation suite (6 consolidated reference guides)
├── mapping/                    # Semantic mapping and offline point cloud processing pipelines
├── models/                     # Deep learning checkpoints (YOLOv8-Seg, YOLOE, Ultralytics assets)
├── outputs/                    # Experiment runs, trajectories, figure renders, and benchmark outputs
├── recording/                  # Synchronized sensor capture and rosbag extraction tools
├── scripts/                    # Unified benchmark execution CLIs and preflight diagnostics
├── src/                        # Core Python source packages
│   ├── estimator_aware_vlm/    # Routing policies, evidence adapters, models, common paths
│   └── scene_graph/            # Spatial perception engine (YOLOE, geometry, causal tracking)
└── tests/                      # Automated unit, integration, and regression test suite
```

### 1.2 Pipeline Architecture

The complete system pipeline is structured into four primary operational subsystems:

1. **Upstream Ingestion, Perception & Tracking**:
   - Ingests synchronized RGB-D frame sequences, camera calibration parameters, and visual odometry trajectories.
   - Detects object instances using open-vocabulary and semantic segmentation models (YOLOE in text/prompt-free/visual modes and YOLOv8-Seg).
   - Projects 2D instance segmentations into 3D metric camera space via pinhole back-projection and aligns the ground coordinate frame via RANSAC plane fitting.
   - Tracks 3D entities across frames using causal Kalman filtering and spatial nearest-neighbor association (`causal_tracker.py`).

2. **Pre-Router Multimodal Evidence Extraction**:
   - Extracts geometric point dispersion, depth containment ratios, and bounding box volumetric consistency without foundation model queries (`geometric_evidence.py`).
   - Tracks temporal persistence, observation frequency, and track fragmentation (`temporal_evidence.py`).
   - Evaluates multi-frame class voting purity and computes Shannon entropy over detection confidence distributions (`semantic_evidence.py`).
   - Combines evidence into a structured, unified observation record via the evidence adapter (`evidence_adapter.py`).

3. **Estimator-Aware Selective Routing**:
   - Evaluates whether an observed entity exhibits unresolved affordance ambiguity.
   - Computes expected downstream state estimator loss reduction $\Delta \hat{L}_{\text{slam}}$ using an empirically calibrated ridge-regularized model.
   - Dispatches inference requests to the local VLM (Qwen2.5-VL-3B-Instruct) only when net query utility $U(q) \ge 0$, otherwise resolving affordances cheaply via geometric heuristics or taxonomic priors (`router_policy.py`, `selective_router.py`).

4. **Contextual Scene Graph Maintenance & Causal SLAM Protection**:
   - Updates persistent 3D landmark nodes in the dynamic temporal scene graph (`contextual_scene_graph_fusion.py`).
   - Decouples historical spatial poses when physical relocations are detected, preserving metric consistency without centroid corruption.
   - Emits pixel-level instance suppression masks (tight polygon contours) to eliminate dynamic feature points from SLAM tracking (Pass B SLAM).

```
[Raw RGB-D Stream & Trajectory] + [2D Segmentations & ByteTrack IDs]
                             │
                             ▼
              Pre-Router Evidence Extraction
     ┌───────────────────────┼────────────────────────┐
     ▼                       ▼                        ▼
Geometric Evidence      Temporal Evidence        Semantic Evidence
(3D Depth containment, (Lifespan, observation   (Class distribution purity,
 bounding box bounds)   frequency, frame gaps)   Shannon entropy)
     └───────────────────────┬────────────────────────┘
                             ▼
              Selective Router Decision Engine
             Computes Net Query Utility: U(q)
                             │
            ┌────────────────┴────────────────┐
            ▼                                 ▼
       U(q) < 0                           U(q) >= 0
(Resolve via Prior / Geometry)       (Escalate to Local VLM)
            │                                 │
            │                      Qwen2.5-VL-3B-Instruct
            │                      (Neutral Schema Prompt ->
            │                       Affordance + Mobility JSON)
            │                                 │
            └────────────────┬────────────────┘
                             ▼
             Cross-Session Hungarian Association
            (Consensus RANSAC SE(2) Registration &
             Uncertainty-Gated Centroid Matching)
                             │
                             ▼
         Contextual Scene Graph Fusion & SLAM Protection
      - Stable Landmarks: Retained with updated spatial extent
      - Relocated Objects: Decoupled historical poses preserved
      - Dynamic Entities: Suppressed via polygon masks in Pass B
```

---

## 2. Core Python Packages (`src/`)

The repository installs two synchronized Python packages in editable mode:

### 2.1 `estimator_aware_vlm` (`src/estimator_aware_vlm/`)

Contains research modules for vision-language routing, evidence extraction, and statistical evaluation:

* **`common/paths.py`**: Centralized canonical path resolution (`get_repo_root()`, `get_benchmarks_dir()`, `get_outputs_dir()`, `get_data_dir()`, `get_configs_dir()`). Resolves paths relative to the repository root, ensuring directory independence regardless of where a script or test is invoked.
* **`selective_router.py`**: Implements `SelectiveRouter` and `SelectiveVLMDecisionEngine`. Orchestrates multi-modal prompt construction, formats OpenAI-compatible API payloads, verifies JSON response schemas, and tracks per-query latencies.
* **`router_policy.py`**: Implements the candidate evaluation policies:
  * `B0_AlwaysVLM` / `B3_VLMAll`: Upper-bound baseline querying the VLM on 100% of candidate instances.
  * `B1_NeverVLM`: Static heuristic resolving strictly by class taxonomy.
  * `B2_GeometryOnly`: Evaluates spatial displacement and velocity thresholds.
  * `B4_RandomRouting`: Budget-matched random query allocation baseline ($K_{\text{B4}} \equiv K_{\text{B6}}$).
  * `B5_ConfidenceThreshold`: Standard heuristic thresholding on detection uncertainty.
  * `B6_EstimatorAware`: Proposed utility-driven policy balancing expected SLAM error reduction against token inference cost.
* **`geometric_evidence.py`**: Implements `GeometricEvidenceExtractor`. Projects 2D bounding box masks into 3D metric point clouds, evaluates axis-aligned bounding boxes, computes inlier depth ratios, and quantifies spatial position variance.
* **`temporal_evidence.py`**: Implements `TemporalEvidenceExtractor`. Tracks multi-frame persistence, observation intervals, frame gap statistics, and flags transient false positives.
* **`semantic_evidence.py`**: Implements `SemanticEvidenceExtractor`. Accumulates open-vocabulary class predictions across tracklets and computes Shannon entropy $H(C) = -\sum p_i \log_2 p_i$.
* **`evidence_adapter.py`**: Implements `EvidenceAdapter`. Bundles extracted evidence records into the standardized benchmark input format without ground-truth label leakage.
* **`calibration/counterfactual.py`**: Implements `CalibratedUtilityModel`. Fits a regularized linear model on empirical single-track SLAM counterfactual interventions using Leave-One-Out Cross-Validation (LOOCV).
* **`cross_session_association.py`**: Implements `CrossSessionCandidateMatcher`. Consolidates fragmented tracklets within sessions and solves inter-session SE(2) registration via consensus RANSAC.
* **`contextual_scene_graph_fusion.py`**: Implements `ContextualSceneGraphFusion`. Merges multi-session observations, distinguishes moved vs stationary objects, and maintains historical pose decoupled state.
* **`statistical_analysis.py`**: Implements Wilson score confidence intervals, McNemar's exact test, paired bootstrap resampling, and Holm-Bonferroni Family-Wise Error Rate (FWER) step-down adjustments.
* **`multimodal_inference_engine.py`**: Architecture-agnostic client for OpenAI-compatible VLM endpoints (`llama-server`, vLLM, Ollama), managing timeouts, retries, and schema validation.

### 2.2 `scene_graph` (`src/scene_graph/`)

Provides upstream spatial perception and metric 3D scene representation:

* **`geometry/`**: Pinhole back-projection, coordinate frame transformations, 3D bounding box estimation, and RANSAC ground plane leveling (`floor_alignment.py`).
* **`perception/`**: Open-vocabulary 2D detection and segmentation wrappers (`YOLOEDetector`, `yoloe_detector.py`) supporting text-prompt, prompt-free, and visual-prompt modes.
* **`tracking/`**: Multi-frame 3D object association and lifespan management (`causal_tracker.py`).
* **`mapping/`**: Global object association, occupancy grid rasterization, and spatial indexing.
* **`ontology/`**: Formal entity and lifecycle definitions (`entity.py`), including `PersistentEntity`, `EntityLifecycleState`, and mobility taxonomies.
* **`graph/`**: Dynamic 3D temporal scene graph data structures (`temporal_graph.py`).

---

## 3. Mathematical Formulations & Coordinate Geometry

### 3.1 Pinhole Back-Projection & Metric Point Clouds

Given an RGB-D camera with intrinsic matrix $K$:

$$K = \begin{bmatrix} f_x & 0 & c_x \\ 0 & f_y & c_y \\ 0 & 0 & 1 \end{bmatrix}$$

Each pixel $(u, v)$ with metric depth $Z = D(u, v)$ is back-projected to 3D camera coordinates:

$$X = \frac{(u - c_x) \cdot Z}{f_x}, \quad Y = \frac{(v - c_y) \cdot Z}{f_y}, \quad Z = D(u, v)$$

Transforming to world coordinates using estimated camera pose $T_{wc} = [R_{wc} \mid t_{wc}]$:

$$p_w = R_{wc} \begin{bmatrix} X \\ Y \\ Z \end{bmatrix} + t_{wc}$$

### 3.2 Coordinate Frames and Rigid SE(2) Registration

In multi-session mapping, coordinate reference frames $\mathcal{F}_1$ (Session 1) and $\mathcal{F}_2$ (Session 2) drift due to uncalibrated start poses or odometry accumulation. Under planar floor movement, the cross-session transformation is modeled as a special Euclidean transformation $\text{SE}(2)$:

$$p_1 = R_{12} p_2 + t_{12}$$

where:
* $R_{12} \in \text{SO}(2)$ is the planar rotation matrix parameterized by yaw angle $\theta$:
  $$R_{12} = \begin{bmatrix} \cos\theta & -\sin\theta \\ \sin\theta & \cos\theta \end{bmatrix}$$
* $t_{12} = [t_x, t_y]^\top \in \mathbb{R}^2$ is the horizontal translation vector.

The transformation is computed using consensus RANSAC over stationary rigid anchors identified across sessions. In the benchmark multi-session recording (`my_desk_recording_01` and `02`), consensus RANSAC converges to:
$$\theta = -15.558^\circ, \quad t_{12} = [-0.1059, 0.0486]^\top \text{ m}$$
achieving an inlier residual RMSE of $0.1198\text{ m}$ across 7 consensus stationary landmarks.

### 3.3 Spatial Uncertainty Propagation and Relocation Gating

Let $c_1$ and $c_2$ denote the 3D centroids of an entity observed in Session 1 and Session 2, with position standard deviations $\sigma_{c1}$ and $\sigma_{c2}$. The estimated world displacement vector is:

$$\Delta p = c_1 - (R_{12} c_2 + t_{12})$$

Total displacement uncertainty standard deviation $\sigma_{\text{disp}}$ combines observation dispersion and registration residual:

$$\sigma_{\text{disp}} = \sqrt{\sigma_{c1}^2 + \sigma_{c2}^2 + \sigma_{\text{reg}}^2}$$

Under the estimator-aware policy, an entity is classified as physically displaced if and only if:

$$\|\Delta p\| \ge d_{\text{min}} \quad \text{and} \quad \|\Delta p\| \ge 3.0 \cdot \sigma_{\text{disp}}$$

where $d_{\text{min}} = 0.25\text{ m}$. This $3\sigma$ gating prevents odometry drift or registration residual noise from erroneously invalidating stationary objects.

---

## 4. Estimator-Aware Query Utility Formulation

The selective router formulates VLM escalation as a decision rule balancing expected downstream SLAM accuracy against computational cost:

$$U(q) = \Delta \hat{L}_{\text{slam}}(q) - \lambda_{\text{cost}} C_{\text{vlm}}$$

where:
* $\Delta \hat{L}_{\text{slam}}(q)$ is the predicted reduction in SLAM tracking error if the entity's mobility state is resolved.
* $C_{\text{vlm}}$ is the normalized inference cost penalty ($C_{\text{vlm}} = 0.50$).
* $\lambda_{\text{cost}}$ is the cost regularization hyperparameter (default $\lambda_{\text{cost}} = 0.50$).

The predicted SLAM benefit $\Delta \hat{L}_{\text{slam}}(q)$ is computed via a regularized linear model fitted to empirical single-track intervention counterfactuals:

$$\Delta \hat{L}_{\text{slam}}(q) = \mathbf{w}^\top \left( \frac{\boldsymbol{\phi}(q) - \boldsymbol{\mu}}{\boldsymbol{\sigma}} \right) + b$$

over standardized feature vector $\boldsymbol{\phi}(q) \in \mathbb{R}^6$:
1. `world_displacement_m`: Estimated physical translation magnitude $\|\Delta p\|$.
2. `displacement_uncertainty_m`: 3D measurement standard deviation $\sigma_{\text{disp}}$.
3. `observation_count`: Total temporal detections supporting the tracklet.
4. `camera_ego_motion_m`: Cumulative camera trajectory distance during tracking.
5. `lifespan_frames`: Duration of tracklet active lifespan.
6. `displacement_to_ego_ratio`: Relative motion ratio $\frac{\|\Delta p\|}{\text{ego\_motion}}$.

When $U(q) < 0$, the router resolves the query cheaply using taxonomic class priors or geometry heuristics at zero latency and zero token cost. When $U(q) \ge 0$, the router escalates to the local VLM.

---

## 5. VLM Inference Contract & Schema

When escalated, the system communicates with a local OpenAI-compatible endpoint hosting **Qwen2.5-VL-3B-Instruct**. The request adheres to a strict contract:

1. **Zero-Shot Prompting**: Uses neutral schema descriptions without class-biasing fixed demonstration exemplars.
2. **Disentangled Physical Representation**:
   * `movability`: Inherent physical affordance category (`IMMOVABLE`, `MOVABLE`, `MOBILE`).
   * `observed_motion`: Current temporal kinematic state (`STATIC`, `DISPLACED`, `DYNAMIC`).
3. **Identity Verification & Anti-Hallucination**: The payload requires echoing the queried `object_id` and `object_class`. Mismatched identifiers or malformed JSON payloads are penalized as invalid responses.
4. **Wire Latency Accounting**: Every query logs actual token counts (prompt + completion) and network wire latency, enabling dual-cost accounting across live and cached evaluation regimes.

---

## 6. Isolated SLAM Runtime (`ORBSLAM3Runner`)

To guarantee reproducible, non-destructive execution, Visual SLAM evaluation is decoupled from host system states via `ORBSLAM3Runner`:

* **Execution Isolation**: Runs ORB-SLAM3 in headless RGB-D mode inside an isolated WSL2 Ubuntu 22.04 container.
* **Strict Immutability**: Raw datasets (`data/raw/*`) remain strictly read-only.
* **Dynamic Mask Modulation**: Suppressed feature regions are dynamically masked in Pass B input frames stored exclusively under `outputs/experiments/<exp_id>/masked_pass_b/`.
* **Telemetry Capture**: Captures standard TUM format trajectories (`CameraTrajectory.txt`, `KeyFrameTrajectory.txt`), sparse 3D point clouds (`MapPoints.ply`), and complete terminal execution logs (`orbslam3_execution.log`).

