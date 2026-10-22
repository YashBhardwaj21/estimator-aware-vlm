# Data and Manifests

## 1. Overview of Data Sources

The perception subsystem operates on multi-session indoor sensor sequences, object crops, tracking logs, and ground truth annotations.

The data layout is structured as follows:

```
vlm/
├── data/
│   ├── raw/
│   │   ├── session_01/                 # Session 1 camera trajectory, timestamps, intrinsics
│   │   └── session_02/                 # Session 2 camera trajectory, timestamps, intrinsics
│   ├── multiview_samples/              # Multi-view cropped objects and contextual frames
│   │   └── instance_<id>/              # Crops and context views per physical object
│   └── observations/                   # Multi-session observation caches
├── benchmarks/
│   ├── paper_benchmark_inputs_expanded.json       # Visual benchmark inputs (42 instances)
│   ├── paper_benchmark_ground_truth_expanded.json  # Visual benchmark ground truth (42 labels)
│   ├── router_benchmark_inputs_expanded.json      # Router benchmark inputs (53 entities)
│   ├── router_benchmark_ground_truth_expanded.json # Router benchmark ground truth (53 labels)
│   ├── controlled_scene_events.json               # Metric tape-measured relocation ground truth
│   └── cross_session_identity_annotations.json    # Human multi-session track identity associations
└── outputs/
    ├── object_observations.jsonl       # Full stream of 7,956 object observations across sessions
    └── stage5_registration_audit.json  # SE(2) consensus registration parameters and anchor residuals
```

---

## 2. Benchmark Populations and Manifests

To evaluate components without data leakage or mixed population denominators, evaluation is strictly divided into two distinct benchmark populations:

### 2.1 Visual Classification Benchmark ($N = 42$)
- **Purpose**: Evaluates the vision-language model's ability to classify physical affordance directly from visual crops and context frames, independent of tracking or SLAM estimators.
- **Inputs**: `vlm/benchmarks/paper_benchmark_inputs_expanded.json`
  - 42 physical object instances possessing valid, verified image crop and context pairs.
  - Instances include 3 `IMMOVABLE` fixtures, 37 `MOVABLE` items, and 2 `MOBILE` agents.
- **Ground Truth**: `vlm/benchmarks/paper_benchmark_ground_truth_expanded.json`
  - Annotations provide ground truth labels for `movability` (`IMMOVABLE`, `MOVABLE`, `MOBILE`) and `observed_motion` (`STATIC`, `DISPLACED`, `DYNAMIC`).

### 2.2 Estimator-Aware Router Benchmark ($N = 53$)
- **Purpose**: Evaluates selective routing decisions, compute cost reduction, and downstream SLAM landmark lifecycle actions.
- **Inputs**: `vlm/benchmarks/router_benchmark_inputs_expanded.json`
  - 53 physical entities evaluated in the multi-session scene graph.
  - Each entry provides tracking lifespan, inlier point ratios, spatial bounding boxes, class priors, and observed SLAM odometry displacement between sessions.
- **Ground Truth**: `vlm/benchmarks/router_benchmark_ground_truth_expanded.json`
  - Complete ground truth labels across all 53 physical scene entities.

### 2.3 Controlled Relocation Ground Truth
- **File**: `vlm/benchmarks/controlled_scene_events.json`
- **Purpose**: Evaluates spatial relocation accuracy against physical tape-measured ground truth for controlled move events:
  - `chair_002` (Caster chair): Ground truth displacement = $0.5630\text{ m}$
  - `chair_003` (Office chair): Ground truth displacement = $0.3860\text{ m}$
  - `storage_box_007` (Floor container): Ground truth displacement = $0.3960\text{ m}$

### 2.4 Cross-Session Identity Annotations
- **File**: `vlm/benchmarks/cross_session_identity_annotations.json`
- **Purpose**: Human-verified identity associations mapping upstream ByteTrack IDs in Session 1 to corresponding IDs in Session 2, used to evaluate tracking recall and Hungarian association accuracy.

---

## 3. Data Schemas

### 3.1 Observation Record Schema (`object_observations.jsonl`)

Each line in `object_observations.jsonl` represents a single object detection frame:

```json
{
  "observation_id": "obs_s1_f00120_t003",
  "session_id": "session_01",
  "frame_id": 120,
  "timestamp": 1696860000.123,
  "track_id": 3,
  "class_name": "chair",
  "class_id": 56,
  "detection_confidence": 0.88,
  "bbox_2d": [145.0, 210.0, 310.0, 480.0],
  "camera_pose_rtabmap": [
    [0.998, 0.012, -0.058, 1.25],
    [-0.012, 0.999, 0.005, 0.42],
    [0.058, -0.005, 0.998, -0.15],
    [0.0, 0.0, 0.0, 1.0]
  ],
  "estimated_3d_centroid": [1.45, 0.38, 2.10],
  "spatial_extent": [0.65, 0.60, 0.85],
  "inlier_depth_ratio": 0.94,
  "depth_variance": 0.018
}
```

### 3.2 Router Input Schema (`router_benchmark_inputs_expanded.json`)

Each item in the router manifest provides complete pre-router context for an evaluated entity:

```json
{
  "entity_id": "chair_002",
  "class_name": "chair",
  "session_observed": ["session_01", "session_02"],
  "track_ids": [3, 47],
  "total_observation_count": 84,
  "lifespan_frames": 165,
  "average_inlier_ratio": 0.91,
  "spatial_extent": [0.65, 0.62, 0.90],
  "s1_centroid": [1.45, 0.38, 2.10],
  "s2_centroid": [1.88, 0.52, 2.35],
  "measured_displacement_meters": 0.4889,
  "displacement_sigma_meters": 0.042,
  "semantic_entropy": 0.05,
  "default_semantic_prior": "MOVABLE",
  "estimator_influence_score": 0.85,
  "has_image_assets": true,
  "crop_path": "vlm/data/multiview_samples/chair_002/crop_view_0.jpg",
  "context_path": "vlm/data/multiview_samples/chair_002/context_view_0.jpg"
}
```

### 3.3 Visual Benchmark Input Schema (`paper_benchmark_inputs_expanded.json`)

```json
{
  "instance_id": "chair_002",
  "class_name": "chair",
  "crop_path": "vlm/data/multiview_samples/chair_002/crop_view_0.jpg",
  "context_path": "vlm/data/multiview_samples/chair_002/context_view_0.jpg",
  "metadata": {
    "session": "session_01",
    "view_angle_deg": 15.0,
    "distance_meters": 2.1
  }
}
```

### 3.4 Scene Graph Node Schema (`scene_graph.json`)

Nodes in the fused 3D scene graph maintain spatial coordinates and decoupled pose histories:

```json
{
  "node_id": "node_chair_002",
  "entity_id": "chair_002",
  "semantic_label": "chair",
  "affordance": "MOVABLE",
  "current_state": "DISPLACED",
  "lifecycle_action": "RETAIN_AND_DECOUPLE_POSE",
  "current_centroid": [1.88, 0.52, 2.35],
  "current_spatial_extent": [0.65, 0.62, 0.90],
  "pose_history": [
    {
      "session_id": "session_01",
      "centroid": [1.45, 0.38, 2.10],
      "status": "HISTORICAL_DECOUPLED"
    },
    {
      "session_id": "session_02",
      "centroid": [1.88, 0.52, 2.35],
      "status": "ACTIVE"
    }
  ],
  "slam_landmark_status": "VALID_CONSTRAINT"
}
```

---

## 4. Data Isolation and Integrity Principles

1. **No Ground Truth Leakage**: Input manifests (`*_inputs_*.json`) never contain target label fields (`movability`, `observed_motion`, `ground_truth_label`). Labels are isolated strictly in ground truth manifests (`*_ground_truth_*.json`).
2. **Strict Population Boundaries**: Metrics for the 42-instance visual evaluation and metrics for the 53-entity router evaluation are never mixed or aggregated into a shared denominator.
3. **Trajectory Ground Truth Boundary**: Because external 6-DoF optical motion capture ground truth was not recorded for the mapping trajectory, trajectory error metrics (Absolute Trajectory Error / Relative Pose Error) are recorded as `NOT_EVALUATED_NO_INDEPENDENT_TRAJECTORY_REFERENCE`. Map and object-level decisions are evaluated against human physical annotations and tape-measured metric displacements.
