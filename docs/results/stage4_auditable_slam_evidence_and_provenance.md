# Stage 4: Auditable SLAM Evidence & Provenance

**Date**: 2026-10-29  
**Specification**: Stage 4 of [Research Implementation Audit](file:///c:/Users/Yash%20Bhardwaj/Desktop/vlm/docs/research_implementation_audit.md#L120-L132)  
**Package**: [`src/estimator_aware_vlm/common/provenance.py`](file:///c:/Users/Yash%20Bhardwaj/Desktop/vlm/src/estimator_aware_vlm/common/provenance.py)  
**Test Suite**: [`tests/test_evidence_provenance.py`](file:///c:/Users/Yash%20Bhardwaj/Desktop/vlm/tests/test_evidence_provenance.py) (6/6 passed), Total Suite: 47/47 passed  

---

## 1. Executive Summary

Stage 4 establishes transparent, tamper-proof scientific provenance for all perception features, tracking attributes, and SLAM telemetry used in the estimator-aware routing pipeline. It resolves a major validity concern identified during the research audit:

> **Audit Finding**: Heuristic proxy quantities (such as 2D bounding-box depth inlier projections or observation counts) must never be disguised as direct backend SLAM optimizer telemetry (such as bundle-adjustment reprojection residuals or Schur complement information gain).

Stage 4 formalizes an auditable three-tier provenance categorization:
1. **`MEASURED_BACKEND`**: Directly acquired from sensor hardware (timestamps, depth, images) or state estimator backend outputs (trajectory poses, graph residuals, inlier sets).
2. **`DERIVED`**: Computed through deterministic, verifiable mathematical transformations without heuristic assumptions (e.g., rigid SE(3) camera ego-motion compensation, 3D Euclidean displacement, lifespan duration, sample standard deviation).
3. **`PROXY`**: Heuristic approximations, 2D projections, or learned model predictions where full 3D bundle-adjustment graph telemetry is unavailable (e.g., 2D bounding-box depth projection, three-tier motion state thresholds, object detector classification votes).

---

## 2. Provenance Taxonomy & Field Mapping

### 2.1 Per-Frame Observation Records (`ObservationRecord`)

| Field | Provenance Level | Source | Scientific Meaning |
| :--- | :---: | :--- | :--- |
| `frame_index` | `MEASURED_BACKEND` | `sensor_stream` | Frame sequence index from camera grabber |
| `timestamp` | `MEASURED_BACKEND` | `sensor_hardware` | Hardware timestamp from depth/RGB sensor |
| `median_depth_m` | `MEASURED_BACKEND` | `depth_sensor` | Metric depth map measurement at object pixel centroid |
| `camera_position_world`| `MEASURED_BACKEND` | `slam_trajectory` | Estimated camera translation from Pass A SLAM |
| `centroid_camera` | `DERIVED` | `pinhole_backprojection` | Inverse projection: $\mathbf{P}_{\text{cam}} = \mathbf{K}^{-1} [u, v, 1]^T \cdot Z$ |
| `centroid_world` | `DERIVED` | `se3_compensation` | Ego-motion compensated position: $\mathbf{P}_{\text{world}} = \mathbf{R}_{wc} \mathbf{P}_{\text{cam}} + \mathbf{t}_{wc}$ |
| `confidence` | `PROXY` | `yolo_detector` | 2D bounding box detection score |
| `bbox_xyxy` | `PROXY` | `yolo_detector` | 2D bounding box pixel coordinates |
| `mask_contour` | `PROXY` | `yolo_segmentation` | 2D polygon segmentation contour |
| `semantic_class` | `PROXY` | `yolo_detector` | COCO class mapped to indoor office ontology |

### 2.2 Multi-Frame Tracked Entities (`TrackedEntity`)

| Field | Provenance Level | Source | Scientific Meaning |
| :--- | :---: | :--- | :--- |
| `first_timestamp` | `MEASURED_BACKEND` | `sensor_hardware` | Hardware timestamp of initial observation |
| `last_timestamp` | `MEASURED_BACKEND` | `sensor_hardware` | Hardware timestamp of final observation |
| `track_id` | `DERIVED` | `causal_tracker` | Deterministic 3D Euclidean nearest-neighbor tracklet association |
| `observation_count` | `DERIVED` | `causal_tracker` | Total observations aggregated in tracklet |
| `lifespan_sec` | `DERIVED` | `causal_tracker` | Temporal duration: $t_{\text{last}} - t_{\text{first}}$ |
| `camera_ego_motion_displacement_m` | `DERIVED` | `slam_trajectory` | Norm of camera translation delta during tracking window |
| `apparent_camera_frame_displacement_m` | `DERIVED` | `sensor_depth` | Norm of camera-frame centroid delta (corrupted by ego-motion) |
| `world_displacement_m` | `DERIVED` | `se3_compensation` | Norm of camera-compensated 3D world centroid displacement |
| `displacement_uncertainty_m` | `DERIVED` | `statistics` | Spatial dispersion standard deviation around mean world centroid |
| `observed_motion` | `PROXY` | `heuristic_thresholds` | Three-tier classification: STATIC ($< 0.12\text{m}$), DISPLACED ($< 0.35\text{m}$), MOVING ($\ge 0.35\text{m}$) |
| `semantic_class` | `PROXY` | `majority_vote` | Temporal majority vote of 2D detector class labels |
| `inlier_ratio` | `PROXY` | `bbox_depth_projection` | Fraction of candidate 3D points projecting inside 2D box |

---

## 3. Telemetry Integrity in Routing Policies

In [`src/estimator_aware_vlm/router_policy.py`](file:///c:/Users/Yash%20Bhardwaj/Desktop/vlm/src/estimator_aware_vlm/router_policy.py#L93-L155):
- `compute_estimator_influence_with_provenance()` now checks the explicit provenance tag of inlier telemetry.
- Direct optimizer telemetry from ORB-SLAM3 bundle adjustment is tagged `MEASURED_BACKEND`.
- Heuristic 2D bounding-box depth inliers are tagged `PROXY_SUPPORT`.
- Zero-inlier conditions return `0.0` with explicit tags (`MEASURED_BACKEND_ZERO_INLIERS` or `PROXY_SUPPORT_ZERO_INLIERS`), strictly preventing unconstrained landmarks from receiving false confidence.

---

## 4. Automated Verification Results

All 6 provenance tests pass in `tests/test_evidence_provenance.py` (total test suite: 47/47 passed):

| Test Case | Responsibility | Status |
| :--- | :--- | :---: |
| `test_evidence_provenance_enum_values` | Rigorous enum matching (`MEASURED_BACKEND`, `DERIVED`, `PROXY`) | PASSED |
| `test_observation_provenance_manifest_completeness` | Complete field mapping for frame observation records | PASSED |
| `test_track_provenance_manifest_completeness` | Complete field mapping for multi-frame tracked entities | PASSED |
| `test_observation_record_serialization` | Serialization of observation records with provenance metadata | PASSED |
| `test_tracked_entity_serialization_and_lifespan` | Serialization of tracked entities, lifespan, and motion states | PASSED |
| `test_estimator_influence_provenance_discrimination` | Separation of true backend telemetry vs proxy support in router | PASSED |

