# Dynamic 3D Scene Graph — Project Overview

## 1. Assignment Scope and Core Objective

This assignment investigates **cross-session dynamic 3D scene graph generation** for indoor robotics. A mobile robot traverses an environment across multiple recording sessions (`recording_01` and `recording_02`), detecting objects, estimating trajectories via RTAB-Map visual SLAM, and constructing an persistent spatial graph of the world.

### Fundamental Scientific Questions
1. **Inherent Movability vs. Observed Physical Motion:**  
   How can a robotic system cleanly separate what an object *is* (e.g., a chair is inherently movable, a door is architecturally fixed, an RC car is mobile) from what an object *did* (was it moved between session 1 and session 2, or did it remain stationary)?
2. **Geometric Integrity & SLAM Invalidation:**  
   When an object relocates, how does the system preserve its historical trajectory while immediately invalidating its spatial landmarks to prevent visual SLAM relocalization failure and map corruption?
3. **Cost-Effective Multimodal Reasoning:**  
   How can a robot decide *when* to query expensive cloud Vision-Language Models (VLMs) for affordance reasoning versus when to resolve instances cheaply using geometric and semantic priors?

---

## 2. Architectural Components

The repository implements a staged multimodal architecture:

```
[RTAB-Map SLAM + 2D/3D Object Detections]
               │
               ▼
[Stage 5: Pre-Router Evidence Extraction]
   ├── Geometric Evidence (3D BBox, extents, point-to-box distance)
   ├── Temporal Tracking (track persistence, lifespan, occlusion recovery)
   └── Semantic Consistency (zero-shot CLIP classification)
               │
               ▼
[Robust SE(2) Inter-Session Alignment]
   └── RANSAC SVD Registration on Stationary Anchors (0.20m threshold)
               │
               ▼
[Cross-Session Data Association Engine]
   ├── Baseline 1: Nearest Centroid
   ├── Baseline 2: Hungarian Bipartite Matching
   ├── Proposed: Uncertainty-Gated Symmetric Matching
   └── Proposed: Relocation-Aware Candidate Association
               │
               ▼
[Stage 6/7: Selective VLM Router & Affordance Reasoning]
   ├── Rule-based Fast Filter (Architectural priors & stationary evidence)
   └── VLM Escalation (Multi-view crops for ambiguous affordances)
               │
               ▼
[Contextual Dynamic Scene Graph Fusion]
   ├── Node Resolution (Global Entity ID persistence)
   ├── Motion Classification (STATIC vs RELOCATED vs NEW)
   ├── Pose History Maintenance (storing pre- and post-relocation poses)
   └── SLAM Landmark Invalidation (tagging visual features as invalid)
```

---

## 3. Current Implementation Status vs. Identified Gaps

| Subsystem | Implemented & Verified | Known Limitation / Gap |
| :--- | :--- | :--- |
| **Stage 5 Evidence** | Full pipeline runs on 7,956 observations (`evidence_pipeline.py`). Generates aligned geometric, temporal, and semantic features. | 44.43% of detector records have classes outside the 3 evaluated taxonomy categories, resulting in 55.57% evaluation coverage. |
| **Registration** | Robust SE(2) SVD estimation with 0.20m inlier threshold (`inlier_rms = 0.1198m`). 5-fold cross-validation implemented. | Peripheral anchors (at sequence boundaries) exhibit up to 0.618m residual drift due to localized visual SLAM odometry drift. |
| **Data Association** | 4 association baselines implemented with physical-instance deduplication. Per-instance outcome diagnostics evaluated against 30 multi-session ground truth instances. | Overall physical instance recall is low (23.33%) across all baselines because 72.78% of tracks are fragmented into <=4 frame bursts, causing spatial gating rejections. |
| **Controlled Relocation** | Ground truth tape measurements recorded for 3 relocated objects (`chair_002`, `chair_003`, `storage box_007`). Relocation vectors calculated. | Mean relative displacement error is 30.42% (mean error 0.1247m) due to centroid estimation noise and 2D-to-3D projection approximations. |
| **Stage 6 Router** | Policy engine (`router_policy.py`) and cached evaluation (`router_evaluation_vlm_enabled.json`) evaluated across 15 validation instances. | Pure VLM query (M3) achieves only 33.33% balanced accuracy without spatial grounding; offline cache required for reproducibility. |
| **Scene Graph Fusion** | Contextual fusion engine (`contextual_scene_graph_fusion.py`) constructs 195 nodes, merges 29 stationary entities, and invalidates 12 SLAM landmarks. | Relocation detection currently relies on fixed distance thresholds rather than full probabilistic spatial hypothesis tracking. |

---

## 4. Key Takeaways for Downstream Researchers

- **Audit-First Integrity:** All original repository code, raw observation files, and ground truth benchmarks have been left intact.
- **Fail-Closed Evaluation Contracts:** The test suite (99 passing pytest unit tests in `source_snapshot/tests/`) enforces strict schema validation, prevents unrecorded latency defaults, and mandates multi-session canonical benchmarks.
- **Traceable Metrics:** Every metric reported in this assignment bundle is tied directly to underlying JSON records with explicitly stated denominators.
