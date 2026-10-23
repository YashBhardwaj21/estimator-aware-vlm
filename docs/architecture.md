# Architecture

## 1. System Overview

The VLM subsystem integrates multi-modal sensor streams (RGB-D observations and 2D bounding boxes) with downstream 3D scene graphs in multi-session indoor SLAM.

The pipeline is structured into four main operational stages:
1. **Upstream Ingestion & Tracking**: Ingests synchronized RGB-D frames, camera odometry from RTAB-Map, and 2D YOLO detections with ByteTrack IDs.
2. **Pre-Router Evidence Extraction (Evidence Extraction Module)**: Evaluates geometric point containment, track temporal persistence, and multi-frame semantic entropy without triggering foundation models.
3. **Estimator-Aware Selective Routing (Selective Router Module)**: Evaluates whether an entity has unresolved affordance ambiguity and whether resolving that ambiguity provides sufficient utility to the SLAM estimator to justify VLM query cost.
4. **Scene Graph Maintenance & Downstream SLAM Protection**: Updates persistent 3D landmark nodes, decouples historical coordinate poses when objects move, and culls dynamic agents from the SLAM map.

---

## 2. Component Pipeline and Data Flow

```
[RGB-D Video & RTAB-Map Poses] + [2D Bounding Boxes & ByteTrack IDs]
                             │
                             ▼
              Pre-Router Evidence Extraction
     ┌───────────────────────┼────────────────────────┐
     ▼                       ▼                        ▼
Geometric Evidence      Temporal Evidence        Semantic Evidence
(Depth distribution,   (Track persistence,      (Class voting purity,
 inlier ratio, bbox)    lifespan, occlusions)    Shannon entropy)
     └───────────────────────┬────────────────────────┘
                             ▼
              Selective Router Policy
                Computes Query Utility U(q)
                             │
            ┌────────────────┴────────────────┐
            ▼                                 ▼
       U(q) <= 0                           U(q) > 0
(Fast Geometry / Prior)             (Escalate to Local VLM)
            │                                 │
            │                         Qwen2.5-VL-3B Server
            │                         (Prompt: neutral schema,
            │                          outputs: affordance JSON)
            │                                 │
            └────────────────┬────────────────┘
                             ▼
             Cross-Session Hungarian Association
           (Intra-session cluster consolidation &
            Uncertainty-gated distance matching)
                             │
                             ▼
         Contextual Scene Graph Fusion & Map Protection
      - Stable landmarks: Retained with updated spatial extent
      - Relocated objects: Pose decoupled, centroid drift prevented
      - Dynamic agents: Purged from static SLAM landmark graph
```

---

## 3. Component Details & Responsibilities

### 3.1 Pre-Router Evidence Extractors (`vlm/src/`)

- **`GeometricEvidenceExtractor` (`geometric_evidence.py`)**:
  - Accumulates 3D points back-projected from depth images within 2D bounding box masks.
  - Computes spatial extent (axis-aligned bounding box dimensions), depth point containment ratio, and depth variance.
  - Detects spatial inconsistency (e.g. depth points falling far outside expected object boundaries).

- **`TemporalEvidenceExtractor` (`temporal_evidence.py`)**:
  - Evaluates track persistence: track age (number of frames observed), frame gap intervals, and tracking confidence history.
  - Flags transient false-positive detections that appear for only a few frames.

- **`SemanticEvidenceExtractor` (`semantic_evidence.py`)**:
  - Accumulates class label votes across frames for the same tracklet.
  - Computes Shannon entropy over class probability distributions to quantify classification ambiguity.
  - Provides default dictionary mobility priors based on object class semantics.

- **`Stage5Adapter` (`evidence_adapter.py`)**:
  - Combines geometric, temporal, and semantic evidence records into a single structured observation context for routing.

---

### 3.2 Coordinate Frames and Rigid SE(2) Registration

Let $\mathcal{F}_1$ and $\mathcal{F}_2$ denote the coordinate reference frames of Session 1 and Session 2. Under the assumption of planar ground movement, the inter-session coordinate transformation is modeled as a special Euclidean transformation $\text{SE}(2)$:

$$p_1 = R_{12} p_2 + t_{12}$$

where:
- $R_{12} \in \text{SO}(2)$ is the planar rotation matrix parameterized by yaw angle $\theta$:
  $$R_{12} = \begin{bmatrix} \cos\theta & -\sin\theta \\ \sin\theta & \cos\theta \end{bmatrix}$$
- $t_{12} = [t_x, t_y]^T \in \mathbb{R}^2$ is the planar translation vector.

The transformation is computed using consensus RANSAC over stationary rigid landmarks identified across sessions. In the benchmark environment, this consensus achieves an inlier residual root mean square error (RMSE) of $0.1198\text{ m}$ across 7 consensus anchors.

---

### 3.3 Spatial Uncertainty Propagation and Relocation Gating

Let $c_1$ and $c_2$ denote the estimated 3D centroids of an entity observed in Session 1 and Session 2, with spatial position covariances $\Sigma_1$ and $\Sigma_2$. The estimated displacement vector is:

$$\Delta p = c_1 - (R_{12} c_2 + t_{12})$$

The total displacement uncertainty standard deviation $\sigma_{\text{disp}}$ combines observation dispersion and registration residual:

$$\sigma_{\text{disp}} = \sqrt{\sigma_{c1}^2 + \sigma_{c2}^2 + \sigma_{\text{reg}}^2}$$

Under the estimator-aware policy, an entity is classified as physically displaced if and only if:

$$\|\Delta p\| \ge d_{\text{min}} \quad \text{and} \quad \|\Delta p\| \ge 3.0 \cdot \sigma_{\text{disp}}$$

where $d_{\text{min}} = 0.25\text{ m}$. This gating prevents registration jitter or camera odometry drift from falsely triggering relocation actions on stationary objects.

---

### 3.4 Estimator-Aware Query Utility Formulation

The selective router decides whether to query the VLM based on expected net utility:

$$U(q) = \Delta L_{\text{SLAM}}(q) - \lambda_{\text{cost}} C_{\text{VLM}}$$

where:
- $\Delta L_{\text{SLAM}}(q) = I_{\text{estimator}} \cdot A(q)$ represents the value of resolving the query for the SLAM system.
- $I_{\text{estimator}} \in [0.0, 1.0]$ is the estimator influence score, reflecting how critical the landmark is to the state estimator (based on track length, inlier point ratio, and spatial consistency).
- $A(q) \in [0.0, 1.0]$ is the affordance ambiguity score, high when an object exhibits borderline displacement, ambiguous mounting class, or high semantic entropy.
- $C_{\text{VLM}}$ is the normalized inference cost (latency and compute penalty, parameterized at $0.5$).
- $\lambda_{\text{cost}}$ is the cost regularization hyperparameter (default $\lambda_{\text{cost}} = 0.5$).

When $U(q) \le 0$, the router bypasses VLM inference and resolves the affordance using fast geometry or class prior. When $U(q) > 0$, the router dispatches a query to the VLM.

---

### 3.5 VLM Inference Contract

When the router escalates an observation, it calls the local Qwen2.5-VL-3B endpoint using a structured prompt schema:
1. **No Fixed Demonstrations**: Uses a neutral schema definition without hardcoded target demonstrations to avoid biasing the model toward specific classes.
2. **Disentangled Classification**:
   - `movability`: Inherent physical category (`IMMOVABLE`, `MOVABLE`, `MOBILE`).
   - `observed_motion`: Observed displacement state (`STATIC`, `DISPLACED`, `DYNAMIC`).
3. **Identity Verification**: The prompt requires the model to echo the queried `object_id` and `object_class`. If the returned ID does not match, the response is rejected as an invalid prediction.

---

### 3.6 Cross-Session Association & Scene Graph Maintenance

- **Intra-Session Cluster Consolidation**: Upstream tracker fragmentation produces brief, broken tracklets. Before cross-session matching, tracks within $< 0.35\text{ m}$ sharing the same class label are merged into cohesive object clusters.
- **Cross-Session Hungarian Matching**: Pairs Session 1 clusters with Session 2 clusters using an uncertainty-weighted Euclidean distance matrix. Unmatched entities are categorized as new or unobserved.
- **Decoupled Pose History**: When an object is confirmed as relocated, the scene graph creates a decoupled historical pose record instead of updating the existing centroid in-place. This preserves metric consistency for both historical mapping trajectories and current navigation goals.
