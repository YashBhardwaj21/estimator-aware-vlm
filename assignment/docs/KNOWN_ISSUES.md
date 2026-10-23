# Catalogue of Known Experimental and Algorithmic Issues

This document catalogues all confirmed and suspected defects within the Dynamic 3D Scene Graph repository. Issues are categorized by severity and impact on the experimental claims.

---

## 1. Confirmed Algorithmic & Experimental Issues

### Issue 1: High False Split Rate (76.67%) & Low Physical Instance Recall (23.33%)
- **Severity:** `CRITICAL`
- **Location:** `vlm/assignment/source_snapshot/src/cross_session_association.py`, `temporal_evidence.py`
- **Confirmed Evidence:**
  - Evaluated on $N=30$ ground truth multi-session physical instances, only 7 instances are successfully re-associated across sessions across all 4 baselines.
  - Per-instance diagnostic analysis (`per_instance_association_diagnostics.json`) proves that 17 instances are dropped due to `UNMATCHED_GATE`.
  - The root cause is severe front-end tracker fragmentation: $1,008$ out of $1,385$ tracks ($72.78\%$) span 4 or fewer frames. These fragmented tracks produce noisy, non-converged spatial centroids that fall outside the Euclidean association gate ($0.45$ m).
- **Impact on Assignment:** The scene graph fails to merge the majority of physical objects across recording sessions, resulting in a disconnected graph where the same object exists as multiple separate nodes.

### Issue 2: Misleading Semantic Macro-F1 (99.76%) Masking 44.43% Coverage Deficit
- **Severity:** `HIGH`
- **Location:** `vlm/assignment/source_snapshot/src/semantic_evidence.py`, `vlm/outputs/prerouter_evidence_report.md`
- **Confirmed Evidence:**
  - Historical reports presented a $99.76\%$ Macro-F1 score as evidence of near-perfect semantic evidence extraction.
  - Audit of `prerouter_evidence_metrics.json` reveals this F1 was computed over only $4,421$ matched observations.
  - $3,535$ out of $7,956$ records ($44.43\%$) have detector labels outside the 3 hardcoded taxonomy classes (`IMMOVABLE`, `MOVABLE`, `MOBILE`) and were completely omitted from evaluation.
- **Impact on Assignment:** Inexperienced researchers may interpret $99.76\%$ as whole-system semantic classification accuracy, whereas the classifier only covers $55.57\%$ of the recorded observations.

### Issue 3: Peripheral Visual SLAM Odometry Drift Corrupting SE(2) Alignment
- **Severity:** `HIGH`
- **Location:** `vlm/assignment/source_snapshot/src/cross_session_association.py`, `benchmarks/registration/registration_audit.json`
- **Confirmed Evidence:**
  - While robust SE(2) registration achieves an inlier RMS of $0.1198$ m on 7 central anchors, the all-anchor RMS jumps to $0.3087$ m (pooled RMSE: $0.3538$ m).
  - Anchors located near the physical trajectory boundaries (e.g., Anchor 6: `desk_023` $\leftrightarrow$ `desk_034`, residual $0.618$ m; Anchor 7: `desk_025` $\leftrightarrow$ `desk_006`, residual $0.5643$ m) suffer from non-rigid RTAB-Map SLAM drift.
- **Impact on Assignment:** Any object located in the boundary region will experience registration displacement exceeding the $0.45$ m static association gate, causing the algorithm to falsely classify static boundary objects as moved or new.

### Issue 4: Significant Relocation Displacement Error (Mean 30.42%)
- **Severity:** `MEDIUM`
- **Location:** `vlm/assignment/source_snapshot/src/contextual_scene_graph_fusion.py`, `metrics/raw/prerouter_evidence_metrics.json`
- **Confirmed Evidence:**
  - Controlled tape measurements for 3 relocated items show:
    - Chair 002: $0.563$ m actual vs $0.6166$ m estimated ($9.52\%$ error).
    - Chair 003: $0.386$ m actual vs $0.2612$ m estimated ($32.34\%$ error).
    - Box 007: $0.396$ m actual vs $0.5917$ m estimated ($49.42\%$ error).
  - Mean absolute displacement error is $0.1247$ m; mean relative error is $30.42\%$.
- **Impact on Assignment:** 2D bounding-box centroid estimation lacks fine 3D point cloud segmentation. Centroid jitter creates up to $0.1957$ m error in physical displacement estimates.

### Issue 5: Artificial 100% Balanced Accuracy in Semantic Prior Baseline (M0)
- **Severity:** `MEDIUM`
- **Location:** `vlm/assignment/source_snapshot/src/router_policy.py`, `metrics/raw/router_ablation_results.json`
- **Confirmed Evidence:**
  - The M0 rule-based semantic prior baseline achieves $100\%$ balanced accuracy and $100\%$ Macro-F1 on the 15 evaluation validation instances.
  - This occurs because the validation instances are drawn from canonical classes (`door`, `chair`, `robot`) that directly match the hard-coded dictionary rules.
- **Impact on Assignment:** The synthetic benchmark creates a false impression that semantic priors alone are sufficient, masking the failure of priors in out-of-distribution or context-dependent scenarios (e.g., a chair anchored to the floor).

---

## 2. Suspected Issues & Open Research Hypotheses

### Issue 6: Fast Yaw Rotations Causing Front-End Tracking Breaches
- **Severity:** `MEDIUM (SUSPECTED)`
- **Symptom:** Observation records show sudden spikes in new `local_track_id` assignments when the robot executes angular turns $> 0.5$ rad/s.
- **Hypothesis:** 2D SORT/ByteTrack feature matching loses visual bounding box overlap during camera motion blur, triggering re-initialization rather than continuous track survival.

### Issue 7: Uncalibrated Spatial Gating Uncertainty
- **Severity:** `LOW (SUSPECTED)`
- **Symptom:** The proposed uncertainty-aware baseline uses scalar residual dispersions ($\sigma_{inlier} = 0.1198$ m) rather than propagating true $3 \times 3$ SE(2) covariance matrices from the SVD alignment Jacobian.
- **Hypothesis:** Propagating full anisotropic spatial covariances would expand gates along the direction of camera trajectory drift, potentially recovering boundary instances without increasing false merges.

### Issue 8: Multi-View Crop Visual Occlusions
- **Severity:** `LOW (SUSPECTED)`
- **Symptom:** In unguided VLM queries (M3), accuracy drops to $33.33\%$.
- **Hypothesis:** Harvested multi-view crops frequently include foreground visual clutter (e.g., desk edges obstructing chair legs), leading the VLM to misclassify affordances.
