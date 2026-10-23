# AI Review Prompt: Root-Cause Investigation & Repair Plan for Dynamic 3D Scene Graph

You are tasked with conducting a rigorous, evidence-based code and experimental audit of the **Dynamic 3D Scene Graph** system (`vlm/assignment/` bundle in repository `YashBhardwaj21/dynamic-3d-scene-graph`). 

The current cross-session data association suffers from **critically low physical instance recall (23.33%)**, a **severe false split rate (76.67%)**, and an **association error rate (41.67% false merges)**, while semantic evidence reports tout a misleading **99.76% Macro-F1** that conceals a **44.43% out-of-vocabulary deficit**.

Your objective is to diagnose exactly why the assignment performs poorly, trace data lineage end-to-end, verify mathematical formulations against actual records, and produce an actionable, prioritized implementation plan.

---

## STRICT OPERATIONAL CONSTRAINTS

1. **NO GUESSING OR SPECULATION:**  
   Every claim must be substantiated by direct citations of source files, line numbers, variable names, and underlying JSON/JSONL records.
2. **NO PREMATURE THRESHOLD TUNING:**  
   Do not arbitrarily adjust distance thresholds (e.g., changing 0.45m to 0.60m) without diagnosing the underlying geometric and tracking distortions.
3. **NO FABRICATION OF EVIDENCE OR LATENCIES:**  
   Never fill missing values with zero or invent unrecorded inference timings.
4. **NO CHEATING VIA GROUND-TRUTH LEAKAGE:**  
   Never use ground-truth identity annotations (`cross_session_identity_annotations.json`) or movability labels as runtime features or shortcuts. Data association must operate purely on sensory evidence and estimated transforms.
5. **FIX THE EXPERIMENTAL CONTRACT FIRST:**  
   Diagnose and resolve schema misalignments, data lineage gaps, and evaluation defects before attempting algorithmic enhancements.
6. **DO NOT REWRITE ALGORITHMS IMMEDIATELY:**  
   You must produce a complete diagnostic report and prioritized repair blueprint before writing code patches.

---

## MANDATORY AUDIT PROTOCOL

### Phase 1: End-to-End Identifier Tracing & Schema Lineage
Trace identifiers across the entire pipeline:
1. Examine `local_track_id`, `observation_id`, and `session_id` in `vlm/assignment/data/observations/object_observations.jsonl`.
2. Trace how `evidence_adapter.py` maps raw observations into `evidence_pipeline.py`.
3. Check how `cross_session_association.py` links track identifiers between `recording_01` and `recording_02` into candidate pairs.
4. Follow how `contextual_scene_graph_fusion.py` assigns `global_entity_id`, updates Kalman centroids, records `pose_history`, and tags `invalidated_slam_landmarks`.
5. Verify whether `track_to_instance_ground_truth.json` and `cross_session_identity_annotations.json` preserve 1-to-1 or 1-to-many relationships, and confirm how track oversegmentation is handled.

### Validation: Inter-Session Registration & Anchor Contamination
Inspect `cross_session_association.py` (`compute_robust_se2`) and `benchmarks/registration/registration_audit.json`:
1. Check whether any relocated object (e.g., `chair_002`, `chair_003`, `storage box_007`) was inadvertently included in the registration anchor set.
2. Inspect the 10 anchor correspondences. Verify why Anchors 1–5 and 10 have small residuals ($0.05$ m to $0.14$ m) while Anchors 6–8 have large residuals ($0.39$ m to $0.62$ m).
3. Determine whether peripheral visual SLAM trajectory drift causes boundary objects to drift beyond the static association gate ($0.45$ m).
4. Evaluate the 5-fold cross-validation results: why does mean fold RMS equal $0.3021$ m while pooled RMSE is $0.3538$ m?

### Phase 3: Spatial Gating, Bipartite Matching, & Association Failures
Audit `cross_session_association.py` (`associate_cross_session_candidates`):
1. Why do all 4 baselines (Nearest Centroid, Hungarian Gated, Uncertainty Symmetric, Relocation Aware) achieve the exact same low physical instance recall ($23.33\%$, 7/30)?
2. Inspect `metrics/derived/per_instance_association_diagnostics.json`. Detail why 17 instances fail with `UNMATCHED_GATE` and 6 instances fail with `FALSE_MERGE`.
3. Cross-reference with `temporal_evidence.py`: how does the 72.78% track fragmentation rate (1,008 tracks spanning $\le 4$ frames) starve spatial gates of reliable centroids?
4. Determine whether the ambiguity rejection mechanism in `PROPOSED_UNCERTAINTY_SYMMETRIC` (rejecting 10 pairs) and `PROPOSED_RELOCATION_AWARE` (rejecting 21 pairs) successfully prevents false merges without destroying recall.

### Phase 4: Ground Truth Contamination & Metric Arithmetic Recomputation
Audit all reported metrics and historical reports:
1. Recompute the Stage 5 semantic Macro-F1 ($99.76\%$) directly from `metrics/raw/prerouter_evidence_metrics.json`.
   - Prove that it only evaluates $4,421$ out of $7,956$ records ($55.57\%$ coverage).
   - Document what happens to the remaining $3,535$ records ($44.43\%$).
2. Check `movability_baseline_results.json` and `router_ablation_results.json`:
   - Verify whether the M0 baseline's $100\%$ balanced accuracy is an artifact of dictionary lookup matching the validation set vocabulary.
   - Analyze why pure unguided VLM queries (M3) drop to $33.33\%$ balanced accuracy.
3. Verify whether any metric in `reports/source_reports/` contradicts the raw JSON data in `metrics/raw/` or `metrics/derived/`.

---

## REQUIRED DELIVERABLES

When executing your review, structure your output into these explicit sections:

1. **Executive Summary & Scorecard:** A concise diagnosis of why physical instance recall is pinned at 23.33% and false split rate is 76.67%.
2. **Defect Classification Table:** Categorize every identified bug into one of five buckets:
   - `ALGORITHMIC_FAILURE` (e.g., rigid gates on noisy boundary tracks)
   - `EVALUATION_DEFECT` (e.g., unstated denominators, reporting matched Macro-F1 as pipeline accuracy)
   - `EXPERIMENTAL_DESIGN_FLAW` (e.g., dictionary lookup on synthetic validation splits)
   - `DATA_QUALITY_ISSUE` (e.g., tracker fragmentation, 2D centroid jitter)
   - `COSMETIC_OR_DOCUMENTATION` (e.g., outdated markdown tables)
3. **End-to-End Lineage & Contract Audit:** File-by-file, function-by-function breakdown of data representations, coordinate frame conversions, and identifier persistence.
4. **Discrepancy Matrix:** Table contrasting claimed metrics in historical markdown reports against actual values computed from raw JSON records.
5. **Prioritized Repair Implementation Plan:**
   - Step 1: Pre-fusion temporal clustering to consolidate fragmented short tracks ($>72\%$ of tracks).
   - Step 2: Anisotropic covariance propagation in SE(2) registration to widen gates along boundary trajectory drift axes.
   - Step 3: Probabilistic two-stage association separating static consensus matching from relocation hypothesis testing.
   - Step 4: Full-vocabulary semantic evaluation contract with explicit unknown/out-of-vocabulary handling.
6. **Required Regression Tests:** Concrete test cases (using `pytest`) to verify each proposed fix without regressing the existing 99 passing unit tests.
