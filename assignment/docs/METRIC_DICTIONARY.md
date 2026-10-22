# Experimental Metric Dictionary

This dictionary defines every quantitative metric used in the Dynamic 3D Scene Graph assignment, providing exact mathematical formulations, evaluation denominators, units, source artifacts, and known failure modes.

---

## 1. Cross-Session Data Association Metrics

### Candidate Pair Precision (%)
- **Definition:** The proportion of predicted cross-session track pairs that belong to the same true physical instance, evaluated over the subset of annotated candidate pairs.
- **Formula:**
  $$\text{Precision} = \frac{TP_{pairs}}{TP_{pairs} + FP_{pairs}} \times 100$$
- **Numerator:** $TP_{pairs} = 7$ (predicted pairs whose session 1 and session 2 tracks share the same annotated `physical_instance_id`).
- **Denominator:** $TP_{pairs} + FP_{pairs} = 12$ (total evaluated candidate pairs with ground truth annotations in baseline 1, 2, and 4; 11 in baseline 3).
- **Unit:** Percentage (%)
- **Source Artifact:** `metrics/derived/association_baselines_summary.json`
- **Caveat:** Does not evaluate unannotated background track pairs.

### Physical-Instance Recall (%)
- **Definition:** The percentage of true multi-session physical object instances that have at least one valid cross-session track association predicted by the algorithm.
- **Formula:**
  $$\text{Instance Recall} = \frac{|\mathcal{I}_{recovered}|}{|\mathcal{I}_{multi-session}|} \times 100$$
- **Numerator:** $|\mathcal{I}_{recovered}| = 7$ (ground truth instances with >=1 correctly associated cross-session pair).
- **Denominator:** $|\mathcal{I}_{multi-session}| = 30$ (total true physical instances with tracks in both Session 1 and Session 2).
- **Unit:** Percentage (%)
- **Value across Baselines:** $23.33\%$ (7 / 30).
- **Static vs Relocated Sub-populations:**
  - **Static Recall:** $\frac{6}{26} = 23.08\%$
  - **Relocated Recall:** $\frac{1}{4} = 25.00\%$
- **Distinction from Pair Recall:** A physical instance may contain multiple fragmented tracks (e.g., 3 tracks in Session 1, 2 tracks in Session 2). Pair-level recall counts pairwise matches, whereas physical instance recall deduplicates multiple tracks per physical object.

### Instance-Level F1 (%)
- **Definition:** Harmonic mean of instance-level precision and physical-instance recall.
- **Formula:**
  $$F_1 = 2 \cdot \frac{\text{Precision}_{inst} \cdot \text{Recall}_{inst}}{\text{Precision}_{inst} + \text{Recall}_{inst}}$$
- **Values:**
  - Baseline 1 (Centroid): $29.79\%$
  - Baseline 2 (Hungarian): $29.79\%$
  - Proposed (Uncertainty Symmetric): $31.11\%$
  - Proposed (Relocation Aware): $30.43\%$

### False Merge Rate (%)
- **Definition:** The proportion of candidate associations that incorrectly link tracks belonging to two distinct physical objects.
- **Formula:**
  $$\text{False Merge Rate} = \frac{FP_{merges}}{TP_{pairs} + FP_{merges}} \times 100$$
- **Values:** $41.67\%$ in Baseline 1, 2, 4 ($5/12$); $36.36\%$ in Baseline 3 ($4/11$).
- **Impact:** High false merge rate corrupts scene graph topology by collapsing distinct objects (e.g., merging two neighboring chairs) into a single entity.

### False Split Rate (%)
- **Definition:** The proportion of true multi-session physical instances that failed to be associated across sessions, remaining split into separate session-specific entities.
- **Formula:**
  $$\text{False Split Rate} = 100\% - \text{Physical Instance Recall} = \frac{30 - 7}{30} \times 100 = 76.67\%$$
- **Denominator:** 30 multi-session ground truth instances.
- **Caveat:** The codebase defines false split rate as the complement of instance recall ($76.67\%$). Do not confuse this with track-level oversegmentation within a single session.

---

## 2. Inter-Session SE(2) Registration Metrics

### Inlier RMS Residual (m)
- **Definition:** Root mean square Euclidean distance between registered stationary anchor centroids after RANSAC SE(2) alignment, evaluated strictly on consensus inliers.
- **Formula:**
  $$\text{RMS}_{inlier} = \sqrt{\frac{1}{|\mathcal{I}_{inliers}|} \sum_{i \in \mathcal{I}_{inliers}} \| R p_{1,i} + t - p_{2,i} \|^2}$$
- **Numerator:** Sum of squared residuals for 7 consensus inliers ($0.1004 \text{ m}^2$).
- **Denominator:** $|\mathcal{I}_{inliers}| = 7$ anchors.
- **Value:** $0.1198$ m (0.0864 m under post-refit strict 0.20m gate).
- **Source Artifact:** `benchmarks/registration/stage5_registration_audit.json`

### All-Anchor RMS Residual (m)
- **Definition:** Root mean square residual across all 10 candidate stationary anchors, including boundary outliers.
- **Value:** $0.3087$ m (pooled 10-anchor RMSE: $0.3538$ m).
- **Outlier Count:** 3 anchors rejected (residuals: $0.3897$ m, $0.5643$ m, $0.6180$ m) due to peripheral SLAM visual drift.

---

## 3. Controlled Relocation Metrics

### Absolute Displacement Error (m)
- **Definition:** Absolute difference between physical tape-measured ground truth relocation distance and SE(2)-registered estimated centroid shift.
- **Formula:**
  $$e_{abs} = |d_{est} - d_{gt}|$$
- **Evaluated Cases (N=3):**
  - `chair_002` (caster chair): $d_{gt} = 0.563$ m, $d_{est} = 0.6166$ m $\implies e_{abs} = 0.0536$ m.
  - `chair_003` (office chair): $d_{gt} = 0.386$ m, $d_{est} = 0.2612$ m $\implies e_{abs} = 0.1248$ m.
  - `storage box_007` (floor box): $d_{gt} = 0.396$ m, $d_{est} = 0.5917$ m $\implies e_{abs} = 0.1957$ m.
- **Mean Absolute Error:** $0.1247$ m.

### Mean Relative Displacement Error (%)
- **Definition:** Mean percentage error relative to true displacement.
- **Formula:**
  $$\text{Rel Error} = \frac{1}{N} \sum_{i=1}^N \frac{|d_{est,i} - d_{gt,i}|}{d_{gt,i}} \times 100 = 30.42\%$$
- **Source Artifact:** `metrics/raw/stage5_evidence_metrics.json`

---

## 4. Semantic Evidence Metrics

### Matched-Support Macro-F1
- **Definition:** Unweighted mean of per-class F1 scores across the 3 evaluated taxonomy categories (`IMMOVABLE`, `MOVABLE`, `MOBILE`), evaluated ONLY on observations with recognized classes.
- **Formula:**
  $$\text{Macro-F1} = \frac{1}{3} (F_{1,\text{IMMOVABLE}} + F_{1,\text{MOVABLE}} + F_{1,\text{MOBILE}}) = \frac{1.0 + 1.0 + 0.9928}{3} = 0.9976$$
- **Evaluated Population (Denominator):** $4,421$ matched observations.
- **Total Records in Dataset:** $7,956$ observations.
- **Evaluation Coverage:** $\frac{4,421}{7,956} = 55.57\%$.
- **CRITICAL CAVEAT:**  
  The saved score of `99.76%` is NOT whole-pipeline classification accuracy. $3,535$ records ($44.43\%$) have detector labels outside the 3 evaluated categories and are completely unmapped. Claiming $99.76\%$ pipeline semantic accuracy without stating the $55.57\%$ coverage is a fatal reporting defect.

---

## 5. Temporal Evidence & Tracking Metrics

### Track Fragmentation Rate (%)
- **Definition:** The proportion of unique tracker track IDs that span 4 or fewer consecutive frames before terminating.
- **Formula:**
  $$\text{Fragmentation Rate} = \frac{N_{tracks \le 4}}{N_{total\_tracks}} \times 100 = \frac{1,008}{1,385} \times 100 = 72.78\%$$
- **Impact:** High fragmentation creates bursts of short-lived tracks, preventing spatial Kalman smoothers from establishing stable centroids and causing cross-session spatial gates to fail.

### Occlusion Recovery Rate (%)
- **Definition:** Percentage of occlusion events where an interrupted track was successfully re-acquired.
- **Formula:** $\frac{355}{355} \times 100 = 100.0\%$.

---

## 6. Selective VLM Router Metrics

### Balanced Accuracy
- **Definition:** Arithmetic mean of recall across all evaluated classes on the validation instance set ($N=15$).
- **Formula:**
  $$\text{Bal Acc} = \frac{1}{K} \sum_{k=1}^K \text{Recall}_k$$
- **Values:**
  - Selective VLM: $88.89\%$ (IMMOVABLE: $66.7\%$, MOVABLE: $100\%$, MOBILE: $100\%$)
  - All-VLM (M3 unguided): $33.33\%$
  - Semantic Prior (M0 lookup): $100.0\%$ (on known dictionary vocabulary)
  - Router Disabled (Pure Prior): $66.67\%$

### VLM Query Rate (% Cost)
- **Definition:** The percentage of evaluated object instances for which the router escalated to query the cloud VLM.
- **Formula:**
  $$\text{Query Rate} = \frac{N_{queries\_invoked}}{N_{total\_instances}} \times 100 = \frac{4}{15} \times 100 = 26.67\%$$
- **Significance:** Demonstrates a $73.33\%$ reduction in multimodal inference calls relative to an all-VLM baseline while maintaining $88.89\%$ balanced accuracy.
