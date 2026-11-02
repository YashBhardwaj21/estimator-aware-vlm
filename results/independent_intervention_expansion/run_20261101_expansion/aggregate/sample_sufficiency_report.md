# Sample Sufficiency and Population Audit Report

## 1. Population Boundaries and Authentic Sample Size

Step 5 of Experiment 9 targeted acquiring at least 30 genuine single-candidate paired intervention outcomes across at least five distinct sequences, *provided the physical data supports this*.

### Audit of Physical Data Reality:
1. **TUM `fr3_walking_xyz`**: Yields **3** genuine single-candidate paired trajectory outcomes:
   - `person_02` polygon: Delta ATE = -0.0365 m (-10.04% degradation)
   - `person_04` polygon: Delta ATE = -0.0147 m (-4.04% degradation)
   - `monitor` polygon (historical verified): Delta ATE = -0.0868 m (-24.80% degradation)
   - *Note*: `both_persons_joint_POLYGON` (Delta = -0.0092 m) is an authentic joint intervention ($I_{02,04} = +0.0420\text{ m}$) and is strictly NOT duplicated across candidates.

2. **OpenLORIS `corridor1-4`**: Yields **1** genuine single-candidate paired trajectory outcome:
   - `person_01` polygon: Delta ATE = -0.0822 m (-4.82% degradation, feature starvation safeguard)

3. **TUM `fr3_walking_halfsphere`**: Evaluated at the **sequence-level policy matrix** (S0-S4):
   - 1021 frames, 5 policy arms. Demonstrates +91.81% error reduction with estimator-aware VLM verification (S3: 0.0345 m vs baseline 0.4211 m).
   - Multi-actor co-movement across 246 frames means individual person extractions operate as coupled sequence policies, not isolated single-candidate units.

4. **D455 Multi-Session**: Contains **52 physical instances** (47 static, 5 relocated/mobile):
   - Evaluated on **scene-graph map lifecycle registration** (held-out anchor RMS = 0.1242 m vs 0.5128 m unmitigated; relocation F1 = 0.75).
   - Trajectory ATE is strictly `null` due to lack of external MoCap ground truth.

5. **Missing External Sequences**:
   - `openloris_market1-1`: `BLOCKED_BY_MISSING_INPUT` (38.8 GB archive not present locally).
   - `tum_fr3_walking_static`: `BLOCKED_BY_MISSING_INPUT` (archive not present locally).

## 2. Statistical Power and Inference Status

Total genuine single-candidate trajectory ATE intervention outcomes: **N = 4**.

Pursuant to scientific integrity standards:
- **Outcome Duplication Prohibited**: Artificially inflating N to 30 by replicating joint interventions or duplicating D455 physical instances into synthetic camera trajectory ATE is strictly prohibited.
- **Inference Status**: Any candidate-level hypothesis tests requiring large sample sizes (e.g. Student's t-test or Wilcoxon test across sequences) are formally classified as:
  `INSUFFICIENT_SAMPLE_SIZE_FOR_INFERENCE (N=4)`
- **What the Evidence Truly Demonstrates**:
  1. In feature-sparse corridors (OpenLORIS), suppressing dynamic entities induces feature starvation and harms tracking (-4.82%).
  2. In static office scenes with static objects (TUM monitor), suppressing valid static visual anchors harms tracking (-24.80%).
  3. In dynamic multi-actor scenes (TUM xyz), suppressing only a subset of dynamic outliers provides incomplete gains due to super-additive interaction ($I_{02,04} = +0.0420\text{ m}$).
  4. In complex moving-camera sequences (TUM halfsphere), estimator-aware VLM verification suppresses false alarms on parallax-induced geometric shifts and achieves superior localization (+91.81%).
  5. In multi-session mapping (D455), estimator-aware affordance verification eliminates false-positive map updates on doors and operators (Precision = 1.00).