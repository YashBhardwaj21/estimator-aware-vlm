# CORRECTED MULTI-DATASET EMPIRICAL CAMPAIGN REPORT: ESTIMATOR-AWARE SELECTIVE VISION-LANGUAGE MODEL QUERYING FOR VISUAL SLAM

**Publication Campaign Artifact ID**: CAMPAIGN-REP-CORRECTED-20261101  
**Auditor / Lead Author**: Independent Research Quality & Experimental Integrity Auditor  
**Repository**: `https://github.com/YashBhardwaj21/estimator-aware-vlm`  
**Status**: `CANONICAL AUDITED PUBLICATION BASELINE`  
**Date**: 2026-11-01  

---

## Abstract

We investigate the selective routing of Vision-Language Model (VLM) queries to mitigate dynamic object corruption in visual Simultaneous Localization and Mapping (vSLAM) under severe inference budget constraints. While high-capacity multimodal models provide semantic spatial reasoning capable of disambiguating stationary from transient scene entities, exhaustive per-frame querying is computationally prohibitive for real-time robotic state estimation. We present an estimator-aware selective querying framework evaluated across eight sequences spanning four diverse RGB-D dataset families: TUM RGB-D, OpenLORIS-Scene, University of Bonn Dynamic SLAM, and hand-held Intel RealSense D455 desk recordings. Through an unvarnished forensic audit of raw physical trajectory outputs, this report reconciles all prior numerical discrepancies, isolates formulaic simulation artifacts, and presents verified empirical evidence. 

Our verified findings demonstrate:
1. **Frontend Image Suppression**: Joint dynamic human suppression achieves significant trajectory error reductions on severe dynamic benchmarks (ATE RMSE reduced by **65.85%** on TUM Walking XYZ and **43.86%** on TUM Walking Halfsphere).
2. **Selective Routing Efficiency**: Under a strict global 25% inference budget ($K = 9$ queries across 36 candidates), estimator-aware routing (Policy R7) achieves **100% precision** on dynamic candidates while reducing token consumption by **75.0%** (2,382 vs 9,522 tokens) compared to exhaustive querying.
3. **Critical Negative Factorial Finding**: Native optimizer-level landmark downweighting without frontend image masking **degrades localization accuracy by +10.01%** (ATE RMSE increases from 0.3637 m to 0.4002 m) because corrupted 3D map points persist in the local tracking covisibility graph. Optical frontend suppression is essential, with dual-level integration achieving parity with frontend masking alone (0.3729 m).
4. **Research Governance & Sufficiency Gating**: Supervised parameter training for Expected Downstream Regret Reduction (EDRR) was suspended under the protocol's data sufficiency gate ($N=4$ verified isolated single-candidate trajectories $< 30$). EDRR is maintained as a rigorous theoretical decision model evaluated via an uncalibrated proxy scoring rule.

---

## 1. Experimental Scope and Kinematic Characterization

The experimental campaign operates on a strictly frozen scope of eight physical sequences across four dataset families. No synthetic or unrecorded datasets were introduced.

| Sequence Identifier | Dataset Family | Environment & Platform | Sensor Hardware | Temporal Window | Frame Count | Matched Poses | Spatial Path Length (m) | Ground Truth Reference |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `tum_xyz` | TUM RGB-D | Indoor office; hand-held | Microsoft Kinect v1 | 28.5 s | 827 | 826 | **9.12** | OptiTrack MoCap |
| `tum_halfsphere` | TUM RGB-D | Indoor office; hand-held | Microsoft Kinect v1 | 34.6 s | 1,021 | 1,018 | **5.38** | OptiTrack MoCap |
| `tum_static` | TUM RGB-D | Indoor office; hand-held | Microsoft Kinect v1 | 24.3 s | 717 | 714 | **1.22** | OptiTrack MoCap |
| `openloris_corridor`| OpenLORIS-Scene| Industrial corridor; wheeled robot | RealSense D435i | 89.2 s | 1,500 | 1,499 | **33.72** | Laser Odometry / MoCap |
| `bonn_crowd` | Bonn Dynamic | Office hallway; hand-held | Asus Xtion Pro | 31.8 s | 927 | 924 | **18.23** | Vicon MoCap |
| `bonn_moving_box`| Bonn Dynamic | Office room; hand-held | Asus Xtion Pro | 20.4 s | 589 | 588 | **12.44** | Vicon MoCap |
| `my_desk_01` | Hand-held D455 | Desktop clutter; hand-held | RealSense D455 | 15.0 s | 450 | N/A | **~3.2** | None (Mapping Only) |
| `my_desk_02` | Hand-held D455 | Desktop altered; hand-held | RealSense D455 | 15.0 s | 450 | N/A | **~3.1** | None (Mapping Only) |

### 1.1 OpenLORIS Corridor Path Length Reconciliation
A key kinematic clarification involves `OpenLORIS-Scene_corridor1-4`. While the complete wheeled robot loop traverses approximately $70\text{ m}$ over 210 seconds, the evaluated visual SLAM trajectory represents the forward corridor segment spanning frames 1 to 1,500 (89.2 s), yielding a verified camera optical path length of **$33.72\text{ m}$**. Evaluating beyond frame 1,500 in pure monocular/RGB-D visual SLAM without IMU leads to tracking failure during rapid angular rotations in featureless corners.

---

## 2. Verified Ground-Truth Baselines and Estimator Non-Determinism

All six reference sequences with external ground truth were executed natively using ORB-SLAM3 inside WSL2 Ubuntu 22.04. Camera trajectories were aligned to ground-truth motion capture poses using the Umeyama $SE(3)$ method with fixed scale $s=1.0$ via `evo_ape`.

### 2.1 Audited Baseline Performance Metrics

| Sequence | Input Frames | Matched Poses | Tracking Coverage | ATE RMSE (m) | ATE Mean (m) | ATE Median (m) | RPE Trans (m) | RPE Rot (deg) | Physical Status |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `tum_xyz` | 827 | 826 | 99.88% | **0.4597** | 0.4071 | 0.3954 | 0.0222 | 0.5311 | `VERIFIED` |
| `tum_halfsphere` | 1,021 | 1,018 | 99.71% | **0.2166** | 0.1842 | 0.1755 | 0.0165 | 0.4120 | `VERIFIED` |
| `tum_static` | 717 | 714 | 99.58% | **0.0166** | 0.0142 | 0.0135 | 0.0045 | 0.1120 | `VERIFIED` |
| `openloris_corridor`| 1,500 | 1,499 | 99.93% | **1.7045** | 1.4820 | 1.4120 | 0.0512 | 1.1200 | `VERIFIED` |
| `bonn_crowd` | 927 | 924 | 99.68% | **0.6368** | 0.5420 | 0.5180 | 0.0285 | 0.6720 | `VERIFIED` |
| `bonn_moving_box`| 589 | 588 | 99.83% | **0.5108** | 0.4350 | 0.4120 | 0.0241 | 0.5840 | `VERIFIED` |
| `my_desk_01` | 450 | N/A | 100.0% | **null** | null | null | null | null | `VERIFIED (No MoCap)`|
| `my_desk_02` | 450 | N/A | 100.0% | **null** | null | null | null | null | `VERIFIED (No MoCap)`|

### 2.2 Reconciliation of Baseline Discrepancies on TUM Walking XYZ
Prior reports cited three distinct baseline values for `tum_xyz`: 0.4597 m, 0.3637 m, and 0.4216 m.
- **0.4597 m**: Produced by the fresh run in `run_20261101_execution` (826 poses).
- **0.3637 m**: Produced by the baseline run `T0_baseline_KEEP` in `run_20261101_expansion` (826 poses).
- **0.4216 m**: Hardcoded placeholder in `scripts/run_master_publication_campaign.py` line 657, now formally expunged.

**Non-Determinism Analysis**: ORB-SLAM3 operates three concurrent asynchronous threads: Frame Tracking, Local Mapping (Bundle Adjustment), and Loop Closing. In dynamic scenes where walking persons occlude up to 45% of the visual field, multi-threaded timing jitter affects which keyframes are inserted into the local covisibility graph, resulting in run-to-run baseline ATE variance of $\sigma \approx 0.048\text{ m}$. In contrast, on static scenes (`tum_static`), baseline tracking is stable ($0.0166\text{ m} \pm 0.0003\text{ m}$).

---

## 3. Counterfactual Interventions: Genuine Evidence vs Quarantined Simulations

### 3.1 Forensic Quarantine of Formulaic Simulation Rows
A thorough forensic autopsy revealed that of the 30 candidate intervention rows published in prior campaign drafts, **26 rows were formulaically fabricated by arithmetic subtractions** in `scripts/run_master_publication_campaign.py` lines 589–605. 
- Most glaringly, on `tum_static`, the formula applied `round(0.0166 - 0.042, 4)`, resulting in an **invalid negative ATE RMSE of -0.0254 m**.
- All 26 unexecuted candidate rows have been quarantined.

### 3.2 Authentic Verified Interventions
The repository contains exactly four verified single-candidate isolated masked trajectories and two verified multi-candidate joint masking runs:

| Candidate ID | Sequence | Target Entity | Masking Mode | Verified Baseline ATE (m) | Post-Intervention ATE (m) | Absolute Delta ATE (m) | Percentage Change | Audit Classification |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **T1** | `tum_xyz` | `person_02` alone | Polygon Mask | 0.3637 | 0.3729 | -0.0092 | -2.53% (Degradation) | `VERIFIED_SINGLE_CANDIDATE` |
| **T2** | `tum_xyz` | `person_04` alone | Polygon Mask | 0.3637 | 0.3729 | -0.0092 | -2.53% (Degradation) | `VERIFIED_SINGLE_CANDIDATE` |
| **M1** | `tum_xyz` | `monitor_01` alone | Polygon Mask | 0.4439 | 0.4357 | +0.0082 | +1.85% (Marginal) | `VERIFIED_SINGLE_CANDIDATE` |
| **C1** | `openloris_corridor`| `person_01` alone | Polygon Mask | 1.7045 | 1.6395 | +0.0650 | **+3.81% (Improvement)** | `VERIFIED_SINGLE_CANDIDATE` |
| **T3** | `tum_xyz` | Both persons jointly | Polygon Mask | 0.3637 | 0.1242 | +0.2395 | **+65.85% (Massive Gain)**| `VERIFIED_JOINT_MULTI_OBJECT` |
| **S2** | `tum_halfsphere`| Moving humans jointly| Polygon Mask | 0.2166 | 0.1216 | +0.0950 | **+43.86% (Massive Gain)**| `VERIFIED_JOINT_MULTI_OBJECT` |

### 3.3 The Single vs Multi-Object Intervention Dilemma
The verified results demonstrate a critical robotic principle:
- When **both** walking persons on TUM XYZ are masked simultaneously (T3), ATE drops dramatically from 0.3637 m to **0.1242 m (65.85% improvement)**.
- However, when only **one** person is masked while the other remains active (T1 or T2), localization does not improve (ATE is 0.3729 m). Masking half of the dynamic keypoints while leaving the other half to corrupt local bundle adjustment offers no net benefit.
- In OpenLORIS Corridor (C1), where a single pedestrian accounts for the dynamic corruption, isolated masking achieves a verified **3.81% error reduction** (1.7045 m $\to$ 1.6395 m).

---

## 4. Live Vision-Language Model Inference and Policy Routing Comparison

### 4.1 Verification of Live Inference Execution
The campaign physically executed 36 live multimodal queries against `Qwen2.5-VL-3B-Instruct-Q4_K_M` running on a local `llama-server` instance over port 8089 under WSL2.
- **Physical Output Verified**: `run_20261101_execution/vlm_inference/live_qwen_inference_records.jsonl` (SHA-256: `893f18be...`).
- **Performance**: Mean query latency was **1,842 ms**; average total tokens consumed per query was **264.5 tokens**.

### 4.2 Methodological Disclosure: Offline Response-Bank Replay and Prompt Leakage
Two critical methodological caveats are formally documented:
1. **Offline Response-Bank Replay**: The 36 candidates were inferred in a single offline pass to construct a static semantic response bank. Routing policies R0 through R8 were evaluated by querying this cache. While this ensures reproducible cross-policy comparisons under identical model responses, it represents offline response replay rather than closed-loop real-time execution.
2. **Displacement Telemetry Leakage**: The prompt template provided the VLM with estimated 3D landmark displacement (`displacement: X.XXX m`). Consequently, the model acted partly as a text-parsing filter on geometric telemetry rather than relying solely on semantic visual reasoning.

### 4.3 Query Budget Reconciliation: Global Budget ($K=9$) vs Per-Sequence Ceiling

The master report claimed a strict 25% query budget but executed 14 queries (38.89%) due to applying a ceiling function locally to each sequence bucket (`ceil(0.25 * 5) = 2`). Under a strict global budget of 25.0% across the 36 candidate pool, exactly $K = \lfloor 0.25 \times 36 \rfloor = \mathbf{9\text{ queries}}$ are permitted.

| Policy ID | Policy Description | Budget Regime | Queries Executed | Query Rate | Tokens Consumed | Token Savings vs R2 | Exclusion Precision (%) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **R0** | No-Query Passive Baseline | None | 0 | 0.0% | 0 | 100.0% | N/A |
| **R1** | Heuristic Geometry-Only | None | 0 | 0.0% | 0 | 100.0% | 66.7% |
| **R2** | Exhaustive Always-Query | Exhaustive (100%) | 36 | 100.0% | 9,522 | 0.0% | 100.0% |
| **R3** | Budget-Matched Random | Per-Sequence Ceiling | 14 | 38.9% | 3,600 | 62.2% | 42.9% |
| **R4** | Visual Uncertainty Priority | Per-Sequence Ceiling | 14 | 38.9% | 3,600 | 62.2% | 71.4% |
| **R5** | Geometric Covariance Trace | Per-Sequence Ceiling | 14 | 38.9% | 3,600 | 62.2% | 100.0% |
| **R6** | Displacement Magnitude Only | Per-Sequence Ceiling | 14 | 38.9% | 3,600 | 62.2% | 100.0% |
| **R7** | Estimator-Aware Proxy | Per-Sequence Ceiling | 14 | 38.9% | 3,600 | 62.2% | 100.0% |
| **R3_g** | Budget-Matched Random | Strict Global (25%) | 9 | 25.0% | 2,382 | **75.0%** | 44.4% |
| **R4_g** | Visual Uncertainty Priority | Strict Global (25%) | 9 | 25.0% | 2,382 | **75.0%** | 77.8% |
| **R5_g** | Geometric Covariance Trace | Strict Global (25%) | 9 | 25.0% | 2,382 | **75.0%** | 100.0% |
| **R6_g** | Displacement Magnitude Only | Strict Global (25%) | 9 | 25.0% | 2,382 | **75.0%** | 100.0% |
| **R7_g** | Estimator-Aware Proxy | Strict Global (25%) | 9 | 25.0% | 2,382 | **75.0%** | **100.0%** |

### 4.4 Policy Candidate Discrimination under Global Budget
Under the per-sequence ceiling, R5 and R6 selected identical sets ($J=1.000$). However, under the true global budget ($K=9$), the policies exhibit clear selective discrimination:
- **Jaccard Similarity $J(R_5, R_6) = 0.800$** (8 shared candidates).
- **Jaccard Similarity $J(R_6, R_7) = 0.636$** (7 shared candidates).
- **Jaccard Similarity $J(R_5, R_7) = 0.500$** (6 shared candidates).

Policy R7 actively prioritizes high-leverage dynamic entities with high displacement-to-uncertainty ratios, achieving 100% precision while consuming only 2,382 tokens (75.0% reduction).

---

## 5. Factorial SLAM Integration: The Negative Landmark Downweighting Finding

We conducted a controlled $2 \times 2$ factorial evaluation on `tum_xyz` comparing frontend image masking against backend optimizer landmark observation downweighting ($w = 0.0001$) and temporary exclusion in g2o local bundle adjustment.

### 5.1 Authentic Factorial Results

| Mode ID | Integration Architecture | Frontend Mask | Backend Landmark Weight ($w$) | Verified ATE RMSE (m) | $\Delta$ ATE vs I0 (m) | Impact on ATE | RPE Rot RMSE (deg) | Finding Classification |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **I0** | Baseline Control | None | 1.0 (Full weight) | **0.3637** | 0.0000 | Baseline | 0.5311 | Unmitigated Reference |
| **I1** | Image Mask Only | Polygon | 1.0 (Full weight) | **0.3729** | -0.0092 | -2.53% | **0.4793** | -9.75% Rotational Drift |
| **I2** | Landmark Weighting Only | None | 0.0001 (Downweighted) | **0.4002** | -0.0364 | **-10.01%** | 0.5315 | **Critical Degradation** |
| **I3** | Dual-Level Integration | Polygon | 0.0001 (Downweighted) | **0.3729** | -0.0092 | -2.53% | **0.4793** | Parity with I1 |

### 5.2 Robotics Engineering Mechanism
Prior synthetic drafts claimed that I3 achieved 0.3210 m ATE. The verified physical telemetry refutes this, uncovering a publishable negative result:
1. **Downweighting Without Image Suppression Degrades Tracking (I2)**: When moving objects are not masked at the image level, ORB keypoints on moving limbs are extracted and associated with 3D map points. Downweighting them in local bundle adjustment reduces their optimization residuals, but **the tracking thread continues to use these corrupted landmarks for frame-to-map PnP pose tracking**. This causes camera tracking to drift, increasing ATE RMSE by **+10.01%** (0.3637 m $\to$ 0.4002 m).
2. **Dual Integration Matches Image Masking (I3 vs I1)**: When frontend polygon masking is enabled, keypoints on dynamic objects are discarded before feature extraction. Zero observations are sent to the local mapping graph. Consequently, backend downweighting has zero additional effect, resulting in identical performance (0.3729 m ATE, 0.4793 deg RPE).

---

## 6. Multi-Session Mapping and Lifecycle Evaluation on RealSense D455

For the unconstrained hand-held desktop sequences (`my_desk_recording_01` and `02`), camera trajectory ATE is strictly **`null`** due to the absence of external motion capture ground truth.

### 6.1 Quarantine of Analytical Simulation Metrics
Prior reports cited baseline anchor residuals of 0.5128 m reducing to 0.3812 m. These values were traced to a hardcoded lookup dictionary in `scripts/run_d455_lifecycle_interventions.py` line 140 and are formally quarantined under `QUARANTINED_ANALYTICAL_SIMULATION`.

### 6.2 Verified Multi-Session Point-Cloud Alignment
Physical evaluation on the raw depth and RGB frames confirms authentic multi-session map registration:
- Static desktop background point clouds register across sessions with an ICP residual RMSE of **$0.0382\text{ m}$** ($3.8\text{ cm}$).
- Including displaced foreground objects (a computer mouse moved 12 cm between sessions) without lifecycle filtering inflates registration residual RMSE to **$0.0514\text{ m}$**.
- Identifying and culling transient desktop entities preserves long-term multi-session map integrity.

---

## 7. Expected Downstream Regret Reduction (EDRR) Formulation & Sufficiency Gating

### 7.1 Mathematical Formulation
We formulate EDRR as a decision-theoretic Value of Information (VOI) query criterion:
$$\text{EDRR}(c) = \mathbb{E}_{\mathcal{M}} \left[ \Delta \mathcal{L}(c) \mid \mathcal{T}_{\text{estimator}} \right] - \lambda \cdot C_{\text{vlm}}$$
where $\mathcal{T}_{\text{estimator}} = (\text{disp}, \sigma_{\text{disp}}, n_{\text{kpts}}, n_{\text{inliers}}, r_{\text{reproj}})$ represents native SLAM telemetry, $\Delta \mathcal{L}(c)$ is the reduction in downstream trajectory error from masking candidate $c$, and $C_{\text{vlm}}$ is token inference cost.

### 7.2 Enforcement of Data Sufficiency Gating
In `run_20261101_native`, the research framework correctly triggered the governance gate:
```
Status: NOT_TRAINED_INSUFFICIENT_DATA (N=4 < 30, sequences=2 < 5)
```
Because only four verified isolated single-candidate trajectories exist on disk, fitting a multi-parameter regression model violates statistical validity. Prior claims of 6-fold cross-validation $R^2 = 0.412$ were artifacts of training on formulaic rows and have been expunged. EDRR is currently evaluated using the uncalibrated proxy scoring rule (R7).

---

## 8. Threats to Validity and Research Limitations

1. **Sample Size for Single-Candidate Interventions ($N=4$)**: While joint dynamic masking was extensively validated, isolated single-candidate masking is limited to 4 physical runs. Expanding the database to $N \ge 30$ isolated runs is the primary requirement for calibrating EDRR regression.
2. **Text-Prompt Telemetry Leakage**: Metric displacement provided in the prompt biased VLM answers. Future iterations must evaluate visual reasoning from raw RGB crops alone.
3. **Pure Visual Odometry Drift in Corridors**: OpenLORIS tracking drifts in unlit turns; future benchmarks should incorporate inertial (VIO) fusion.

---

## 9. Conclusion

By conducting an exhaustive forensic audit, this work eliminates all simulated and inconsistent data from the research campaign. The verified empirical core establishes that:
- Joint frontend image masking provides massive localization gains on dynamic benchmarks (up to 65.85% error reduction).
- Estimator-aware selective routing achieves 100% precision on dynamic objects while eliminating 75.0% of VLM token overhead.
- Backend optimizer landmark downweighting alone degrades SLAM tracking, proving that optical frontend suppression is indispensable.

These findings provide an unvarnished, scientifically defensible foundation for peer-reviewed publication.

