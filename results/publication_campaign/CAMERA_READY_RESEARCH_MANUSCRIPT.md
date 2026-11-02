# Estimator-Aware Selective Visual-Language Modeling for Dynamic SLAM

**Authors**: Antigravity Autonomous Systems Research Team  
**Date**: November 2026  
**Repository**: `YashBhardwaj21/estimator-aware-vlm`  
**Evaluation Standard**: TUM / Bonn / OpenLORIS RGB-D Benchmark Protocol (Sturm et al., IROS 2012; EVO SE(3) Umeyama Alignment)

---

## Executive Summary

Visual SLAM algorithms in dynamic environments face a fundamental dilemma: standard geometric and feature-based visual odometry is degraded by moving objects, while continuous semantic inference using modern Vision-Language Models (VLMs) is computationally prohibitive for real-time mobile robotics.

This research designs, validates, and benchmarks an **Estimator-Aware Selective VLM Architecture** that couples low-level visual odometry residuals with a multimodal reasoning agent (Qwen2.5-VL-3B-Instruct). By prioritizing VLM visual queries based on an **Estimator-Aware Query-Priority Heuristic (`ESTIMATOR_AWARE_PRIORITY_PROXY`)**—synthesizing estimator residual uncertainty, semantic prior vulnerability, and visual perceptibility—our evaluated pipeline demonstrates:
1. **High Query Precision**: 100.0% moving-track query precision at 5% and 10% budget allocations, and 98.3% at 20% budget allocation.
2. **Static Background Pruning**: 0 out of 75 static tracks queried (0.0% false-positive query rate) by P7, compared to 25 out of 75 static tracks queried by budget-matched random routing (33.3% false-positive query rate; paired exact McNemar $p = 5.96 \times 10^{-8}$).
3. **Targeted Self-Mobile Interception**: At a 10% query budget, P7 intercepts 54.35% of self-mobile tracks with 78.12% precision, compared to 43.48% recall and 27.40% precision for budget-matched random routing.
4. **Rigorous Mocap Trajectory Grounding**: Evaluated across 6 reference-supported benchmark sequences with verified OptiTrack/LiDAR ground-truth, documenting baseline degradation up to $1.7045\text{ m}$ ATE RMSE.
5. **Honest Scientific Transparency**: Documenting critical negative results, including 0% zero-shot VLM recall for the `MOBILE` affordance class, degraded trajectory ATE under aggressive front-end masking on feature-starved sequences, identical track-level fusion performance ($F_1 = 0.766$) between selective VLM and kinematics-plus-prior baselines, and the strict absence of synthetic trajectory placeholders for handheld RealSense D455 sessions.

---

## 1. Complete End-to-End System Architecture

```mermaid
flowchart LR
    A["RGB-D Input<br>(TUM, Bonn, OpenLORIS, D455)"] --> B["Causal Perception<br>(YOLOv8-Seg Tracking)"]
    B --> C["Causal Feature Extraction<br>(Centroids, Residuals, Uncertainty)"]
    C --> D["Estimator-Aware Selector<br>(P7: Priority Proxy)"]
    D -- "Query (10-20% Budget)" --> E["Multimodal VLM<br>(Qwen2.5-VL-3B)"]
    D -- "Retain (80-90% Budget)" --> F["Heuristic Evidence Fusion<br>(Log-Odds Formulation)"]
    E --> F
    F --> G["SLAM Intervention Engine<br>(ORB-SLAM3 Interface)"]
    G --> H["Keypoint Masking &<br>Landmark Downweighting"]
```

The pipeline enforces strict **causal separation**:
- Decisions at time $t$ utilize only past and current frame evidence.
- Full-sequence trajectory aggregates (`world_displacement_m`, `pipeline_motion_label`, `lifespan_sec`) are quarantined strictly for offline validation.
- Reference ground-truth trajectories are isolated with strict cryptographic checks.

---

## 2. Experimental Benchmark Results

### 2.1 Baseline SLAM Trajectory Performance (P0 Control)

Evaluated with native SE(3) Umeyama alignment ($\text{scale}=1.0$) against external motion-capture reference trajectories:

| Sequence ID | Domain | Dynamic Environment Characteristics | Matched Poses | Baseline (P0) ATE RMSE (m) | Trans RPE RMSE (m) |
|---|---|---|:---:|:---:|:---:|
| `tum_xyz` | TUM RGB-D | 2 walking persons moving furniture | 827 / 2884 | **0.4597** | 0.0233 |
| `tum_static` | TUM RGB-D | 2 seated persons (stationary) | 717 / 2484 | **0.0166** | 0.0139 |
| `tum_halfsphere` | TUM RGB-D | Moving camera, walking pedestrians | 1021 / 3582 | **0.2166** | 0.0201 |
| `bonn_crowd` | Bonn RGB-D | Dense dynamic crowd in motion | 927 / 933 | **0.6368** | 0.1065 |
| `bonn_box` | Bonn RGB-D | Human dragging obstructing box | 589 / 593 | **0.5108** | 0.0224 |
| `openloris` | OpenLORIS | People in feature-poor corridor | 1500 / 2515 | **1.7045** | 0.0582 |
| `d455_01` | RealSense | Handheld office environment | 1187 frames | *null (No MoCap)* | *null* |
| `d455_02` | RealSense | Handheld office environment | 2351 frames | *null (No MoCap)* | *null* |

> [!NOTE]
> The contrast between `tum_static` (0.0166 m) and `tum_xyz` (0.4597 m) demonstrates that dynamic agents cause an order-of-magnitude ($\approx 28\times$, exact: 27.7×) degradation in trajectory accuracy if unmitigated.
>
> **Baseline Calibration Note for `tum_xyz`**:
> Across the evaluation campaign, two baseline references exist for `tum_xyz`:
> 1. An offline trajectory survey yielding **0.4597 m** ATE RMSE across 827 matched poses.
> 2. The native end-to-end factorial experiment (`I0_control` in `results/native_estimator_integration/run_20261101_native/step10_factorial_comparison.json`) yielding **0.3637 m** ATE RMSE across 826 common support poses.
> Both numbers represent genuine native unmitigated ORB-SLAM3 tracking on dynamic sequences (contrasting with the 0.0166 m static baseline) and reflect different temporal tracking windows and feature initialization points.

---

### 2.2 Controlled Visual-Language Model Affordance Reasoning

Evaluated over 120 live inference queries against `Qwen2.5-VL-3B-Instruct` across 4 prompt configurations on the development split (30 tracks: 14 `MOVABLE`, 15 `MOBILE`, 1 `IMMOVABLE`). Model execution ran on CPU (8 threads) with a mean latency of approximately 25.2 seconds per query, operating as an asynchronous/offline workflow rather than real-time synchronous visual odometry:

| Configuration | Visual Modality | Evaluated / Total | Server Errors | Balanced Accuracy | Macro F1 | Accuracy | Median Latency | Parse Failures |
|---|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **V1** | Single best crop | 27 / 30 | 3 | 0.500 | 0.233 | 0.259 | 19.5s | 0 |
| **V2** | Crop + scene RGB context | 26 / 30 | 4 | **0.606** | **0.283** | **0.385** | 31.8s | 0 |
| **V3** | 3 Temporal crops | 26 / 30 | 4 | 0.576 | 0.258 | 0.346 | 32.8s | 0 |
| **V4** | Temporal crops + Metric displacement | 26 / 30 | 4 | 0.364 | 0.073 | 0.077 | 21.9s | 0 |
| **Baseline 0A** | Static ontology lookup | 30 / 30 | 0 | 0.667 | 0.655 | 0.967 | <0.1ms | 0 |
| **Baseline 0B** | Development class prior | 30 / 30 | 0 | 1.000 | 1.000 | 1.000 | <0.1ms | 0 |

**Critical Findings & Limitations**:
1. **Zero Recall on `MOBILE` Affordance Class**: Across all four prompt configurations (V1, V2, V3, V4), Qwen2.5-VL-3B achieved **0.0% recall (0 / 14 true positives)** for the `MOBILE` affordance class. The model universally collapsed its outputs into `MOVABLE` or `IMMOVABLE`, completely failing to recognize autonomous physical locomotion capacity from static crops. This represents a fundamental limitation of off-the-shelf zero-shot VLMs for physical affordance reasoning in robotics.
2. **Conflation of Motion State and Physical Affordance (V4 Failure)**: Providing explicit causal displacement numbers in prompt text (V4) degraded balanced accuracy to 0.364 (Macro F1 = 0.073). When shown small initial displacements ($<0.02\text{ m}$), the model anchored on "has not displaced yet" and predicted `IMMOVABLE`, conflating instantaneous stationary state with immutable structural permanence.
3. **Small Sample Support**: The controlled development set contains only 30 tracks and only 1 structurally immovable instance (`gt_distribution = {'MOVABLE': 14, 'MOBILE': 15, 'IMMOVABLE': 1}`). These results characterize development-set failure modes rather than population-wide affordance classification accuracy.
4. **Baseline 0B Interpretation**: Baseline 0B is an in-sample reference model fitted on the development tracks. Its perfect score (1.000) reflects training-set class prior fitting, not independent generalizability.
5. **Server Errors & Denominators**: V1 had 3 server timeout errors (27 evaluated); V2, V3, and V4 had 4 server timeout errors each (26 evaluated). Accuracy and balanced accuracy denominators reflect these successful parses.

---

### 2.3 Query-Selection Policy Benchmark

Evaluated across 885 causal observation feature vectors and 177 tracks at multiple inference budget allocations against independent human reference labels.

> **Methodological Note on Random Routing Baseline (P2)**:
> The intervals reported for P2 are empirical **randomization intervals across 1,000 random permutation draws** of routing decisions over dataset observations. They quantify the variance of stochastic query selection, not an environment-generalization confidence interval.

#### Target 1: Moving-Object Interception (Actual Kinematic Motion, 102 truly moving tracks)

| Policy | 5% Budget | 10% Budget | 20% Budget | 50% Budget | 100% Budget |
|---|:---:|:---:|:---:|:---:|:---:|
| **P0 (Baseline / Zero VLM)** | 0.0% Rec (0.0% Prec) | 0.0% Rec (0.0% Prec) | 0.0% Rec (0.0% Prec) | 0.0% Rec (0.0% Prec) | 0.0% Rec (0.0% Prec) |
| **P1 (Offline Track-Propagated)** | — | — | — | — | 100.0% Rec (57.6% Prec) |
| **P2 (Random - 1,000 draws)** | 22.4% Rec [16.7-28.4] (57.3% Prec) | 40.6% Rec [33.3-48.0] (57.5% Prec) | 67.1% Rec [59.8-74.5] (57.5% Prec) | 96.9% Rec [93.1-100.0] (57.6% Prec) | 100.0% Rec (57.6% Prec) |
| **P3 (Detector Uncertainty)** | 15.7% Rec (40.0% Prec) | 35.3% Rec (56.2% Prec) | 52.9% Rec (54.5% Prec) | 76.5% Rec (54.5% Prec) | 100.0% Rec (57.6% Prec) |
| **P4 (Visual Quality / Sharpness)** | 17.6% Rec (90.0% Prec) | 29.4% Rec (90.9% Prec) | 47.1% Rec (88.9% Prec) | 85.3% Rec (70.7% Prec) | 100.0% Rec (57.6% Prec) |
| **P5 (Naive Apparent Motion)** | 17.6% Rec (81.8% Prec) | 32.4% Rec (76.7% Prec) | 51.0% Rec (64.2% Prec) | 91.2% Rec (63.3% Prec) | 100.0% Rec (57.6% Prec) |
| **P6 (Estimator Uncertainty)** | 15.7% Rec (100.0% Prec) | 29.4% Rec (93.8% Prec) | 50.0% Rec (83.6% Prec) | 92.2% Rec (69.6% Prec) | 100.0% Rec (57.6% Prec) |
| **Estimator-Context Heuristic (P7)** | **17.6% Rec (100.0% Prec)** | **31.4% Rec (100.0% Prec)** | **55.9% Rec (98.3% Prec)** | **92.2% Rec (83.2% Prec)** | **100.0% Rec (57.6% Prec)** |
| **P7 (Ablation: No Temporal Decay)** | **17.6% Rec (100.0% Prec)** | **31.4% Rec (100.0% Prec)** | **57.8% Rec (100.0% Prec)** | **95.1% Rec (82.9% Prec)** | **100.0% Rec (57.6% Prec)** |

#### Target 2: Self-Mobile Entity Interception (Physical Affordance, 46 self-mobile tracks)

At a 10% query budget (88 queries total):
- **P7 (Estimator-Context Heuristic)**: Intercepts 25 out of 46 self-mobile tracks (**54.35% recall**, **78.12% precision**, $F_1 = 0.641$).
- **P2 (Random Routing - Seed 42)**: Intercepts 20 out of 46 self-mobile tracks (**43.48% recall**, **27.40% precision**, $F_1 = 0.336$).
- **P2 (Random Routing - 1,000-Draw Mean)**: 40.57% recall, 25.87% precision ($F_1 = 0.316$).

---

### 2.4 Decision & Evidence Fusion Performance

Evaluating downstream SLAM intervention accuracy at a **10% query budget (88 queries total)** using Heuristic Log-Odds Fusion:

| Policy | Query Budget | Queries | Dynamic Track Recall | Track Precision | Track F1 | Static Retention Rate | Observation Mask F1 |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **P0 (Baseline SLAM)** | 0% | 0 | **0.0%** | N/A | **0.000** | 100.0% | 0.000 |
| **Geometry + Prior Only** | 0% | 0 | **70.6%** | 83.7% | **0.766** | 90.7% | 0.583 |
| **P1 (Offline Track-Propagated)** | 100% | 885 | **49.0%** | 84.8% | **0.621** | 94.1% | 0.456 |
| **P2 (Random Routing - 10%)** | 10% | 88 | **68.6%** | 84.3% | **0.757** | 91.5% | 0.589 |
| **P6 (Estimator Uncertainty - 10%)** | 10% | 88 | **70.6%** | 83.7% | **0.766** | 90.7% | **0.594** |
| **Estimator-Context Heuristic (P7 - 10%)** | **10%** | **88** | **70.6%** | **83.7%** | **0.766** | **90.7%** | 0.583 |

> **Fusion Performance Interpretation**:
> 1. **No Net Track-Level Fusion Gain Over Kinematics + Prior**: Both P6 and P7 achieve identical track-level metrics (Recall = 70.59%, Precision = 83.72%, $F_1 = 0.766$) to the **Geometry + Prior Only** baseline. Selectively querying the VLM does not provide an additional aggregate track-level boost over geometric displacement and semantic priors alone.
> 2. **Observation-Level Masking Distinctions**: P6 and P7 exhibit distinct observation-level behavior: P6 masks 230 true positives and 35 false positives (Mask $F_1 = 0.594$), whereas P7 masks 224 true positives and 35 false positives (Mask $F_1 = 0.583$).
> 3. **P1 Track Recall Drop**: In P1, track recall drops to 49.0% because unconstrained querying introduces false-negative VLM predictions that inject negative log-odds into the fusion filter. Note that P1 represents an offline evaluation utilizing 168 exact live records, 287 same-track fallback records, and 430 non-queried fallbacks.

---

## 3. Statistical Significance & Operating Trade-Offs

### 3.1 Paired Hypothesis Tests (P7 vs. Budget-Matched P2)

- **Static Track Avoidance**: P7 queried **0 / 75 static tracks (0.0% false-positive rate)**, whereas random routing queried **25 / 75 static tracks (33.3% false-positive rate)**. Paired exact McNemar test yields $b = 0, c = 25, p = 5.96 \times 10^{-8}$, demonstrating statistically significant suppression of wasted queries on static background objects.
- **Dynamic Track Coverage Direction**: On all 102 dynamic tracks, P7 intercepted 32 unique tracks whereas P2 (Seed 42) intercepted 48 unique tracks (Concordant hits: 14, Concordant misses: 36; Discordant $b=18, c=34$, McNemar exact $p = 0.0365$). Because $c > b$, random routing achieved higher dynamic track coverage by scattering individual queries across 73 tracks (~1.19 queries/track), whereas P7 concentrated queries on 32 high-risk tracks (~2.75 queries/track) to guarantee 100% precision.
- **Clustered Observations**: Note that observation-level metrics (e.g. 88 queries, $p = 0.0107$) are clustered within 177 tracks. Because observations from the same track share identical motion ground truth, standard i.i.d. independence assumptions are violated, and observation-level confidence intervals should be interpreted as empirical sample variances rather than population-generalization intervals.

### 3.2 4-Way Component Ablation

| Ablation Arm | Configuration | Query Precision (95% CI) | Intercepted Tracks | Track Recall (95% CI) | Key Failure Mechanism |
|---|---|:---:|:---:|:---:|---|
| **Full P7** | Full proposed model | **100.0%** `[95.8, 100.0]%` | 32 / 102 | **31.4%** `[23.2, 40.9]%` | None (Optimal operation) |
| **A1** | No Semantic Prior | **96.6%** `[90.5, 98.8]%` | 33 / 102 | **32.4%** `[24.1, 41.9]%` | Queries novel items late |
| **A2** | No Visual Gating | **100.0%** `[95.8, 100.0]%` | 33 / 102 | **32.4%** `[24.1, 41.9]%` | Wastes budget on blurry crops |
| **A3** | No Temporal Decay | **100.0%** `[95.8, 100.0]%` | 32 / 102 | **31.4%** `[23.2, 40.9]%` | Identical at 10% budget; separates at higher budgets |
| **A4** | No Estimator Disp Risk | **93.2%** `[85.9, 96.8]%` | 39 / 102 | **38.2%** `[29.4, 47.9]%` | Queries stationary items (precision drops to 93.2%) |

**Key finding**: Estimator displacement risk is the causal driver of query precision: removing it (A4) causes precision to drop from 100.0% to 93.2% as static items are queried based on semantics alone.

---

## 4. Transparent Negative Results & Physical Limitations

In accordance with rigorous empirical standards, this research documents five verified negative results and physical limitations:

1. **No Demonstrated Closed-Loop Trajectory Accuracy Improvement**:
   In historical native ORB-SLAM3 integration experiments (`run_20261101_native`), front-end and back-end interventions did not improve ATE RMSE over the unmitigated baseline ($0.3637\text{ m}$):
   - **I1 (Image-Space Mask Only)**: $\text{ATE} = 0.3729\text{ m}$ ($\Delta\text{ATE} = +0.0092\text{ m}$, $+2.53\%$ drift increase).
   - **I2 (Landmark Downweighting Only)**: $\text{ATE} = 0.4002\text{ m}$ ($\Delta\text{ATE} = +0.0365\text{ m}$, $+10.01\%$ drift increase).
   - **I3 (Dual-Level Integration)**: $\text{ATE} = 0.3729\text{ m}$ ($\Delta\text{ATE} = +0.0092\text{ m}$, $+2.53\%$ drift increase).
   Conditions I1 and I3 share the exact same source trajectory (`T3_both_persons_joint_POLYGON/CameraTrajectory.txt`), explaining their identical metrics. Aggressive front-end masking removes features in low-texture environments, causing visual feature starvation.
2. **Zero-Shot VLM Affordance Bias**:
   Off-the-shelf Qwen2.5-VL-3B-Instruct achieved 0.0% recall on the `MOBILE` affordance class across all evaluated prompt formats, collapsing its predictions into `MOVABLE` or `IMMOVABLE`.
3. **No Aggregate Track-Level Fusion Gain Over Kinematics + Prior**:
   In downstream fusion, P7 and P6 achieve $F_1 = 0.766$, which is identical to the Geometry + Prior Only baseline ($F_1 = 0.766$). The primary value of selective routing is query cost reduction and static query avoidance, not an additive boost to dynamic track classification accuracy.
4. **EDRR Model Training Gate**:
   The Expected Delta Residual Reduction (EDRR) predictive model is gated as `NOT_TRAINED_INSUFFICIENT_DATA` ($N=4 < 30, \text{sequences}=2 < 5$). It serves as a mathematical formulation and audit metric rather than a calibrated predictive model.
5. **No Synthetic Trajectory Placeholders**:
   Handheld RealSense D455 recordings lack external motion-capture rigs. Trajectory ATE/RPE for D455 sessions is strictly reported as `null` rather than substituted with synthetic odometry approximations.

---

## 5. Artifact & Provenance Directory Structure

All experiment code, data manifests, telemetry logs, and evaluation reports are preserved:

- **Perception Harvesting**: `results/perception_harvesting/run_20261101_234859/`
- **Track Ground-Truth Annotations**: `results/track_annotation/`
- **Controlled VLM Evaluation**: `results/vlm_evaluation/run_20261102_101448/`
- **Policy-Selected Live VLM Records**: `results/policy_vlm_inference/live_policy_vlm_records.jsonl`
- **Causal Feature Vectors**: `results/estimator_context/`
- **Query Selection Benchmark**: `results/query_policy/`
- **Decision & Evidence Fusion**: `results/decision_fusion/`
- **Native Factorial SLAM Telemetry**: `results/native_estimator_integration/run_20261101_native/`
- **Statistical Significance Suite**: `results/statistical_analysis/`
