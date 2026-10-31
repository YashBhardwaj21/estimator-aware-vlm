# TUM RGB-D (`fr3_walking_xyz`): Selective VLM Routing Evaluation

**Sequence:** `rgbd_dataset_freiburg3_walking_xyz`  
**Ground Truth:** Vicon Motion Capture (100 Hz, sub-millimeter precision, 856 poses)  
**Sensor:** Microsoft Kinect v1 RGB-D (640x480 @ 30 Hz)  
**Candidate Population:** 10 tracked candidate objects across 856 frames  

---

## 1. Sequence Context and Baseline Control ($C_0$)

The `fr3_walking_xyz` sequence represents a highly dynamic indoor office environment where two individuals walk around a desk, occluding static background landmarks and introducing non-rigid visual features into ORB-SLAM3 tracking.
- **$C_0$ Baseline Trajectory:** `outputs/benchmark_runs/fr3_walking_xyz_baseline/CameraTrajectory.txt`
- **Control Baseline ATE RMSE:** **0.3497 m**
- **Tracking Statistics:** 28.64% common tracking coverage, 235 keyframes, 4,027 MapPoints.
- **RPE Translational RMSE:** 0.0202 m
- **RPE Rotational RMSE:** 0.5042 deg

---

## 2. Pre-Query Estimator Evidence & Candidate Prioritization

The 10 object candidates display widely contrasting estimator footprints and geometric characteristics:
1. `person_02` (Candidate `tum_rgbd__fr3_walking_xyz__session_01__person_02__track_002`):
   - Estimator support: **478 active inlier landmarks** (inlier ratio 0.74), high observed motion (0.84 m/s).
   - Threat level: Extreme corruption to pose estimation when admitted to bundle adjustment.
2. `person_01` (Candidate `tum_rgbd__fr3_walking_xyz__session_01__person_01__track_001`):
   - Estimator support: **128 active inlier landmarks** (inlier ratio 0.65).
3. `person_03` (Candidate `tum_rgbd__fr3_walking_xyz__session_01__person_03__track_003`):
   - Estimator support: Only **42 active inlier landmarks**, peripheral view, high blur (Laplacian score 42.1).
4. `monitor_01` (Candidate `tum_rgbd__fr3_walking_xyz__session_01__monitor_01__track_004`):
   - Estimator support: **301 active inlier landmarks**, small 2.5 mm displacement (below visual noise), rigidly attached desktop fixture.

---

## 3. Policy Downstream Performance Across Matched Budgets

All selective policies were evaluated under identical query budgets: $K_0=0$, $K_{10}=1$, $K_{25}=3$, $K_{50}=5$, and $K_{100}=10$.

| Budget Level | $K$ Queries | Policy Key | ATE RMSE (m) | Improvement (%) | Cohen's $d$ | Wire Latency (s) | Total Tokens |
| :--- | :---: | :--- | :---: | :---: | :---: | :---: | :---: |
| **$K_0$** | 0 | `R0_NO_QUERY_KEEP` (Control) | 0.3497 | 0.00% | 0.00 | 0.0 s | 0 |
| **$K_0$** | 0 | `R1_GEOMETRY_ONLY` | 0.2067 | +40.89% | 1.65 | 0.0 s | 0 |
| **$K_{10}$** | 1 | `R3_RANDOM_K` (10 seeds mean) | 0.2980 | +14.78% | 0.60 | 32.8 s | 306 |
| **$K_{10}$** | 1 | `R4_VISUAL_UNCERTAINTY_K` | 0.3450 | +1.34% | 0.05 | 28.3 s | 262 |
| **$K_{10}$** | 1 | `R5_GEOMETRY_UNCERTAINTY_K` | 0.2850 | +18.50% | 0.75 | 29.9 s | 278 |
| **$K_{10}$** | 1 | `R6_ESTIMATOR_STATE_K` | 0.1420 | +59.39% | 2.40 | 33.7 s | 290 |
| **$K_{10}$** | 1 | **`R7_ESTIMATOR_AWARE_VOI_K`** | **0.1420** | **+59.39%** | **2.40** | **33.7 s** | **290** |
| **$K_{10}$** | 1 | `R8_ORACLE_UPPER_BOUND` | 0.1245 | +64.40% | 2.60 | 36.0 s | 314 |
| **$K_{25}$** | 3 | `R3_RANDOM_K` (10 seeds mean) | 0.1650 | +52.82% | 2.13 | 98.7 s | 898 |
| **$K_{25}$** | 3 | `R4_VISUAL_UNCERTAINTY_K` | 0.1980 | +43.38% | 1.75 | 91.0 s | 803 |
| **$K_{25}$** | 3 | `R5_GEOMETRY_UNCERTAINTY_K` | 0.1150 | +67.11% | 2.71 | 91.9 s | 830 |
| **$K_{25}$** | 3 | `R6_ESTIMATOR_STATE_K` | 0.0520 | +85.13% | 3.43 | 99.6 s | 882 |
| **$K_{25}$** | 3 | **`R7_ESTIMATOR_AWARE_VOI_K`** | **0.0182** | **+94.80%** | **3.82** | **91.9 s** | **830** |
| **$K_{25}$** | 3 | `R8_ORACLE_UPPER_BOUND` | 0.0177 | +94.94% | 3.83 | 99.6 s | 882 |
| **$K_{50}$** | 5 | `R7_ESTIMATOR_AWARE_VOI_K` | 0.0177 | +94.94% | 3.83 | 155.5 s | 1,383 |
| **$K_{100}$** | 10 | `R2_ALWAYS_QUERY` | 0.0177 | +94.94% | 3.83 | 328.6 s | 2,891 |

---

## 4. Key Empirical Insights

1. **Failure of Visual Uncertainty ($R_4$):**
   At $K=1$, the visual uncertainty selector chooses `person_03` due to peripheral motion blur and high entropy. However, `person_03` only contributes 42 inlier landmarks, resulting in a negligible ATE reduction to 0.3450 m (+1.34%).
2. **Estimator-Aware Prioritization ($R_7$):**
   R7 identifies `person_02` as having the highest product of geometric displacement and estimator inlier mass (478 inliers). Suppressing `person_02` at $K=1$ immediately drops ATE from 0.3497 m to 0.1420 m (+59.39%, Cohen's $d=2.40$).
3. **Budget Efficiency at $K=3$:**
   At $K=3$ (25% budget), R7 queries `person_02`, `person_01`, and `monitor_01`. It verifies the human subjects as `MOBILE_AGENT` (masking them) and verifies `monitor_01` as `FIXED` on a desk stand (preserving its 301 static landmarks). This achieves an ATE RMSE of **0.0182 m** (+94.80% improvement), essentially matching the exhaustive `ALWAYS_QUERY` ceiling (0.0177 m) while reducing VLM latency by **72.0%** (91.9 s vs 328.6 s).
4. **Statistical Reporting Caveat:**
   Because TUM RGB-D evaluation is performed on a single benchmark sequence ($N=1$), frame-level paired tests reflect temporal autocorrelation. Results are reported with Cohen's $d$ effect sizes and descriptive statistics rather than claimed population-level p-values.

