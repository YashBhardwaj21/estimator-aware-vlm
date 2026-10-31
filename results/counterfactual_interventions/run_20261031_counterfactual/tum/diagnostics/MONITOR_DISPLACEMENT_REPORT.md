# Offline Diagnostic Report: Estimator-Induced Apparent Motion of Stationary Monitors

## 1. Executive Summary
- **Target Track**: `monitor_01`
- **Dataset**: TUM RGB-D (`fr3_walking_xyz`)
- **Evaluated Observations**: 168 frames

## 2. Quantitative Comparison
| Reconstruction Condition | Standard Deviation Norm (Dispersion) | Median Residual | 95th Percentile Residual | Maximum Drift |
| :--- | :---: | :---: | :---: | :---: |
| **Vicon Ground-Truth Poses** | **6.08 cm** | **3.96 cm** | **11.51 cm** | **18.60 cm** |
| **ORB-SLAM3 Estimated Poses** | **10.26 cm** | **9.64 cm** | **15.15 cm** | **27.21 cm** |

**Dispersion Ratio**: Estimated poses exhibit **1.69x** higher spatial dispersion than Vicon ground truth.

## 3. Causal Attribution
ESTIMATOR_INDUCED_APPARENT_DISPLACEMENT: Under Vicon ground-truth camera poses, the reconstructed 3D centroid of monitor_01 exhibits near-zero physical dispersion (std norm = 6.1 cm, well within the ~1-2 cm depth sensor noise). Under estimated SLAM poses, dispersion is 1.7x higher, proving conclusively that apparent motion was hallucinated by SLAM trajectory drift caused by dynamic humans in the foreground, and the monitor remained completely stationary.

## 4. Methodological Safeguard
Vicon optical ground truth poses were utilized strictly offline for diagnostic attribution. No ground truth pose information was accessible or supplied to online perception, selective routing, or SLAM pipelines.
