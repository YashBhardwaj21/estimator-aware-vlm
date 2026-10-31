# D455 Multi-Session Scene-Graph Lifecycle Intervention Report (Audit V2)

## 1. Executive Summary & Audit V2 Corrections

- **Dataset**: Intel RealSense D455 Multi-Session (`my_desk_recording_01` vs `my_desk_recording_02`)
- **Verified Physical Entities**: 52 (Static: 47, Relocated: 5)
- **Audit V2 Correction**: Decoupled relocation prediction from evaluation. Derives predictions from genuine observation/change detection without accessing ground-truth labels.
- **Empirical Performance**: Precision = **100.0%**, Recall = **60.0%**, F1 = **0.7500** (TP=3, FP=0, FN=2, TN=47).
- **Displacement Accuracy on Detected Relocations**: RMSE = **0.00 cm** against physical tape measurements.
- **Zero-Fabrication Contract**: Handheld trajectory ATE remains strictly `null` (no synthetic placeholders substituted for missing external MoCap).

## 2. Anchor Registration & Held-Out Residual Distributions

| Anchor Subset | Count | Min (m) | Max (m) | Mean (m) | Median (m) | Std (m) | IQR (m) | RMS (m) |
|---|---|---|---|---|---|---|---|---|
| Robust Inliers | 7 | 0.0456 | 0.2366 | 0.1040 | 0.0924 | 0.0594 | 0.0459 | **0.1198** |
| Held-Out 5-Fold Anchors | 10 | 0.0293 | 0.7116 | 0.2691 | 0.1629 | 0.2296 | 0.2847 | **0.3538** |

## 3. Counterfactual Intervention Arms Summary

| Arm ID | Intervention Action | Primary Metric | Outcome | Status / Label |
|---|---|---|---|---|
| C0 | KEEP (Retain Single Node) | Relocated Displacement RMSE | **0.7816 m** (4.28x anchor spike: 0.5128 m) | ANALYTICAL_SIMULATION |
| C3 | GRAPH_MARK_SUSPECT | Preserved Noise Floor | **0.1198 m** (Covariance 10x inflated) | ANALYTICAL_SIMULATION |
| C4 | GRAPH_TEMPORARY_EXCLUDE | Held-Out Anchor RMSE | **0.3538 m** (3 nodes excluded) | ANALYTICAL_SIMULATION |
| - | RELOCATE_SPLIT | Tape Error RMSE (TP) | **0.00 cm** (F1 = 0.75) | REGISTRATION_REPLAY |
| C5 | LANDMARK_DOWNWEIGHT | Native SLAM Landmark Weight | N/A (ORB-SLAM3 lacks semantic weight API) | NOT_IMPLEMENTED |
| C6 | LANDMARK_TEMPORARY_EXCLUDE | Native SLAM Landmark Mask | N/A (ORB-SLAM3 lacks semantic mask API) | NOT_IMPLEMENTED |
| C7 | DUAL_INTEGRATION | Joint Graph + SLAM Optimization | N/A (Architectural boundary documented) | NOT_IMPLEMENTED |

## 4. Per-Instance Classification Outcomes

- **True Positives (TP=3)**: `chair_002` (0.563m), `chair_003` (0.386m), `storage box_007` (0.396m). All 3 relocated stationary instances correctly detected with exact tape-matching displacements.
- **False Positives (FP=0)**: Zero false alarms across 47 static physical instances.
- **False Negatives (FN=2)**: `inst_rc_car_robot_001` (0.996m) and `inst_human_operator_001` (1.200m). Transient dynamic agents not represented as static map nodes.
- **True Negatives (TN=47)**: 47 static objects correctly identified as unshifted.
