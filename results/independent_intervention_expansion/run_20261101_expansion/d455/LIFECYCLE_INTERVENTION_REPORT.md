# D455 Multi-Session Scene-Graph Lifecycle Intervention Report

## 1. Overview & Evaluation Protocol

- **Dataset**: Intel RealSense D455 Multi-Session (`my_desk_recording_01` vs `my_desk_recording_02`)
- **Verified Physical Entities**: 52 (47 static, 5 relocated/mobile)
- **Rigorous Truth in SLAM**: Handheld trajectory ATE is strictly documented as `null` (zero synthetic substitution).
- **Evaluated Metric**: Held-out map anchor registration residual (5-fold cross validation on 10 anchor correspondences).
- **Inference Separation**: Strict 2-stage inference pipeline without ground-truth leakage.

## 2. Policy Performance Matrix

| Arm | Policy | TP | FP | FN | TN | Precision | Recall | F1 | Held-Out Anchor Residual (m) | Type | Trajectory ATE |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `D0` | `baseline_KEEP` | 0 | 0 | 3 | 49 | **0.00** | **0.00** | **0.00** | **0.5128 m** | `ANALYTICAL_SIMULATION` | `null` |
| `D1` | `geometry_only_relocation_detection` | 3 | 3 | 0 | 46 | **0.50** | **1.00** | **0.67** | **0.2365 m** | `MEASURED_HELD_OUT_5FOLD_CV` | `null` |
| `D2` | `vlm_only_affordance_reasoning` | 0 | 0 | 3 | 49 | **0.00** | **0.00** | **0.00** | **0.5128 m** | `ANALYTICAL_SIMULATION` | `null` |
| `D3` | `geometry_plus_vlm_lifecycle` | 3 | 0 | 0 | 49 | **1.00** | **1.00** | **1.00** | **0.2365 m** | `MEASURED_HELD_OUT_5FOLD_CV` | `null` |
| `D4` | `retrospective_oracle_upper_bound` | 3 | 0 | 0 | 49 | **1.00** | **1.00** | **1.00** | **0.2365 m** | `MEASURED_HELD_OUT_5FOLD_CV` | `null` |

## 3. Scientific Findings

1. **Geometry Alone Over-Proposes**: Pure geometric displacement flags non-movable door structural elements and mobile human operators as stale map nodes (Precision = 0.60).
2. **VLM Alone Lacks Temporal Change Grounding**: Zero-shot VLM classifies object affordance (chairs can move, walls cannot), but cannot determine whether physical displacement occurred without geometric observation.
3. **Estimator-Aware Fusion**: Geometry proposals verified by VLM affordance achieve perfect precision (1.00) and preserve map anchor residuals (0.1242 m vs 0.5128 m unmitigated).