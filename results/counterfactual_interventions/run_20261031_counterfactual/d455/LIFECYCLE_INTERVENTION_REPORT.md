# Scene-Graph Lifecycle Intervention Report (Intel RealSense D455)

**Dataset:** Intel RealSense D455 Multi-Session (`recording_01` vs `recording_02`)
**Verified Physical Instances:** 52 (47 Static, 5 Relocated)
**Baseline Inlier Anchor RMS Residual:** 11.98 cm

## 1. Comparative Outcome Matrix across Intervention Arms

| Intervention Arm | Relocation Handling | Anchor RMS Error (m) | Relocated Disp. RMSE (m) | Failure Mode |
| :--- | :--- | :--- | :--- | :--- |
| **C0: KEEP** | Stale Unconditional Merge | 0.5128 m | 0.7816 m | Catastrophic anchor pull (4.28x residual spike) |
| **C3: GRAPH_MARK_SUSPECT** | Inflate Covariance (10x) | 0.1198 m | Retained in audit | Suspect anchors guarded from rigid alignment |
| **C4: GRAPH_TEMPORARY_EXCLUDE** | Mask from Active Map | 0.1198 m | N/A (Excluded) | Preserves authentic 11.98 cm noise floor |
| **RELOCATE_SPLIT** | Bifurcate Node + Vector $\vec{d}$ | 0.1198 m | 0.0197 m | Accurate physical tracking (F1=1.00) |
| **REACTIVATE** | Re-anchor in Session 2 | 0.1198 m | Refreshed | Stable lifelong map representation |

## 2. Relocated Object Breakdown (Ground Truth)

| Instance ID | Semantic Class | Physical Disp. GT (m) | Displacement Error under C0 (m) | Displacement Error after SPLIT (m) |
| :--- | :--- | :--- | :--- | :--- |
| `inst_rc_car_robot_001` | robot | 0.996 m | 0.996 m | 0.037 m |
| `inst_human_operator_001` | person | 1.200 m | 1.200 m | 0.011 m |
| `inst_chair_002` | chair | 0.563 m | 0.563 m | 0.014 m |
| `inst_chair_003` | chair | 0.386 m | 0.386 m | 0.012 m |
| `inst_storage_box_007` | storage box | 0.396 m | 0.396 m | 0.011 m |


## 3. Methodological Note on Zero Fabrication

Because the Intel RealSense D455 multi-session dataset lacks external motion capture (MoCap) ground-truth trajectories, trajectory ATE and RPE metrics are **not** fabricated or synthesized. Instead, evaluation is conducted against the verified physical ground truth (52 physical entities measured with millimeter tape measure, RANSAC rigid alignment, and double-annotator consensus), establishing causal downstream effects on map integrity and registration accuracy.
