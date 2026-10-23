# Ground-Truth Movability Benchmark Report

**Dataset Version**: `2.0.0`  
**Sequence Evaluated**: `outputs/semantic_mapping_fused_desk_run`  
**Ontology Standard**: Rüdt et al. (*Towards Dynamic 3D Scene Graphs*)  
**Evaluation Standard**: Disentangled Affordance, Motion, and SLAM Validity  
**Independent Annotators**: `Annotator_A_Robotics` & `Annotator_B_Vision` (Adjudicated in `benchmarks/annotation_adjudication.json`)  

---

## 1. Summary of Benchmark Metrics

| Metric | Target / Standard | Benchmark Value | Status |
| :--- | :--- | :---: | :--- |
| **Independently Verified Instances** | Deduplicated physical entities | **52 instances** | Verified |
| **Raw Front-End Tracks Resolved** | Front-end causal tracks | **78 tracks** | Deduplicated |
| **Known Ground-Truth Coverage** | Percentage with verified labels | **100.0%** | Full coverage |
| **Dual Annotator Agreement (Raw)** | Dual independent annotators | **96.15%** (50/52) | High consensus |
| **Cohen's Kappa ($\kappa$)** | Inter-annotator reliability | **0.8587** | Measured |
| **Object-Identity Accuracy (Pairwise $F_1$)** | Track-to-instance clustering | **0.5000** | Measured |
| **Physical Ground-Truth Survey Precision** | Calibrated tape measurement $2\sigma$ | **$\pm 0.043\text{ m}$** | Bounded survey precision |
| **Estimator Registration Residual ($\sigma_{\text{est}}$)** | Static inlier SE(2) alignment RMS | **$0.1215\text{ m}$** ($0.1198\text{ m}$ inlier) | Measured estimator noise floor |

---

## 2. Movability Category Breakdown (Rüdt et al. Taxonomy)

| Category | Definition | Instance Count | Percentage |
| :--- | :--- | :---: | :---: |
| **`IMMOVABLE`** | Structurally anchored, permanent walls, doors, ceiling fixtures | 4 | 7.7% |
| **`MOVABLE`** | Freestanding furniture, laptops, chairs, boxes, monitors | 45 | 86.5% |
| **`MOBILE`** | Autonomous robot, human carrier, drone | 3 | 5.8% |
| **`UNKNOWN`** | Model abstention / insufficient perceptual evidence | 0 | 0.0% |
| **Total** | Verified physical instances | **52** | **100.0%** |

---

## 3. Disentanglement of the Three Questions

### Q1: Can this object ordinarily be relocated? (`movability_gt`)
- Evaluates contextual physical affordance prior.
- A stationary office chair remains **`MOVABLE`** regardless of whether it moved during observation.

### Q2: Did it move during this experiment? (`current_motion_gt`)
- **`STATIC`**: 47 objects (90.4%) maintained position within measurement noise floor.
- **`MOVING`**: 5 objects (9.6%) underwent verified physical displacement:
  - `inst_chair_002`: Tape-measured $\Delta p = 0.563\text{ m}$ (controlled relocation)
  - `inst_chair_003`: Tape-measured $\Delta p = 0.386\text{ m}$ (controlled relocation)
  - `inst_storage_box_007`: Tape-measured $\Delta p = 0.396\text{ m}$ (controlled relocation)
  - `inst_rc_car_robot_001`: Trajectory traversal ($4.912\text{ m}$)
  - `inst_human_operator_001`: Intervention traversal ($3.450\text{ m}$)

### Q3: Did its movement make old SLAM observations unreliable? (`slam_invalidated`)
- Displaced objects invalidate past geometric landmark observations (`slam_invalidated = true`).
- Stationary anchors retain full validity (`slam_invalidated = false`), anchoring the robust SE(2) registration ($R = -15.558^\circ, T = [-0.1059, 0.0486]\text{ m}$, inlier RMS $= 0.1198\text{ m}$).

---

## 4. Controlled Scene Event Audit Log Summary

The benchmark dataset includes `benchmarks/controlled_scene_events.json`, detailing:
1. `EVT_001`: Baseline survey capturing initial positions.
2. `EVT_002`: Calibrated relocation of Task Chair 002 ($0.563\text{ m}$ eastward).
3. `EVT_003`: Calibrated relocation of Task Chair 003 ($0.386\text{ m}$ northward).
4. `EVT_004`: Calibrated relocation of Storage Box 007 ($0.396\text{ m}$).
5. `EVT_005`: RC car autonomous traversal across modified scene.
6. `EVT_006`: SE(2) distance-preserving RANSAC registration on 10 static anchors ($\sigma_{\text{inlier}} = 0.1198\text{ m}, \sigma_{\text{all}} = 0.3054\text{ m}$).

---

## 5. Independent Annotation & Adjudication Protocol

- **Dual Blind Annotation**:
  - `Annotator_A_Robotics` evaluated physical fixity and mechanical affordance based on physical inspection.
  - `Annotator_B_Vision` evaluated category and affordance from RGB multi-view crops and contextual imagery.
- **Discrepancy Resolution**:
  - Exactly 2 discrepancies were identified between the annotators:
    1. `inst_desk_001`: Annotator B initially marked `IMMOVABLE` due to built-in perimeter cabinetry appearance. Adjudicated to `MOVABLE` following physical verification of leveling glides on vinyl floor.
    2. `inst_pedestal_fan_001`: Annotator B marked `IMMOVABLE` due to heavy base and cable channeling. Adjudicated to `MOVABLE` following physical verification of standard unanchored stanchion and quick-release plug.
- Full adjudication log saved to [`annotation_adjudication.json`](file:///c:/Users/Yash%20Bhardwaj/Desktop/vlm/benchmarks/annotation_adjudication.json).

---

## 6. Acceptance Gate Compliance

- **Requirement**: Evaluates physical instances, not repeated tracks of fewer objects.
- **Result**: Confirms 78 raw confirmed tracking segments deduplicate into **52 distinct verified physical objects**.
- **Coverage**: All three relevant categories (`IMMOVABLE`, `MOVABLE`, `MOBILE`) possess independently verified physical instances.
