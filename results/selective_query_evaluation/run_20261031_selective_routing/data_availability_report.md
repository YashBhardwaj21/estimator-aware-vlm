# Data Availability and Candidate Population Report (Phase 1)

**Experiment Run ID:** `run_20261031_selective_routing`  
**Audit Timestamp:** `2026-10-31 16:08:18 UTC`  

---

## 1. Measured Candidate Populations

| Dataset | Unique Candidates | Image Eligible | Geometry Eligible | Human-Labelled | Independent Sequences | Reference Trajectory Type |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **tum_rgbd** | 10 | 10 | 10 | 10 | 1 | Vicon MoCap (100 Hz) |
| **openloris** | 1 | 1 | 1 | 1 | 1 | LiDAR Odometry |
| **realsense_d455** | 52 | 41 | 52 | 52 | 2 | Physical Ground Truth (Tape Calibrated; MoCap null) |

---

## 2. Experimental Capacity & Feasibility Analysis

### TUM RGB-D (`fr3_walking_xyz`)
- **Candidates:** Exactly 10 unique candidate tracks.
- **Eligibility:** 10/10 (100%) are image-eligible with extracted visual crops and scene context.
- **Experimental Role:** Primary dynamic evaluation sequence. Supports budget testing at $K \in \{0, 1, 3, 5, 10\}$ (corresponding to $K_0, K_{10}, K_{25}, K_{50}, K_{100}$).
- **Reference:** Vicon optical motion capture ($100\text{ Hz}$).

### OpenLORIS-Scene (`corridor1-4`)
- **Candidates:** Exactly 1 unique candidate track (`person_01`). Its 10 visible frames represent repeated observations of a single physical human, not 10 independent candidates.
- **Eligibility:** 1/1 (100%) image-eligible.
- **Experimental Role:** Feature-starvation case study. Supports binary query comparison ($K=0$ vs $K=1$). Cannot support multi-candidate ranking or random sampling distribution.
- **Reference:** LiDAR odometry (strictly not motion capture).

### RealSense D455 Multi-Session (`my_desk_multi_session`)
- **Candidates:** 52 verified physical entities across two recording sessions (`recording_01` and `recording_02`).
- **Eligibility:** 41/52 (78.8%) are image-eligible with verified multiview crops on disk. 11 entities (load-bearing walls, floor, unobserved fixtures) are non-visual or lack segmented crops.
- **Experimental Role:** Multi-session 3D scene-graph map integrity, relocation classification ($F_1$), and static-anchor alignment evaluation.
- **Reference:** Calibrated physical tape measurements (displacement RMSE) and held-out static anchors (SE(2) alignment residual). Camera trajectory ground truth is strictly `null` (zero fabrication enforced).

---

## 3. Candidate Identity Contract

Every candidate is assigned a globally unique composite identifier:
`candidate_id = dataset_id + '__' + sequence_id + '__' + session_id + '__' + physical_entity_id + '__' + track_id`

This contract eliminates collision between track IDs across datasets and prevents treating multiple frames of the same physical object as independent query candidates.