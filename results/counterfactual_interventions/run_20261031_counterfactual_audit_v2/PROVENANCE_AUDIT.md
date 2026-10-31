# Counterfactual Benchmark V2 Provenance Audit

**Run Identifier:** `run_20261031_counterfactual_audit_v2`  
**Generated UTC:** `2026-10-31T15:25:05Z`  
**Git Commit:** `3e77b74e4099e7cb2818b615858edf960f0a3250` (Branch: `main`)  
**Repository:** `https://github.com/YashBhardwaj21/estimator-aware-vlm`  

---

## 1. SLAM Estimator Environment & Executable Verification

All ORB-SLAM3 trajectory evaluations in this benchmark were executed inside an isolated WSL Ubuntu 22.04 environment using the official compiled binary:

- **Distribution:** `Ubuntu-22.04`
- **User:** `saturn`
- **Executable Path:** `/home/saturn/ORB_SLAM3/Examples/RGB-D/rgbd_tum`
- **Executable SHA-256:** `cd9162f319423483a93e1c4e33ffc86ec9bb5bc3e931ea44816ebc5db08e999c`
- **ORB-SLAM3 Commit:** `4452a3c4ab75b1cde34e5505a36ec3f9edcdc4c4`
- **Vocabulary Path:** `/home/saturn/ORB_SLAM3/Vocabulary/ORBvoc.txt`

---

## 2. Analysis Code Cryptographic Hashes

| Component / Script | SHA-256 Checksum | Lines | Size (Bytes) |
| :--- | :--- | :--- | :--- |
| `scripts/evaluate_tum_monitor_overlap_ablation.py` | `cd8f9ff1aaf495a1...` | 444 | 22639 |
| `scripts/evaluate_fast_feature_retention.py` | `6d6021e4c6c17f23...` | 376 | 19341 |
| `scripts/evaluate_graph_lifecycle_interventions.py` | `07260ebdb1893523...` | 42 | 2173 |
| `scripts/generate_counterfactual_audit_v2.py` | `c70dda49b5bbf2fd...` | 385 | 20893 |
| `scripts/plot_counterfactual_audit_v2.py` | `0e451081efdba29f...` | 152 | 7162 |
| `scripts/audit_and_generate_provenance_v2.py` | `6e2738bdc7b51b1e...` | 297 | 12710 |
| `src/estimator_aware_vlm/interventions/lifecycle.py` | `863f3ea4ec0f5071...` | 526 | 27313 |
| `src/estimator_aware_vlm/interventions/evaluation.py` | `117956e8874360ce...` | 153 | 7279 |
| `src/estimator_aware_vlm/interventions/schema.py` | `9641306d08523d8c...` | 146 | 6773 |
| `src/estimator_aware_vlm/evaluation/paired_metrics.py` | `f652041ef8c91a7f...` | 298 | 15051 |
| `src/estimator_aware_vlm/slam/runtime.py` | `f66e82496400a429...` | 235 | 10525 |

---

## 3. Dataset Configuration & Sensor Calibration Hashes

| Configuration / Sensor File | SHA-256 Checksum | Size (Bytes) |
| :--- | :--- | :--- |
| `configs/datasets/tum_rgbd_fr3_walking_xyz.yaml` | `2efc6508cf1d200dd426099ba9c5b95a10f9c239143b9d7741cb04955403cadb` | 1262 |
| `configs/datasets/openloris_corridor1_4.yaml` | `653ca21a03fdf2a4d8794b0a439f562f15905bd6f2c3062ebd05696c9974e18d` | 1334 |
| `configs/datasets/realsense_d455_recording_01.yaml` | `e89cc98e1f83aa576da69b0036ed4bd2a659756cf95474f0a9af76633a48e54f` | 1152 |
| `configs/datasets/realsense_d455_recording_02.yaml` | `dadc66653f2bdf505804821b03550cfa6c333b14de212d8048567428692c54dd` | 1152 |
| `configs/sensors/OpenLORIS_D400.yaml` | `d2a4cd6ebaf9636896c4b0f3265977f7681ab44e7165563ecdabbae42a2e1bd2` | 1726 |
| `configs/sensors/realsense_d455_indoor.yaml` | *MISSING* | - |
| `benchmarks/movability_ground_truth.json` | `d989c0503708527837d296f8393b1a5b4c898319136546bcda3418a6fe209cdc` | 39758 |

---

## 4. Evaluated Trajectories Provenance

| Trajectory File | SHA-256 Checksum | Poses |
| :--- | :--- | :--- |
| `tum/trajectories/tum_fr3_walking_xyz_baseline_KEEP_r01.txt` | `770669d2bc2ee7669f5c4911d6bce89b08282e9d41b9e15c8202e504955a6bf9` | 827 |
| `tum/trajectories/tum_fr3_walking_xyz_monitor_01_POLYGON_r01.txt` | *MISSING* | - |
| `tum/trajectories/tum_fr3_walking_xyz_monitor_01_BBOX_r01.txt` | *MISSING* | - |
| `openloris/trajectories/openloris_corridor1-4_baseline_KEEP_r01.txt` | `721dff86b198a7e93c1956b5b8c183ba9331d17dbc59352c8b1ccb7cba47763d` | 1500 |
| `openloris/trajectories/openloris_corridor1-4_person_01_POLYGON_r01.txt` | *MISSING* | - |
| `openloris/trajectories/openloris_corridor1-4_person_01_BBOX_r01.txt` | *MISSING* | - |

---

## 5. Execution Protocol

| Pipeline Step | Exact Command | Execution Environment |
| :--- | :--- | :--- |
| Task 1 & 2: Canonical Audit Table & Statistical Analysis | `python scripts/generate_counterfactual_audit_v2.py` | Windows PowerShell (Sandboxed / Standard) |
| Task 3: D455 Non-Oracle Scene-Graph Lifecycle & Anchor Integrity | `python scripts/evaluate_graph_lifecycle_interventions.py` | Windows PowerShell (Sandboxed / Standard) |
| Task 4: TUM Monitor Overlap 6-Condition Ablation in Live ORB-SLAM3 | `python scripts/evaluate_tum_monitor_overlap_ablation.py` | WSL Ubuntu-22.04 with /home/saturn/ORB_SLAM3/Examples/RGB-D/rgbd_tum |
| Task 5: Corrected FAST Feature Boundary Analysis (5px Margin) | `python scripts/evaluate_fast_feature_retention.py` | Windows PowerShell (Sandboxed / Standard) |
| Visual Synthesis: Publication Figures Generation | `python scripts/plot_counterfactual_audit_v2.py` | Windows PowerShell (Sandboxed / Standard) |
