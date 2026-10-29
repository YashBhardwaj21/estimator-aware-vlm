# Research Implementation Audit: Estimator-Aware VLM for RGB-D SLAM

**Audit Date**: October 2026  
**Repository State**: Git HEAD `1fce23632c717b104f0952597c85b25273b5dd75` on branch `main`  
**System Environment**: Ubuntu 22.04 LTS (WSL2), Python 3.10.12, ORB-SLAM3 (`rgbd_tum`), local `llama-server` hosting `Qwen2.5-VL-3B-Instruct` (Q4_K_M GGUF)

---

## 1. Current Architecture & Execution Flow

The current repository implements a two-pass experimental SLAM & perception pipeline:
1. **Pass A (Unfiltered Geometric Baseline)**:
   - Synchronized RGB-D frames are ingested by `ORB-SLAM3` (`rgbd_tum` executable) without object filtering.
   - Outputs estimated camera trajectory `CameraTrajectory.txt`.
   - `scripts/run_baseline_slam.py` evaluates estimated poses against ground truth using rigid SE(3) Umeyama alignment (`evaluate_trajectory()`).
2. **Perception & 3D Sequence Tracking**:
   - `SequenceTracker` (`src/estimator_aware_vlm/perception/sequence_tracker.py`) ingests color frames and Pass A camera poses.
   - Detects object instances with `models/yolov8n-seg.pt` (COCO indoor classes).
   - Back-projects pixels to 3D camera coordinates using depth and intrinsics ($P_{\text{cam}} = K^{-1} [u, v, 1]^T \cdot Z$).
   - Compensates camera ego-motion using Pass A poses to track physical 3D displacement in world coordinates ($P_{\text{world}} = R_{wc} P_{\text{cam}} + t_{wc}$).
   - Extracts multi-view crops and deterministic per-frame polygon segmentation contours serialized to `per_frame_instance_observations.jsonl`.
3. **Selective Multimodal Routing Experiment**:
   - `scripts/run_routing_and_vlm_experiment.py` evaluates candidate entities across policies B0 through B6.
   - Queries a live local `llama-server` (`Qwen2.5-VL-3B-Instruct` on port 8089) for target crops with a strict JSON schema (`semantic_class`, `contextual_movability`, `reasoning`).
   - Exports `routing_decisions.json`.
4. **Pass B (Filtered SLAM Execution)**:
   - `scripts/run_pass_b_slam.py` ingests `routing_decisions.json` and `per_frame_instance_observations.jsonl`.
   - Masks only entities flagged as `MASK_FROM_TRACKING` using polygon contours (`cv2.fillPoly`).
   - Re-runs `ORB-SLAM3` on the masked sequence.
   - Evaluates Pass B trajectory against ground truth and computes error reduction relative to Pass A.

---

## 2. What Runs Successfully vs What Has Not Been Demonstrated

### What Runs Successfully:
- **TUM RGB-D (`fr3_walking_xyz`) Full Pipeline**:
  - Pass A Baseline ATE RMSE: `0.4372 m` (43.72 cm), 236 valid matched poses.
  - 10 candidate entities tracked; polygon masks extracted via `yolov8n-seg.pt`.
  - Live VLM queries executed with Qwen2.5-VL-3B-Instruct (10 entities queried, 6 dynamic flagged for masking).
  - Pass B Estimator-Aware ATE RMSE: `0.0167 m` (1.67 cm), matched 28.64% tracking coverage, **96.17% error reduction**.
- **OpenLORIS (`corridor1-4`) Pipeline**:
  - Calibration: RealSense D400 configuration (`configs/sensors/OpenLORIS_D400.yaml`) with `Stereo.b: 0.0745`.
  - Pass A Baseline SLAM: Executed on 1,500 frames; ATE RMSE: `1.6983 m` (169.83 cm), matched 1,500 ground-truth poses (59.60% sequence coverage).
  - Tracking: Discovered `person_01` (0.9100 m displacement, observed motion `MOVING`).
  - Live VLM Query: Classified `person_01` as `person (MOBILE)` in 32.95s.
  - Pass B SLAM: Executed on 1,500 frames with `person_01` polygon-masked; ATE RMSE: `1.6925 m` (169.25 cm).
- **Test Suite**: 20/20 targeted tests pass (including `test_occupancy_grid.py`, `test_tum_adapter.py`, `test_pass_b_causality.py`, `test_baseline_matching.py`, `test_sequence_tracker.py`).
- **Figure Generation**: All 9 publication figures generated and cataloged in `outputs/run_manifest.json`.

### What Has NOT Been Demonstrated:
- **Counterfactual-Calibrated Utility**: The current B6 decision function in `run_routing_and_vlm_experiment.py` uses heuristic thresholds ($\text{utility} = \Delta L_{\text{slam}} - 0.05 \cdot C_{\text{vlm}}$ where $\Delta L_{\text{slam}} = 0.5 \cdot \text{obs\_norm} + 0.5 \cdot \text{disp\_norm}$). This is a heuristic proxy, not a predictor trained or calibrated against empirical downstream SLAM counterfactual outcomes.
- **Controlled Observation-Treatment Interface**: The current implementation only supports pixel-level masking in Pass B. It does not support feature downweighting or map-point rejection within the SLAM optimizer itself.
- **Cross-Session Landmark Persistence**: Both available sequences are single-session recordings. The claim that estimator-aware VLM maintains long-term maps across environmental reconfigurations remains **UNTESTED** on empirical data.
- **Direct Estimator Telemetry**: Feature counts and inlier proportions are estimated via proxies or bounding-box heuristics rather than direct hooks into ORB-SLAM3's bundle adjustment graph.
- **Multi-Sequence Statistical Generalization**: Testing on two single sequences (TUM fr3 and OpenLORIS corridor1-4) is exploratory and does not establish statistical significance across robot domains.

---

## 3. Existing Dataset Support & Limitations

| Sequence | Raw Files on Disk | Adapter Status | Calibration & Intrinsics | Ground Truth Provenance | Limitations |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **TUM RGB-D** (`fr3_walking_xyz`) | 827 color PNGs, 827 depth PNGs, `rgb.txt`, `depth.txt`, `groundtruth.txt` | Fully implemented via `TUMRGBDAdapter` | Calibrated Kinect intrinsics ($f_x=535.4, f_y=539.2, c_x=320.1, c_y=247.6$). Depth scale: 0.0002 (5000/m). | High-precision MoCap (8 cameras, 100 Hz). 827 reference poses. | Office lab with walking humans. Single session; no environmental reconfiguration between sessions. |
| **OpenLORIS** (`corridor1-4`) | 1,500 color PNGs, 1,500 aligned depth PNGs, `color.txt` (1,889 entries), `aligned_depth.txt`, `groundtruth.txt` (2,517 poses) | Ingested via `TUMRGBDAdapter` with disk existence filtering | RealSense D400 color stream ($848 \times 480$, $f_x=611.451, f_y=611.486, c_x=433.204, c_y=249.473$). Depth scale: 0.001 (1000/m). | Wheel-robot platform with offline LiDAR-odometry SLAM ground truth in global coordinates. | First 1,500 frames on disk (rosbag truncated at 1,500). Pedestrians appear in only 10 frames (0.67%). Feature-poor corridor causes structural drift. |

---

## 4. Existing Metrics & Limitations

- **Absolute Trajectory Error (ATE)**:
  - Computed via Horn/Umeyama SE(3) rigid alignment (`align_trajectories_se3()`).
  - *Limitation*: When evaluated on an individually matched support, a policy that loses tracking during fast rotations can deceptively appear to have lower ATE if difficult frames are dropped from the evaluation. A common timestamp support must be enforced for strict paired A/B comparisons.
- **Relative Pose Error (RPE)**:
  - Computed frame-to-frame ($\Delta = 1$).
  - *Limitation*: Fixed-index delta ($\Delta=1$) does not account for variable temporal gap when frames are dropped. Fixed-time ($\Delta t = 1.0\text{ s}$) or fixed-distance RPE evaluation is needed.
- **Zero-Pose Handling**:
  - `compute_ate()` previously returned `0.0` when no poses matched. This must fail explicitly with an `EvaluationError` or `NaN`.

---

## 5. Existing Routing-Policy Behavior (B0–B6)

- **B0 (Blind RETAIN)**: Masks 0 candidates, queries 0 VLM calls.
- **B1 (Blind INVALIDATE)**: Masks 100% of candidate object tracks.
- **B2 (Geometry Only)**: Masks tracks where camera-compensated world displacement $\ge 0.30\text{ m}$.
- **B3 (VLM All)**: Queries live VLM for 100% of candidates; masks any entity predicted as `MOBILE` or (`MOVABLE` with displacement $> 0.15\text{ m}$).
- **B4 (Random)**: Queries a random subset of candidates.
- **B5 (Confidence / Ambiguity)**: Queries candidates with high spatial dispersion ($\sigma_d > 0.08\text{ m}$) or borderline displacement ($0.12 \le d \le 0.35\text{ m}$).
- **B6 (Estimator-Aware)**: Queries when utility $U(q) = \Delta L_{\text{slam}} - \lambda C_{\text{vlm}} > 0$.
- *Core Limitation*: B6 uses an uncalibrated proxy utility rather than a true counterfactual-calibrated benefit function.

---

## 6. Current Masking / Intervention Implementation

- **Mechanism**: Pixel-level dynamic feature suppression on RGB images (`cv2.fillPoly` on polygon contours).
- **Depth Handling**: Depth is currently untouched or zeroed in corresponding mask regions.
- **Causality**: Validated by `mask_causality_report.json` and cryptographic hashes.
- *Core Limitation*: This is an image-space input filter (Pass B input modulation), not a native SLAM backend landmark lifecycle management hook.

---

## 7. Important Scientific-Validity Issues & Risks

1. **Raw Data Immutability Violation**:
   - `associations_pass_b.txt`, `associations_pass_b.manifest.json`, and `masked_pass_b/` directories were written directly into `data/raw/` subdirectories.
   - *Requirement*: Raw dataset directories must remain strictly read-only. All generated associations, masks, and manifests must reside under `outputs/experiments/<experiment_id>/`.
2. **Missing Dedicated OpenLORIS Adapter**:
   - OpenLORIS was loaded through `TUMRGBDAdapter` by passing custom YAML paths. A dedicated `OpenLORISAdapter` should be implemented and registered via a factory to cleanly handle OpenLORIS-specific metadata, sensors.yaml, and LiDAR-groundtruth conventions.
3. **Uncalibrated Proxy Utility**:
   - $\Delta L_{\text{slam}}$ in B6 was computed from normalized observation counts and displacements, not calibrated against counterfactual trajectory improvements.
4. **Shared VLM Cache Accounting**:
   - When policies replay offline from a shared cache, reporting 0.00s latency for subsequent policies obscures hypothetical model costs. Latency accounting must distinguish live execution from replay accounting.
5. **No Ground Truth Leakage**:
   - Confirmed: Ground-truth poses are isolated to `TrajectoryRole.GROUND_TRUTH` and are never ingested by `SequenceTracker` or routing policies.

---

## 8. Ordered Implementation Roadmap & Acceptance Criteria

| Phase | Milestone | Priority | Acceptance Criteria |
| :--- | :--- | :---: | :--- |
| **Stage 1** | **Dataset-Independent Input & Immutability** | P0 | Dedicated `OpenLORISAdapter`; raw data directories are 100% read-only; strict 1-to-1 association; automated preflight tests pass. |
| **Stage 2** | **Isolated & Configurable SLAM Runtime** | P0 | ORB-SLAM3 paths configurable via CLI/env; isolated runs in `outputs/experiments/<id>/`; fail-closed on stale trajectories or exit codes. |
| **Stage 3** | **Trustworthy Trajectory Evaluation Metrics** | P0 | Paired evaluation on common timestamp support; non-zero error on failure; fixed-time RPE; tests pass for empty/mismatched trajectories. |
| **Stage 4** | **Auditable SLAM Evidence & Provenance** | P1 | Explicit provenance tags (`MEASURED_BACKEND`, `DERIVED`, `PROXY`) for all track fields; no invented telemetry. |
| **Stage 5** | **Counterfactual-Calibrated Query Utility** | P2 | Single-track intervention runs compute empirical benefit $\Delta L(i, a)$; utility calibrated without held-out leakage. |
| **Stage 6** | **Fair Policy Comparisons (P0–P6) & Cost Accounting** | P2 | Equal query budgets enforced; hypothetical cost calculated accurately during cache replay; clear separation of query vs intervention. |
| **Stage 7** | **Valid & Causally Interpretable Interventions** | P3 | Mask artifacts stored in experiment run dirs; box-mask vs polygon-mask ablation; full SLAM reruns. |
| **Stage 8–9** | **Hypothesis Testing & Cross-Dataset Case Studies** | P3 | Sequence capability tables generated; H1–H5 evaluated with honest limitations; Pareto accuracy-cost curves. |
| **Stage 10–14**| **Extensions, Verification & Final Research Report** | P4 | Clean publication figures; automated tests pass; reproducible CLI; final report in `docs/experimental_results_and_limitations.md`. |
