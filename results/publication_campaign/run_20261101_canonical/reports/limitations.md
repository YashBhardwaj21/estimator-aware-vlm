# COMPREHENSIVE EXPERIMENTAL LIMITATIONS AND THREATS TO VALIDITY

**Campaign Identifier**: `run_20261101_canonical`  
**Target Journal**: *Robotics and Autonomous Systems*  
**Auditor**: Independent Research Quality & Experimental Integrity Auditor  
**Date**: 2026-11-01  

---

## 1. Executive Summary

This report documents the methodological boundaries, structural constraints, and threats to validity of this empirical study. Transparent reporting of limitations ensures that reviewers and readers can assess the generalizability of our findings without over-interpreting experimental scope.

---

## 2. Inventory of Experimental Limitations

### 2.1 Sample Size for Single-Candidate Counterfactual Interventions ($N = 4$)
- **Current State**: The repository contains exactly four verified single-candidate isolated masked trajectories across two sequences (`tum_xyz` and `openloris_corridor`).
- **Impact on EDRR**: While decision-theoretic Expected Downstream Regret Reduction (EDRR) is fully formulated as a theoretical Value of Information framework, supervised regression training of its 8 parameters was suspended under the protocol's data sufficiency gate ($N < 30$, sequences $< 5$).
- **Mitigation in Paper**: EDRR is evaluated via an uncalibrated estimator-aware priority proxy (Policy R7), which achieves 100% precision on dynamic candidates under the global 25% query budget. Expanding the physical database to $N \ge 30$ isolated runs represents immediate future work.

---

### 2.2 Absence of Ground-Truth Motion Capture on Hand-Held RealSense D455 Sequences
- **Current State**: `my_desk_recording_01` and `my_desk_recording_02` were captured using a hand-held Intel RealSense D455 in a realistic desktop environment without OptiTrack or Vicon tracking marker rigs.
- **Protocol Enforcement**: Camera Trajectory ATE RMSE is strictly reported as **`null`**.
- **Scope Restriction**: These sequences validate multi-session semantic map persistence and landmark outlier rejection via point-cloud ICP ($3.8\text{ cm}$ static alignment residual), not trajectory estimation accuracy.

---

### 2.3 Visual Odometry Sensitivity to Fast Angular Rotations in Low-Texture Corridors
- **Current State**: In `OpenLORIS-Scene_corridor1-4`, the mobile robot traverses a feature-rich corridor before rotating rapidly into an unlit cross-corridor.
- **Evaluation Windowing**: Evaluation was restricted to frames 1–1,500 ($33.72\text{ m}$ camera path), where visual feature tracking remained unbroken. Beyond frame 1,500, pure RGB-D visual SLAM without IMU experiences tracking loss due to motion blur and featureless white walls.
- **Future Direction**: Full loop evaluation across all 70 meters requires visual-inertial sensor fusion (RGBD-Inertial mode).

---

### 2.4 Multimodal Model Inference Latency on Embedded Hardware
- **Current State**: In our local deployment, `Qwen2.5-VL-3B-Instruct` executes on CPU under WSL2, requiring approximately **16.8 seconds per multimodal query**.
- **Systems Implications**: While selective query routing (Policy R7) reduces query frequency and token consumption by **75.0%**, a 16.8-second latency is too slow for synchronous high-frequency frame tracking (30 Hz).
- **Architectural Solution**: The VLM query loop must operate as an asynchronous background reasoning thread that provides periodic object lifecycle annotations to the real-time SLAM backend. Hardware acceleration on edge platforms (e.g., NVIDIA Jetson Orin with TensorRT-LLM) is required for real-time edge robotics.

---

### 2.5 Offline Response-Bank Replay vs Closed-Loop Streaming
- **Current State**: In the policy comparison experiment, candidate entities were evaluated using an offline semantic response bank to guarantee identical perception outputs across policies R2 through R8.
- **Validity Consideration**: While response-bank evaluation is standard for reproducible benchmark comparison, it does not evaluate dynamic feedback loops where an earlier masking decision alters the subsequent tracking trajectory and changes which future objects are observed.

