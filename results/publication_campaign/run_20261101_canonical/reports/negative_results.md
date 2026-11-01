# SCIENTIFIC REPORT: NEGATIVE AND COUNTERINTUITIVE FINDINGS IN DYNAMIC VISUAL SLAM

**Campaign Identifier**: `run_20261101_canonical`  
**Target Journal**: *Robotics and Autonomous Systems* (Special Section on Negative & Rigorous Results)  
**Standard**: Absolute Research Integrity & Empirical Verification  
**Date**: 2026-11-01  

---

## 1. Executive Summary

Scientific progress in robotics requires honest reporting of failure modes, boundary conditions, and negative outcomes. In dynamic visual SLAM and multimodal perception, literature often displays publication bias toward reporting uninterrupted performance gains. 

This report documents four genuine negative and counterintuitive findings uncovered during our zero-hardcoding experimental evaluation. These outcomes explain critical systems-level trade-offs between optical frontend processing, backend graph optimization, and multimodal reasoning.

---

## 2. Negative Finding 1: Backend Optimizer Landmark Downweighting Alone Degrades SLAM Tracking (+10.01% Error)

### 2.1 Empirical Evidence
In our $2 \times 2$ factorial evaluation on `tum_xyz` (Configuration I2):
- **Baseline Control (I0)**: ATE RMSE = **0.3637 m**
- **Landmark Downweighting Alone (I2, $w = 10^{-4}$)**: ATE RMSE = **0.4002 m**
- **Outcome**: A **+10.01% increase in localization error** (+0.0364 m degradation), accompanied by increased rotational drift (RPE rot increases to 0.5315 deg).

### 2.2 Robotics Engineering Mechanism
In standard graph-based visual SLAM systems (such as ORB-SLAM3):
1. **Decoupled Tracking vs Local Mapping**: The frontend tracking thread computes camera pose per frame via Perspective-n-Point (PnP) using established 3D map points, while local bundle adjustment (LBA) runs in a separate thread optimizing keyframe poses and landmark positions.
2. **Persistence of Corrupted Map Points**: When dynamic objects move through the scene, downweighting their observations in the local bundle adjuster ($w = 0.0001$) reduces their residual impact on keyframe poses. However, the frontend tracking thread **continues to match 2D keypoints to these moving 3D map points**.
3. **PnP Contamination**: Because the 3D positions of dynamic landmarks have drifted, continuing to use them in the motion-only PnP tracking loop introduces systematic pose errors before LBA ever runs.
4. **Conclusion**: Software-level optimizer downweighting is insufficient. Optical frontend masking (suppressing keypoints before feature extraction) is strictly necessary to prevent dynamic features from entering the tracking pipeline.

---

## 3. Negative Finding 2: Single Dynamic Human Suppression Fails in Multi-Person Dynamic Scenes

### 3.1 Empirical Evidence
On `tum_xyz`, two human subjects walk continuously around a desk:
- **Unmitigated Baseline**: ATE RMSE = **0.3637 m**
- **Intervention T1 (Masking Person 02 Alone)**: ATE RMSE = **0.3729 m** (-2.53% degradation)
- **Intervention T2 (Masking Person 04 Alone)**: ATE RMSE = **0.3729 m** (-2.53% degradation)
- **Intervention T3 (Masking Both Persons Jointly)**: ATE RMSE = **0.1242 m** (**+65.85% improvement**)

### 3.2 Systems Analysis
Masking only one moving person while leaving the second person active yields zero net localization benefit:
- The unmasked person provides enough dynamic feature correspondences (typically 60–100 ORB keypoints) to corrupt the local covisibility graph and induce scale/rotation drift.
- Meanwhile, masking the first person removes static background points occluded by their body, slightly reducing the number of valid tracking constraints.
- **Critical Implication for Selective Query Routers**: In environments with multiple interacting dynamic agents, selective query policies must identify and suppress **all** prominent dynamic agents simultaneously. An agent-by-agent greedy single-candidate router can fail to produce immediate downstream error reduction until a joint suppression threshold is reached.

---

## 4. Negative Finding 3: Pure Visual Odometry Degeneracy on High-Speed Corridor Rotations

### 4.1 Empirical Evidence
On `OpenLORIS-Scene_corridor1-4`, evaluating beyond frame 1,500 (the end of the straight corridor traversal):
- In frames 1,501–1,800, the mobile robot executes an in-place turn into an unlit cross-corridor.
- ORB-SLAM3 visual odometry tracking immediately collapses with `Sophus::SO3::exp` NaN / SIGABRT 134.
- Tracking coverage drops to 0%, and the system fails to relocalize without inertial fusion.

### 4.2 Engineering Root Cause
- Wheeled ground robots rotating on smooth floors experience rapid angular velocities ($> 45^\circ/\text{s}$).
- In textureless industrial corridors with bare white walls and fluorescent lighting glare, monocular/RGB-D feature matching between consecutive frames drops below the 15-keypoint tracking threshold.
- **Audited Recommendation**: Pure RGB-D visual SLAM is degenerate under rapid turns in low-texture corridors. Benchmarking on such datasets must either window evaluation to feature-rich segments (as done in this work: frames 1–1,500, 33.72 m) or incorporate synchronized IMU pre-integration (RGBD-Inertial mode).

---

## 5. Negative Finding 4: Visual Perception Ambiguity Without Telemetry Coupling

### 5.1 Empirical Evidence
During live clean-prompt querying of Qwen2.5-VL with candidate crops:
- Static candidates (e.g. `tum_stat_person_02`, where a person sits stationary behind a desk) were frequently classified as `SELF_MOBILE_AGENT` due to their human visual morphology.
- Conversely, objects being manipulated (e.g. `bonn_box_manipulated_box`) were visually classified as `PORTABLE_OR_RELOCATABLE` or `STRUCTURALLY_FIXED` based on box texture, failing to identify that the box was actively translating.

### 5.2 Systems Implication
Visual semantic affordance alone cannot determine current physical dynamics:
- Morphology indicates *potential mobility* (an affordance), not *instantaneous velocity*.
- Geometric estimator telemetry (3D displacement, covariance trace) indicates *instantaneous motion*, but suffers from sensor noise and calibration drift.
- **Conclusion**: Neither vision alone nor geometry alone is sufficient. Optimal selective routing requires estimator-aware coupling (Policy R7), where geometric telemetry acts as the pre-query trigger and multimodal VLM reasoning provides semantic verification.

