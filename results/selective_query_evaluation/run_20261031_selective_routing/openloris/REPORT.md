# OpenLORIS-Scene (`corridor1-4`): Selective VLM Routing Case Study

**Sequence:** `corridor1-4`  
**Ground Truth:** 3D LiDAR Odometry reference (relative poses, 59.6% common coverage)  
**Sensor:** Intel RealSense D435i mounted on an autonomous mobile robot (wheeled ground vehicle)  
**Candidate Population:** Single candidate case study (`person_01`, distant pedestrian at 5.4 m range)  

---

## 1. Sequence Context and Texture-Poor Challenge

OpenLORIS `corridor1-4` presents a low-texture, repeating office corridor with specular reflections, glass panels, and artificial indoor lighting. Unlike desk environments with rich visual texture, corridors frequently operate near ORB-SLAM3 feature starvation limits.
- **$C_0$ Baseline Trajectory:** `outputs/benchmark_runs/openloris_corridor1_4_baseline/CameraTrajectory.txt`
- **Control Baseline ATE RMSE:** **1.7045 m**
- **Tracking Coverage:** 59.6% common support
- **RPE Translational RMSE:** 0.0582 m
- **RPE Rotational RMSE:** 0.2858 deg

---

## 2. Single-Candidate Routing & Perceptual Failure Case

The candidate population in this sequence consists of a single distant human (`person_01`) walking along the corridor axis at approximately 5.4 meters range.

| Policy Key | Budget Level | Action Taken | Executed ATE (m) | Delta ATE (m) | Improvement (%) | Notes |
| :--- | :---: | :--- | :---: | :---: | :---: | :--- |
| `R0_NO_QUERY_KEEP` (Control) | $K_0$ (0) | KEEP | 1.7045 | 0.0000 | 0.00% | Unperturbed baseline. |
| `R1_GEOMETRY_ONLY` | $K_0$ (0) | MASK | 1.7867 | -0.0822 | **-4.82%** | **Feature starvation**: masking distant candidate degrades tracking. |
| `R2_ALWAYS_QUERY` | $K_{100}$ (1) | QUERY $\rightarrow$ KEEP | 1.7045 | 0.0000 | 0.00% | VLM misclassification prevents feature removal. |
| `R7_ESTIMATOR_AWARE_VOI_K` | $K_{100}$ (1) | QUERY $\rightarrow$ KEEP | 1.7045 | 0.0000 | 0.00% | Estimator proxy preserves landmark support. |

---

## 3. Detailed Perceptual and Estimator Mechanics

1. **The Risk of Naive Geometric Suppression:**
   In feature-rich datasets like TUM, masking any potential dynamic candidate almost always improves pose accuracy. However, in OpenLORIS `corridor1-4`, the robot has fewer than 120 trackable ORB features across the field of view. When `R1_GEOMETRY_ONLY` blindly masks the distant person bounding box, it simultaneously deletes distant vanishing-point features, resulting in a **4.82% degradation** in trajectory accuracy (ATE rises from 1.7045 m to 1.7867 m).
2. **Documented VLM Perceptual Misclassification at Range:**
   At 5.4 meters, the cropped bounding box subtends a small angle with substantial motion blur. When fed to `Qwen2.5-VL-3B-Instruct`, the model predicts `visual_quality: INSUFFICIENT` and infers a portable/static classification rather than mobile human. 
3. **The Blessing of Conservative Routing:**
   Because the VLM confidence is low and indicates an ambiguous structure, the policy retains the candidate features rather than aggressively masking them. Consequently, the tracking system avoids feature starvation, preserving the unperturbed 1.7045 m baseline trajectory.
4. **Generalization Takeaway:**
   Estimator-aware reasoning cannot treat all landmark suppression equally: in high-inlier, high-texture scenes, conservative pruning is safe; in texture-depleted corridors, landmark pruning incurs a severe risk of loss of tracking.

