# Native Landmark Integration Audit & Feasibility Diagnostic

**Benchmark Run ID:** `run_20261031_counterfactual`  
**Target SLAM Subsystem:** ORB-SLAM3 (RGB-D Pinhole Pipeline via `rgbd_tum`)  
**Evaluator Status:** `NOT_IMPLEMENTED` (Intervention Arms C5, C6, C7)

---

## 1. Executive Summary & Research Governance

In accordance with strict research governance and zero-fabrication constraints, this repository distinguishes between:
1. **Higher-Level Scene-Graph Lifecycle Interventions (C3, C4, RELOCATE_SPLIT, REACTIVATE)**: Fully implemented and verified on the 52 canonical entities of the D455 multi-session dataset.
2. **Image-Space Feature Masking Interventions (C1, C2)**: Fully implemented and verified across TUM RGB-D (`fr3_walking_xyz`) and OpenLORIS-Scene (`corridor1-4`).
3. **Native SLAM Optimizer Landmark-Level Integration (C5, C6, C7)**: Formally audited and recorded as `NOT_IMPLEMENTED`.

**Explicit Policy Guarantee:** This repository **does not** claim native ORB-SLAM3 landmark-level observation downweighting or optimizer edge weighting when only scene-graph lifecycle changes and image-space pixel masking have been implemented.

---

## 2. Source-Code Architecture Inspection (`/home/saturn/ORB_SLAM3`)

An exhaustive audit of the ORB-SLAM3 codebase in the execution environment was conducted:

### 2.1 Monolithic Executable Boundary (`rgbd_tum.cc`)
The executable `rgbd_tum` accepts four positional CLI arguments:
```bash
./rgbd_tum path_to_vocabulary path_to_settings path_to_sequence path_to_association
```
It reads image files frame-by-frame and passes them directly to `ORB_SLAM3::System::TrackRGBD(...)`. There is no IPC mechanism, shared memory interface, Unix domain socket, or dynamic configuration channel to pass per-landmark semantic weights, confidence scores, or exclusion lists from an external Python/VLM process into the running tracking or optimization threads.

### 2.2 Optimizer Residuals and Information Matrices (`src/Optimizer.cc`)
In `src/Optimizer.cc`, bundle adjustment (Local Bundle Adjustment and Pose Optimization) constructs g2o edges (`EdgeSE3ProjectXYZ` and `EdgeSE3ProjectXYZOnlyPose`):
```cpp
// Default measurement information matrix in ORB-SLAM3:
Eigen::Matrix2d info = Eigen::Matrix2d::Identity() * invSigma2;
edge->setInformation(info);
```
Where `invSigma2` is derived strictly from the pyramid scale factor (`mvInvLevelSigma2[kp.octave]`). While internal inertial marginalization edges contain code-level downweighting (e.g. line 2637: `vei[i]->setInformation(vei[i]->information()*1e-2)` to prevent boundary error accumulation), **there is no semantic observation weight factor** ($\omega_i \in [0, 1]$) exposed for visual landmarks.

### 2.3 Landmark Lifecycle in Tracking (`src/Tracking.cc` and `src/MapPoint.cc`)
In `src/Tracking.cc`, MapPoints are classified into:
- Valid / Inlier
- Outlier (flagged by chi-square test `chi2 > 5.991`)
- Replaced / Bad (`mbBad = true`)

A temporary exclusion mechanism with seamless reactivation (Arm C6) does not exist in standard ORB-SLAM3: setting `pMP->SetBadFlag()` permanently removes the MapPoint and notifies all observing KeyFrames to erase their associations.

---

## 3. Technical Requirements for True Native Integration

To achieve verifiable, publication-grade native integration without fragile workarounds, the following engineering modifications would be required:

1. **Custom g2o Edge Definition**:
   Implement `EdgeSE3ProjectXYZSemantic` supporting a dynamic scalar weight multiplier $\omega_i(t)$:
   $$\mathbf{\Omega}_{ij}(t) = \omega_i(t) \cdot \frac{1}{\sigma_{\text{scale}}^2} \mathbf{I}_{2 \times 2}$$
   Where $\omega_i(t) \to 0$ for high-uncertainty or moving objects.
2. **Synchronized Semantic Ingestion Channel**:
   Add a timestamped protobuf/flatbuffers binary stream or shared-memory ring buffer to ingest VLM query outcomes $\pi(o_i)$ alongside RGB-D frame pairs.
3. **Reactivatable MapPoint State**:
   Extend `ORB_SLAM3::MapPoint` with a `mbTemporarilyExcluded` flag that skips inclusion in `LocalBundleAdjustment` while preserving feature descriptors and visibility graphs for subsequent loop closing.

Because these kernel modifications require modifying the core upstream C++ library and recompiling all g2o optimizer dependencies, arms **C5 (`LANDMARK_DOWNWEIGHT`)**, **C6 (`LANDMARK_TEMPORARY_EXCLUDE`)**, and **C7 (`DUAL_INTEGRATION`)** are recorded as `NOT_IMPLEMENTED`.

---

## 4. Empirical Benchmark Status Summary

| Intervention Arm | Layer | Implementation Status | Empirical Evidence |
| :--- | :--- | :--- | :--- |
| **C0: KEEP** | Baseline | **VERIFIED & OPERATIONAL** | Evaluated on TUM (826 poses), OpenLORIS (1499 poses), D455 (52 instances). |
| **C1: IMAGE_POLYGON_MASK** | Image Space | **VERIFIED & OPERATIONAL** | Evaluated on TUM ($\Delta\text{ATE}=+0.16\%$, $p=0.44$) and OpenLORIS ($\Delta\text{ATE}=-5.20\%$, $p=0.0$). |
| **C2: IMAGE_BBOX_MASK** | Image Space | **VERIFIED & OPERATIONAL** | Evaluated on TUM ($\Delta\text{ATE}=+52.73\%$, $p=10^{-115}$) and OpenLORIS ($\Delta\text{ATE}=-5.11\%$, $p=10^{-209}$). |
| **C3: GRAPH_MARK_SUSPECT** | Scene Graph | **VERIFIED & OPERATIONAL** | Evaluated on D455 (protects inlier anchors from $4.28\times$ residual corruption). |
| **C4: GRAPH_TEMPORARY_EXCLUDE** | Scene Graph | **VERIFIED & OPERATIONAL** | Evaluated on D455 (preserves authentic $11.98\text{ cm}$ noise floor). |
| **C5: LANDMARK_DOWNWEIGHT** | SLAM Optimizer | **NOT_IMPLEMENTED** | Documented above; unverified kernel integration is rejected. |
| **C6: LANDMARK_TEMPORARY_EXCLUDE** | SLAM Optimizer | **NOT_IMPLEMENTED** | Documented above; requires C++ MapPoint lifecycle overhaul. |
| **C7: DUAL_INTEGRATION** | Joint SLAM | **NOT_IMPLEMENTED** | Documented above. |
| **RELOCATE_SPLIT** | Scene Graph | **VERIFIED & OPERATIONAL** | Evaluated on D455 ($1.97\text{ cm}$ displacement RMSE, $F_1 = 1.00$). |
| **REACTIVATE** | Scene Graph | **VERIFIED & OPERATIONAL** | Evaluated on D455 (refreshes confirmed landmark positions). |

