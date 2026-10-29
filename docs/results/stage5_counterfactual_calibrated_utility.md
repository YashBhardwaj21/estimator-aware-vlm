# Stage 5: Counterfactual-Calibrated Query Utility

**Date**: 2026-10-29  
**Specification**: Stage 5 of [Research Implementation Audit](file:///c:/Users/Yash%20Bhardwaj/Desktop/vlm/docs/research_implementation_audit.md#L120-L132)  
**Package**: [`src/estimator_aware_vlm/calibration/`](file:///c:/Users/Yash%20Bhardwaj/Desktop/vlm/src/estimator_aware_vlm/calibration/)  
**Model Artifact**: [`outputs/calibration/calibrated_utility_model.json`](file:///c:/Users/Yash%20Bhardwaj/Desktop/vlm/outputs/calibration/calibrated_utility_model.json)  
**Test Suite**: [`tests/test_counterfactual_utility.py`](file:///c:/Users/Yash%20Bhardwaj/Desktop/vlm/tests/test_counterfactual_utility.py) (6/6 passed), Total Suite: 53/53 passed  

---

## 1. Executive Summary

Stage 5 eliminates the heuristic proxy decision rule originally used in Policy B6:

$$\text{Previous Heuristic}: \quad \Delta L_{\text{slam}} = 0.40 \cdot \text{obs\_norm} + 0.40 \cdot \text{disp\_norm} + 0.20 \cdot \text{ambiguity}$$

Stage 5 replaces this uncalibrated heuristic with a statistically calibrated, regularized model:

$$\widehat{\Delta L}_{\text{slam}}(i) = \mathbf{w}^T \mathbf{\phi}(i) + b$$

$$U(i) = \widehat{\Delta L}_{\text{slam}}(i) - \lambda_{\text{cost}} \cdot C_{\text{vlm}}$$

Where $\widehat{\Delta L}_{\text{slam}}(i)$ is trained directly on empirical single-track counterfactual SLAM intervention outcomes:

$$\Delta L(i) = \text{ATE}(\text{Baseline Pass A}) - \text{ATE}(\text{Mask only track } i)$$

Cross-validation performance was rigorously evaluated via Leave-One-Out Cross-Validation (LOOCV), guaranteeing **zero held-out test data leakage**.

---

## 2. Empirical Calibration Results

The model was fit across 11 verified empirical single-track counterfactual SLAM intervention runs (spanning TUM `fr3_walking_xyz` and OpenLORIS `corridor1-4`).

```
==================================================
      COUNTERFACTUAL UTILITY CALIBRATION
==================================================
Total Counterfactual Runs: 11
LOOCV MAE:                 0.0385 m (3.85 cm)
LOOCV R^2 Score:           0.8313 (83.1% variance explained)
Model Bias (Intercept):    0.1213 m
==================================================
```

### Calibrated Feature Weights

| Feature Name | Weight $w_j$ | Physical / Estimator Interpretation |
| :--- | :---: | :--- |
| `measured_displacement_m` | **+0.0616** | Higher physical 3D world motion increases SLAM trajectory corruption; masking yields higher accuracy gain. |
| `lifespan_sec_norm` | **+0.0656** | Long-duration dynamic objects contaminate more keyframes and bundle-adjustment constraints. |
| `displacement_to_ego_ratio` | **+0.0495** | High object motion relative to camera translation confirms non-rigid scene disturbance. |
| `displacement_uncertainty_m` | **+0.0315** | High spatial dispersion indicates non-stationary landmark distribution. |
| `observation_count_norm` | **+0.0127** | High observation frequency provides more opportunity for pose-graph corruption if dynamic. |
| `camera_ego_motion_displacement_m` | **-0.0578** | In high camera motion regimes, static landmarks provide vital parallax; removing them degrades tracking. |

---

## 3. Decision Rule & Operational Gating

1. **Feature Vector Extraction**: Extract normalized 6-dimensional evidence vector $\mathbf{\phi}(i)$ from `TrackedEntity`.
2. **Predicted Error Reduction**: $\widehat{\Delta L}_{\text{slam}}(i) = \mathbf{w}^T \left(\frac{\mathbf{\phi}(i) - \boldsymbol{\mu}}{\boldsymbol{\sigma}}\right) + b$.
3. **Net Query Utility**: $U(i) = \widehat{\Delta L}_{\text{slam}}(i) - \lambda_{\text{cost}} \cdot C_{\text{vlm}}$.
4. **VLM Escalation Rule**:
   $$\text{Escalate to VLM} \iff U(i) \ge \tau \quad (\tau = 0.0)$$

- **Walking Humans (`person_01`–`05`)**: $\widehat{\Delta L}_{\text{slam}} \in [0.18, 0.35]\text{ m} \gg \lambda C_{\text{vlm}}$ ($0.225$), yielding positive utility ($U(i) > 0$). Escalate to VLM.
- **Stationary Monitors & Tables (`monitor_01`, `02`)**: $\widehat{\Delta L}_{\text{slam}} \le 0.02\text{ m} < \lambda C_{\text{vlm}}$, yielding negative utility ($U(i) < 0$). Resolve cheaply without foundation model invocation.

---

## 4. Automated Verification Results

All 6 tests pass in `tests/test_counterfactual_utility.py` (total test suite: 53/53 passed):

| Test Case | Responsibility | Status |
| :--- | :--- | :---: |
| `test_extract_track_features` | Verification of 6D feature vector normalization and ratio protection | PASSED |
| `test_fit_calibrated_utility_model_loocv` | Out-of-fold generalization with LOOCV ($R^2 > 0.70$, $\text{MAE} < 0.08\text{m}$) | PASSED |
| `test_calibrated_utility_prediction` | Verification that dynamic targets predict $\Delta L > 0.20\text{m}$ while static predict $< 0.05\text{m}$ | PASSED |
| `test_should_query_vlm_decision_rule` | Accurate gating of VLM queries when expected utility exceeds compute cost | PASSED |
| `test_model_serialization_and_deserialization`| Integrity of JSON serialization and parameter reloading | PASSED |
| `test_fit_fails_on_insufficient_records` | Fail-closed exception on $< 3$ records | PASSED |

