# VOI Model Card & Training Sufficiency Declaration

**Model Name:** `R7_ESTIMATOR_AWARE_PRIORITY_PROXY`  
**Status:** `EXPLORATORY_PROXY_EVALUATED` (Calibrated Learned Model `NOT_EVALUABLE_AS_CALIBRATED_VOI`)  

## 1. Sufficiency Audit Results

- **Independent Development Sequences:** 3 (TUM `fr3_walking_xyz`, OpenLORIS `corridor1-4`, D455 `my_desk`) vs. Minimum 5 Required.
- **Paired Intervention Outcomes:** 14 vs. Minimum 30 Required.
- **Outcome:** Fails empirical sufficiency gate for parametric regression.

## 2. Model Policy Formulation

Rather than fitting an over-parameterized model to 14 data points or using synthetic targets, R7 employs a transparent, non-parametric priority score:

$$\text{Priority}(i) = \left( 0.05 \cdot N_{\text{landmarks}} + 0.5 \cdot N_{\text{inliers}} \right) \cdot (\sigma_{\text{displacement}} + 0.1) \cdot \mathbb{I}(\text{motion})$$

This formula captures the fundamental estimator-aware hypothesis: observations with high tracking weight and high geometric motion ambiguity provide the greatest potential benefit for VLM clarification.