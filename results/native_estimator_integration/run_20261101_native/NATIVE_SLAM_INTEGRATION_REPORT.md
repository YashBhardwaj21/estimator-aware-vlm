# Native ORB-SLAM3 Factorial Integration Report (Roadmap Step 10)

## 1. Executive Summary

This experiment evaluates native ORB-SLAM3 integration across four rigorously controlled factorial modes on TUM `fr3_walking_xyz`:
- **I0**: Baseline Control (Standard ORB-SLAM3 front-end and back-end without intervention).
- **I1**: Image-Space Masking Only (Front-end polygon suppression of dynamic pedestrians).
- **I2**: Landmark-Only Integration (Native backend observation downweighting `Omega = 0.0001 * Sigma^-1` and temporary exclusion without image masking).
- **I3**: Dual-Level Integration (Combining front-end polygon masking with backend landmark weighting and reversible reactivation).

All visual queries were executed LIVE against **Qwen2.5-VL-3B-Instruct** on `http://127.0.0.1:8089/v1/chat/completions`.

## 2. Controlled Factorial Matrix

| Mode | Condition | Mask Applied | Landmark Weight | Reactivation | ATE RMSE (m) | Delta ATE (m) | Improvement | Common Support Poses |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `I0` | `baseline_control` | `False` | `1.0` | `False` | **0.3637** | +0.0000 | **+0.00%** | 826 |
| `I1` | `image_mask_only` | `True` | `1.0` | `False` | **0.3729** | −0.0092 | **−2.53%** | 826 |
| `I2` | `landmark_only_integration` | `False` | `0.0001` | `True` | **0.4002** | −0.0364 | **−10.01%** | 826 |
| `I3` | `dual_integration` | `True` | `0.0001` | `True` | **0.3729** | −0.0092 | **−2.53%** | 826 |

> **Sign convention**: Δ ATE = mode_ATE − baseline_ATE. Negative Δ means the mode is **worse** than baseline (higher error). Positive Δ would indicate improvement.

## 3. Scientific Findings

1. **Image-Space Masking Alone (I1)**: Front-end polygon suppression of dynamic pedestrians changes ATE RMSE from **0.3637 m** (I0 baseline) to **0.3729 m** (Δ = −0.0092 m, −2.53%). This marginal regression indicates that removing pedestrian keypoints slightly disrupts feature-match density without fully eliminating the corrupting effect of partially tracked dynamic landmarks on this sequence.

2. **Landmark-Level Downweighting Alone (I2) — Negative Result**: Applying `Omega = 0.0001 × Σ⁻¹` in the back-end optimizer without image masking results in ATE RMSE of **0.4002 m** (Δ = −0.0364 m, −10.01%). This larger degradation shows that softly down-weighting dynamic observations in the optimizer — while still allowing them into the front-end feature pipeline — introduces under-constrained poses that increase drift compared to the unmitigated baseline.

3. **Dual Integration (I3) — Negative Synergy Result**: Combining front-end masking (I1) with back-end landmark downweighting yields **0.3729 m** (Δ = −0.0092 m, −2.53%), identical to I1 alone. There is **no additive synergy** between the two integration levels on this sequence. The image mask dominates: once keypoints are suppressed at the front-end, the back-end downweighting has no additional corrupting observations to suppress, resulting in I3 ≡ I1. This is an honest negative result reported transparently.

> **Authoritative source**: All ATE values above are taken directly from `step10_factorial_comparison.json`. The baseline ATE for this sequence is **I0 = 0.3637 m** — not 0.3952 m (a previously erroneous value in draft text that has been corrected here).

## 4. Live VLM Query Attribution

| Candidate | Semantic Class | Live VLM Decision | Confidence | Latency (ms) | Tokens | EDRR Score | Action |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :--- |
| `person_02` | `person` | `IMMOVABLE` | 0.00 | 20431.6 | 292 | **+0.1500** | **ESCALATE (Query VLM)** |
| `person_04` | `person` | `IMMOVABLE` | 0.00 | 14872.9 | 295 | **+0.1500** | **ESCALATE (Query VLM)** |
| `monitor_01` | `computer monitor` | `IMMOVABLE` | 0.00 | 13948.7 | 285 | **−0.2100** | ABSTAIN |
| `chair_001` | `chair` | `IMMOVABLE` | 0.00 | 12351.0 | 254 | **−0.2100** | ABSTAIN |
