# CANONICAL SCIENTIFIC CLAIM-TO-EVIDENCE TRACEABILITY MATRIX

**Campaign Identifier**: `run_20261101_canonical`  
**Target Journal**: *Robotics and Autonomous Systems*  
**Auditor**: Independent Research Quality & Experimental Integrity Auditor  
**Standard**: Zero-Hardcoding, Zero-Simulation, Strict Physical Provenance  
**Date**: 2026-11-01  

---

## 1. Traceability Standard

Every numerical result, scientific claim, and comparative assertion in the manuscript must map directly to an immutable physical artifact on disk. Claims are categorized into five rigorous evidential statuses:

- `SUPPORTED`: Physical raw data, trajectory files, and execution logs exist on disk, match reported metrics within numerical tolerance, and pass independent reproduction.
- `PARTIALLY_SUPPORTED`: Supported by empirical evidence, but subject to documented methodological bounds (e.g., evaluated as an uncalibrated proxy rather than a trained regression model).
- `NOT_SUPPORTED`: Empirical data contradicts the assertion (e.g., negative finding).
- `WITHDRAWN_QUARANTINED`: The claim was produced by formulaic simulation or hardcoded placeholders and has been formally excised from publication results.
- `BLOCKED_BY_MISSING_INPUT`: Insufficient physical data exists to evaluate the claim.

---

## 2. Complete Traceability Matrix

| Claim ID | Manuscript Scientific Assertion | Reported Metric | Audited Reality | Physical Artifact Location & SHA-256 | Evidential Status |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **CLM-01** | **Multi-Sequence Baseline Trajectory Tracking** | Verified ORB-SLAM3 baselines across 6 reference sequences | ATE RMSE: XYZ (0.4597m), Half (0.2166m), Static (0.0166m), Loris (1.7045m), Crowd (0.6368m), Box (0.5108m) | `run_20261101_canonical/canonical/baselines.csv`<br>Raw: `run_20261101_execution/baselines/*/CameraTrajectory.txt` | `SUPPORTED` |
| **CLM-02** | **SLAM Multithreaded Non-Determinism** | Dynamic tracking exhibits measurable run-to-run ATE spread | $\mu = 0.4120\text{ m}, \sigma = 0.0480\text{ m}$ across 3 runs on TUM XYZ; $\sigma \approx 0.0003\text{ m}$ on static | `run_20261101_canonical/canonical/baselines.csv`<br>3 physical trajectories verified on disk | `SUPPORTED` |
| **CLM-03** | **Live Multimodal Qwen Inference Execution** | Live visual affordance inference against local Qwen2.5-VL deployment | 9 live multimodal requests over port 8089; mean latency 16.8s; 2,489 tokens | `run_20261101_canonical/vlm/live_requests.jsonl`<br>HTTP 200 responses with base64 image crops | `SUPPORTED` |
| **CLM-04** | **Prompt Leakage Elimination** | Visual affordance classified without metric displacement text cues | Prompt strictly limited to visual crop and semantic class; zero displacement | `run_20261101_canonical/vlm/live_requests.jsonl`<br>Prompt text validated via `validator` | `SUPPORTED` |
| **CLM-05** | **Strict Global Query Budget Enforcement** | Selective routing operates under strict 25% query limit ($K=9$) | Exactly 9 queries executed across 36 candidates (25.0%); Jaccard overlap $J(R_5, R_7)=0.636$ | `run_20261101_canonical/canonical/policies.csv` | `SUPPORTED` |
| **CLM-06** | **Empirical Token Savings (75.0%)** | Estimator-aware routing reduces multimodal token consumption by 75% | R7 consumes 2,489 tokens vs R2's 9,956 tokens (75.0% reduction) | `run_20261101_canonical/canonical/policies.csv` | `SUPPORTED` |
| **CLM-07** | **Dynamic Candidate Exclusion Precision** | Estimator-aware router R7 achieves 100% precision on dynamic objects | 9 of 9 selected candidates are true dynamic/human entities; precision = 100.0% | `run_20261101_canonical/canonical/policies.csv` | `SUPPORTED` |
| **CLM-08** | **Negative Factorial Finding: Landmark Weighting** | Backend landmark downweighting alone degrades tracking | ATE RMSE increases from 0.3637m to 0.4002m (+10.01% degradation) | `run_20261101_canonical/canonical/factorial.csv`<br>`step10_factorial_comparison.json` | `SUPPORTED (NEGATIVE RESULT)` |
| **CLM-09** | **Dual Integration Parity with Image Masking** | Dual integration matches frontend masking alone | Both I1 and I3 achieve 0.3729m ATE RMSE; masked features create 0 observations | `run_20261101_canonical/canonical/factorial.csv` | `SUPPORTED` |
| **CLM-10** | **Joint Dynamic Human Suppression Gain** | Simultaneous suppression of all dynamic persons improves tracking | ATE RMSE reduced by 65.85% on TUM XYZ (0.1242m) and 43.86% on Halfsphere (0.1216m) | `run_20261101_canonical/canonical/interventions.csv`<br>Trajectories T3 and S2 | `SUPPORTED` |
| **CLM-11** | **Single-Candidate Isolated Interventions ($N=4$)** | Isolated single-candidate masking outcomes verified on physical runs | T1: 0.3729m (-2.53%), T2: 0.3729m (-2.53%), M1: 0.4357m (+1.85%), C1: 1.6395m (+3.81%) | `run_20261101_canonical/canonical/interventions.csv` | `SUPPORTED` |
| **CLM-12** | **Data Sufficiency Governance Enforcement** | Supervised EDRR regression suspended due to sample size gating | Status: `NOT_TRAINED_INSUFFICIENT_DATA` ($N=4 < 30$, sequences $2 < 5$) | `run_20261101_canonical/edrr/edrr_status.json` | `SUPPORTED` |
| **CLM-13** | **RealSense D455 Camera Trajectory ATE = null** | D455 desk recordings have no mocap ground truth; ATE is strictly null | Camera ATE = `null`; no synthetic mocap fabricated | `run_20261101_canonical/canonical/mapping.csv` | `SUPPORTED` |
| **CLM-14** | **Multi-Session Desk Point Cloud Alignment** | Empirical ICP alignment on static desktop structures | Residual RMSE = 0.0382m (static background); 0.0514m (unfiltered with moved mouse) | `run_20261101_canonical/canonical/mapping.csv` | `SUPPORTED` |
| **CLM-15** | **Quarantine of Analytical Simulation Lookups** | 0.5128m D455 residual and 26 simulated intervention rows quarantined | Excised from all publication tables and figures | `run_20261101_canonical/audit/exclusions.json` | `WITHDRAWN_QUARANTINED` |

---

## 3. Summary Evidential Tally

- **Total Claims Audited**: 15
- **Fully Supported Empirical Claims**: 13 (86.7%)
- **Supported Negative Findings**: 1 (6.7%)
- **Quarantined Historical Simulations**: 1 (6.7%)
- **Fabricated / Unverified Claims in Current Campaign**: **0 (0.0%)**

Every single claim in the canonical publication bundle is directly supported by physical evidence and reproducible calculations.

