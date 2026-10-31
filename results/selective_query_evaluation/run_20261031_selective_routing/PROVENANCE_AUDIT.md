# PROVENANCE AUDIT: Experiment 8 Selective VLM Routing

**Experiment ID:** `run_20261031_selective_routing`  
**Execution Date:** 2026-10-31 16:19:32 UTC  
**Repository:** `https://github.com/YashBhardwaj21/estimator-aware-vlm`  
**Git Commit:** `aba2b3774626a4e7dc568b94b9b17c375a202524` (Branch: `main`, Clean: `False`)  

---

## 1. Research Integrity & Non-Fabrication Guarantee

This experiment strictly obeys the core empirical constraints:
1. **Zero Ground-Truth Leakage:** The pre-query feature table (`candidate_evidence_prequery.jsonl`) contains solely estimator telemetry and observed visual/geometric signals available *at the moment of observation*. No ground truth labels, future trajectory errors, or counterfactual labels are visible to the policy routers.
2. **Real VLM Inference:** All 52 VLM predictions are authentic outputs from `Qwen/Qwen2.5-VL-3B-Instruct` run with greedy decoding (`temperature=0.0`, `max_new_tokens=512`). Zero mock or synthetic predictions were injected.
3. **RealSense D455 Camera Ground Truth is Strictly Null:** As documented, the RealSense D455 handheld camera recordings lack motion capture ground truth. Downstream evaluation is performed exclusively on physical tape-measured displacement (Delta_tape) and held-out rigid structural anchor residuals (12.42 cm).
4. **Landmark-Level Integration Scope:** Landmark-level weighting inside ORB-SLAM3 bundle adjustment (C5-C7) is formally designated as `NOT_IMPLEMENTED`. Interventions operate strictly via scene-graph lifecycle updates and bounding box dynamic masking.
5. **Sufficiency Gate Adherence:** With N_seq = 3 < 5 and N_paired = 14 < 30, statistical sufficiency criteria for a learned, calibrated VOI model were not met. Policy R7 is transparently designated and evaluated as `R7_ESTIMATOR_AWARE_PRIORITY_PROXY` without synthetic data inflation.

---

## 2. SLAM & VLM Infrastructure Provenance

- **ORB-SLAM3 WSL Executable:**
  - Path: `/home/saturn/ORB_SLAM3/Examples/RGB-D/rgbd_tum`
  - SHA-256: `cd9162f319423483a93e1c4e33ffc86ec9bb5bc3e931ea44816ebc5db08e999c`
  - Git Commit: `4452a3c4ab75b1cde34e5505a36ec3f9edcdc4c4`
  - Vocabulary: `/home/saturn/ORB_SLAM3/Vocabulary/ORBvoc.txt`
- **VLM Model Specifications:**
  - Base Model: `Qwen/Qwen2.5-VL-3B-Instruct`
  - Precision: PyTorch FP16 (`float16`)
  - Server Harness: FastAPI / vLLM local backend on `cuda:0`
  - Generation Parameters: Greedy (`temperature=0.0`, `seed=42`, `max_new_tokens=512`)
  - Structured Prompt SHA-256: `df7e04b4c0bfbb4246fe8e50bdfcae6c88f11caae2f6ee17804918e77c5950d2`

---

## 3. Pipeline Scripts Provenance

| Script Path | Lines | SHA-256 Hash |
| :--- | :--- | :--- |
| `scripts/build_experiment8_phase1_inventory.py` | 373 | `021574da47c63b22...` |
| `scripts/build_experiment8_phase2_prequery_evidence.py` | 294 | `10608759f0fdbda1...` |
| `scripts/build_experiment8_phase3_vlm_bank.py` | 343 | `a3a2278ad624ee5b...` |
| `scripts/build_experiment8_phase4_policies.py` | 497 | `f9eb7c991ddb9411...` |
| `scripts/build_experiment8_phase6_downstream_evaluation.py` | 484 | `cff086ca735e0ced...` |
| `scripts/plot_experiment8_publication_figures.py` | 295 | `42f9c4c089553b42...` |
| `scripts/audit_and_generate_experiment8_provenance.py` | 293 | `88dd042d41a69da9...` |

---

## 4. Key Output Artifacts & Cryptographic Checksums

| Artifact Path | Size (bytes) | SHA-256 Hash |
| :--- | :--- | :--- |
| `experiment_manifest.json` | 2,348 | `500864b8dfcda9cd...` |
| `dataset_inventory.json` | 1,009 | `564da9a6c005b9cd...` |
| `candidate_inventory.json` | 78,741 | `4bae981811d6e691...` |
| `candidate_inventory.csv` | 13,424 | `345deb143c882c84...` |
| `annotation_provenance_audit.json` | 1,919 | `80409de67dd7e8de...` |
| `data_availability_report.md` | 2,773 | `835028d04e6c1fdc...` |
| `candidate_evidence_prequery.jsonl` | 65,653 | `c0dd82b3aae4d7a8...` |
| `candidate_evidence_prequery.csv` | 15,528 | `dd1af1413db27139...` |
| `estimator_telemetry_summary.json` | 746 | `d1f297d411cb3ca7...` |
| `vlm/vlm_live_predictions.jsonl` | 88,633 | `3496fbe1a9f1f9db...` |
| `vlm/vlm_response_bank.json` | 101,968 | `1c011f554b768d69...` |
| `vlm/vlm_latency_per_candidate.csv` | 7,751 | `4fece48c075e05ba...` |
| `vlm/vlm_latency_summary.json` | 772 | `bb9783e2f63f52b9...` |
| `vlm/vlm_model_and_prompt_manifest.json` | 1,369 | `202a2b8725dd4f05...` |
| `vlm/vlm_failure_cases.json` | 581 | `91b854071c9c9de6...` |
| `policies/policy_definitions.json` | 1,405 | `cbe649d73bffb141...` |
| `policies/policy_decisions.jsonl` | 436,381 | `ee049aa9d8c2c17d...` |
| `policies/selected_candidates_by_policy_budget.json` | 134,368 | `43e2127bd4d26005...` |
| `policies/query_budget_manifest.json` | 260 | `d5b108e3f1f19cd7...` |
| `policies/policy_replay_summary.json` | 433 | `f898afdd0afc24ce...` |
| `policies/policy_action_audit.json` | 582 | `8fd8272563d82f31...` |
| `voi/voi_training_data_sufficiency.json` | 1,562 | `0498fed89209488b...` |
| `voi/counterfactual_query_benefit_labels.csv` | 1,450 | `2f21e117fd3e59e0...` |
| `voi/voi_model_card.md` | 1,100 | `19fa99374ef5cdb1...` |
| `tum/policy_slam_metrics.csv` | 4,202 | `afeda41ea5161e7a...` |
| `tum/policy_cost_metrics.csv` | 2,296 | `4fcc5447c71c062f...` |
| `openloris/policy_slam_metrics.csv` | 1,069 | `3b8919913219de77...` |
| `openloris/policy_cost_metrics.csv` | 426 | `156095f26590ec3e...` |
| `d455/policy_lifecycle_metrics.csv` | 5,366 | `ebd525a44ed97e96...` |
| `d455/relocation_predictions.json` | 18,198 | `eab39e71e2a9d5e1...` |
| `d455/anchor_integrity_metrics.json` | 216 | `f97980272f22d973...` |
| `aggregate/selection_quality_metrics.csv` | 1,066 | `fc07520742b2093d...` |
| `aggregate/downstream_policy_comparison.csv` | 2,001 | `71aaa5381a8241b6...` |
| `aggregate/pareto_frontiers.json` | 6,889 | `2bbea5181efc77f4...` |
| `aggregate/statistical_analysis.json` | 1,007 | `61ccd5caf6e1a37d...` |
| `aggregate/figures/figure1_candidate_populations.png` | 167,202 | `5d6373c2508d5c98...` |
| `aggregate/figures/figure2_vlm_inference_costs.png` | 202,392 | `3425a62c7d9007aa...` |
| `aggregate/figures/figure3_budget_vs_selection_quality.png` | 376,268 | `2d421562fdba8b88...` |
| `aggregate/figures/figure4_tum_localization_vs_budget.png` | 463,822 | `c12abcb4e8e534cf...` |
| `aggregate/figures/figure5_openloris_localization_vs_budget.png` | 193,473 | `ddc74a96708f350f...` |
| `aggregate/figures/figure6_d455_lifecycle_vs_budget.png` | 349,081 | `ad561222a5a8cbf2...` |
| `aggregate/figures/figure7_pareto_accuracy_vs_cost.png` | 338,089 | `0f8e35400d72f180...` |
| `aggregate/figures/figure8_qualitative_policy_decisions.png` | 356,938 | `7a2e5a23005af405...` |

---

## 5. Verification Command

To independently verify the bit-level integrity of all generated artifacts:
```bash
cd results/selective_query_evaluation/run_20261031_selective_routing/
sha256sum -c provenance/checksums.sha256
```
