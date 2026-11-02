# Decision & Evidence Fusion Report

Generated from 885 causal observations across 177 tracks.  
Inference budget evaluated: **10.0% (88 queries)**  
VLM configuration consumed: **V4**  
Fusion formulation: **Heuristic Log-Odds Fusion**

---

## 1. Track-Level SLAM Intervention Performance

| Policy | Budget | Queries | Dynamic Track Recall | Masking Precision | Track F1 | Static Retention Rate |
| --- | :---: | :---: | :---: | :---: | :---: | :---: |
| P0 (Baseline SLAM) | 0% | 0 | **0.0%** | N/A | **0.000** | 100.0% |
| Geometry + Prior Only | 0% | 0 | **70.6%** | 83.7% | **0.766** | 90.7% |
| P1 (Offline Track-Propagated) | 100% | 885 | **49.0%** | 84.8% | **0.621** | 94.1% |
| P2 (Random Routing - 10%) | 10% | 88 | **68.6%** | 84.3% | **0.757** | 91.5% |
| P6 (Estimator Uncertainty - 10%) | 10% | 88 | **70.6%** | 83.7% | **0.766** | 90.7% |
| **Estimator-Context Heuristic (P7 - 10%)** | 10% | 88 | **70.6%** | 83.7% | **0.766** | 90.7% |

---

## 2. Observation-Level Masking Performance

| Policy | Budget | Evaluated Obs | TP Masks | FP Masks | FN Masks | TN Retained | Mask Precision | Mask Recall | Mask F1 | Static Retention |
| --- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| P0 (Baseline SLAM) | 0% | 885 | 0 | 0 | 510 | 375 | N/A | 0.0% | 0.000 | 100.0% |
| P1 (Offline Track-Propagated) | 100% | 885 | 157 | 22 | 353 | 353 | 87.7% | 30.8% | 0.456 | 94.1% |
| P2 (Random Routing - 10%) | 10% | 885 | 226 | 32 | 284 | 343 | 87.6% | 44.3% | 0.589 | 91.5% |
| P6 (Estimator Uncertainty - 10%) | 10% | 885 | 230 | 35 | 280 | 340 | 86.8% | 45.1% | **0.594** | 90.7% |
| **Estimator-Context Heuristic (P7 - 10%)** | 10% | 885 | 224 | 35 | 286 | 340 | 86.5% | 43.9% | 0.583 | 90.7% |

---

## 3. Key Scientific Observations

1. **Track-Level Equivalence to Kinematics + Prior Baseline**: Both P6 and P7 achieve identical track-level metrics ($F_1 = 0.766$) to the kinematics-plus-prior-only baseline. Selective VLM routing does not provide an additional aggregate track-level boost over geometric displacement and semantic priors alone.
2. **Observation-Level Distinctions**: P6 and P7 exhibit distinct per-frame masking behavior (P6: 230 TP / 35 FP; P7: 224 TP / 35 FP), reflecting their differing ranking priorities at the observation boundary.
3. **P1 Provenance & Fallback Qualification**: P1 represents an offline evaluation utilizing 168 exact live records, 287 same-track fallback records, and 430 non-queried fallbacks. Its reduced recall (49.0%) results from propagating false-negative VLM classifications across tracks.
4. **Static Landmark Retention**: By decoupling physical affordance from observed motion, stationary objects are preserved as localization anchors (90.7% retention).
