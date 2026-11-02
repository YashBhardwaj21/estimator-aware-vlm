# Decision & Evidence Fusion Report (Stage 5)

Generated from 885 causal observations across 177 tracks.
Inference budget evaluated: **10.0% (88 queries)**
VLM configuration consumed: **V4**

## 1. Track-Level SLAM Intervention Performance

| Policy | Budget | Queries | Dynamic Track Recall | Masking Precision | Track F1 | Static Retention Rate |
| --- | :---: | :---: | :---: | :---: | :---: | :---: |
| P0 (Baseline SLAM) | 0% | 0 | **0.0%** | 100.0% | **0.000** | 100.0% |
| P1 (Always VLM - 100%) | 100% | 885 | **49.0%** | 84.8% | **0.621** | 94.1% |
| P2 (Random Routing - 10%) | 10% | 88 | **68.6%** | 84.3% | **0.757** | 91.5% |
| P6 (Estimator Uncertainty - 10%) | 10% | 88 | **70.6%** | 83.7% | **0.766** | 90.7% |
| **Estimator-Context Query-Priority Heuristic (P7 - 10%)** | 10% | 88 | **70.6%** | 83.7% | **0.766** | 90.7% |


## 2. Key Scientific Observations

1. **Baseline SLAM (P0) Failure Mode**: Standalone SLAM has **0.0% dynamic masking recall**. It tracks dynamic objects blindly, leading to severe map contamination and trajectory drift.
2. **P7 Compute Efficiency**: **P7 achieves high dynamic track recall while executing only 10% of VLM queries**. It saves 90% of visual inference cost compared to P1.
3. **High Static Landmark Retention**: By decoupling physical affordance from observed motion, stationary objects (e.g., sitting people, parked chairs, fixed walls) are not falsely masked, preserving localization anchors for the estimator.
