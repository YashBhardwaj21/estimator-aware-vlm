# Query-Selection Policy Benchmark Report

Generated from 885 causal observation feature vectors across 177 tracks.

> **Methodological Note on Random Routing Baseline (P2)**: The intervals reported for P2 are empirical **randomization intervals across 1,000 random permutation draws** of routing decisions over dataset observations. They quantify the variance of stochastic query selection, not an environment-generalization confidence interval.

## Evaluation Split: `full`

- Total Observations: **885**
- Total Tracks: **177**
- Truly Moving Tracks (Target 1 Reference): **102**
- Self-Mobile Tracks (Target 2 Reference): **46**

### Policy Comparison: Moving-Object Interception (Target 1: Actual Motion)

| Policy | Metric | 5% Budget | 10% Budget | 20% Budget | 50% Budget | 100% Budget |
| --- | --- | :---: | :---: | :---: | :---: | :---: |
| P0 (Zero VLM / Baseline) | **Recall** | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% |
| | Precision | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% |
| P1 (Always Query / Naive) | **Recall** | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% |
| | Precision | 57.6% | 57.6% | 57.6% | 57.6% | 57.6% |
| P2 (Random Subsampling - 1,000 draws) | **Recall** | 22.4% [16.7-28.4] | 40.6% [33.3-48.0] | 67.1% [59.8-74.5] | 96.9% [93.1-100.0] | 100.0% [100.0-100.0] |
| | Precision | 57.3% [43.9-70.0] | 57.5% [48.6-66.2] | 57.5% [52.5-62.5] | 57.6% [56.2-59.1] | 57.6% [57.6-57.6] |
| P3 (Detector Uncertainty) | **Recall** | 15.7% | 35.3% | 52.9% | 76.5% | 100.0% |
| | Precision | 40.0% | 56.2% | 54.5% | 54.5% | 57.6% |
| P4 (Visual Quality / Sharpness) | **Recall** | 17.6% | 29.4% | 47.1% | 85.3% | 100.0% |
| | Precision | 90.0% | 90.9% | 88.9% | 70.7% | 57.6% |
| P5 (Naive Apparent Motion) | **Recall** | 17.6% | 32.4% | 51.0% | 91.2% | 100.0% |
| | Precision | 81.8% | 76.7% | 64.2% | 63.3% | 57.6% |
| P6 (Estimator Uncertainty) | **Recall** | 15.7% | 29.4% | 50.0% | 92.2% | 100.0% |
| | Precision | 100.0% | 93.8% | 83.6% | 69.6% | 57.6% |
| **Estimator-Context Query-Priority Heuristic (P7)** | **Recall** | 17.6% | 31.4% | 55.9% | 92.2% | 100.0% |
| | Precision | 100.0% | 100.0% | 98.3% | 83.2% | 57.6% |
| P7 (Ablation: No Temporal Decay) | **Recall** | 17.6% | 31.4% | 57.8% | 95.1% | 100.0% |
| | Precision | 100.0% | 100.0% | 100.0% | 82.9% | 57.6% |

### Policy Comparison: Self-Mobile Affordance Interception (Target 2: Mobile Affordance)

| Policy | Metric | 5% Budget | 10% Budget | 20% Budget | 50% Budget | 100% Budget |
| --- | --- | :---: | :---: | :---: | :---: | :---: |
| P0 (Zero VLM / Baseline) | **Recall** | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% |
| | Precision | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% |
| P1 (Always Query / Naive) | **Recall** | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% |
| | Precision | 26.0% | 26.0% | 26.0% | 26.0% | 26.0% |
| P2 (Random Subsampling - 1,000 draws) | **Recall** | 22.5% | 40.6% | 67.1% | 96.9% | 100.0% |
| | Precision | 25.9% | 25.9% | 25.9% | 26.0% | 26.0% |
| P3 (Detector Uncertainty) | **Recall** | 13.0% | 23.9% | 41.3% | 65.2% | 100.0% |
| | Precision | 15.0% | 17.2% | 19.2% | 21.0% | 26.0% |
| P4 (Visual Quality / Sharpness) | **Recall** | 10.9% | 15.2% | 34.8% | 76.1% | 100.0% |
| | Precision | 25.0% | 21.2% | 29.6% | 28.5% | 26.0% |
| P5 (Naive Apparent Motion) | **Recall** | 28.3% | 45.6% | 69.6% | 89.1% | 100.0% |
| | Precision | 59.1% | 48.8% | 39.5% | 27.9% | 26.0% |
| P6 (Estimator Uncertainty) | **Recall** | 32.6% | 52.2% | 67.4% | 89.1% | 100.0% |
| | Precision | 93.8% | 75.0% | 50.8% | 30.4% | 26.0% |
| **Estimator-Context Query-Priority Heuristic (P7)** | **Recall** | 32.6% | 54.4% | 78.3% | 97.8% | 100.0% |
| | Precision | 83.3% | 78.1% | 62.1% | 39.8% | 26.0% |
| P7 (Ablation: No Temporal Decay) | **Recall** | 37.0% | 56.5% | 76.1% | 93.5% | 100.0% |
| | Precision | 94.4% | 81.2% | 59.3% | 36.8% | 26.0% |

---

## Evaluation Split: `development`

- Total Observations: **175**
- Total Tracks: **35**
- Truly Moving Tracks (Target 1 Reference): **25**
- Self-Mobile Tracks (Target 2 Reference): **17**

### Policy Comparison: Moving-Object Interception (Target 1: Actual Motion)

| Policy | Metric | 5% Budget | 10% Budget | 20% Budget | 50% Budget | 100% Budget |
| --- | --- | :---: | :---: | :---: | :---: | :---: |
| P0 (Zero VLM / Baseline) | **Recall** | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% |
| | Precision | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% |
| P1 (Always Query / Naive) | **Recall** | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% |
| | Precision | 71.4% | 71.4% | 71.4% | 71.4% | 71.4% |
| P2 (Random Subsampling - 1,000 draws) | **Recall** | 23.6% [12.0-32.0] | 42.2% [28.0-56.0] | 67.6% [52.0-80.0] | 97.1% [88.0-100.0] | 100.0% [100.0-100.0] |
| | Precision | 71.7% [42.9-100.0] | 71.3% [53.3-87.5] | 71.3% [60.9-81.8] | 71.4% [68.8-73.5] | 71.4% [71.4-71.4] |
| P3 (Detector Uncertainty) | **Recall** | 24.0% | 28.0% | 44.0% | 80.0% | 100.0% |
| | Precision | 75.0% | 58.3% | 61.1% | 66.7% | 71.4% |
| P4 (Visual Quality / Sharpness) | **Recall** | 16.0% | 28.0% | 44.0% | 76.0% | 100.0% |
| | Precision | 100.0% | 100.0% | 100.0% | 86.4% | 71.4% |
| P5 (Naive Apparent Motion) | **Recall** | 20.0% | 32.0% | 52.0% | 96.0% | 100.0% |
| | Precision | 100.0% | 100.0% | 86.7% | 80.0% | 71.4% |
| P6 (Estimator Uncertainty) | **Recall** | 16.0% | 32.0% | 48.0% | 88.0% | 100.0% |
| | Precision | 100.0% | 100.0% | 100.0% | 78.6% | 71.4% |
| **Estimator-Context Query-Priority Heuristic (P7)** | **Recall** | 16.0% | 32.0% | 52.0% | 88.0% | 100.0% |
| | Precision | 100.0% | 100.0% | 100.0% | 100.0% | 71.4% |
| P7 (Ablation: No Temporal Decay) | **Recall** | 16.0% | 40.0% | 52.0% | 92.0% | 100.0% |
| | Precision | 100.0% | 100.0% | 100.0% | 100.0% | 71.4% |

### Policy Comparison: Self-Mobile Affordance Interception (Target 2: Mobile Affordance)

| Policy | Metric | 5% Budget | 10% Budget | 20% Budget | 50% Budget | 100% Budget |
| --- | --- | :---: | :---: | :---: | :---: | :---: |
| P0 (Zero VLM / Baseline) | **Recall** | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% |
| | Precision | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% |
| P1 (Always Query / Naive) | **Recall** | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% |
| | Precision | 48.6% | 48.6% | 48.6% | 48.6% | 48.6% |
| P2 (Random Subsampling - 1,000 draws) | **Recall** | 23.4% | 42.2% | 67.7% | 97.0% | 100.0% |
| | Precision | 48.4% | 48.5% | 48.5% | 48.5% | 48.6% |
| P3 (Detector Uncertainty) | **Recall** | 5.9% | 11.8% | 35.3% | 82.3% | 100.0% |
| | Precision | 12.5% | 16.7% | 33.3% | 46.7% | 48.6% |
| P4 (Visual Quality / Sharpness) | **Recall** | 23.5% | 41.2% | 52.9% | 76.5% | 100.0% |
| | Precision | 100.0% | 100.0% | 81.8% | 59.1% | 48.6% |
| P5 (Naive Apparent Motion) | **Recall** | 29.4% | 47.1% | 70.6% | 94.1% | 100.0% |
| | Precision | 100.0% | 100.0% | 80.0% | 53.3% | 48.6% |
| P6 (Estimator Uncertainty) | **Recall** | 23.5% | 47.1% | 70.6% | 94.1% | 100.0% |
| | Precision | 100.0% | 100.0% | 100.0% | 57.1% | 48.6% |
| **Estimator-Context Query-Priority Heuristic (P7)** | **Recall** | 23.5% | 47.1% | 76.5% | 100.0% | 100.0% |
| | Precision | 100.0% | 100.0% | 100.0% | 77.3% | 48.6% |
| P7 (Ablation: No Temporal Decay) | **Recall** | 23.5% | 58.8% | 76.5% | 100.0% | 100.0% |
| | Precision | 100.0% | 100.0% | 100.0% | 73.9% | 48.6% |

---

## Evaluation Split: `validation`

- Total Observations: **65**
- Total Tracks: **13**
- Truly Moving Tracks (Target 1 Reference): **9**
- Self-Mobile Tracks (Target 2 Reference): **5**

### Policy Comparison: Moving-Object Interception (Target 1: Actual Motion)

| Policy | Metric | 5% Budget | 10% Budget | 20% Budget | 50% Budget | 100% Budget |
| --- | --- | :---: | :---: | :---: | :---: | :---: |
| P0 (Zero VLM / Baseline) | **Recall** | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% |
| | Precision | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% |
| P1 (Always Query / Naive) | **Recall** | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% |
| | Precision | 69.2% | 69.2% | 69.2% | 69.2% | 69.2% |
| P2 (Random Subsampling - 1,000 draws) | **Recall** | 21.5% [0.0-33.3] | 39.7% [22.2-55.6] | 69.3% [44.4-88.9] | 97.4% [88.9-100.0] | 100.0% [100.0-100.0] |
| | Precision | 69.0% [0.0-100.0] | 69.8% [40.0-100.0] | 70.0% [55.6-88.9] | 69.2% [66.7-75.0] | 69.2% [69.2-69.2] |
| P3 (Detector Uncertainty) | **Recall** | 22.2% | 33.3% | 77.8% | 88.9% | 100.0% |
| | Precision | 66.7% | 75.0% | 87.5% | 72.7% | 69.2% |
| P4 (Visual Quality / Sharpness) | **Recall** | 22.2% | 22.2% | 44.4% | 66.7% | 100.0% |
| | Precision | 100.0% | 100.0% | 100.0% | 85.7% | 69.2% |
| P5 (Naive Apparent Motion) | **Recall** | 33.3% | 33.3% | 44.4% | 100.0% | 100.0% |
| | Precision | 100.0% | 100.0% | 66.7% | 75.0% | 69.2% |
| P6 (Estimator Uncertainty) | **Recall** | 11.1% | 22.2% | 33.3% | 100.0% | 100.0% |
| | Precision | 100.0% | 100.0% | 75.0% | 81.8% | 69.2% |
| **Estimator-Context Query-Priority Heuristic (P7)** | **Recall** | 11.1% | 22.2% | 66.7% | 88.9% | 100.0% |
| | Precision | 100.0% | 100.0% | 100.0% | 80.0% | 69.2% |
| P7 (Ablation: No Temporal Decay) | **Recall** | 22.2% | 22.2% | 55.6% | 100.0% | 100.0% |
| | Precision | 100.0% | 100.0% | 100.0% | 81.8% | 69.2% |

### Policy Comparison: Self-Mobile Affordance Interception (Target 2: Mobile Affordance)

| Policy | Metric | 5% Budget | 10% Budget | 20% Budget | 50% Budget | 100% Budget |
| --- | --- | :---: | :---: | :---: | :---: | :---: |
| P0 (Zero VLM / Baseline) | **Recall** | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% |
| | Precision | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% |
| P1 (Always Query / Naive) | **Recall** | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% |
| | Precision | 38.5% | 38.5% | 38.5% | 38.5% | 38.5% |
| P2 (Random Subsampling - 1,000 draws) | **Recall** | 21.4% | 39.4% | 68.6% | 97.6% | 100.0% |
| | Precision | 38.2% | 38.6% | 38.4% | 38.5% | 38.5% |
| P3 (Detector Uncertainty) | **Recall** | 20.0% | 40.0% | 60.0% | 80.0% | 100.0% |
| | Precision | 33.3% | 50.0% | 37.5% | 36.4% | 38.5% |
| P4 (Visual Quality / Sharpness) | **Recall** | 40.0% | 40.0% | 80.0% | 80.0% | 100.0% |
| | Precision | 100.0% | 100.0% | 100.0% | 57.1% | 38.5% |
| P5 (Naive Apparent Motion) | **Recall** | 60.0% | 60.0% | 60.0% | 100.0% | 100.0% |
| | Precision | 100.0% | 100.0% | 50.0% | 41.7% | 38.5% |
| P6 (Estimator Uncertainty) | **Recall** | 20.0% | 40.0% | 40.0% | 100.0% | 100.0% |
| | Precision | 100.0% | 100.0% | 50.0% | 45.5% | 38.5% |
| **Estimator-Context Query-Priority Heuristic (P7)** | **Recall** | 20.0% | 40.0% | 100.0% | 100.0% | 100.0% |
| | Precision | 100.0% | 100.0% | 83.3% | 50.0% | 38.5% |
| P7 (Ablation: No Temporal Decay) | **Recall** | 40.0% | 40.0% | 60.0% | 100.0% | 100.0% |
| | Precision | 100.0% | 100.0% | 60.0% | 45.5% | 38.5% |

---

## Evaluation Split: `held_out`

- Total Observations: **125**
- Total Tracks: **25**
- Truly Moving Tracks (Target 1 Reference): **12**
- Self-Mobile Tracks (Target 2 Reference): **7**

### Policy Comparison: Moving-Object Interception (Target 1: Actual Motion)

| Policy | Metric | 5% Budget | 10% Budget | 20% Budget | 50% Budget | 100% Budget |
| --- | --- | :---: | :---: | :---: | :---: | :---: |
| P0 (Zero VLM / Baseline) | **Recall** | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% |
| | Precision | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% |
| P1 (Always Query / Naive) | **Recall** | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% |
| | Precision | 48.0% | 48.0% | 48.0% | 48.0% | 48.0% |
| P2 (Random Subsampling - 1,000 draws) | **Recall** | 21.8% [8.3-41.7] | 40.2% [16.7-58.3] | 68.3% [41.7-91.7] | 97.1% [83.3-100.0] | 100.0% [100.0-100.0] |
| | Precision | 47.3% [16.7-83.3] | 48.0% [25.0-72.7] | 48.2% [33.3-62.5] | 48.0% [43.5-52.2] | 48.0% [48.0-48.0] |
| P3 (Detector Uncertainty) | **Recall** | 0.0% | 16.7% | 50.0% | 66.7% | 100.0% |
| | Precision | 0.0% | 20.0% | 42.9% | 42.1% | 48.0% |
| P4 (Visual Quality / Sharpness) | **Recall** | 33.3% | 58.3% | 75.0% | 91.7% | 100.0% |
| | Precision | 100.0% | 87.5% | 75.0% | 68.8% | 48.0% |
| P5 (Naive Apparent Motion) | **Recall** | 16.7% | 41.7% | 66.7% | 91.7% | 100.0% |
| | Precision | 100.0% | 100.0% | 66.7% | 52.4% | 48.0% |
| P6 (Estimator Uncertainty) | **Recall** | 25.0% | 41.7% | 50.0% | 100.0% | 100.0% |
| | Precision | 100.0% | 100.0% | 75.0% | 66.7% | 48.0% |
| **Estimator-Context Query-Priority Heuristic (P7)** | **Recall** | 16.7% | 41.7% | 58.3% | 91.7% | 100.0% |
| | Precision | 100.0% | 100.0% | 100.0% | 73.3% | 48.0% |
| P7 (Ablation: No Temporal Decay) | **Recall** | 25.0% | 41.7% | 58.3% | 100.0% | 100.0% |
| | Precision | 100.0% | 100.0% | 100.0% | 75.0% | 48.0% |

### Policy Comparison: Self-Mobile Affordance Interception (Target 2: Mobile Affordance)

| Policy | Metric | 5% Budget | 10% Budget | 20% Budget | 50% Budget | 100% Budget |
| --- | --- | :---: | :---: | :---: | :---: | :---: |
| P0 (Zero VLM / Baseline) | **Recall** | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% |
| | Precision | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% |
| P1 (Always Query / Naive) | **Recall** | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% |
| | Precision | 28.0% | 28.0% | 28.0% | 28.0% | 28.0% |
| P2 (Random Subsampling - 1,000 draws) | **Recall** | 22.1% | 40.3% | 68.5% | 97.0% | 100.0% |
| | Precision | 27.8% | 28.0% | 28.2% | 28.0% | 28.0% |
| P3 (Detector Uncertainty) | **Recall** | 0.0% | 14.3% | 57.1% | 71.4% | 100.0% |
| | Precision | 0.0% | 10.0% | 28.6% | 26.3% | 28.0% |
| P4 (Visual Quality / Sharpness) | **Recall** | 57.1% | 85.7% | 100.0% | 100.0% | 100.0% |
| | Precision | 100.0% | 75.0% | 58.3% | 43.8% | 28.0% |
| P5 (Naive Apparent Motion) | **Recall** | 28.6% | 57.1% | 85.7% | 100.0% | 100.0% |
| | Precision | 100.0% | 80.0% | 50.0% | 33.3% | 28.0% |
| P6 (Estimator Uncertainty) | **Recall** | 42.9% | 71.4% | 85.7% | 100.0% | 100.0% |
| | Precision | 100.0% | 100.0% | 75.0% | 38.9% | 28.0% |
| **Estimator-Context Query-Priority Heuristic (P7)** | **Recall** | 28.6% | 71.4% | 85.7% | 100.0% | 100.0% |
| | Precision | 100.0% | 100.0% | 85.7% | 46.7% | 28.0% |
| P7 (Ablation: No Temporal Decay) | **Recall** | 42.9% | 71.4% | 100.0% | 100.0% | 100.0% |
| | Precision | 100.0% | 100.0% | 100.0% | 43.8% | 28.0% |

---

