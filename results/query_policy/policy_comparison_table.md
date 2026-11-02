# Stage 5 — Query-Selection Policy Benchmark Report

Generated from 885 causal observation feature vectors.

## Evaluation Split: `full`

- Total Observations: **885**
- Total Tracks: **177**
- Total Dynamic / Mobile Tracks: **131**

### Policy Comparison across Budgets (Dynamic Track Recall & Query Precision)

| Policy | Metric | 5% Budget | 10% Budget | 20% Budget | 50% Budget | 100% Budget |
| --- | --- | :---: | :---: | :---: | :---: | :---: |
| P0 (Zero VLM / Baseline) | **Recall** | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% |
| | Precision | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% |
| P1 (Always Query / Naive) | **Recall** | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% |
| | Precision | 74.0% | 74.0% | 74.0% | 74.0% | 74.0% |
| P2 (Random Subsampling) | **Recall** | 19.9% | 42.0% | 64.1% | 97.7% | 100.0% |
| | Precision | 65.9% | 75.0% | 72.9% | 75.3% | 74.0% |
| P3 (Detector Uncertainty) | **Recall** | 22.1% | 37.4% | 54.2% | 76.3% | 100.0% |
| | Precision | 72.7% | 78.4% | 74.6% | 69.2% | 74.0% |
| P4 (Visual Quality / Sharpness) | **Recall** | 13.0% | 20.6% | 36.6% | 76.3% | 100.0% |
| | Precision | 72.7% | 78.4% | 83.6% | 82.8% | 74.0% |
| P5 (Naive Apparent Motion) | **Recall** | 6.1% | 12.2% | 25.9% | 57.2% | 100.0% |
| | Precision | 88.6% | 88.6% | 94.3% | 84.8% | 74.0% |
| P6 (Estimator Uncertainty) | **Recall** | 6.9% | 13.7% | 27.5% | 67.2% | 100.0% |
| | Precision | 100.0% | 100.0% | 100.0% | 98.9% | 74.0% |
| **P7 (Proposed Estimator VOI)** | **Recall** | 19.1% | 27.5% | 41.2% | 78.6% | 100.0% |
| | Precision | 100.0% | 100.0% | 100.0% | 94.6% | 74.0% |

---

## Evaluation Split: `development`

- Total Observations: **175**
- Total Tracks: **35**
- Total Dynamic / Mobile Tracks: **29**

### Policy Comparison across Budgets (Dynamic Track Recall & Query Precision)

| Policy | Metric | 5% Budget | 10% Budget | 20% Budget | 50% Budget | 100% Budget |
| --- | --- | :---: | :---: | :---: | :---: | :---: |
| P0 (Zero VLM / Baseline) | **Recall** | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% |
| | Precision | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% |
| P1 (Always Query / Naive) | **Recall** | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% |
| | Precision | 82.9% | 82.9% | 82.9% | 82.9% | 82.9% |
| P2 (Random Subsampling) | **Recall** | 17.2% | 37.9% | 69.0% | 96.5% | 100.0% |
| | Precision | 66.7% | 77.8% | 88.6% | 86.4% | 82.9% |
| P3 (Detector Uncertainty) | **Recall** | 20.7% | 31.0% | 44.8% | 82.8% | 100.0% |
| | Precision | 66.7% | 66.7% | 60.0% | 68.2% | 82.9% |
| P4 (Visual Quality / Sharpness) | **Recall** | 13.8% | 24.1% | 37.9% | 75.9% | 100.0% |
| | Precision | 100.0% | 100.0% | 100.0% | 100.0% | 82.9% |
| P5 (Naive Apparent Motion) | **Recall** | 6.9% | 13.8% | 24.1% | 62.1% | 100.0% |
| | Precision | 100.0% | 100.0% | 100.0% | 100.0% | 82.9% |
| P6 (Estimator Uncertainty) | **Recall** | 6.9% | 13.8% | 24.1% | 62.1% | 100.0% |
| | Precision | 100.0% | 100.0% | 100.0% | 100.0% | 82.9% |
| **P7 (Proposed Estimator VOI)** | **Recall** | 20.7% | 37.9% | 37.9% | 82.8% | 100.0% |
| | Precision | 100.0% | 100.0% | 100.0% | 98.9% | 82.9% |

---

## Evaluation Split: `validation`

- Total Observations: **65**
- Total Tracks: **13**
- Total Dynamic / Mobile Tracks: **12**

### Policy Comparison across Budgets (Dynamic Track Recall & Query Precision)

| Policy | Metric | 5% Budget | 10% Budget | 20% Budget | 50% Budget | 100% Budget |
| --- | --- | :---: | :---: | :---: | :---: | :---: |
| P0 (Zero VLM / Baseline) | **Recall** | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% |
| | Precision | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% |
| P1 (Always Query / Naive) | **Recall** | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% |
| | Precision | 92.3% | 92.3% | 92.3% | 92.3% | 92.3% |
| P2 (Random Subsampling) | **Recall** | 25.0% | 33.3% | 50.0% | 100.0% | 100.0% |
| | Precision | 100.0% | 83.3% | 92.3% | 87.5% | 92.3% |
| P3 (Detector Uncertainty) | **Recall** | 25.0% | 33.3% | 58.3% | 83.3% | 100.0% |
| | Precision | 100.0% | 100.0% | 92.3% | 87.5% | 92.3% |
| P4 (Visual Quality / Sharpness) | **Recall** | 16.7% | 16.7% | 33.3% | 50.0% | 100.0% |
| | Precision | 100.0% | 100.0% | 100.0% | 87.5% | 92.3% |
| P5 (Naive Apparent Motion) | **Recall** | 8.3% | 16.7% | 25.0% | 58.3% | 100.0% |
| | Precision | 100.0% | 100.0% | 100.0% | 100.0% | 92.3% |
| P6 (Estimator Uncertainty) | **Recall** | 8.3% | 16.7% | 25.0% | 58.3% | 100.0% |
| | Precision | 100.0% | 100.0% | 100.0% | 100.0% | 92.3% |
| **P7 (Proposed Estimator VOI)** | **Recall** | 16.7% | 25.0% | 33.3% | 75.0% | 100.0% |
| | Precision | 100.0% | 100.0% | 100.0% | 96.9% | 92.3% |

---

## Evaluation Split: `held_out`

- Total Observations: **125**
- Total Tracks: **25**
- Total Dynamic / Mobile Tracks: **18**

### Policy Comparison across Budgets (Dynamic Track Recall & Query Precision)

| Policy | Metric | 5% Budget | 10% Budget | 20% Budget | 50% Budget | 100% Budget |
| --- | --- | :---: | :---: | :---: | :---: | :---: |
| P0 (Zero VLM / Baseline) | **Recall** | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% |
| | Precision | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% |
| P1 (Always Query / Naive) | **Recall** | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% |
| | Precision | 72.0% | 72.0% | 72.0% | 72.0% | 72.0% |
| P2 (Random Subsampling) | **Recall** | 22.2% | 50.0% | 55.6% | 94.4% | 100.0% |
| | Precision | 83.3% | 83.3% | 64.0% | 67.7% | 72.0% |
| P3 (Detector Uncertainty) | **Recall** | 22.2% | 33.3% | 55.6% | 77.8% | 100.0% |
| | Precision | 66.7% | 58.3% | 72.0% | 74.2% | 72.0% |
| P4 (Visual Quality / Sharpness) | **Recall** | 22.2% | 44.4% | 61.1% | 77.8% | 100.0% |
| | Precision | 100.0% | 100.0% | 88.0% | 83.9% | 72.0% |
| P5 (Naive Apparent Motion) | **Recall** | 11.1% | 16.7% | 27.8% | 61.1% | 100.0% |
| | Precision | 100.0% | 100.0% | 100.0% | 88.7% | 72.0% |
| P6 (Estimator Uncertainty) | **Recall** | 11.1% | 16.7% | 27.8% | 72.2% | 100.0% |
| | Precision | 100.0% | 100.0% | 100.0% | 100.0% | 72.0% |
| **P7 (Proposed Estimator VOI)** | **Recall** | 27.8% | 38.9% | 38.9% | 83.3% | 100.0% |
| | Precision | 100.0% | 100.0% | 100.0% | 95.2% | 72.0% |

---

