# OpenLORIS corridor1-4 Benchmark & Feature-Starvation Report

## 1. Overview

OpenLORIS `corridor1-4` represents an indoor service-robot sequence with low environmental visual texture. Unlike TUM `fr3_walking_xyz` where walking humans corrupt landmark mapping, here masking the single walking pedestrian (`person_01`) induces feature starvation, causing localization error to increase.

## 2. Experimental Results

| Arm | Condition | Classification | ATE RMSE (m) | Delta ATE (m) | Improvement | Ref Coverage | Input Tracking |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `C0` | `baseline_KEEP` | `BASELINE_CONTROL` | **1.7045** | +0.0000 | **+0.00%** | 59.60% | 100.00% |
| `C1` | `person_01_IMAGE_POLYGON_MASK` | `INDEPENDENT_SINGLE_CANDIDATE` | **1.7867** | -0.0822 | **-4.82%** | 59.60% | 100.00% |
| `C2` | `person_01_IMAGE_BBOX_MASK` | `DIAGNOSTIC_BBOX` | **1.7853** | -0.0808 | **-4.74%** | 59.60% | 100.00% |

## 3. Scientific Finding: Feature-Starvation Safeguard

- **Baseline Control (C0)**: ATE RMSE = **1.7045 m**
- **Polygon Masking (C1)**: ATE RMSE = **1.7867 m** (Delta = -0.0822 m, -4.82% degradation)
- **Bounding Box Masking (C2)**: ATE RMSE = **1.7853 m** (Delta = -0.0808 m, -4.74% degradation)

This empirical degradation establishes the necessity of an estimator-aware routing policy: when local visual feature density is low, indiscriminately suppressing moving entities can starve the visual odometry front-end, worsening trajectory tracking.