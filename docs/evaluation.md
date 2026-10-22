# Evaluation

## 1. Evaluation Protocol and Populations

The evaluation protocol measures affordance classification accuracy, compute efficiency, and downstream SLAM map reliability across separate, rigorously controlled populations.

### 1.1 Evaluated Populations

1. **Visual Classification Benchmark ($N = 42$)**:
   - 42 physical object instances with verified multi-view crop and context images (3 `IMMOVABLE`, 37 `MOVABLE`, 2 `MOBILE`).
   - Evaluates visual affordance prediction under `CROP_ONLY` and `CROP_PLUS_CONTEXT` view modes.
2. **Estimator-Aware Router Benchmark ($N = 53$)**:
   - 53 physical entities evaluated in the multi-session scene graph.
   - Evaluates decision routing, query cost reduction, and downstream landmark lifecycle actions across 12 distinct policies.
3. **Controlled Relocation Dataset ($N = 3$)**:
   - 3 physical objects subjected to physical, tape-measured relocations between mapping sessions (`chair_002`, `chair_003`, `storage_box_007`).

---

## 2. Baseline Policies

The routing framework evaluates eight primary policies on the 53-entity benchmark dataset:

| Policy ID | Policy Name | Description |
|---|---|---|
| **B0** | `STATIC_MAJORITY` | Trivial baseline predicting `MOVABLE` for all objects; zero VLM queries. |
| **B1** | `SEMANTIC_PRIOR` | Static lookup mapping object class names to default affordances; zero VLM queries. |
| **B2** | `GEOMETRY_MOTION` | Threshold-based displacement classification without semantic reasoning; zero VLM queries. |
| **B3** | `ZERO_SHOT_CROP_ONLY` | Always queries VLM using isolated 2D bounding box crops. |
| **B4** | `ZERO_SHOT_CROP_PLUS_CONTEXT` | Always queries VLM using crop and full contextual frame. |
| **B5** | `ALWAYS_VLM` | Dispatches VLM inference for all image-eligible instances. |
| **B6** | `ESTIMATOR_AWARE_ROUTER` | **Proposed policy**: Dispatches VLM only when estimator utility exceeds query cost ($U(q) > 0$). |
| **B7** | `ORACLE` | Theoretical upper bound with optimal routing dispatch. |

---

## 3. Metrics and Mathematical Formulations

### 3.1 Classification Metrics

- **Overall Accuracy**:
  $$\text{Accuracy} = \frac{\sum_{c} \text{TP}_c}{N}$$
- **Balanced Accuracy**: Macro-average of recall across all classes, preventing majority class dominance:
  $$\text{Balanced Accuracy} = \frac{1}{|C|} \sum_{c \in C} \frac{\text{TP}_c}{\text{TP}_c + \text{FN}_c}$$
- **Macro-F1**: Unweighted mean of per-class F1 scores:
  $$\text{Macro-F1} = \frac{1}{|C|} \sum_{c \in C} \frac{2 \cdot \text{Precision}_c \cdot \text{Recall}_c}{\text{Precision}_c + \text{Recall}_c}$$

### 3.2 Compute & Efficiency Metrics

- **VLM Query Rate**: Proportion of instances escalated to the VLM:
  $$\text{Query Rate} = \frac{N_{\text{queries}}}{N_{\text{total}}}$$
- **Relative Query Reduction**: Reduction in VLM queries relative to the Always-VLM baseline (B5):
  $$\text{Reduction} = \frac{N_{\text{queries}}(B5) - N_{\text{queries}}(B6)}{N_{\text{queries}}(B5)}$$
- **Normalized Compute Cost**: Parameterized inference cost ($\lambda_{\text{cost}} = 0.5$):
  $$\text{Cost} = 0.5 \times \text{Query Rate}$$

### 3.3 Downstream SLAM Landmark Lifecycle Metrics

- **Landmark Action Accuracy**: Proportion of entities receiving the correct downstream action (e.g. retain, decouple pose, or purge).
- **Landmark Retention Recall**: Proportion of true stationary landmarks correctly retained in the SLAM map:
  $$\text{Retention Recall} = \frac{\text{Retained Stationary Landmarks}}{\text{Total Stationary Landmarks}}$$
- **False Invalidation Rate**: Proportion of stationary landmarks incorrectly culled from the SLAM map.
- **Landmark Corruption Risk**: Count of dynamic agents or moved objects incorrectly retained as static landmarks in the SLAM map (lower is better; target = 0).
- **Mobile Agent Rejection Rate**: Proportion of dynamic mobile agents correctly purged from static landmark constraints (target = 100%).

### 3.4 Relocation Precision Metrics

- **Mean Absolute Error (MAE)**:
  $$\text{MAE} = \frac{1}{K} \sum_{k=1}^K |d_{\text{measured}, k} - d_{\text{gt}, k}|$$
- **Mean Relative Error (MRE)**:
  $$\text{MRE} = \frac{1}{K} \sum_{k=1}^K \frac{|d_{\text{measured}, k} - d_{\text{gt}, k}|}{d_{\text{gt}, k}}$$

---

## 4. Empirical Benchmark Results

### 4.1 Visual Classification Results (Qwen2.5-VL-3B, $N = 42$)

Evaluated on the expanded 42-instance visual dataset:

| View Mode | Overall Accuracy | Balanced Accuracy | Macro-F1 | Mean Latency (s) | Median Latency (s) |
|---|---|---|---|---|---|
| `CROP_ONLY` | 92.9% | 94.4% | 85.1% | 2.01s | 1.88s |
| `CROP_PLUS_CONTEXT` | **95.2%** | **98.2%** | **90.7%** | 2.15s | 1.96s |

Providing contextual scene frames alongside the tight bounding box crop improves Macro-F1 by +5.6 percentage points, resolving ambiguities where isolated crops lack scale and mounting cues.

---

### 4.2 Router Policy Comparison ($N = 53$)

Evaluated across all policies on the complete 53-entity physical population:

| Policy | Overall Acc | Balanced Acc | Macro-F1 | Query Rate | Compute Cost | Landmark Retention | Corruption Risk | Agent Rejection |
|---|---|---|---|---|---|---|---|---|
| **B0** (Static Majority) | 84.9% | 33.3% | 30.6% | 0.0% (0/53) | 0.000 | 100.0% | 2 instances | 0.0% |
| **B1** (Semantic Prior) | 98.1% | 93.3% | 95.9% | 0.0% (0/53) | 0.000 | 96.0% | 0 instances | 100.0% |
| **B2** (Geometry & Motion) | 96.2% | 91.1% | 90.9% | 0.0% (0/53) | 0.000 | 94.0% | 0 instances | 100.0% |
| **B3** (Crop Only) | 73.6% | 74.4% | 72.8% | 79.2% (42/53) | 0.396 | 76.0% | 1 instance | 100.0% |
| **B4** (Crop + Context) | 75.5% | 76.7% | 75.8% | 79.2% (42/53) | 0.396 | 78.0% | 1 instance | 100.0% |
| **B5** (Always VLM) | 75.5% | 76.7% | 75.8% | 79.2% (42/53) | 0.396 | 78.0% | 1 instance | 100.0% |
| **B6** (Estimator-Aware) | **98.1%** | **99.3%** | **96.6%** | **17.0%** (9/53) | **0.085** | **94.0%** | **0 instances** | **100.0%** |
| **B7** (Oracle) | 100.0% | 100.0% | 100.0% | 1.9% (1/53) | 0.009 | 100.0% | 0 instances | 100.0% |

#### Key Takeaways:
1. **Accuracy vs Cost Frontier**: The proposed estimator-aware router (B6) matches the highest overall accuracy (98.1%) and achieves the highest Balanced Accuracy (99.3%) and Macro-F1 (96.6%) among non-oracle policies.
2. **Query Reduction**: B6 queries the VLM on only 9 of 53 instances (17.0% query rate), achieving a **78.6% reduction in VLM inference calls** compared to Always-VLM (42 queries).
3. **Always-VLM Failure Mode**: Policies that query the VLM unconditionally (B3, B4, B5) suffer from linguistic mobility bias, incorrectly reclassifying immovable fixtures (such as wall-mounted displays and ceiling fixtures) as movable objects.
4. **Map Protection**: B6 achieves **0 landmark corruption risk** (no moving or displaced objects retained as static constraints) and **100% dynamic agent rejection**, while retaining 94.0% of valid stationary landmarks.

---

### 4.3 Controlled Relocation Precision

Measured across the three physical relocation events:

| Object ID | Description | Ground Truth Disp | Measured Disp | Absolute Error | Relative Error |
|---|---|---|---|---|---|
| `chair_002` | Caster chair | 0.5630 m | 0.4889 m | 0.0741 m | 13.16% |
| `chair_003` | Office chair | 0.3860 m | 0.3512 m | 0.0348 m | 9.02% |
| `storage_box_007` | Floor container | 0.3960 m | 0.5729 m | 0.1769 m | 44.66% |
| **Aggregate** | — | — | — | **0.0953 m (9.53 cm)** | **22.28%** |

The aggregate mean relative error of 22.28% strictly satisfies the $\le 30.0\%$ acceptance threshold. Stationary control objects demonstrated 100.0% static stability with 0.0% false motion detections.

---

## 5. Scope Boundaries and Trajectory Evaluation

### Trajectory ATE/RPE Boundary
Because external 6-DoF optical motion capture tracking (e.g. Vicon / OptiTrack) was not recorded during dataset collection, camera poses are derived solely from RTAB-Map visual odometry and loop-closure constraints.

Calculating Absolute Trajectory Error (ATE) or Relative Pose Error (RPE) against RTAB-Map's own trajectory would evaluate internal loop-closure residual, not ground-truth physical trajectory error. Consequently, all trajectory metric fields are explicitly recorded as:

```
NOT_EVALUATED_NO_INDEPENDENT_TRAJECTORY_REFERENCE
```

Evaluation strictly focuses on component-wise ablation of semantic mapping, physical object affordances, relocation measurement, and landmark lifecycle protection against human ground-truth annotations and physical tape measurements.
