# Stage 3: Trustworthy Trajectory Evaluation Metrics

**Date**: 2026-10-29  
**Specification**: Stage 3 of [Research Implementation Audit](file:///c:/Users/Yash%20Bhardwaj/Desktop/vlm/docs/research_implementation_audit.md#L120-L132)  
**Package**: [`src/estimator_aware_vlm/evaluation/`](file:///c:/Users/Yash%20Bhardwaj/Desktop/vlm/src/estimator_aware_vlm/evaluation/)  
**Test Suite**: [`tests/test_paired_trajectory_metrics.py`](file:///c:/Users/Yash%20Bhardwaj/Desktop/vlm/tests/test_paired_trajectory_metrics.py) (9/9 passed), [`tests/test_trajectory_metrics.py`](file:///c:/Users/Yash%20Bhardwaj/Desktop/vlm/tests/test_trajectory_metrics.py) (3/3 passed), Total Suite: 41/41 passed  

---

## 1. Executive Summary

Stage 3 eliminates three major sources of evaluation bias and reporting failure identified in the research implementation audit:

1. **Fail-Closed on Evaluation Failures**: Replaced deceptive returns (such as returning `0.0` ATE RMSE on empty or completely lost trajectories) with strict fail-closed [`EvaluationError`](file:///c:/Users/Yash%20Bhardwaj/Desktop/vlm/src/estimator_aware_vlm/evaluation/trajectory_metrics.py#L22) exceptions and explicit `NaN` values for non-fatal reporting.
2. **Paired Trajectory Evaluation on Common Support**: Implemented [`evaluate_paired_trajectories_common_support()`](file:///c:/Users/Yash%20Bhardwaj/Desktop/vlm/src/estimator_aware_vlm/evaluation/paired_metrics.py#L65). When comparing baseline SLAM (Pass A) against filtered SLAM (Pass B), evaluation is computed over the exact common intersection of ground-truth timestamps ($\mathcal{T}_{\text{common}} = \mathcal{T}_A \cap \mathcal{T}_B$). This prevents unprincipled policies that drop difficult frames during dynamic motion from deceptively appearing more accurate.
3. **Fixed-Time Interval RPE ($\Delta t$)**: Extended [`compute_rpe()`](file:///c:/Users/Yash%20Bhardwaj/Desktop/vlm/src/estimator_aware_vlm/evaluation/trajectory_metrics.py#L188) to support fixed-time interval evaluation ($\Delta t = 1.0\text{ s}$), overcoming frame-rate jitter and variable camera capture rates.
4. **Paired Hypothesis Testing**: Integrated point-by-point residual deltas ($\Delta e_i = e_{B, i} - e_{A, i}$) alongside two-tailed paired t-tests and Wilcoxon signed-rank tests to verify statistical significance ($p < 0.05$).

---

## 2. Mathematical Formulations & Implementations

### 2.1 Common Support Intersection & Jaccard Index
Given ground truth timestamps $\mathcal{T}_{\text{gt}}$, Trajectory A matches $\mathcal{T}_A \subseteq \mathcal{T}_{\text{gt}}$ and Trajectory B matches $\mathcal{T}_B \subseteq \mathcal{T}_{\text{gt}}$.

$$\mathcal{T}_{\text{common}} = \mathcal{T}_A \cap \mathcal{T}_B$$

$$\text{Jaccard Support Index} = \frac{|\mathcal{T}_A \cap \mathcal{T}_B|}{|\mathcal{T}_A \cup \mathcal{T}_B|}$$

Both trajectories are aligned via rigid Horn/Umeyama SE(3) transformation onto the identical ground-truth subset:

$$\min_{\mathbf{R} \in \mathrm{SO}(3), \mathbf{t} \in \mathbb{R}^3} \sum_{i \in \mathcal{T}_{\text{common}}} \|\mathbf{R} \mathbf{p}_i + \mathbf{t} - \mathbf{q}_i\|^2$$

$$\text{ATE}_{\text{RMSE}} = \sqrt{\frac{1}{|\mathcal{T}_{\text{common}}|} \sum_{i \in \mathcal{T}_{\text{common}}} \|\mathbf{p}_i^{\text{aligned}} - \mathbf{q}_i\|^2}$$

### 2.2 Fixed-Time Relative Pose Error (RPE)
For each pose $i$ at timestamp $t_i$, a target pose $j$ is matched such that $|(t_j - t_i) - \Delta t| \le \epsilon$. The relative transformation error is:

$$\mathbf{E}_{ij} = (\mathbf{Q}_i^{-1} \mathbf{Q}_j)^{-1} (\mathbf{P}_i^{-1} \mathbf{P}_j)$$

- **Translation Drift**: $d_{ij} = \|\mathbf{E}_{ij}[:3, 3]\|_2$
- **Rotation Drift**: $\theta_{ij} = \arccos\left(\mathrm{clip}\left(\frac{\mathrm{Tr}(\mathbf{E}_{ij}[:3, :3]) - 1}{2}, -1, 1\right)\right)$

### 2.3 Pointwise Residual Delta & Significance Testing
For each pose on common support:

$$\Delta e_i = \|\mathbf{p}_{B, i}^{\text{aligned}} - \mathbf{q}_i\|_2 - \|\mathbf{p}_{A, i}^{\text{aligned}} - \mathbf{q}_i\|_2$$

A negative $\Delta e_i$ confirms local accuracy improvement from dynamic feature suppression. The two-tailed paired t-test calculates:

$$t = \frac{\overline{\Delta e}}{s_{\Delta e} / \sqrt{N}}, \quad p = 2 \cdot (1 - F_{t_{N-1}}(|t|))$$

---

## 3. Package & Script Integration

1. **[`src/estimator_aware_vlm/evaluation/paired_metrics.py`](file:///c:/Users/Yash%20Bhardwaj/Desktop/vlm/src/estimator_aware_vlm/evaluation/paired_metrics.py)**: Added `PairedEvaluationResult` dataclass and `evaluate_paired_trajectories_common_support()`.
2. **[`src/estimator_aware_vlm/evaluation/trajectory_metrics.py`](file:///c:/Users/Yash%20Bhardwaj/Desktop/vlm/src/estimator_aware_vlm/evaluation/trajectory_metrics.py)**: Added `EvaluationError`, `rpe_fixed_time` field, and fail-closed checks on 0-pair evaluations.
3. **[`scripts/run_pass_b_slam.py`](file:///c:/Users/Yash%20Bhardwaj/Desktop/vlm/scripts/run_pass_b_slam.py)**: Automatically detects Pass A baseline trajectory in the experiment suite, computes common-support paired metrics, and exports `paired_common_support_metrics.json`.

---

## 4. Automated Verification Results

All 41 tests pass across the entire pipeline:

```text
tests/test_dataset_contracts.py          ...........                       [ 26%]
tests/test_tum_adapter.py                ......                            [ 41%]
tests/test_slam_runtime.py               ............                      [ 70%]
tests/test_paired_trajectory_metrics.py  .........                         [ 92%]
tests/test_trajectory_metrics.py         ...                               [100%]
============================== 41 passed in 36.89s ==============================
```

| Test Case | Responsibility | Status |
| :--- | :--- | :---: |
| `test_compute_ate_fail_closed_on_empty` | Rejection of 0-matched pairs with `EvaluationError` | PASSED |
| `test_align_trajectories_se3_fail_closed_on_insufficient_points` | Rejection of $< 3$ points for SE(3) alignment | PASSED |
| `test_evaluate_trajectory_fail_closed_on_disjoint_timestamps` | Detection of non-overlapping estimated vs GT timestamps | PASSED |
| `test_compute_rpe_fixed_time` | Verification of fixed-time ($\Delta t = 1.0\text{ s}$) drift calculation | PASSED |
| `test_compute_rpe_fixed_time_with_dropped_frames` | Bridging dropped frames based on elapsed time | PASSED |
| `test_paired_evaluation_common_support_identical` | Exact 0-reduction recovery for identical trajectories | PASSED |
| `test_paired_evaluation_prevents_partial_tracking_bias` | Strict enforcement of common support when a policy loses tracking | PASSED |
| `test_paired_evaluation_statistical_significance` | Detection of significant error reduction via paired t-test ($p < 0.05$) | PASSED |
| `test_paired_evaluation_fails_on_disjoint_support` | Fail-closed when trajectories have disjoint tracking supports | PASSED |

