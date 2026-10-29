"""Paired Trajectory Evaluation Pipeline on Common Timestamp Support.

Provides mathematically rigorous, unbiased paired evaluation between two SLAM trajectories
(e.g., Pass A Unfiltered Baseline vs Pass B Estimator-Aware VLM) on their common ground-truth
support intersection:
    T_common = T_matched(A) ∩ T_matched(B)

Prevents policies that drop difficult frames during dynamic motion or rapid rotations from
deceptively claiming lower ATE on smaller, easier subsets. Also performs paired statistical
significance testing (two-tailed paired t-test and Wilcoxon signed-rank test).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Dict, List, Optional, Tuple
import numpy as np
import scipy.stats

from estimator_aware_vlm.datasets.base import (
    Trajectory,
    TrajectoryPose,
    TrajectoryRole,
)
from estimator_aware_vlm.evaluation.trajectory_metrics import (
    ATEStats,
    EvaluationError,
    RPEStats,
    align_trajectories_se3,
    compute_ate,
    compute_rpe,
    match_trajectories_by_timestamp,
)


@dataclass(frozen=True)
class PairedEvaluationResult:
    """Report comparing two trajectories against ground truth on identical common support."""
    sequence_name: str
    traj_a_name: str
    traj_b_name: str
    num_ground_truth_poses: int
    num_common_poses: int
    traj_a_matched_poses: int
    traj_b_matched_poses: int
    common_coverage: float
    traj_a_coverage: float
    traj_b_coverage: float
    jaccard_support_index: float
    ate_a_common: ATEStats
    ate_b_common: ATEStats
    ate_a_independent: ATEStats
    ate_b_independent: ATEStats
    rpe_a_common: RPEStats
    rpe_b_common: RPEStats
    rpe_fixed_time_a_common: Optional[RPEStats]
    rpe_fixed_time_b_common: Optional[RPEStats]
    absolute_ate_reduction_common_m: float
    percentage_ate_improvement_common: float
    mean_pointwise_error_delta_m: float
    ttest_stat: float
    ttest_pvalue: float
    wilcoxon_stat: Optional[float]
    wilcoxon_pvalue: Optional[float]
    is_statistically_significant: bool

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)


def evaluate_paired_trajectories_common_support(
    traj_a: Trajectory,
    traj_b: Trajectory,
    ground_truth: Trajectory,
    max_dt_sec: float = 0.02,
    delta_sec: Optional[float] = 1.0,
    sequence_name: str = "paired_evaluation",
) -> PairedEvaluationResult:
    """Evaluate Trajectory A and Trajectory B against Ground Truth on identical common support.

    Args:
        traj_a: Baseline estimated trajectory (TrajectoryRole.ESTIMATED).
        traj_b: Intervention estimated trajectory (TrajectoryRole.ESTIMATED).
        ground_truth: Reference motion capture or LiDAR trajectory (TrajectoryRole.GROUND_TRUTH).
        max_dt_sec: Maximum timestamp difference for matching with ground truth.
        delta_sec: Interval for fixed-time RPE drift computation.
        sequence_name: Name of evaluated sequence.

    Returns:
        PairedEvaluationResult with common-support ATE/RPE and paired hypothesis tests.

    Raises:
        EvaluationError: If common timestamp support is fewer than 3 poses.
        PermissionError: If trajectory roles are violated.
    """
    traj_a.assert_role(TrajectoryRole.ESTIMATED)
    traj_b.assert_role(TrajectoryRole.ESTIMATED)
    ground_truth.assert_role(TrajectoryRole.GROUND_TRUTH)

    num_gt = len(ground_truth)
    if num_gt < 3:
        raise EvaluationError(f"Ground truth has insufficient poses: {num_gt} < 3")

    gt_timestamps = ground_truth.timestamps

    # 1. Match Trajectory A -> Ground Truth
    matched_a_by_gt_idx: Dict[int, TrajectoryPose] = {}
    used_gt_a = set()
    for pa in traj_a.poses:
        diffs = np.abs(gt_timestamps - pa.timestamp)
        min_idx = int(np.argmin(diffs))
        if diffs[min_idx] <= max_dt_sec and min_idx not in used_gt_a:
            used_gt_a.add(min_idx)
            matched_a_by_gt_idx[min_idx] = pa

    # 2. Match Trajectory B -> Ground Truth
    matched_b_by_gt_idx: Dict[int, TrajectoryPose] = {}
    used_gt_b = set()
    for pb in traj_b.poses:
        diffs = np.abs(gt_timestamps - pb.timestamp)
        min_idx = int(np.argmin(diffs))
        if diffs[min_idx] <= max_dt_sec and min_idx not in used_gt_b:
            used_gt_b.add(min_idx)
            matched_b_by_gt_idx[min_idx] = pb

    len_a_matched = len(matched_a_by_gt_idx)
    len_b_matched = len(matched_b_by_gt_idx)

    # 3. Intersect Ground Truth indices for common support
    common_gt_indices = sorted(list(set(matched_a_by_gt_idx.keys()) & set(matched_b_by_gt_idx.keys())))
    union_gt_indices = set(matched_a_by_gt_idx.keys()) | set(matched_b_by_gt_idx.keys())
    num_common = len(common_gt_indices)
    jaccard_index = float(num_common / len(union_gt_indices)) if union_gt_indices else 0.0

    if num_common < 3:
        raise EvaluationError(
            f"Insufficient common timestamp support between '{traj_a.name}' and '{traj_b.name}': "
            f"only {num_common} common matched poses (threshold=3, tolerance={max_dt_sec}s). "
            f"Tracking supports do not meaningfully overlap."
        )

    # 4. Extract common pose triplets (Pose_A, Pose_B, Pose_GT)
    common_a_poses: List[TrajectoryPose] = [matched_a_by_gt_idx[idx] for idx in common_gt_indices]
    common_b_poses: List[TrajectoryPose] = [matched_b_by_gt_idx[idx] for idx in common_gt_indices]
    common_gt_poses: List[TrajectoryPose] = [ground_truth.poses[idx] for idx in common_gt_indices]

    gt_common_pos = np.array([p.position for p in common_gt_poses], dtype=np.float64)
    a_common_pos = np.array([p.position for p in common_a_poses], dtype=np.float64)
    b_common_pos = np.array([p.position for p in common_b_poses], dtype=np.float64)

    # 5. Perform Horn / Umeyama SE(3) alignment on identical common ground-truth support
    aligned_a_pos, _, _ = align_trajectories_se3(a_common_pos, gt_common_pos)
    aligned_b_pos, _, _ = align_trajectories_se3(b_common_pos, gt_common_pos)

    # 6. Compute ATE on common support
    ate_a_common = compute_ate(aligned_a_pos, gt_common_pos, fail_on_empty=True)
    ate_b_common = compute_ate(aligned_b_pos, gt_common_pos, fail_on_empty=True)

    # 7. Pointwise residual delta and paired statistical testing
    errors_a = np.linalg.norm(aligned_a_pos - gt_common_pos, axis=1)
    errors_b = np.linalg.norm(aligned_b_pos - gt_common_pos, axis=1)
    delta_errors = errors_b - errors_a  # Negative values indicate Traj B improved over Traj A

    # Paired t-test
    t_stat, p_val_ttest = scipy.stats.ttest_rel(errors_a, errors_b)

    # Wilcoxon signed-rank test
    try:
        if np.allclose(errors_a, errors_b, atol=1e-7):
            w_stat = 0.0
            w_pval = 1.0
        else:
            w_res = scipy.stats.wilcoxon(errors_a, errors_b)
            w_stat = float(w_res.statistic)
            w_pval = float(w_res.pvalue)
    except Exception:
        w_stat = None
        w_pval = None

    is_significant = bool(p_val_ttest < 0.05)

    # 8. Compute RPE on common support
    rpe_a_common = compute_rpe(common_a_poses, common_gt_poses, delta=1, fail_on_empty=False)
    rpe_b_common = compute_rpe(common_b_poses, common_gt_poses, delta=1, fail_on_empty=False)

    rpe_fixed_a_common = None
    rpe_fixed_b_common = None
    if delta_sec is not None:
        rpe_fixed_a_common = compute_rpe(common_a_poses, common_gt_poses, delta_sec=delta_sec, max_dt_sec=0.1, fail_on_empty=False)
        rpe_fixed_b_common = compute_rpe(common_b_poses, common_gt_poses, delta_sec=delta_sec, max_dt_sec=0.1, fail_on_empty=False)

    # 9. Compute independent ATE on individual supports for reference
    # Individual A
    indep_a_est = [matched_a_by_gt_idx[idx] for idx in sorted(matched_a_by_gt_idx.keys())]
    indep_a_gt = [ground_truth.poses[idx] for idx in sorted(matched_a_by_gt_idx.keys())]
    if len(indep_a_est) >= 3:
        aligned_indep_a, _, _ = align_trajectories_se3(
            np.array([p.position for p in indep_a_est], dtype=np.float64),
            np.array([p.position for p in indep_a_gt], dtype=np.float64),
        )
        ate_a_indep = compute_ate(aligned_indep_a, np.array([p.position for p in indep_a_gt], dtype=np.float64))
    else:
        ate_a_indep = compute_ate(np.empty((0, 3)), np.empty((0, 3)), fail_on_empty=False)

    # Individual B
    indep_b_est = [matched_b_by_gt_idx[idx] for idx in sorted(matched_b_by_gt_idx.keys())]
    indep_b_gt = [ground_truth.poses[idx] for idx in sorted(matched_b_by_gt_idx.keys())]
    if len(indep_b_est) >= 3:
        aligned_indep_b, _, _ = align_trajectories_se3(
            np.array([p.position for p in indep_b_est], dtype=np.float64),
            np.array([p.position for p in indep_b_gt], dtype=np.float64),
        )
        ate_b_indep = compute_ate(aligned_indep_b, np.array([p.position for p in indep_b_gt], dtype=np.float64))
    else:
        ate_b_indep = compute_ate(np.empty((0, 3)), np.empty((0, 3)), fail_on_empty=False)

    # 10. Performance improvement on common support
    ate_reduction_m = ate_a_common.rmse_m - ate_b_common.rmse_m
    pct_improvement = (ate_reduction_m / ate_a_common.rmse_m * 100.0) if ate_a_common.rmse_m > 0 else 0.0

    return PairedEvaluationResult(
        sequence_name=sequence_name,
        traj_a_name=traj_a.name or "trajectory_a",
        traj_b_name=traj_b.name or "trajectory_b",
        num_ground_truth_poses=num_gt,
        num_common_poses=num_common,
        traj_a_matched_poses=len_a_matched,
        traj_b_matched_poses=len_b_matched,
        common_coverage=float(num_common / num_gt),
        traj_a_coverage=float(len_a_matched / num_gt),
        traj_b_coverage=float(len_b_matched / num_gt),
        jaccard_support_index=jaccard_index,
        ate_a_common=ate_a_common,
        ate_b_common=ate_b_common,
        ate_a_independent=ate_a_indep,
        ate_b_independent=ate_b_indep,
        rpe_a_common=rpe_a_common,
        rpe_b_common=rpe_b_common,
        rpe_fixed_time_a_common=rpe_fixed_a_common,
        rpe_fixed_time_b_common=rpe_fixed_b_common,
        absolute_ate_reduction_common_m=round(ate_reduction_m, 6),
        percentage_ate_improvement_common=round(pct_improvement, 2),
        mean_pointwise_error_delta_m=float(np.mean(delta_errors)),
        ttest_stat=float(t_stat),
        ttest_pvalue=float(p_val_ttest),
        wilcoxon_stat=w_stat,
        wilcoxon_pvalue=w_pval,
        is_statistically_significant=is_significant,
    )
