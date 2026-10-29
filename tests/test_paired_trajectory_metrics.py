"""Unit and contract tests for trustworthy trajectory evaluation and paired metrics (Stage 3)."""

import math
from typing import Tuple
import numpy as np
import pytest

from estimator_aware_vlm.datasets import (
    Trajectory,
    TrajectoryPose,
    TrajectoryRole,
)
from estimator_aware_vlm.evaluation import (
    ATEStats,
    EvaluationError,
    RPEStats,
    align_trajectories_se3,
    compute_ate,
    compute_rpe,
    evaluate_paired_trajectories_common_support,
    evaluate_trajectory,
)


def make_linear_trajectory(
    timestamps: np.ndarray,
    velocity_mps: float = 1.0,
    offset: Tuple[float, float, float] = (0.0, 0.0, 0.0),
    role: TrajectoryRole = TrajectoryRole.GROUND_TRUTH,
    name: str = "linear_traj",
) -> Trajectory:
    poses = []
    for t in timestamps:
        pos = (offset[0] + velocity_mps * float(t), offset[1], offset[2])
        poses.append(
            TrajectoryPose(
                timestamp=float(t),
                position=pos,
                orientation=(0.0, 0.0, 0.0, 1.0),
                role=role,
            )
        )
    return Trajectory(poses=poses, role=role, name=name)


def test_compute_ate_fail_closed_on_empty():
    """Verify compute_ate raises EvaluationError on empty input when fail_on_empty=True."""
    with pytest.raises(EvaluationError, match="Cannot compute ATE on 0 matched position pairs"):
        compute_ate(np.empty((0, 3)), np.empty((0, 3)), fail_on_empty=True)

    stats = compute_ate(np.empty((0, 3)), np.empty((0, 3)), fail_on_empty=False)
    assert math.isnan(stats.rmse_m)
    assert stats.num_pairs == 0


def test_align_trajectories_se3_fail_closed_on_insufficient_points():
    """Verify rigid alignment rejects fewer than 3 points."""
    pts1 = np.array([[0, 0, 0], [1, 1, 1]])
    pts2 = np.array([[0, 0, 0], [1, 1, 1]])
    with pytest.raises(EvaluationError, match="At least 3 matched points are required"):
        align_trajectories_se3(pts1, pts2)


def test_evaluate_trajectory_fail_closed_on_disjoint_timestamps():
    """Verify evaluate_trajectory raises EvaluationError when no timestamps match."""
    t1 = np.linspace(0, 10, 20)
    t2 = np.linspace(100, 110, 20)

    gt_traj = make_linear_trajectory(t1, role=TrajectoryRole.GROUND_TRUTH)
    est_traj = make_linear_trajectory(t2, role=TrajectoryRole.ESTIMATED, name="disjoint_est")

    with pytest.raises(EvaluationError, match="Insufficient timestamp-aligned poses"):
        evaluate_trajectory(est_traj, gt_traj, max_dt_sec=0.02)


def test_compute_rpe_fixed_time():
    """Verify compute_rpe with delta_sec=1.0 pairs poses based on elapsed time."""
    # 30 Hz for 3 seconds -> 91 frames
    t = np.linspace(0.0, 3.0, 91)
    gt_traj = make_linear_trajectory(t, velocity_mps=1.0, role=TrajectoryRole.GROUND_TRUTH)
    # Estimated trajectory has a drift of 0.02 m/s
    est_traj = make_linear_trajectory(t, velocity_mps=1.02, role=TrajectoryRole.ESTIMATED)

    rpe_fixed = compute_rpe(
        est_traj.poses,
        gt_traj.poses,
        delta_sec=1.0,
        max_dt_sec=0.05,
        fail_on_empty=True,
    )

    assert rpe_fixed.delta_sec == 1.0
    assert rpe_fixed.num_pairs > 0
    # Over 1.0 second, velocity difference of 0.02 m/s produces approx 0.02m drift
    assert abs(rpe_fixed.trans_rmse_m - 0.02) < 0.005


def test_compute_rpe_fixed_time_with_dropped_frames():
    """Verify fixed-time RPE correctly bridges dropped/irregular frames."""
    # Frames at t = 0.0, 0.1, 0.2, then a gap to 1.0, 1.1, 1.2
    t = np.array([0.0, 0.1, 0.2, 1.0, 1.1, 1.2])
    gt_traj = make_linear_trajectory(t, velocity_mps=1.0, role=TrajectoryRole.GROUND_TRUTH)
    est_traj = make_linear_trajectory(t, velocity_mps=1.0, role=TrajectoryRole.ESTIMATED)

    rpe_fixed = compute_rpe(
        est_traj.poses,
        gt_traj.poses,
        delta_sec=1.0,
        max_dt_sec=0.05,
        fail_on_empty=True,
    )

    # Pairs: (0.0, 1.0), (0.1, 1.1), (0.2, 1.2) -> exactly 3 pairs!
    assert rpe_fixed.num_pairs == 3
    assert rpe_fixed.trans_rmse_m < 1e-6


def test_paired_evaluation_common_support_identical():
    """Verify paired evaluation on identical trajectories yields zero reduction."""
    t = np.linspace(0, 5, 50)
    gt_traj = make_linear_trajectory(t, role=TrajectoryRole.GROUND_TRUTH)
    traj_a = make_linear_trajectory(t, role=TrajectoryRole.ESTIMATED, name="traj_a")
    traj_b = make_linear_trajectory(t, role=TrajectoryRole.ESTIMATED, name="traj_b")

    res = evaluate_paired_trajectories_common_support(traj_a, traj_b, gt_traj)
    assert res.num_common_poses == 50
    assert res.common_coverage == 1.0
    assert res.jaccard_support_index == 1.0
    assert res.ate_a_common.rmse_m < 1e-6
    assert res.ate_b_common.rmse_m < 1e-6
    assert res.absolute_ate_reduction_common_m == 0.0


def test_paired_evaluation_prevents_partial_tracking_bias():
    """Verify paired evaluation strictly evaluates on common support when one policy drops frames."""
    t_full = np.linspace(0, 10, 100)
    gt_traj = make_linear_trajectory(t_full, role=TrajectoryRole.GROUND_TRUTH)

    # Trajectory A tracks all 100 frames with small noise
    noise_a = np.sin(t_full) * 0.1
    poses_a = [
        TrajectoryPose(
            timestamp=float(t),
            position=(float(t) + float(noise_a[i]), 0.0, 0.0),
            orientation=(0.0, 0.0, 0.0, 1.0),
            role=TrajectoryRole.ESTIMATED,
        )
        for i, t in enumerate(t_full)
    ]
    traj_a = Trajectory(poses=poses_a, role=TrajectoryRole.ESTIMATED, name="full_tracker_a")

    # Trajectory B only tracks the first 25 easy frames with 0 noise, then loses tracking completely
    t_partial = t_full[:25]
    poses_b = [
        TrajectoryPose(
            timestamp=float(t),
            position=(float(t), 0.0, 0.0),
            orientation=(0.0, 0.0, 0.0, 1.0),
            role=TrajectoryRole.ESTIMATED,
        )
        for t in t_partial
    ]
    traj_b = Trajectory(poses=poses_b, role=TrajectoryRole.ESTIMATED, name="partial_tracker_b")

    res = evaluate_paired_trajectories_common_support(traj_a, traj_b, gt_traj)

    # Verify that the comparison evaluates on the 25 common frames only
    assert res.num_common_poses == 25
    assert res.common_coverage == 0.25
    assert res.traj_a_coverage == 1.0
    assert res.traj_b_coverage == 0.25
    assert res.jaccard_support_index == 0.25  # 25 / 100

    # On common support, ATE of both A and B are evaluated on the exact same 25 poses
    assert res.ate_a_common.num_pairs == 25
    assert res.ate_b_common.num_pairs == 25


def test_paired_evaluation_statistical_significance():
    """Verify paired evaluation detects statistically significant trajectory improvement."""
    t = np.linspace(0, 10, 60)
    gt_traj = make_linear_trajectory(t, role=TrajectoryRole.GROUND_TRUTH)

    # Trajectory A: baseline with significant dynamic-object disturbance (0.20m drift)
    poses_a = [
        TrajectoryPose(
            timestamp=float(t[i]),
            position=(float(t[i]), 0.20 + 0.02 * np.random.randn(), 0.0),
            orientation=(0.0, 0.0, 0.0, 1.0),
            role=TrajectoryRole.ESTIMATED,
        )
        for i in range(len(t))
    ]
    traj_a = Trajectory(poses=poses_a, role=TrajectoryRole.ESTIMATED, name="noisy_baseline")

    # Trajectory B: filtered SLAM with minimal error (0.02m)
    poses_b = [
        TrajectoryPose(
            timestamp=float(t[i]),
            position=(float(t[i]), 0.02 + 0.005 * np.random.randn(), 0.0),
            orientation=(0.0, 0.0, 0.0, 1.0),
            role=TrajectoryRole.ESTIMATED,
        )
        for i in range(len(t))
    ]
    traj_b = Trajectory(poses=poses_b, role=TrajectoryRole.ESTIMATED, name="filtered_slam")

    res = evaluate_paired_trajectories_common_support(traj_a, traj_b, gt_traj)

    assert res.absolute_ate_reduction_common_m > 0.0
    assert res.percentage_ate_improvement_common > 50.0
    assert res.mean_pointwise_error_delta_m < 0.0
    assert res.is_statistically_significant is True
    assert res.ttest_pvalue < 1e-4


def test_paired_evaluation_fails_on_disjoint_support():
    """Verify evaluate_paired_trajectories_common_support fails closed when support is disjoint."""
    t_a = np.linspace(0, 5, 20)
    t_b = np.linspace(50, 55, 20)
    gt_traj = make_linear_trajectory(np.linspace(0, 60, 100), role=TrajectoryRole.GROUND_TRUTH)

    traj_a = make_linear_trajectory(t_a, role=TrajectoryRole.ESTIMATED, name="traj_a")
    traj_b = make_linear_trajectory(t_b, role=TrajectoryRole.ESTIMATED, name="traj_b")

    with pytest.raises(EvaluationError, match="Insufficient common timestamp support"):
        evaluate_paired_trajectories_common_support(traj_a, traj_b, gt_traj)
