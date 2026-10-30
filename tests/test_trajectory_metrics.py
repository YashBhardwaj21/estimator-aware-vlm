"""Unit and contract tests for SE(3) trajectory alignment, ATE/RPE metrics, and paired evaluation.

Verifies:
1. Rigid SE(3) Horn alignment and exact translation/rotation recovery.
2. Trajectory role enforcement and permission isolation (ground-truth vs estimated).
3. Absolute Trajectory Error (ATE) and Relative Pose Error (RPE) computation.
4. Fail-closed contract invariants (empty inputs, insufficient points, disjoint timestamps).
5. Paired trajectory evaluation on common temporal support to prevent partial-tracking bias.
6. Statistical significance analysis of trajectory improvements.
7. Baseline SLAM trajectory ingestion and evaluation.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Tuple
import numpy as np
import pytest

from estimator_aware_vlm.common.paths import get_repo_root
from estimator_aware_vlm.datasets import (
    Trajectory,
    TrajectoryPose,
    TrajectoryRole,
    parse_tum_ground_truth,
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
from scripts.run_baseline_slam import parse_estimated_trajectory


def make_linear_trajectory(
    timestamps: np.ndarray,
    velocity_mps: float = 1.0,
    offset: Tuple[float, float, float] = (0.0, 0.0, 0.0),
    role: TrajectoryRole = TrajectoryRole.GROUND_TRUTH,
    name: str = "linear_traj",
) -> Trajectory:
    """Helper to generate synthetic linear trajectories."""
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


def test_align_trajectories_se3_exact_recovery():
    """Verify rigid alignment recovers exact translation and 90-deg yaw rotation."""
    theta = np.linspace(0, 2 * np.pi, 20)
    model_pos = np.stack([np.cos(theta), np.sin(theta), np.zeros_like(theta)], axis=1)

    R_true = np.array([
        [0.0, -1.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0],
    ])
    t_true = np.array([10.0, -5.0, 2.0])
    data_pos = (R_true @ model_pos.T).T + t_true

    aligned_pos, R_est, t_est = align_trajectories_se3(model_pos, data_pos)

    assert np.allclose(R_est, R_true, atol=1e-5)
    assert np.allclose(t_est, t_true, atol=1e-5)
    assert np.allclose(aligned_pos, data_pos, atol=1e-5)

    ate = compute_ate(aligned_pos, data_pos)
    assert ate.rmse_m < 1e-6
    assert ate.num_pairs == 20


def test_evaluate_trajectory_role_isolation():
    """Verify evaluation rejects ground truth spoofed as an estimated trajectory."""
    poses_gt = [
        TrajectoryPose(
            timestamp=float(i),
            position=(float(i), 0.0, 0.0),
            orientation=(0.0, 0.0, 0.0, 1.0),
            role=TrajectoryRole.GROUND_TRUTH,
        )
        for i in range(10)
    ]
    gt_traj = Trajectory(poses=poses_gt, role=TrajectoryRole.GROUND_TRUTH)

    with pytest.raises(PermissionError, match="Trajectory role violation"):
        evaluate_trajectory(estimated=gt_traj, ground_truth=gt_traj)


def test_evaluate_trajectory_with_simulated_slam_error():
    """Verify evaluation on simulated noisy estimated trajectory."""
    timestamps = np.linspace(0, 10, 50)
    gt_poses = [
        TrajectoryPose(
            timestamp=float(t),
            position=(float(t), 0.0, 1.0),
            orientation=(0.0, 0.0, 0.0, 1.0),
            role=TrajectoryRole.GROUND_TRUTH,
        )
        for t in timestamps
    ]
    gt_traj = Trajectory(poses=gt_poses, role=TrajectoryRole.GROUND_TRUTH)

    est_poses = [
        TrajectoryPose(
            timestamp=float(t) + 0.005,
            position=(float(t) + 0.05, 0.02, 1.0),
            orientation=(0.0, 0.0, 0.0, 1.0),
            role=TrajectoryRole.ESTIMATED,
        )
        for t in timestamps
    ]
    est_traj = Trajectory(poses=est_poses, role=TrajectoryRole.ESTIMATED)

    result = evaluate_trajectory(est_traj, gt_traj, max_dt_sec=0.02)
    assert result.num_matched_poses == 50
    assert result.tracking_coverage == 1.0
    assert result.ate.rmse_m < 0.01


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
    t = np.linspace(0.0, 3.0, 91)
    gt_traj = make_linear_trajectory(t, velocity_mps=1.0, role=TrajectoryRole.GROUND_TRUTH)
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
    assert abs(rpe_fixed.trans_rmse_m - 0.02) < 0.005


def test_compute_rpe_fixed_time_with_dropped_frames():
    """Verify fixed-time RPE correctly bridges dropped/irregular frames."""
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

    assert res.num_common_poses == 25
    assert res.common_coverage == 0.25
    assert res.traj_a_coverage == 1.0
    assert res.traj_b_coverage == 0.25
    assert res.jaccard_support_index == 0.25

    assert res.ate_a_common.num_pairs == 25
    assert res.ate_b_common.num_pairs == 25


def test_paired_evaluation_statistical_significance():
    """Verify paired evaluation detects statistically significant trajectory improvement."""
    np.random.seed(42)
    t = np.linspace(0, 10, 60)
    gt_traj = make_linear_trajectory(t, role=TrajectoryRole.GROUND_TRUTH)

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


def test_baseline_trajectory_ingestion_and_evaluation():
    """Verify baseline trajectory loads and produces expected dynamic error range."""
    repo_root = get_repo_root()
    traj_path = repo_root / "outputs" / "baseline_slam" / "CameraTrajectory.txt"
    if not traj_path.exists():
        pytest.skip("Baseline CameraTrajectory.txt not yet generated.")

    est_traj = parse_estimated_trajectory(traj_path)
    assert len(est_traj) >= 800
    assert est_traj.role == TrajectoryRole.ESTIMATED

    gt_path = repo_root / "data" / "raw" / "rgbd_dataset_freiburg3_walking_xyz" / "groundtruth.txt"
    if not gt_path.exists():
        pytest.skip("TUM groundtruth.txt not present on disk.")

    gt_traj = parse_tum_ground_truth(gt_path)
    assert len(gt_traj) > 2000
    assert gt_traj.role == TrajectoryRole.GROUND_TRUTH

    result = evaluate_trajectory(est_traj, gt_traj, max_dt_sec=0.02)
    assert result.num_matched_poses >= 800
    assert 0.30 <= result.ate.rmse_m <= 0.60
    assert result.rpe.trans_rmse_m > 0.01
    assert result.rpe.rot_rmse_deg > 0.1
