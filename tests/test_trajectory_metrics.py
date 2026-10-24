"""Unit tests for SE(3) trajectory alignment, ATE, RPE, and role verification."""

import math
import numpy as np
import pytest

from estimator_aware_vlm.datasets import Trajectory, TrajectoryPose, TrajectoryRole
from estimator_aware_vlm.evaluation import (
    align_trajectories_se3,
    compute_ate,
    compute_rpe,
    evaluate_trajectory,
)


def test_align_trajectories_se3_exact_recovery():
    """Verify rigid alignment recovers exact translation and 90-deg yaw rotation."""
    # Synthetic circular trajectory in XY
    theta = np.linspace(0, 2 * np.pi, 20)
    model_pos = np.stack([np.cos(theta), np.sin(theta), np.zeros_like(theta)], axis=1)

    # Rotate by 90 degrees around Z and translate by (10, -5, 2)
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

    # Attempting to pass ground-truth trajectory as estimated must raise PermissionError
    with pytest.raises(PermissionError, match="Trajectory role violation"):
        evaluate_trajectory(estimated=gt_traj, ground_truth=gt_traj)


def test_evaluate_trajectory_with_simulated_slam_error():
    """Verify evaluation on simulated noisy estimated trajectory."""
    timestamps = np.linspace(0, 10, 50)
    
    # Ground truth: linear motion along X
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

    # Estimated: has 0.05m drift and slight delay
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
    # After SE(3) alignment, the systematic constant offset is aligned, leaving negligible residual
    assert result.ate.rmse_m < 0.01
