"""Smoke test for Baseline SLAM trajectory ingestion and evaluation (Pass A)."""

from pathlib import Path
import pytest

from estimator_aware_vlm.common.paths import get_repo_root
from estimator_aware_vlm.datasets import (
    TrajectoryRole,
    parse_tum_ground_truth,
)
from estimator_aware_vlm.evaluation import evaluate_trajectory
from scripts.run_baseline_slam import parse_estimated_trajectory


def test_baseline_trajectory_ingestion_and_evaluation():
    """Verify Pass A baseline trajectory loads and produces expected dynamic error range."""
    repo_root = get_repo_root()
    traj_path = repo_root / "outputs" / "baseline_slam" / "CameraTrajectory.txt"
    if not traj_path.exists():
        pytest.skip("Baseline CameraTrajectory.txt not yet generated.")

    # Ingest estimated trajectory
    est_traj = parse_estimated_trajectory(traj_path)
    assert len(est_traj) >= 800
    assert est_traj.role == TrajectoryRole.ESTIMATED

    # Ingest ground truth trajectory
    gt_path = repo_root / "data" / "raw" / "rgbd_dataset_freiburg3_walking_xyz" / "groundtruth.txt"
    gt_traj = parse_tum_ground_truth(gt_path)
    assert len(gt_traj) > 2000
    assert gt_traj.role == TrajectoryRole.GROUND_TRUTH

    # Evaluate
    result = evaluate_trajectory(est_traj, gt_traj, max_dt_sec=0.02)
    assert result.num_matched_poses >= 800
    
    # Baseline error on fr3_walking_xyz under unmitigated human motion is typically 35-50 cm ATE RMSE
    assert 0.30 <= result.ate.rmse_m <= 0.60
    assert result.rpe.trans_rmse_m > 0.01
    assert result.rpe.rot_rmse_deg > 0.1
