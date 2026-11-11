"""Tests for baseline trajectory and mapping quality evaluation."""

from __future__ import annotations

import csv
import json
from pathlib import Path
import numpy as np
import pytest

from estimator_aware_vlm.common.paths import get_repo_root
from estimator_aware_vlm.datasets.base import Trajectory, TrajectoryPose, TrajectoryRole
from estimator_aware_vlm.evaluation.baseline_evaluator import (
    compute_map_point_stats,
    compute_trajectory_continuity,
    parse_estimated_trajectory,
    parse_native_execution_log,
)


def test_trajectory_continuity_computation():
    """Verify continuity metrics on a synthetic trajectory with known gaps."""
    poses = [
        TrajectoryPose(0.0, (0.0, 0.0, 0.0), (0.0, 0.0, 0.0, 1.0), TrajectoryRole.ESTIMATED),
        TrajectoryPose(0.033, (1.0, 0.0, 0.0), (0.0, 0.0, 0.0, 1.0), TrajectoryRole.ESTIMATED),
        TrajectoryPose(0.066, (2.0, 0.0, 0.0), (0.0, 0.0, 0.0, 1.0), TrajectoryRole.ESTIMATED),
        TrajectoryPose(0.500, (3.0, 0.0, 0.0), (0.0, 0.0, 0.0, 1.0), TrajectoryRole.ESTIMATED),
        TrajectoryPose(0.533, (4.0, 0.0, 0.0), (0.0, 0.0, 0.0, 1.0), TrajectoryRole.ESTIMATED),
    ]
    traj = Trajectory(poses=poses, role=TrajectoryRole.ESTIMATED)
    stats = compute_trajectory_continuity(traj)

    assert stats.total_poses == 5
    assert pytest.approx(stats.duration_sec, rel=1e-3) == 0.533
    assert pytest.approx(stats.max_gap_sec, rel=1e-3) == 0.434
    assert stats.gaps_over_100ms_count == 1
    assert stats.longest_continuous_run_poses == 3
    assert pytest.approx(stats.path_length_m, rel=1e-3) == 4.0


def test_map_point_stats_parser(tmp_path: Path):
    """Verify ASCII PLY parsing and bounding volume metrics."""
    ply_file = tmp_path / "test_points.ply"
    ply_content = """ply
format ascii 1.0
element vertex 4
property float x
property float y
property float z
end_header
0.0 0.0 0.0
10.0 0.0 0.0
0.0 5.0 0.0
0.0 0.0 2.0
"""
    ply_file.write_text(ply_content, encoding="utf-8")

    stats = compute_map_point_stats(ply_file, path_length_m=10.0)
    assert stats.vertex_count == 4
    assert stats.linear_density_pts_per_m == 0.4
    assert stats.robust_volume_m3 > 0.0


def test_parse_native_execution_log(tmp_path: Path):
    """Verify log parsing extracts tracking losses, relocalizations, and merges."""
    log_file = tmp_path / "test.log"
    log_content = """
New Map created with 500 points
Fail to track local map!
Fail to track local map!
Relocalized!!
*Merge detected
*Loop detected
median tracking time: 0.025
mean tracking time: 0.030
"""
    log_file.write_text(log_content, encoding="utf-8")

    stats = parse_native_execution_log(log_file)
    assert stats.tracking_failure_events == 2
    assert stats.relocalization_events == 1
    assert stats.maps_created_count == 1
    assert stats.map_merges_detected == 1
    assert stats.loop_closures_detected == 1
    assert pytest.approx(stats.median_tracking_time_ms, rel=1e-3) == 25.0
    assert pytest.approx(stats.mean_tracking_time_ms, rel=1e-3) == 30.0


def test_baseline_deliverables_integrity():
    """Verify generated Step 5 deliverables exist, have correct schema, and valid values."""
    repo_root = get_repo_root()
    baselines_dir = repo_root / "results" / "baselines"

    csv_path = baselines_dir / "baseline_comparison.csv"
    assert csv_path.exists(), "baseline_comparison.csv missing"

    with open(csv_path, "r", encoding="utf-8") as f:
        reader = list(csv.DictReader(f))

    assert len(reader) == 8, f"Expected 8 baseline sequences, got {len(reader)}"
    expected_cols = {
        "sequence_name",
        "dataset_type",
        "ground_truth_available",
        "estimated_poses",
        "ground_truth_poses",
        "matched_poses",
        "frame_tracking_rate_pct",
        "ground_truth_sample_coverage_pct",
        "ate_rmse_m",
        "trajectory_path_length_m",
        "map_points_count",
    }
    assert expected_cols.issubset(set(reader[0].keys()))

    # Verify JSON structure
    json_path = baselines_dir / "baseline_metrics.json"
    assert json_path.exists(), "baseline_metrics.json missing"

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    assert data["total_sequences"] == 8
    assert "ground_truth_evaluated" in data
    assert "ground_truth_free" in data
    assert len(data["ground_truth_evaluated"]) == 6
    assert len(data["ground_truth_free"]) == 2

    # Verify TUM XYZ values match authentic SLAM metrics
    tum_xyz = data["ground_truth_evaluated"]["rgbd_dataset_freiburg3_walking_xyz"]
    assert tum_xyz["trajectory_poses"] == 827
    assert tum_xyz["frame_tracking_rate"] == 1.0
    assert pytest.approx(tum_xyz["ate"]["rmse_m"], abs=1e-3) == 0.3957

    # Verify D455 values
    d455_01 = data["ground_truth_free"]["my_desk_recording_01"]
    assert d455_01["trajectory_poses"] == 4980
    assert pytest.approx(d455_01["frame_tracking_rate"], abs=1e-3) == 0.9932
    assert d455_01["cross_session_registration"]["robust_inliers"] == 7

    # Verify Plots
    plots_dir = baselines_dir / "plots"
    assert (plots_dir / "gt_benchmark_trajectories.png").stat().st_size > 10000
    assert (plots_dir / "d455_cross_session_alignment.png").stat().st_size > 10000
