"""Unit tests for Bonn RGB-D Dynamic dataset adapter and configurations."""

import pytest
from pathlib import Path

from estimator_aware_vlm.common.paths import get_repo_root
from estimator_aware_vlm.datasets import (
    CameraIntrinsics,
    FrameRecord,
    Trajectory,
    TrajectoryRole,
    get_dataset_adapter,
    load_dataset_config_from_yaml,
)
from estimator_aware_vlm.datasets.bonn import BonnDatasetAdapter, BONN_XTION_INTRINSICS


@pytest.fixture
def repo_root() -> Path:
    return get_repo_root()


def test_bonn_crowd_adapter_contract(repo_root: Path):
    """Test Bonn crowd dataset loading and contract conformance."""
    cfg_path = repo_root / "configs" / "datasets" / "bonn_crowd.yaml"
    assert cfg_path.exists(), f"Configuration missing: {cfg_path}"

    raw_dir = repo_root / "data" / "raw" / "rgbd_bonn_crowd"
    if not raw_dir.exists():
        pytest.skip("Bonn crowd dataset not present on disk")

    adapter = get_dataset_adapter(cfg_path)
    assert isinstance(adapter, BonnDatasetAdapter)
    assert adapter.get_sequence_name() == "rgbd_bonn_crowd"
    assert len(adapter) == 927

    intr = adapter.get_camera_intrinsics()
    assert isinstance(intr, CameraIntrinsics)
    assert pytest.approx(intr.fx, abs=1e-2) == 542.82
    assert pytest.approx(intr.fy, abs=1e-2) == 542.58
    assert pytest.approx(intr.cx, abs=1e-2) == 315.59
    assert pytest.approx(intr.cy, abs=1e-2) == 237.76
    assert intr.width == 640
    assert intr.height == 480

    # Test FrameRecord contract
    f0 = adapter.get_frame(0)
    assert isinstance(f0, FrameRecord)
    assert f0.rgb_path.exists()
    assert f0.depth_path.exists()
    assert pytest.approx(f0.depth_scale, abs=1e-6) == 0.0002

    # Frame 1 has ground truth pose (MoCap begins 0.02s after frame 0)
    f1 = adapter.get_frame(1)
    assert f1.ground_truth_pose is not None
    assert f1.ground_truth_pose.role == TrajectoryRole.GROUND_TRUTH

    # Test ground truth trajectory
    gt = adapter.get_ground_truth_trajectory()
    assert isinstance(gt, Trajectory)
    assert gt.role == TrajectoryRole.GROUND_TRUTH
    assert len(gt.poses) == 933


def test_bonn_moving_box_adapter_contract(repo_root: Path):
    """Test Bonn moving obstructing box dataset loading and contract conformance."""
    cfg_path = repo_root / "configs" / "datasets" / "bonn_moving_obstructing_box.yaml"
    assert cfg_path.exists(), f"Configuration missing: {cfg_path}"

    raw_dir = repo_root / "data" / "raw" / "rgbd_bonn_moving_obstructing_box"
    if not raw_dir.exists():
        pytest.skip("Bonn moving obstructing box dataset not present on disk")

    adapter = get_dataset_adapter(cfg_path)
    assert isinstance(adapter, BonnDatasetAdapter)
    assert adapter.get_sequence_name() == "rgbd_bonn_moving_obstructing_box"
    assert len(adapter) == 589

    f0 = adapter.get_frame(0)
    assert f0.rgb_path.exists()
    assert f0.depth_path.exists()
    assert pytest.approx(f0.depth_scale, abs=1e-6) == 0.0002

    gt = adapter.get_ground_truth_trajectory()
    assert gt.role == TrajectoryRole.GROUND_TRUTH
    assert len(gt.poses) == 593


def test_bonn_immutability_guarantee(repo_root: Path, tmp_path: Path):
    """Verify that export_associations refuses to write into the raw dataset directory."""
    cfg_path = repo_root / "configs" / "datasets" / "bonn_crowd.yaml"
    if not (repo_root / "data" / "raw" / "rgbd_bonn_crowd").exists():
        pytest.skip("Bonn crowd dataset not present on disk")

    adapter = get_dataset_adapter(cfg_path)

    # Attempt to write into raw directory must raise PermissionError
    forbidden_target = repo_root / "data" / "raw" / "rgbd_bonn_crowd" / "associations.txt"
    with pytest.raises(PermissionError, match="Immutability violation"):
        adapter.export_associations(forbidden_target)

    # Writing to tmp_path should succeed
    safe_target = tmp_path / "associations.txt"
    exported = adapter.export_associations(safe_target)
    assert exported.exists()
    assert exported.stat().st_size > 0


def test_tum_static_adapter_contract(repo_root: Path):
    """Test TUM freiburg3_walking_static dataset loading and contract conformance."""
    cfg_path = repo_root / "configs" / "datasets" / "tum_rgbd_fr3_walking_static.yaml"
    assert cfg_path.exists(), f"Configuration missing: {cfg_path}"

    raw_dir = repo_root / "data" / "raw" / "rgbd_dataset_freiburg3_walking_static"
    if not raw_dir.exists():
        pytest.skip("TUM static dataset not present on disk")

    adapter = get_dataset_adapter(cfg_path)
    assert adapter.get_sequence_name() == "rgbd_dataset_freiburg3_walking_static"
    assert len(adapter) == 717

    f0 = adapter.get_frame(0)
    assert f0.rgb_path.exists()
    assert f0.depth_path.exists()
    assert pytest.approx(f0.depth_scale, abs=1e-6) == 0.0002

    gt = adapter.get_ground_truth_trajectory()
    assert gt.role == TrajectoryRole.GROUND_TRUTH
    assert len(gt.poses) == 2484

