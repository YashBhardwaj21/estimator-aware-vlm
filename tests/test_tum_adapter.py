"""Unit tests for TUM RGB-D dataset adapter and trajectory role isolation."""

from pathlib import Path
import pytest
import yaml

from estimator_aware_vlm.common.paths import get_repo_root
from estimator_aware_vlm.datasets import (
    CameraIntrinsics,
    DatasetConfig,
    FrameRecord,
    TUMRGBDAdapter,
    TUM_FREIBURG3_INTRINSICS,
    Trajectory,
    TrajectoryPose,
    TrajectoryRole,
    associate_streams,
)


@pytest.fixture
def repo_root() -> Path:
    return get_repo_root()


@pytest.fixture
def fr3_config(repo_root: Path) -> DatasetConfig:
    seq_dir = repo_root / "data" / "raw" / "rgbd_dataset_freiburg3_walking_xyz"
    return DatasetConfig(
        dataset_type="tum_rgbd",
        sequence_name="rgbd_dataset_freiburg3_walking_xyz",
        sequence_dir=seq_dir,
        depth_scale=0.0002,
        association_tolerance_sec=0.02,
        intrinsics=TUM_FREIBURG3_INTRINSICS,
    )


def test_associate_streams_logic():
    """Verify stream association respecting max tolerance."""
    rgb_stream = [(1.00, "rgb/1.png"), (1.03, "rgb/2.png"), (1.10, "rgb/3.png")]
    depth_stream = [(1.01, "depth/1.png"), (1.06, "depth/2.png")]

    # tolerance 0.02 -> 1.00 matches 1.01 (dt=0.01), 1.03 has no match (closest 1.01 dt=0.02, 1.06 dt=0.03)
    matches = associate_streams(rgb_stream, depth_stream, max_difference_sec=0.015)
    assert len(matches) == 1
    assert matches[0] == (1.00, "rgb/1.png", 1.01, "depth/1.png")

    # With wider tolerance 0.04
    matches_wide = associate_streams(rgb_stream, depth_stream, max_difference_sec=0.04)
    assert len(matches_wide) >= 2


def test_tum_adapter_live_ingestion(fr3_config: DatasetConfig):
    """Verify TUM adapter correctly loads raw TUM sequence if present."""
    if not fr3_config.sequence_dir.exists():
        pytest.skip("Raw TUM fr3_walking_xyz sequence directory not present on disk")

    adapter = TUMRGBDAdapter(fr3_config)
    assert len(adapter) > 800  # Expect ~837 synchronized frames
    assert adapter.get_sequence_name() == "rgbd_dataset_freiburg3_walking_xyz"

    intr = adapter.get_camera_intrinsics()
    assert intr.fx == 535.4
    assert intr.width == 640
    assert intr.height == 480

    first_frame = adapter.get_frame(0)
    assert first_frame.rgb_path.exists()
    assert first_frame.depth_path.exists()
    assert first_frame.depth_scale == 0.0002
    assert first_frame.get_safe_estimated_pose() is None

    # Motion capture ground truth begins ~0.04s after RGB-D stream starts;
    # verify that the overwhelming majority of frames (>800) have matched GT poses.
    gt_frames = [f for f in adapter.get_frames() if f.get_evaluation_ground_truth_pose() is not None]
    assert len(gt_frames) > 800
    assert gt_frames[0].get_evaluation_ground_truth_pose() is not None


def test_ground_truth_role_isolation(fr3_config: DatasetConfig):
    """Verify strict prohibition against treating ground-truth trajectory as estimated trajectory."""
    gt_pose = TrajectoryPose(
        timestamp=100.0,
        position=(1.0, 2.0, 3.0),
        orientation=(0.0, 0.0, 0.0, 1.0),
        role=TrajectoryRole.GROUND_TRUTH,
    )

    # Valid assertion of its actual role
    gt_pose.assert_role(TrajectoryRole.GROUND_TRUTH)

    # Attempting to use ground truth as estimated pose must raise PermissionError
    with pytest.raises(PermissionError, match="TrajectoryPose role violation"):
        gt_pose.assert_role(TrajectoryRole.ESTIMATED)

    # Trajectory container validation
    traj = Trajectory(poses=[gt_pose], role=TrajectoryRole.GROUND_TRUTH)
    with pytest.raises(PermissionError, match="Ground-truth trajectory cannot be used as an estimator input"):
        traj.assert_role(TrajectoryRole.ESTIMATED)

    # FrameRecord isolation check
    frame = FrameRecord(
        frame_index=0,
        timestamp=100.0,
        rgb_path=Path("dummy_rgb.png"),
        depth_path=Path("dummy_depth.png"),
        depth_scale=0.0002,
        intrinsics=TUM_FREIBURG3_INTRINSICS,
        estimated_pose=gt_pose,  # Inadvertent leak of GT into estimated slot
    )
    with pytest.raises(PermissionError, match="TrajectoryPose role violation"):
        frame.get_safe_estimated_pose()


def test_yaml_config_loading(repo_root: Path):
    """Verify YAML configuration for fr3_walking_xyz matches benchmark specification."""
    cfg_path = repo_root / "configs" / "datasets" / "tum_rgbd_fr3_walking_xyz.yaml"
    assert cfg_path.exists(), f"Missing config: {cfg_path}"

    with open(cfg_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    assert data["dataset_type"] == "tum_rgbd"
    assert data["depth_scale"] == 0.0002
    assert data["association_tolerance_sec"] == 0.02
    assert data["intrinsics"]["fx"] == 535.4
    assert data["intrinsics"]["fy"] == 539.2
    assert data["trajectory_contract"]["ground_truth_role"] == "GROUND_TRUTH"


def test_load_dataset_config_from_yaml_openloris(repo_root: Path):
    """Verify OpenLORIS YAML profile loads into typed DatasetConfig."""
    from estimator_aware_vlm.datasets import load_dataset_config_from_yaml

    cfg_path = repo_root / "configs" / "datasets" / "openloris_corridor1_4.yaml"
    cfg = load_dataset_config_from_yaml(cfg_path)

    assert cfg.sequence_name == "OpenLORIS-Scene_corridor1-4"
    assert cfg.depth_scale == 0.001
    assert cfg.intrinsics is not None
    assert cfg.intrinsics.width == 848
    assert cfg.intrinsics.height == 480
    assert pytest.approx(cfg.intrinsics.fx, 0.01) == 611.45
    assert cfg.rgb_file is not None and cfg.rgb_file.exists()
    assert cfg.depth_file is not None and cfg.depth_file.exists()


def test_openloris_adapter_ingestion(repo_root: Path):
    """Verify OpenLORISAdapter and get_dataset_adapter ingest OpenLORIS sequence with RealSense intrinsics."""
    from estimator_aware_vlm.datasets import (
        OpenLORISAdapter,
        get_dataset_adapter,
        load_dataset_config_from_yaml,
    )

    cfg_path = repo_root / "configs" / "datasets" / "openloris_corridor1_4.yaml"
    if not (repo_root / "data" / "raw" / "OpenLORIS-Scene_corridor1-4").exists():
        pytest.skip("OpenLORIS raw data directory not present")

    cfg = load_dataset_config_from_yaml(cfg_path)
    assert cfg.dataset_type == "openloris"

    adapter = get_dataset_adapter(cfg)
    assert isinstance(adapter, OpenLORISAdapter)

    assert len(adapter) == 1500  # Exactly 1,500 frames on disk
    assert adapter.get_sequence_name() == "OpenLORIS-Scene_corridor1-4"
    assert adapter.get_camera_intrinsics().width == 848

    f0 = adapter.get_frame(0)
    assert f0.rgb_path.exists()
    assert f0.depth_path.exists()
    assert f0.depth_scale == 0.001

