"""Dataset Adapter contract parity, 1-to-1 synchronization, interpolation, and immutability tests."""

from pathlib import Path
import numpy as np
import pytest

from estimator_aware_vlm.common.paths import get_repo_root
from estimator_aware_vlm.datasets import (
    CameraIntrinsics,
    DatasetAdapter,
    DatasetConfig,
    FrameRecord,
    OpenLORISAdapter,
    TUMRGBDAdapter,
    Trajectory,
    TrajectoryPose,
    TrajectoryRole,
    associate_streams,
    get_dataset_adapter,
    load_dataset_config_from_yaml,
    quaternion_slerp,
)


@pytest.fixture
def repo_root() -> Path:
    return get_repo_root()


# ---------------------------------------------------------------------------
# 1. Strict 1-to-1 Stream Synchronization Verification
# ---------------------------------------------------------------------------

def test_associate_streams_empty_streams():
    """Verify empty streams return empty list safely."""
    assert associate_streams([], []) == []
    assert associate_streams([(1.0, "a.png")], []) == []
    assert associate_streams([], [(1.0, "b.png")]) == []


def test_associate_streams_duplicate_timestamps():
    """Verify duplicate timestamps are deduplicated deterministically."""
    stream1 = [(1.0, "a1.png"), (1.0, "a2_dup.png"), (2.0, "a3.png")]
    stream2 = [(1.005, "b1.png"), (2.005, "b2.png")]
    matches = associate_streams(stream1, stream2, max_difference_sec=0.02, one_to_one=True)
    assert len(matches) == 2
    assert matches[0] == (1.0, "a1.png", 1.005, "b1.png")
    assert matches[1] == (2.0, "a3.png", 2.005, "b2.png")


def test_associate_streams_strict_one_to_one():
    """Verify no frame in either stream is paired more than once."""
    # stream1 has two frames close to b1; only the mutually closest should match b1
    # dt(a1, b1) = 0.005; dt(a2, b1) = 0.010
    stream1 = [(1.00, "a1.png"), (1.015, "a2.png")]
    stream2 = [(1.005, "b1.png"), (1.04, "b2.png")]

    matches = associate_streams(stream1, stream2, max_difference_sec=0.02, one_to_one=True)
    # a1 matches b1; a2 is within 0.02 of b1 (diff=0.01) but b1 is already taken by a1!
    # And distance from a2 to b2 is 0.025 > 0.02 (rejected)
    assert len(matches) == 1
    assert matches[0] == (1.00, "a1.png", 1.005, "b1.png")

    # Check that stream2 frame b1 is not matched twice
    matched_depth_paths = [m[3] for m in matches]
    assert len(matched_depth_paths) == len(set(matched_depth_paths))


def test_associate_streams_tolerance_boundary():
    """Verify association strictly respects max_difference_sec boundary."""
    stream1 = [(1.00, "a.png")]
    stream2 = [(1.02, "b_exact.png"), (1.025, "b_outside.png")]

    # Exact boundary (0.02)
    matches_exact = associate_streams(stream1, stream2, max_difference_sec=0.02)
    assert len(matches_exact) == 1
    assert matches_exact[0][2] == 1.02

    # Stricter boundary (0.015)
    matches_strict = associate_streams(stream1, stream2, max_difference_sec=0.015)
    assert len(matches_strict) == 0


# ---------------------------------------------------------------------------
# 2. SE(3) Trajectory Interpolation and Manifold Operations
# ---------------------------------------------------------------------------

def test_quaternion_slerp_properties():
    """Verify SLERP respects rotation interpolation, normalization, and antipodal symmetry."""
    q_id = (0.0, 0.0, 0.0, 1.0)
    # 90 degree rotation around Z: cos(pi/4) = 0.7071, sin(pi/4) = 0.7071
    q_z90 = (0.0, 0.0, float(np.sin(np.pi / 4)), float(np.cos(np.pi / 4)))

    # Midpoint t=0.5 -> 45 degree rotation around Z
    q_mid = quaternion_slerp(q_id, q_z90, 0.5)
    expected_z = float(np.sin(np.pi / 8))
    expected_w = float(np.cos(np.pi / 8))
    assert pytest.approx(q_mid[2], abs=1e-4) == expected_z
    assert pytest.approx(q_mid[3], abs=1e-4) == expected_w

    # Antipodal symmetry: -q represents identical SO(3) rotation
    q_z90_neg = (-q_z90[0], -q_z90[1], -q_z90[2], -q_z90[3])
    q_mid_neg = quaternion_slerp(q_id, q_z90_neg, 0.5)
    assert pytest.approx(abs(q_mid_neg[2]), abs=1e-4) == expected_z
    assert pytest.approx(abs(q_mid_neg[3]), abs=1e-4) == expected_w


def test_trajectory_interpolation_no_extrapolation():
    """Verify trajectory interpolation rejects queries outside timestamp range."""
    p1 = TrajectoryPose(10.0, (0.0, 0.0, 0.0), (0.0, 0.0, 0.0, 1.0), TrajectoryRole.GROUND_TRUTH)
    p2 = TrajectoryPose(11.0, (1.0, 2.0, 3.0), (0.0, 0.0, 0.0, 1.0), TrajectoryRole.GROUND_TRUTH)
    traj = Trajectory(poses=[p1, p2], role=TrajectoryRole.GROUND_TRUTH)

    # Within range
    interp = traj.interpolate_pose_at(10.5)
    assert interp is not None
    assert pytest.approx(interp.position[0]) == 0.5
    assert pytest.approx(interp.position[1]) == 1.0
    assert pytest.approx(interp.position[2]) == 1.5
    assert interp.role == TrajectoryRole.GROUND_TRUTH

    # Strictly no extrapolation
    assert traj.interpolate_pose_at(9.99) is None
    assert traj.interpolate_pose_at(11.01) is None


def test_trajectory_interpolation_gap_rejection():
    """Verify interpolation rejects intervals exceeding max_gap_sec."""
    p1 = TrajectoryPose(10.0, (0.0, 0.0, 0.0), (0.0, 0.0, 0.0, 1.0), TrajectoryRole.GROUND_TRUTH)
    p2 = TrajectoryPose(15.0, (1.0, 0.0, 0.0), (0.0, 0.0, 0.0, 1.0), TrajectoryRole.GROUND_TRUTH)
    traj = Trajectory(poses=[p1, p2], role=TrajectoryRole.GROUND_TRUTH)

    # Gap of 5.0s > default max_gap_sec=0.5 -> returns None
    assert traj.interpolate_pose_at(12.5) is None


# ---------------------------------------------------------------------------
# 3. Adapter Contract Parity and Role Isolation Verification
# ---------------------------------------------------------------------------

def test_adapter_factory_dispatch(repo_root: Path):
    """Verify get_dataset_adapter dispatches correctly for both datasets."""
    tum_cfg_path = repo_root / "configs" / "datasets" / "tum_rgbd_fr3_walking_xyz.yaml"
    openloris_cfg_path = repo_root / "configs" / "datasets" / "openloris_corridor1_4.yaml"

    tum_adapter = get_dataset_adapter(tum_cfg_path)
    assert isinstance(tum_adapter, TUMRGBDAdapter)
    assert isinstance(tum_adapter, DatasetAdapter)

    openloris_adapter = get_dataset_adapter(openloris_cfg_path)
    assert isinstance(openloris_adapter, OpenLORISAdapter)
    assert isinstance(openloris_adapter, DatasetAdapter)


def test_adapter_canonical_frame_contract(repo_root: Path):
    """Verify contract parity across TUM and OpenLORIS adapters."""
    tum_cfg_path = repo_root / "configs" / "datasets" / "tum_rgbd_fr3_walking_xyz.yaml"
    openloris_cfg_path = repo_root / "configs" / "datasets" / "openloris_corridor1_4.yaml"

    adapters = [
        get_dataset_adapter(tum_cfg_path),
        get_dataset_adapter(openloris_cfg_path),
    ]

    for adapter in adapters:
        assert len(adapter) > 0
        name = adapter.get_sequence_name()
        assert len(name) > 0

        intr = adapter.get_camera_intrinsics()
        assert isinstance(intr, CameraIntrinsics)
        assert intr.fx > 0 and intr.fy > 0
        assert intr.width > 0 and intr.height > 0

        gt_traj = adapter.get_ground_truth_trajectory()
        assert isinstance(gt_traj, Trajectory)
        assert gt_traj.role == TrajectoryRole.GROUND_TRUTH

        # Verify first frame contract
        f0 = adapter.get_frame(0)
        assert isinstance(f0, FrameRecord)
        assert f0.rgb_path.exists()
        assert f0.depth_path.exists()
        assert f0.depth_scale > 0
        # Safe estimated pose must be None initially
        assert f0.get_safe_estimated_pose() is None


# ---------------------------------------------------------------------------
# 4. Raw Data Immutability and Security Guard Verification
# ---------------------------------------------------------------------------

def test_raw_data_immutability_prohibition(repo_root: Path):
    """Verify export_associations raises PermissionError when targeted into raw data dir."""
    tum_cfg = load_dataset_config_from_yaml(
        repo_root / "configs" / "datasets" / "tum_rgbd_fr3_walking_xyz.yaml"
    )
    adapter = TUMRGBDAdapter(tum_cfg)

    # Attempting to export directly inside sequence_dir must be rejected
    forbidden_target = tum_cfg.sequence_dir / "associations_stray.txt"
    with pytest.raises(PermissionError, match="Immutability violation"):
        adapter.export_associations(forbidden_target)


def test_raw_data_directory_is_clean(repo_root: Path):
    """Verify data/raw directories contain NO generated association or mask artifacts."""
    raw_dirs = [
        repo_root / "data" / "raw" / "rgbd_dataset_freiburg3_walking_xyz",
        repo_root / "data" / "raw" / "OpenLORIS-Scene_corridor1-4",
    ]

    for d in raw_dirs:
        if not d.exists():
            continue
        # No generated associations or masks should exist
        assert not (d / "associations.txt").exists(), f"Stray associations.txt in {d}"
        assert not (d / "associations_pass_b.txt").exists(), f"Stray associations_pass_b.txt in {d}"
        assert not (d / "associations_pass_b.manifest.json").exists(), f"Stray manifest in {d}"
        assert not (d / "masked_pass_b").exists(), f"Stray masked_pass_b dir in {d}"


# ---------------------------------------------------------------------------
# 5. Canonical Path & Preflight Execution Contracts
# ---------------------------------------------------------------------------

def test_canonical_paths_structure(repo_root: Path):
    """Verify canonical directory structure helpers return valid paths."""
    from estimator_aware_vlm.common.paths import (
        get_raw_data_dir,
        get_processed_data_dir,
        get_mapping_outputs_dir,
        resolve_path,
    )
    assert repo_root.exists()
    assert (repo_root / "pyproject.toml").exists()

    assert get_raw_data_dir() == repo_root / "data" / "raw"
    assert get_processed_data_dir() == repo_root / "data" / "processed"
    assert get_mapping_outputs_dir() == repo_root / "outputs" / "mapping"

    resolved = resolve_path("configs/default.yaml")
    assert resolved == (repo_root / "configs" / "default.yaml").resolve()


def test_preflight_tool_execution():
    """Verify system preflight check returns non-error diagnostic dict."""
    from scripts.check_system_dependencies import run_preflight
    summary = run_preflight()
    assert isinstance(summary, dict)
    assert "system" in summary
    assert "python" in summary
    assert "core_dependencies" in summary
    assert "overall_status" in summary
