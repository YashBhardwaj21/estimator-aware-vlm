"""Unit tests for RGB-D object geometry extraction and 3D coordinate transformation."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np
import pytest

from estimator_aware_vlm.datasets.base import (
    CameraIntrinsics,
    Trajectory,
    TrajectoryPose,
    TrajectoryRole,
)
from estimator_aware_vlm.perception.object_geometry import (
    ObjectGeometryResult,
    compute_boundary_depth_discontinuity,
    extract_object_geometry,
)
from estimator_aware_vlm.perception.unified_pipeline import (
    SequenceSpec,
    UnifiedPerceptionPipeline,
)


@pytest.fixture
def standard_intrinsics() -> CameraIntrinsics:
    return CameraIntrinsics(
        fx=500.0,
        fy=500.0,
        cx=320.0,
        cy=240.0,
        width=640,
        height=480,
    )


# ---------------------------------------------------------------------------
# 1. Analytic Pinhole Back-Projection Contract
# ---------------------------------------------------------------------------

def test_pinhole_back_projection_analytic_parity(standard_intrinsics: CameraIntrinsics):
    """Verify vectorized back-projection matches analytic pinhole equation:
    X = (u - cx) * Z / fx
    Y = (v - cy) * Z / fy
    Z = Z
    """
    depth_raw = np.zeros((480, 640), dtype=np.uint16)
    # Place synthetic object at u=370, v=290 with depth 2.0m (scale 0.001 -> raw 2000)
    # Box from (360, 280) to (380, 300)
    depth_raw[280:300, 360:380] = 2000
    depth_scale = 0.001

    bbox = (360, 280, 380, 300)
    geom = extract_object_geometry(
        depth_img_raw=depth_raw,
        depth_scale=depth_scale,
        bbox_xyxy=bbox,
        intrinsics=standard_intrinsics,
        min_depth_m=0.25,
        max_depth_m=5.0,
    )

    assert geom is not None
    # Expected centroid: u_mid = 369.5, v_mid = 289.5
    # X_expected = (369.5 - 320) * 2.0 / 500 = 0.198m
    # Y_expected = (289.5 - 240) * 2.0 / 500 = 0.198m
    # Z_expected = 2.0m
    expected_x = (369.5 - 320.0) * 2.0 / 500.0
    expected_y = (289.5 - 240.0) * 2.0 / 500.0

    assert pytest.approx(geom.centroid_camera[0], abs=0.01) == expected_x
    assert pytest.approx(geom.centroid_camera[1], abs=0.01) == expected_y
    assert pytest.approx(geom.centroid_camera[2], abs=0.001) == 2.0


# ---------------------------------------------------------------------------
# 2. Camera-to-World SE(3) Rigid Transformation
# ---------------------------------------------------------------------------

def test_se3_rigid_body_transformation(standard_intrinsics: CameraIntrinsics):
    """Verify world-coordinate transformation: P_world = R_wc * P_cam + t_wc."""
    depth_raw = np.zeros((480, 640), dtype=np.uint16)
    # Object centered at principal point (320, 240) at depth 3.0m
    depth_raw[230:250, 310:330] = 3000
    depth_scale = 0.001

    # Camera pose: camera positioned at (1.0, 2.0, 0.5) in world frame,
    # rotated 90 degrees around Z axis: (x_w, y_w, z_w) = (-y_c + 1, x_c + 2, z_c + 0.5)
    # Quaternion for 90 deg rotation around Z: (0, 0, sin(45), cos(45)) = (0, 0, 0.7071068, 0.7071068)
    q_z90 = (0.0, 0.0, math.sin(math.pi / 4), math.cos(math.pi / 4))
    t_wc = (1.0, 2.0, 0.5)

    cam_pose = TrajectoryPose(
        timestamp=100.0,
        position=t_wc,
        orientation=q_z90,
        role=TrajectoryRole.ESTIMATED,
    )

    bbox = (310, 230, 330, 250)
    geom = extract_object_geometry(
        depth_img_raw=depth_raw,
        depth_scale=depth_scale,
        bbox_xyxy=bbox,
        intrinsics=standard_intrinsics,
        camera_pose=cam_pose,
    )

    assert geom is not None
    assert geom.world_position_valid is True
    assert geom.world_pose_provenance == "ESTIMATED_SLAM"
    assert geom.centroid_world is not None

    # In camera frame, point is approximately (0, 0, 3.0)
    # Applying R_z(90 deg):
    # [ 0 -1  0 ] [ 0 ]   [ 0 ]
    # [ 1  0  0 ] [ 0 ] = [ 0 ]
    # [ 0  0  1 ] [ 3 ]   [ 3 ]
    # Adding t_wc: (0 + 1.0, 0 + 2.0, 3.0 + 0.5) = (1.0, 2.0, 3.5)
    assert pytest.approx(geom.centroid_world[0], abs=0.05) == 1.0
    assert pytest.approx(geom.centroid_world[1], abs=0.05) == 2.0
    assert pytest.approx(geom.centroid_world[2], abs=0.05) == 3.5


# ---------------------------------------------------------------------------
# 3. Mask-Level Depth Isolation vs. Box Background Contamination
# ---------------------------------------------------------------------------

def test_mask_guided_depth_isolation_eliminates_background(standard_intrinsics: CameraIntrinsics):
    """Verify that using mask contour isolates foreground depth and excludes background."""
    depth_raw = np.zeros((480, 640), dtype=np.uint16)
    # Fill bounding box area with background depth = 5000 (5.0m)
    depth_raw[100:200, 100:200] = 5000

    # Put a triangular foreground object inside the box with foreground depth = 1500 (1.5m)
    pts = np.array([[110, 110], [180, 110], [145, 180]], dtype=np.int32)
    cv2.fillPoly(depth_raw, [pts], 1500)

    bbox = (100, 100, 200, 200)
    contour = pts.tolist()

    # Geometry with mask contour: should strictly measure ~1.5m
    geom_with_mask = extract_object_geometry(
        depth_img_raw=depth_raw,
        depth_scale=0.001,
        bbox_xyxy=bbox,
        intrinsics=standard_intrinsics,
        mask_contour=contour,
    )

    # Geometry without mask contour (bounding-box fallback): contaminated by 5.0m background
    geom_box_only = extract_object_geometry(
        depth_img_raw=depth_raw,
        depth_scale=0.001,
        bbox_xyxy=bbox,
        intrinsics=standard_intrinsics,
        mask_contour=None,
    )

    assert geom_with_mask is not None
    assert geom_box_only is not None

    # Mask contour must isolate foreground
    assert pytest.approx(geom_with_mask.depth_median_m, abs=0.01) == 1.5
    assert geom_with_mask.depth_mad_m < 0.05

    # Bounding box without mask is contaminated by background
    assert geom_box_only.depth_median_m > 3.0


# ---------------------------------------------------------------------------
# 4. Robust Depth Statistics and Dispersion
# ---------------------------------------------------------------------------

def test_robust_depth_dispersion_and_percentiles(standard_intrinsics: CameraIntrinsics):
    """Verify median, MAD, 10th and 90th percentiles are calculated correctly."""
    depth_raw = np.zeros((480, 640), dtype=np.uint16)
    # Create gradient depth within a 20x20 box: values from 1000 to 2000 (1.0m to 2.0m)
    vals = np.linspace(1000, 2000, 400).reshape((20, 20)).astype(np.uint16)
    depth_raw[100:120, 100:120] = vals

    bbox = (100, 100, 120, 120)
    geom = extract_object_geometry(
        depth_img_raw=depth_raw,
        depth_scale=0.001,
        bbox_xyxy=bbox,
        intrinsics=standard_intrinsics,
    )

    assert geom is not None
    assert pytest.approx(geom.depth_median_m, abs=0.02) == 1.5
    assert geom.depth_p10_m < geom.depth_median_m
    assert geom.depth_p90_m > geom.depth_median_m
    assert pytest.approx(geom.depth_p10_m, abs=0.05) == 1.1
    assert pytest.approx(geom.depth_p90_m, abs=0.05) == 1.9
    assert geom.depth_mad_m > 0.15


# ---------------------------------------------------------------------------
# 5. Mask Boundary Depth Discontinuity
# ---------------------------------------------------------------------------

def test_boundary_depth_discontinuity():
    """Verify boundary depth step measurement across object perimeter."""
    depth_full = np.full((100, 100), 4.0, dtype=np.float64)  # Background at 4.0m
    mask = np.zeros((100, 100), dtype=np.uint8)
    mask[30:70, 30:70] = 1  # Foreground at 1.5m
    depth_full[30:70, 30:70] = 1.5

    discontinuity = compute_boundary_depth_discontinuity(
        depth_m_full=depth_full,
        mask_binary=mask,
        kernel_size=5,
    )

    # Discontinuity should be approximately |4.0 - 1.5| = 2.5m
    assert pytest.approx(discontinuity, abs=0.1) == 2.5


# ---------------------------------------------------------------------------
# 6. Role Integrity: Rejection of Ground Truth Poses
# ---------------------------------------------------------------------------

def test_ground_truth_pose_rejection(standard_intrinsics: CameraIntrinsics):
    """Verify that TrajectoryRole.GROUND_TRUTH poses are strictly rejected."""
    depth_raw = np.full((480, 640), 2000, dtype=np.uint16)
    gt_pose = TrajectoryPose(
        timestamp=1.0,
        position=(0.0, 0.0, 0.0),
        orientation=(0.0, 0.0, 0.0, 1.0),
        role=TrajectoryRole.GROUND_TRUTH,
    )

    with pytest.raises((PermissionError, ValueError, RuntimeError), match="role violation"):
        extract_object_geometry(
            depth_img_raw=depth_raw,
            depth_scale=0.001,
            bbox_xyxy=(100, 100, 200, 200),
            intrinsics=standard_intrinsics,
            camera_pose=gt_pose,
        )


# ---------------------------------------------------------------------------
# 7. Prohibition of Identity Fallback for Moving Cameras
# ---------------------------------------------------------------------------

def test_moving_camera_identity_fallback_prohibited(tmp_path: Path):
    """Verify that moving-camera sequences raise RuntimeError if estimated trajectory is missing."""
    pipeline = UnifiedPerceptionPipeline(
        repo_root=tmp_path,
        output_dir=tmp_path / "out",
        yolo_model_name="models/yolov8s-seg.pt",
    )

    moving_spec = SequenceSpec(
        sequence_id="tum_xyz",
        dataset_yaml="configs/datasets/tum_rgbd_fr3_walking_xyz.yaml",
        estimated_trajectory_path="nonexistent_trajectory.txt",
    )

    with pytest.raises(RuntimeError, match="Missing estimated camera trajectory for moving-camera sequence"):
        pipeline._load_trajectory(moving_spec, [1.0, 2.0, 3.0])


# ---------------------------------------------------------------------------
# 8. Serialization Roundtrip Integrity
# ---------------------------------------------------------------------------

def test_object_geometry_serialization_roundtrip(standard_intrinsics: CameraIntrinsics):
    """Verify ObjectGeometryResult to_dict and from_dict roundtrip."""
    depth_raw = np.full((480, 640), 2000, dtype=np.uint16)
    cam_pose = TrajectoryPose(
        timestamp=1.0,
        position=(1.0, 0.5, 0.2),
        orientation=(0.0, 0.0, 0.0, 1.0),
        role=TrajectoryRole.ESTIMATED,
    )

    geom = extract_object_geometry(
        depth_img_raw=depth_raw,
        depth_scale=0.001,
        bbox_xyxy=(100, 100, 200, 200),
        intrinsics=standard_intrinsics,
        camera_pose=cam_pose,
    )
    assert geom is not None

    d = geom.to_dict()
    roundtripped = ObjectGeometryResult.from_dict(d)

    assert roundtripped.centroid_camera == pytest.approx(geom.centroid_camera)
    assert roundtripped.centroid_world == pytest.approx(geom.centroid_world)
    assert roundtripped.depth_median_m == pytest.approx(geom.depth_median_m)
    assert roundtripped.depth_mad_m == pytest.approx(geom.depth_mad_m)
    assert roundtripped.valid_depth_fraction == pytest.approx(geom.valid_depth_fraction)
    assert roundtripped.world_position_valid == geom.world_position_valid
    assert roundtripped.world_pose_provenance == geom.world_pose_provenance


# ---------------------------------------------------------------------------
# 9. Deliverable Integrity and Schema Verification
# ---------------------------------------------------------------------------

def test_geometry_deliverables_exist():
    """Verify all Step 7 deliverables exist and are non-empty."""
    from estimator_aware_vlm.common.paths import get_repo_root
    root = get_repo_root()
    geom_dir = root / "results" / "geometry"

    obs_file = geom_dir / "observations_3d.jsonl"
    metrics_file = geom_dir / "geometry_metrics.json"
    report_file = geom_dir / "geometry_quality_report.md"

    assert obs_file.exists(), f"Missing {obs_file}"
    assert metrics_file.exists(), f"Missing {metrics_file}"
    assert report_file.exists(), f"Missing {report_file}"

    assert obs_file.stat().st_size > 0
    assert metrics_file.stat().st_size > 0
    assert report_file.stat().st_size > 0


def test_geometry_observations_schema_and_provenance():
    """Verify that every 3D observation contains required geometry attributes and valid provenance."""
    from estimator_aware_vlm.common.paths import get_repo_root
    root = get_repo_root()
    obs_file = root / "results" / "geometry" / "observations_3d.jsonl"

    count = 0
    with open(obs_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            count += 1

            # Required identifiers
            assert "obs_id" in rec
            assert "sequence_id" in rec
            assert "track_id" in rec
            assert "frame_index" in rec
            assert "timestamp" in rec

            # Required 3D geometric properties
            assert "centroid_camera" in rec
            assert len(rec["centroid_camera"]) == 3
            assert all(math.isfinite(x) for x in rec["centroid_camera"])

            assert "dimensions_3d_m" in rec
            assert len(rec["dimensions_3d_m"]) == 3
            assert all(d > 0 for d in rec["dimensions_3d_m"])

            assert "centroid_covariance_camera" in rec
            assert len(rec["centroid_covariance_camera"]) == 9

            # Depth statistics
            assert "median_depth_m" in rec
            assert 0.25 <= rec["median_depth_m"] <= 8.0
            assert "depth_mad_m" in rec
            assert rec["depth_mad_m"] >= 0.0
            assert "valid_depth_fraction" in rec
            assert 0.0 < rec["valid_depth_fraction"] <= 1.0
            assert "depth_confidence" in rec
            assert 0.0 <= rec["depth_confidence"] <= 1.0

            # World transformation and provenance
            assert rec["world_position_valid"] is True
            assert rec["world_pose_provenance"] == "ESTIMATED_SLAM"
            assert rec["centroid_world"] is not None
            assert len(rec["centroid_world"]) == 3
            assert all(math.isfinite(x) for x in rec["centroid_world"])

    assert count >= 1000, f"Expected at least 1000 observations, found {count}"


def test_geometry_metrics_summary():
    """Verify that geometry metrics cover all eight benchmark sequences with 100% world transformation rate."""
    from estimator_aware_vlm.common.paths import get_repo_root
    root = get_repo_root()
    metrics_file = root / "results" / "geometry" / "geometry_metrics.json"

    with open(metrics_file, "r", encoding="utf-8") as f:
        metrics = json.load(f)

    assert "global" in metrics
    assert "sequences" in metrics

    assert len(metrics["sequences"]) == 8
    seq_ids = {s["sequence_id"] for s in metrics["sequences"]}
    expected_ids = {
        "tum_xyz",
        "tum_halfsphere",
        "tum_static",
        "openloris",
        "bonn_crowd",
        "bonn_box",
        "d455_01",
        "d455_02",
    }
    assert seq_ids == expected_ids

    # Global transformation rate must be 100%
    assert metrics["global"]["global_world_transformation_rate"] == 1.0
    assert metrics["global"]["total_valid_3d_observations"] >= 1000
    assert metrics["global"]["mean_depth_confidence"] > 0.80
    assert metrics["global"]["mean_depth_mad_m"] < 0.15
