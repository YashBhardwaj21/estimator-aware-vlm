"""Unit tests for Pre-Router Multimodal Evidence Pipeline."""

import pytest
import math
import sys
from pathlib import Path
import numpy as np

# Add project root and vlm to sys.path
root_dir = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root_dir))
sys.path.insert(0, str(root_dir / "vlm"))

from estimator_aware_vlm.geometric_evidence import (
    GeometricEvidenceExtractor,
    compute_depth_uncertainty,
    quaternion_to_rotation_matrix,
    invert_se3_transform,
    project_points_to_camera,
    compute_point_to_bbox_distances,
    FX, FY, CX, CY, STEREO_BASELINE_M
)
from estimator_aware_vlm.temporal_evidence import TemporalEvidenceExtractor
from estimator_aware_vlm.semantic_evidence import SemanticEvidenceExtractor, CLASS_MOBILITY_PRIORS


def test_geometric_depth_uncertainty_model():
    """Verify physical RealSense D455 stereo triangulation uncertainty formula.
    sigma_z = (z^2 / (f_x * B)) * sigma_d
    """
    sigma_1m = compute_depth_uncertainty(1.0)
    sigma_2m = compute_depth_uncertainty(2.0)
    sigma_3m = compute_depth_uncertainty(3.0)

    # Monotonic quadratic growth with depth
    assert sigma_1m < sigma_2m < sigma_3m
    assert 0.001 <= sigma_1m <= 0.005
    assert 0.010 <= sigma_3m <= 0.035


def test_geometric_quaternion_orthonormality():
    """Verify quaternion to rotation matrix produces valid orthonormal SO(3) matrix."""
    R_eye = quaternion_to_rotation_matrix(0, 0, 0, 1)
    assert R_eye.shape == (3, 3)
    assert math.isclose(R_eye[0, 0], 1.0, abs_tol=1e-5)
    assert math.isclose(R_eye[1, 1], 1.0, abs_tol=1e-5)
    assert math.isclose(R_eye[2, 2], 1.0, abs_tol=1e-5)

    R = quaternion_to_rotation_matrix(0.1, -0.2, 0.3, 0.9)
    I_test = R @ R.T
    for i in range(3):
        for j in range(3):
            expected = 1.0 if i == j else 0.0
            assert math.isclose(I_test[i, j], expected, abs_tol=1e-4)


def test_se3_transform_inversion_roundtrip():
    """Verify SE(3) transformation inversion round-trip: T_cw * T_wc = I."""
    R_wc = quaternion_to_rotation_matrix(0.1, 0.2, -0.3, 0.9)
    t_wc = np.array([1.5, -0.8, 0.4], dtype=np.float64)

    R_cw, t_cw = invert_se3_transform(R_wc, t_wc)

    # Test round-trip on arbitrary 3D test point
    p_orig = np.array([2.5, 1.2, 0.9], dtype=np.float64)
    # Forward: world to camera
    p_cam = R_cw @ p_orig + t_cw
    # Inverse: camera back to world
    p_recov = R_wc @ p_cam + t_wc

    np.testing.assert_allclose(p_recov, p_orig, atol=1e-6)


def test_known_3d_to_2d_camera_projection():
    """Verify known 3D-to-2D pinhole projection with known pixel coordinates."""
    # Camera at origin looking along +Z
    cam_pose = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]

    # Point directly on optical axis at Z = 2.0m -> must project to principal point (CX, CY)
    p_center = np.array([[0.0, 0.0, 2.0]], dtype=np.float64)
    pixels, valid_z, _ = project_points_to_camera(p_center, cam_pose)
    assert valid_z[0] is True or valid_z[0] == 1
    assert math.isclose(pixels[0, 0], CX, abs_tol=1e-4)
    assert math.isclose(pixels[0, 1], CY, abs_tol=1e-4)

    # Point at X = 0.5m, Y = 0.2m, Z = 2.0m
    # u = FX * (0.5 / 2.0) + CX = 383.924896 * 0.25 + 320.123016 = 416.10424
    # v = FY * (0.2 / 2.0) + CY = 383.924896 * 0.10 + 244.569885 = 282.96237
    p_offset = np.array([[0.5, 0.2, 2.0]], dtype=np.float64)
    pix_off, val_off, _ = project_points_to_camera(p_offset, cam_pose)
    assert val_off[0] is True or val_off[0] == 1
    expected_u = FX * (0.5 / 2.0) + CX
    expected_v = FY * (0.2 / 2.0) + CY
    assert math.isclose(pix_off[0, 0], expected_u, abs_tol=1e-2)
    assert math.isclose(pix_off[0, 1], expected_v, abs_tol=1e-2)


def test_points_behind_camera_filtered():
    """Verify that points behind camera (Z <= 0.15m) are filtered out."""
    cam_pose = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]
    pts = np.array([
        [0.0, 0.0, -1.0],  # Behind camera
        [0.0, 0.0, 0.05],  # Near plane violation (<0.15m)
        [0.0, 0.0, 2.0],   # Valid
    ], dtype=np.float64)

    _, valid_z, _ = project_points_to_camera(pts, cam_pose)
    assert bool(valid_z[0]) is False
    assert bool(valid_z[1]) is False
    assert bool(valid_z[2]) is True


def test_point_to_bbox_distances_inside_vs_outside():
    """Verify points inside box have 0 distance; points outside have Euclidean distance."""
    bbox = [100.0, 100.0, 200.0, 200.0]  # [bx1, by1, bx2, by2]
    pixels = np.array([
        [150.0, 150.0],  # Inside -> distance 0 px
        [210.0, 150.0],  # 10 px to the right -> distance 10 px
        [150.0, 80.0],   # 20 px above -> distance 20 px
        [203.0, 204.0],  # 3 px right, 4 px down -> distance 5 px (sqrt(9+16))
    ], dtype=np.float64)
    valid_z = np.ones(4, dtype=bool)

    dists, inlier_mask = compute_point_to_bbox_distances(pixels, bbox, valid_z, margin_px=8.0)
    assert math.isclose(dists[0], 0.0, abs_tol=1e-4)
    assert math.isclose(dists[1], 10.0, abs_tol=1e-4)
    assert math.isclose(dists[2], 20.0, abs_tol=1e-4)
    assert math.isclose(dists[3], 5.0, abs_tol=1e-4)

    # Inliers within margin (margin = 8px)
    assert bool(inlier_mask[0]) is True   # inside
    assert bool(inlier_mask[1]) is False  # 10px > 8px
    assert bool(inlier_mask[2]) is False  # 20px > 8px
    assert bool(inlier_mask[3]) is True   # 3px, 4px <= 8px


def test_geometric_missing_pose_or_floor_transform():
    """Verify that missing pose or floor transform sets explicit missing provenance."""
    extractor = GeometricEvidenceExtractor()

    # Case 1: Valid session but missing camera pose
    sample_obs_s1 = {
        "obs_id": "test_missing_pose",
        "frame_index": 10,
        "timestamp": 1.0,
        "source_session": "semantic_mapping_my_desk_recording_01",
        "global_entity_id": "desk_001",
        "local_track_id": "trk_001",
        "centroid_camera": [0.0, 0.0, 2.0],
        "bbox_min_floor": [1.0, -1.0, 0.0],
        "bbox_max_floor": [2.0, -0.5, 0.8],
        "camera_pose": None,  # Missing pose
        "bbox_xyxy": [100.0, 100.0, 200.0, 200.0],
    }

    rec1 = extractor.extract_observation_geometry(sample_obs_s1)
    assert rec1.depth_valid is True
    assert rec1.point_to_bbox_valid is False
    assert rec1.point_to_bbox_provenance in ["MISSING_FLOOR_TRANSFORM", "MISSING_CAMERA_POSE"]
    assert rec1.feature_reprojection_residual_valid is False
    assert rec1.feature_reprojection_provenance == "UNAVAILABLE_NO_KEYPOINT_CORRESPONDENCES"

    # Case 2: Unknown session must produce MISSING_SESSION_CONFIG (zero cross-session fallback)
    sample_obs_unknown = dict(sample_obs_s1)
    sample_obs_unknown["source_session"] = "unknown_session_xyz"
    rec2 = extractor.extract_observation_geometry(sample_obs_unknown)
    assert rec2.point_to_bbox_provenance == "MISSING_SESSION_CONFIG"
    assert rec2.provenance == "MISSING_SESSION_CONFIG"


def test_temporal_missing_centroid_honest_handling():
    """Verify that missing centroid does NOT get replaced with [0, 0, 0] and sets UNCERTAIN."""
    extractor = TemporalEvidenceExtractor()
    obs_missing = {
        "obs_id": "t_miss",
        "frame_index": 5,
        "timestamp": 0.5,
        "global_entity_id": "desk_miss",
        "local_track_id": "trk_miss",
        "centroid_floor": None,  # Missing!
        "camera_pose": [0, 0, 0, 0, 0, 0, 1],
    }
    rec = extractor.extract_observation_temporal(obs_missing)
    assert rec.has_valid_evidence is False
    assert rec.current_motion == "UNCERTAIN"
    assert rec.missing_reason == "MISSING_CENTROID_FLOOR"


def test_temporal_ego_motion_compensation():
    """Verify that moving the camera while object stays still yields STATIC motion."""
    extractor = TemporalEvidenceExtractor()

    obs1 = {
        "obs_id": "step_01",
        "frame_index": 0,
        "timestamp": 0.0,
        "global_entity_id": "chair_001",
        "local_track_id": "trk_c1",
        "centroid_floor": [2.0, -1.0, 0.4],
        "camera_pose": [0.0, 0.0, 0.5, 0, 0, 0, 1],
    }
    rec1 = extractor.extract_observation_temporal(obs1)
    assert rec1.current_motion == "TRANSIENT"
    assert rec1.world_displacement_m == 0.0

    obs2 = {
        "obs_id": "step_02",
        "frame_index": 10,
        "timestamp": 0.5,
        "global_entity_id": "chair_001",
        "local_track_id": "trk_c1",
        "centroid_floor": [2.01, -1.0, 0.4],
        "camera_pose": [1.5, 0.2, 0.5, 0, 0, 0, 1],
    }
    extractor.extract_observation_temporal(obs2)

    obs3 = {
        "obs_id": "step_03",
        "frame_index": 20,
        "timestamp": 1.0,
        "global_entity_id": "chair_001",
        "local_track_id": "trk_c1",
        "centroid_floor": [2.015, -0.995, 0.4],
        "camera_pose": [2.5, 0.4, 0.5, 0, 0, 0, 1],
    }
    rec3 = extractor.extract_observation_temporal(obs3)

    assert rec3.current_motion == "STATIC"
    assert rec3.world_displacement_m < 0.05
    assert rec3.camera_ego_displacement_m > 2.0
    assert rec3.ego_motion_compensation_ratio > 10.0


def test_temporal_displaced_object_detection():
    """Verify that true physical relocation (>0.15m) triggers MOVING/DISPLACED."""
    extractor = TemporalEvidenceExtractor()

    extractor.extract_observation_temporal({
        "obs_id": "t1", "frame_index": 0, "timestamp": 0.0,
        "global_entity_id": "chair_reloc", "local_track_id": "trk_r",
        "centroid_floor": [1.0, 1.0, 0.4], "camera_pose": [0, 0, 0, 0, 0, 0, 1],
    })
    extractor.extract_observation_temporal({
        "obs_id": "t2", "frame_index": 5, "timestamp": 0.25,
        "global_entity_id": "chair_reloc", "local_track_id": "trk_r",
        "centroid_floor": [1.0, 1.0, 0.4], "camera_pose": [0, 0, 0, 0, 0, 0, 1],
    })
    rec3 = extractor.extract_observation_temporal({
        "obs_id": "t3", "frame_index": 30, "timestamp": 1.5,
        "global_entity_id": "chair_reloc", "local_track_id": "trk_r",
        "centroid_floor": [1.56, 1.0, 0.4], "camera_pose": [0, 0, 0, 0, 0, 0, 1],
    })

    assert rec3.current_motion in ["MOVING", "DISPLACED"]
    assert math.isclose(rec3.world_displacement_m, 0.56, abs_tol=1e-2)
    assert rec3.motion_significance_ratio > 3.0


def test_semantic_entropy_and_cheap_prior():
    """Verify semantic Shannon entropy and deterministic M0 priors."""
    extractor = SemanticEvidenceExtractor()

    rec1 = extractor.extract_observation_semantics({
        "obs_id": "s1", "frame_index": 0, "timestamp": 0.0,
        "global_entity_id": "chair_001", "class_name": "chair", "confidence": 0.90,
    })
    assert rec1.class_prior_mobility == "MOVABLE"
    assert rec1.semantic_entropy == 0.0
    assert rec1.semantic_purity == 1.0

    rec2 = extractor.extract_observation_semantics({
        "obs_id": "s2", "frame_index": 1, "timestamp": 0.1,
        "global_entity_id": "chair_001", "class_name": "table", "confidence": 0.40,
    })
    assert rec2.semantic_entropy > 0.0
    assert rec2.semantic_purity == 0.50

    door_rec = extractor.extract_observation_semantics({
        "obs_id": "s3", "frame_index": 2, "timestamp": 0.2,
        "global_entity_id": "door_001", "class_name": "door", "confidence": 0.95,
    })
    assert door_rec.class_prior_mobility == "IMMOVABLE"
    assert door_rec.class_prior_confidence >= 0.90
