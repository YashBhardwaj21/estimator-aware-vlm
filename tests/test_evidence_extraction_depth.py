"""Regression and acceptance tests for depth and geometric evidence extraction:
- Step 1: Authoritative input resolution, structured run result, and evaluator alignment.
- Step 2: Fail-closed source schema validation, reconciliation, and duplicate guard.
- Step 3: Explicit session mapping, rigid transform validation, strict PLY parsing, and honest metric naming.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
import numpy as np
import pytest

from estimator_aware_vlm.evidence_pipeline import (
    compute_file_sha256,
    resolve_observations_input,
    run_stage5_pipeline,
)
from estimator_aware_vlm.evidence_validation import SourceRecordValidator, is_finite_number, is_finite_vector
from estimator_aware_vlm.geometric_evidence import (
    GeometricEvidenceExtractor,
    SessionGeometricConfig,
    quaternion_to_rotation_matrix,
    project_points_to_camera,
)


# ==============================================================================
# Step 1 Tests: Authoritative Observations Input & Pipeline Run Result
# ==============================================================================

def test_step1_missing_input_raises_clear_filenotfound(tmp_path):
    """Verify that a missing observations file raises FileNotFoundError with candidate list."""
    non_existent = tmp_path / "non_existent_obs.jsonl"
    with pytest.raises(FileNotFoundError) as exc_info:
        resolve_observations_input(obs_jsonl_path=non_existent, repo_root=tmp_path)
    assert "Explicitly specified observations input file not found" in str(exc_info.value)


def test_step1_canonical_resolution_and_explicit_override(tmp_path):
    """Verify explicit override takes precedence over canonical default, with exact SHA-256."""
    fake_repo = tmp_path / "repo"
    canonical_dir = fake_repo / "vlm" / "data" / "observations"
    canonical_dir.mkdir(parents=True, exist_ok=True)
    canonical_file = canonical_dir / "object_observations.jsonl"
    canonical_file.write_text('{"obs_id": "c1", "source_session": "s1", "frame_index": 0, "timestamp": 0.0}\n', encoding="utf-8")

    custom_file = tmp_path / "custom_obs.jsonl"
    custom_file.write_text('{"obs_id": "cust1", "source_session": "s1", "frame_index": 0, "timestamp": 0.0}\n', encoding="utf-8")

    # 1. Without override -> selects canonical
    path1, sha1, reason1 = resolve_observations_input(None, fake_repo)
    assert path1 == canonical_file
    assert reason1 == "CANONICAL_DEFAULT"
    assert sha1 == compute_file_sha256(canonical_file)

    # 2. With explicit override -> selects custom
    path2, sha2, reason2 = resolve_observations_input(custom_file, fake_repo)
    assert path2 == custom_file.resolve()
    assert reason2 == "EXPLICIT_USER_OVERRIDE"
    assert sha2 == compute_file_sha256(custom_file)
    assert sha1 != sha2


def test_step1_run_pipeline_returns_manifest(tmp_path):
    """Verify run_stage5_pipeline returns a complete structured run manifest."""
    obs_file = tmp_path / "mini_obs.jsonl"
    obs_file.write_text(
        '{"obs_id": "obs_01", "source_session": "semantic_mapping_my_desk_recording_01", "frame_index": 1, "timestamp": 0.1, "local_track_id": "trk_01"}\n'
        '{"obs_id": "obs_02", "source_session": "semantic_mapping_my_desk_recording_01", "frame_index": 2, "timestamp": 0.2, "local_track_id": "trk_01"}\n',
        encoding="utf-8",
    )
    out_dir = tmp_path / "run_out"
    manifest = run_stage5_pipeline(obs_jsonl_path=obs_file, output_dir=out_dir)

    assert manifest["reconciliation_passed"] is True
    assert manifest["total_non_empty_lines"] == 2
    assert manifest["accepted_count"] == 2
    assert manifest["rejected_count"] == 0
    assert Path(manifest["fused_records_file"]).exists()
    assert Path(manifest["geometric_records_file"]).exists()
    assert Path(manifest["temporal_records_file"]).exists()
    assert Path(manifest["semantic_records_file"]).exists()
    assert manifest["input_obs_sha256"] == compute_file_sha256(obs_file)


# ==============================================================================
# Step 2 Tests: Source Record Schema & Fail-Closed Validation
# ==============================================================================

def test_step2_validator_reconciles_accepted_and_rejected():
    """Verify non_empty_lines == accepted + rejected accounting with permissive mode."""
    validator = SourceRecordValidator(strict_mode=False)

    lines = [
        "",  # empty -> ignored
        "   ",  # whitespace -> ignored
        '{"obs_id": "v1", "source_session": "s1", "frame_index": 0, "timestamp": 0.0}',  # valid
        "not json",  # malformed JSON -> rejected
        '["a", "list", "not", "dict"]',  # non-object -> rejected
        '{"source_session": "s1", "frame_index": 1, "timestamp": 0.1}',  # missing obs_id -> rejected
        '{"obs_id": "v1", "source_session": "s1", "frame_index": 2, "timestamp": 0.2}',  # duplicate (s1, v1) -> rejected
        '{"obs_id": "v2", "source_session": "s1", "frame_index": -1, "timestamp": 0.2}',  # negative frame -> rejected
        '{"obs_id": "v3", "source_session": "s1", "frame_index": 3, "timestamp": -0.5}',  # negative timestamp -> rejected
        '{"obs_id": "v4", "source_session": "s2", "frame_index": 0, "timestamp": 0.0}',  # valid s2
    ]

    for line_no, l in enumerate(lines, start=1):
        validator.validate_line(l, line_no)

    summary = validator.reconcile_summary()
    assert summary["total_lines_read"] == len(lines)
    assert summary["total_non_empty_lines"] == 8
    assert summary["total_accepted"] == 2
    assert summary["total_rejected"] == 6
    assert summary["reconciliation_passed"] is True
    assert summary["unique_composite_ids_count"] == 2
    assert len(summary["rejections"]) == 6


def test_step2_strict_mode_raises_on_malformed_input():
    """Verify that strict mode immediately raises on invalid JSON and invalid types."""
    v_strict = SourceRecordValidator(strict_mode=True)
    with pytest.raises(ValueError) as exc_info:
        v_strict.validate_line("invalid json line", 1)
    assert "Strict mode rejected malformed JSON" in str(exc_info.value)

    with pytest.raises(TypeError):
        v_strict.validate_line("[1, 2, 3]", 2)


def test_step2_duplicate_composite_id_rejected():
    """Verify that duplicate (session, obs_id) pairs are rejected while distinct sessions with same obs_id succeed."""
    validator = SourceRecordValidator(strict_mode=False)
    # Session 1 obs_001
    rec1, _, _ = validator.validate_line('{"obs_id": "obs_001", "source_session": "sess1", "frame_index": 0, "timestamp": 0.0}', 1)
    assert rec1 is not None

    # Session 2 obs_001 -> DISTINCT composite key, must SUCCEED
    rec2, _, _ = validator.validate_line('{"obs_id": "obs_001", "source_session": "sess2", "frame_index": 0, "timestamp": 0.0}', 2)
    assert rec2 is not None

    # Session 1 obs_001 AGAIN -> DUPLICATE, must REJECT
    rec3, _, rej = validator.validate_line('{"obs_id": "obs_001", "source_session": "sess1", "frame_index": 1, "timestamp": 0.1}', 3)
    assert rec3 is None
    assert rej.reason == "DUPLICATE_COMPOSITE_ID"


# ==============================================================================
# Step 3 Tests: Session Map, Transform, Calibration & Metric Semantics
# ==============================================================================

def test_step3_zero_or_invalid_quaternion_rejected():
    """Verify quaternion_to_rotation_matrix rejects zero/near-zero quaternion and non-finite values."""
    with pytest.raises(ValueError) as exc_zero:
        quaternion_to_rotation_matrix(0.0, 0.0, 0.0, 0.0)
    assert "near-zero norm" in str(exc_zero.value)

    with pytest.raises(ValueError):
        quaternion_to_rotation_matrix(float("nan"), 0.0, 0.0, 1.0)


def test_step3_zero_quaternion_in_project_points_returns_empty():
    """Verify project_points_to_camera cleanly returns empty/invalid for zero quaternion pose."""
    pts = np.array([[1.0, 2.0, 3.0]], dtype=np.float64)
    zero_q_pose = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    pixels, valid_z, _ = project_points_to_camera(pts, zero_q_pose)
    assert len(pixels) == 0
    assert len(valid_z) == 0


def test_step3_unknown_session_does_not_use_session1_map(tmp_path):
    """Verify that an observation with an unknown session produces MISSING_SESSION_CONFIG."""
    # Create valid Session 1 map
    ply_s1 = tmp_path / "s1_MapPoints.ply"
    ply_s1.write_text("ply\nformat ascii 1.0\nelement vertex 1\nproperty float x\nproperty float y\nproperty float z\nend_header\n1.0 2.0 3.0\n")

    fp_s1 = tmp_path / "s1_floor_plane.json"
    fp_s1.write_text(json.dumps({"T_world_floor": np.eye(4).tolist()}))

    cfg_s1 = SessionGeometricConfig(
        session_id="session_01",
        mappoints_ply_path=ply_s1,
        floor_plane_path=fp_s1,
    )

    extractor = GeometricEvidenceExtractor(session_configs={"session_01": cfg_s1})

    # Observation from Session 1
    obs_s1 = {
        "obs_id": "o1",
        "source_session": "session_01",
        "frame_index": 1,
        "timestamp": 0.1,
        "centroid_camera": [0.0, 0.0, 2.0],
        "camera_pose": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
    }
    rec_s1 = extractor.extract_observation_geometry(obs_s1)
    assert rec_s1.provenance != "MISSING_SESSION_CONFIG"

    # Observation from UNKNOWN session -> MUST NOT use session_01 map
    obs_unknown = dict(obs_s1)
    obs_unknown["source_session"] = "unknown_session_rogue"
    rec_unk = extractor.extract_observation_geometry(obs_unknown)
    assert rec_unk.provenance == "MISSING_SESSION_CONFIG"
    assert rec_unk.point_to_bbox_provenance == "MISSING_SESSION_CONFIG"
    assert rec_unk.point_to_bbox_valid is False


def test_step3_ply_parser_rejects_binary_format(tmp_path):
    """Verify PLY parser rejects non-ASCII format."""
    bad_ply = tmp_path / "binary_map.ply"
    bad_ply.write_text("ply\nformat binary_little_endian 1.0\nelement vertex 1\nend_header\n")

    extractor = GeometricEvidenceExtractor(session_configs={})
    with pytest.raises(ValueError) as exc:
        extractor._read_ply_points(bad_ply)
    assert "Unsupported PLY format" in str(exc.value)


def test_step3_floor_plane_validator_rejects_invalid_rigid_transform(tmp_path):
    """Verify floor plane loader rejects non-rigid (sheared or singular) matrices."""
    bad_fp = tmp_path / "bad_floor.json"
    # Singular matrix (det = 0)
    singular_mat = np.zeros((4, 4))
    singular_mat[3, 3] = 1.0
    bad_fp.write_text(json.dumps({"T_world_floor": singular_mat.tolist()}))

    extractor = GeometricEvidenceExtractor(session_configs={})
    T_wf, has_fp = extractor._validate_and_load_floor_plane(bad_fp)
    assert has_fp is False


def test_step3_metric_naming_does_not_alias_point_to_bbox_as_reprojection():
    """Verify that mean_reprojection_residual_px and reprojection_residual_valid are NOT aliased to bbox distance."""
    extractor = GeometricEvidenceExtractor(session_configs={})
    obs = {
        "obs_id": "test_naming",
        "source_session": "s1",
        "frame_index": 1,
        "timestamp": 0.1,
        "centroid_camera": [0.0, 0.0, 2.0],
    }
    rec = extractor.extract_observation_geometry(obs)
    d = rec.to_dict()

    # Feature reprojection must be honestly unavailable
    assert rec.feature_reprojection_residual_valid is False
    assert rec.feature_reprojection_residual_px is None
    assert rec.feature_reprojection_provenance == "UNAVAILABLE_NO_KEYPOINT_CORRESPONDENCES"

    # Compatibility properties must return None / False, never bbox distance
    assert rec.mean_reprojection_residual_px is None
    assert rec.reprojection_residual_valid is False
    assert d["mean_reprojection_residual_px"] is None
    assert d["reprojection_residual_valid"] is False


def test_step3_observation_points_count_separated_from_landmark_support():
    """Verify that observation_points_count and mappoints_candidate_count remain distinct."""
    extractor = GeometricEvidenceExtractor(session_configs={})
    obs = {
        "obs_id": "test_counts",
        "source_session": "s1",
        "frame_index": 1,
        "timestamp": 0.1,
        "points_count": 1420,  # Detector point count
        "centroid_camera": [0.0, 0.0, 2.0],
    }
    rec = extractor.extract_observation_geometry(obs)
    assert rec.observation_points_count == 1420
    assert rec.mappoints_candidate_count == 0  # No SLAM mappoints in empty session
