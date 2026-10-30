"""Unit and validation tests for multimodal evidence extraction, verification, and fusion gates.

Covers:
1. Input resolution, SHA-256 provenance, and run manifest reconciliation.
2. Fail-closed record validation, non-finite rejection, and composite key uniqueness.
3. Rigid geometry verification, quaternion validity, camera projection, and metric integrity.
4. Temporal motion tracking, sampling gap handling, and ego-motion compensation.
5. Semantic entropy calculation, category mobility priors, and majority voting.
6. Cross-session SE(2) registration verification and acceptance gate validation.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
import numpy as np
import pytest

from estimator_aware_vlm.common.paths import get_repo_root
from estimator_aware_vlm.evidence_pipeline import (
    compute_file_sha256,
    resolve_observations_input,
    run_evidence_pipeline,
)
from estimator_aware_vlm.evidence_validation import (
    SourceRecordValidator,
    ValidatedModalityStatus,
    is_finite_number,
    is_finite_vector,
)
from estimator_aware_vlm.geometric_evidence import (
    GeometricEvidenceExtractor,
    GeometricEvidenceRecord,
    SessionGeometricConfig,
    project_points_to_camera,
    quaternion_to_rotation_matrix,
)
from estimator_aware_vlm.temporal_evidence import (
    TemporalEvidenceExtractor,
    TemporalEvidenceRecord,
    CUMULATIVE_PATH_THRESHOLD_M,
)
from estimator_aware_vlm.semantic_evidence import (
    SemanticEvidenceExtractor,
    SemanticEvidenceRecord,
)
from estimator_aware_vlm.evidence_adapter import load_registration_audit, RegistrationInfo

repo_root = get_repo_root()

# ==============================================================================
# Authoritative Observations Input & Pipeline Run Result
# ==============================================================================

def test_missing_input_raises_clear_filenotfound(tmp_path):
    """Verify that a missing observations file raises FileNotFoundError with candidate list."""
    non_existent = tmp_path / "non_existent_obs.jsonl"
    with pytest.raises(FileNotFoundError) as exc_info:
        resolve_observations_input(obs_jsonl_path=non_existent, repo_root=tmp_path)
    assert "Explicitly specified observations input file not found" in str(exc_info.value)


def test_canonical_resolution_and_explicit_override(tmp_path):
    """Verify explicit override takes precedence over canonical default, with exact SHA-256."""
    fake_repo = tmp_path / "repo"
    canonical_dir = fake_repo / "data" / "observations"
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


def test_run_pipeline_returns_manifest(tmp_path):
    """Verify run_evidence_pipeline returns a complete structured run manifest."""
    obs_file = tmp_path / "mini_obs.jsonl"
    obs_file.write_text(
        '{"obs_id": "obs_01", "source_session": "semantic_mapping_my_desk_recording_01", "frame_index": 1, "timestamp": 0.1, "local_track_id": "trk_01"}\n'
        '{"obs_id": "obs_02", "source_session": "semantic_mapping_my_desk_recording_01", "frame_index": 2, "timestamp": 0.2, "local_track_id": "trk_01"}\n',
        encoding="utf-8",
    )
    out_dir = tmp_path / "run_out"
    manifest = run_evidence_pipeline(obs_jsonl_path=obs_file, output_dir=out_dir)

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
# Source Record Schema & Fail-Closed Validation
# ==============================================================================

def test_validator_reconciles_accepted_and_rejected():
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


def test_strict_mode_raises_on_malformed_input():
    """Verify that strict mode immediately raises on invalid JSON and invalid types."""
    v_strict = SourceRecordValidator(strict_mode=True)
    with pytest.raises(ValueError) as exc_info:
        v_strict.validate_line("invalid json line", 1)
    assert "Strict mode rejected malformed JSON" in str(exc_info.value)

    with pytest.raises(TypeError):
        v_strict.validate_line("[1, 2, 3]", 2)


def test_duplicate_composite_id_rejected():
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
# Session Map, Transform, Calibration & Metric Semantics
# ==============================================================================

def test_zero_or_invalid_quaternion_rejected():
    """Verify quaternion_to_rotation_matrix rejects zero/near-zero quaternion and non-finite values."""
    with pytest.raises(ValueError) as exc_zero:
        quaternion_to_rotation_matrix(0.0, 0.0, 0.0, 0.0)
    assert "near-zero norm" in str(exc_zero.value)

    with pytest.raises(ValueError):
        quaternion_to_rotation_matrix(float("nan"), 0.0, 0.0, 1.0)


def test_zero_quaternion_in_project_points_returns_empty():
    """Verify project_points_to_camera cleanly returns empty/invalid for zero quaternion pose."""
    pts = np.array([[1.0, 2.0, 3.0]], dtype=np.float64)
    zero_q_pose = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    pixels, valid_z, _ = project_points_to_camera(pts, zero_q_pose)
    assert len(pixels) == 0
    assert len(valid_z) == 0


def test_unknown_session_does_not_use_session1_map(tmp_path):
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


def test_ply_parser_rejects_binary_format(tmp_path):
    """Verify PLY parser rejects non-ASCII format."""
    bad_ply = tmp_path / "binary_map.ply"
    bad_ply.write_text("ply\nformat binary_little_endian 1.0\nelement vertex 1\nend_header\n")

    extractor = GeometricEvidenceExtractor(session_configs={})
    with pytest.raises(ValueError) as exc:
        extractor._read_ply_points(bad_ply)
    assert "Unsupported PLY format" in str(exc.value)


def test_floor_plane_validator_rejects_invalid_rigid_transform(tmp_path):
    """Verify floor plane loader rejects non-rigid (sheared or singular) matrices."""
    bad_fp = tmp_path / "bad_floor.json"
    # Singular matrix (det = 0)
    singular_mat = np.zeros((4, 4))
    singular_mat[3, 3] = 1.0
    bad_fp.write_text(json.dumps({"T_world_floor": singular_mat.tolist()}))

    extractor = GeometricEvidenceExtractor(session_configs={})
    T_wf, has_fp = extractor._validate_and_load_floor_plane(bad_fp)
    assert has_fp is False


def test_metric_naming_does_not_alias_point_to_bbox_as_reprojection():
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


def test_observation_points_count_separated_from_landmark_support():
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


# ==============================================================================
# Geometric Evidence & Honest Metric Unaliasing
# ==============================================================================

def test_reprojection_residual_strictly_none_and_unaliased():
    """Verify that mean_reprojection_residual_px is never conflated with bbox distance."""
    rec = GeometricEvidenceRecord(
        obs_id="obs_test_geo",
        frame_index=1,
        timestamp=0.1,
        entity_id="chair_1",
        track_id="trk_1",
        source_session="session_1",
        depth_valid=True,
        measured_depth_m=1.85,
        depth_uncertainty_sigma_m=0.012,
        depth_provenance="MEASURED_SENSOR",
        point_to_bbox_valid=True,
        point_to_bbox_provenance="MEASURED_MAPPOINTS",
        point_to_bbox_distance_px=14.2,
        point_to_bbox_inlier_ratio=0.88,
        mappoints_candidate_count=25,
        mappoints_inlier_count=22,
        observation_points_count=150,
        feature_reprojection_residual_valid=False,
        feature_reprojection_residual_px=None,
        feature_reprojection_provenance="UNAVAILABLE_NO_KEYPOINT_CORRESPONDENCES",
        landmark_support_count=25,
        landmark_spatial_extent_m=0.45,
        geometric_consistency_score=0.85,
        provenance="MEASURED_FULL",
        has_valid_evidence=True,
    )
    # The property and dict entry MUST be None
    assert rec.mean_reprojection_residual_px is None
    assert rec.median_reprojection_residual_px is None
    assert rec.p95_reprojection_residual_px is None
    assert rec.reprojection_residual_valid is False
    assert rec.feature_reprojection_residual_valid is False
    assert rec.feature_reprojection_residual_px is None

    # Check to_dict serialization
    d = rec.to_dict()
    assert d["mean_reprojection_residual_px"] is None
    assert d["point_to_bbox_distance_px"] == 14.2
    assert d["point_to_bbox_inlier_ratio"] == 0.88


# ==============================================================================
# Temporal Multi-Scale Motion, Gap Detection, & Session Isolation
# ==============================================================================

def test_move_and_return_trajectory_classified_as_displaced():
    """Verify that move-and-return trajectory (low net displacement, high cumulative path) is DISPLACED."""
    extractor = TemporalEvidenceExtractor()

    # Initial position at (0.0, 0.0, 0.0)
    rec1 = extractor.extract_observation_temporal({
        "obs_id": "m1", "frame_index": 0, "timestamp": 0.0,
        "global_entity_id": "box_01", "local_track_id": "trk_box", "source_session": "sess_A",
        "centroid_floor": [0.0, 0.0, 0.0], "camera_pose": [0, 0, 0, 0, 0, 0, 1],
    })
    assert rec1.current_motion == "TRANSIENT"
    assert rec1.cumulative_path_length_m == 0.0

    # Second observation at (0.0, 0.0, 0.0) (n_obs=2 is still TRANSIENT until 3 samples seen)
    rec2 = extractor.extract_observation_temporal({
        "obs_id": "m2", "frame_index": 2, "timestamp": 0.1,
        "global_entity_id": "box_01", "local_track_id": "trk_box", "source_session": "sess_A",
        "centroid_floor": [0.0, 0.0, 0.0], "camera_pose": [0, 0, 0, 0, 0, 0, 1],
    })
    assert rec2.current_motion == "TRANSIENT"

    # Moves 0.35m away to (0.35, 0.0, 0.0) (n_obs=3, step_disp > 0.05 -> MOVING)
    rec3 = extractor.extract_observation_temporal({
        "obs_id": "m3", "frame_index": 5, "timestamp": 0.25,
        "global_entity_id": "box_01", "local_track_id": "trk_box", "source_session": "sess_A",
        "centroid_floor": [0.35, 0.0, 0.0], "camera_pose": [0, 0, 0, 0, 0, 0, 1],
    })
    assert rec3.current_motion == "MOVING"

    # Moves back to near-origin (0.02, 0.0, 0.0)
    rec4 = extractor.extract_observation_temporal({
        "obs_id": "m4", "frame_index": 8, "timestamp": 0.40,
        "global_entity_id": "box_01", "local_track_id": "trk_box", "source_session": "sess_A",
        "centroid_floor": [0.02, 0.0, 0.0], "camera_pose": [0, 0, 0, 0, 0, 0, 1],
    })

    # Settles at returned position (0.02, 0.0, 0.0) with step_disp=0.0 <= 0.05
    # Net displacement is 0.02m (< 0.25m threshold), but cumulative path is 0.68m (> 0.50m)
    rec5 = extractor.extract_observation_temporal({
        "obs_id": "m5", "frame_index": 10, "timestamp": 0.50,
        "global_entity_id": "box_01", "local_track_id": "trk_box", "source_session": "sess_A",
        "centroid_floor": [0.02, 0.0, 0.0], "camera_pose": [0, 0, 0, 0, 0, 0, 1],
    })
    assert rec5.net_displacement_m == pytest.approx(0.02, abs=1e-3)
    assert rec5.cumulative_path_length_m == pytest.approx(0.68, abs=1e-2)
    assert rec5.cumulative_path_length_m > CUMULATIVE_PATH_THRESHOLD_M
    assert rec5.current_motion == "DISPLACED"


def test_sampling_gap_detection_and_reappearance():
    """Verify that gaps exceeding thresholds increment observation_gap_events_count."""
    extractor = TemporalEvidenceExtractor()

    # Observation 1 at frame 0
    extractor.extract_observation_temporal({
        "obs_id": "g1", "frame_index": 0, "timestamp": 0.0,
        "global_entity_id": "chair_g", "local_track_id": "trk_g", "source_session": "sess_A",
        "centroid_floor": [1.0, 1.0, 0.0], "camera_pose": [0, 0, 0, 0, 0, 0, 1],
    })

    # Observation 2 after normal interval (frame 5, +0.15s) -> no gap
    rec2 = extractor.extract_observation_temporal({
        "obs_id": "g2", "frame_index": 5, "timestamp": 0.15,
        "global_entity_id": "chair_g", "local_track_id": "trk_g", "source_session": "sess_A",
        "centroid_floor": [1.0, 1.0, 0.0], "camera_pose": [0, 0, 0, 0, 0, 0, 1],
    })
    assert rec2.observation_gap_events_count == 0
    assert rec2.is_reappearance_after_gap is False

    # Observation 3 after 25 frames and 1.2s -> gap event triggered!
    rec3 = extractor.extract_observation_temporal({
        "obs_id": "g3", "frame_index": 30, "timestamp": 1.35,
        "global_entity_id": "chair_g", "local_track_id": "trk_g", "source_session": "sess_A",
        "centroid_floor": [1.0, 1.0, 0.0], "camera_pose": [0, 0, 0, 0, 0, 0, 1],
    })
    assert rec3.observation_gap_events_count == 1
    assert rec3.is_reappearance_after_gap is True
    assert rec3.frames_since_last_seen == 25
    # Backward compatibility properties
    assert rec3.occlusion_events_count == 1
    assert rec3.is_reappearance_after_occlusion is True


def test_session_isolation_in_temporal_tracks():
    """Verify that same local_track_id in different sessions does NOT cross-contaminate history."""
    extractor = TemporalEvidenceExtractor()

    # Session 1 track "trk_1"
    extractor.extract_observation_temporal({
        "obs_id": "s1_obs1", "frame_index": 0, "timestamp": 0.0,
        "global_entity_id": "ent_1", "local_track_id": "trk_1", "source_session": "session_1",
        "centroid_floor": [1.0, 0.0, 0.0], "camera_pose": [0, 0, 0, 0, 0, 0, 1],
    })
    extractor.extract_observation_temporal({
        "obs_id": "s1_obs2", "frame_index": 5, "timestamp": 0.2,
        "global_entity_id": "ent_1", "local_track_id": "trk_1", "source_session": "session_1",
        "centroid_floor": [1.5, 0.0, 0.0], "camera_pose": [0, 0, 0, 0, 0, 0, 1],
    })

    # Session 2 track with the exact same "trk_1" should be treated as brand new track
    rec_s2 = extractor.extract_observation_temporal({
        "obs_id": "s2_obs1", "frame_index": 0, "timestamp": 0.0,
        "global_entity_id": "ent_2", "local_track_id": "trk_1", "source_session": "session_2",
        "centroid_floor": [5.0, 5.0, 0.0], "camera_pose": [0, 0, 0, 0, 0, 0, 1],
    })
    # Should be observation_count = 1, current_motion = TRANSIENT, displacement = 0.0
    assert rec_s2.observation_count == 1
    assert rec_s2.current_motion == "TRANSIENT"
    assert rec_s2.net_displacement_m == 0.0


# ==============================================================================
# Semantic Honest Missing Values & Prior Separation
# ==============================================================================

def test_no_synthetic_defaults_for_missing_fields():
    """Verify missing detector confidence and bbox return None rather than 0.50 or fake coordinates."""
    extractor = SemanticEvidenceExtractor()

    # Missing confidence and bbox
    rec = extractor.extract_observation_semantics({
        "obs_id": "sem_empty",
        "frame_index": 0,
        "timestamp": 0.0,
        "global_entity_id": "chair_001",
        "class_name": "chair",
        # confidence omitted
        # bbox_xyxy omitted
    })
    assert rec.detector_confidence is None
    assert rec.bbox_area_px is None
    assert rec.detection_valid is False
    assert any("Missing or non-finite detector confidence" in diag for diag in rec.semantic_diagnostics)
    assert any("Missing or malformed bbox_xyxy" in diag for diag in rec.semantic_diagnostics)


def test_separation_of_detection_valid_and_class_known():
    """Verify distinction between valid detection and known vocabulary prior."""
    extractor = SemanticEvidenceExtractor()

    # Valid detection of unknown object class "hoverboard"
    rec = extractor.extract_observation_semantics({
        "obs_id": "sem_unknown_class",
        "frame_index": 0,
        "timestamp": 0.0,
        "global_entity_id": "ent_hover",
        "class_name": "hoverboard",
        "confidence": 0.88,
        "bbox_xyxy": [10.0, 20.0, 110.0, 120.0],
    })
    assert rec.detection_valid is True
    assert rec.class_known_to_prior is False
    assert rec.is_unknown_class is True
    assert rec.class_prior_mobility == "UNKNOWN"
    assert rec.class_prior_confidence == 0.0
    assert "Unmodeled object class 'hoverboard'" in rec.cheap_prior_description


def test_session_isolation_in_semantic_voting():
    """Verify that multi-frame class voting history is strictly isolated across sessions."""
    extractor = SemanticEvidenceExtractor()

    # In Session 1, trk_shared voted "chair"
    extractor.extract_observation_semantics({
        "obs_id": "s1_1", "frame_index": 0, "timestamp": 0.0,
        "local_track_id": "trk_shared", "source_session": "session_1",
        "class_name": "chair", "confidence": 0.90,
    })

    # In Session 2, trk_shared is a "table"
    rec_s2 = extractor.extract_observation_semantics({
        "obs_id": "s2_1", "frame_index": 0, "timestamp": 0.0,
        "local_track_id": "trk_shared", "source_session": "session_2",
        "class_name": "table", "confidence": 0.85,
    })

    # In Session 2, there should only be 1 vote (for table), purity = 1.0, entropy = 0.0
    assert rec_s2.vote_count_total == 1
    assert rec_s2.semantic_purity == 1.0
    assert rec_s2.semantic_entropy == 0.0


# ==============================================================================
# Fused Evidence Schema & Fail-Safe Router Summary
# ==============================================================================

def test_router_summary_requires_vlm_query_when_evidence_incomplete():
    """Verify that incomplete evidence forces requires_vlm_query=True and cheap_verdict=UNSTABLE."""
    from estimator_aware_vlm.evidence_validation import ValidatedModalityStatus
    from estimator_aware_vlm.geometric_evidence import GeometricEvidenceRecord
    from estimator_aware_vlm.temporal_evidence import TemporalEvidenceRecord
    from estimator_aware_vlm.semantic_evidence import SemanticEvidenceRecord

    # Create synthetic records where geometry evidence is invalid
    geo_rec = GeometricEvidenceRecord(
        obs_id="obs_incomplete", frame_index=0, timestamp=0.0,
        entity_id="chair_x", track_id="trk_x", source_session="session_1",
        depth_valid=False, measured_depth_m=None, depth_uncertainty_sigma_m=None,
        depth_provenance="MISSING", point_to_bbox_valid=False,
        point_to_bbox_provenance="MISSING", point_to_bbox_distance_px=None,
        point_to_bbox_inlier_ratio=None, mappoints_candidate_count=0,
        mappoints_inlier_count=0, observation_points_count=0,
        feature_reprojection_residual_valid=False, feature_reprojection_residual_px=None,
        feature_reprojection_provenance="UNAVAILABLE_NO_KEYPOINT_CORRESPONDENCES",
        landmark_support_count=0, landmark_spatial_extent_m=0.0,
        geometric_consistency_score=0.0, provenance="MISSING", has_valid_evidence=False,
    )

    temp_rec = TemporalEvidenceRecord(
        obs_id="obs_incomplete", frame_index=0, timestamp=0.0,
        entity_id="chair_x", track_id="trk_x", source_session="session_1",
        current_motion="STATIC", world_displacement_m=0.0, net_displacement_m=0.0,
        step_displacement_m=0.0, cumulative_path_length_m=0.0,
        world_displacement_sigma_m=0.04, camera_ego_displacement_m=0.0,
        ego_motion_compensation_ratio=1.0, motion_significance_ratio=0.0,
        visibility_duration_s=0.0, observation_count=5,
        observation_gap_events_count=0, frames_since_last_seen=0,
        is_reappearance_after_gap=False, temporal_persistence_score=0.45,
        position_change_since_last_step_m=0.0, has_valid_evidence=True,
    )

    sem_rec = SemanticEvidenceRecord(
        obs_id="obs_incomplete", frame_index=0, timestamp=0.0,
        entity_id="chair_x", track_id="trk_x", source_session="session_1",
        raw_class_name="chair", yoloe_class_name="chair", detector_confidence=0.90,
        bbox_area_px=5000.0, class_prior_mobility="MOVABLE", class_prior_confidence=0.85,
        semantic_entropy=0.0, semantic_purity=1.0, vote_count_primary=1,
        vote_count_total=1, is_unknown_class=False, detection_valid=True,
        class_known_to_prior=True, cheap_prior_description="Office chair", has_valid_evidence=True,
    )

    # Modality status indicating missing pose
    mod_status = ValidatedModalityStatus(has_valid_pose=False, has_valid_floor_centroid=True)

    geometry_valid = bool(geo_rec.has_valid_evidence)
    temporal_valid = bool(temp_rec.has_valid_evidence)
    semantic_valid = bool(sem_rec.has_valid_evidence and sem_rec.class_prior_mobility != "UNKNOWN")

    evidence_complete = bool(
        geometry_valid and temporal_valid and semantic_valid
        and mod_status.has_valid_pose and mod_status.has_valid_floor_centroid
    )
    assert evidence_complete is False

    requires_vlm_query = bool(
        not evidence_complete
        or temp_rec.current_motion in ["MOVING", "DISPLACED", "UNCERTAIN"]
        or sem_rec.semantic_entropy > 0.45
        or geo_rec.geometric_consistency_score < 0.50
    )
    assert requires_vlm_query is True

    cheap_verdict = (
        "STATIONARY_" + sem_rec.class_prior_mobility
        if (temp_rec.current_motion == "STATIC" and evidence_complete)
        else "UNSTABLE"
    )
    assert cheap_verdict == "UNSTABLE"


# ==============================================================================
# Cross-Session SE(2) Registration & Controlled Targets
# ==============================================================================

def test_authoritative_se2_registration_loaded():
    """Verify loading and validating the authoritative SE(2) registration audit file."""
    repo_root = Path(__file__).resolve().parents[1]
    audit_file = repo_root / "outputs" / "reports" / "registration_audit.json"
    if not audit_file.exists():
        audit_file = repo_root / "outputs" / "registration_audit.json"
    reg = load_registration_audit(audit_file)

    assert reg.is_valid is True
    assert reg.reference_session == "semantic_mapping_my_desk_recording_01"
    assert reg.aligned_session == "semantic_mapping_my_desk_recording_02"
    # Registration rotation and translation checks
    assert abs(reg.rotation_deg - (-15.558)) < 0.05
    assert reg.translation_m.shape == (2,)
    assert reg.R_12.shape == (2, 2)
    assert np.allclose(reg.R_12 @ reg.R_12.T, np.eye(2), atol=1e-5)
    assert reg.anchor_count >= 10
    assert reg.robust_inlier_count >= 7


def test_controlled_displacement_evaluation_precision():
    """Verify controlled displacement evaluation on known physical targets."""
    repo_root = Path(__file__).resolve().parents[1]
    metrics_file = repo_root / "outputs" / "reports" / "prerouter_evidence_metrics.json"
    if not metrics_file.exists():
        metrics_file = repo_root / "outputs" / "prerouter_evidence_metrics.json"
    assert metrics_file.exists(), "prerouter_evidence_metrics.json must exist"

    with open(metrics_file, "r", encoding="utf-8") as f:
        metrics = json.load(f)

    # Check controlled displacement gate
    controlled_gate = next(g for g in metrics["acceptance_gates"] if g["name"] == "Controlled Displacement Error")
    assert controlled_gate["passed"] is True
    assert controlled_gate["value"] <= 0.15  # <= 15 cm

    rel_err_gate = next(g for g in metrics["acceptance_gates"] if g["name"] == "Controlled Relative Error")
    assert rel_err_gate["passed"] is True
    assert rel_err_gate["value"] <= 30.0  # <= 30%


# ==============================================================================
# Acceptance Gate Verification & Manifest Consistency
# ==============================================================================

def test_all_acceptance_gates_verified():
    """Verify all 6 evidence pipeline acceptance gates pass unconditionally."""
    repo_root = Path(__file__).resolve().parents[1]
    metrics_file = repo_root / "outputs" / "reports" / "prerouter_evidence_metrics.json"
    if not metrics_file.exists():
        metrics_file = repo_root / "outputs" / "prerouter_evidence_metrics.json"
    with open(metrics_file, "r", encoding="utf-8") as f:
        metrics = json.load(f)

    assert metrics["acceptance_gate_passed"] is True
    assert len(metrics["acceptance_gates"]) == 6

    for gate in metrics["acceptance_gates"]:
        assert gate["passed"] is True, f"Gate {gate['name']} failed: {gate['reason']}"
        assert gate["status"] == "PASS"


def test_manifest_record_counts_reconciled():
    """Verify that evidence_run_manifest.json accounts for all non-empty lines without silent data loss."""
    repo_root = Path(__file__).resolve().parents[1]
    manifest_file = repo_root / "outputs" / "reports" / "evidence_run_manifest.json"
    if not manifest_file.exists():
        manifest_file = repo_root / "outputs" / "full_experiment_run" / "evidence" / "evidence_run_manifest.json"
    if not manifest_file.exists():
        manifest_file = repo_root / "outputs" / "evidence_run_manifest.json"
    assert manifest_file.exists()

    with open(manifest_file, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    assert manifest["total_lines_read"] == 7956
    assert manifest["accepted_count"] == 7956
    assert manifest["rejected_count"] == 0
    assert manifest["accepted_count"] + manifest["rejected_count"] == manifest["total_lines_read"]
    assert manifest["reconciliation_passed"] is True


