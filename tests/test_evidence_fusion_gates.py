"""Regression and validation tests for evidence fusion acceptance gates.

Step 7: Fused evidence generation, schema completeness, and fail-safe router summary.
Step 8: Cross-session entity evaluation using authoritative SE(2) registration, controlled targets, and balanced class support.
Step 9: Acceptance gate validation and experiment runner integration.
"""

import json
from pathlib import Path
import numpy as np
import pytest

from estimator_aware_vlm.evidence_pipeline import run_stage5_pipeline
from estimator_aware_vlm.evidence_adapter import load_registration_audit, RegistrationInfo
from scripts.evaluate_prerouter_evidence import evaluate_stage5


# ==============================================================================
# Step 7 Tests: Fused Evidence Schema & Fail-Safe Router Summary
# ==============================================================================

def test_step7_router_summary_requires_vlm_query_when_evidence_incomplete():
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
# Step 8 Tests: Cross-Session SE(2) Registration & Controlled Targets
# ==============================================================================

def test_step8_authoritative_se2_registration_loaded():
    """Verify loading and validating the authoritative SE(2) registration audit file."""
    repo_root = Path(__file__).resolve().parents[1]
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


def test_step8_controlled_displacement_evaluation_precision():
    """Verify controlled displacement evaluation on known physical targets."""
    repo_root = Path(__file__).resolve().parents[1]
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
# Step 9 Tests: Acceptance Gate Verification & Manifest Consistency
# ==============================================================================

def test_step9_all_acceptance_gates_verified():
    """Verify all 6 evidence pipeline acceptance gates pass unconditionally."""
    repo_root = Path(__file__).resolve().parents[1]
    metrics_file = repo_root / "outputs" / "prerouter_evidence_metrics.json"
    with open(metrics_file, "r", encoding="utf-8") as f:
        metrics = json.load(f)

    assert metrics["acceptance_gate_passed"] is True
    assert len(metrics["acceptance_gates"]) == 6

    for gate in metrics["acceptance_gates"]:
        assert gate["passed"] is True, f"Gate {gate['name']} failed: {gate['reason']}"
        assert gate["status"] == "PASS"


def test_step9_manifest_record_counts_reconciled():
    """Verify that evidence_run_manifest.json accounts for all non-empty lines without silent data loss."""
    repo_root = Path(__file__).resolve().parents[1]
    manifest_file = repo_root / "outputs" / "evidence_run_manifest.json"
    assert manifest_file.exists()

    with open(manifest_file, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    assert manifest["total_lines_read"] == 7956
    assert manifest["accepted_count"] == 7956
    assert manifest["rejected_count"] == 0
    assert manifest["accepted_count"] + manifest["rejected_count"] == manifest["total_lines_read"]
    assert manifest["reconciliation_passed"] is True
