"""Unit tests for Auditable SLAM Evidence & Provenance Framework (Stage 4)."""

import pytest

from estimator_aware_vlm.common.provenance import (
    EvidenceProvenance,
    FieldProvenanceMeta,
    OBSERVATION_FIELD_PROVENANCE,
    TRACK_FIELD_PROVENANCE,
    get_observation_provenance_manifest,
    get_track_provenance_manifest,
)
from estimator_aware_vlm.perception.sequence_tracker import (
    MotionObservationState,
    ObservationRecord,
    TrackedEntity,
)
from estimator_aware_vlm.router_policy import compute_estimator_influence_with_provenance


def test_evidence_provenance_enum_values():
    assert EvidenceProvenance.MEASURED_BACKEND.value == "MEASURED_BACKEND"
    assert EvidenceProvenance.DERIVED.value == "DERIVED"
    assert EvidenceProvenance.PROXY.value == "PROXY"


def test_observation_provenance_manifest_completeness():
    manifest = get_observation_provenance_manifest()
    expected_fields = [
        "frame_index",
        "timestamp",
        "median_depth_m",
        "camera_position_world",
        "centroid_camera",
        "centroid_world",
        "confidence",
        "bbox_xyxy",
        "mask_contour",
        "semantic_class",
    ]
    for field_name in expected_fields:
        assert field_name in manifest
        meta = manifest[field_name]
        assert meta["provenance"] in [p.value for p in EvidenceProvenance]
        assert len(meta["source"]) > 0
        assert len(meta["description"]) > 0

    # Verify classification rigor
    assert manifest["timestamp"]["provenance"] == EvidenceProvenance.MEASURED_BACKEND.value
    assert manifest["centroid_world"]["provenance"] == EvidenceProvenance.DERIVED.value
    assert manifest["confidence"]["provenance"] == EvidenceProvenance.PROXY.value


def test_track_provenance_manifest_completeness():
    manifest = get_track_provenance_manifest()
    expected_fields = [
        "first_timestamp",
        "last_timestamp",
        "track_id",
        "observation_count",
        "lifespan_sec",
        "camera_ego_motion_displacement_m",
        "apparent_camera_frame_displacement_m",
        "world_displacement_m",
        "measured_displacement_m",
        "displacement_uncertainty_m",
        "observed_motion",
        "semantic_class",
        "inlier_ratio",
    ]
    for field_name in expected_fields:
        assert field_name in manifest
        meta = manifest[field_name]
        assert meta["provenance"] in [p.value for p in EvidenceProvenance]

    # Verify classification rigor
    assert manifest["first_timestamp"]["provenance"] == EvidenceProvenance.MEASURED_BACKEND.value
    assert manifest["measured_displacement_m"]["provenance"] == EvidenceProvenance.DERIVED.value
    assert manifest["observed_motion"]["provenance"] == EvidenceProvenance.PROXY.value
    assert manifest["inlier_ratio"]["provenance"] == EvidenceProvenance.PROXY.value


def test_observation_record_serialization():
    obs = ObservationRecord(
        frame_index=10,
        timestamp=1341846315.123,
        semantic_class="chair",
        confidence=0.88,
        bbox_xyxy=(100, 150, 200, 300),
        centroid_camera=(0.2, -0.1, 1.5),
        centroid_world=(1.2, 0.4, 1.5),
        area_px=15000,
        median_depth_m=1.52,
        sharpness=85.5,
        camera_position_world=(1.0, 0.5, 0.0),
        rgb_path="rgb/10.png",
        track_id="chair_01",
    )
    d = obs.to_dict()
    assert d["frame_index"] == 10
    assert d["timestamp"] == 1341846315.123
    assert "provenance" in d
    assert d["provenance"]["timestamp"]["provenance"] == "MEASURED_BACKEND"
    assert d["provenance"]["centroid_world"]["provenance"] == "DERIVED"


def test_tracked_entity_serialization_and_lifespan():
    obs1 = ObservationRecord(
        frame_index=0,
        timestamp=100.0,
        semantic_class="person",
        confidence=0.9,
        bbox_xyxy=(10, 10, 50, 100),
        centroid_camera=(0.0, 0.0, 2.0),
        centroid_world=(0.0, 0.0, 2.0),
        area_px=4000,
        median_depth_m=2.0,
        sharpness=70.0,
        camera_position_world=(0.0, 0.0, 0.0),
    )
    obs2 = ObservationRecord(
        frame_index=30,
        timestamp=101.5,
        semantic_class="person",
        confidence=0.92,
        bbox_xyxy=(20, 10, 60, 100),
        centroid_camera=(0.8, 0.0, 2.0),
        centroid_world=(0.8, 0.0, 2.0),
        area_px=4000,
        median_depth_m=2.0,
        sharpness=72.0,
        camera_position_world=(0.0, 0.0, 0.0),
    )
    entity = TrackedEntity(track_id="person_01", observations=[obs1, obs2])
    entity.compute_motion_statistics()

    assert entity.observation_count == 2
    assert entity.lifespan_sec == 1.5
    assert entity.measured_displacement_m == 0.8
    assert entity.observed_motion == MotionObservationState.MOVING

    d = entity.to_dict()
    assert d["track_id"] == "person_01"
    assert d["lifespan_sec"] == 1.5
    assert "provenance" in d
    assert d["provenance"]["measured_displacement_m"]["provenance"] == "DERIVED"
    assert d["provenance"]["observed_motion"]["provenance"] == "PROXY"


def test_estimator_influence_provenance_discrimination():
    # 1. Direct measured SLAM backend telemetry
    evidence_measured = {
        "has_valid_pose": True,
        "pose_valid": True,
        "depth_valid": True,
        "inlier_count": 15,
        "inlier_ratio": 0.75,
        "inlier_provenance": "MEASURED_BACKEND",
        "total_observations": 12,
    }
    score, prov = compute_estimator_influence_with_provenance(evidence_measured)
    assert score > 0.0
    assert prov == "MEASURED_BACKEND"

    # 2. Proxy inlier telemetry (bounding-box depth projection)
    evidence_proxy = {
        "has_valid_pose": True,
        "pose_valid": True,
        "depth_valid": True,
        "inlier_count": 15,
        "inlier_ratio": 0.75,
        "inlier_provenance": "PROXY",
        "total_observations": 12,
    }
    score, prov = compute_estimator_influence_with_provenance(evidence_proxy)
    assert score > 0.0
    assert prov == "PROXY_SUPPORT"

    # 3. Measured zero inliers
    evidence_zero = {
        "has_valid_pose": True,
        "pose_valid": True,
        "depth_valid": True,
        "inlier_count": 0,
        "inlier_provenance": "MEASURED_BACKEND",
    }
    score, prov = compute_estimator_influence_with_provenance(evidence_zero)
    assert score == 0.0
    assert prov == "MEASURED_BACKEND_ZERO_INLIERS"

