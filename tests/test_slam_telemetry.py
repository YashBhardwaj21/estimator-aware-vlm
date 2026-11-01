"""Unit tests for SLAM telemetry schemas and validation."""

import json
from pathlib import Path
import pytest

from estimator_aware_vlm.slam.telemetry import (
    FrameFeatureAssociationRecord,
    LandmarkSupportRecord,
    OptimizerObservationRecord,
    OptimizerSummary,
    TelemetryManifest,
    compute_file_sha256,
)


def test_frame_feature_association_record():
    rec = FrameFeatureAssociationRecord(
        frame_id=42,
        timestamp=1341847980.1234,
        keypoint_index=7,
        u=320.5,
        v=240.2,
        map_point_id=105,
        associated_object_id="person_01",
        semantic_class="person",
        is_in_dynamic_mask=True,
    )
    d = rec.to_dict()
    assert d["frame_id"] == 42
    assert d["keypoint_index"] == 7
    assert d["map_point_id"] == 105
    assert d["is_in_dynamic_mask"] is True


def test_landmark_support_record():
    lm = LandmarkSupportRecord(
        landmark_id=105,
        x=1.23,
        y=-0.45,
        z=2.67,
        total_observations=18,
        semantic_weight=0.10,
        dynamic_state="TEMPORARILY_EXCLUDED",
        associated_object_id="person_01",
        semantic_class="person",
        consecutive_inliers=0,
    )
    d = lm.to_dict()
    assert d["landmark_id"] == 105
    assert d["dynamic_state"] == "TEMPORARILY_EXCLUDED"
    assert d["semantic_weight"] == 0.10


def test_optimizer_observation_record():
    obs = OptimizerObservationRecord(
        optimization_type="POSE_OPTIMIZATION",
        frame_id=42,
        iteration=3,
        keypoint_index=7,
        map_point_id=105,
        residual_u=0.8,
        residual_v=-0.6,
        residual_norm=1.0,
        chi2_raw=3.2,
        semantic_weight=0.10,
        chi2_weighted=0.32,
        is_tracking_inlier=True,
        is_optimizer_inlier=True,
    )
    d = obs.to_dict()
    assert d["optimization_type"] == "POSE_OPTIMIZATION"
    assert d["chi2_weighted"] == 0.32
    assert d["is_optimizer_inlier"] is True


def test_telemetry_manifest_and_sha256(tmp_path: Path):
    test_file = tmp_path / "test.txt"
    test_file.write_text("telemetry content", encoding="utf-8")
    sha = compute_file_sha256(test_file)
    assert len(sha) == 64

    manifest = TelemetryManifest(
        sequence_id="tum_fr3_walking_xyz",
        run_id="run_20261101_native",
        integration_mode="I3_DUAL",
        files={"test.txt": {"sha256": sha, "bytes": test_file.stat().st_size}},
    )
    out_json = tmp_path / "telemetry_manifest.json"
    manifest.save_json(out_json)
    assert out_json.exists()
    data = json.loads(out_json.read_text(encoding="utf-8"))
    assert data["integration_mode"] == "I3_DUAL"
    assert data["files"]["test.txt"]["sha256"] == sha

