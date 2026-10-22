"""Comprehensive unit tests for Stage5EvidenceAdapter and cross-session displacement.

Validates the 10 manifest edge cases and synthetic displacement geometry:
1. Valid ID pair
2. Unresolving observation ID
3. Entity-ID / observation-ID session mismatch
4. Missing registration audit file
5. Malformed registration audit
6. Missing centroid in observations
7. Missing camera pose
8. Duplicate observation IDs
9. Stage 5 output missing a fused record
10. Valid evidence for one session but invalid for the other

Synthetic Displacement Tests:
- Pure translation with known displacement
- SE(2) rotation + translation recovery
- Uncertainty propagation consistency
"""

import json
import math
import numpy as np
import pytest
from pathlib import Path
from vlm.src.stage5_adapter import Stage5EvidenceAdapter, load_registration_audit, RegistrationInfo


@pytest.fixture
def minimal_environment(tmp_path):
    """Create a temporary environment with synthetic audit, observations, and fused files."""
    audit_file = tmp_path / "audit_summary.json"
    obs_file = tmp_path / "observations.jsonl"
    fused_file = tmp_path / "fused.jsonl"

    # Known SE(2) transform: 90 deg rotation + translation [1.0, 2.0]
    # R_12 = [[0, -1], [1, 0]], t_12 = [1.0, 2.0]
    audit_data = {
        "registration": {
            "reference_session": "semantic_mapping_my_desk_recording_01",
            "aligned_session": "semantic_mapping_my_desk_recording_02",
            "rotation_deg": 90.0,
            "translation_m": [1.0, 2.0],
            "anchors": [
                {"anchor_id": "a1", "c1": [1.0, 2.0, 0.0], "c2": [0.0, 0.0, 0.0]},
                {"anchor_id": "a2", "c1": [0.0, 3.0, 0.0], "c2": [1.0, 1.0, 0.0]},
                {"anchor_id": "a3", "c1": [2.0, 3.0, 0.0], "c2": [1.0, -1.0, 0.0]},
            ],
        }
    }
    with open(audit_file, "w") as f:
        json.dump(audit_data, f)

    # Observations:
    # S1 at [1.0, 2.0]
    # S2 at [0.0, 0.0] -> Transformed: R @ [0,0] + [1,2] = [1.0, 2.0] (Displacement = 0.0)
    obs_lines = [
        {
            "obs_id": "obs_s1_01",
            "global_entity_id": "chair_001",
            "local_track_id": "trk_01",
            "source_session": "semantic_mapping_my_desk_recording_01",
            "centroid_floor": [1.0, 2.0, 0.5],
            "centroid_camera": [0.0, 0.0, 1.5],
            "camera_pose": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
            "has_valid_pose": True,
        },
        {
            "obs_id": "obs_s2_01",
            "global_entity_id": "chair_001_s2",
            "local_track_id": "trk_02",
            "source_session": "semantic_mapping_my_desk_recording_02",
            "centroid_floor": [0.0, 0.0, 0.5],
            "centroid_camera": [0.0, 0.0, 1.5],
            "camera_pose": [1.0, 2.0, 0.0, 0.0, 0.0, 0.0, 1.0],
            "has_valid_pose": True,
        },
    ]
    with open(obs_file, "w") as f:
        for o in obs_lines:
            f.write(json.dumps(o) + "\n")

    fused_lines = [
        {
            "obs_id": "obs_s1_01",
            "entity_id": "chair_001",
            "source_session": "semantic_mapping_my_desk_recording_01",
            "geometry": {"geometric_consistency_score": 0.85, "provenance": "MEASURED_FULL"},
            "semantics": {"semantic_entropy": 0.12},
            "temporal": {"world_displacement_m": 0.0, "world_displacement_sigma_m": 0.03, "current_motion": "STATIC"},
        },
        {
            "obs_id": "obs_s2_01",
            "entity_id": "chair_001_s2",
            "source_session": "semantic_mapping_my_desk_recording_02",
            "geometry": {"geometric_consistency_score": 0.80, "provenance": "MEASURED_FULL"},
            "semantics": {"semantic_entropy": 0.15},
            "temporal": {"world_displacement_m": 0.0, "world_displacement_sigma_m": 0.03, "current_motion": "STATIC"},
        },
    ]
    with open(fused_file, "w") as f:
        for fl in fused_lines:
            f.write(json.dumps(fl) + "\n")

    return {
        "audit": audit_file,
        "obs": obs_file,
        "fused": fused_file,
        "root": tmp_path,
    }


# ==============================================================================
# Task 1: 10 Manifest Edge Case Tests
# ==============================================================================

def test_case_1_valid_id_pair(minimal_environment):
    """Case 1: Valid S1 and S2 ID pair resolves and computes accurate displacement."""
    env = minimal_environment
    adapter = Stage5EvidenceAdapter(
        repo_root=env["root"],
        audit_path=env["audit"],
        obs_path=env["obs"],
        fused_path=env["fused"],
    )

    manifest_item = {
        "entity_id": "inst_chair_synth",
        "s1_obs_id": "obs_s1_01",
        "s2_obs_id": "obs_s2_01",
        "semantic_class": "chair",
    }
    ev = adapter.build_instance_evidence(manifest_item)
    assert ev["relocation_evaluable"] is True
    assert ev["world_displacement_m"] == pytest.approx(0.0, abs=1e-3)
    assert ev["observed_motion"] == "STATIC"
    assert ev["has_valid_pose"] is True
    assert ev["depth_valid"] is True
    assert ev["s1_obs_count"] == 1
    assert ev["s2_obs_count"] == 1


def test_case_2_unresolving_observation_id(minimal_environment):
    """Case 2: Observation ID that does not resolve reports diagnostic."""
    env = minimal_environment
    adapter = Stage5EvidenceAdapter(
        repo_root=env["root"],
        audit_path=env["audit"],
        obs_path=env["obs"],
        fused_path=env["fused"],
    )

    manifest_item = {
        "entity_id": "inst_ghost",
        "s1_obs_id": "non_existent_obs_9999",
        "s2_obs_id": "obs_s2_01",
        "semantic_class": "chair",
    }
    ev = adapter.build_instance_evidence(manifest_item)
    assert ev["relocation_evaluable"] is False
    assert any("could not be resolved" in d for d in ev["diagnostics"])
    assert ev["s1_obs_count"] == 0


def test_case_3_entity_session_mismatch(minimal_environment):
    """Case 3: Observation ID belonging to Session 2 provided as Session 1 ID."""
    env = minimal_environment
    adapter = Stage5EvidenceAdapter(
        repo_root=env["root"],
        audit_path=env["audit"],
        obs_path=env["obs"],
        fused_path=env["fused"],
    )

    # Provide s2 observation for s1_obs_id
    manifest_item = {
        "entity_id": "inst_mismatch",
        "s1_obs_id": "obs_s2_01",  # belongs to recording_02!
        "s2_obs_id": "obs_s2_01",
        "semantic_class": "chair",
    }
    ev = adapter.build_instance_evidence(manifest_item)
    assert any("session mismatch" in d for d in ev["diagnostics"])


def test_case_4_missing_registration(tmp_path, minimal_environment):
    """Case 4: Missing registration audit file marks displacement NOT_EVALUABLE."""
    env = minimal_environment
    non_existent_audit = tmp_path / "missing_audit.json"
    adapter = Stage5EvidenceAdapter(
        repo_root=env["root"],
        audit_path=non_existent_audit,
        obs_path=env["obs"],
        fused_path=env["fused"],
    )

    manifest_item = {
        "entity_id": "inst_chair_synth",
        "s1_obs_id": "obs_s1_01",
        "s2_obs_id": "obs_s2_01",
        "semantic_class": "chair",
    }
    ev = adapter.build_instance_evidence(manifest_item)
    assert ev["relocation_evaluable"] is False
    assert ev["registration_status"] in ["NOT_EVALUABLE", "REGISTRATION_FILE_MISSING"]
    assert any("registration failed" in d for d in ev["diagnostics"])


def test_case_5_malformed_registration_audit(tmp_path, minimal_environment):
    """Case 5: Malformed registration audit lacks verified quality."""
    env = minimal_environment
    bad_audit = tmp_path / "bad_audit.json"
    with open(bad_audit, "w") as f:
        json.dump({"alignment_summary": {"status": "FAILED_CONVERGENCE"}}, f)

    adapter = Stage5EvidenceAdapter(
        repo_root=env["root"],
        audit_path=bad_audit,
        obs_path=env["obs"],
        fused_path=env["fused"],
    )

    manifest_item = {
        "entity_id": "inst_chair_synth",
        "s1_obs_id": "obs_s1_01",
        "s2_obs_id": "obs_s2_01",
        "semantic_class": "chair",
    }
    ev = adapter.build_instance_evidence(manifest_item)
    assert ev["relocation_evaluable"] is False
    assert ev["registration_status"] in ["NOT_EVALUABLE", "MISSING_REGISTRATION_BLOCK", "MALFORMED_REGISTRATION_PARAMETERS"]


def test_case_6_missing_centroid(tmp_path, minimal_environment):
    """Case 6: Missing centroid prevents cross-session displacement calculation."""
    env = minimal_environment
    obs_file = tmp_path / "obs_no_centroid.jsonl"
    with open(obs_file, "w") as f:
        f.write(json.dumps({
            "obs_id": "obs_no_cent",
            "source_session": "semantic_mapping_my_desk_recording_01",
            "centroid_floor": None,
            "has_valid_pose": True,
        }) + "\n")

    adapter = Stage5EvidenceAdapter(
        repo_root=env["root"],
        audit_path=env["audit"],
        obs_path=obs_file,
        fused_path=env["fused"],
    )

    manifest_item = {
        "entity_id": "inst_no_cent",
        "s1_obs_id": "obs_no_cent",
        "s2_obs_id": "obs_s2_01",
        "semantic_class": "chair",
    }
    ev = adapter.build_instance_evidence(manifest_item)
    # S1 points is empty, cannot compute 2-session displacement
    assert ev["relocation_evaluable"] is False


def test_case_7_missing_pose(tmp_path, minimal_environment):
    """Case 7: Missing camera pose marks has_valid_pose False."""
    env = minimal_environment
    obs_file = tmp_path / "obs_no_pose.jsonl"
    with open(obs_file, "w") as f:
        f.write(json.dumps({
            "obs_id": "obs_no_pose",
            "source_session": "semantic_mapping_my_desk_recording_01",
            "centroid_floor": [1.0, 2.0, 0.0],
            "centroid_camera": [0.0, 0.0, 1.5],
            "camera_pose": None,
            "has_valid_pose": False,
        }) + "\n")

    adapter = Stage5EvidenceAdapter(
        repo_root=env["root"],
        audit_path=env["audit"],
        obs_path=obs_file,
        fused_path=env["fused"],
    )

    manifest_item = {
        "entity_id": "inst_no_pose",
        "s1_obs_id": "obs_no_pose",
        "semantic_class": "chair",
    }
    ev = adapter.build_instance_evidence(manifest_item)
    assert ev["has_valid_pose"] is False


def test_case_8_duplicate_observation_ids(tmp_path, minimal_environment):
    """Case 8: Duplicate observation IDs are tracked in diagnostics."""
    env = minimal_environment
    obs_file = tmp_path / "obs_dups.jsonl"
    with open(obs_file, "w") as f:
        f.write(json.dumps({"obs_id": "dup_01", "source_session": "recording_01"}) + "\n")
        f.write(json.dumps({"obs_id": "dup_01", "source_session": "recording_01"}) + "\n")

    adapter = Stage5EvidenceAdapter(
        repo_root=env["root"],
        audit_path=env["audit"],
        obs_path=obs_file,
        fused_path=env["fused"],
    )
    assert "dup_01" in adapter.duplicate_obs_ids


def test_case_9_missing_fused_record(tmp_path, minimal_environment):
    """Case 9: Observation exists but has no Stage 5 fused evidence record."""
    env = minimal_environment
    fused_empty = tmp_path / "empty_fused.jsonl"
    fused_empty.touch()

    adapter = Stage5EvidenceAdapter(
        repo_root=env["root"],
        audit_path=env["audit"],
        obs_path=env["obs"],
        fused_path=fused_empty,
    )

    manifest_item = {
        "entity_id": "inst_no_fused",
        "s1_obs_id": "obs_s1_01",
        "semantic_class": "chair",
    }
    ev = adapter.build_instance_evidence(manifest_item)
    assert any("missing fused evidence" in d for d in ev["diagnostics"])


def test_case_10_valid_one_session_invalid_other(minimal_environment):
    """Case 10: Valid evidence for S1, but S2 identifier unresolvable."""
    env = minimal_environment
    adapter = Stage5EvidenceAdapter(
        repo_root=env["root"],
        audit_path=env["audit"],
        obs_path=env["obs"],
        fused_path=env["fused"],
    )

    manifest_item = {
        "entity_id": "inst_one_sided",
        "s1_obs_id": "obs_s1_01",
        "s2_obs_id": "missing_s2_obs",
        "semantic_class": "chair",
    }
    ev = adapter.build_instance_evidence(manifest_item)
    # Can still fall back to S1 single-session motion
    assert ev["s1_obs_count"] == 1
    assert ev["s2_obs_count"] == 0
    assert any("could not be resolved" in d for d in ev["diagnostics"])


# ==============================================================================
# Task 2: Synthetic Displacement & Coordinate Geometry Unit Tests
# ==============================================================================

def test_synthetic_pure_translation():
    """Verify displacement computation under known pure translation."""
    audit = RegistrationInfo(
        is_valid=True,
        reference_session="s1",
        aligned_session="s2",
        rotation_deg=0.0,
        R_12=np.eye(2),
        translation_m=np.array([0.5, 0.0]),
        anchor_rms_residual_m=0.02,
        anchor_count=5,
        status="SE2_REGISTERED_VERIFIED",
    )

    # Object was at [1.0, 1.0] in s1.
    # In s2, object was moved by +0.3m along Y, so in s1 coords it is at [1.0, 1.3].
    # Since c_1 = c_2_aligned + t_12:
    # c_2_aligned = [1.0, 1.3] - [0.5, 0.0] = [0.5, 1.3].
    c1 = np.array([1.0, 1.0])
    c2 = np.array([0.5, 1.3])

    c2_trans = audit.R_12 @ c2 + audit.translation_m
    disp = float(np.linalg.norm(c2_trans - c1))

    assert disp == pytest.approx(0.30, abs=1e-5)


def test_synthetic_rotation_and_translation():
    """Verify displacement computation under 90-degree counter-clockwise rotation and translation."""
    # R_12 for 90 deg CCW: [[0, -1], [1, 0]]
    # c2 = [2.0, 1.0]
    # R @ c2 = [-1.0, 2.0]
    # t = [3.0, 4.0]
    # c2_trans = [2.0, 6.0]
    # c1 = [2.0, 5.0] -> disp = 1.0m
    audit = RegistrationInfo(
        is_valid=True,
        reference_session="s1",
        aligned_session="s2",
        rotation_deg=90.0,
        R_12=np.array([[0.0, -1.0], [1.0, 0.0]]),
        translation_m=np.array([3.0, 4.0]),
        anchor_rms_residual_m=0.01,
        anchor_count=10,
        status="SE2_REGISTERED_VERIFIED",
    )

    c1 = np.array([2.0, 5.0])
    c2 = np.array([2.0, 1.0])

    c2_trans = audit.R_12 @ c2 + audit.translation_m
    disp = float(np.linalg.norm(c2_trans - c1))

    assert disp == pytest.approx(1.0, abs=1e-5)


def test_uncertainty_propagation_formula():
    """Verify documented formula: sigma_disp = sqrt(sem1^2 + sem2^2 + (RMS / sqrt(K))^2)."""
    s1 = 0.10
    n1 = 25  # sem1 = 0.10 / 5 = 0.02
    s2 = 0.15
    n2 = 9   # sem2 = 0.15 / 3 = 0.05
    rms_reg = 0.30
    k_anchors = 25  # reg_sem = 0.30 / 5 = 0.06

    sem1 = s1 / math.sqrt(n1)
    sem2 = s2 / math.sqrt(n2)
    reg_sem = rms_reg / math.sqrt(k_anchors)

    expected_sigma = math.sqrt(sem1**2 + sem2**2 + reg_sem**2)
    # sqrt(0.0004 + 0.0025 + 0.0036) = sqrt(0.0065) = 0.0806225...
    assert expected_sigma == pytest.approx(0.08062, abs=1e-4)


# ==============================================================================
# Strict Registration Validation Edge Case Tests
# ==============================================================================

def test_registration_reversed_session_direction(tmp_path):
    """Reversed session direction or unexpected session names fail validation."""
    p = tmp_path / "reg_reversed.json"
    with open(p, "w") as f:
        json.dump({
            "registration": {
                "reference_session": "semantic_mapping_my_desk_recording_02",
                "aligned_session": "semantic_mapping_my_desk_recording_01",
                "rotation_deg": 0.0,
                "translation_m": [0.0, 0.0],
                "anchors": [
                    {"c1": [0.0, 0.0], "c2": [0.0, 0.0]},
                    {"c1": [1.0, 0.0], "c2": [1.0, 0.0]},
                    {"c1": [0.0, 1.0], "c2": [0.0, 1.0]},
                ],
            }
        }, f)
    reg = load_registration_audit(
        p,
        expected_ref_session="semantic_mapping_my_desk_recording_01",
        expected_align_session="semantic_mapping_my_desk_recording_02",
    )
    assert not reg.is_valid
    assert reg.status == "SESSION_DIRECTION_INVALID"


def test_registration_fewer_than_three_anchors(tmp_path):
    """Fewer than 3 anchors is strictly rejected."""
    p = tmp_path / "reg_two_anchors.json"
    with open(p, "w") as f:
        json.dump({
            "registration": {
                "reference_session": "s1",
                "aligned_session": "s2",
                "rotation_deg": 0.0,
                "translation_m": [0.0, 0.0],
                "anchors": [
                    {"c1": [0.0, 0.0], "c2": [0.0, 0.0]},
                    {"c1": [1.0, 0.0], "c2": [1.0, 0.0]},
                ],
            }
        }, f)
    reg = load_registration_audit(p)
    assert not reg.is_valid
    assert reg.status == "INSUFFICIENT_ANCHOR_CORRESPONDENCES"


def test_registration_collinear_anchors(tmp_path):
    """Anchors that lie on a 1D straight line fail rank/geometry check."""
    p = tmp_path / "reg_collinear.json"
    with open(p, "w") as f:
        json.dump({
            "registration": {
                "reference_session": "s1",
                "aligned_session": "s2",
                "rotation_deg": 0.0,
                "translation_m": [0.0, 0.0],
                "anchors": [
                    {"c1": [1.0, 1.0], "c2": [1.0, 1.0]},
                    {"c1": [2.0, 2.0], "c2": [2.0, 2.0]},
                    {"c1": [3.0, 3.0], "c2": [3.0, 3.0]},
                ],
            }
        }, f)
    reg = load_registration_audit(p)
    assert not reg.is_valid
    assert reg.status == "DEGENERATE_ANCHOR_GEOMETRY"


def test_registration_duplicate_anchors(tmp_path):
    """Duplicate / coincident anchor positions fail validation."""
    p = tmp_path / "reg_dup.json"
    with open(p, "w") as f:
        json.dump({
            "registration": {
                "reference_session": "s1",
                "aligned_session": "s2",
                "rotation_deg": 0.0,
                "translation_m": [0.0, 0.0],
                "anchors": [
                    {"c1": [1.0, 1.0], "c2": [1.0, 1.0]},
                    {"c1": [1.0, 1.0], "c2": [1.0, 1.0]},
                    {"c1": [2.0, 3.0], "c2": [2.0, 3.0]},
                ],
            }
        }, f)
    reg = load_registration_audit(p)
    assert not reg.is_valid
    assert reg.status == "DEGENERATE_ANCHOR_GEOMETRY"


def test_registration_nan_inf_rejected(tmp_path):
    """NaN / Inf values in translation or rotation are rejected."""
    p = tmp_path / "reg_nan.json"
    with open(p, "w") as f:
        f.write('{"registration": {"reference_session": "s1", "aligned_session": "s2", "rotation_deg": NaN, "translation_m": [0.0, 0.0]}}')
    # NaN in standard JSON parses to float("nan") with standard python json
    reg = load_registration_audit(p)
    assert not reg.is_valid
    assert reg.status in ["REGISTRATION_FILE_MALFORMED", "MALFORMED_TRANSFORM_PARAMETERS"]


def test_registration_stored_residual_mismatch(tmp_path):
    """Stored residual differing from recomputed residual beyond 0.02m fails closed."""
    p = tmp_path / "reg_mismatch.json"
    with open(p, "w") as f:
        json.dump({
            "registration": {
                "reference_session": "s1",
                "aligned_session": "s2",
                "rotation_deg": 0.0,
                "translation_m": [0.0, 0.0],
                "anchors": [
                    {"c1": [0.0, 0.0], "c2": [0.0, 0.0], "residual_m": 0.00},
                    {"c1": [1.0, 0.0], "c2": [1.0, 0.0], "residual_m": 0.00},
                    {"c1": [0.0, 1.0], "c2": [0.0, 1.0], "residual_m": 0.25},  # Stored claims 0.25m, actual is 0.00m
                ],
            }
        }, f)
    reg = load_registration_audit(p)
    assert not reg.is_valid
    assert reg.status == "STORED_RESIDUAL_DISCREPANCY"


def test_registration_inconsistent_counts(tmp_path):
    """Inconsistent inlier + outlier count != total anchors fails validation."""
    p = tmp_path / "reg_counts.json"
    with open(p, "w") as f:
        json.dump({
            "registration": {
                "reference_session": "s1",
                "aligned_session": "s2",
                "rotation_deg": 0.0,
                "translation_m": [0.0, 0.0],
                "anchors": [
                    {"c1": [0.0, 0.0], "c2": [0.0, 0.0]},
                    {"c1": [1.0, 0.0], "c2": [1.0, 0.0]},
                    {"c1": [0.0, 1.0], "c2": [0.0, 1.0]},
                ],
            },
            "residuals": {
                "inlier_count": 5,  # 5 + 2 = 7 != 3 total anchors!
                "outlier_count": 2,
            }
        }, f)
    reg = load_registration_audit(p)
    assert not reg.is_valid
    assert reg.status == "INCONSISTENT_INLIER_COUNTS"


def test_registration_no_hardcoded_defaults_on_missing_optional_blocks(tmp_path):
    """Missing residuals block does NOT populate fake 0.1198 or 7 constants."""
    p = tmp_path / "reg_no_blocks.json"
    with open(p, "w") as f:
        json.dump({
            "registration": {
                "reference_session": "s1",
                "aligned_session": "s2",
                "rotation_deg": 0.0,
                "translation_m": [0.0, 0.0],
                "anchors": [
                    {"c1": [0.0, 0.0], "c2": [0.0, 0.0]},
                    {"c1": [1.0, 0.0], "c2": [1.0, 0.0]},
                    {"c1": [0.0, 1.0], "c2": [0.0, 1.0]},
                ],
            }
        }, f)
    reg = load_registration_audit(p)
    assert reg.is_valid
    # Inliers recomputed from actual residuals (all 0.0 <= 0.20m), not hardcoded 7 or 0.1198
    assert reg.robust_inlier_count == 3
    assert reg.robust_outlier_count == 0
    assert reg.post_refit_strict_inlier_rms_residual_m == pytest.approx(0.0, abs=1e-5)

