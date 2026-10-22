"""Unit and regression tests for Stage 9A: Cross-Session Physical Identity Association."""

import math
import numpy as np
import pytest

from vlm.src.stage5_adapter import RegistrationInfo
from vlm.src.cross_session_association import (
    EntityCluster,
    CrossSessionCandidateMatcher,
    evaluate_matcher_against_ground_truth,
    build_canonical_session_track_mapping,
    compare_association_baselines,
)


def _make_cluster(
    entity_id: str,
    semantic_class: str,
    x: float,
    y: float,
    session: str = "rec01",
    extent: float = 0.50,
    n_obs: int = 10,
    dispersion: float = 0.05,
) -> EntityCluster:
    return EntityCluster(
        entity_id=entity_id,
        semantic_class=semantic_class,
        source_session=session,
        centroid_floor=np.array([x, y], dtype=np.float64),
        centroid_std=dispersion,
        spatial_extent_m=extent,
        observations_count=n_obs,
        local_track_ids=[entity_id],
        observation_ids=[f"obs_{entity_id}_{i}" for i in range(n_obs)],
        position_uncertainty_m=dispersion / math.sqrt(n_obs),
    )


def _make_reg(
    is_valid: bool = True,
    status: str = "VERIFIED",
    translation_m: np.ndarray = None,
    rotation_deg: float = 0.0,
    R_12: np.ndarray = None,
    anchor_rms_residual_m: float = 0.02,
    anchor_count: int = 3,
) -> RegistrationInfo:
    if translation_m is None:
        translation_m = np.zeros(2, dtype=np.float64)
    if R_12 is None:
        R_12 = np.eye(2, dtype=np.float64)
    return RegistrationInfo(
        is_valid=is_valid,
        reference_session="rec01",
        aligned_session="rec02",
        rotation_deg=rotation_deg,
        translation_m=translation_m,
        R_12=R_12,
        anchor_rms_residual_m=anchor_rms_residual_m,
        anchor_count=anchor_count,
        status=status,
    )


def test_pure_translation_matching():
    """Test matching with known pure translation SE(2)."""
    # Session 1: entity at (1.0, 2.0)
    s1 = {"chair_001": _make_cluster("chair_001", "chair", 1.0, 2.0, session="rec01")}
    # Session 2: shifted by dx=-0.5, dy=1.0 -> entity in S2 is at (1.5, 1.0)
    # Registration: c1 = I @ c2 + [-0.5, 1.0] -> [-0.5, 1.0] + [1.5, 1.0] = [1.0, 2.0]
    s2 = {"chair_009": _make_cluster("chair_009", "chair", 1.5, 1.0, session="rec02")}

    reg = _make_reg(
        translation_m=np.array([-0.5, 1.0], dtype=np.float64),
    )

    matcher = CrossSessionCandidateMatcher(max_static_distance_m=0.45)
    res = matcher.match_entities(s1, s2, reg)

    assert res.is_valid
    assert len(res.matched_pairs) == 1
    m = res.matched_pairs[0]
    assert m.s1_entity_id == "chair_001"
    assert m.s2_entity_id == "chair_009"
    assert m.distance_m < 0.01
    assert m.status == "MATCHED"
    assert len(res.unmatched_s1) == 0
    assert len(res.unmatched_s2) == 0


def test_se2_rotation_and_translation_matching():
    """Test matching with full SE(2) rotation and translation."""
    theta = np.deg2rad(45.0)
    R = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    t = np.array([2.0, -1.0])

    # S1 positions
    c1_pos = np.array([3.0, 4.0])
    s1 = {"desk_001": _make_cluster("desk_001", "desk", c1_pos[0], c1_pos[1], session="rec01")}

    # S2 position: c2 = R^T (c1 - t)
    c2_pos = R.T @ (c1_pos - t)
    s2 = {"desk_020": _make_cluster("desk_020", "desk", c2_pos[0], c2_pos[1], session="rec02")}

    reg = _make_reg(
        translation_m=t,
        rotation_deg=45.0,
        R_12=R,
        anchor_rms_residual_m=0.03,
        anchor_count=4,
    )

    matcher = CrossSessionCandidateMatcher()
    res = matcher.match_entities(s1, s2, reg)

    assert res.is_valid
    assert len(res.matched_pairs) == 1
    assert res.matched_pairs[0].s1_entity_id == "desk_001"
    assert res.matched_pairs[0].s2_entity_id == "desk_020"
    assert res.matched_pairs[0].distance_m < 0.01


def test_nearby_distinct_chairs_rejected_if_beyond_gate():
    """Objects of same class beyond max_static_distance_m are rejected."""
    s1 = {"chair_001": _make_cluster("chair_001", "chair", 1.0, 1.0)}
    # Distinct chair 0.8m away
    s2 = {"chair_002": _make_cluster("chair_002", "chair", 1.8, 1.0)}

    reg = _make_reg()

    matcher = CrossSessionCandidateMatcher(max_static_distance_m=0.45)
    res = matcher.match_entities(s1, s2, reg)

    assert len(res.matched_pairs) == 0
    assert "chair_001" in res.unmatched_s1
    assert "chair_002" in res.unmatched_s2


def test_repeated_desks_globally_assigned():
    """Hungarian algorithm disambiguates row of identical desks cleanly."""
    # Row of desks at x=1.0, 2.0, 3.0
    s1 = {
        "desk_A": _make_cluster("desk_A", "desk", 1.0, 0.0),
        "desk_B": _make_cluster("desk_B", "desk", 2.0, 0.0),
        "desk_C": _make_cluster("desk_C", "desk", 3.0, 0.0),
    }
    # S2 desks with small jitter (0.05m)
    s2 = {
        "desk_1": _make_cluster("desk_1", "desk", 1.05, 0.0),
        "desk_2": _make_cluster("desk_2", "desk", 2.02, -0.01),
        "desk_3": _make_cluster("desk_3", "desk", 2.98, 0.03),
    }

    reg = _make_reg()

    matcher = CrossSessionCandidateMatcher(max_static_distance_m=0.45, ambiguity_margin_m=0.30)
    res = matcher.match_entities(s1, s2, reg)

    assert len(res.matched_pairs) == 3
    mapping = {m.s1_entity_id: m.s2_entity_id for m in res.matched_pairs}
    assert mapping["desk_A"] == "desk_1"
    assert mapping["desk_B"] == "desk_2"
    assert mapping["desk_C"] == "desk_3"


def test_ambiguity_margin_detection():
    """If two candidates in S1 are virtually equidistant to S2 candidate, detect ambiguity."""
    # S1 has two chairs at (1.0, 0.0) and (1.06, 0.0)
    s1 = {
        "chair_A": _make_cluster("chair_A", "chair", 1.0, 0.0),
        "chair_B": _make_cluster("chair_B", "chair", 1.06, 0.0),
    }
    # S2 chair at (1.03, 0.0)
    s2 = {
        "chair_1": _make_cluster("chair_1", "chair", 1.03, 0.0),
    }

    reg = _make_reg()

    # Ambiguity margin of 0.15m: distance to chair_A is 0.03m, to chair_B is 0.03m -> diff is 0.00m < 0.15m!
    matcher = CrossSessionCandidateMatcher(max_static_distance_m=0.45, ambiguity_margin_m=0.15)
    res = matcher.match_entities(s1, s2, reg)

    assert len(res.matched_pairs) == 0
    assert len(res.ambiguous_pairs) == 1
    assert res.ambiguous_pairs[0].status == "AMBIGUOUS"


def test_invalid_registration_blocks_matching():
    """Unverified or invalid registration immediately gates out cross-session matches."""
    s1 = {"chair_001": _make_cluster("chair_001", "chair", 1.0, 2.0)}
    s2 = {"chair_002": _make_cluster("chair_002", "chair", 1.0, 2.0)}

    reg = _make_reg(
        is_valid=False,
        status="FAILED_RESIDUAL_TOO_HIGH",
        anchor_rms_residual_m=0.85,
        anchor_count=1,
    )

    matcher = CrossSessionCandidateMatcher()
    res = matcher.match_entities(s1, s2, reg)

    assert not res.is_valid
    assert len(res.matched_pairs) == 0
    assert "chair_001" in res.unmatched_s1
    assert "chair_002" in res.unmatched_s2


def test_semantic_class_mismatch_gated():
    """Objects at same location but different semantic classes are not matched."""
    s1 = {"chair_001": _make_cluster("chair_001", "chair", 1.0, 1.0)}
    s2 = {"desk_001": _make_cluster("desk_001", "desk", 1.0, 1.0)}

    reg = _make_reg()

    matcher = CrossSessionCandidateMatcher()
    res = matcher.match_entities(s1, s2, reg)

    assert len(res.matched_pairs) == 0
    assert len(res.unmatched_s1) == 1
    assert len(res.unmatched_s2) == 1


def test_extent_gate_rejection():
    """Objects at same location but incompatible bounding extents are rejected."""
    s1 = {"desk_small": _make_cluster("desk_small", "desk", 1.0, 1.0, extent=0.40)}
    s2 = {"desk_giant": _make_cluster("desk_giant", "desk", 1.0, 1.0, extent=1.50)}

    reg = _make_reg()

    # Relative diff: abs(0.40 - 1.50) / 1.50 = 0.733 > 0.60
    matcher = CrossSessionCandidateMatcher(max_extent_relative_diff=0.60)
    res = matcher.match_entities(s1, s2, reg)

    assert len(res.matched_pairs) == 0


def test_symmetric_ambiguity_direction_b():
    """Test ambiguity when S1 has 1 candidate and S2 has 2 competing nearby candidates."""
    # S1 has one chair at (1.0, 0.0)
    s1 = {"chair_A": _make_cluster("chair_A", "chair", 1.0, 0.0)}
    # S2 has two chairs at (1.02, 0.0) and (1.05, 0.0) -> both within 0.15m of each other
    s2 = {
        "chair_1": _make_cluster("chair_1", "chair", 1.02, 0.0),
        "chair_2": _make_cluster("chair_2", "chair", 1.05, 0.0),
    }

    reg = _make_reg()

    matcher = CrossSessionCandidateMatcher(max_static_distance_m=0.45, ambiguity_margin_m=0.15)
    res = matcher.match_entities(s1, s2, reg)

    # Must be marked AMBIGUOUS due to symmetric check in Direction B
    assert len(res.matched_pairs) == 0
    assert len(res.ambiguous_pairs) == 1
    assert "competing S2 neighbor" in res.ambiguous_pairs[0].diagnostics[0]


def test_true_relocation_beyond_static_gate():
    """Objects that moved >0.45m between sessions are not forcibly merged as static objects."""
    s1 = {"chair_001": _make_cluster("chair_001", "chair", 1.0, 1.0)}
    # Physically relocated by 1.2m
    s2 = {"chair_001_s2": _make_cluster("chair_001_s2", "chair", 2.2, 1.0)}

    reg = _make_reg()

    matcher = CrossSessionCandidateMatcher(max_static_distance_m=0.45)
    res = matcher.match_entities(s1, s2, reg)

    assert len(res.matched_pairs) == 0
    assert "chair_001" in res.unmatched_s1
    assert "chair_001_s2" in res.unmatched_s2


def test_duplicate_track_ids_different_objects():
    """Ensure spatial geometry dictates physical identity, NOT identical session-local track names."""
    # S1 has chair_001 at (1.0, 1.0)
    s1 = {"chair_001": _make_cluster("chair_001", "chair", 1.0, 1.0)}
    # S2 renumbered: chair_001 is at (5.0, 5.0), while chair_010 is at (1.02, 1.01)
    s2 = {
        "chair_001": _make_cluster("chair_001", "chair", 5.0, 5.0),
        "chair_010": _make_cluster("chair_010", "chair", 1.02, 1.01),
    }

    reg = _make_reg()

    matcher = CrossSessionCandidateMatcher(max_static_distance_m=0.45)
    res = matcher.match_entities(s1, s2, reg)

    assert len(res.matched_pairs) == 1
    m = res.matched_pairs[0]
    # Matched with chair_010 based on true physical position, NOT with chair_001!
    assert m.s1_entity_id == "chair_001"
    assert m.s2_entity_id == "chair_010"
    assert "chair_001" in res.unmatched_s2


def test_canonical_session_track_mapping_loads_verified_annotations():
    """Verify loading the canonical ground truth mapping with 113 explicit session-track annotations."""
    mapping, pairs = build_canonical_session_track_mapping()
    assert len(mapping) == 113, f"Expected exactly 113 canonical mappings, got {len(mapping)}"

    # Check known entries
    assert mapping[("semantic_mapping_my_desk_recording_01", "desk_001")] == "inst_desk_001"
    assert mapping[("semantic_mapping_my_desk_recording_02", "desk_001")] == "inst_desk_001"


def test_canonical_mapping_detects_contradictions(tmp_path):
    """Verify that contradictory or conflicting annotations raise a ValueError."""
    import json

    conflict_data = {
        "annotations": [
            {
                "source_session": "rec01",
                "local_track_id": "desk_001",
                "physical_instance_id": "inst_desk_001",
            },
            {
                "source_session": "rec01",
                "local_track_id": "desk_001",
                "physical_instance_id": "inst_desk_CONTRADICTING",
            },
        ]
    }
    p = tmp_path / "conflict_annotations.json"
    with open(p, "w", encoding="utf-8") as f:
        json.dump(conflict_data, f)

    with pytest.raises(ValueError, match="Contradictory annotation for"):
        build_canonical_session_track_mapping(p)


def test_canonical_mapping_track_name_reuse_clean_separation():
    """Verify that tracker ID reuse across sessions maps to different physical instances."""
    mapping, _ = build_canonical_session_track_mapping()
    # desk_016 in Session 1 is inst_desk_016; desk_016 in Session 2 is inst_desk_008
    assert mapping[("semantic_mapping_my_desk_recording_01", "desk_016")] == "inst_desk_016"
    assert mapping[("semantic_mapping_my_desk_recording_02", "desk_016")] == "inst_desk_008"


def test_canonical_mapping_single_session_and_multi_track_support():
    """Verify single-session objects and multiple track IDs per physical instance."""
    mapping, _ = build_canonical_session_track_mapping()

    # Single-session object (quadcopter drone only in recording 02; chair_004 only in recording 01)
    assert ("semantic_mapping_my_desk_recording_02", "quadcopter drone_001") in mapping
    assert ("semantic_mapping_my_desk_recording_01", "quadcopter drone_001") not in mapping
    assert ("semantic_mapping_my_desk_recording_01", "chair_004") in mapping
    assert ("semantic_mapping_my_desk_recording_02", "chair_004") not in mapping

    # Multi-track object across sessions (robot platform observed as robot_platform_base in S1 and robot_001 in S2)
    assert mapping[("semantic_mapping_my_desk_recording_01", "robot_platform_base")] == "inst_rc_car_robot_001"
    assert mapping[("semantic_mapping_my_desk_recording_02", "robot_001")] == "inst_rc_car_robot_001"


def test_compare_association_baselines_contract():
    """Verify the 3-baseline comparison contract on synthetic clusters."""
    # Synthetic clusters with 1 clear match and 1 ambiguous pair
    s1 = {
        "desk_01": _make_cluster("desk_01", "desk", 1.0, 1.0),
        "chair_A": _make_cluster("chair_A", "chair", 3.0, 3.0),
        "chair_B": _make_cluster("chair_B", "chair", 3.05, 3.0),
    }
    s2 = {
        "desk_01": _make_cluster("desk_01", "desk", 1.02, 1.01),
        "chair_1": _make_cluster("chair_1", "chair", 3.02, 3.0),
    }
    reg = _make_reg()

    results = compare_association_baselines(s1, s2, reg)

    assert "baseline_1_nearest_centroid" in results
    assert "baseline_2_hungarian_gated" in results
    assert "proposed_uncertainty_symmetric" in results

    for key, data in results.items():
        res = data["result"]
        eval_dict = data["evaluation"]
        assert hasattr(res, "matched_pairs")
        assert "algorithm_name" in eval_dict
        assert "precision_on_annotated_subset_pct" in eval_dict
        assert "true_positive_matches" in eval_dict
        assert "false_merges" in eval_dict
        assert "annotated_eval_pairs_count" in eval_dict


def test_registration_runtime_robust_transform_agreement():
    """Verify that load_registration_audit matches the RANSAC robust transform and residuals."""
    from pathlib import Path
    from vlm.src.stage5_adapter import load_registration_audit

    repo_root = Path(__file__).resolve().parent.parent.parent
    reg_path = repo_root / "vlm" / "outputs" / "stage5_registration_audit.json"
    if not reg_path.exists():
        pytest.skip("Registration audit artifact missing")

    reg = load_registration_audit(reg_path)
    assert reg.is_valid is True
    assert reg.rotation_deg == pytest.approx(-15.558, abs=0.01)
    assert np.allclose(reg.translation_m, [-0.1059, 0.0486], atol=1e-3)
    assert reg.anchor_count == 10
    # All-anchor RMS residual ~ 0.3087m
    assert reg.anchor_rms_residual_m == pytest.approx(0.3087, abs=0.01)
    # Robust inlier RMS residual ~ 0.1198m
    assert reg.inlier_rms_residual_m == pytest.approx(0.1198, abs=0.01)
    assert reg.median_residual_m == pytest.approx(0.0924, abs=0.01)


def test_canonical_benchmark_missing_fails_closed(tmp_path):
    """Verify that missing canonical identity annotations file fails closed with FileNotFoundError."""
    missing_file = tmp_path / "nonexistent_annotations.json"
    with pytest.raises(FileNotFoundError, match="Canonical cross-session benchmark missing"):
        build_canonical_session_track_mapping(missing_file)


def test_relocation_aware_candidate_matching():
    """Stage 10 Core Contract: Relocation-aware matching associates moved objects beyond static gate.
    
    Verifies that an object relocated by 0.563m (e.g. chair_002) is gated out by static matching,
    but successfully resolved as MATCHED_RELOCATED by the relocation-aware matcher.
    """
    # S1 chair at (1.747, -1.960)
    s1 = {"chair_002": _make_cluster("chair_002", "chair", 1.747, -1.960, extent=0.55, session="rec01")}
    # S2 chair after 0.563m relocation at (2.308, -2.000)
    s2 = {"chair_010": _make_cluster("chair_010", "chair", 2.308, -2.000, extent=0.55, session="rec02")}

    reg = _make_reg()  # Identity transform for unit test

    matcher = CrossSessionCandidateMatcher(
        max_static_distance_m=0.45,
        max_relocation_distance_m=1.20,
        min_relocation_distance_m=0.25,
    )

    # 1. Static matcher must NOT match (0.563m > 0.45m static gate)
    res_static = matcher.match_entities(s1, s2, reg)
    assert len(res_static.matched_pairs) == 0
    assert "chair_002" in res_static.unmatched_s1
    assert "chair_010" in res_static.unmatched_s2

    # 2. Relocation-aware matcher recognizes the moved physical instance
    res_reloc = matcher.match_entities_relocation_aware(s1, s2, reg)
    assert len(res_reloc.matched_pairs) == 1
    m = res_reloc.matched_pairs[0]
    assert m.s1_entity_id == "chair_002"
    assert m.s2_entity_id == "chair_010"
    assert m.status == "MATCHED_RELOCATED"
    assert m.outcome == "MATCHED_RELOCATED"
    assert m.distance_m == pytest.approx(0.563, abs=0.01)
    assert len(res_reloc.unmatched_s1) == 0
    assert len(res_reloc.unmatched_s2) == 0



