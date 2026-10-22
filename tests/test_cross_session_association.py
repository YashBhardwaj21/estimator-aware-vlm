"""Unit and regression tests for Stage 9A: Cross-Session Physical Identity Association."""

import math
import numpy as np
import pytest

from vlm.src.stage5_adapter import RegistrationInfo
from vlm.src.cross_session_association import (
    EntityCluster,
    PhysicalInstanceMatch,
    AssociationResult,
    CrossSessionCandidateMatcher,
    evaluate_matcher_against_ground_truth,
    build_canonical_session_track_mapping,
    compare_association_baselines,
    reconcile_evaluation_invariants,
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

    # Single-session object (quadcopter drone only in recording 01; chair_004 only in recording 01)
    assert ("semantic_mapping_my_desk_recording_01", "quadcopter drone_001") in mapping
    assert ("semantic_mapping_my_desk_recording_02", "quadcopter drone_001") not in mapping
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


def test_consolidate_intra_session_clusters_merges_close_tracklets():
    """Verify that co-located tracklets of same class are merged, while distinct objects remain separate."""
    # Two fragmented chair tracks at (1.0, 1.0) and (1.1, 1.05) - distance ~0.11m < 0.35m
    c1 = _make_cluster("chair_A", "chair", 1.0, 1.0, n_obs=10)
    c2 = _make_cluster("chair_B", "chair", 1.1, 1.05, n_obs=5)
    # Distinct chair at (2.5, 2.5) - distance > 0.35m
    c3 = _make_cluster("chair_C", "chair", 2.5, 2.5, n_obs=8)
    # Desk at (1.05, 1.02) - distance ~0.05m but different class
    d1 = _make_cluster("desk_1", "desk", 1.05, 1.02, n_obs=12)

    raw_clusters = {"chair_A": c1, "chair_B": c2, "chair_C": c3, "desk_1": d1}
    consolidated = CrossSessionCandidateMatcher.consolidate_intra_session_clusters(
        raw_clusters, max_merge_distance_m=0.35
    )

    # chair_A and chair_B should be merged into 1 cluster, chair_C separate, desk_1 separate -> total 3
    assert len(consolidated) == 3
    assert "chair_C" in consolidated
    assert "desk_1" in consolidated

    # Representative should be chair_A (higher obs count: 10 vs 5)
    assert "chair_A" in consolidated
    merged_chair = consolidated["chair_A"]
    assert merged_chair.observations_count == 15
    assert set(merged_chair.local_track_ids) == {"chair_A", "chair_B"}
    assert set(merged_chair.merged_entity_ids) == {"chair_A", "chair_B"}
    # Weighted centroid: (1.0*10 + 1.1*5)/15 = 15.5/15 = 1.0333
    assert merged_chair.centroid_floor[0] == pytest.approx(1.0333, abs=0.01)


def test_consolidate_empty_clusters_returns_empty():
    """Edge case: consolidating empty dict returns empty dict."""
    res = CrossSessionCandidateMatcher.consolidate_intra_session_clusters({})
    assert res == {}


def test_consolidate_intra_session_clusters_covisible_rejection():
    """Two nearby tracklets of same class with overlapping lifespans must NOT be merged."""
    # Two chairs 0.20m apart, but co-visible for 100 frames simultaneously
    c1 = _make_cluster("chair_left", "chair", 1.0, 1.0, n_obs=100)
    c1.start_frame = 100
    c1.end_frame = 200

    c2 = _make_cluster("chair_right", "chair", 1.20, 1.0, n_obs=100)
    c2.start_frame = 100
    c2.end_frame = 200

    raw = {"chair_left": c1, "chair_right": c2}
    consolidated = CrossSessionCandidateMatcher.consolidate_intra_session_clusters(
        raw, max_merge_distance_m=0.35, max_covisible_overlap_frames=5
    )

    # Co-visible chairs must NOT be merged
    assert len(consolidated) == 2
    assert "chair_left" in consolidated
    assert "chair_right" in consolidated


def test_consolidate_intra_session_clusters_3d_height_rejection():
    """Two objects at same 2D position but different 3D heights must NOT be merged."""
    c1 = _make_cluster("chair_floor", "chair", 1.0, 1.0, n_obs=20)
    c1.centroid_3d = np.array([1.0, 1.0, 0.45])  # chair height

    c2 = _make_cluster("chair_stacked", "chair", 1.05, 1.02, n_obs=20)
    c2.centroid_3d = np.array([1.05, 1.02, 1.20])  # stacked high height

    raw = {"chair_floor": c1, "chair_stacked": c2}
    consolidated = CrossSessionCandidateMatcher.consolidate_intra_session_clusters(
        raw, max_merge_distance_m=0.35
    )
    assert len(consolidated) == 2


def test_match_entities_uncertainty_adaptive():
    """Test adaptive matching gate in peripheral drift zones with 3D height gating."""
    # Peripheral entity at (-1.5, -6.5) where registration drift is higher
    c1 = _make_cluster("desk_periph_1", "desk", -1.5, -6.5)
    c1.centroid_3d = np.array([-1.5, -6.5, 0.75])

    # S2 cluster aligned at distance 0.52m (beyond fixed 0.45m gate, within adaptive gate)
    c2 = _make_cluster("desk_periph_2", "desk", -1.5, -6.5 + 0.52)
    c2.centroid_3d = np.array([-1.5, -6.5 + 0.52, 0.75])

    s1 = {"desk_periph_1": c1}
    s2 = {"desk_periph_2": c2}

    reg = _make_reg(rotation_deg=0.0, translation_m=[0.0, 0.0])
    matcher = CrossSessionCandidateMatcher()

    # Fixed gate 0.45m fails to match at 0.52m
    res_fixed = matcher.match_entities_hungarian_gated(s1, s2, reg)
    assert len(res_fixed.matched_pairs) == 0

    # Adaptive gate succeeds in peripheral zone
    res_adaptive = matcher.match_entities_uncertainty_adaptive(s1, s2, reg, use_3d_height_gate=True)
    assert len(res_adaptive.matched_pairs) == 1
    m = res_adaptive.matched_pairs[0]
    assert m.s1_entity_id == "desk_periph_1"
    assert m.s2_entity_id == "desk_periph_2"
    assert m.distance_m == pytest.approx(0.52, abs=0.01)


def _make_synth_benchmark(annotations: list) -> dict:
    return {
        "version": "3.0.0",
        "title": "Synthetic Test Benchmark",
        "total_physical_instances": len(set(a["physical_instance_id"] for a in annotations)),
        "annotations": annotations,
    }


def _make_match(s1_id: str, s2_id: str, pid: str, dist: float = 0.1, status: str = "MATCHED") -> PhysicalInstanceMatch:
    return PhysicalInstanceMatch(
        persistent_entity_id=pid,
        s1_entity_id=s1_id,
        s2_entity_id=s2_id,
        semantic_class="chair",
        distance_m=dist,
        position_uncertainty_m=0.02,
        extent_diff_m=0.01,
        match_score=0.95,
        confidence=0.95,
        status=status,
        s1_obs_count=10,
        s2_obs_count=10,
    )


def test_evaluator_missed_instance_classified_as_unmatched_not_false_split():
    """An unrecovered physical object must be reported as UNMATCHED with 0% false-split rate."""
    gt = _make_synth_benchmark([
        {"source_session": "rec01", "local_track_id": "c1", "physical_instance_id": "inst_1", "presence": "PRESENT_BOTH", "motion": "STATIC"},
        {"source_session": "rec02", "local_track_id": "c1", "physical_instance_id": "inst_1", "presence": "PRESENT_BOTH", "motion": "STATIC"},
    ])
    res = AssociationResult(
        is_valid=True,
        registration_status="VERIFIED",
        total_s1_entities=1,
        total_s2_entities=1,
        matched_pairs=[],
        unmatched_s1=["c1"],
        unmatched_s2=["c1"],
        ambiguous_pairs=[],
    )
    s1_cls = {"c1": _make_cluster("c1", "chair", 0.0, 0.0, session="rec01")}
    s2_cls = {"c1": _make_cluster("c1", "chair", 1.0, 1.0, session="rec02")}

    eval_out = evaluate_matcher_against_ground_truth(res, gt_path=gt, s1_clusters=s1_cls, s2_clusters=s2_cls)

    assert eval_out["instance_recall"]["value_pct"] == 0.0
    assert eval_out["instance_miss_rate"]["value_pct"] == 100.0
    # Crucial: missed instance is NOT a false split!
    assert eval_out["split_instance_rate"]["value_pct"] == 0.0
    assert eval_out["true_false_split_rate_pct"] == 0.0
    assert eval_out["false_split_rate_pct"] == 0.0

    per_inst = eval_out["per_instance_results"]
    assert len(per_inst) == 1
    assert per_inst[0]["outcome"] == "UNMATCHED"
    assert per_inst[0]["is_correct"] is False
    assert per_inst[0]["involved_in_split"] is False


def test_evaluator_genuine_identity_split_increments_split_rate():
    """A physical object split across multiple predicted persistent IDs must increment split rate."""
    gt = _make_synth_benchmark([
        {"source_session": "rec01", "local_track_id": "desk_part1", "physical_instance_id": "inst_desk", "presence": "PRESENT_BOTH", "motion": "STATIC"},
        {"source_session": "rec01", "local_track_id": "desk_part2", "physical_instance_id": "inst_desk", "presence": "PRESENT_BOTH", "motion": "STATIC"},
        {"source_session": "rec02", "local_track_id": "desk_s2_a", "physical_instance_id": "inst_desk", "presence": "PRESENT_BOTH", "motion": "STATIC"},
        {"source_session": "rec02", "local_track_id": "desk_s2_b", "physical_instance_id": "inst_desk", "presence": "PRESENT_BOTH", "motion": "STATIC"},
    ])
    # Predictions associate the same physical object into two different persistent entity IDs
    m1 = _make_match("desk_part1", "desk_s2_a", pid="persistent_id_A")
    m2 = _make_match("desk_part2", "desk_s2_b", pid="persistent_id_B")
    res = AssociationResult(
        is_valid=True,
        registration_status="VERIFIED",
        total_s1_entities=2,
        total_s2_entities=2,
        matched_pairs=[m1, m2],
        unmatched_s1=[],
        unmatched_s2=[],
        ambiguous_pairs=[],
    )
    s1_cls = {
        "desk_part1": _make_cluster("desk_part1", "desk", 0.0, 0.0, session="rec01"),
        "desk_part2": _make_cluster("desk_part2", "desk", 0.2, 0.0, session="rec01"),
    }
    s2_cls = {
        "desk_s2_a": _make_cluster("desk_s2_a", "desk", 0.0, 0.0, session="rec02"),
        "desk_s2_b": _make_cluster("desk_s2_b", "desk", 0.2, 0.0, session="rec02"),
    }

    eval_out = evaluate_matcher_against_ground_truth(res, gt_path=gt, s1_clusters=s1_cls, s2_clusters=s2_cls)

    # It was recovered, but it genuinely split
    assert eval_out["instance_recall"]["value_pct"] == 100.0
    assert eval_out["split_instance_rate"]["value_pct"] == 100.0
    assert eval_out["true_false_split_rate_pct"] == 100.0
    assert eval_out["false_split_rate_pct"] == 100.0

    per_inst = eval_out["per_instance_results"]
    assert len(per_inst) == 1
    assert per_inst[0]["outcome"] == "CORRECTLY_MATCHED"
    assert per_inst[0]["is_correct"] is True
    assert per_inst[0]["involved_in_split"] is True


def test_evaluator_false_merge_increments_merge_rate_and_rejects_correctness():
    """A predicted pair joining two distinct physical objects must never mark both as correct."""
    gt = _make_synth_benchmark([
        {"source_session": "rec01", "local_track_id": "c1", "physical_instance_id": "inst_chair_1", "presence": "PRESENT_BOTH", "motion": "STATIC"},
        {"source_session": "rec02", "local_track_id": "c1", "physical_instance_id": "inst_chair_1", "presence": "PRESENT_BOTH", "motion": "STATIC"},
        {"source_session": "rec01", "local_track_id": "d2", "physical_instance_id": "inst_desk_2", "presence": "PRESENT_BOTH", "motion": "STATIC"},
        {"source_session": "rec02", "local_track_id": "d2", "physical_instance_id": "inst_desk_2", "presence": "PRESENT_BOTH", "motion": "STATIC"},
    ])
    # Faulty matcher merges c1 (chair_1) with d2 (desk_2)
    m = _make_match("c1", "d2", pid="fused_wrong_pid")
    res = AssociationResult(
        is_valid=True,
        registration_status="VERIFIED",
        total_s1_entities=2,
        total_s2_entities=2,
        matched_pairs=[m],
        unmatched_s1=["d2"],
        unmatched_s2=["c1"],
        ambiguous_pairs=[],
    )
    s1_cls = {
        "c1": _make_cluster("c1", "chair", 0.0, 0.0, session="rec01"),
        "d2": _make_cluster("d2", "desk", 2.0, 2.0, session="rec01"),
    }
    s2_cls = {
        "c1": _make_cluster("c1", "chair", 0.0, 0.0, session="rec02"),
        "d2": _make_cluster("d2", "desk", 2.0, 2.0, session="rec02"),
    }

    eval_out = evaluate_matcher_against_ground_truth(res, gt_path=gt, s1_clusters=s1_cls, s2_clusters=s2_cls)

    # 1 annotated pair, 0 true positives, 1 false merge
    assert eval_out["pair_precision"]["value_pct"] == 0.0
    assert eval_out["pair_false_merge_rate"]["value_pct"] == 100.0
    # Both instances are involved in the merge
    assert eval_out["merge_instance_rate"]["numerator"] == 2
    assert eval_out["merge_instance_rate"]["value_pct"] == 100.0
    assert eval_out["instance_recall"]["value_pct"] == 0.0

    for rec in eval_out["per_instance_results"]:
        assert rec["outcome"] == "FALSE_MERGE"
        assert rec["is_correct"] is False
        assert rec["involved_in_merge"] is True


def test_evaluator_ambiguous_candidate_rejected_classified_as_ambiguous():
    """When a candidate pair is rejected for ambiguity with no accepted match, outcome is AMBIGUOUS."""
    gt = _make_synth_benchmark([
        {"source_session": "rec01", "local_track_id": "c1", "physical_instance_id": "inst_1", "presence": "PRESENT_BOTH", "motion": "STATIC"},
        {"source_session": "rec02", "local_track_id": "c1", "physical_instance_id": "inst_1", "presence": "PRESENT_BOTH", "motion": "STATIC"},
    ])
    ambig_m = _make_match("c1", "c1", pid="ambig_pid", status="AMBIGUOUS")
    res = AssociationResult(
        is_valid=True,
        registration_status="VERIFIED",
        total_s1_entities=1,
        total_s2_entities=1,
        matched_pairs=[],
        unmatched_s1=[],
        unmatched_s2=[],
        ambiguous_pairs=[ambig_m],
    )
    s1_cls = {"c1": _make_cluster("c1", "chair", 0.0, 0.0, session="rec01")}
    s2_cls = {"c1": _make_cluster("c1", "chair", 0.0, 0.0, session="rec02")}

    eval_out = evaluate_matcher_against_ground_truth(res, gt_path=gt, s1_clusters=s1_cls, s2_clusters=s2_cls)

    assert eval_out["ambiguity_rate"]["value_pct"] == 100.0
    assert eval_out["instance_recall"]["value_pct"] == 0.0
    assert eval_out["per_instance_results"][0]["outcome"] == "AMBIGUOUS"
    assert eval_out["per_instance_results"][0]["is_correct"] is False
    assert eval_out["per_instance_results"][0]["had_ambiguous_candidates"] is True


def test_evaluator_unannotated_prediction_scope_separated():
    """Unannotated predictions must be reported as OUT_OF_BENCHMARK_SCOPE, not silent errors."""
    gt = _make_synth_benchmark([
        {"source_session": "rec01", "local_track_id": "c1", "physical_instance_id": "inst_1", "presence": "PRESENT_BOTH", "motion": "STATIC"},
        {"source_session": "rec02", "local_track_id": "c1", "physical_instance_id": "inst_1", "presence": "PRESENT_BOTH", "motion": "STATIC"},
    ])
    # Matcher accepts unannotated objects
    m_unann = _make_match("unknown_a", "unknown_b", pid="pid_unann")
    res = AssociationResult(
        is_valid=True,
        registration_status="VERIFIED",
        total_s1_entities=2,
        total_s2_entities=2,
        matched_pairs=[m_unann],
        unmatched_s1=["c1"],
        unmatched_s2=["c1"],
        ambiguous_pairs=[],
    )
    s1_cls = {
        "c1": _make_cluster("c1", "chair", 0.0, 0.0, session="rec01"),
        "unknown_a": _make_cluster("unknown_a", "chair", 5.0, 5.0, session="rec01"),
    }
    s2_cls = {
        "c1": _make_cluster("c1", "chair", 0.0, 0.0, session="rec02"),
        "unknown_b": _make_cluster("unknown_b", "chair", 5.0, 5.0, session="rec02"),
    }

    eval_out = evaluate_matcher_against_ground_truth(res, gt_path=gt, s1_clusters=s1_cls, s2_clusters=s2_cls)

    assert eval_out["unannotated_candidate_pairs_count"] == 1
    assert eval_out["annotated_eval_pairs_count"] == 0
    assert eval_out["annotation_coverage"]["value_pct"] == 0.0
    # Pair precision is null because denominator is 0
    assert eval_out["pair_precision"]["value_pct"] is None


def test_evaluator_alias_conflict_cluster_retains_all_conflicting_ids():
    """A cluster merging tracks from contradictory ground-truth instances results in ALIAS_CONFLICT."""
    gt = _make_synth_benchmark([
        {"source_session": "rec01", "local_track_id": "track_alpha", "physical_instance_id": "inst_A", "presence": "PRESENT_BOTH", "motion": "STATIC"},
        {"source_session": "rec01", "local_track_id": "track_beta", "physical_instance_id": "inst_B", "presence": "PRESENT_BOTH", "motion": "STATIC"},
        {"source_session": "rec02", "local_track_id": "track_s2", "physical_instance_id": "inst_A", "presence": "PRESENT_BOTH", "motion": "STATIC"},
    ])
    # Consolidated cluster in S1 erroneously contains both alpha and beta
    cl_conflict = _make_cluster("cl_merged", "chair", 0.0, 0.0, session="rec01")
    cl_conflict.merged_entity_ids = ["track_alpha", "track_beta"]

    s1_cls = {"cl_merged": cl_conflict}
    s2_cls = {"track_s2": _make_cluster("track_s2", "chair", 0.0, 0.0, session="rec02")}

    m = _make_match("cl_merged", "track_s2", pid="fused_conflict_pid")
    res = AssociationResult(
        is_valid=True,
        registration_status="VERIFIED",
        total_s1_entities=1,
        total_s2_entities=1,
        matched_pairs=[m],
        unmatched_s1=[],
        unmatched_s2=[],
        ambiguous_pairs=[],
    )

    eval_out = evaluate_matcher_against_ground_truth(res, gt_path=gt, s1_clusters=s1_cls, s2_clusters=s2_cls)

    assert eval_out["alias_conflict_clusters_count"] == 1
    # Both inst_A and inst_B are marked ALIAS_CONFLICT in primary outcome
    outcomes = {r["physical_instance_id"]: r["outcome"] for r in eval_out["per_instance_results"]}
    assert outcomes["inst_A"] == "ALIAS_CONFLICT"


def test_evaluator_empty_population_reports_null_metrics():
    """When evaluable population is 0, metric values must be None rather than 0.0%."""
    gt = _make_synth_benchmark([])
    res = AssociationResult(
        is_valid=True,
        registration_status="VERIFIED",
        total_s1_entities=0,
        total_s2_entities=0,
        matched_pairs=[],
        unmatched_s1=[],
        unmatched_s2=[],
        ambiguous_pairs=[],
    )
    eval_out = evaluate_matcher_against_ground_truth(res, gt_path=gt, s1_clusters={}, s2_clusters={})
    assert eval_out["instance_recall"]["value_pct"] is None
    assert eval_out["instance_miss_rate"]["value_pct"] is None


def test_evaluator_reconciliation_invariants_enforcement():
    """Tampering with results to break reconciliation invariants raises ValueError."""
    fake_eval = {
        "recovered_all_instances_count": 5,
        "instance_recall": {"numerator": 5, "denominator": 10, "value_pct": 50.0},
        "per_instance_results": [
            {"physical_instance_id": "inst_1", "outcome": "CORRECTLY_MATCHED", "is_correct": True},
            # Missing remaining 9 instances!
        ],
    }
    violations = reconcile_evaluation_invariants(fake_eval, evaluable_instance_count=10)
    assert len(violations) > 0
    assert "does not equal evaluable benchmark population" in violations[0]
