"""Unit tests for Decoupled Router Policies (R0-R4) and Estimator Influence."""

import pytest
from estimator_aware_vlm.router_policy import (
    compute_estimator_influence,
    evaluate_policy_r0,
    evaluate_policy_r1,
    evaluate_policy_r2,
    evaluate_policy_r3,
    evaluate_policy_r4,
    evaluate_router_policy,
)


def test_r0_pure_prior_policy():
    """R0 must resolve strictly by taxonomy without requesting VLM."""
    ev_door = {"entity_id": "inst_01", "semantic_class": "door"}
    r_door = evaluate_policy_r0(ev_door)
    assert r_door.decision == "RESOLVE_CHEAP"
    assert r_door.cheap_label == "IMMOVABLE"
    assert not r_door.vlm_required

    ev_chair = {"entity_id": "inst_02", "semantic_class": "chair"}
    r_chair = evaluate_policy_r0(ev_chair)
    assert r_chair.decision == "RESOLVE_CHEAP"
    assert r_chair.cheap_label == "MOVABLE"
    assert not r_chair.vlm_required

    ev_robot = {"entity_id": "inst_03", "semantic_class": "robot"}
    r_robot = evaluate_policy_r0(ev_robot)
    assert r_robot.decision == "RESOLVE_CHEAP"
    assert r_robot.cheap_label == "MOBILE"
    assert not r_robot.vlm_required


def test_r1_geometry_motion_policy():
    """R1 resolves based on observed motion and displacement without VLM."""
    # Displaced chair classified as MOVABLE without VLM
    ev_displaced = {
        "entity_id": "inst_chair_002",
        "semantic_class": "chair",
        "observed_motion": "DISPLACED",
        "world_displacement_m": 0.56,
    }
    r = evaluate_policy_r1(ev_displaced)
    assert r.decision == "RESOLVE_CHEAP"
    assert r.cheap_label == "MOVABLE"
    assert not r.vlm_required
    assert "R1_GEOMETRY_MOTION_DISPLACED" in r.reason_codes


def test_r2_non_estimator_aware_policy():
    """R2 escalates on ambiguity blind to SLAM landmark influence."""
    ev = {
        "entity_id": "inst_pedestal_fan_001",
        "semantic_class": "pedestal fan",
        "has_valid_pose": True,
        "depth_valid": True,
        "observed_motion": "STATIC",
        "has_images": True,
        "views_count": 1,
        "s1_obs_count": 0,
        "s2_obs_count": 0,
    }
    r = evaluate_policy_r2(ev)
    assert r.decision == "ESCALATE"
    assert r.vlm_required


def test_r3_estimator_aware_dependency():
    """R3 demonstrably uses the estimator-relevance signal."""
    # Condition A: High observation support and valid pose -> high estimator influence -> ESCALATE
    ev_high_influence = {
        "entity_id": "inst_chair_002",
        "semantic_class": "chair",
        "has_valid_pose": True,
        "depth_valid": True,
        "observed_motion": "DISPLACED",
        "world_displacement_m": 0.563,
        "displacement_uncertainty_m": 0.04,
        "has_images": True,
        "views_count": 2,
        "s1_obs_count": 15,
        "s2_obs_count": 10,
        "geometric_consistency_score": 0.95,
    }
    r_high = evaluate_policy_r3(ev_high_influence)
    assert r_high.decision == "ESCALATE"
    assert r_high.vlm_required
    assert r_high.estimator_influence is not None and r_high.estimator_influence > 0.50
    assert "HIGH_ESTIMATOR_RELEVANCE" in r_high.reason_codes

    # Condition B: Invalid pose -> DEFER
    ev_invalid_pose = dict(ev_high_influence)
    ev_invalid_pose["has_valid_pose"] = False
    r_inv = evaluate_policy_r3(ev_invalid_pose)
    assert r_inv.decision == "DEFER"
    assert not r_inv.vlm_required
    assert r_inv.estimator_influence == 0.0


def test_r4_all_eligible_vlm_policy():
    """R4 requests VLM for every image-eligible instance."""
    ev = {
        "entity_id": "inst_desk_001",
        "has_images": True,
        "views_count": 1,
    }
    r = evaluate_policy_r4(ev)
    assert r.decision == "ESCALATE"
    assert r.vlm_required
    assert "R4_ALL_ELIGIBLE_VLM" in r.reason_codes


def test_estimator_influence_computation():
    """Verify compute_estimator_influence scales properly with observation support."""
    ev_few = {"has_valid_pose": True, "s1_obs_count": 1, "s2_obs_count": 1}
    inf_few = compute_estimator_influence(ev_few)

    ev_many = {"has_valid_pose": True, "s1_obs_count": 10, "s2_obs_count": 10}
    inf_many = compute_estimator_influence(ev_many)

    assert inf_few < inf_many
    assert inf_many <= 1.0


def test_unified_evaluate_router_policy():
    """Verify master evaluate_router_policy dispatches correctly by name."""
    ev = {
        "entity_id": "inst_01",
        "semantic_class": "door",
        "has_valid_pose": True,
        "depth_valid": True,
        "observed_motion": "STATIC",
        "has_images": True,
        "views_count": 1,
    }
    r0 = evaluate_router_policy(ev, policy="R0")
    assert r0.decision == "RESOLVE_CHEAP"
    assert not r0.vlm_required

    r4 = evaluate_router_policy(ev, policy="R4")
    assert r4.decision == "ESCALATE"
    assert r4.vlm_required


def test_r2_vs_r3_divergence_on_estimator_influence():
    """Verify that R2 and R3 make divergent decisions when estimator utility warrants it.

    Under ambiguous physical support (e.g. pedestal fan):
    - Case A (High influence): Key landmark actively constraining SLAM -> R2 and R3 both escalate.
    - Case B (Low influence): Background clutter with poor geometry -> R2 escalates (blind to SLAM),
      but R3 avoids query (resolves cheap) because expected SLAM loss reduction does not justify VLM cost.
    """
    # High influence case
    ev_high = {
        "entity_id": "inst_fan_high",
        "semantic_class": "pedestal fan",
        "has_valid_pose": True,
        "depth_valid": True,
        "observed_motion": "STATIC",
        "has_images": True,
        "views_count": 2,
        "s1_obs_count": 8,
        "geometric_consistency_score": 0.90,
    }
    r2_high = evaluate_policy_r2(ev_high)
    r3_high = evaluate_policy_r3(ev_high)
    assert r2_high.decision == "ESCALATE"
    assert r3_high.decision == "ESCALATE"
    assert r3_high.query_utility is not None and r3_high.query_utility >= 0.0

    # Low influence case (poor geometric consistency / peripheral clutter)
    ev_low = {
        "entity_id": "inst_fan_low",
        "semantic_class": "pedestal fan",
        "has_valid_pose": True,
        "depth_valid": True,
        "observed_motion": "STATIC",
        "has_images": True,
        "views_count": 1,
        "total_observations": 1,
        "geometric_consistency_score": 0.10,
    }
    r2_low = evaluate_policy_r2(ev_low)
    r3_low = evaluate_policy_r3(ev_low)
    # R2 escalates purely on affordance ambiguity
    assert r2_low.decision == "ESCALATE"
    assert r2_low.vlm_required is True
    # R3 avoids query because expected SLAM loss reduction does not justify VLM compute cost
    assert r3_low.decision == "RESOLVE_CHEAP"
    assert r3_low.vlm_required is False
    assert r3_low.query_utility is not None and r3_low.query_utility < 0.0
    assert "SUBTHRESHOLD_QUERY_UTILITY_AVOIDED" in r3_low.reason_codes


def test_backend_zero_inliers_sets_zero_influence():
    """Verify that backend telemetry with zero inliers drops estimator influence to 0.0."""
    from estimator_aware_vlm.router_policy import compute_estimator_influence_with_provenance
    ev_zero_inliers = {
        "entity_id": "clutter_01",
        "has_valid_pose": True,
        "depth_valid": True,
        "inlier_count": 0,
        "inlier_ratio": 0.0,
        "s1_obs_count": 10,
    }
    score, prov = compute_estimator_influence_with_provenance(ev_zero_inliers)
    assert score == 0.0
    assert prov == "MEASURED_BACKEND_ZERO_INLIERS"
