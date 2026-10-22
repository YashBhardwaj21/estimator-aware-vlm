"""Unit tests for Stage 6 Selective Router Policy, Dispatcher, and Adapter."""

import pytest
import sys
from pathlib import Path

# Add project root and vlm to sys.path
root_dir = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(root_dir))
sys.path.insert(0, str(root_dir / "vlm"))

from vlm.src.router_policy import evaluate_router_policy, RouterDecision
from vlm.src.selective_router import SelectiveRouter, clean_json_text, parse_vlm_output
from vlm.src.stage5_adapter import Stage5EvidenceAdapter


def test_gate1_invalid_pose_deferred():
    """Verify that untrustworthy camera pose causes DEFER decision."""
    ev = {
        "entity_id": "chair_001",
        "semantic_class": "chair",
        "has_valid_pose": False,
        "depth_valid": True,
    }
    decision = evaluate_router_policy(ev)
    assert decision.decision == "DEFER"
    assert "UNTRUSTWORTHY_ESTIMATOR_POSE" in decision.reason_codes
    assert not decision.vlm_required


def test_gate1_invalid_depth_on_moving_target():
    """Verify that moving target with invalid depth is deferred."""
    ev = {
        "entity_id": "chair_002",
        "semantic_class": "chair",
        "has_valid_pose": True,
        "depth_valid": False,
        "observed_motion": "MOVING",
        "world_displacement_m": 0.50,
    }
    decision = evaluate_router_policy(ev)
    assert decision.decision == "DEFER"
    assert "INVALID_DEPTH_ON_MOVING_TARGET" in decision.reason_codes


def test_canonical_mobile_agent_resolved_cheaply():
    """Verify person or robot resolves immediately to MOBILE without VLM."""
    for cls_name in ["person", "robot", "quadcopter drone"]:
        ev = {
            "entity_id": f"inst_{cls_name}_001",
            "semantic_class": cls_name,
            "has_valid_pose": True,
            "depth_valid": True,
            "observed_motion": "DYNAMIC",
        }
        decision = evaluate_router_policy(ev)
        assert decision.decision == "RESOLVE_CHEAP"
        assert decision.cheap_label == "MOBILE"
        assert not decision.vlm_required
        assert "CANONICAL_MOBILE_AGENT_PRIOR" in decision.reason_codes


def test_canonical_immovable_resolved_cheaply():
    """Verify door or ceiling fan without displacement resolves to IMMOVABLE without VLM."""
    for cls_name in ["door", "ceiling fan", "wall"]:
        ev = {
            "entity_id": f"inst_{cls_name}_001",
            "semantic_class": cls_name,
            "has_valid_pose": True,
            "depth_valid": True,
            "observed_motion": "STATIC",
            "world_displacement_m": 0.02,
        }
        decision = evaluate_router_policy(ev)
        assert decision.decision == "RESOLVE_CHEAP"
        assert decision.cheap_label == "IMMOVABLE"
        assert not decision.vlm_required
        assert "CANONICAL_IMMOVABLE_ARCHITECTURAL_PRIOR" in decision.reason_codes


def test_stationary_movable_resolved_cheaply():
    """Verify stationary desk or laptop resolves to MOVABLE without VLM."""
    for cls_name in ["desk", "laptop", "backpack"]:
        ev = {
            "entity_id": f"inst_{cls_name}_001",
            "semantic_class": cls_name,
            "has_valid_pose": True,
            "depth_valid": True,
            "observed_motion": "STATIC",
            "world_displacement_m": 0.03,
            "displacement_uncertainty_m": 0.02,
        }
        decision = evaluate_router_policy(ev)
        assert decision.decision == "RESOLVE_CHEAP"
        assert decision.cheap_label == "MOVABLE"
        assert not decision.vlm_required


def test_significant_displacement_escalated_with_images():
    """Verify relocated chair with images escalates to VLM."""
    ev = {
        "entity_id": "inst_chair_002",
        "semantic_class": "chair",
        "has_valid_pose": True,
        "depth_valid": True,
        "observed_motion": "DISPLACED",
        "world_displacement_m": 0.563,
        "displacement_uncertainty_m": 0.04,
        "has_images": True,
        "views_count": 2,
    }
    decision = evaluate_router_policy(ev)
    assert decision.decision == "ESCALATE"
    assert decision.vlm_required
    assert "SIGNIFICANT_DISPLACEMENT_AFFORDANCE_UNRESOLVED" in decision.reason_codes


def test_significant_displacement_deferred_without_images():
    """Verify relocated chair without images defers instead of failing silently."""
    ev = {
        "entity_id": "inst_chair_002",
        "semantic_class": "chair",
        "has_valid_pose": True,
        "depth_valid": True,
        "observed_motion": "DISPLACED",
        "world_displacement_m": 0.563,
        "has_images": False,
        "views_count": 0,
    }
    decision = evaluate_router_policy(ev)
    assert decision.decision == "DEFER"
    assert not decision.vlm_required
    assert "MISSING_REQUIRED_VISUAL_CONTEXT" in decision.reason_codes


def test_ambiguous_affordance_escalation():
    """Verify pedestal fan escalates for visual mounting affordance check."""
    ev = {
        "entity_id": "inst_pedestal_fan_001",
        "semantic_class": "pedestal fan",
        "has_valid_pose": True,
        "depth_valid": True,
        "observed_motion": "STATIC",
        "has_images": True,
        "views_count": 1,
    }
    decision = evaluate_router_policy(ev)
    assert decision.decision == "ESCALATE"
    assert decision.vlm_required
    assert "AMBIGUOUS_PHYSICAL_SUPPORT_AFFORDANCE" in decision.reason_codes


def test_router_output_contract():
    """Verify SelectiveRouter outputs conform to required dictionary schema."""
    router = SelectiveRouter(enable_vlm=False)
    ev = {
        "entity_id": "inst_desk_001",
        "semantic_class": "desk",
        "has_valid_pose": True,
        "depth_valid": True,
        "observed_motion": "STATIC",
        "world_displacement_m": 0.01,
        "has_images": True,
        "views_count": 3,
    }
    res = router.route_entity(ev)

    required_keys = [
        "entity_id", "decision", "reason_codes", "evidence_valid",
        "evidence_provenance", "vlm_required", "vlm_model", "views_used",
        "prediction", "abstained", "decision_latency_ms", "inference_latency_ms"
    ]
    for k in required_keys:
        assert k in res
    assert res["decision"] == "RESOLVE_CHEAP"
    assert res["prediction"] == "MOVABLE"
    assert not res["abstained"]
    assert res["decision_latency_ms"] < 5.0


def test_stage5_adapter_zero_leakage():
    """Verify that Stage5EvidenceAdapter produces evidence WITHOUT target ground truth fields."""
    adapter = Stage5EvidenceAdapter(repo_root=root_dir)
    manifest_entry = {
        "entity_id": "inst_chair_002",
        "s1_obs_id": "chair_002",
        "s2_obs_id": "chair_001",
        "semantic_class": "chair",
        "contextual_movability_gt": "MOVABLE",
        "observed_motion_gt": "DISPLACED",
        "expected_decision": "ESCALATE",
        "expected_reason": "SIGNIFICANT_DISPLACEMENT_AFFORDANCE_UNRESOLVED",
        "has_images": True,
        "views_count": 1,
        "crops_dir": "vlm/outputs/multiview_samples/inst_chair_002",
    }

    evidence = adapter.build_instance_evidence(manifest_entry)

    # Benchmark target fields MUST be absent from routing evidence
    forbidden_keys = [
        "contextual_movability_gt",
        "observed_motion_gt",
        "expected_decision",
        "expected_reason",
    ]
    for k in forbidden_keys:
        assert k not in evidence

    # Estimated fields must be present and correctly typed
    assert evidence["entity_id"] == "inst_chair_002"
    assert evidence["semantic_class"] == "chair"
    assert isinstance(evidence["world_displacement_m"], float)
    assert evidence["world_displacement_m"] > 0.30  # Estimated displacement from real observations
    assert evidence["observed_motion"] in ["DISPLACED", "MOVING"]
    assert evidence["has_valid_pose"] is True
    assert evidence["depth_valid"] is True


def test_vlm_output_validation_and_safe_fallback():
    """Verify that malformed or adversarial VLM outputs fail gracefully."""
    # Case 1: Malformed JSON
    res, is_json, is_schema = parse_vlm_output("not valid json at all", "inst_01", "chair")
    assert not is_json
    assert not is_schema
    assert res["abstain"] is True
    assert res["movability"] == "UNKNOWN"

    # Case 2: Invalid movability label (e.g. "FRUIT")
    raw_invalid = '{"object_id": "inst_01", "object_class": "chair", "movability": "FRUIT"}'
    res2, is_json2, is_schema2 = parse_vlm_output(raw_invalid, "inst_01", "chair")
    assert is_json2
    assert res2["abstain"] is True
    assert res2["movability"] == "UNKNOWN"

    # Case 3: Valid structured response
    raw_valid = '{"object_id": "inst_01", "object_class": "chair", "movability": "MOVABLE", "abstain": false, "physical_support": "freestanding_on_surface"}'
    res3, is_json3, is_schema3 = parse_vlm_output(raw_valid, "inst_01", "chair")
    assert is_json3
    assert is_schema3
    assert res3["abstain"] is False
    assert res3["movability"] == "MOVABLE"
    assert res3["physical_support"] == "freestanding_on_surface"
