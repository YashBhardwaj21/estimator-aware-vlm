"""Unit tests for Stage 6 Selective Router Policy, Dispatcher, and Adapter."""

import pytest
import sys
from pathlib import Path
from PIL import Image

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


def test_wrong_object_id_rejected():
    """Phase 2: A response naming the wrong object must not be accepted as valid."""
    raw_mismatched = '{"object_id": "inst_wrong_999", "object_class": "chair", "movability": "MOVABLE", "abstain": false}'
    res, is_json, is_schema = parse_vlm_output(raw_mismatched, "inst_chair_001", "chair")
    assert is_json
    assert not is_schema
    assert not res["id_matched"]
    assert res["abstain"] is True
    assert res["movability"] == "UNKNOWN"
    assert res["validation_error"] == "object_id_mismatch"


def test_ontology_alias_normalization():
    """Phase 2: Verify alias normalization to Rüdt et al. canonical classes."""
    # IMMOVABLE aliases
    for alias in ["FIXED", "STRUCTURAL", "ARCHITECTURAL", "WALL_MOUNTED"]:
        raw = f'{{"object_id": "inst_01", "object_class": "door", "movability": "{alias}", "abstain": false}}'
        res, is_json, is_schema = parse_vlm_output(raw, "inst_01", "door")
        assert res["movability"] == "IMMOVABLE"
        assert not res["abstain"]

    # MOVABLE aliases
    for alias in ["PORTABLE", "RELOCATABLE", "FURNITURE", "FREESTANDING"]:
        raw = f'{{"object_id": "inst_01", "object_class": "chair", "movability": "{alias}", "abstain": false}}'
        res, is_json, is_schema = parse_vlm_output(raw, "inst_01", "chair")
        assert res["movability"] == "MOVABLE"
        assert not res["abstain"]

    # MOBILE aliases
    for alias in ["DYNAMIC", "SELF_PROPELLED", "ROBOT", "AGENT"]:
        raw = f'{{"object_id": "inst_01", "object_class": "robot", "movability": "{alias}", "abstain": false}}'
        res, is_json, is_schema = parse_vlm_output(raw, "inst_01", "robot")
        assert res["movability"] == "MOBILE"
        assert not res["abstain"]


def test_missing_views_rejected_before_inference(tmp_path):
    """Phase 2: Empty visual context must cause DEFER and never execute text-only inference."""
    router = SelectiveRouter(enable_vlm=True, mode="replay")
    empty_dir = tmp_path / "empty_crops"
    empty_dir.mkdir()

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
        "crops_dir": str(empty_dir),
    }

    res = router.route_entity(ev)
    assert res["decision"] == "DEFER"
    assert "MISSING_REQUIRED_VISUAL_CONTEXT" in res["reason_codes"]
    assert not res["vlm_inference_executed"]
    assert res["prediction"] == "UNKNOWN"
    assert res["abstained"] is True


def test_replay_mode_cache_miss_semantics():
    """Phase 2: Replay mode must record explicit cache miss rather than calling server."""
    router = SelectiveRouter(enable_vlm=True, mode="replay", cache_path=None)
    ev = {
        "entity_id": "inst_pedestal_fan_001",
        "semantic_class": "pedestal fan",
        "has_valid_pose": True,
        "depth_valid": True,
        "observed_motion": "STATIC",
        "has_images": True,
        "views_count": 1,
        "crops_dir": "vlm/outputs/multiview_samples/inst_pedestal_fan_001",
    }
    res = router.route_entity(ev, repo_root=root_dir)
    # If the key is not in cache, replay mode cleanly returns miss without network
    assert res["execution_mode"] in ["replay", "CACHED_HISTORICAL"]
    assert not res["vlm_inference_executed"]


def test_wrong_object_class_rejected():
    """Phase 9: Response naming the wrong semantic class must be rejected."""
    raw_class_mismatch = '{"object_id": "inst_chair_001", "object_class": "refrigerator", "movability": "MOVABLE", "abstain": false}'
    res, is_json, is_schema = parse_vlm_output(raw_class_mismatch, "inst_chair_001", "chair")
    assert is_json
    assert not is_schema
    assert not res["class_matched"]
    assert res["abstain"] is True
    assert res["movability"] == "UNKNOWN"
    assert res["validation_error"] == "object_class_mismatch"


def test_corrupt_unreadable_image_rejected(tmp_path):
    """Phase 9: Corrupt or unreadable image must fail decoding validation and prevent inference."""
    router = SelectiveRouter(enable_vlm=True, mode="replay")
    corrupt_dir = tmp_path / "corrupt_inst"
    corrupt_dir.mkdir()
    corrupt_crop = corrupt_dir / "crop_view_0.jpg"
    corrupt_crop.write_bytes(b"NOT_A_VALID_JPEG_HEADER_CORRUPTED_BYTES")

    valid_images, telemetry = router.validate_image_inputs(corrupt_dir, expected_views=1)
    assert len(valid_images) == 0
    assert len(telemetry["decoding_failures"]) == 1
    assert "crop_view_0.jpg" in telemetry["decoding_failures"][0]


def test_cache_invalidation_on_image_modification(tmp_path):
    """Phase 9: Changing image binary contents must alter fingerprint and invalidate cache."""
    router = SelectiveRouter(enable_vlm=True, mode="replay")
    test_dir = tmp_path / "mod_inst"
    test_dir.mkdir()
    crop_file = test_dir / "crop_view_0.jpg"

    # Create valid dummy image 1
    im1 = Image.new("RGB", (64, 64), color="red")
    im1.save(crop_file, format="JPEG")
    imgs1, tel1 = router.validate_image_inputs(test_dir, expected_views=1)
    key1 = router._compute_cache_key("inst_mod", {"semantic_class": "chair"}, imgs1)

    # Modify image binary
    im2 = Image.new("RGB", (64, 64), color="blue")
    im2.save(crop_file, format="JPEG")
    imgs2, tel2 = router.validate_image_inputs(test_dir, expected_views=1)
    key2 = router._compute_cache_key("inst_mod", {"semantic_class": "chair"}, imgs2)

    assert key1 != key2
    assert tel1["image_hashes"][0] != tel2["image_hashes"][0]


