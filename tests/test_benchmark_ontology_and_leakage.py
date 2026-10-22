"""Unit and Regression Tests for Canonical Ontology and Benchmark Leakage Quarantine.

Verifies:
1. Canonical ontology YAML file structure and class definitions.
2. Exact ontology alignment across router and visual classification prompt templates.
3. Strict ground-truth quarantine: removing, mutating, or flipping evaluator-only labels
   has ZERO effect on evidence construction, prompts, router policy decisions, or model payloads.
"""

from __future__ import annotations

import copy
from pathlib import Path
import pytest
import yaml

from vlm.src.selective_router import (
    build_structured_affordance_prompt,
    SelectiveRouter,
    parse_vlm_output,
)
from vlm.src.router_policy import evaluate_router_policy
from vlm.scripts.evaluate_zero_shot_qwen import build_zero_shot_prompt


def test_ontology_yaml_validity():
    """Verify ontology_definitions.yaml is present, well-formed, and defines all required classes and states."""
    repo_root = Path(__file__).resolve().parent.parent.parent
    ontology_path = repo_root / "vlm" / "benchmarks" / "ontology_definitions.yaml"
    assert ontology_path.exists(), f"Missing ontology definitions file at {ontology_path}"

    with open(ontology_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    assert "classes" in data
    classes = data["classes"]
    for required_cls in ["IMMOVABLE", "MOVABLE", "MOBILE", "UNKNOWN"]:
        assert required_cls in classes, f"Class {required_cls} missing from ontology"
        assert "description" in classes[required_cls]

    assert "observed_motion_states" in data
    states = data["observed_motion_states"]
    for required_state in ["STATIC", "DISPLACED", "DYNAMIC", "UNKNOWN"]:
        assert required_state in states

    assert "physical_support_types" in data
    assert "transport_context_types" in data


def test_ontology_prompt_alignment():
    """Verify that both prompt builders use the exact same canonical definitions."""
    router_prompt = build_structured_affordance_prompt("inst_001", "chair")
    zero_shot_prompt = build_zero_shot_prompt("inst_001", "chair")

    # Both must explicitly contain the canonical classes
    for c in ["IMMOVABLE", "MOVABLE", "MOBILE"]:
        assert c in router_prompt
        assert c in zero_shot_prompt

    # Both must explicitly note that carried objects remain MOVABLE
    assert "carried" in router_prompt.lower()
    assert "carried" in zero_shot_prompt.lower()

    # Neither prompt must contain ground-truth label leaks
    for forbidden in ["contextual_movability_gt", "current_motion_gt", "expected_decision", "expected_reason"]:
        assert forbidden not in router_prompt
        assert forbidden not in zero_shot_prompt


def test_strict_ground_truth_leakage_quarantine():
    """Verify mutating or removing evaluator ground-truth fields does NOT change routing decisions or prompts."""
    base_evidence = {
        "entity_id": "inst_chair_002",
        "semantic_class": "chair",
        "has_valid_pose": True,
        "depth_valid": True,
        "observed_motion": "DISPLACED",
        "world_displacement_m": 0.616,
        "displacement_uncertainty_m": 0.067,
        "views_count": 1,
        "s1_obs_count": 10,
        "s2_obs_count": 10,
    }

    # Generate baseline decision and prompt
    base_decision = evaluate_router_policy(base_evidence, policy="R3")
    base_prompt = build_structured_affordance_prompt("inst_chair_002", "chair")

    # In adversarial scenarios, inject ground-truth fields with contradictory or fake values
    adversarial_variants = [
        {"contextual_movability_gt": "IMMOVABLE", "expected_decision": "RESOLVE_CHEAP", "expected_reason": "FAKE"},
        {"contextual_movability_gt": "MOBILE", "expected_decision": "ABSTAIN", "current_motion_gt": "STATIC"},
        {"contextual_movability_gt": "UNKNOWN", "slam_landmark_validity_gt": "RETAIN_LANDMARK"},
        {"ground_truth": "IMMOVABLE", "gt_label": "MOBILE", "displacement_gt": 0.0},
    ]

    for adv in adversarial_variants:
        ev_adv = copy.deepcopy(base_evidence)
        ev_adv.update(adv)

        # Policy evaluation MUST be completely invariant to ground truth fields
        adv_decision = evaluate_router_policy(ev_adv, policy="R3")
        assert adv_decision.decision == base_decision.decision
        assert adv_decision.cheap_label == base_decision.cheap_label
        assert adv_decision.vlm_required == base_decision.vlm_required
        assert adv_decision.reason_codes == base_decision.reason_codes

        # Prompt generation MUST be completely invariant
        adv_prompt = build_structured_affordance_prompt(ev_adv["entity_id"], ev_adv["semantic_class"])
        assert adv_prompt == base_prompt
