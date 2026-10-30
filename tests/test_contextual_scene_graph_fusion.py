"""Unit tests for Contextual Semantic Map and Dynamic 3D Scene Graph Fusion."""

import pytest
import math
import sys
from pathlib import Path
import numpy as np

root_dir = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root_dir / "src"))

from estimator_aware_vlm.contextual_scene_graph_fusion import (
    ContextualMovability,
    ObservedMotionState,
    ContextualSceneGraphNode,
    ContextualSceneGraphFusion,
)


def test_contextual_node_initialization():
    """Verify that a newly instantiated scene graph node separates movability and motion."""
    fusion = ContextualSceneGraphFusion()
    routed = {
        "prediction": "MOVABLE",
        "observed_motion": "STATIC",
        "physical_support": "freestanding_on_surface",
        "evidence_provenance": "MEASURED_FULL",
        "decision": "RESOLVE_CHEAP",
        "reason_codes": ["STATIC_CANONICAL_MOVABLE_PRIOR"],
        "vlm_required": False,
    }
    meta = {
        "centroid": [1.5, -0.8, 0.45],
        "spatial_extent_m": 0.65,
        "coordinate_frame": "floor_map",
        "source_session": "session_01",
        "obs_id": "obs_001",
        "timestamp": 100.0,
    }

    node = fusion.update_or_create_node("chair_001", "chair", routed, meta)

    assert node.entity_id == "chair_001"
    assert node.semantic_class == "chair"
    assert node.contextual_movability == ContextualMovability.MOVABLE
    assert node.observed_motion == ObservedMotionState.STATIC
    assert node.physical_support == "freestanding_on_surface"
    assert math.isclose(node.centroid_world[0], 1.5, abs_tol=1e-3)
    assert node.vlm_decision == "RESOLVE_CHEAP"


def test_geometry_preserved_when_movability_unknown():
    """Verify that an UNKNOWN or DEFERRED semantic verdict does not destroy valid 3D geometry."""
    fusion = ContextualSceneGraphFusion()
    routed = {
        "prediction": "UNKNOWN",
        "observed_motion": "STATIC",
        "decision": "DEFER",
        "reason_codes": ["MISSING_REQUIRED_VISUAL_CONTEXT"],
        "vlm_required": False,
    }
    meta = {
        "centroid": [2.2, 1.1, 0.8],
        "spatial_extent_m": 1.2,
        "source_session": "session_01",
        "obs_id": "obs_002",
        "timestamp": 101.0,
    }

    node = fusion.update_or_create_node("whiteboard_001", "whiteboard", routed, meta)

    assert node.contextual_movability == ContextualMovability.UNKNOWN
    # Valid geometric state MUST be preserved
    assert math.isclose(node.centroid_world[0], 2.2, abs_tol=1e-3)
    assert math.isclose(node.spatial_extent_m, 1.2, abs_tol=1e-3)
    assert node.vlm_decision == "DEFER"


def test_physical_relocation_updates_motion_without_corrupting_movability():
    """Verify that physical relocation marks DISPLACED without forcing IMMOVABLE to change."""
    fusion = ContextualSceneGraphFusion()
    # Step 1: Initial static observation
    routed1 = {"prediction": "MOVABLE", "observed_motion": "STATIC"}
    meta1 = {"centroid": [1.0, 1.0, 0.4], "source_session": "session_01", "timestamp": 0.0}
    fusion.update_or_create_node("chair_reloc", "chair", routed1, meta1)

    # Step 2: Relocated in Session 2
    routed2 = {
        "prediction": "MOVABLE",
        "observed_motion": "DISPLACED",
        "decision": "ESCALATE",
        "vlm_required": True,
        "vlm_inference_executed": True,
    }
    meta2 = {"centroid": [1.56, 1.0, 0.4], "source_session": "session_02", "timestamp": 500.0}
    node = fusion.update_or_create_node("chair_reloc", "chair", routed2, meta2)

    assert node.contextual_movability == ContextualMovability.MOVABLE
    assert node.observed_motion == ObservedMotionState.DISPLACED
    assert "session_01" in node.source_sessions
    assert "session_02" in node.source_sessions


def test_contradictory_view_handling():
    """Verify that contradictory predictions decay confidence rather than thrashing."""
    fusion = ContextualSceneGraphFusion()
    # Step 1: View 1 predicts IMMOVABLE with high confidence
    routed1 = {"prediction": "IMMOVABLE", "observed_motion": "STATIC"}
    node = fusion.update_or_create_node("fixture_001", "shelf", routed1, {"timestamp": 1.0})
    assert node.contextual_movability == ContextualMovability.IMMOVABLE
    initial_conf = node.semantic_confidence

    # Step 2: View 2 contradicts with MOVABLE
    routed2 = {"prediction": "MOVABLE", "observed_motion": "STATIC"}
    node_updated = fusion.update_or_create_node("fixture_001", "shelf", routed2, {"timestamp": 2.0})

    # Confidence must decay when contradicted
    assert node_updated.semantic_confidence < initial_conf


# ---------------------------------------------------------------------------
# MovabilityState Data Contract & Coexistence Invariants
# ---------------------------------------------------------------------------

from estimator_aware_vlm.state import MovabilityState


def test_stationary_movable_object_coexistence():
    """Verify stationary movable object retains high reliability and observation treatment without conflation."""
    chair_state = MovabilityState(
        contextual_mobility="MOVABLE",
        contextual_confidence=0.92,
        current_motion="STATIONARY",
        displacement_m=0.01,
        displacement_sigma_m=0.03,
        observation_reliability=0.95,
        observation_treatment="RETAIN",
        vlm_views_used=3,
        metadata={"instance_id": "chair_004", "evidence": "wheeled base, stable in observation"},
    )

    assert chair_state.contextual_mobility == "MOVABLE"
    assert chair_state.current_motion == "STATIONARY"
    assert chair_state.observation_reliability == 0.95
    assert chair_state.observation_treatment == "RETAIN"
    assert chair_state.displacement_m == 0.01


def test_relocated_movable_object():
    """Verify relocated movable object receives downweighted observation treatment."""
    displaced_chair = MovabilityState(
        contextual_mobility="MOVABLE",
        contextual_confidence=0.88,
        current_motion="MOVING",
        displacement_m=0.75,
        displacement_sigma_m=0.04,
        observation_reliability=0.15,
        observation_treatment="DOWNWEIGHT",
    )

    assert displaced_chair.contextual_mobility == "MOVABLE"
    assert displaced_chair.current_motion == "MOVING"
    assert displaced_chair.observation_reliability < 0.3
    assert displaced_chair.observation_treatment == "DOWNWEIGHT"


def test_structural_fixed_object():
    """Verify structural fixed fixture retains maximum reliability and retain treatment."""
    fan_state = MovabilityState(
        contextual_mobility="FIXED",
        contextual_confidence=0.99,
        current_motion="STATIONARY",
        displacement_m=0.00,
        observation_reliability=0.99,
        observation_treatment="RETAIN",
    )

    assert fan_state.contextual_mobility == "FIXED"
    assert fan_state.current_motion == "STATIONARY"
    assert fan_state.observation_treatment == "RETAIN"


def test_serialization_roundtrip():
    """Ensure state serializes to dict and restores losslessly."""
    original = MovabilityState(
        contextual_mobility="MOVABLE",
        contextual_confidence=0.8421,
        current_motion="STATIONARY",
        displacement_m=0.0215,
        displacement_sigma_m=0.0102,
        observation_reliability=0.8914,
        observation_treatment="RETAIN",
        vlm_views_used=2,
        metadata={"class_prior": "chair"},
    )

    data = original.to_dict()
    restored = MovabilityState.from_dict(data)

    assert restored.contextual_mobility == original.contextual_mobility
    assert restored.current_motion == original.current_motion
    assert restored.contextual_confidence == round(original.contextual_confidence, 4)
    assert restored.displacement_m == round(original.displacement_m, 4)
    assert restored.observation_treatment == original.observation_treatment
    assert restored.metadata == original.metadata


# ---------------------------------------------------------------------------
# Downstream SLAM Landmark Action & Lifecycle Metrics
# ---------------------------------------------------------------------------

from scripts.evaluate_downstream_lifecycle import (
    determine_downstream_landmark_action,
    evaluate_policy_lifecycle,
)


def test_determine_downstream_landmark_action():
    """Verify mapping from mobility prediction and observed motion to downstream SLAM landmark actions."""
    # 1. Stationary architectural object -> RETAIN
    act1, reason1, decouple1 = determine_downstream_landmark_action("IMMOVABLE", "STATIC", 0.0, 0.02)
    assert act1 == "RETAIN_LANDMARK"
    assert decouple1 is False

    # 2. Stationary movable furniture -> RETAIN
    act2, reason2, decouple2 = determine_downstream_landmark_action("MOVABLE", "STATIC", 0.0, 0.02)
    assert act2 == "RETAIN_LANDMARK"
    assert decouple2 is False

    # 3. Displaced chair -> INVALIDATE_RELOCATED with pose decoupling
    act3, reason3, decouple3 = determine_downstream_landmark_action("MOVABLE", "DISPLACED", 0.616, 0.067)
    assert act3 == "INVALIDATE_RELOCATED"
    assert decouple3 is True
    assert "PURGED_RELOCATED" in reason3

    # 4. Mobile agent -> INVALIDATE_DYNAMIC_AGENT
    act4, reason4, decouple4 = determine_downstream_landmark_action("MOBILE", "DYNAMIC", 1.85, 0.15)
    assert act4 == "INVALIDATE_DYNAMIC_AGENT"
    assert decouple4 is False


def test_evaluate_policy_lifecycle_metrics():
    """Verify landmark retention precision and false invalidation rate computation."""
    inputs_map = {
        "inst_1": {"physical_instance_id": "inst_1", "semantic_class": "door", "estimator_evidence": {"observed_motion": "STATIC", "measured_displacement_m": 0.0, "displacement_uncertainty_m": 0.02}},
        "inst_2": {"physical_instance_id": "inst_2", "semantic_class": "chair", "estimator_evidence": {"observed_motion": "DISPLACED", "measured_displacement_m": 0.55, "displacement_uncertainty_m": 0.05}},
        "inst_3": {"physical_instance_id": "inst_3", "semantic_class": "person", "estimator_evidence": {"observed_motion": "DYNAMIC", "measured_displacement_m": 1.5, "displacement_uncertainty_m": 0.10}},
    }

    gt_map = {
        "inst_1": {"slam_landmark_validity_gt": "RETAIN_LANDMARK"},
        "inst_2": {"slam_landmark_validity_gt": "INVALIDATE_RELOCATED"},
        "inst_3": {"slam_landmark_validity_gt": "INVALIDATE_DYNAMIC_AGENT"},
    }

    predictions = [
        {"physical_instance_id": "inst_1", "semantic_class": "door", "parsed_prediction": "IMMOVABLE", "response_received": True, "prediction_accepted": True},
        {"physical_instance_id": "inst_2", "semantic_class": "chair", "parsed_prediction": "MOVABLE", "response_received": True, "prediction_accepted": True},
        {"physical_instance_id": "inst_3", "semantic_class": "person", "parsed_prediction": "MOBILE", "response_received": True, "prediction_accepted": True},
    ]

    res = evaluate_policy_lifecycle("TEST_POLICY", predictions, inputs_map, gt_map)
    assert res["landmark_retention"]["precision"] == 1.0
    assert res["landmark_retention"]["false_invalidation_rate"] == 0.0
    assert res["landmark_retention"]["false_positives_corruption_risk"] == 0
    assert res["relocation_handling"]["decoupled_pose_count"] == 1
