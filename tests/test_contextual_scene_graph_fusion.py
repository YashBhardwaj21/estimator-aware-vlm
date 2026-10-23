"""Unit tests for Contextual Semantic Map and Dynamic 3D Scene Graph Fusion."""

import pytest
import math
import sys
from pathlib import Path
import numpy as np

root_dir = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root_dir))
sys.path.insert(0, str(root_dir / "vlm"))

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
