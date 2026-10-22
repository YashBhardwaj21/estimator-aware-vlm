"""Unit tests for Stage 2 MovabilityState data contract."""

import pytest
import sys
from pathlib import Path

# Add project root and vlm to sys.path
root_dir = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(root_dir))
sys.path.insert(0, str(root_dir / "vlm"))

from vlm.src.state import MovabilityState


def test_stationary_movable_object_coexistence():
    """Pass condition test:
    A stationary chair must be able to have STATIONARY motion, MOVABLE contextual mobility,
    and high observation reliability (e.g. 0.95, RETAIN) simultaneously without conflation.
    """
    chair_state = MovabilityState(
        contextual_mobility="MOVABLE",
        contextual_confidence=0.92,
        current_motion="STATIONARY",
        displacement_m=0.01,
        displacement_sigma_m=0.03,
        observation_reliability=0.95,
        observation_treatment="RETAIN",
        vlm_views_used=3,
        metadata={"instance_id": "chair_004", "evidence": "wheeled base, stable in observation"}
    )

    assert chair_state.contextual_mobility == "MOVABLE"
    assert chair_state.current_motion == "STATIONARY"
    assert chair_state.observation_reliability == 0.95
    assert chair_state.observation_treatment == "RETAIN"
    assert chair_state.displacement_m == 0.01


def test_relocated_movable_object():
    """Case B: Object is contextually movable and has actually displaced."""
    displaced_chair = MovabilityState(
        contextual_mobility="MOVABLE",
        contextual_confidence=0.88,
        current_motion="MOVING",
        displacement_m=0.75,
        displacement_sigma_m=0.04,
        observation_reliability=0.15,
        observation_treatment="DOWNWEIGHT"
    )

    assert displaced_chair.contextual_mobility == "MOVABLE"
    assert displaced_chair.current_motion == "MOVING"
    assert displaced_chair.observation_reliability < 0.3
    assert displaced_chair.observation_treatment == "DOWNWEIGHT"


def test_structural_fixed_object():
    """Case: Structural fixture (e.g. ceiling fan or wall) that is FIXED."""
    fan_state = MovabilityState(
        contextual_mobility="FIXED",
        contextual_confidence=0.99,
        current_motion="STATIONARY",
        displacement_m=0.00,
        observation_reliability=0.99,
        observation_treatment="RETAIN"
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
        metadata={"class_prior": "chair"}
    )

    data = original.to_dict()
    restored = MovabilityState.from_dict(data)

    assert restored.contextual_mobility == original.contextual_mobility
    assert restored.current_motion == original.current_motion
    assert restored.contextual_confidence == round(original.contextual_confidence, 4)
    assert restored.displacement_m == round(original.displacement_m, 4)
    assert restored.observation_treatment == original.observation_treatment
    assert restored.metadata == original.metadata
