"""Disentangled Object Movability and Reliability State Contract.

Explicitly separates:
1. Contextual Movability: VLM target (could this object ordinarily be relocated?)
2. Current Motion: Geometric & temporal tracking (has it physically moved during observation?)
3. Observation Reliability: Evidence fusion result (should landmarks contribute to SLAM?)
4. Observation Treatment: Downstream action taken by the SLAM estimator.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any, Dict, Literal, Optional


@dataclass
class MovabilityState:
    """Disentangled movability, motion, and SLAM reliability state."""

    # 1. Contextual Movability (VLM contextual inference according to Rüdt et al. ontology)
    contextual_mobility: Literal["IMMOVABLE", "MOVABLE", "MOBILE", "FIXED", "UNKNOWN"] = "UNKNOWN"
    contextual_confidence: Optional[float] = None
    vlm_views_used: int = 0
    vlm_abstained: bool = False

    # 2. Observed Geometric Motion (Geometric & temporal displacement tracking)
    current_motion: Literal["STATIC", "STATIONARY", "MOVING", "UNCERTAIN"] = "UNCERTAIN"
    displacement_m: Optional[float] = None
    displacement_sigma_m: Optional[float] = None

    # 3. SLAM Observation Reliability (Evidence fusion result)
    observation_reliability: Optional[float] = None  # Scale 0.0 (unreliable) to 1.0 (highly reliable)
    slam_invalidated: Optional[bool] = None  # Whether movement makes prior SLAM landmark observations invalid

    # 4. SLAM Downstream Treatment (Action taken by SLAM optimization)
    observation_treatment: Literal["RETAIN", "DOWNWEIGHT", "REJECT", "UNDECIDED"] = "UNDECIDED"

    # Additional provenance & diagnostic metadata
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert state to serializable dictionary."""
        return {
            "contextual_mobility": self.contextual_mobility,
            "contextual_confidence": (
                round(float(self.contextual_confidence), 4)
                if self.contextual_confidence is not None
                else None
            ),
            "vlm_views_used": int(self.vlm_views_used),
            "vlm_abstained": bool(self.vlm_abstained),
            "current_motion": self.current_motion,
            "displacement_m": (
                round(float(self.displacement_m), 4)
                if self.displacement_m is not None
                else None
            ),
            "displacement_sigma_m": (
                round(float(self.displacement_sigma_m), 4)
                if self.displacement_sigma_m is not None
                else None
            ),
            "observation_reliability": (
                round(float(self.observation_reliability), 4)
                if self.observation_reliability is not None
                else None
            ),
            "slam_invalidated": self.slam_invalidated,
            "observation_treatment": self.observation_treatment,
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> MovabilityState:
        """Create state instance from dictionary."""
        return cls(
            contextual_mobility=data.get("contextual_mobility", "UNKNOWN"),
            contextual_confidence=data.get("contextual_confidence"),
            vlm_views_used=data.get("vlm_views_used", 0),
            vlm_abstained=data.get("vlm_abstained", False),
            current_motion=data.get("current_motion", "UNCERTAIN"),
            displacement_m=data.get("displacement_m"),
            displacement_sigma_m=data.get("displacement_sigma_m"),
            observation_reliability=data.get("observation_reliability"),
            slam_invalidated=data.get("slam_invalidated"),
            observation_treatment=data.get("observation_treatment", "UNDECIDED"),
            metadata=dict(data.get("metadata", {})),
        )
