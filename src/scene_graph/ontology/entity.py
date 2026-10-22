from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional
import numpy as np


class EntityType(str, Enum):
    WORLD = "WORLD"
    PLACE = "PLACE"
    SURFACE = "SURFACE"
    CONTAINER = "CONTAINER"
    OBJECT = "OBJECT"
    AGENT = "AGENT"
    ROBOT = "ROBOT"


class EntityLifecycleState(str, Enum):
    TENTATIVE = "TENTATIVE"
    CONFIRMED = "CONFIRMED"
    VISIBLE = "VISIBLE"
    OCCLUDED = "OCCLUDED"
    STALE = "STALE"
    LOST = "LOST"
    ARCHIVED = "ARCHIVED"


class VisibilityState(str, Enum):
    VISIBLE = "VISIBLE"
    OCCLUDED = "OCCLUDED"
    OUT_OF_VIEW = "OUT_OF_VIEW"
    UNKNOWN = "UNKNOWN"


@dataclass
class SemanticHypothesis:
    label: str
    confidence: float
    source: str = "detector"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "label": self.label,
            "confidence": float(self.confidence),
            "source": self.source,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> SemanticHypothesis:
        return cls(
            label=str(data.get("label", "unknown")),
            confidence=float(data.get("confidence", 0.0)),
            source=str(data.get("source", "detector")),
        )


@dataclass
class PersistentEntity:
    entity_id: str
    entity_type: EntityType = EntityType.OBJECT
    semantic_hypotheses: List[SemanticHypothesis] = field(default_factory=list)
    pose: np.ndarray = field(default_factory=lambda: np.eye(4, dtype=np.float64))
    geometry: Optional[Any] = None
    velocity: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float64))
    uncertainty: np.ndarray = field(default_factory=lambda: np.eye(3, dtype=np.float64) * 0.1)
    first_seen: float = 0.0
    last_seen: float = 0.0
    last_observed: float = 0.0
    visibility_state: VisibilityState = VisibilityState.UNKNOWN
    lifecycle_state: EntityLifecycleState = EntityLifecycleState.TENTATIVE
    source_track_ids: List[str] = field(default_factory=list)
    parent_context_id: Optional[str] = None
    attributes: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if isinstance(self.entity_type, str) and not isinstance(self.entity_type, EntityType):
            self.entity_type = EntityType(self.entity_type)
        if isinstance(self.visibility_state, str) and not isinstance(self.visibility_state, VisibilityState):
            self.visibility_state = VisibilityState(self.visibility_state)
        if isinstance(self.lifecycle_state, str) and not isinstance(self.lifecycle_state, EntityLifecycleState):
            self.lifecycle_state = EntityLifecycleState(self.lifecycle_state)

        if not isinstance(self.pose, np.ndarray) or self.pose.shape != (4, 4):
            self.pose = np.eye(4, dtype=np.float64)
        if not isinstance(self.velocity, np.ndarray) or self.velocity.shape != (3,):
            self.velocity = np.zeros(3, dtype=np.float64)
        if not isinstance(self.uncertainty, np.ndarray) or self.uncertainty.shape != (3, 3):
            self.uncertainty = np.eye(3, dtype=np.float64) * 0.1

    @property
    def primary_label(self) -> str:
        if not self.semantic_hypotheses:
            return "unknown"
        best = max(self.semantic_hypotheses, key=lambda h: h.confidence)
        return best.label

    @property
    def primary_confidence(self) -> float:
        if not self.semantic_hypotheses:
            return 0.0
        best = max(self.semantic_hypotheses, key=lambda h: h.confidence)
        return float(best.confidence)

    @property
    def is_active(self) -> bool:
        return self.lifecycle_state in (
            EntityLifecycleState.CONFIRMED,
            EntityLifecycleState.VISIBLE,
            EntityLifecycleState.OCCLUDED,
            EntityLifecycleState.STALE,
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "entity_id": self.entity_id,
            "entity_type": self.entity_type.value,
            "semantic_hypotheses": [h.to_dict() for h in self.semantic_hypotheses],
            "pose": self.pose.tolist(),
            "velocity": self.velocity.tolist(),
            "uncertainty": self.uncertainty.tolist(),
            "first_seen": float(self.first_seen),
            "last_seen": float(self.last_seen),
            "last_observed": float(self.last_observed),
            "visibility_state": self.visibility_state.value,
            "lifecycle_state": self.lifecycle_state.value,
            "source_track_ids": list(self.source_track_ids),
            "parent_context_id": self.parent_context_id,
            "attributes": dict(self.attributes),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> PersistentEntity:
        hypotheses = [
            SemanticHypothesis.from_dict(h)
            for h in data.get("semantic_hypotheses", [])
        ]
        pose = np.asarray(data.get("pose", np.eye(4, dtype=np.float64)), dtype=np.float64)
        velocity = np.asarray(data.get("velocity", np.zeros(3, dtype=np.float64)), dtype=np.float64)
        uncertainty = np.asarray(data.get("uncertainty", np.eye(3, dtype=np.float64) * 0.1), dtype=np.float64)

        return cls(
            entity_id=str(data["entity_id"]),
            entity_type=EntityType(data.get("entity_type", EntityType.OBJECT.value)),
            semantic_hypotheses=hypotheses,
            pose=pose,
            velocity=velocity,
            uncertainty=uncertainty,
            first_seen=float(data.get("first_seen", 0.0)),
            last_seen=float(data.get("last_seen", 0.0)),
            last_observed=float(data.get("last_observed", 0.0)),
            visibility_state=VisibilityState(data.get("visibility_state", VisibilityState.UNKNOWN.value)),
            lifecycle_state=EntityLifecycleState(data.get("lifecycle_state", EntityLifecycleState.TENTATIVE.value)),
            source_track_ids=list(data.get("source_track_ids", [])),
            parent_context_id=data.get("parent_context_id"),
            attributes=dict(data.get("attributes", {})),
        )
