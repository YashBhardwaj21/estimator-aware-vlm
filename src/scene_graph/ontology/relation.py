from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple


class RelationCategory(str, Enum):
    STRUCTURAL = "STRUCTURAL"
    SPATIAL = "SPATIAL"
    VISIBILITY = "VISIBILITY"


class RelationState(str, Enum):
    CANDIDATE = "CANDIDATE"
    CONFIRMED = "CONFIRMED"
    WEAKENING = "WEAKENING"
    OCCLUDED = "OCCLUDED"
    UNKNOWN = "UNKNOWN"
    CONTRADICTED = "CONTRADICTED"
    ENDED = "ENDED"


class RelationPredicate(str, Enum):
    SUPPORTED_BY = "SUPPORTED_BY"
    INSIDE = "INSIDE"
    ATTACHED_TO = "ATTACHED_TO"

    NEAR = "NEAR"
    TOUCHING = "TOUCHING"
    LEFT_OF = "LEFT_OF"
    RIGHT_OF = "RIGHT_OF"
    ABOVE = "ABOVE"
    BELOW = "BELOW"
    FRONT_OF = "FRONT_OF"
    BEHIND = "BEHIND"

    OCCLUDES = "OCCLUDES"

    SUPPORTS = "SUPPORTS"
    CONTAINS = "CONTAINS"
    OCCLUDED_BY = "OCCLUDED_BY"


CANONICAL_PREDICATES: Set[str] = {
    RelationPredicate.SUPPORTED_BY.value,
    RelationPredicate.INSIDE.value,
    RelationPredicate.ATTACHED_TO.value,
    RelationPredicate.NEAR.value,
    RelationPredicate.TOUCHING.value,
    RelationPredicate.LEFT_OF.value,
    RelationPredicate.RIGHT_OF.value,
    RelationPredicate.ABOVE.value,
    RelationPredicate.BELOW.value,
    RelationPredicate.FRONT_OF.value,
    RelationPredicate.BEHIND.value,
    RelationPredicate.OCCLUDES.value,
}

PREDICATE_CATEGORIES: Dict[str, RelationCategory] = {
    RelationPredicate.SUPPORTED_BY.value: RelationCategory.STRUCTURAL,
    RelationPredicate.INSIDE.value: RelationCategory.STRUCTURAL,
    RelationPredicate.ATTACHED_TO.value: RelationCategory.STRUCTURAL,
    RelationPredicate.SUPPORTS.value: RelationCategory.STRUCTURAL,
    RelationPredicate.CONTAINS.value: RelationCategory.STRUCTURAL,

    RelationPredicate.NEAR.value: RelationCategory.SPATIAL,
    RelationPredicate.TOUCHING.value: RelationCategory.SPATIAL,
    RelationPredicate.LEFT_OF.value: RelationCategory.SPATIAL,
    RelationPredicate.RIGHT_OF.value: RelationCategory.SPATIAL,
    RelationPredicate.ABOVE.value: RelationCategory.SPATIAL,
    RelationPredicate.BELOW.value: RelationCategory.SPATIAL,
    RelationPredicate.FRONT_OF.value: RelationCategory.SPATIAL,
    RelationPredicate.BEHIND.value: RelationCategory.SPATIAL,

    RelationPredicate.OCCLUDES.value: RelationCategory.VISIBILITY,
    RelationPredicate.OCCLUDED_BY.value: RelationCategory.VISIBILITY,
}

INVERSE_PREDICATES: Dict[str, str] = {
    RelationPredicate.SUPPORTED_BY.value: RelationPredicate.SUPPORTS.value,
    RelationPredicate.SUPPORTS.value: RelationPredicate.SUPPORTED_BY.value,
    RelationPredicate.INSIDE.value: RelationPredicate.CONTAINS.value,
    RelationPredicate.CONTAINS.value: RelationPredicate.INSIDE.value,
    RelationPredicate.ATTACHED_TO.value: RelationPredicate.ATTACHED_TO.value,

    RelationPredicate.NEAR.value: RelationPredicate.NEAR.value,
    RelationPredicate.TOUCHING.value: RelationPredicate.TOUCHING.value,
    RelationPredicate.LEFT_OF.value: RelationPredicate.RIGHT_OF.value,
    RelationPredicate.RIGHT_OF.value: RelationPredicate.LEFT_OF.value,
    RelationPredicate.ABOVE.value: RelationPredicate.BELOW.value,
    RelationPredicate.BELOW.value: RelationPredicate.ABOVE.value,
    RelationPredicate.FRONT_OF.value: RelationPredicate.BEHIND.value,
    RelationPredicate.BEHIND.value: RelationPredicate.FRONT_OF.value,

    RelationPredicate.OCCLUDES.value: RelationPredicate.OCCLUDED_BY.value,
    RelationPredicate.OCCLUDED_BY.value: RelationPredicate.OCCLUDES.value,
}

SYMMETRIC_PREDICATES: Set[str] = {
    RelationPredicate.NEAR.value,
    RelationPredicate.TOUCHING.value,
    RelationPredicate.ATTACHED_TO.value,
}

LEGACY_PREDICATE_MAP: Dict[str, str] = {
    "ON": RelationPredicate.SUPPORTED_BY.value,
    "UNDER": RelationPredicate.SUPPORTS.value,
    "CONTAINING": RelationPredicate.CONTAINS.value,
    "IN_FRONT_OF": RelationPredicate.FRONT_OF.value,
    "OCCLUDING": RelationPredicate.OCCLUDES.value,
    "FAR": "FAR",
}


def normalize_predicate(predicate: str) -> str:
    cleaned = str(predicate).strip().upper()
    return LEGACY_PREDICATE_MAP.get(cleaned, cleaned)


def get_inverse_predicate(predicate: str) -> str:
    normalized = normalize_predicate(predicate)
    if normalized in INVERSE_PREDICATES:
        return INVERSE_PREDICATES[normalized]
    raise ValueError(f"Unknown inverse for predicate: {predicate} (normalized: {normalized})")


def is_canonical_predicate(predicate: str) -> bool:
    normalized = normalize_predicate(predicate)
    return normalized in CANONICAL_PREDICATES


def get_predicate_category(predicate: str) -> RelationCategory:
    normalized = normalize_predicate(predicate)
    if normalized in PREDICATE_CATEGORIES:
        return PREDICATE_CATEGORIES[normalized]
    raise ValueError(f"Unknown predicate category for: {predicate}")


@dataclass
class RelationEdge:
    relation_id: str
    subject_entity_id: str
    predicate: str
    object_entity_id: str
    state: RelationState = RelationState.CANDIDATE
    confidence: float = 0.0
    uncertainty: float = 1.0
    first_confirmed: float = -1.0
    last_confirmed: float = -1.0
    last_evidence: Optional[Any] = None
    source_estimators: List[str] = field(default_factory=list)
    evidence_summary: Dict[str, Any] = field(default_factory=dict)
    frame_id: int = -1
    world_timestamp: float = -1.0
    attributes: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.predicate = normalize_predicate(self.predicate)
        if isinstance(self.state, str) and not isinstance(self.state, RelationState):
            self.state = RelationState(self.state)

    @property
    def canonical_key(self) -> Tuple[str, str, str]:
        return (self.subject_entity_id, self.object_entity_id, self.predicate)

    @property
    def is_active(self) -> bool:
        return self.state in (
            RelationState.CONFIRMED,
            RelationState.WEAKENING,
            RelationState.OCCLUDED,
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "relation_id": self.relation_id,
            "subject_entity_id": self.subject_entity_id,
            "predicate": self.predicate,
            "object_entity_id": self.object_entity_id,
            "state": self.state.value,
            "confidence": float(self.confidence),
            "uncertainty": float(self.uncertainty),
            "first_confirmed": float(self.first_confirmed),
            "last_confirmed": float(self.last_confirmed),
            "source_estimators": list(self.source_estimators),
            "evidence_summary": dict(self.evidence_summary),
            "frame_id": int(self.frame_id),
            "world_timestamp": float(self.world_timestamp),
            "attributes": dict(self.attributes),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> RelationEdge:
        return cls(
            relation_id=str(data["relation_id"]),
            subject_entity_id=str(data["subject_entity_id"]),
            predicate=str(data["predicate"]),
            object_entity_id=str(data["object_entity_id"]),
            state=RelationState(data.get("state", RelationState.CANDIDATE.value)),
            confidence=float(data.get("confidence", 0.0)),
            uncertainty=float(data.get("uncertainty", 1.0)),
            first_confirmed=float(data.get("first_confirmed", -1.0)),
            last_confirmed=float(data.get("last_confirmed", -1.0)),
            source_estimators=list(data.get("source_estimators", [])),
            evidence_summary=dict(data.get("evidence_summary", {})),
            frame_id=int(data.get("frame_id", -1)),
            world_timestamp=float(data.get("world_timestamp", -1.0)),
            attributes=dict(data.get("attributes", {})),
        )
