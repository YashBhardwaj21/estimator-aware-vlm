from dataclasses import dataclass, field
from typing import Dict, Any, Optional

from scene_graph.relations.evidence import RelationEvidence
from scene_graph.temporal.relation_state import RelationState
from scene_graph.graph.participation_state import GraphParticipationState
from scene_graph.ontology.relation import (
    RelationEdge,
    RelationState as OntoRelationState,
    normalize_predicate,
)


@dataclass
class GraphEdge:
    """Represents a relational edge between two nodes in the scene graph."""
    
    predicate: str
    subject_id: str
    object_id: str
    state: RelationState
    latest_evidence: RelationEvidence
    
    participation: GraphParticipationState = GraphParticipationState.ACTIVE
    
    start_time: float = -1.0
    start_frame: int = -1
    end_time: float = -1.0
    end_frame: int = -1
    
    attributes: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def canonical_predicate(self) -> str:
        """Return the authoritative canonical predicate name."""
        return normalize_predicate(self.predicate)

    @property
    def is_active(self) -> bool:
        """Returns True if the relation is currently active in the graph."""
        return self.participation == GraphParticipationState.ACTIVE and self.state in (
            RelationState.SUPPORTED,
            RelationState.ACTIVE,
            RelationState.CONFIRMED,
        )

    def to_relation_edge(self) -> RelationEdge:
        """Export this edge as an authoritative RelationEdge."""
        conf = float(self.latest_evidence.confidence) if self.latest_evidence is not None else 0.0
        details = dict(self.latest_evidence.details) if self.latest_evidence is not None and hasattr(self.latest_evidence, "details") else {}
        ev_type = str(self.latest_evidence.evidence_type) if self.latest_evidence is not None and hasattr(self.latest_evidence, "evidence_type") else "geometric"

        if self.is_active:
            onto_state = OntoRelationState.CONFIRMED
        elif self.state in (RelationState.DECAYING, RelationState.WEAKENING):
            onto_state = OntoRelationState.WEAKENING
        elif self.state == RelationState.OCCLUDED:
            onto_state = OntoRelationState.OCCLUDED
        elif self.state == RelationState.CONTRADICTED:
            onto_state = OntoRelationState.CONTRADICTED
        elif self.state in (RelationState.TERMINATED, RelationState.ENDED):
            onto_state = OntoRelationState.ENDED
        else:
            onto_state = OntoRelationState.UNKNOWN

        return RelationEdge(
            relation_id=f"{self.subject_id}_{self.canonical_predicate}_{self.object_id}",
            subject_entity_id=self.subject_id,
            predicate=self.canonical_predicate,
            object_entity_id=self.object_id,
            state=onto_state,
            confidence=conf,
            uncertainty=max(0.0, 1.0 - conf),
            first_confirmed=self.start_time,
            last_confirmed=self.latest_evidence.timestamp if self.latest_evidence is not None else self.start_time,
            last_evidence=self.latest_evidence,
            source_estimators=[ev_type],
            evidence_summary=details,
            frame_id=self.start_frame,
            world_timestamp=self.start_time,
            attributes=dict(self.attributes),
        )

