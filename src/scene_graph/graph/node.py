from dataclasses import dataclass, field
from typing import Dict, Any, Optional
import numpy as np


from scene_graph.tracking.track import Track, TrackState
from scene_graph.temporal.object_state import ObjectState
from scene_graph.graph.participation_state import GraphParticipationState
from scene_graph.ontology.entity import (
    EntityType,
    EntityLifecycleState,
    VisibilityState,
    SemanticHypothesis,
    PersistentEntity,
)


@dataclass
class GraphNode:
    """Represents an entity node in the scene graph."""
    
    object_id: str
    class_name: str
    track: Track
    state: ObjectState
    participation: GraphParticipationState = GraphParticipationState.ACTIVE
    spatial_context_id: str = "world"
    is_spatial_anchor: bool = False
    attributes: Dict[str, Any] = field(default_factory=dict)
    entity_type: EntityType = EntityType.OBJECT
    persistent_entity_id: Optional[str] = None

    def to_persistent_entity(self) -> PersistentEntity:
        """Export this node as an authoritative PersistentEntity."""
        ent_id = self.persistent_entity_id or self.object_id
        if self.is_observed:
            vis = VisibilityState.VISIBLE
        elif not self.attributes.get("in_frustum", getattr(self.track, "is_in_frustum", True)):
            vis = VisibilityState.OUT_OF_VIEW
        else:
            vis = VisibilityState.OCCLUDED
        life = EntityLifecycleState.CONFIRMED if self.is_active else EntityLifecycleState.STALE
        
        obs_stamp = float(self.track.last_timestamp) if hasattr(self.track, "last_timestamp") and self.track.last_timestamp is not None else 0.0
        first_stamp = obs_stamp
        
        conf = getattr(self.track, "detection_confidence", 0.0)
        hypotheses = [
            SemanticHypothesis(
                label=self.class_name,
                confidence=float(conf),
                source="tracker",
            )
        ]

        centroid = self.track.centroid_world if hasattr(self.track, "centroid_world") and self.track.centroid_world is not None else np.zeros(3, dtype=np.float64)
        pose = np.eye(4, dtype=np.float64)
        pose[:3, 3] = centroid

        cov = getattr(self.track, "position_covariance_world", None)
        vel = getattr(self.track, "velocity_world", None)

        track_id_str = getattr(self.track, "track_id", getattr(self.track, "object_id", self.object_id))

        return PersistentEntity(
            entity_id=ent_id,
            entity_type=self.entity_type,
            semantic_hypotheses=hypotheses,
            pose=pose,
            velocity=vel if vel is not None else np.zeros(3, dtype=np.float64),
            uncertainty=cov[:3, :3] if cov is not None and cov.shape >= (3, 3) else np.eye(3, dtype=np.float64) * 0.1,
            first_seen=first_stamp,
            last_seen=obs_stamp,
            last_observed=obs_stamp,
            visibility_state=vis,
            lifecycle_state=life,
            source_track_ids=[str(track_id_str)],
            parent_context_id=self.spatial_context_id if self.spatial_context_id != "world" else None,
            attributes=dict(self.attributes),
        )

    
    @property
    def is_active(self) -> bool:
        """Returns True if the object is currently active in the scene graph world model."""
        return (
            self.participation == GraphParticipationState.ACTIVE
            and self.track.state in (
                TrackState.ACTIVE,
                TrackState.TEMPORARILY_UNOBSERVED,
            )
        )

    @property
    def is_observed(self) -> bool:
        """Returns True if the node is active and directly observed in the current frame."""
        return self.is_active and self.track.state == TrackState.ACTIVE

