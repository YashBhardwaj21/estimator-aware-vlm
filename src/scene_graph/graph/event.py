from enum import Enum
from dataclasses import dataclass, field
from typing import Dict, Any, Optional

class GraphEventType(Enum):
    NODE_ADDED = "node_added"
    NODE_REMOVED = "node_removed"
    EDGE_ADDED = "edge_added"
    EDGE_REMOVED = "edge_removed"

@dataclass
class GraphEvent:
    """Represents a discrete structural change in the scene graph over time."""
    event_type: GraphEventType
    timestamp: float
    frame_index: int
    subject_id: str
    object_id: Optional[str] = None
    predicate: Optional[str] = None
    attributes: Dict[str, Any] = field(default_factory=dict)
