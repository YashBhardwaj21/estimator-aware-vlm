from collections import deque
import json
from typing import List, Dict, Optional
from scene_graph.graph.event import GraphEvent, GraphEventType


class GraphHistory:
    """Maintains a chronological bounded log of structural changes to the graph."""
    
    def __init__(self, maxlen: int = 1000, dump_file: Optional[str] = None):
        self.maxlen = int(maxlen)
        self.dump_file = dump_file
        self.events: deque[GraphEvent] = deque(maxlen=self.maxlen)
        self._dump_fp = None
        if self.dump_file is not None:
            self._dump_fp = open(self.dump_file, "a", encoding="utf-8")
        
    def add_event(self, event: GraphEvent):
        """Record a new event into bounded deque and optional disk stream."""
        self.events.append(event)
        if self._dump_fp is not None:
            data = {
                "event_type": event.event_type.value if hasattr(event.event_type, "value") else str(event.event_type),
                "timestamp": float(event.timestamp),
                "frame_index": int(event.frame_index),
                "subject_id": str(event.subject_id),
                "object_id": str(event.object_id) if event.object_id is not None else None,
                "predicate": str(event.predicate) if event.predicate is not None else None,
            }
            self._dump_fp.write(json.dumps(data) + "\n")
            self._dump_fp.flush()
        
    def get_events_for_object(self, object_id: str) -> List[GraphEvent]:
        """Retrieve all events (node or edge) involving a specific object."""
        return [
            e for e in self.events 
            if e.subject_id == object_id or e.object_id == object_id
        ]
        
    def get_events_in_window(self, start_time: float, end_time: float) -> List[GraphEvent]:
        """Retrieve all events that occurred within a time window."""
        return [
            e for e in self.events 
            if start_time <= e.timestamp <= end_time
        ]

    def __len__(self) -> int:
        return len(self.events)

    def close(self):
        if self._dump_fp is not None:
            self._dump_fp.close()
            self._dump_fp = None
