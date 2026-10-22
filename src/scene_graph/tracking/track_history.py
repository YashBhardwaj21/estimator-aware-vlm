from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Any


@dataclass
class TrackStateTransition:
    frame_index: int
    from_state: str
    to_state: str
    reason: str


class TrackHistory:
    """External store for tracking history.
    
    Used when config.experiment.save_intermediates == True.
    Stores full observations and state transitions for all tracks.
    """
    
    def __init__(self):
        self.transitions: Dict[str, List[TrackStateTransition]] = defaultdict(list)
        self.observations: Dict[str, List[Any]] = defaultdict(list)
        
    def record_transition(self, track_id: str, frame_index: int, from_state: str, to_state: str, reason: str):
        self.transitions[track_id].append(
            TrackStateTransition(frame_index, from_state, to_state, reason)
        )
        
    def record_observation(self, track_id: str, observation: Any):
        self.observations[track_id].append(observation)
