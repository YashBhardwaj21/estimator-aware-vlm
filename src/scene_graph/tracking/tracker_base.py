from abc import ABC, abstractmethod
from typing import List

from scene_graph.perception.observation import Observation
from scene_graph.tracking.track import Track


class TrackerInterface(ABC):
    
    @abstractmethod
    def update(self, observations: List[Observation], frame_index: int, timestamp: float) -> List[Track]:
        """Update tracker with new observations and return all non-LOST tracks."""
        pass
