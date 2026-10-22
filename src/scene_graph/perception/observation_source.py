from abc import ABC, abstractmethod
from typing import List
from scene_graph.perception.observation import Observation
from scene_graph.data.frame_packet import FramePacket

class ObservationSource(ABC):
    """Interface for retrieving pre-computed observations (e.g. from disk)."""
    
    @abstractmethod
    def observations_for_frame(self, frame_index: int) -> List[Observation]:
        """Return observations for the given frame index."""
        pass

class ObservationProducer(ABC):
    """Interface for actively computing observations (e.g. running a model)."""
    
    @abstractmethod
    def detect(self, packet: FramePacket) -> List[Observation]:
        """Produce observations from the given frame packet."""
        pass
