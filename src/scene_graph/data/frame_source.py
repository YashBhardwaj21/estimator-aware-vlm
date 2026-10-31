from abc import ABC, abstractmethod
from typing import Iterator

from scene_graph.data.frame_packet import FramePacket

class FrameSource(ABC):
    """Abstract base class for all frame sources (Replay or Live)."""
    
    @abstractmethod
    def __iter__(self) -> Iterator[FramePacket]:
        """Yield frame packets one at a time."""
        pass
