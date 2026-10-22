from abc import ABC, abstractmethod
from abc import ABC, abstractmethod
from typing import List, Tuple

from scene_graph.tracking.track import Track
from scene_graph.relations.context import FrameContext
from scene_graph.relations.evidence import RelationEvidence


class RelationModule(ABC):
    
    @abstractmethod
    def predicates(self) -> List[str]:
        """Return the list of predicate names this module computes (e.g., ['NEAR', 'FAR'])."""
        pass
        
    def compute_pairs(self, pairs: List[Tuple[Track, Track]], context: FrameContext) -> List[RelationEvidence]:
        """Compute relation evidence for all given pairs. Default implementation iterates."""
        evidences = []
        for subj, obj in pairs:
            evidences.extend(self.compute(subj, obj, context))
        return evidences
        
    @abstractmethod
    def compute(self, subject: Track, object: Track, context: FrameContext) -> List[RelationEvidence]:
        """Compute relation evidence for the given subject and object."""
        pass
