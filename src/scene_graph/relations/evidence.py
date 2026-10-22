from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, Optional

from scene_graph.geometry.reference_frame import ReferenceFrameType


class EvidenceResult(Enum):
    SUPPORTED = "supported"
    CONTRADICTED = "contradicted"
    NOT_APPLICABLE = "not_applicable"
    INSUFFICIENT_DEPTH = "insufficient_depth"
    INSUFFICIENT_GEOMETRY = "insufficient_geometry"
    MISSING_MASK = "missing_mask"


@dataclass(frozen=True)
class RelationEvidence:
    predicate: str
    subject_id: str
    object_id: str
    frame_index: int
    timestamp: float
    result: EvidenceResult
    value: Optional[float]
    threshold: Optional[float]
    confidence: float
    reference_frame: ReferenceFrameType
    evidence_type: str
    details: Dict[str, Any]

    @property
    def evidence_strength(self) -> float:
        """Alias for confidence, expressing strength of evidence in [0.0, 1.0]."""
        return self.confidence

    def __post_init__(self):
        if not isinstance(self.reference_frame, ReferenceFrameType):
            raise TypeError(
                f"RelationEvidence.reference_frame must be ReferenceFrameType enum, got {type(self.reference_frame)}: {self.reference_frame!r}"
            )
        if not isinstance(self.result, EvidenceResult):
            raise TypeError(
                f"RelationEvidence.result must be EvidenceResult enum, got {type(self.result)}: {self.result!r}"
            )
        try:
            val = float(self.confidence)
        except (ValueError, TypeError):
            raise ValueError(f"RelationEvidence.confidence must be a float, got {self.confidence!r}")

        import math
        if not math.isfinite(val) or not (0.0 <= val <= 1.0):
            raise ValueError(
                f"RelationEvidence.confidence must be finite and in [0.0, 1.0], got {self.confidence}"
            )