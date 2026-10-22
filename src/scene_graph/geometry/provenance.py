from enum import Enum


class GeometrySource(Enum):
    """Provenance indicator for object geometry."""
    OBSERVED = "observed"
    PREDICTED = "predicted"
    PERSISTED = "persisted"
