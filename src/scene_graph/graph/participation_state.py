from enum import Enum

class GraphParticipationState(Enum):
    ABSENT = "absent"
    ACTIVE = "active"
    TEMPORARILY_UNOBSERVED = "temporarily_unobserved"
    REMOVED = "removed"
