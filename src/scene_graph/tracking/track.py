from collections import deque
from dataclasses import dataclass, field
from enum import Enum
import numpy as np

from scene_graph.tracking.state import KalmanState
from scene_graph.ontology.entity import VisibilityState


class TrackState(Enum):
    CANDIDATE = "candidate"
    ACTIVE = "active"
    TEMPORARILY_UNOBSERVED = "temporarily_unobserved"
    LOST = "lost"


@dataclass(eq=False)
class Track:
    object_id: str
    class_name: str
    state: TrackState
    _initial_centroid: np.ndarray
    last_observed_frame: int
    first_observed_frame: int
    observation_count: int
    missing_count: int
    detection_confidence: float
    track_observation_ratio: float
    last_timestamp: float
    recent_observations: deque
    kalman_state: KalmanState | None = None
    size_world: np.ndarray | None = None
    label_belief: dict[str, float] = field(default_factory=dict)
    last_observed_timestamp: float | None = None
    predicted_centroid_world: np.ndarray | None = None
    predicted_covariance_world: np.ndarray | None = None
    is_in_frustum: bool = True
    visibility_state: VisibilityState = VisibilityState.VISIBLE
    coordinate_frame: str = "world"
    centroid_camera: np.ndarray | None = None
    pose_timestamp: float | None = None

    def __post_init__(self):
        if not self.label_belief and self.class_name:
            self.label_belief[self.class_name] = 1.0
        if self.last_observed_timestamp is None:
            self.last_observed_timestamp = self.last_timestamp

    @property
    def primary_label(self) -> str:
        """Top-1 label derived from accumulated label belief."""
        if self.label_belief:
            return max(self.label_belief.items(), key=lambda item: item[1])[0]
        return self.class_name

    def update_label_belief(self, observed_label: str, confidence: float, decay: float = 0.95):
        """Update label belief distribution with exponential decay.
        
        Example: track_0017 semantic: cup: 0.63, mug: 0.31, bottle: 0.06
        """
        total = 0.0
        for k in list(self.label_belief.keys()):
            self.label_belief[k] *= decay
            total += self.label_belief[k]

        weight = max(float(confidence), 0.05)
        self.label_belief[observed_label] = self.label_belief.get(observed_label, 0.0) + weight
        total += weight

        if total > 0:
            for k in list(self.label_belief.keys()):
                self.label_belief[k] /= total

        self.class_name = self.primary_label

    @property
    def centroid_world(self) -> np.ndarray | None:
        if self.coordinate_frame != "world":
            return None
        if self.state == TrackState.TEMPORARILY_UNOBSERVED and self.predicted_centroid_world is not None:
            return self.predicted_centroid_world
        if self.kalman_state is not None:
            return self.kalman_state.position
        return self._initial_centroid

    @property
    def smoothed_position(self) -> np.ndarray | None:
        """Alias for centroid_world (Kalman-filtered 3D position)."""
        return self.centroid_world

    @property
    def velocity_world(self) -> np.ndarray | None:
        if self.coordinate_frame != "world":
            return None
        if self.kalman_state is not None:
            return self.kalman_state.velocity
        return None

    @property
    def position_covariance_world(self) -> np.ndarray | None:
        if self.coordinate_frame != "world":
            return None
        if self.state == TrackState.TEMPORARILY_UNOBSERVED and self.predicted_covariance_world is not None:
            return self.predicted_covariance_world
        if self.kalman_state is not None:
            return self.kalman_state.position_covariance
        return None
    
    # Store the actual object instances (e.g. Observation) over time if needed
    # internally by tracker, bounded by the deque.
    
    def __hash__(self):
        return hash(self.object_id)
        
    def __eq__(self, other):
        if not isinstance(other, Track):
            return False
        return self.object_id == other.object_id
