from enum import Enum
from typing import Dict, List

from scene_graph.tracking.track import Track, TrackState


class ObjectState(Enum):
    UNKNOWN = "unknown"
    STABLE = "stable"
    UNSTABLE = "unstable"


class ObjectStateMachine:
    """Maintains causal object lifecycle state from tracker output."""

    def __init__(self):
        self.states: Dict[str, ObjectState] = {}
        self._last_frame_index: int | None = None

    def update(self, tracks: List[Track], frame_index: int) -> Dict[str, ObjectState]:
        if frame_index < 0:
            raise ValueError("frame_index must be non-negative.")

        if self._last_frame_index is not None and frame_index < self._last_frame_index:
            raise ValueError(
                f"Non-monotonic frame index: {frame_index} < {self._last_frame_index}"
            )

        track_lookup = {track.object_id: track for track in tracks}

        for object_id in list(self.states):
            track = track_lookup.get(object_id)

            if track is None:
                self.states[object_id] = ObjectState.UNKNOWN
                continue

            self.states[object_id] = self._state_from_track(track)

        for track in tracks:
            if track.object_id not in self.states:
                self.states[track.object_id] = self._state_from_track(track)

        self._last_frame_index = frame_index

        return dict(self.states)

    @staticmethod
    def _state_from_track(track: Track) -> ObjectState:
        if track.state == TrackState.ACTIVE:
            return ObjectState.STABLE

        if track.state == TrackState.TEMPORARILY_UNOBSERVED:
            return ObjectState.UNSTABLE

        if track.state in (TrackState.CANDIDATE, TrackState.LOST):
            return ObjectState.UNKNOWN

        raise ValueError(f"Unsupported track state: {track.state}")