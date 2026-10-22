"""relation_state.py: Temporal belief state machine for scene graph relations.

Canonical lifecycle:
    PROPOSED -> ACTIVE -> DECAYING -> OCCLUDED -> UNKNOWN -> TERMINATED
                            \
                       CONTRADICTED

Compatibility aliases (accepted for backward compatibility with older
snapshots; equality comparisons canonicalise them):
    SUPPORTED == ACTIVE
    CONFIRMED == ACTIVE
    HYPOTHESIZED == PROPOSED
    WEAKENING == DECAYING
    ENDED == TERMINATED
"""

from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np

from scene_graph.config import RelationTemporalConfig
from scene_graph.relations.evidence import EvidenceResult, RelationEvidence


RelationKey = Tuple[str, str, str]

# Canonical alias map: maps every state value (including legacy aliases) to its
# canonical equivalent. Shared by RelationState.__hash__ and __eq__.
_RELATION_STATE_CANON: dict = {
    "active": "active",
    "supported": "active",
    "confirmed": "active",
    "proposed": "proposed",
    "hypothesized": "proposed",
    "candidate": "proposed",
    "decaying": "decaying",
    "weakening": "decaying",
    "occluded": "occluded",
    "unknown": "unknown",
    "contradicted": "contradicted",
    "terminated": "terminated",
    "ended": "terminated",
}


class RelationState(str, Enum):
    """Lifecycle states for temporal relation beliefs.

    Canonical states are PROPOSED, ACTIVE, DECAYING, OCCLUDED, UNKNOWN,
    CONTRADICTED, and TERMINATED. Legacy aliases compare equal to their
    canonical equivalents via __eq__.
    """

    # Canonical lifecycle states
    PROPOSED = "proposed"
    ACTIVE = "active"
    DECAYING = "decaying"
    OCCLUDED = "occluded"
    UNKNOWN = "unknown"
    CONTRADICTED = "contradicted"
    TERMINATED = "terminated"

    # Compatibility aliases (kept for backward-compatible deserialisation)
    SUPPORTED = "supported"
    CONFIRMED = "confirmed"
    HYPOTHESIZED = "hypothesized"
    WEAKENING = "weakening"
    ENDED = "ended"

    @classmethod
    def _missing_(cls, value: object) -> Optional["RelationState"]:
        """Case-insensitive lookup so YAML/JSON deserialisation is forgiving."""
        if isinstance(value, str):
            val_lower = value.lower()
            for member in cls:
                if member.value == val_lower or member.name.lower() == val_lower:
                    return member
        return super()._missing_(value)

    def __hash__(self) -> int:
        return hash(_RELATION_STATE_CANON.get(self.value, self.value))

    def __eq__(self, other: Any) -> bool:
        if self is other:
            return True
        my_c = _RELATION_STATE_CANON.get(self.value, self.value)
        if isinstance(other, RelationState):
            return my_c == _RELATION_STATE_CANON.get(other.value, other.value)
        if isinstance(other, str):
            return my_c == _RELATION_STATE_CANON.get(other.lower(), other.lower())
        return super().__eq__(other)


class RelationStateMachine:
    """Temporal belief state machine governing relation edge lifecycle.

    Each relation is keyed by ``(subject_id, object_id, predicate)`` and carries
    a continuous belief score in ``[-1.0, 1.0]``.  Positive evidence increments
    the belief; contradicting evidence decrements it.  Belief decays at
    ``decay_per_second`` when no fresh evidence arrives.

    State transitions follow the canonical :class:`RelationState` lifecycle.
    The machine returns only relations involving *active* object IDs when
    ``active_object_ids`` is provided to :meth:`update`.
    """

    def __init__(self, config: RelationTemporalConfig):
        if config is None:
            raise ValueError("RelationTemporalConfig is required.")

        if config.confirmation_threshold <= 0.0:
            raise ValueError("confirmation_threshold must be positive.")

        if config.contradiction_threshold >= 0.0:
            raise ValueError("contradiction_threshold must be negative.")

        if config.decay_per_second < 0.0:
            raise ValueError("decay_per_second must be non-negative.")

        if config.unknown_after_seconds < 0.0:
            raise ValueError("unknown_after_seconds must be non-negative.")

        if config.lost_after_seconds < 0.0:
            raise ValueError("lost_after_seconds must be non-negative.")

        if config.lost_after_seconds < config.unknown_after_seconds:
            raise ValueError("lost_after_seconds must be greater than or equal to unknown_after_seconds.")

        if getattr(config, "min_confirmation_hits", 1) < 1:
            raise ValueError("min_confirmation_hits must be at least 1.")

        self.config = config
        self.states: Dict[RelationKey, RelationState] = {}
        self.beliefs: Dict[RelationKey, float] = {}
        self.last_update_times: Dict[RelationKey, float] = {}
        self.hit_counts: Dict[RelationKey, int] = {}
        self.miss_counts: Dict[RelationKey, int] = {}
        self.first_observed_times: Dict[RelationKey, float] = {}
        self.last_observed_times: Dict[RelationKey, float] = {}
        self._last_frame_index: Optional[int] = None
        self._last_timestamp: Optional[float] = None

    def _select_evidence(self, evidences: List[RelationEvidence]) -> Dict[RelationKey, RelationEvidence]:
        selected: Dict[RelationKey, RelationEvidence] = {}
        min_conf = float(getattr(self.config, "min_evidence_confidence", 0.0))

        for evidence in evidences:
            if evidence.subject_id == evidence.object_id:
                continue

            if evidence.result not in (EvidenceResult.SUPPORTED, EvidenceResult.CONTRADICTED):
                continue

            confidence = float(evidence.confidence)
            if not np.isfinite(confidence) or confidence < 0.0 or confidence > 1.0:
                raise ValueError(
                    f"Non-finite or out-of-bounds evidence confidence ({confidence}) for relation "
                    f"({evidence.subject_id}, {evidence.predicate}, {evidence.object_id})"
                )

            if confidence < min_conf:
                continue

            key = (evidence.subject_id, evidence.object_id, evidence.predicate)
            current = selected.get(key)

            if current is None:
                selected[key] = evidence
                continue

            current_confidence = float(current.confidence)

            if confidence > current_confidence:
                selected[key] = evidence
            elif confidence == current_confidence and evidence.result == EvidenceResult.SUPPORTED:
                selected[key] = evidence

        return selected

    def _validate_frame(self, frame_index: int, timestamp: float) -> None:
        if frame_index < 0:
            raise ValueError("frame_index must be non-negative.")

        if not np.isfinite(timestamp):
            raise ValueError("timestamp must be finite.")

        if self._last_frame_index is not None and frame_index < self._last_frame_index:
            raise ValueError(
                f"Non-monotonic frame index: {frame_index} < {self._last_frame_index}"
            )

        if self._last_timestamp is not None and timestamp < self._last_timestamp:
            raise ValueError(
                f"Non-monotonic timestamp: {timestamp} < {self._last_timestamp}"
            )

    def _apply_decay(self, key: RelationKey, timestamp: float) -> None:
        last_update = self.last_update_times.get(key)

        if last_update is None:
            self.beliefs.setdefault(key, 0.0)
            return

        dt = timestamp - last_update

        if dt < 0.0:
            raise ValueError(
                f"Non-monotonic timestamp for relation {key}: {timestamp} < {last_update}"
            )

        if dt == 0.0:
            return

        belief = self.beliefs.get(key, 0.0)
        decay = self.config.decay_per_second * dt

        if belief > 0.0:
            belief = max(0.0, belief - decay)
        elif belief < 0.0:
            belief = min(0.0, belief + decay)

        self.beliefs[key] = float(belief)

    @staticmethod
    def _evidence_delta(evidence: RelationEvidence) -> float:
        confidence = float(evidence.confidence)

        if not np.isfinite(confidence):
            raise ValueError(
                f"Non-finite evidence confidence for relation "
                f"({evidence.subject_id}, {evidence.predicate}, {evidence.object_id})"
            )

        confidence = float(np.clip(confidence, 0.0, 1.0))

        if evidence.result == EvidenceResult.SUPPORTED:
            return confidence

        if evidence.result == EvidenceResult.CONTRADICTED:
            return -confidence

        return 0.0

    def _update_state(
        self,
        key: RelationKey,
        evidence: Optional[RelationEvidence] = None,
        timestamp: Optional[float] = None,
    ) -> RelationState:
        belief = self.beliefs.get(key, 0.0)
        prev_state = self.states.get(key)
        min_hits = getattr(self.config, "min_confirmation_hits", 1)

        if evidence is not None:
            if evidence.result == EvidenceResult.CONTRADICTED:
                if belief <= self.config.contradiction_threshold:
                    return RelationState.CONTRADICTED
                if prev_state in (RelationState.ACTIVE, RelationState.SUPPORTED, RelationState.DECAYING, RelationState.WEAKENING):
                    return RelationState.DECAYING
                if belief > 0.0:
                    return RelationState.PROPOSED
                return RelationState.UNKNOWN

            # evidence.result == EvidenceResult.SUPPORTED
            hits = self.hit_counts.get(key, 0)
            if belief >= self.config.confirmation_threshold and hits >= min_hits:
                return RelationState.SUPPORTED
            elif prev_state in (RelationState.ACTIVE, RelationState.SUPPORTED) and belief >= self.config.confirmation_threshold:
                return RelationState.SUPPORTED
            elif prev_state in (RelationState.ACTIVE, RelationState.SUPPORTED, RelationState.DECAYING, RelationState.WEAKENING):
                return RelationState.DECAYING
            elif belief > 0.0:
                return RelationState.PROPOSED
            else:
                return RelationState.UNKNOWN
        else:
            if timestamp is not None:
                last_obs = self.last_observed_times.get(key, self.last_update_times.get(key, timestamp))
                missing_time = timestamp - last_obs

                if missing_time >= self.config.lost_after_seconds:
                    return RelationState.TERMINATED
                if missing_time >= self.config.unknown_after_seconds:
                    return RelationState.UNKNOWN

            if prev_state in (RelationState.ACTIVE, RelationState.SUPPORTED):
                if belief >= self.config.confirmation_threshold:
                    return RelationState.SUPPORTED
                return RelationState.DECAYING
            if prev_state in (RelationState.DECAYING, RelationState.WEAKENING):
                return RelationState.DECAYING
            if prev_state in (RelationState.PROPOSED, RelationState.HYPOTHESIZED):
                if belief > 0.0:
                    return RelationState.PROPOSED
                return RelationState.UNKNOWN

            if belief >= self.config.confirmation_threshold:
                return RelationState.SUPPORTED
            if belief <= self.config.contradiction_threshold:
                return RelationState.CONTRADICTED
            if belief > 0.0:
                return RelationState.PROPOSED
            return RelationState.UNKNOWN

    def _filter_active_relations(
        self,
        relation_states: Dict[RelationKey, RelationState],
        active_object_ids: Optional[Set[str]],
    ) -> Dict[RelationKey, RelationState]:
        if active_object_ids is None:
            return dict(relation_states)

        return {
            key: state
            for key, state in relation_states.items()
            if key[0] in active_object_ids
            and key[1] in active_object_ids
        }

    def update(
        self,
        evidences: List[RelationEvidence],
        frame_index: int,
        timestamp: float,
        active_object_ids: Optional[Set[str]] = None,
    ) -> Dict[RelationKey, RelationState]:

        self._validate_frame(frame_index, timestamp)

        frame_evidence = self._select_evidence(evidences)

        all_keys = sorted(
            set(self.states) | set(frame_evidence),
            key=lambda key: (key[0], key[1], key[2]),
        )

        keys_to_remove: List[RelationKey] = []

        for key in all_keys:
            self._apply_decay(key, timestamp)

            evidence = frame_evidence.get(key)

            if evidence is not None:
                delta = self._evidence_delta(evidence)
                belief = self.beliefs.get(key, 0.0)

                self.beliefs[key] = float(
                    np.clip(
                        belief + delta,
                        -1.0,
                        1.0,
                    )
                )
                self.last_update_times[key] = timestamp
                if key not in self.first_observed_times:
                    self.first_observed_times[key] = timestamp

                if evidence.result == EvidenceResult.SUPPORTED:
                    self.hit_counts[key] = self.hit_counts.get(key, 0) + 1
                    self.last_observed_times[key] = timestamp
                    self.miss_counts[key] = 0
                elif evidence.result == EvidenceResult.CONTRADICTED:
                    self.hit_counts[key] = 0
            else:
                self.hit_counts[key] = 0
                self.miss_counts[key] = self.miss_counts.get(key, 0) + 1

            state = self._update_state(key, evidence=evidence, timestamp=timestamp)

            if state == RelationState.TERMINATED:
                keys_to_remove.append(key)
                continue

            self.states[key] = state

        for key in keys_to_remove:
            self.states.pop(key, None)
            self.beliefs.pop(key, None)
            self.last_update_times.pop(key, None)
            self.hit_counts.pop(key, None)
            self.miss_counts.pop(key, None)
            self.first_observed_times.pop(key, None)
            self.last_observed_times.pop(key, None)

        self._last_frame_index = frame_index
        self._last_timestamp = timestamp

        return self._filter_active_relations(
            self.states,
            active_object_ids,
        )

    def get_belief(self, key: RelationKey) -> float:
        """Return the current continuous belief [-1.0, 1.0] for a relation key."""
        return float(self.beliefs.get(key, 0.0))

    def get_state(self, key: RelationKey) -> RelationState:
        """Return the current temporal relation state for a relation key."""
        return self.states.get(key, RelationState.UNKNOWN)

    def get_hit_count(self, key: RelationKey) -> int:
        """Return consecutive supported observation hit count."""
        return int(self.hit_counts.get(key, 0))

    def get_miss_count(self, key: RelationKey) -> int:
        """Return consecutive missed observation count."""
        return int(self.miss_counts.get(key, 0))

    def is_confirmed(self, key: RelationKey) -> bool:
        """Return True if the relation is confirmed active."""
        return self.states.get(key) in (RelationState.ACTIVE, RelationState.SUPPORTED, RelationState.CONFIRMED)