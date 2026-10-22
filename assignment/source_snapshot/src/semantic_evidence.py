"""Stage 5: Semantic Evidence Extractor for Non-VLM Pre-Routing Pipeline.

Extracts detector semantics and cheap mobility priors:
1. Ingests YOLOE detection labels, confidences, and 2D bounding boxes/masks.
2. Quantifies multi-frame semantic uncertainty using Shannon entropy.
3. Computes label purity and label flip frequency.
4. Generates cheap class priors (M0) without triggering expensive VLM reasoning.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Literal, Optional, Tuple


# Standard deterministic M0 class mobility priors and physical affordances
CLASS_MOBILITY_PRIORS: Dict[str, Tuple[Literal["IMMOVABLE", "MOVABLE", "MOBILE", "UNKNOWN"], float, str]] = {
    # Structural / Architectural fixtures
    "door": ("IMMOVABLE", 0.94, "Architectural boundary hinge fixture"),
    "wall": ("IMMOVABLE", 0.99, "Load-bearing or partition structural wall"),
    "ceiling fan": ("IMMOVABLE", 0.98, "Ceiling electrical junction fixture"),
    "window": ("IMMOVABLE", 0.96, "Building envelope glazing fixture"),
    "floor": ("IMMOVABLE", 0.99, "Horizontal substrate floor"),
    "ceiling": ("IMMOVABLE", 0.99, "Horizontal overhead substrate"),
    "pillar": ("IMMOVABLE", 0.99, "Vertical structural column"),
    # Freestanding movable objects
    "chair": ("MOVABLE", 0.92, "Freestanding task or side seating"),
    "desk": ("MOVABLE", 0.78, "Freestanding laboratory or office workstation table"),
    "table": ("MOVABLE", 0.85, "Freestanding meeting or side table"),
    "laptop": ("MOVABLE", 0.98, "Portable computing hardware"),
    "computer monitor": ("MOVABLE", 0.94, "Workstation display peripheral"),
    "monitor": ("MOVABLE", 0.94, "Workstation display peripheral"),
    "backpack": ("MOVABLE", 0.97, "Personal soft storage luggage"),
    "handbag": ("MOVABLE", 0.97, "Personal soft accessory"),
    "storage box": ("MOVABLE", 0.90, "Stackable plastic/cardboard container"),
    "box": ("MOVABLE", 0.90, "Stackable container"),
    "bottle": ("MOVABLE", 0.96, "Liquid container"),
    "cell phone": ("MOVABLE", 0.98, "Personal handheld device"),
    "cup": ("MOVABLE", 0.96, "Drinkware container"),
    "keyboard": ("MOVABLE", 0.95, "Computer input peripheral"),
    "mouse": ("MOVABLE", 0.95, "Computer pointing device"),
    "pedestal fan": ("MOVABLE", 0.92, "Freestanding floor cooling fan"),
    "fan": ("MOVABLE", 0.88, "Floor or desk fan"),
    # Intrinsically mobile carriers
    "robot": ("MOBILE", 0.96, "Autonomously mobile wheeled robotic platform"),
    "rc car": ("MOBILE", 0.96, "Radio-controlled mobile chassis"),
    "person": ("MOBILE", 0.99, "Dynamic human agent"),
}


@dataclass
class SemanticEvidenceRecord:
    """Timestamp-aligned semantic evidence record."""
    obs_id: str
    frame_index: int
    timestamp: float
    entity_id: str
    track_id: str
    yoloe_class_name: str
    detector_confidence: float
    bbox_area_px: float
    class_prior_mobility: Literal["IMMOVABLE", "MOVABLE", "MOBILE", "UNKNOWN"]
    class_prior_confidence: float
    semantic_entropy: float
    semantic_purity: float
    vote_count_primary: int
    vote_count_total: int
    is_unknown_class: bool
    cheap_prior_description: str
    has_valid_evidence: bool

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class SemanticEvidenceExtractor:
    """Extracts semantic category evidence, multi-frame class entropy, and cheap priors."""

    def __init__(self):
        # Key: entity_id or track_id -> dict of vote counts {class_name: count}
        self._vote_history: Dict[str, Dict[str, int]] = {}

    def extract_observation_semantics(
        self,
        obs: Dict[str, Any],
    ) -> SemanticEvidenceRecord:
        """Extract semantic evidence for a timestamped observation."""
        obs_id = obs.get("obs_id", "")
        frame_idx = int(obs.get("frame_index", 0))
        ts = float(obs.get("timestamp", 0.0))
        entity_id = obs.get("global_entity_id", "")
        track_id = obs.get("local_track_id", "")
        key = entity_id if entity_id else track_id

        raw_class = str(obs.get("class_name", "unknown")).lower().strip()
        confidence = float(obs.get("confidence", 0.50))
        bbox = obs.get("bbox_xyxy", [0, 0, 100, 100])
        area = max(1.0, float((bbox[2] - bbox[0]) * (bbox[3] - bbox[1])))

        # Update vote history
        if key not in self._vote_history:
            self._vote_history[key] = {}
        votes = self._vote_history[key]
        votes[raw_class] = votes.get(raw_class, 0) + 1

        total_votes = sum(votes.values())
        primary_class = max(votes.keys(), key=lambda c: votes[c])
        primary_count = votes[primary_class]

        # Calculate Shannon entropy: H = -sum(p * log2(p))
        entropy = 0.0
        for count in votes.values():
            p = count / total_votes
            if p > 0:
                entropy -= p * math.log2(p)

        # Semantic purity: dominant class vote share
        purity = primary_count / max(1, total_votes)

        # Lookup M0 cheap prior
        prior_mobility, prior_conf, prior_desc = CLASS_MOBILITY_PRIORS.get(
            raw_class,
            ("UNKNOWN", 0.50, f"Unmodeled object class '{raw_class}'")
        )
        is_unknown = (prior_mobility == "UNKNOWN")

        return SemanticEvidenceRecord(
            obs_id=obs_id,
            frame_index=frame_idx,
            timestamp=ts,
            entity_id=entity_id,
            track_id=track_id,
            yoloe_class_name=raw_class,
            detector_confidence=round(confidence, 4),
            bbox_area_px=round(area, 1),
            class_prior_mobility=prior_mobility,
            class_prior_confidence=round(prior_conf, 3),
            semantic_entropy=round(entropy, 4),
            semantic_purity=round(purity, 4),
            vote_count_primary=primary_count,
            vote_count_total=total_votes,
            is_unknown_class=is_unknown,
            cheap_prior_description=prior_desc,
            has_valid_evidence=True,
        )
