"""Semantic Evidence Extractor for Non-VLM Pre-Routing Pipeline.

Extracts detector semantics and cheap mobility priors honestly:
1. Ingests YOLOE detection labels, confidences, and 2D bounding boxes without synthetic defaults.
2. Quantifies multi-frame semantic uncertainty using online Shannon entropy.
3. Computes label purity and dominant category shares.
4. Distinguishes detector validity from mobility prior coverage (zero fabricated priors).
5. Enforces session-qualified track keys to prevent cross-session voting collisions.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
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
    "projector": ("IMMOVABLE", 0.95, "Overhead ceiling or wall mounted projection fixture"),
    "whiteboard": ("IMMOVABLE", 0.94, "Wall-mounted dry-erase communication board"),
    "toolbox": ("MOVABLE", 0.91, "Portable hardware tool storage container"),
    "computer mouse": ("MOVABLE", 0.95, "Computer pointing device peripheral"),
    # Intrinsically mobile carriers
    "robot": ("MOBILE", 0.96, "Autonomously mobile wheeled robotic platform"),
    "rc car": ("MOBILE", 0.96, "Radio-controlled mobile chassis"),
    "quadcopter drone": ("MOBILE", 0.96, "Aerial autonomous or teleoperated robotic platform"),
    "person": ("MOBILE", 0.99, "Dynamic human agent"),
}


@dataclass
class SemanticEvidenceRecord:
    """Timestamp-aligned semantic evidence record with honest missing field representation."""
    obs_id: str
    frame_index: int
    timestamp: float
    entity_id: str
    track_id: str
    source_session: str
    raw_class_name: str
    yoloe_class_name: str
    detector_confidence: Optional[float]
    bbox_area_px: Optional[float]
    class_prior_mobility: Literal["IMMOVABLE", "MOVABLE", "MOBILE", "UNKNOWN"]
    class_prior_confidence: float
    semantic_entropy: float
    semantic_purity: float
    vote_count_primary: int
    vote_count_total: int
    is_unknown_class: bool
    detection_valid: bool
    class_known_to_prior: bool
    cheap_prior_description: str
    has_valid_evidence: bool
    semantic_diagnostics: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class SemanticEvidenceExtractor:
    """Extracts semantic category evidence, multi-frame class entropy, and cheap priors honestly."""

    def __init__(self):
        # Key: (session, track_id) -> dict of online vote counts {class_name: count}
        self._vote_history: Dict[Tuple[str, str], Dict[str, int]] = {}

    def extract_observation_semantics(
        self,
        obs: Dict[str, Any],
    ) -> SemanticEvidenceRecord:
        """Extract semantic evidence for a timestamped observation without fabricated defaults."""
        obs_id = str(obs.get("obs_id", "")).strip()
        frame_idx = int(obs.get("frame_index", 0))
        ts = float(obs.get("timestamp", 0.0))
        entity_id = str(obs.get("global_entity_id", "")).strip()
        track_id = str(obs.get("local_track_id", "")).strip()
        session = str(obs.get("source_session", "")).strip()

        # Session-qualified track key (never bare track_id)
        sess_key = session if session else "__unspecified_session__"
        if track_id:
            track_key = (sess_key, track_id)
        elif entity_id:
            track_key = (sess_key, entity_id)
        else:
            # Isolated key for un-tracked observations
            track_key = (sess_key, f"__isolated_{obs_id}__")

        diagnostics: List[str] = []

        # 1. Raw class name extraction & normalization
        raw_c = obs.get("class_name")
        if raw_c is not None and isinstance(raw_c, str) and raw_c.strip():
            raw_class = raw_c.strip().lower()
        else:
            raw_class = "unknown"
            diagnostics.append("Missing or empty class_name in observation.")

        # 2. Honest detector confidence validation (no 0.50 fallback)
        raw_conf = obs.get("confidence")
        detector_conf: Optional[float] = None
        if raw_conf is not None and isinstance(raw_conf, (int, float)) and math.isfinite(raw_conf):
            val_c = float(raw_conf)
            if 0.0 <= val_c <= 1.0:
                detector_conf = val_c
            else:
                diagnostics.append(f"Confidence {val_c} out of valid probability range [0.0, 1.0].")
        else:
            diagnostics.append("Missing or non-finite detector confidence.")

        # 3. Honest bounding box validation (no [0, 0, 100, 100] fallback)
        bbox = obs.get("bbox_xyxy")
        bbox_area: Optional[float] = None
        bbox_valid = False
        if bbox is not None and isinstance(bbox, (list, tuple)) and len(bbox) == 4:
            if all(isinstance(v, (int, float)) and math.isfinite(v) for v in bbox):
                x1, y1, x2, y2 = bbox
                if x1 < x2 and y1 < y2:
                    bbox_area = float((x2 - x1) * (y2 - y1))
                    bbox_valid = True
                else:
                    diagnostics.append(f"Inverted bounding box dimensions: [{x1}, {y1}, {x2}, {y2}].")
            else:
                diagnostics.append("Non-finite coordinates in bbox_xyxy.")
        else:
            diagnostics.append("Missing or malformed bbox_xyxy.")

        # 4. Multi-Frame Online Category Voting
        if track_key not in self._vote_history:
            self._vote_history[track_key] = {}
        votes = self._vote_history[track_key]

        # Only vote for recognized, non-empty classes
        if raw_class != "unknown":
            votes[raw_class] = votes.get(raw_class, 0) + 1

        total_votes = sum(votes.values())
        if total_votes > 0:
            primary_class = max(votes.keys(), key=lambda c: votes[c])
            primary_count = votes[primary_class]
            # Shannon entropy: H = -sum(p * log2(p))
            entropy = 0.0
            for count in votes.values():
                p = count / total_votes
                if p > 0:
                    entropy -= p * math.log2(p)
            purity = primary_count / total_votes
        else:
            primary_class = raw_class
            primary_count = 0
            entropy = 0.0
            purity = 0.0

        # 5. M0 Mobility Prior Lookup
        prior_lookup = CLASS_MOBILITY_PRIORS.get(raw_class)
        if prior_lookup is not None:
            prior_mobility, prior_conf, prior_desc = prior_lookup
            class_known = True
        else:
            prior_mobility = "UNKNOWN"
            prior_conf = 0.0
            prior_desc = f"Unmodeled object class '{raw_class}'"
            class_known = False

        is_unknown = (prior_mobility == "UNKNOWN")
        detection_valid = bool(raw_class != "unknown" and detector_conf is not None)
        has_valid = bool(detection_valid and (bbox_valid or detector_conf >= 0.25))

        return SemanticEvidenceRecord(
            obs_id=obs_id,
            frame_index=frame_idx,
            timestamp=ts,
            entity_id=entity_id,
            track_id=track_id,
            source_session=session,
            raw_class_name=raw_class,
            yoloe_class_name=raw_class,
            detector_confidence=round(detector_conf, 4) if detector_conf is not None else None,
            bbox_area_px=round(bbox_area, 1) if bbox_area is not None else None,
            class_prior_mobility=prior_mobility,
            class_prior_confidence=round(prior_conf, 3),
            semantic_entropy=round(entropy, 4),
            semantic_purity=round(purity, 4),
            vote_count_primary=primary_count,
            vote_count_total=total_votes,
            is_unknown_class=is_unknown,
            detection_valid=detection_valid,
            class_known_to_prior=class_known,
            cheap_prior_description=prior_desc,
            has_valid_evidence=has_valid,
            semantic_diagnostics=diagnostics,
        )
