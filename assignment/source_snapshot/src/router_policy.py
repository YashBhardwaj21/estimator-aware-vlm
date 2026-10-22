"""Stage 6: Pure Deterministic Router Policy for Estimator-Aware VLM Querying.

This module contains pure, deterministic decision logic. It accepts structured
evidence and returns a decision plus machine-readable reasons without invoking
a VLM, accessing files, or altering state estimators.

Gates:
  Gate 1 (Estimator & Evidence Validity): Check coordinate frames, depth, and pose.
  Gate 2 (Visual Context Utility): Check image availability if escalation is needed.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Optional


DecisionType = Literal["RESOLVE_CHEAP", "ESCALATE", "DEFER", "ABSTAIN"]

CANONICAL_IMMOVABLE = {
    "door", "wall", "ceiling fan", "window", "floor", "ceiling", "pillar"
}

CANONICAL_MOBILE = {
    "person", "robot", "drone", "quadcopter drone", "human", "operator", "pet"
}

CANONICAL_MOVABLE = {
    "chair", "desk", "table", "laptop", "computer monitor", "monitor",
    "backpack", "handbag", "storage box", "keyboard", "mouse", "computer mouse",
    "bottle", "cup", "trash can", "toolbox"
}

# Ambiguous categories that require visual affordance verification (e.g. mounted vs freestanding)
AMBIGUOUS_AFFORDANCE_CLASSES = {
    "pedestal fan", "fan", "whiteboard", "projector", "screen", "shelf", "cabinet"
}

MIN_SIGNIFICANT_DISPLACEMENT_M = 0.25
ENTROPY_AMBIGUITY_THRESHOLD = 0.35


@dataclass
class RouterDecision:
    """Outcome contract for an evaluated evidence record."""
    entity_id: str
    decision: DecisionType
    reason_codes: List[str]
    vlm_required: bool
    cheap_label: Optional[str] = None
    confidence: float = 1.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "entity_id": self.entity_id,
            "decision": self.decision,
            "reason_codes": self.reason_codes,
            "vlm_required": self.vlm_required,
            "cheap_label": self.cheap_label,
            "confidence": round(self.confidence, 4),
        }


def evaluate_router_policy(evidence: Dict[str, Any]) -> RouterDecision:
    """Evaluate pure deterministic routing policy on an aggregated entity evidence record."""
    entity_id = str(evidence.get("entity_id", "unknown"))
    sem_class = str(evidence.get("semantic_class", "object")).lower().strip()
    prior_mobility = evidence.get("class_prior_mobility")
    observed_motion = str(evidence.get("observed_motion", "STATIC")).upper()
    world_disp = evidence.get("world_displacement_m")
    disp_sigma = evidence.get("displacement_uncertainty_m", 0.04) or 0.04
    semantic_entropy = float(evidence.get("semantic_entropy", 0.0) or 0.0)

    # Pose and depth validity checks
    has_valid_pose = evidence.get("has_valid_pose")
    has_valid_depth = evidence.get("depth_valid")
    has_images = bool(evidence.get("has_images", False))
    views_count = int(evidence.get("views_count", 0))

    # -------------------------------------------------------------
    # GATE 1: Estimator & Evidence Validity Check
    # -------------------------------------------------------------
    if has_valid_pose is False:
        return RouterDecision(
            entity_id=entity_id,
            decision="DEFER",
            reason_codes=["UNTRUSTWORTHY_ESTIMATOR_POSE"],
            vlm_required=False,
            cheap_label=None,
            confidence=0.0,
        )

    if has_valid_depth is False and observed_motion in ["MOVING", "DISPLACED"]:
        return RouterDecision(
            entity_id=entity_id,
            decision="DEFER",
            reason_codes=["INVALID_DEPTH_ON_MOVING_TARGET"],
            vlm_required=False,
            cheap_label=None,
            confidence=0.0,
        )

    reasons: List[str] = []

    # -------------------------------------------------------------
    # Rule A: Canonical Mobile Agents
    # -------------------------------------------------------------
    if sem_class in CANONICAL_MOBILE or prior_mobility == "MOBILE":
        return RouterDecision(
            entity_id=entity_id,
            decision="RESOLVE_CHEAP",
            reason_codes=["CANONICAL_MOBILE_AGENT_PRIOR"],
            vlm_required=False,
            cheap_label="MOBILE",
            confidence=0.98,
        )

    # -------------------------------------------------------------
    # Rule B: Check for Significant Physical Relocation / Motion
    # -------------------------------------------------------------
    if world_disp is not None:
        is_significantly_displaced = (
            world_disp >= MIN_SIGNIFICANT_DISPLACEMENT_M
            and world_disp >= 3.0 * max(0.02, disp_sigma)
            and observed_motion in ["DISPLACED", "MOVING"]
        )
        if is_significantly_displaced:
            reasons.append("SIGNIFICANT_DISPLACEMENT_AFFORDANCE_UNRESOLVED")
    elif observed_motion in ["DISPLACED", "MOVING"]:
        reasons.append("SIGNIFICANT_DISPLACEMENT_AFFORDANCE_UNRESOLVED")

    # -------------------------------------------------------------
    # Rule C: Ambiguous Physical Affordance / Mounting Support
    # -------------------------------------------------------------
    if sem_class in AMBIGUOUS_AFFORDANCE_CLASSES:
        reasons.append("AMBIGUOUS_PHYSICAL_SUPPORT_AFFORDANCE")

    # -------------------------------------------------------------
    # Rule D: High Semantic Detector Ambiguity
    # -------------------------------------------------------------
    if semantic_entropy >= ENTROPY_AMBIGUITY_THRESHOLD:
        reasons.append("HIGH_SEMANTIC_AMBIGUITY")

    # -------------------------------------------------------------
    # GATE 2: Decision Routing
    # -------------------------------------------------------------
    if reasons:
        # Escalation is requested. Check Gate 2 (Visual context availability).
        if not has_images or views_count <= 0:
            return RouterDecision(
                entity_id=entity_id,
                decision="DEFER",
                reason_codes=reasons + ["MISSING_REQUIRED_VISUAL_CONTEXT"],
                vlm_required=False,
                cheap_label=None,
                confidence=0.30,
            )
        return RouterDecision(
            entity_id=entity_id,
            decision="ESCALATE",
            reason_codes=reasons,
            vlm_required=True,
            cheap_label=None,
            confidence=0.90,
        )

    # -------------------------------------------------------------
    # Rule E: Canonical Immovable Architectural Elements (Stationary)
    # -------------------------------------------------------------
    if sem_class in CANONICAL_IMMOVABLE or prior_mobility == "IMMOVABLE":
        return RouterDecision(
            entity_id=entity_id,
            decision="RESOLVE_CHEAP",
            reason_codes=["CANONICAL_IMMOVABLE_ARCHITECTURAL_PRIOR"],
            vlm_required=False,
            cheap_label="IMMOVABLE",
            confidence=0.99,
        )

    # -------------------------------------------------------------
    # Rule F: Canonical Movable Furniture (Stationary & Stable)
    # -------------------------------------------------------------
    if sem_class in CANONICAL_MOVABLE or prior_mobility == "MOVABLE":
        return RouterDecision(
            entity_id=entity_id,
            decision="RESOLVE_CHEAP",
            reason_codes=["STATIC_CANONICAL_MOVABLE_PRIOR"],
            vlm_required=False,
            cheap_label="MOVABLE",
            confidence=0.95,
        )

    # Fallback: Unknown category
    if has_images and views_count > 0:
        return RouterDecision(
            entity_id=entity_id,
            decision="ESCALATE",
            reason_codes=["UNKNOWN_CATEGORY_ESCALATION"],
            vlm_required=True,
            cheap_label=None,
            confidence=0.50,
        )

    return RouterDecision(
        entity_id=entity_id,
        decision="ABSTAIN",
        reason_codes=["UNKNOWN_CATEGORY_NO_IMAGES"],
        vlm_required=False,
        cheap_label="UNKNOWN",
        confidence=0.0,
    )
