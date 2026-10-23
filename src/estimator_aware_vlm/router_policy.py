"""Estimator-Aware & Baseline Router Policies.

Implements five distinct, reproducible routing policies for empirical comparison:
- R0: Semantic-class prior lookup only (zero VLM queries).
- R1: Geometry and observed motion only without VLM (zero VLM queries).
- R2: Non-estimator-aware selective routing (ambiguity-driven, blind to SLAM influence).
- R3: Estimator-aware selective routing (explicit SLAM loss reduction utility vs VLM cost).
- R4: All-eligible VLM reference (unguided contextual reasoning ceiling).

Core formulation for R3 (Estimator-Aware):
    query_utility = delta_L_SLAM - lambda_cost * C_vlm
where delta_L_SLAM quantifies the expected SLAM trajectory loss reduction if an ambiguous
or displaced object's true affordance is resolved to prevent map corruption and false loop closures.
"""

from __future__ import annotations

import math
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

# Ambiguous categories requiring visual affordance verification (mounted vs freestanding)
AMBIGUOUS_AFFORDANCE_CLASSES = {
    "pedestal fan", "fan", "whiteboard", "projector", "screen", "shelf", "cabinet"
}

MIN_SIGNIFICANT_DISPLACEMENT_M = 0.25
ENTROPY_AMBIGUITY_THRESHOLD = 0.35
DEFAULT_VLM_COST = 0.50


def extract_displacement_evidence(evidence: Dict[str, Any]) -> Optional[float]:
    """Safely extract measured or world displacement in meters.

    Supports both 'world_displacement_m' and 'measured_displacement_m',
    correctly handling 0.0, None, and numeric strings.
    """
    disp = evidence.get("world_displacement_m")
    if disp is None:
        disp = evidence.get("measured_displacement_m")
    if disp is not None:
        try:
            return float(disp)
        except (ValueError, TypeError):
            return None
    return None


@dataclass
class RouterDecision:
    """Outcome contract for an evaluated evidence record."""
    entity_id: str
    decision: DecisionType
    reason_codes: List[str]
    vlm_required: bool
    cheap_label: Optional[str] = None
    confidence: float = 1.0
    query_utility: Optional[float] = None
    estimator_influence: Optional[float] = None
    estimator_provenance: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "entity_id": self.entity_id,
            "decision": self.decision,
            "reason_codes": self.reason_codes,
            "vlm_required": self.vlm_required,
            "cheap_label": self.cheap_label,
            "confidence": round(self.confidence, 4),
            "query_utility": round(self.query_utility, 4) if self.query_utility is not None else None,
            "estimator_influence": round(self.estimator_influence, 4) if self.estimator_influence is not None else None,
            "estimator_provenance": self.estimator_provenance,
        }


def compute_estimator_influence_with_provenance(evidence: Dict[str, Any]) -> Tuple[float, str]:
    """Compute normalized SLAM landmark influence score in [0.0, 1.0] and its provenance.

    Evaluates whether the landmark observation physically and geometrically constrains
    the state estimator (e.g. optimizer inlier support, residuals, depth uncertainty,
    observation count, pose validity, and displacement state).

    Distinguishes:
    - 'MEASURED_BACKEND': low-level SLAM optimizer residuals and inlier counts exist.
    - 'PROXY_SUPPORT': heuristic fallback based on observation frequency and consistency.
    """
    has_valid_pose = evidence.get("has_valid_pose", True)
    if not has_valid_pose or evidence.get("pose_valid") is False:
        return 0.0, "INVALID_ESTIMATOR_POSE"

    if evidence.get("depth_valid") is False:
        return 0.0, "INVALID_DEPTH_ESTIMATOR_UNCONSTRAINED"

    # Check for direct measured SLAM backend telemetry
    measured_inliers = evidence.get("inlier_count")
    if measured_inliers is None:
        measured_inliers = evidence.get("mappoints_inlier_count")
    inlier_ratio = evidence.get("inlier_ratio")
    if inlier_ratio is None:
        inlier_ratio = evidence.get("point_to_bbox_inlier_ratio")
    mean_residual = evidence.get("mean_reprojection_residual_px")
    if mean_residual is None:
        mean_residual = evidence.get("mean_residual_px")

    s1_cnt = int(evidence.get("s1_obs_count", 0))
    s2_cnt = int(evidence.get("s2_obs_count", 0))
    total_obs = s1_cnt + s2_cnt
    if total_obs == 0:
        total_obs = int(evidence.get("total_observations", evidence.get("observation_count", 0)))

    if total_obs > 0:
        obs_factor = min(1.0, total_obs / 10.0)
    else:
        views_cnt = int(evidence.get("views_count", 1))
        obs_factor = min(1.0, max(0.70, views_cnt / 2.0))

    # If backend optimizer telemetry exists:
    if inlier_ratio is not None or measured_inliers is not None:
        # Zero inliers means the landmark provides ZERO constraint to the optimizer
        if (measured_inliers is not None and measured_inliers == 0) or (inlier_ratio is not None and inlier_ratio == 0.0):
            return 0.0, "MEASURED_BACKEND_ZERO_INLIERS"

        ratio_val = float(inlier_ratio) if inlier_ratio is not None else min(1.0, float(measured_inliers or 0) / 20.0)
        # Residual penalty: large residuals degrade metric trust
        res_factor = 1.0
        if mean_residual is not None and mean_residual > 20.0:
            res_factor = max(0.0, 1.0 - (float(mean_residual) - 20.0) / 80.0)

        world_disp = extract_displacement_evidence(evidence)
        disp_factor = 1.25 if (world_disp is not None and world_disp > 0.25) else 1.0

        influence = min(1.0, obs_factor * ratio_val * res_factor * disp_factor)
        return round(influence, 4), "MEASURED_BACKEND"

    # Fallback / Proxy Support
    geo_score = evidence.get("geometric_consistency_score")
    if geo_score is None:
        geo_factor = 0.85 if evidence.get("depth_valid", True) else 0.0
    else:
        geo_factor = max(0.0, min(1.0, float(geo_score)))  # Strict: no artificial 0.2 floor

    world_disp = extract_displacement_evidence(evidence)
    disp_factor = 1.25 if (world_disp is not None and world_disp > 0.25) else 1.0

    influence = min(1.0, obs_factor * geo_factor * disp_factor)
    return round(influence, 4), "PROXY_SUPPORT"


def compute_estimator_influence(evidence: Dict[str, Any]) -> float:
    """Compute normalized SLAM landmark influence score in [0.0, 1.0]."""
    score, _ = compute_estimator_influence_with_provenance(evidence)
    return score


def evaluate_policy_r0(evidence: Dict[str, Any]) -> RouterDecision:
    """R0: Semantic-Class Prior Lookup Only (Zero VLM queries)."""
    entity_id = str(evidence.get("entity_id", "unknown"))
    sem_class = str(evidence.get("semantic_class", "object")).lower().strip()
    prior = evidence.get("class_prior_mobility")

    if sem_class in CANONICAL_MOBILE or prior == "MOBILE":
        return RouterDecision(
            entity_id=entity_id,
            decision="RESOLVE_CHEAP",
            reason_codes=["R0_PRIOR_CANONICAL_MOBILE"],
            vlm_required=False,
            cheap_label="MOBILE",
            confidence=0.98,
        )
    if sem_class in CANONICAL_IMMOVABLE or prior == "IMMOVABLE":
        return RouterDecision(
            entity_id=entity_id,
            decision="RESOLVE_CHEAP",
            reason_codes=["R0_PRIOR_CANONICAL_IMMOVABLE"],
            vlm_required=False,
            cheap_label="IMMOVABLE",
            confidence=0.99,
        )
    if sem_class in CANONICAL_MOVABLE or prior == "MOVABLE":
        return RouterDecision(
            entity_id=entity_id,
            decision="RESOLVE_CHEAP",
            reason_codes=["R0_PRIOR_CANONICAL_MOVABLE"],
            vlm_required=False,
            cheap_label="MOVABLE",
            confidence=0.95,
        )
    return RouterDecision(
        entity_id=entity_id,
        decision="ABSTAIN",
        reason_codes=["R0_PRIOR_UNKNOWN_CATEGORY"],
        vlm_required=False,
        cheap_label="UNKNOWN",
        confidence=0.0,
    )


def evaluate_policy_r1(evidence: Dict[str, Any]) -> RouterDecision:
    """R1: Geometry & Temporal Motion Only (Zero VLM queries)."""
    entity_id = str(evidence.get("entity_id", "unknown"))
    sem_class = str(evidence.get("semantic_class", "object")).lower().strip()
    observed_motion = str(evidence.get("observed_motion", "STATIC")).upper()
    world_disp = extract_displacement_evidence(evidence)

    # If observed moving, classify passively movable objects as MOVABLE and agents as MOBILE
    if observed_motion in ["MOVING", "DISPLACED"] or (world_disp and world_disp > 0.25):
        if sem_class in CANONICAL_MOBILE:
            label = "MOBILE"
        else:
            label = "MOVABLE"
        return RouterDecision(
            entity_id=entity_id,
            decision="RESOLVE_CHEAP",
            reason_codes=["R1_GEOMETRY_MOTION_DISPLACED"],
            vlm_required=False,
            cheap_label=label,
            confidence=0.90,
        )

    # Static: fallback to prior
    return evaluate_policy_r0(evidence)


def evaluate_policy_r2(evidence: Dict[str, Any]) -> RouterDecision:
    """R2: Non-Estimator-Aware Selective Routing (Ambiguity-Driven)."""
    entity_id = str(evidence.get("entity_id", "unknown"))
    sem_class = str(evidence.get("semantic_class", "object")).lower().strip()
    prior_mobility = evidence.get("class_prior_mobility")
    observed_motion = str(evidence.get("observed_motion", "STATIC")).upper()
    world_disp = extract_displacement_evidence(evidence)
    disp_sigma = evidence.get("displacement_uncertainty_m", 0.04) or 0.04
    semantic_entropy = float(evidence.get("semantic_entropy", 0.0) or 0.0)
    has_valid_pose = evidence.get("has_valid_pose")
    has_valid_depth = evidence.get("depth_valid")
    has_images = bool(evidence.get("has_images", False))
    views_count = int(evidence.get("views_count", 0))

    # Gate 1: Estimator & Evidence Validity
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

    # Mobile agents resolved cheaply
    if sem_class in CANONICAL_MOBILE or prior_mobility == "MOBILE":
        return RouterDecision(
            entity_id=entity_id,
            decision="RESOLVE_CHEAP",
            reason_codes=["CANONICAL_MOBILE_AGENT_PRIOR"],
            vlm_required=False,
            cheap_label="MOBILE",
            confidence=0.98,
        )

    reasons: List[str] = []

    # Significant displacement
    if world_disp is not None:
        is_displaced = (
            world_disp >= MIN_SIGNIFICANT_DISPLACEMENT_M
            and world_disp >= 3.0 * max(0.02, disp_sigma)
            and observed_motion in ["DISPLACED", "MOVING"]
        )
        if is_displaced:
            reasons.append("SIGNIFICANT_DISPLACEMENT_AFFORDANCE_UNRESOLVED")
    elif observed_motion in ["DISPLACED", "MOVING"]:
        reasons.append("SIGNIFICANT_DISPLACEMENT_AFFORDANCE_UNRESOLVED")

    # Ambiguous mounting class
    if sem_class in AMBIGUOUS_AFFORDANCE_CLASSES:
        reasons.append("AMBIGUOUS_PHYSICAL_SUPPORT_AFFORDANCE")

    # High semantic entropy
    if semantic_entropy >= ENTROPY_AMBIGUITY_THRESHOLD:
        reasons.append("HIGH_SEMANTIC_AMBIGUITY")

    if reasons:
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

    # Stationary immovable
    if sem_class in CANONICAL_IMMOVABLE or prior_mobility == "IMMOVABLE":
        return RouterDecision(
            entity_id=entity_id,
            decision="RESOLVE_CHEAP",
            reason_codes=["CANONICAL_IMMOVABLE_ARCHITECTURAL_PRIOR"],
            vlm_required=False,
            cheap_label="IMMOVABLE",
            confidence=0.99,
        )

    # Stationary movable
    if sem_class in CANONICAL_MOVABLE or prior_mobility == "MOVABLE":
        return RouterDecision(
            entity_id=entity_id,
            decision="RESOLVE_CHEAP",
            reason_codes=["STATIC_CANONICAL_MOVABLE_PRIOR"],
            vlm_required=False,
            cheap_label="MOVABLE",
            confidence=0.95,
        )

    return evaluate_policy_r0(evidence)


def evaluate_policy_r3(
    evidence: Dict[str, Any],
    lambda_cost: float = DEFAULT_VLM_COST,
    utility_threshold: float = 0.0,
) -> RouterDecision:
    """R3: Estimator-Aware Selective Routing.

    Evaluates:
        query_utility = delta_L_SLAM - lambda_cost * C_VLM
    where delta_L_SLAM quantifies expected SLAM trajectory/map loss reduction if an
    ambiguous or displaced object's true affordance is resolved to prevent map corruption.

    Operational Decision Rule:
        If query_utility >= utility_threshold:
            ESCALATE to VLM (expected SLAM loss reduction justifies inference compute).
        Else:
            RESOLVE_CHEAP (low estimator relevance or sub-threshold utility; query avoided).
    """
    entity_id = str(evidence.get("entity_id", "unknown"))
    sem_class = str(evidence.get("semantic_class", "object")).lower().strip()
    prior_mobility = evidence.get("class_prior_mobility")
    observed_motion = str(evidence.get("observed_motion", "STATIC")).upper()
    world_disp = extract_displacement_evidence(evidence)
    disp_sigma = evidence.get("displacement_uncertainty_m", 0.04) or 0.04
    semantic_entropy = float(evidence.get("semantic_entropy", 0.0) or 0.0)

    has_valid_pose = evidence.get("has_valid_pose")
    has_valid_depth = evidence.get("depth_valid")
    has_images = bool(evidence.get("has_images", False))
    views_count = int(evidence.get("views_count", 0))

    # Gate 1: Estimator & Evidence Validity
    if has_valid_pose is False:
        return RouterDecision(
            entity_id=entity_id,
            decision="DEFER",
            reason_codes=["UNTRUSTWORTHY_ESTIMATOR_POSE"],
            vlm_required=False,
            cheap_label=None,
            confidence=0.0,
            query_utility=-1.0,
            estimator_influence=0.0,
            estimator_provenance="INVALID_ESTIMATOR_POSE",
        )

    if has_valid_depth is False and observed_motion in ["MOVING", "DISPLACED"]:
        return RouterDecision(
            entity_id=entity_id,
            decision="DEFER",
            reason_codes=["INVALID_DEPTH_ON_MOVING_TARGET"],
            vlm_required=False,
            cheap_label=None,
            confidence=0.0,
            query_utility=-1.0,
            estimator_influence=0.0,
            estimator_provenance="INVALID_DEPTH_ON_MOVING_TARGET",
        )

    # Calculate explicit estimator influence signal and provenance
    est_influence, est_prov = compute_estimator_influence_with_provenance(evidence)

    # Canonical mobile agents resolved cheaply without VLM
    if sem_class in CANONICAL_MOBILE or prior_mobility == "MOBILE":
        return RouterDecision(
            entity_id=entity_id,
            decision="RESOLVE_CHEAP",
            reason_codes=["CANONICAL_MOBILE_AGENT_PRIOR"],
            vlm_required=False,
            cheap_label="MOBILE",
            confidence=0.98,
            query_utility=0.0,
            estimator_influence=est_influence,
            estimator_provenance=est_prov,
        )

    # Determine relocation and affordance ambiguity factors
    ambiguity_factors = []
    is_displaced = False
    if world_disp is not None:
        is_displaced = (
            world_disp >= MIN_SIGNIFICANT_DISPLACEMENT_M
            and world_disp >= 3.0 * max(0.02, disp_sigma)
            and observed_motion in ["DISPLACED", "MOVING"]
        )
        if is_displaced:
            ambiguity_factors.append("SIGNIFICANT_DISPLACEMENT_AFFORDANCE_UNRESOLVED")
    elif observed_motion in ["DISPLACED", "MOVING"]:
        is_displaced = True
        ambiguity_factors.append("SIGNIFICANT_DISPLACEMENT_AFFORDANCE_UNRESOLVED")

    if sem_class in AMBIGUOUS_AFFORDANCE_CLASSES:
        ambiguity_factors.append("AMBIGUOUS_PHYSICAL_SUPPORT_AFFORDANCE")

    if semantic_entropy >= ENTROPY_AMBIGUITY_THRESHOLD:
        ambiguity_factors.append("HIGH_SEMANTIC_AMBIGUITY")

    # If evidence is ambiguous, calculate operational query utility:
    if ambiguity_factors:
        # Base expected loss reduction:
        # Displaced landmark: severe risk of loop-closure distortion (1.0).
        # Physical support ambiguity (mounted vs freestanding): intermediate risk (0.70).
        # Semantic entropy alone: lower risk (0.40).
        if is_displaced:
            base_risk = 1.0
        elif sem_class in AMBIGUOUS_AFFORDANCE_CLASSES:
            base_risk = 0.85
        else:
            base_risk = 0.60

        delta_L_slam = base_risk * est_influence

        # Query utility = delta_L_slam - lambda_cost * C_vlm (where C_vlm = 1.0 normalized)
        utility = delta_L_slam - (lambda_cost * 1.0)

        # Check visual context availability
        if not has_images or views_count <= 0:
            return RouterDecision(
                entity_id=entity_id,
                decision="DEFER",
                reason_codes=ambiguity_factors + ["MISSING_REQUIRED_VISUAL_CONTEXT"],
                vlm_required=False,
                cheap_label=None,
                confidence=0.30,
                query_utility=round(utility, 4),
                estimator_influence=est_influence,
                estimator_provenance=est_prov,
            )

        # Operational Estimator-Utility Gating:
        # Escalate if and only if query utility meets or exceeds decision threshold
        if utility >= utility_threshold:
            return RouterDecision(
                entity_id=entity_id,
                decision="ESCALATE",
                reason_codes=ambiguity_factors + [
                    "HIGH_ESTIMATOR_RELEVANCE",
                    "POSITIVE_QUERY_UTILITY",
                    f"ESTIMATOR_INFLUENCE_{est_influence:.2f}",
                    f"PROVENANCE_{est_prov}",
                ],
                vlm_required=True,
                cheap_label=None,
                confidence=0.90,
                query_utility=round(utility, 4),
                estimator_influence=est_influence,
                estimator_provenance=est_prov,
            )
        else:
            if sem_class in CANONICAL_IMMOVABLE or sem_class == "whiteboard" or prior_mobility == "IMMOVABLE":
                fallback_label = "IMMOVABLE"
            elif sem_class in CANONICAL_MOBILE or prior_mobility == "MOBILE":
                fallback_label = "MOBILE"
            else:
                fallback_label = "MOVABLE"
            resolved_label = "MOVABLE" if (is_displaced and sem_class not in CANONICAL_IMMOVABLE) else fallback_label
            return RouterDecision(
                entity_id=entity_id,
                decision="RESOLVE_CHEAP",
                reason_codes=ambiguity_factors + [
                    "SUBTHRESHOLD_QUERY_UTILITY_AVOIDED",
                    f"ESTIMATOR_INFLUENCE_{est_influence:.2f}",
                ],
                vlm_required=False,
                cheap_label=resolved_label,
                confidence=0.70,
                query_utility=round(utility, 4),
                estimator_influence=est_influence,
                estimator_provenance=est_prov,
            )

    # Stationary architectural elements
    if sem_class in CANONICAL_IMMOVABLE or prior_mobility == "IMMOVABLE":
        return RouterDecision(
            entity_id=entity_id,
            decision="RESOLVE_CHEAP",
            reason_codes=["CANONICAL_IMMOVABLE_ARCHITECTURAL_PRIOR"],
            vlm_required=False,
            cheap_label="IMMOVABLE",
            confidence=0.99,
            query_utility=0.0,
            estimator_influence=est_influence,
            estimator_provenance=est_prov,
        )

    # Stationary movable furniture
    if sem_class in CANONICAL_MOVABLE or prior_mobility == "MOVABLE":
        return RouterDecision(
            entity_id=entity_id,
            decision="RESOLVE_CHEAP",
            reason_codes=["STATIC_CANONICAL_MOVABLE_PRIOR"],
            vlm_required=False,
            cheap_label="MOVABLE",
            confidence=0.95,
            query_utility=0.0,
            estimator_influence=est_influence,
            estimator_provenance=est_prov,
        )

    # Unknown category: escalate if images available and influence exists
    if has_images and views_count > 0:
        return RouterDecision(
            entity_id=entity_id,
            decision="ESCALATE",
            reason_codes=["UNKNOWN_CATEGORY_ESCALATION"],
            vlm_required=True,
            cheap_label=None,
            confidence=0.50,
            query_utility=round(est_influence - lambda_cost, 4),
            estimator_influence=est_influence,
        )

    return RouterDecision(
        entity_id=entity_id,
        decision="ABSTAIN",
        reason_codes=["UNKNOWN_CATEGORY_NO_IMAGES"],
        vlm_required=False,
        cheap_label="UNKNOWN",
        confidence=0.0,
        query_utility=-1.0,
        estimator_influence=est_influence,
    )


def evaluate_policy_r4(evidence: Dict[str, Any]) -> RouterDecision:
    """R4: All-Eligible VLM Reference Policy."""
    entity_id = str(evidence.get("entity_id", "unknown"))
    has_images = bool(evidence.get("has_images", False))
    views_count = int(evidence.get("views_count", 0))

    if has_images and views_count > 0:
        return RouterDecision(
            entity_id=entity_id,
            decision="ESCALATE",
            reason_codes=["R4_ALL_ELIGIBLE_VLM"],
            vlm_required=True,
            cheap_label=None,
            confidence=0.90,
        )

    return RouterDecision(
        entity_id=entity_id,
        decision="DEFER",
        reason_codes=["R4_MISSING_IMAGES"],
        vlm_required=False,
        cheap_label="UNKNOWN",
        confidence=0.0,
    )


def evaluate_router_policy(
    evidence: Dict[str, Any],
    policy: Literal["R0", "R1", "R2", "R3", "R4"] = "R3",
    lambda_cost: float = DEFAULT_VLM_COST,
) -> RouterDecision:
    """Unified policy evaluator defaulting to R3 (Estimator-Aware)."""
    if policy == "R0":
        return evaluate_policy_r0(evidence)
    elif policy == "R1":
        return evaluate_policy_r1(evidence)
    elif policy == "R2":
        return evaluate_policy_r2(evidence)
    elif policy == "R4":
        return evaluate_policy_r4(evidence)
    return evaluate_policy_r3(evidence, lambda_cost=lambda_cost)
