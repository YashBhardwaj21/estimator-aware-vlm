"""Expected Downstream Regret Reduction (EDRR) Formulation.

Formal decision-theoretic query-selection formulation:
    EDRR_i(x_i) = E_{y_i ~ P(y|x_i)} [ R(pi_no_query, Z_i) - R(pi_query(y_i), Z_i) ] - lambda_C * C_i

where:
    - R(pi, Z_i) = L(pi, Z_i) - L(pi^*, Z_i) is the downstream regret under policy pi.
    - L(pi, Z_i) is the multi-component downstream SLAM loss.
    - Since L(pi^*, Z_i) cancels, Regret Reduction identically equals Downstream Loss Reduction:
          Delta L_i(y_i) = L(pi_no_query, Z_i) - L(pi_query(y_i), Z_i)
    - lambda_C is the resource exchange rate (tradeoff parameter).
    - C_i is the normalized inference cost of querying the VLM.

Under fixed query budget K:
    Select candidate set S* = argmax_{|S| <= K} sum_{i in S} EDRR_i(x_i) subject to EDRR_i > 0.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Tuple
import numpy as np

from estimator_aware_vlm.edrr.loss import (
    DownstreamSLAMLoss,
    SLAMLossWeights,
    compute_normalized_slam_loss,
)


@dataclass
class CandidateQueryProfile:
    """Perceptual and estimator features for a candidate visual query."""
    candidate_id: str
    sequence_id: str
    semantic_class: str
    measured_displacement_m: float
    displacement_uncertainty_m: float
    observation_count: int
    camera_ego_motion_displacement_m: float
    lifespan_sec: float
    local_keypoint_count: int
    nominal_keypoint_count: int = 150
    estimated_query_cost: float = 0.50
    prior_p_dynamic: float = 0.50

    def to_feature_vector(self) -> np.ndarray:
        """Convert profile to normalized 6D feature vector."""
        w_disp = float(self.measured_displacement_m)
        disp_unc = float(self.displacement_uncertainty_m)
        obs_cnt_norm = min(1.0, float(self.observation_count) / 100.0)
        cam_disp = float(self.camera_ego_motion_displacement_m)
        lifespan_norm = min(1.0, float(self.lifespan_sec) / 10.0)
        disp_to_ego_ratio = min(5.0, w_disp / (cam_disp + 1e-4))
        starvation_risk = max(0.0, 1.0 - (float(self.local_keypoint_count) / float(self.nominal_keypoint_count)))

        return np.array([
            w_disp,
            disp_unc,
            obs_cnt_norm,
            cam_disp,
            lifespan_norm,
            disp_to_ego_ratio,
            starvation_risk,
        ], dtype=np.float64)


@dataclass
class EDRRDecision:
    """Decision record for a single query candidate under EDRR."""
    candidate_id: str
    sequence_id: str
    semantic_class: str
    expected_delta_loss: float
    vlm_query_cost: float
    lambda_cost: float
    edrr_score: float
    should_query: bool
    rank: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "sequence_id": self.sequence_id,
            "semantic_class": self.semantic_class,
            "expected_delta_loss": round(self.expected_delta_loss, 6),
            "vlm_query_cost": round(self.vlm_query_cost, 4),
            "lambda_cost": round(self.lambda_cost, 4),
            "edrr_score": round(self.edrr_score, 6),
            "should_query": self.should_query,
            "rank": self.rank,
            "metadata": self.metadata,
        }


def compute_candidate_edrr(
    profile: CandidateQueryProfile,
    predicted_delta_loss: float,
    lambda_cost: float = 0.40,
) -> EDRRDecision:
    """Compute EDRR score for a candidate object.

    Args:
        profile: CandidateQueryProfile holding sensory observations and sequence context.
        predicted_delta_loss: Predicted downstream SLAM loss reduction Delta L.
        lambda_cost: Tradeoff parameter weighting inference cost relative to loss reduction.

    Returns:
        EDRRDecision with computed net score and query recommendation.
    """
    cost = profile.estimated_query_cost
    net_edrr = float(predicted_delta_loss - (lambda_cost * cost))
    should_query = net_edrr > 0.0

    return EDRRDecision(
        candidate_id=profile.candidate_id,
        sequence_id=profile.sequence_id,
        semantic_class=profile.semantic_class,
        expected_delta_loss=predicted_delta_loss,
        vlm_query_cost=cost,
        lambda_cost=lambda_cost,
        edrr_score=net_edrr,
        should_query=should_query,
    )


def select_top_k_edrr(
    candidates: List[Tuple[CandidateQueryProfile, float]],
    budget_k: int,
    lambda_cost: float = 0.40,
    enforce_positive_utility: bool = True,
) -> List[EDRRDecision]:
    """Rank and select up to K candidates under EDRR formulation.

    Args:
        candidates: List of (CandidateQueryProfile, predicted_delta_loss) pairs.
        budget_k: Maximum allowable VLM queries.
        lambda_cost: Inference cost tradeoff multiplier.
        enforce_positive_utility: If True, only select candidates with EDRR > 0.

    Returns:
        Ordered list of EDRRDecision objects with assigned ranks and should_query flags.
    """
    decisions: List[EDRRDecision] = []
    for prof, pred_dl in candidates:
        dec = compute_candidate_edrr(prof, pred_dl, lambda_cost=lambda_cost)
        decisions.append(dec)

    # Sort descending by EDRR score
    decisions.sort(key=lambda d: d.edrr_score, reverse=True)

    selected_count = 0
    for idx, dec in enumerate(decisions):
        dec.rank = idx + 1
        if selected_count < budget_k:
            if not enforce_positive_utility or dec.edrr_score > 0.0:
                dec.should_query = True
                selected_count += 1
            else:
                dec.should_query = False
        else:
            dec.should_query = False

    return decisions

