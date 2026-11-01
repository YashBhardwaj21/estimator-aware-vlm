"""Multi-Component Unit-Consistent Downstream SLAM Loss Formulation.

Formulates downstream SLAM estimation loss across four critical operational dimensions:
    L(pi, Z) = alpha_p * L_pose + alpha_t * L_tracking + alpha_m * L_map + alpha_f * L_feature_starvation

All components are normalized into dimensionless quantities on [0, 1] scale or calibrated
multipliers to ensure unit consistency, with non-negative convex combination weights:
    alpha_p + alpha_t + alpha_m + alpha_f = 1.0,  alpha_* >= 0
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Any, Dict, Optional


@dataclass(frozen=True)
class SLAMLossWeights:
    """Normalized weighting coefficients for multi-component downstream SLAM loss."""
    alpha_pose: float = 0.40
    alpha_tracking: float = 0.25
    alpha_map: float = 0.20
    alpha_feature_starvation: float = 0.15

    def __post_init__(self) -> None:
        total = self.alpha_pose + self.alpha_tracking + self.alpha_map + self.alpha_feature_starvation
        if not math.isclose(total, 1.0, rel_tol=1e-5):
            raise ValueError(f"Loss weights must sum to 1.0, got {total}")
        for k, v in asdict(self).items():
            if v < 0.0:
                raise ValueError(f"Weight {k} must be non-negative, got {v}")


@dataclass
class DownstreamSLAMLoss:
    """Evaluated multi-component downstream SLAM loss for an estimation state."""
    l_pose: float
    l_tracking: float
    l_map: float
    l_feature_starvation: float
    total_loss: float
    weights: SLAMLossWeights

    def to_dict(self) -> Dict[str, Any]:
        return {
            "l_pose": round(self.l_pose, 6),
            "l_tracking": round(self.l_tracking, 6),
            "l_map": round(self.l_map, 6),
            "l_feature_starvation": round(self.l_feature_starvation, 6),
            "total_loss": round(self.total_loss, 6),
            "weights": asdict(self.weights),
        }


def compute_normalized_slam_loss(
    ate_rmse_m: Optional[float],
    tracking_lost_ratio: float,
    map_anchor_residual_m: Optional[float],
    min_feature_count: int,
    nominal_feature_count: int = 150,
    max_tolerable_ate_m: float = 2.0,
    max_tolerable_residual_m: float = 1.0,
    weights: Optional[SLAMLossWeights] = None,
) -> DownstreamSLAMLoss:
    """Compute unit-consistent, normalized multi-component downstream SLAM loss.

    Args:
        ate_rmse_m: Absolute Trajectory Error RMSE in meters (None if MoCap unavailable, e.g. D455).
        tracking_lost_ratio: Ratio of lost tracking frames in [0, 1].
        map_anchor_residual_m: Scene graph or landmark anchor registration residual in meters.
        min_feature_count: Minimum active keypoint/feature count observed in the scene.
        nominal_feature_count: Nominal baseline feature count before starvation.
        max_tolerable_ate_m: Normalization scale for ATE (meters).
        max_tolerable_residual_m: Normalization scale for map anchor residual (meters).
        weights: SLAMLossWeights instance summing to 1.0.

    Returns:
        DownstreamSLAMLoss containing individual components and total loss.
    """
    w = weights or SLAMLossWeights()

    # 1. Normalized Pose Error Component L_pose in [0, 1]
    if ate_rmse_m is not None and not math.isnan(ate_rmse_m):
        l_pose = min(1.0, max(0.0, float(ate_rmse_m) / max_tolerable_ate_m))
    else:
        # For datasets without ground-truth trajectory MoCap (e.g. handheld D455),
        # pose loss falls back to tracking continuity proxy
        l_pose = min(1.0, max(0.0, float(tracking_lost_ratio)))

    # 2. Normalized Tracking Loss Component L_tracking in [0, 1]
    l_tracking = min(1.0, max(0.0, float(tracking_lost_ratio)))

    # 3. Normalized Map Integrity Component L_map in [0, 1]
    if map_anchor_residual_m is not None and not math.isnan(map_anchor_residual_m):
        l_map = min(1.0, max(0.0, float(map_anchor_residual_m) / max_tolerable_residual_m))
    else:
        l_map = 0.0

    # 4. Normalized Feature-Starvation Component L_feature_starvation in [0, 1]
    # Penalizes dropping below minimum tracking threshold
    starvation_deficit = max(0, nominal_feature_count - min_feature_count)
    l_starvation = min(1.0, float(starvation_deficit) / float(nominal_feature_count))

    total = (
        w.alpha_pose * l_pose
        + w.alpha_tracking * l_tracking
        + w.alpha_map * l_map
        + w.alpha_feature_starvation * l_starvation
    )

    return DownstreamSLAMLoss(
        l_pose=l_pose,
        l_tracking=l_tracking,
        l_map=l_map,
        l_feature_starvation=l_starvation,
        total_loss=total,
        weights=w,
    )

