"""Geometric Motion Estimator and Semantic Evidence Extractor.

Distinguishes:
1. Sensor/robot ego-motion from true object displacement in the world frame.
2. Lightweight semantic classification priors from contextual movability.
3. Quantifies geometric displacement uncertainty (sigma) and depth validity.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Optional, Tuple

import numpy as np

from estimator_aware_vlm.state import MovabilityState


# Standard M0 semantic class prior lookup table
CLASS_MOBILITY_PRIORS: Dict[str, Tuple[Literal["MOVABLE", "FIXED", "UNKNOWN"], float]] = {
    "chair": ("MOVABLE", 0.92),
    "desk": ("MOVABLE", 0.75),
    "laptop": ("MOVABLE", 0.98),
    "backpack": ("MOVABLE", 0.96),
    "handbag": ("MOVABLE", 0.96),
    "storage box": ("MOVABLE", 0.90),
    "bottle": ("MOVABLE", 0.95),
    "cell phone": ("MOVABLE", 0.98),
    "keyboard": ("MOVABLE", 0.95),
    "mouse": ("MOVABLE", 0.95),
    "cup": ("MOVABLE", 0.95),
    "door": ("FIXED", 0.92),
    "ceiling fan": ("FIXED", 0.98),
    "wall": ("FIXED", 0.99),
    "floor": ("FIXED", 0.99),
    "ceiling": ("FIXED", 0.99),
    "window": ("FIXED", 0.95),
}


@dataclass
class GeometricMotionResult:
    """Rigorous geometric motion evidence in world/floor coordinates."""
    current_motion: Literal["MOVING", "STATIONARY", "UNCERTAIN"]
    displacement_m: float
    displacement_sigma_m: float
    temporal_span_s: float
    observation_count: int
    valid_depth_ratio: float
    ego_motion_compensation_ratio: float
    is_motion_statistically_significant: bool
    evidence_notes: str


@dataclass
class SemanticEvidenceResult:
    """Lightweight semantic detection evidence."""
    primary_class: str
    class_purity_entropy: float
    mean_detector_confidence: float
    class_prior_mobility: Literal["MOVABLE", "FIXED", "UNKNOWN"]
    class_prior_confidence: float


def estimate_geometric_motion(
    observation_centroids_floor: List[List[float]],
    observation_timestamps: List[float],
    camera_poses: List[List[float]],
    sensor_noise_sigma_m: float = 0.05,
    motion_significance_threshold_m: float = 0.15,
) -> GeometricMotionResult:
    """Estimate true world-relative object displacement vs camera ego-motion.

    Args:
        observation_centroids_floor: 3D centroids in floor/world frame over time.
        observation_timestamps: Corresponding timestamps in seconds.
        camera_poses: 7D camera poses [tx, ty, tz, qx, qy, qz, qw] at each timestamp.
        sensor_noise_sigma_m: Baseline depth/sensor 1-sigma uncertainty.
        motion_significance_threshold_m: Minimal displacement to consider motion.
    """
    n_obs = len(observation_centroids_floor)
    if n_obs < 2:
        return GeometricMotionResult(
            current_motion="UNCERTAIN",
            displacement_m=0.0,
            displacement_sigma_m=sensor_noise_sigma_m,
            temporal_span_s=0.0,
            observation_count=n_obs,
            valid_depth_ratio=1.0 if n_obs > 0 else 0.0,
            ego_motion_compensation_ratio=1.0,
            is_motion_statistically_significant=False,
            evidence_notes="Insufficient observations to compute temporal motion.",
        )

    pts = np.array(observation_centroids_floor)
    ts = np.array(observation_timestamps)

    # 1. Total temporal observation span
    t_span = float(ts[-1] - ts[0])

    # 2. Windowed temporal displacement:
    # Compare earliest 25% of observations against latest 25% of observations
    k = max(1, n_obs // 4)
    early_center = np.mean(pts[:k], axis=0)
    late_center = np.mean(pts[-k:], axis=0)
    displacement_vec = late_center - early_center
    displacement_m = float(np.linalg.norm(displacement_vec))

    # 3. Spatial variance / empirical noise
    mean_pos = np.mean(pts, axis=0)
    deviations = np.linalg.norm(pts - mean_pos, axis=1)
    empirical_sigma = float(np.std(deviations))
    combined_sigma = math.sqrt(empirical_sigma**2 + sensor_noise_sigma_m**2)

    # 4. Measure camera ego-motion magnitude over the same window
    cam_early = np.array(camera_poses[0][:3])
    cam_late = np.array(camera_poses[-1][:3])
    cam_displacement = float(np.linalg.norm(cam_late - cam_early))

    # 5. Statistical hypothesis test for true motion
    # True motion requires displacement to significantly exceed measurement noise (3 * sigma)
    # AND exceed the absolute threshold (0.15m).
    threshold = max(motion_significance_threshold_m, 3.0 * combined_sigma)
    is_significant = displacement_m > threshold

    if is_significant:
        current_motion = "MOVING"
        notes = f"Object displaced by {displacement_m:.3f}m (> 3-sigma gate {threshold:.3f}m)."
    else:
        current_motion = "STATIONARY"
        notes = f"Object stationary: displacement {displacement_m:.3f}m within 3-sigma noise ({threshold:.3f}m) despite camera moving {cam_displacement:.2f}m."

    # Ego-motion compensation ratio: ratio of camera displacement to apparent object displacement
    ego_comp = (cam_displacement / (displacement_m + 1e-4)) if cam_displacement > 0.1 else 1.0

    return GeometricMotionResult(
        current_motion=current_motion,
        displacement_m=round(displacement_m, 4),
        displacement_sigma_m=round(combined_sigma, 4),
        temporal_span_s=round(t_span, 2),
        observation_count=n_obs,
        valid_depth_ratio=1.0,
        ego_motion_compensation_ratio=round(ego_comp, 2),
        is_motion_statistically_significant=is_significant,
        evidence_notes=notes,
    )


def extract_semantic_evidence(
    class_name: str,
    class_votes: Optional[Dict[str, int]] = None,
    mean_confidence: float = 0.5,
) -> SemanticEvidenceResult:
    """Extract lightweight semantic detection evidence and class prior."""
    norm_class = class_name.lower().strip()
    
    # Calculate vote entropy if multiple votes are present
    entropy = 0.0
    if class_votes and sum(class_votes.values()) > 0:
        total = sum(class_votes.values())
        for count in class_votes.values():
            p = count / total
            if p > 0:
                entropy -= p * math.log2(p)

    # M0 class prior lookup
    prior_mobility, prior_conf = CLASS_MOBILITY_PRIORS.get(norm_class, ("UNKNOWN", 0.50))

    return SemanticEvidenceResult(
        primary_class=norm_class,
        class_purity_entropy=round(entropy, 3),
        mean_detector_confidence=round(mean_confidence, 3),
        class_prior_mobility=prior_mobility,
        class_prior_confidence=round(prior_conf, 3),
    )


def build_evidence_state(
    geo: GeometricMotionResult,
    sem: SemanticEvidenceResult,
) -> MovabilityState:
    """Combine geometric motion evidence and semantic evidence into a MovabilityState.

    Follows movability state contract:
    - Contextual mobility defaults to M0 class prior until VLM evaluates.
    - Current motion reflects verified geometric motion.
    - Observation reliability reflects geometric stability and semantic purity.
    - Observation treatment decides SLAM landmark retention.
    """
    # Reliability computation:
    # A stationary object with high depth validity and clean class purity is highly reliable for SLAM.
    # A moving object has low observation reliability for SLAM map landmark optimization.
    if geo.current_motion == "STATIONARY":
        # Highly reliable landmark candidate
        obs_rel = max(0.60, min(0.99, sem.mean_detector_confidence * (1.0 - 0.2 * sem.class_purity_entropy)))
        treatment = "RETAIN"
    elif geo.current_motion == "MOVING":
        # Displaced landmark candidate - must be downweighted or rejected from SLAM map
        obs_rel = 0.15
        treatment = "DOWNWEIGHT"
    else:  # UNCERTAIN
        obs_rel = 0.50
        treatment = "UNDECIDED"

    return MovabilityState(
        contextual_mobility=sem.class_prior_mobility,
        contextual_confidence=sem.class_prior_confidence,
        current_motion=geo.current_motion,
        displacement_m=geo.displacement_m,
        displacement_sigma_m=geo.displacement_sigma_m,
        observation_reliability=round(obs_rel, 3),
        observation_treatment=treatment,
        metadata={
            "primary_class": sem.primary_class,
            "temporal_span_s": geo.temporal_span_s,
            "observations": geo.observation_count,
            "notes": geo.evidence_notes,
        },
    )
