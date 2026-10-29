"""Evaluation package for Estimator-Aware VLM."""

from estimator_aware_vlm.evaluation.trajectory_metrics import (
    ATEStats,
    EvaluationError,
    RPEStats,
    TrajectoryEvaluationResult,
    align_trajectories_se3,
    compute_ate,
    compute_rpe,
    evaluate_trajectory,
    match_trajectories_by_timestamp,
)
from estimator_aware_vlm.evaluation.paired_metrics import (
    PairedEvaluationResult,
    evaluate_paired_trajectories_common_support,
)

__all__ = [
    "ATEStats",
    "EvaluationError",
    "RPEStats",
    "TrajectoryEvaluationResult",
    "PairedEvaluationResult",
    "align_trajectories_se3",
    "compute_ate",
    "compute_rpe",
    "evaluate_trajectory",
    "evaluate_paired_trajectories_common_support",
    "match_trajectories_by_timestamp",
]
