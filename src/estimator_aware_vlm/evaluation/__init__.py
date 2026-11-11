"""Evaluation package for Estimator-Aware VLM."""

from estimator_aware_vlm.evaluation.trajectory_metrics import (
    ATEStats,
    EvaluationError,
    RPEStats,
    TrajectoryEvaluationResult,
    align_trajectories_se3,
    compute_ate,
    compute_frame_tracking_rate,
    compute_rpe,
    evaluate_trajectory,
    match_trajectories_by_timestamp,
)
from estimator_aware_vlm.evaluation.paired_metrics import (
    PairedEvaluationResult,
    evaluate_paired_trajectories_common_support,
)
from estimator_aware_vlm.evaluation.baseline_evaluator import (
    CrossSessionAnchorStats,
    MapQualityStats,
    NativeLogStats,
    TrajectoryContinuityStats,
    compute_map_point_stats,
    compute_trajectory_continuity,
    load_cross_session_anchors,
    parse_native_execution_log,
)

__all__ = [
    "ATEStats",
    "EvaluationError",
    "RPEStats",
    "TrajectoryEvaluationResult",
    "PairedEvaluationResult",
    "CrossSessionAnchorStats",
    "MapQualityStats",
    "NativeLogStats",
    "TrajectoryContinuityStats",
    "align_trajectories_se3",
    "compute_ate",
    "compute_frame_tracking_rate",
    "compute_rpe",
    "compute_map_point_stats",
    "compute_trajectory_continuity",
    "evaluate_trajectory",
    "evaluate_paired_trajectories_common_support",
    "load_cross_session_anchors",
    "match_trajectories_by_timestamp",
    "parse_native_execution_log",
]
