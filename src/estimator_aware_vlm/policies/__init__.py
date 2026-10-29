"""Estimator-Aware Policy Evaluation and Comparison Suite."""

from estimator_aware_vlm.policies.comparison_suite import (
    CachedVLMProvider,
    LiveVLMProvider,
    PolicyExecutionSummary,
    TrackDecisionRecord,
    VLMProvider,
    VLMQueryResult,
    evaluate_b0_blind_retain,
    evaluate_b1_blind_invalidate,
    evaluate_b2_geometry_only,
    evaluate_b3_vlm_all,
    evaluate_b4_random,
    evaluate_b5_confidence_threshold,
    evaluate_b6_estimator_aware,
    run_fair_policy_comparison_suite,
)

__all__ = [
    "CachedVLMProvider",
    "LiveVLMProvider",
    "PolicyExecutionSummary",
    "TrackDecisionRecord",
    "VLMProvider",
    "VLMQueryResult",
    "evaluate_b0_blind_retain",
    "evaluate_b1_blind_invalidate",
    "evaluate_b2_geometry_only",
    "evaluate_b3_vlm_all",
    "evaluate_b4_random",
    "evaluate_b5_confidence_threshold",
    "evaluate_b6_estimator_aware",
    "run_fair_policy_comparison_suite",
]
