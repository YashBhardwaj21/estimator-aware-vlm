"""Unit and contract tests for Stage 6: Fair Policy Comparisons and Cost Accounting.

Verifies:
1. Budget matching fairness: K_B4 == K_B6.
2. Calibrated utility integration in B6 vs uncalibrated fallback.
3. Dual latency accounting: observed vs hypothetical wire latency and token counts.
4. Clean decoupling between VLM query decision and downstream SLAM action.
5. Strict zero ground-truth leakage into routing decisions.
6. Deterministic seeding and reproducibility for random baselines.
"""

from __future__ import annotations

import copy
from pathlib import Path
import random
from typing import Any, Dict, List
import pytest

from estimator_aware_vlm.calibration.counterfactual import CalibratedUtilityModel
from estimator_aware_vlm.policies.comparison_suite import (
    CachedVLMProvider,
    PolicyExecutionSummary,
    TrackDecisionRecord,
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


@pytest.fixture
def mock_candidate_tracks() -> List[Dict[str, Any]]:
    """Synthesize 10 candidate tracks with diverse geometry, motion, and visual availability."""
    return [
        {
            "track_id": "monitor_static",
            "semantic_class": "monitor",
            "observation_count": 150,
            "measured_displacement_m": 0.05,
            "displacement_uncertainty_m": 0.02,
            "camera_ego_motion_displacement_m": 1.2,
            "lifespan_sec": 15.0,
            "observed_motion": "STATIC",
            "crop_paths": ["/dummy/crop_monitor.jpg"],
        },
        {
            "track_id": "monitor_displaced",
            "semantic_class": "monitor",
            "observation_count": 180,
            "measured_displacement_m": 0.85,
            "displacement_uncertainty_m": 0.04,
            "camera_ego_motion_displacement_m": 1.5,
            "lifespan_sec": 20.0,
            "observed_motion": "MOVING",
            "crop_paths": ["/dummy/crop_monitor2.jpg"],
        },
        {
            "track_id": "chair_moving",
            "semantic_class": "chair",
            "observation_count": 95,
            "measured_displacement_m": 0.65,
            "displacement_uncertainty_m": 0.09,
            "camera_ego_motion_displacement_m": 1.1,
            "lifespan_sec": 12.0,
            "observed_motion": "MOVING",
            "crop_paths": ["/dummy/crop_chair.jpg"],
        },
        {
            "track_id": "chair_stationary",
            "semantic_class": "chair",
            "observation_count": 40,
            "measured_displacement_m": 0.02,
            "displacement_uncertainty_m": 0.01,
            "camera_ego_motion_displacement_m": 0.8,
            "lifespan_sec": 8.0,
            "observed_motion": "STATIC",
            "crop_paths": ["/dummy/crop_chair2.jpg"],
        },
        {
            "track_id": "person_active",
            "semantic_class": "person",
            "observation_count": 210,
            "measured_displacement_m": 1.45,
            "displacement_uncertainty_m": 0.12,
            "camera_ego_motion_displacement_m": 2.0,
            "lifespan_sec": 22.0,
            "observed_motion": "MOVING",
            "crop_paths": ["/dummy/crop_person.jpg"],
        },
        {
            "track_id": "keyboard_tabletop",
            "semantic_class": "keyboard",
            "observation_count": 60,
            "measured_displacement_m": 0.01,
            "displacement_uncertainty_m": 0.01,
            "camera_ego_motion_displacement_m": 0.5,
            "lifespan_sec": 5.0,
            "observed_motion": "STATIC",
            "crop_paths": ["/dummy/crop_keyboard.jpg"],
        },
        {
            "track_id": "desk_heavy",
            "semantic_class": "desk",
            "observation_count": 120,
            "measured_displacement_m": 0.04,
            "displacement_uncertainty_m": 0.01,
            "camera_ego_motion_displacement_m": 1.4,
            "lifespan_sec": 18.0,
            "observed_motion": "STATIC",
            "crop_paths": ["/dummy/crop_desk.jpg"],
        },
        {
            "track_id": "bottle_small",
            "semantic_class": "bottle",
            "observation_count": 30,
            "measured_displacement_m": 0.03,
            "displacement_uncertainty_m": 0.01,
            "camera_ego_motion_displacement_m": 0.4,
            "lifespan_sec": 4.0,
            "observed_motion": "STATIC",
            "crop_paths": ["/dummy/crop_bottle.jpg"],
        },
        {
            "track_id": "box_borderline",
            "semantic_class": "box",
            "observation_count": 55,
            "measured_displacement_m": 0.22,
            "displacement_uncertainty_m": 0.15,
            "camera_ego_motion_displacement_m": 0.9,
            "lifespan_sec": 7.0,
            "observed_motion": "MOVING",
            "crop_paths": ["/dummy/crop_box.jpg"],
        },
        {
            "track_id": "unseen_object",
            "semantic_class": "unknown",
            "observation_count": 10,
            "measured_displacement_m": 0.05,
            "displacement_uncertainty_m": 0.05,
            "camera_ego_motion_displacement_m": 0.2,
            "lifespan_sec": 1.0,
            "observed_motion": "STATIC",
            "crop_paths": [],  # No images available
        },
    ]


@pytest.fixture
def mock_cached_provider() -> CachedVLMProvider:
    """Pre-recorded mock VLM results with realistic wire latency and token counts."""
    telemetry = {
        "monitor_static": {
            "semantic_class": "monitor",
            "contextual_movability": "IMMOVABLE",
            "reasoning": "Mounted on workstation desk.",
            "latency_ms": 1150.0,
            "prompt_tokens": 230,
            "completion_tokens": 45,
        },
        "monitor_displaced": {
            "semantic_class": "monitor",
            "contextual_movability": "IMMOVABLE",
            "reasoning": "Desktop display; camera moved past it.",
            "latency_ms": 1220.0,
            "prompt_tokens": 240,
            "completion_tokens": 52,
        },
        "chair_moving": {
            "semantic_class": "chair",
            "contextual_movability": "MOVABLE",
            "reasoning": "Office chair on casters being pushed.",
            "latency_ms": 980.0,
            "prompt_tokens": 215,
            "completion_tokens": 40,
        },
        "chair_stationary": {
            "semantic_class": "chair",
            "contextual_movability": "MOVABLE",
            "reasoning": "Empty office chair tucked under desk.",
            "latency_ms": 950.0,
            "prompt_tokens": 210,
            "completion_tokens": 38,
        },
        "person_active": {
            "semantic_class": "person",
            "contextual_movability": "MOBILE",
            "reasoning": "Walking human researcher.",
            "latency_ms": 1400.0,
            "prompt_tokens": 255,
            "completion_tokens": 60,
        },
        "keyboard_tabletop": {
            "semantic_class": "keyboard",
            "contextual_movability": "MOVABLE",
            "reasoning": "Computer keyboard lying on desk.",
            "latency_ms": 890.0,
            "prompt_tokens": 200,
            "completion_tokens": 35,
        },
        "desk_heavy": {
            "semantic_class": "desk",
            "contextual_movability": "IMMOVABLE",
            "reasoning": "Heavy wooden office desk.",
            "latency_ms": 1100.0,
            "prompt_tokens": 225,
            "completion_tokens": 42,
        },
        "bottle_small": {
            "semantic_class": "bottle",
            "contextual_movability": "MOVABLE",
            "reasoning": "Water bottle on table.",
            "latency_ms": 820.0,
            "prompt_tokens": 195,
            "completion_tokens": 30,
        },
        "box_borderline": {
            "semantic_class": "box",
            "contextual_movability": "MOVABLE",
            "reasoning": "Storage box sitting on floor.",
            "latency_ms": 910.0,
            "prompt_tokens": 205,
            "completion_tokens": 36,
        },
    }
    return CachedVLMProvider(telemetry)


@pytest.fixture
def mock_calibrated_model() -> CalibratedUtilityModel:
    """Pre-calibrated empirical utility model for unit testing."""
    return CalibratedUtilityModel(
        feature_names=[
            "measured_displacement_m",
            "displacement_uncertainty_m",
            "observation_count_norm",
            "camera_ego_motion_displacement_m",
            "lifespan_sec_norm",
            "displacement_to_ego_ratio",
        ],
        weights=[0.35, 0.10, 0.25, -0.05, 0.15, 0.08],
        bias=0.12,
        feature_mean=[0.30, 0.05, 0.60, 0.80, 0.50, 0.40],
        feature_std=[0.25, 0.03, 0.30, 0.40, 0.25, 0.30],
        loocv_mae=0.035,
        loocv_r2=0.82,
        lambda_cost=0.45,
        c_vlm=0.50,
    )


def test_budget_matching_b4_to_b6(
    mock_candidate_tracks: List[Dict[str, Any]],
    mock_cached_provider: CachedVLMProvider,
    mock_calibrated_model: CalibratedUtilityModel,
) -> None:
    """Test that B4 (Random) is strictly budget-matched to B6's query count."""
    results = run_fair_policy_comparison_suite(
        tracks=mock_candidate_tracks,
        vlm_provider=mock_cached_provider,
        calibrated_model=mock_calibrated_model,
        lambda_cost=0.45,
        random_seed=42,
    )

    b6_summary = results["B6_Estimator_Aware"]
    b4_summary = results["B4_Random"]

    # Strict budget equality
    assert b4_summary.query_count == b6_summary.query_count, (
        f"Fairness violation: B4 query count {b4_summary.query_count} != B6 query count {b6_summary.query_count}"
    )
    assert b4_summary.budget_matched_to == "B6_Estimator_Aware"
    assert b4_summary.query_rate == b6_summary.query_rate


def test_calibrated_utility_vs_uncalibrated_fallback(
    mock_candidate_tracks: List[Dict[str, Any]],
    mock_cached_provider: CachedVLMProvider,
    mock_calibrated_model: CalibratedUtilityModel,
) -> None:
    """Verify B6 correctly incorporates empirical calibrated model and tags provenance."""
    # Run with calibrated model
    b6_calib = evaluate_b6_estimator_aware(
        tracks=mock_candidate_tracks,
        vlm_provider=mock_cached_provider,
        calibrated_model=mock_calibrated_model,
    )

    # Run with uncalibrated proxy
    b6_proxy = evaluate_b6_estimator_aware(
        tracks=mock_candidate_tracks,
        vlm_provider=mock_cached_provider,
        calibrated_model=None,
    )

    assert b6_calib.query_count >= 1
    assert b6_proxy.query_count >= 1

    # Verify provenance tag
    calib_reasons = [r for d in b6_calib.decisions for r in d.get("reason_codes", [])]
    assert any("PROVENANCE_CALIBRATED_EMPIRICAL" in r for r in calib_reasons)

    proxy_reasons = [r for d in b6_proxy.decisions for r in d.get("reason_codes", [])]
    assert any("PROVENANCE_PROXY_SUPPORT" in r for r in proxy_reasons)


def test_dual_latency_and_cost_accounting(
    mock_candidate_tracks: List[Dict[str, Any]],
    mock_cached_provider: CachedVLMProvider,
    mock_calibrated_model: CalibratedUtilityModel,
) -> None:
    """Verify dual latency tracking: sub-millisecond replay vs honest wire latency."""
    results = run_fair_policy_comparison_suite(
        tracks=mock_candidate_tracks,
        vlm_provider=mock_cached_provider,
        calibrated_model=mock_calibrated_model,
        lambda_cost=0.45,
    )

    for pkey, summary in results.items():
        # Latencies must be non-negative
        assert summary.total_observed_latency_ms >= 0.0
        assert summary.total_hypothetical_wire_latency_ms >= 0.0

        if summary.query_count > 0:
            # During replay, observed latency is fast (< 50ms total)
            assert summary.total_observed_latency_ms < 50.0
            # Hypothetical wire latency reflects real model execution (> 500ms per query)
            assert summary.total_hypothetical_wire_latency_ms >= summary.query_count * 500.0
            # Token counts must be positive for querying policies
            assert summary.total_prompt_tokens > 0
            assert summary.total_completion_tokens > 0
            assert summary.normalized_compute_cost > 0.0
        else:
            # Non-querying baselines (B0, B1, B2)
            assert summary.total_hypothetical_wire_latency_ms == 0.0
            assert summary.total_prompt_tokens == 0
            assert summary.total_completion_tokens == 0
            assert summary.normalized_compute_cost == 0.0


def test_decoupled_query_and_action(
    mock_candidate_tracks: List[Dict[str, Any]],
    mock_cached_provider: CachedVLMProvider,
    mock_calibrated_model: CalibratedUtilityModel,
) -> None:
    """Verify that VLM querying is distinct from downstream SLAM intervention."""
    results = run_fair_policy_comparison_suite(
        tracks=mock_candidate_tracks,
        vlm_provider=mock_cached_provider,
        calibrated_model=mock_calibrated_model,
        lambda_cost=0.45,
    )

    b6_decisions = results["B6_Estimator_Aware"].decisions

    # Verify presence of queried + RETAIN (e.g. monitor resolved as static landmark)
    queried_and_retained = [
        d for d in b6_decisions if d["vlm_queried"] is True and d["action"] == "RETAIN"
    ]
    assert len(queried_and_retained) >= 1, "Expected at least one entity queried but retained as static landmark"
    assert queried_and_retained[0]["resolved_affordance"] == "IMMOVABLE"

    # Verify presence of queried + MASK (e.g. person or moving chair)
    queried_and_masked = [
        d for d in b6_decisions if d["vlm_queried"] is True and d["action"] == "MASK_FROM_TRACKING"
    ]
    assert len(queried_and_masked) >= 1, "Expected at least one entity queried and masked"


def test_zero_ground_truth_leakage(
    mock_candidate_tracks: List[Dict[str, Any]],
    mock_cached_provider: CachedVLMProvider,
) -> None:
    """Verify that routing policies do not access ground truth fields."""
    tracks_with_gt_poison = copy.deepcopy(mock_candidate_tracks)
    for t in tracks_with_gt_poison:
        # Inject deceptive ground truth tags
        t["groundtruth_action"] = "INVENTED_GT_LEAKAGE"
        t["groundtruth_class"] = "INVENTED_CLASS"
        t["gt_pose"] = [999.0, 999.0, 999.0]

    results = run_fair_policy_comparison_suite(
        tracks=tracks_with_gt_poison,
        vlm_provider=mock_cached_provider,
    )

    # Decisions must never contain poisoned tags
    for pkey, summary in results.items():
        for d in summary.decisions:
            assert "INVENTED_GT_LEAKAGE" not in d["action"]
            assert "INVENTED_CLASS" not in d["resolved_affordance"]


def test_reproducibility_with_seed(
    mock_candidate_tracks: List[Dict[str, Any]],
    mock_cached_provider: CachedVLMProvider,
) -> None:
    """Verify that B4 (Random) generates identical queries under the same seed."""
    b4_run1 = evaluate_b4_random(
        tracks=mock_candidate_tracks,
        vlm_provider=mock_cached_provider,
        target_budget_k=4,
        random_seed=1234,
    )
    b4_run2 = evaluate_b4_random(
        tracks=mock_candidate_tracks,
        vlm_provider=mock_cached_provider,
        target_budget_k=4,
        random_seed=1234,
    )
    b4_run3 = evaluate_b4_random(
        tracks=mock_candidate_tracks,
        vlm_provider=mock_cached_provider,
        target_budget_k=4,
        random_seed=9999,
    )

    tids_1 = [d["track_id"] for d in b4_run1.decisions if d["vlm_queried"]]
    tids_2 = [d["track_id"] for d in b4_run2.decisions if d["vlm_queried"]]
    tids_3 = [d["track_id"] for d in b4_run3.decisions if d["vlm_queried"]]

    assert tids_1 == tids_2, "Same seed produced different random query selections"
    assert tids_1 != tids_3 or len(tids_1) == 0, "Different seeds unexpectedly produced identical selections"

