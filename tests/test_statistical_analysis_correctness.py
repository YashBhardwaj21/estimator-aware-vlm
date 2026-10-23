"""Regression tests for statistical analysis correctness, denominator consistency, and claim checks."""

import json
from pathlib import Path
import pytest

repo_root = Path(__file__).resolve().parents[1]
from estimator_aware_vlm.statistical_analysis import (
    wilson_score_interval,
    mcnemar_exact_test,
    holm_bonferroni_correction,
    generate_claim_to_evidence_matrix,
)


def test_wilson_score_exact_denominators():
    # Test valid denominator behavior
    low, high = wilson_score_interval(12, 14)
    assert 0.5 < low < high <= 1.0

    # Test edge case n=0
    low0, high0 = wilson_score_interval(0, 0)
    assert (low0, high0) == (0.0, 0.0)


def test_mcnemar_test_discordance():
    # Concordant pairs only
    res = mcnemar_exact_test([True, True, False], [True, True, False])
    assert res["discordant_pairs"] == 0
    assert res["p_value"] == 1.0

    # Discordant: A wins 5 times, B wins 0 times
    res_disc = mcnemar_exact_test([True] * 5, [False] * 5)
    assert res_disc["discordant_pairs"] == 5
    # Exact binomial two-sided: 2 * (0.5^5) = 2 * 0.03125 = 0.0625
    assert abs(res_disc["p_value"] - 0.0625) < 1e-3


def test_holm_bonferroni_monotonicity():
    p_vals = {
        "test_1": 0.01,
        "test_2": 0.04,
        "test_3": 0.03,
    }
    corrected = holm_bonferroni_correction(p_vals)
    # Sorted order: test_1 (0.01 * 3 = 0.03), test_3 (0.03 * 2 = 0.06), test_2 (0.04 * 1 = 0.04 -> max(0.06, 0.04) = 0.06)
    assert corrected["test_1"]["adjusted_p"] == 0.03
    assert corrected["test_3"]["adjusted_p"] == 0.06
    assert corrected["test_2"]["adjusted_p"] == 0.06
    # Adjusted p-values must be non-decreasing with rank
    ranks = sorted(corrected.values(), key=lambda d: d["rank"])
    for i in range(len(ranks) - 1):
        assert ranks[i]["adjusted_p"] <= ranks[i + 1]["adjusted_p"]


def test_claim_matrix_denominator_and_ate_exclusion():
    router_summary = {
        "policy_summaries": {
            "B6_ESTIMATOR_AWARE_ROUTER": {
                "metrics": {
                    "total_instances": 15,
                    "overall_accuracy": 0.8667,
                    "vlm_query_rate": 0.5333,
                    "vlm_queries_count": 8,
                    "normalized_compute_cost": 0.533,
                    "balanced_accuracy": 0.85,
                    "macro_f1": 0.85,
                }
            },
            "B4_ZERO_SHOT_CROP_PLUS_CONTEXT": {
                "metrics": {
                    "total_instances": 15,
                    "overall_accuracy": 0.8000,
                    "vlm_query_rate": 0.9333,
                    "vlm_queries_count": 14,
                    "normalized_compute_cost": 0.933,
                    "balanced_accuracy": 0.78,
                    "macro_f1": 0.79,
                }
            },
            "B1_SEMANTIC_PRIOR": {"metrics": {}},
            "A3_NO_PRIOR": {"metrics": {}},
            "A2_NO_DISPLACEMENT": {"metrics": {}},
        }
    }

    lifecycle_summary = {
        "policies": {
            "B6_ESTIMATOR_AWARE_ROUTER": {
                "landmark_retention": {
                    "precision": 1.0,
                    "false_invalidation_rate": 0.0,
                    "false_positives_corruption_risk": 0,
                    "retained_landmarks_count": 12,
                    "false_invalidation_count": 0,
                    "ground_truth_stationary_count": 12,
                },
                "relocation_handling": {
                    "decoupled_pose_count": 1,
                    "expected_decoupled_count": 1,
                }
            }
        }
    }

    stats_summary = {
        "comparisons": {
            "delta_query_rate_B6_minus_B4_always_vlm": {"observed_delta": -0.40, "ci_95": [-0.60, -0.20]},
            "delta_acc_B6_minus_B4_always_vlm": {"observed_delta": 0.0667, "ci_95": [-0.10, 0.25]},
        },
        "mcnemar_tests": {
            "B6_vs_B4": {"p_value": 0.625}
        },
        "wilson_score_intervals": {
            "B6_ESTIMATOR_AWARE_ROUTER": {"accuracy_wilson_ci": [0.62, 0.96]},
            "B4_ZERO_SHOT_CROP_PLUS_CONTEXT": {"accuracy_wilson_ci": [0.55, 0.93]},
        }
    }

    matrix_md = generate_claim_to_evidence_matrix(router_summary, lifecycle_summary, stats_summary)

    # 1. Denominator consistency: C3 Wilson CI must NOT copy B6 accuracy CI ([62.0%, 96.0%])
    # Retention was 12/12, Wilson CI for 12/12 is [75.8%, 100.0%]
    assert "Retention Wilson 95% CI: `[75.8, 100.0]%`" in matrix_md
    assert "Wilson 95% CI: `[62.0, 96.0]%`" not in matrix_md.split("| **C3** |")[1].split("| **C4** |")[0]

    # 2. Distinct reporting of absolute pp vs relative reduction in calls
    # Absolute: 93.3% - 53.3% = 40.0 pp
    # Relative: (14 - 8) / 14 = 42.9%
    assert "40.0 pp absolute reduction" in matrix_md
    assert "42.9% relative call reduction" in matrix_md

    # 3. Trajectory ATE/RPE claim must be strictly NOT_EVALUATED and REJECTED AS PAPER CLAIM
    assert "| **C5** |" in matrix_md
    assert "NOT_EVALUATED" in matrix_md
    assert "REJECTED AS PAPER CLAIM" in matrix_md
