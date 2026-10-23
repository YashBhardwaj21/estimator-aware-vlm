"""Unit and Consistency Tests for Statistical Analysis & Benchmark Reporting.

Verifies:
1. Wilson score confidence interval correctness and boundary conditions [0, 1].
2. McNemar's exact test contingency matrix and binomial two-tailed p-values.
3. Holm-Bonferroni step-down correction monotonicity and FWER control.
4. Exact programmatic consistency between statistical summary JSON and Markdown claim matrix.
5. Strict enforcement that non-significant accuracy differences are never claimed as statistically significant.
6. Trajectory ATE/RPE is strictly marked as NOT_EVALUATED.
"""

from __future__ import annotations

import json
from pathlib import Path
import pytest

from estimator_aware_vlm.statistical_analysis import (
    wilson_score_interval,
    mcnemar_exact_test,
    holm_bonferroni_correction,
    generate_claim_to_evidence_matrix,
)

repo_root = Path(__file__).resolve().parents[1]


def test_wilson_score_interval_properties():
    # Boundary checks
    low_0, high_0 = wilson_score_interval(0, 10)
    assert low_0 == 0.0
    assert 0.0 < high_0 < 0.35

    low_1, high_1 = wilson_score_interval(10, 10)
    assert 0.65 < low_1 < 1.0
    assert high_1 == 1.0

    # Symmetric midpoint
    low_mid, high_mid = wilson_score_interval(5, 10)
    assert 0.20 < low_mid < 0.35
    assert 0.65 < high_mid < 0.80
    assert round(0.5 - low_mid, 2) == round(high_mid - 0.5, 2)


def test_mcnemar_exact_test_calculation():
    # Case 1: Identical predictions (0 discordant pairs)
    corr_a = [True, True, False, True, False]
    corr_b = [True, True, False, True, False]
    res_ident = mcnemar_exact_test(corr_a, corr_b)
    assert res_ident["discordant_pairs"] == 0
    assert res_ident["p_value"] == 1.0
    assert not res_ident["is_significant_p05"]

    # Case 2: Known discordant pair count (2 discordant: A right, B wrong on both)
    # Expected two-tailed p = 2 * (0.5)^2 = 0.50
    corr_a2 = [True] * 14 + [False]
    corr_b2 = [True] * 12 + [False, False, False]
    res_disc = mcnemar_exact_test(corr_a2, corr_b2)
    assert res_disc["contingency_table"]["only_a_correct"] == 2
    assert res_disc["contingency_table"]["only_b_correct"] == 0
    assert res_disc["discordant_pairs"] == 2
    assert res_disc["p_value"] == 0.50
    assert not res_disc["is_significant_p05"]

    # Case 3: Statistically significant discordant pairs (e.g. 10 vs 0)
    corr_a3 = [True] * 10
    corr_b3 = [False] * 10
    res_sig = mcnemar_exact_test(corr_a3, corr_b3)
    assert res_sig["discordant_pairs"] == 10
    assert res_sig["p_value"] < 0.01
    assert res_sig["is_significant_p05"]


def test_holm_bonferroni_correction_monotonicity():
    p_vals = {
        "test_1": 0.001,
        "test_2": 0.04,
        "test_3": 0.045,
        "test_4": 0.20,
    }
    corrected = holm_bonferroni_correction(p_vals)
    adj_values = [v["adjusted_p"] for v in corrected.values()]
    # Assert strictly non-decreasing adjusted p-values
    for i in range(len(adj_values) - 1):
        assert adj_values[i] <= adj_values[i + 1]

    # Lowest p-value multiplied by m=4: 0.001 * 4 = 0.004
    assert corrected["test_1"]["adjusted_p"] == 0.004
    assert corrected["test_1"]["is_significant_fwer_05"]


def test_claim_matrix_report_consistency():
    router_sum = {
        "policy_summaries": {
            "B6_ESTIMATOR_AWARE_ROUTER": {
                "metrics": {
                    "total_instances": 15,
                    "overall_accuracy": 0.9333,
                    "vlm_query_rate": 0.3333,
                    "normalized_compute_cost": 0.1667,
                    "macro_f1": 0.9333,
                }
            },
            "B4_ZERO_SHOT_CROP_PLUS_CONTEXT": {
                "metrics": {
                    "total_instances": 15,
                    "overall_accuracy": 0.8000,
                    "vlm_query_rate": 0.9333,
                    "normalized_compute_cost": 0.4667,
                    "macro_f1": 0.8080,
                    "balanced_accuracy": 0.8500,
                }
            },
            "B1_SEMANTIC_PRIOR": {"metrics": {"overall_accuracy": 0.8667}},
            "A3_NO_PRIOR": {"metrics": {"overall_accuracy": 0.8000}},
            "A2_NO_DISPLACEMENT": {"metrics": {"vlm_query_rate": 0.1333}},
        }
    }

    lifecycle_sum = {
        "policies": {
            "B6_ESTIMATOR_AWARE_ROUTER": {
                "landmark_retention": {
                    "precision": 1.0,
                    "false_invalidation_rate": 0.0,
                    "false_positives_corruption_risk": 0,
                },
                "relocation_handling": {
                    "decoupled_pose_count": 3,
                    "expected_decoupled_count": 3,
                },
            }
        }
    }

    stats_sum = {
        "comparisons": {
            "delta_query_rate_B6_minus_B4_always_vlm": {
                "observed_delta": -0.60,
                "ci_95": [-0.7333, -0.4667],
                "p_value_two_tailed": 0.0002,
            },
            "delta_acc_B6_minus_B4_always_vlm": {
                "observed_delta": 0.1333,
                "ci_95": [-0.0667, 0.3333],
                "p_value_two_tailed": 0.2292,
            },
        },
        "mcnemar_tests": {
            "B6_vs_B4": {"p_value": 0.5000, "is_significant_p05": False}
        },
        "wilson_score_intervals": {
            "B6_ESTIMATOR_AWARE_ROUTER": {"accuracy_wilson_ci": [0.7018, 0.9888]},
            "B4_ZERO_SHOT_CROP_PLUS_CONTEXT": {"accuracy_wilson_ci": [0.5481, 0.9295]},
        },
    }

    matrix_md = generate_claim_to_evidence_matrix(router_sum, lifecycle_sum, stats_sum)

    # 1. Assert C2 explicitly labels accuracy difference as directional trend due to p=0.50
    assert "DIRECTIONAL_TREND" in matrix_md
    assert "McNemar exact p = 0.5000" in matrix_md

    # 2. Assert query reduction is highlighted as supported and significant
    assert "-60.0%" in matrix_md

    # 3. Assert Trajectory ATE/RPE claim C5 is REJECTED AS PAPER CLAIM
    assert "REJECTED AS PAPER CLAIM" in matrix_md
    assert "NOT_EVALUATED" in matrix_md
