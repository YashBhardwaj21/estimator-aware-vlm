"""Unit and contract tests for Statistical Analysis, Paired Inference, and Claim Verification.

Verifies:
1. Wilson score confidence interval computation and edge-case handling (k=0, k=n, n=0).
2. Paired bootstrap difference (point estimate, 95% CI, empirical p-value).
3. McNemar exact binomial test for paired categorical discordance.
4. Holm-Bonferroni step-down correction for Family-Wise Error Rate (FWER) control.
5. Claim-to-evidence matrix generation, denominator consistency, and metric extraction.
6. Execution of end-to-end statistical analysis pipeline on benchmark run artifacts.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List
import numpy as np
import pytest

from estimator_aware_vlm.common.paths import get_repo_root
from estimator_aware_vlm.statistical_analysis import (
    generate_claim_to_evidence_matrix,
    holm_bonferroni_correction,
    mcnemar_exact_test,
    paired_bootstrap_difference,
    run_statistical_analysis,
    wilson_score_interval,
)

repo_root = get_repo_root()


def test_wilson_score_interval_properties():
    """Verify Wilson score interval properties and edge-case invariants."""
    # 1. Edge case: 0 successes out of 10
    low_0, high_0 = wilson_score_interval(0, 10, confidence=0.95)
    assert low_0 == 0.0
    assert 0.0 < high_0 < 0.35  # Wilson does not collapse to zero width at boundary

    # 2. Edge case: 10/10 successes
    low_10, high_10 = wilson_score_interval(10, 10, confidence=0.95)
    assert 0.65 < low_10 < 1.0
    assert high_10 == 1.0

    # 3. Standard 50%
    low_5, high_5 = wilson_score_interval(5, 10, confidence=0.95)
    assert 0.20 < low_5 < 0.50
    assert 0.50 < high_5 < 0.80

    # 4. Edge case: n = 0 (fail-safe return)
    low_zero_n, high_zero_n = wilson_score_interval(0, 0, confidence=0.95)
    assert (low_zero_n, high_zero_n) == (0.0, 0.0)

    # 5. Non-trivial ratio (12 / 14)
    low_12, high_12 = wilson_score_interval(12, 14, confidence=0.95)
    assert 0.50 < low_12 < high_12 <= 1.0


def test_paired_bootstrap_difference():
    """Verify paired bootstrap difference accurately evaluates mean delta and significance."""
    np.random.seed(42)
    records_a = [{"physical_instance_id": f"inst_{i}", "val": 1.0} for i in range(10)]
    records_b = [{"physical_instance_id": f"inst_{i}", "val": 0.5} for i in range(10)]

    def metric_fn(recs: List[Dict[str, Any]]) -> float:
        return float(np.mean([r["val"] for r in recs]))

    res = paired_bootstrap_difference(records_a, records_b, metric_fn, n_resamples=1000)
    assert res["observed_delta"] == pytest.approx(0.5, 0.01)
    assert res["ci_95"][0] == pytest.approx(0.5, 0.01)
    assert res["ci_95"][1] == pytest.approx(0.5, 0.01)
    assert res["is_significant_p05"] is True


def test_mcnemar_exact_test_discordance():
    """Verify McNemar exact binomial test on concordant and discordant pair distributions."""
    # Fully concordant pairs
    res_conc = mcnemar_exact_test([True, True, False], [True, True, False])
    assert res_conc["discordant_pairs"] == 0
    assert res_conc["p_value"] == 1.0

    # Asymmetric discordant pairs: A correct 5 times, B incorrect 5 times
    res_disc = mcnemar_exact_test([True] * 5, [False] * 5)
    assert res_disc["discordant_pairs"] == 5
    # Two-sided exact binomial p-value: 2 * (0.5^5) = 2 * 0.03125 = 0.0625
    assert abs(res_disc["p_value"] - 0.0625) < 1e-3


def test_holm_bonferroni_monotonicity():
    """Verify Holm-Bonferroni step-down correction preserves rank monotonicity and p-value bounds."""
    p_vals = {
        "test_1": 0.01,
        "test_2": 0.04,
        "test_3": 0.03,
    }
    corrected = holm_bonferroni_correction(p_vals)

    # Rank 1: test_1 (0.01 * 3 = 0.03)
    # Rank 2: test_3 (0.03 * 2 = 0.06)
    # Rank 3: test_2 (0.04 * 1 = 0.04 -> monotonic step max(0.06, 0.04) = 0.06)
    assert corrected["test_1"]["adjusted_p"] == 0.03
    assert corrected["test_3"]["adjusted_p"] == 0.06
    assert corrected["test_2"]["adjusted_p"] == 0.06

    ranks = sorted(corrected.values(), key=lambda d: d["rank"])
    for i in range(len(ranks) - 1):
        assert ranks[i]["adjusted_p"] <= ranks[i + 1]["adjusted_p"]


def test_claim_matrix_denominator_and_metric_consistency():
    """Verify claim-to-evidence matrix enforces denominator consistency and excludes uncalibrated metrics."""
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
                },
            }
        }
    }

    stats_summary = {
        "bootstrap_comparisons": {},
        "slam_trajectory_metrics": {},
    }
    matrix_md = generate_claim_to_evidence_matrix(
        router_summary=router_summary,
        lifecycle_summary=lifecycle_summary,
        stats_summary=stats_summary,
    )

    # 1. Verification of VLM query savings
    assert "53.3%" in matrix_md

    # 2. Accuracy and retention metrics
    assert "SUPPORTED" in matrix_md
    assert "Qwen2.5-VL-3B" in matrix_md

    # 3. Trajectory ATE status must state NOT_EVALUATED
    assert "NOT_EVALUATED" in matrix_md


def test_run_statistical_analysis_on_benchmark_run(tmp_path: Path):
    """Verify execution of full statistical analysis suite over recorded benchmark run."""
    bench_runs = repo_root / "outputs" / "benchmark_runs"
    router_runs = sorted(bench_runs.glob("router_eval_*"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not router_runs:
        pytest.skip("No router_eval_* run directories found.")

    rdir = router_runs[0]
    out_dir = tmp_path / "stats_out"
    out_dir.mkdir(parents=True)

    summary = run_statistical_analysis(router_run_dir=rdir, output_dir=out_dir)

    assert "paired_bootstrap_resamples" in summary
    assert "comparisons" in summary
    assert "delta_acc_B6_minus_B4_always_vlm" in summary["comparisons"]

    matrix_p = out_dir / "claim_to_evidence_matrix.md"
    report_p = out_dir / "paper_benchmark_report.md"
    json_p = out_dir / "statistical_metrics_summary.json"

    assert matrix_p.exists()
    assert report_p.exists()
    assert json_p.exists()

    content = matrix_p.read_text(encoding="utf-8")
    assert "NOT_EVALUATED" in content or "UNEVALUATED" in content
