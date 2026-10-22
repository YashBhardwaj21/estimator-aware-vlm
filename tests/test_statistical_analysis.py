"""Unit and Integration Tests for Statistical Analysis & Paired Inference."""

from __future__ import annotations

import json
from pathlib import Path
import pytest
import numpy as np

from vlm.src.statistical_analysis import (
    wilson_score_interval,
    paired_bootstrap_difference,
    run_statistical_analysis,
    generate_claim_to_evidence_matrix,
)

repo_root = Path(__file__).resolve().parent.parent.parent


def test_wilson_score_interval():
    # 1. Edge case: 0 successes
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


def test_paired_bootstrap_difference():
    records_a = [
        {"physical_instance_id": f"inst_{i}", "val": 1.0} for i in range(10)
    ]
    records_b = [
        {"physical_instance_id": f"inst_{i}", "val": 0.5} for i in range(10)
    ]

    def metric_fn(recs):
        return float(np.mean([r["val"] for r in recs]))

    res = paired_bootstrap_difference(records_a, records_b, metric_fn, n_resamples=1000)
    assert res["observed_delta"] == pytest.approx(0.5, 0.01)
    assert res["ci_95"][0] == pytest.approx(0.5, 0.01)
    assert res["ci_95"][1] == pytest.approx(0.5, 0.01)
    assert res["is_significant_p05"] is True


def test_run_statistical_analysis_on_latest_run(tmp_path):
    # Find latest router run
    bench_runs = repo_root / "vlm" / "outputs" / "benchmark_runs"
    router_runs = sorted(bench_runs.glob("router_eval_*"), key=lambda p: p.stat().st_mtime, reverse=True)
    assert router_runs, "No router_eval_* run directories found"
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
    assert "NOT_EVALUATED" in content
    assert "Qwen2.5-VL-3B" in content
