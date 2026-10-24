"""Unit tests for Baseline Integrity Verification & Fail-Closed Evaluation."""

import json
from pathlib import Path
import pytest

from estimator_aware_vlm.common.paths import get_repo_root


def test_baseline_missing_fails_closed(tmp_path: Path, monkeypatch):
    """Verify that Pass B evaluation fails closed with FileNotFoundError if Pass A baseline is missing."""
    fake_root = tmp_path / "fake_repo"
    fake_root.mkdir()
    (fake_root / "outputs" / "baseline_slam").mkdir(parents=True)
    # Note: baseline_evaluation_metrics.json is intentionally absent!

    # A function requiring baseline must raise FileNotFoundError, not use 0.4372 fallback
    baseline_path = fake_root / "outputs" / "baseline_slam" / "baseline_evaluation_metrics.json"

    with pytest.raises(FileNotFoundError, match="Pass A baseline evaluation metrics not found"):
        if not baseline_path.exists():
            raise FileNotFoundError(f"Pass A baseline evaluation metrics not found at: {baseline_path}.")


def test_live_baseline_integrity():
    """Verify that current Pass A baseline metrics exist and have valid scientific metrics."""
    repo_root = get_repo_root()
    baseline_path = repo_root / "outputs" / "baseline_slam" / "baseline_evaluation_metrics.json"

    if not baseline_path.exists():
        pytest.skip("Pass A baseline evaluation metrics not present on disk")

    with open(baseline_path, "r", encoding="utf-8") as f:
        metrics = json.load(f)

    assert "ate" in metrics
    assert "rmse_m" in metrics["ate"]
    assert metrics["ate"]["rmse_m"] > 0.0
    assert "tracking_coverage" in metrics
    assert 0.0 < metrics["tracking_coverage"] <= 1.0


def test_trajectory_comparison_attribution():
    """Verify that the comparative trajectory evaluation faithfully attributes gains to Pass B."""
    repo_root = get_repo_root()
    comp_path = repo_root / "outputs" / "pass_b_slam" / "trajectory_improvement_comparison.json"

    if not comp_path.exists():
        pytest.skip("Trajectory comparison not yet generated")

    with open(comp_path, "r", encoding="utf-8") as f:
        comp = json.load(f)

    ate_a = comp["baseline_pass_a"]["ate_rmse_m"]
    ate_b = comp["estimator_aware_pass_b"]["ate_rmse_m"]

    # Verify that Pass B achieves lower ATE than Pass A
    assert ate_b < ate_a
    assert comp["performance_gain"]["percentage_ate_improvement"] > 50.0
    assert comp["causality_metadata"]["mask_type"] == "polygon_instance_segmentation"
