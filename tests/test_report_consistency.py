"""Unit tests for Report Generation Consistency."""

import json
import pytest
import sys
from pathlib import Path

from estimator_aware_vlm.common.paths import get_outputs_dir, get_repo_root

root_dir = get_repo_root()
sys.path.insert(0, str(root_dir))

from scripts.evaluate_prerouter_evidence import generate_markdown_report as generate_evidence_markdown_report
from scripts.run_router_ablation import generate_ablation_markdown_report as generate_router_ablation_report


def test_evidence_report_consistency_with_json(tmp_path):
    """Verify that Evidence Markdown report renders all gate metrics from JSON without divergence."""
    metrics_file = get_outputs_dir() / "reports" / "prerouter_evidence_metrics.json"
    if not metrics_file.exists():
        metrics_file = get_outputs_dir() / "prerouter_evidence_metrics.json"
    if not metrics_file.exists():
        pytest.skip("prerouter_evidence_metrics.json not yet generated")

    with open(metrics_file, "r", encoding="utf-8-sig") as f:
        data = json.load(f)

    temp_report = tmp_path / "evidence_report_test.md"
    generate_evidence_markdown_report(data, temp_report)
    content = temp_report.read_text(encoding="utf-8")

    # Check that key numerical values from JSON appear accurately in the report
    depth_val = f"{data['geometry']['depth_validity_percentage']}%"
    assert depth_val in content

    false_motion = f"{data['observed_motion']['false_motion_rate_pct']}%"
    assert false_motion in content

    macro_f1 = f"{data['semantics']['macro_f1'] * 100:.2f}%"
    assert macro_f1 in content


def test_router_report_consistency_with_json(tmp_path):
    """Verify that Router Markdown report renders all ablation matrix values from JSON."""
    ablation_file = get_outputs_dir() / "router_ablation" / "router_ablation_results.json"
    if not ablation_file.exists():
        pytest.skip("router_ablation_results.json not yet generated")

    with open(ablation_file, "r", encoding="utf-8-sig") as f:
        data = json.load(f)

    temp_report = tmp_path / "router_report_test.md"
    generate_router_ablation_report(data, temp_report)
    content = temp_report.read_text(encoding="utf-8")

    # Verify that selective router accuracy and queries match JSON
    sel_m = data["experiments"]["router_selective_vlm"]["metrics"]
    acc_str = f"{sel_m['quality']['balanced_accuracy']*100:.1f}%"
    query_str = f"{sel_m['efficiency']['vlm_query_rate_pct']:.1f}%"

    assert acc_str in content
    assert query_str in content


def test_downstream_slam_effects_no_tum_and_no_fallbacks():
    """Verify Evaluation isolation contract: zero TUM references in VLM downstream evaluation and no synthetic fallbacks."""
    downstream_json = get_outputs_dir() / "reports" / "downstream_slam_effects.json"
    if not downstream_json.exists():
        downstream_json = get_outputs_dir() / "downstream_slam_effects.json"
    downstream_report = get_outputs_dir() / "reports" / "downstream_slam_effects_report.md"
    if not downstream_report.exists():
        downstream_report = get_outputs_dir() / "downstream_slam_effects_report.md"

    assert downstream_json.exists()
    assert downstream_report.exists()

    with open(downstream_json, "r", encoding="utf-8") as f:
        data = json.load(f)

    # 1. Trajectory metrics must be null/unevaluated without MoCap
    slam_m = data["slam_trajectory_metrics"]
    assert slam_m["ate_rmse_m"] is None
    assert slam_m["rpe_rmse_m"] is None
    assert slam_m["has_optical_mocap_ground_truth"] is False
    assert slam_m["trajectory_status"] == "UNEVALUATED_NO_MOCAP_GROUND_TRUTH"
    assert slam_m["evaluation_sequence"] == "my_desk_sequence"

    # 2. Zero TUM occurrences in downstream evaluation artifacts
    json_text = downstream_json.read_text(encoding="utf-8").lower()
    report_text = downstream_report.read_text(encoding="utf-8").lower()
    assert "tum_fr1_desk" not in json_text
    assert "tum" not in json_text
    assert "tum_fr1_desk" not in report_text
    assert "tum" not in report_text
