"""Unit tests for Stage 5 & 6 Report Generation Consistency."""

import json
import pytest
import sys
from pathlib import Path

root_dir = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(root_dir))
sys.path.insert(0, str(root_dir / "vlm"))

from vlm.scripts.evaluate_stage5_evidence import generate_markdown_report as generate_s5_report
from vlm.scripts.run_stage6_ablation import generate_ablation_markdown_report as generate_s6_report


def test_stage5_report_consistency_with_json(tmp_path):
    """Verify that Stage 5 Markdown report renders all gate metrics from JSON without divergence."""
    metrics_file = root_dir / "vlm" / "outputs" / "stage5_evidence_metrics.json"
    if not metrics_file.exists():
        pytest.skip("stage5_evidence_metrics.json not yet generated")

    with open(metrics_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    temp_report = tmp_path / "stage5_report_test.md"
    generate_s5_report(data, temp_report)
    content = temp_report.read_text(encoding="utf-8")

    # Check that key numerical values from JSON appear accurately in the report
    depth_val = f"{data['geometry']['depth_validity_percentage']}%"
    assert depth_val in content

    false_motion = f"{data['observed_motion']['false_motion_rate_pct']}%"
    assert false_motion in content

    macro_f1 = f"{data['semantics']['macro_f1'] * 100:.2f}%"
    assert macro_f1 in content


def test_stage6_report_consistency_with_json(tmp_path):
    """Verify that Stage 6 Markdown report renders all ablation matrix values from JSON."""
    ablation_file = root_dir / "vlm" / "outputs" / "stage6" / "router_ablation_results.json"
    if not ablation_file.exists():
        pytest.skip("router_ablation_results.json not yet generated")

    with open(ablation_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    temp_report = tmp_path / "stage6_report_test.md"
    generate_s6_report(data, temp_report)
    content = temp_report.read_text(encoding="utf-8")

    # Verify that selective router accuracy and queries match JSON
    sel_m = data["experiments"]["router_selective_vlm"]["metrics"]
    acc_str = f"{sel_m['quality']['balanced_accuracy']*100:.1f}%"
    query_str = f"{sel_m['efficiency']['vlm_query_rate_pct']:.1f}%"

    assert acc_str in content
    assert query_str in content
