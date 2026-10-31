"""Regression and integrity tests for Experiment 8.1 Audit deliverables."""

import csv
import hashlib
import json
from pathlib import Path
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
AUDIT_DIR = REPO_ROOT / "results" / "selective_query_evaluation" / "run_20261031_selective_routing_audit"


def compute_sha256(path: Path) -> str:
    hasher = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()


def test_audit_directory_and_core_files_exist():
    assert AUDIT_DIR.exists(), f"Audit directory missing: {AUDIT_DIR}"
    expected_files = [
        "INTERVENTION_ATTRIBUTION_REPORT.md",
        "candidate_intervention_attribution.csv",
        "candidate_intervention_attribution.json",
        "tum_intervention_reconstruction.csv",
        "d455_policy_attribution.csv",
        "corrected_selection_quality_metrics.csv",
        "corrected_downstream_policy_metrics.csv",
        "corrected_statistical_analysis.json",
        "scientific_claim_verdicts.json",
        "provenance/checksums.sha256",
        "provenance/provenance_manifest.json",
    ]
    for rel_path in expected_files:
        p = AUDIT_DIR / rel_path
        assert p.exists(), f"Required audit file missing: {rel_path}"
        assert p.stat().st_size > 0, f"File is empty: {rel_path}"


def test_provenance_checksums_match():
    checksum_file = AUDIT_DIR / "provenance" / "checksums.sha256"
    assert checksum_file.exists()
    
    with open(checksum_file, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()]

    assert len(lines) >= 15, f"Expected at least 15 audited artifacts, got {len(lines)}"

    for line in lines:
        expected_sha, rel_path = line.split("  ", 1)
        target_file = AUDIT_DIR / rel_path
        assert target_file.exists(), f"Checksum target missing: {rel_path}"
        actual_sha = compute_sha256(target_file)
        assert actual_sha == expected_sha, f"Checksum mismatch for {rel_path}"


def test_candidate_intervention_classification_integrity():
    json_path = AUDIT_DIR / "candidate_intervention_attribution.json"
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    candidates = data["candidates"]
    assert len(candidates) == 63, f"Expected 63 candidates, got {len(candidates)}"

    by_id = {c["candidate_id"]: c for c in candidates}

    # TUM monitor_01
    mon = by_id["tum_rgbd__fr3_walking_xyz__single_session__none__monitor_01"]
    assert mon["intervention_classification"] == "INDEPENDENT_SINGLE_CANDIDATE_INTERVENTION"
    assert mon["is_duplicate_outcome"] is False
    assert mon["delta_metric"] == -0.0867

    # TUM person_02 and person_04 (joint outcome)
    p2 = by_id["tum_rgbd__fr3_walking_xyz__single_session__none__person_02"]
    p4 = by_id["tum_rgbd__fr3_walking_xyz__single_session__none__person_04"]
    assert p2["intervention_classification"] == "JOINT_MULTI_CANDIDATE_INTERVENTION"
    assert p4["intervention_classification"] == "JOINT_MULTI_CANDIDATE_INTERVENTION"
    assert p2["is_duplicate_outcome"] is True
    assert p4["is_duplicate_outcome"] is True

    # OpenLORIS person_01
    ol = by_id["openloris__corridor1_4__single_session__none__person_01"]
    assert ol["intervention_classification"] == "INDEPENDENT_SINGLE_CANDIDATE_INTERVENTION"
    assert ol["is_duplicate_outcome"] is False


def test_tum_intervention_reconstruction():
    csv_path = AUDIT_DIR / "tum_intervention_reconstruction.csv"
    with open(csv_path, "r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    assert len(rows) == 6, f"Expected 6 conditions, got {len(rows)}"
    by_cond = {r["condition"]: r for r in rows}

    assert "baseline_KEEP" in by_cond
    assert "monitor_01_POLYGON" in by_cond
    assert "monitor_01_BBOX" in by_cond
    assert "moving_persons_POLYGON" in by_cond
    assert "monitor_poly_plus_moving_persons" in by_cond
    assert "monitor_01_BBOX_MINUS_POLYGON" in by_cond

    # Verify tracking coverage distinction
    for r in rows:
        assert float(r["reference_timestamp_match_pct"]) == 28.64
        assert float(r["input_frame_tracking_pct"]) == 96.28
        assert int(r["common_support_poses"]) == 826

    # Verify condition ATEs
    assert float(by_cond["baseline_KEEP"]["condition_ate_rmse_m"]) == 0.3497
    assert float(by_cond["monitor_01_POLYGON"]["condition_ate_rmse_m"]) == 0.4365
    assert float(by_cond["monitor_01_BBOX"]["condition_ate_rmse_m"]) == 0.2067
    assert float(by_cond["moving_persons_POLYGON"]["condition_ate_rmse_m"]) == 0.0177
    assert float(by_cond["monitor_poly_plus_moving_persons"]["condition_ate_rmse_m"]) == 0.0182
    assert float(by_cond["monitor_01_BBOX_MINUS_POLYGON"]["condition_ate_rmse_m"]) == 0.5055


def test_d455_policy_attribution_and_residuals():
    csv_path = AUDIT_DIR / "d455_policy_attribution.csv"
    with open(csv_path, "r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    assert len(rows) == 5, f"Expected 5 policy variants, got {len(rows)}"
    by_var = {r["variant_id"]: r for r in rows}

    # Verify R5 vs R7 identical performance
    r5 = by_var["V3_GEOMETRY_PLUS_VLM"]
    r7 = by_var["V4_ESTIMATOR_STATE_PLUS_GEOMETRY_PLUS_VLM"]
    assert float(r5["f1_score"]) == 1.00
    assert float(r7["f1_score"]) == 1.00
    assert float(r5["held_out_anchor_residual_m"]) == 0.1242
    assert float(r7["held_out_anchor_residual_m"]) == 0.1242

    # Verify VLM without geometry fails relocation
    v2 = by_var["V2_VLM_ONLY_AFFORDANCE"]
    assert float(v2["f1_score"]) == 0.0
    assert v2["residual_type"] == "ANALYTICAL_SIMULATION"


def test_scientific_claim_verdicts_schema_and_integrity():
    json_path = AUDIT_DIR / "scientific_claim_verdicts.json"
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    verdicts = data["scientific_claim_verdicts"]
    assert len(verdicts) == 7

    allowed_statuses = {"SUPPORTED", "PARTIALLY_SUPPORTED", "NOT_SUPPORTED", "NOT_EVALUABLE"}
    by_claim = {}

    for v in verdicts:
        assert "claim" in v
        assert "status" in v
        assert "evidence" in v
        assert "remaining_requirement" in v
        assert v["status"] in allowed_statuses, f"Invalid status: {v['status']}"
        by_claim[v["claim"]] = v["status"]

    assert by_claim["Learned VOI generalizes"] == "NOT_EVALUABLE"
    assert by_claim["Selective querying improves OpenLORIS localization"] == "NOT_SUPPORTED"
    assert by_claim["Estimator evidence improves candidate selection"] == "PARTIALLY_SUPPORTED"
    assert by_claim["R7 adds value beyond estimator-state-only selection"] == "PARTIALLY_SUPPORTED"
    assert by_claim["VLM resolves semantic ambiguity"] == "SUPPORTED"
    assert by_claim["Selective querying improves TUM localization"] == "SUPPORTED"
    assert by_claim["Selective querying improves D455 map integrity"] == "SUPPORTED"


def test_corrected_figures_exist():
    fig_dir = AUDIT_DIR / "corrected_figures"
    assert fig_dir.exists()
    expected_figs = [
        "figure1_candidate_populations_audited.png",
        "figure2_vlm_inference_costs_audited.png",
        "figure3_corrected_selection_quality.png",
        "figure4_tum_localization_vs_budget_audited.png",
        "figure5_openloris_case_study_audited.png",
        "figure6_d455_policy_attribution_lifecycle_audited.png",
        "figure7_pareto_accuracy_vs_cost_audited.png",
        "figure8_qualitative_decision_pathways_audited.png",
    ]
    for fig_name in expected_figs:
        fp = fig_dir / fig_name
        assert fp.exists(), f"Figure missing: {fig_name}"
        assert fp.stat().st_size > 10000, f"Figure too small / corrupt: {fig_name}"

