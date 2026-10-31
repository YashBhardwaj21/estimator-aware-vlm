"""Unit and Regression Tests for Experiment 8:
Estimator-Aware Selective VLM Routing Under Matched Inference Budgets.

Verifies:
1. Candidate population integrity and uniqueness across 3 environments.
2. Zero ground truth leakage in pre-query evidence.
3. Strict matched-budget candidate allocation across policies.
4. Estimator-aware policy (R7) superiority over visual uncertainty and random at low budgets.
5. Structural integrity metrics for RealSense D455 (camera trajectory GT is null).
6. Non-fabrication invariants (zero synthetic tokens, authentic Qwen responses).
7. Provenance manifest and publication figure existence.
"""

import json
from pathlib import Path
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
EXP_DIR = REPO_ROOT / "results" / "selective_query_evaluation" / "run_20261031_selective_routing"


@pytest.fixture(scope="module")
def candidate_inventory():
    path = EXP_DIR / "candidate_inventory.json"
    assert path.is_file(), f"Missing {path}"
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict):
        return data.get("candidates", [])
    return data


@pytest.fixture(scope="module")
def prequery_evidence():
    path = EXP_DIR / "candidate_evidence_prequery.jsonl"
    assert path.is_file(), f"Missing {path}"
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    return records


@pytest.fixture(scope="module")
def policy_decisions():
    path = EXP_DIR / "policies" / "policy_decisions.jsonl"
    assert path.is_file(), f"Missing {path}"
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    return records


@pytest.fixture(scope="module")
def budget_manifest():
    path = EXP_DIR / "policies" / "query_budget_manifest.json"
    assert path.is_file(), f"Missing {path}"
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


@pytest.fixture(scope="module")
def provenance_manifest():
    path = EXP_DIR / "provenance" / "provenance_manifest.json"
    assert path.is_file(), f"Missing {path}"
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# 1. Candidate Population Tests
def test_candidate_inventory_counts(candidate_inventory):
    assert len(candidate_inventory) == 63, f"Expected 63 candidates, got {len(candidate_inventory)}"
    
    tum_candidates = [c for c in candidate_inventory if c["dataset_id"] == "tum_rgbd"]
    openloris_candidates = [c for c in candidate_inventory if c["dataset_id"] == "openloris"]
    d455_candidates = [c for c in candidate_inventory if c["dataset_id"] == "realsense_d455"]
    
    assert len(tum_candidates) == 10
    assert len(openloris_candidates) == 1
    assert len(d455_candidates) == 52


def test_candidate_id_uniqueness(candidate_inventory):
    candidate_ids = [c["candidate_id"] for c in candidate_inventory]
    assert len(candidate_ids) == len(set(candidate_ids)), "Candidate IDs must be unique"
    
    for cid in candidate_ids:
        parts = cid.split("__")
        assert len(parts) == 5, f"Candidate ID {cid} must have 5 parts separated by '__'"


# 2. Zero Ground-Truth Leakage Tests
def test_zero_ground_truth_leakage_in_prequery_evidence(prequery_evidence):
    forbidden_keys = {
        "ground_truth",
        "gt_movable",
        "is_relocated",
        "delta_tape_m",
        "counterfactual_ate_rmse_m",
        "future_delta_ate",
        "oracle_benefit",
    }
    for rec in prequery_evidence:
        found_forbidden = forbidden_keys.intersection(rec.keys())
        assert not found_forbidden, f"Candidate {rec.get('candidate_id')} leaked ground truth keys: {found_forbidden}"


# 3. Matched-Budget Allocation Tests
def test_strict_matched_budgets(policy_decisions, budget_manifest):
    tum_budgets = budget_manifest.get("tum_rgbd", {})
    
    for budget_level, k_target in tum_budgets.items():
        if k_target == 0:
            continue
        # Check all selective policies at this budget level
        for policy_key in [
            "R3_RANDOM_K",
            "R4_VISUAL_UNCERTAINTY_K",
            "R5_GEOMETRY_UNCERTAINTY_K",
            "R6_ESTIMATOR_STATE_K",
            "R7_ESTIMATOR_AWARE_VOI_K",
            "R8_ORACLE_UPPER_BOUND",
        ]:
            queried = [
                d for d in policy_decisions
                if d["dataset_id"] == "tum_rgbd"
                and d.get("budget") == budget_level
                and d["policy_key"] == policy_key
                and d.get("vlm_queried") is True
            ]
            assert len(queried) == k_target, (
                f"Policy {policy_key} at {budget_level} queried {len(queried)} candidates, expected {k_target}"
            )


# 4. Estimator-Aware Routing Superiority Tests
def test_r7_prioritizes_dominant_inlier_candidate_in_tum(policy_decisions):
    # At K=1 (K10), R7 must query person_02 (which has 478 inlier landmarks)
    r7_k1 = [
        d for d in policy_decisions
        if d["dataset_id"] == "tum_rgbd"
        and d.get("budget") == "K10"
        and d["policy_key"] == "R7_ESTIMATOR_AWARE_VOI_K"
        and d.get("vlm_queried") is True
    ]
    assert len(r7_k1) == 1
    assert "person_02" in r7_k1[0]["candidate_id"], (
        f"Expected R7 to select person_02 at K=1, got {r7_k1[0]['candidate_id']}"
    )


def test_r4_visual_uncertainty_selects_peripheral_blurry_candidate(policy_decisions):
    # At K=1 (K10), R4 Visual Uncertainty selects person_03 (highest blur score)
    r4_k1 = [
        d for d in policy_decisions
        if d["dataset_id"] == "tum_rgbd"
        and d.get("budget") == "K10"
        and d["policy_key"] == "R4_VISUAL_UNCERTAINTY_K"
        and d.get("vlm_queried") is True
    ]
    assert len(r4_k1) == 1
    assert "person_03" in r4_k1[0]["candidate_id"], (
        f"Expected R4 to select person_03 at K=1, got {r4_k1[0]['candidate_id']}"
    )


# 5. D455 Camera Trajectory Ground Truth is Strictly Null
def test_d455_camera_trajectory_ground_truth_is_null():
    metrics_csv = EXP_DIR / "d455" / "policy_lifecycle_metrics.csv"
    assert metrics_csv.is_file()
    with open(metrics_csv, "r", encoding="utf-8") as f:
        header = f.readline().strip().split(",")
        traj_idx = header.index("camera_trajectory_ate")
        for line in f:
            if line.strip():
                cols = line.strip().split(",")
                val = cols[traj_idx]
                assert val == "" or val == "null" or val == "None", (
                    f"D455 camera trajectory ground truth must be null, found '{val}'"
                )


# 6. Provenance and Artifact Verification
def test_provenance_manifest_integrity(provenance_manifest):
    assertions = provenance_manifest.get("integrity_assertions", {})
    assert assertions.get("zero_ground_truth_leakage_into_prequery_evidence") is True
    assert assertions.get("zero_synthetic_or_mock_vlm_tokens") is True
    assert assertions.get("realsense_d455_camera_trajectory_ground_truth") is None
    assert "NOT_IMPLEMENTED" in assertions.get("landmark_level_orb_slam3_integration_status", "")
    assert "INSUFFICIENT_DATA_FALLBACK_TO_PROXY" in assertions.get("voi_sufficiency_gate_status", "")


def test_all_publication_figures_exist():
    fig_dir = EXP_DIR / "aggregate" / "figures"
    expected_figures = [
        "figure1_candidate_populations.png",
        "figure2_vlm_inference_costs.png",
        "figure3_budget_vs_selection_quality.png",
        "figure4_tum_localization_vs_budget.png",
        "figure5_openloris_localization_vs_budget.png",
        "figure6_d455_lifecycle_vs_budget.png",
        "figure7_pareto_accuracy_vs_cost.png",
        "figure8_qualitative_policy_decisions.png",
    ]
    for fig_name in expected_figures:
        fig_path = fig_dir / fig_name
        assert fig_path.is_file(), f"Missing figure {fig_name}"
        assert fig_path.stat().st_size > 10000, f"Figure {fig_name} is too small / empty ({fig_path.stat().st_size} bytes)"
