"""Unit and Mock Integration Tests for Axis 2: Router Policy Suite & Baselines.

Verifies:
1. extract_evidence_from_input_record converts inference inputs to evidence without ground truth leakage.
2. execute_policy_decision behaves correctly across all baselines B0-B7 and ablations A1-A4.
3. compute_policy_evaluation_metrics accurately calculates query rates, costs, and accuracies.
4. evaluate_router_policies runs end-to-end in replay mode and generates valid output files.
"""

from __future__ import annotations

import json
from pathlib import Path
import pytest

from scripts.evaluate_router_policies import (
    extract_evidence_from_input_record,
    execute_policy_decision,
    compute_policy_evaluation_metrics,
    evaluate_router_policies,
    POLICY_KEYS,
)
from estimator_aware_vlm.router_policy import RouterDecision

repo_root = Path(__file__).resolve().parents[1]


def test_extract_evidence_zero_leakage():
    input_rec = {
        "physical_instance_id": "inst_chair_002",
        "semantic_class": "chair",
        "source_session": "session_01",
        "source_obs_ids": ["chair_002", "chair_001"],
        "visual_inputs": {
            "has_images": True,
            "usable_views": 1,
            "crop_view_paths": ["vlm/outputs/multiview_samples/inst_chair_002/crop_view_0.jpg"],
            "scene_view_paths": ["vlm/outputs/multiview_samples/inst_chair_002/scene_view_0.jpg"],
            "image_sha256": ["hash1", "hash2"],
        },
        "split": "PILOT_BENCHMARK_FROZEN",
        "estimator_evidence": {
            "observed_motion": "DISPLACED",
            "world_displacement_m": 0.616,
            "displacement_uncertainty_m": 0.067,
            "has_valid_pose": True,
            "depth_valid": True,
        },
    }
    evidence = extract_evidence_from_input_record(input_rec)

    assert evidence["entity_id"] == "inst_chair_002"
    assert evidence["semantic_class"] == "chair"
    assert evidence["class_prior_mobility"] == "MOVABLE"
    assert evidence["observed_motion"] == "DISPLACED"
    assert evidence["has_images"] is True
    assert evidence["views_count"] == 1

    # Check that NO ground-truth labels are present
    assert "contextual_movability_gt" not in evidence
    assert "current_motion_gt" not in evidence
    assert "expected_decision" not in evidence


def test_policy_decisions_behavior():
    # 1. Static door (Canonical immovable)
    ev_door = {
        "entity_id": "inst_door_001",
        "semantic_class": "door",
        "class_prior_mobility": "IMMOVABLE",
        "observed_motion": "STATIC",
        "world_displacement_m": 0.0,
        "displacement_uncertainty_m": 0.02,
        "has_images": True,
        "views_count": 1,
        "has_valid_pose": True,
        "depth_valid": True,
    }

    # B0: Majority -> MOVABLE
    dec_b0, _ = execute_policy_decision("B0_MAJORITY", ev_door)
    assert dec_b0.decision == "RESOLVE_CHEAP"
    assert dec_b0.cheap_label == "MOVABLE"

    # B1: Prior -> IMMOVABLE
    dec_b1, _ = execute_policy_decision("B1_SEMANTIC_PRIOR", ev_door)
    assert dec_b1.decision == "RESOLVE_CHEAP"
    assert dec_b1.cheap_label == "IMMOVABLE"

    # B6: Estimator-Aware -> Static door resolves cheaply to IMMOVABLE without VLM
    dec_b6, _ = execute_policy_decision("B6_ESTIMATOR_AWARE_ROUTER", ev_door)
    assert dec_b6.decision == "RESOLVE_CHEAP"
    assert dec_b6.cheap_label == "IMMOVABLE"
    assert dec_b6.vlm_required is False

    # 2. Displaced chair (Requires VLM escalation in B6)
    ev_chair_displaced = {
        "entity_id": "inst_chair_002",
        "semantic_class": "chair",
        "class_prior_mobility": "MOVABLE",
        "observed_motion": "DISPLACED",
        "world_displacement_m": 0.616,
        "displacement_uncertainty_m": 0.067,
        "has_images": True,
        "views_count": 1,
        "has_valid_pose": True,
        "depth_valid": True,
    }

    dec_chair_b6, vmode_chair = execute_policy_decision("B6_ESTIMATOR_AWARE_ROUTER", ev_chair_displaced)
    assert dec_chair_b6.decision == "ESCALATE"
    assert dec_chair_b6.vlm_required is True
    assert vmode_chair == "CROP_PLUS_CONTEXT"

    # A4: Crop-only router uses CROP_ONLY
    dec_a4, vmode_a4 = execute_policy_decision("A4_CROP_ONLY", ev_chair_displaced)
    assert dec_a4.decision == "ESCALATE"
    assert vmode_a4 == "CROP_ONLY"


def test_compute_policy_metrics():
    gt_map = {
        "inst_1": {"contextual_movability_gt": "IMMOVABLE"},
        "inst_2": {"contextual_movability_gt": "MOVABLE"},
        "inst_3": {"contextual_movability_gt": "MOVABLE"},
        "inst_4": {"contextual_movability_gt": "MOBILE"},
    }

    eval_recs = [
        {"physical_instance_id": "inst_1", "decision": "RESOLVE_CHEAP", "vlm_invoked": False, "final_prediction": "IMMOVABLE", "total_latency_ms": 0.1},
        {"physical_instance_id": "inst_2", "decision": "RESOLVE_CHEAP", "vlm_invoked": False, "final_prediction": "MOVABLE", "total_latency_ms": 0.1},
        {"physical_instance_id": "inst_3", "decision": "ESCALATE", "vlm_invoked": True, "final_prediction": "MOVABLE", "total_latency_ms": 1500.0},
        {"physical_instance_id": "inst_4", "decision": "RESOLVE_CHEAP", "vlm_invoked": False, "final_prediction": "MOBILE", "total_latency_ms": 0.1},
    ]

    metrics = compute_policy_evaluation_metrics(eval_recs, gt_map, vlm_cost=0.5)
    assert metrics["total_instances"] == 4
    assert metrics["vlm_queries_count"] == 1
    assert metrics["vlm_query_rate"] == 0.25
    assert metrics["overall_accuracy"] == 1.0
    assert metrics["balanced_accuracy"] == 1.0
    assert metrics["selective_accuracy_on_vlm"] == 1.0
    assert metrics["cheap_accuracy"] == 1.0
    assert metrics["normalized_compute_cost"] == pytest.approx(0.125, 0.001)


def test_evaluate_router_policies_replay(tmp_path):
    """Test full router evaluation pipeline in replay mode with synthetic replay files."""
    # Create synthetic replay directory
    mock_run = tmp_path / "mock_vis_run"
    mock_run.mkdir(parents=True)

    inputs_path = repo_root / "benchmarks" / "paper_benchmark_inputs.json"
    with open(inputs_path, "r", encoding="utf-8") as f:
        instances = json.load(f)["instances"]

    gt_path = repo_root / "benchmarks" / "paper_benchmark_ground_truth.json"
    with open(gt_path, "r", encoding="utf-8") as f:
        gt_map = json.load(f)["labels"]

    # Write mock predictions for both view modes
    for vmode in ["crop_only", "crop_plus_context"]:
        pfile = mock_run / f"visual_classification_predictions_qwen2_5_vl_3b_{vmode}.jsonl"
        with open(pfile, "w", encoding="utf-8") as pf:
            for inst in instances:
                eid = inst["physical_instance_id"]
                gt_cls = gt_map.get(eid, {}).get("contextual_movability_gt", "MOVABLE")
                rec = {
                    "physical_instance_id": eid,
                    "model_key": "qwen2_5_vl_3b",
                    "view_mode": vmode.upper(),
                    "parsed_prediction": gt_cls,
                    "client_latency_ms": 1200.0,
                    "response_received": True,
                    "schema_valid": True,
                    "prediction_accepted": True,
                }
                pf.write(json.dumps(rec) + "\n")

    out_run = tmp_path / "router_run"
    results = evaluate_router_policies(
        policy_keys=["B0_MAJORITY", "B1_SEMANTIC_PRIOR", "B6_ESTIMATOR_AWARE_ROUTER"],
        model_key="qwen2_5_vl_3b",
        execution_mode="replay",
        replay_run_dir=mock_run,
        run_dir=out_run,
    )

    assert "policy_summaries" in results
    assert "B0_MAJORITY" in results["policy_summaries"]
    assert "B1_SEMANTIC_PRIOR" in results["policy_summaries"]
    assert "B6_ESTIMATOR_AWARE_ROUTER" in results["policy_summaries"]

    b6_m = results["policy_summaries"]["B6_ESTIMATOR_AWARE_ROUTER"]["metrics"]
    assert b6_m["total_instances"] == 15
    assert b6_m["vlm_queries_count"] == 8
    assert b6_m["vlm_query_rate"] == pytest.approx(8 / 15, 0.01)

    summary_file = out_run / "router_policies_summary.json"
    assert summary_file.exists()
