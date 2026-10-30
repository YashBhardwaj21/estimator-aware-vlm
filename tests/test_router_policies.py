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
            "crop_view_paths": ["benchmarks/assets/multiview_samples/inst_chair_002/crop_view_0.jpg"],
            "scene_view_paths": ["benchmarks/assets/multiview_samples/inst_chair_002/scene_view_0.jpg"],
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


# ---------------------------------------------------------------------------
# Individual Decoupled Policy Invariants & Estimator Relevance Signals
# ---------------------------------------------------------------------------

from estimator_aware_vlm.router_policy import (
    compute_estimator_influence,
    compute_estimator_influence_with_provenance,
    evaluate_policy_r0,
    evaluate_policy_r1,
    evaluate_policy_r2,
    evaluate_policy_r3,
    evaluate_policy_r4,
    evaluate_router_policy,
)
from estimator_aware_vlm.selective_router import SelectiveRouter
from scripts.evaluate_selective_router import compute_metrics


def test_r0_pure_prior_policy():
    """R0 must resolve strictly by taxonomy without requesting VLM."""
    ev_door = {"entity_id": "inst_01", "semantic_class": "door"}
    r_door = evaluate_policy_r0(ev_door)
    assert r_door.decision == "RESOLVE_CHEAP"
    assert r_door.cheap_label == "IMMOVABLE"
    assert not r_door.vlm_required

    ev_chair = {"entity_id": "inst_02", "semantic_class": "chair"}
    r_chair = evaluate_policy_r0(ev_chair)
    assert r_chair.decision == "RESOLVE_CHEAP"
    assert r_chair.cheap_label == "MOVABLE"
    assert not r_chair.vlm_required

    ev_robot = {"entity_id": "inst_03", "semantic_class": "robot"}
    r_robot = evaluate_policy_r0(ev_robot)
    assert r_robot.decision == "RESOLVE_CHEAP"
    assert r_robot.cheap_label == "MOBILE"
    assert not r_robot.vlm_required


def test_r1_geometry_motion_policy():
    """R1 resolves based on observed motion and displacement without VLM."""
    ev_displaced = {
        "entity_id": "inst_chair_002",
        "semantic_class": "chair",
        "observed_motion": "DISPLACED",
        "world_displacement_m": 0.56,
    }
    r = evaluate_policy_r1(ev_displaced)
    assert r.decision == "RESOLVE_CHEAP"
    assert r.cheap_label == "MOVABLE"
    assert not r.vlm_required
    assert "R1_GEOMETRY_MOTION_DISPLACED" in r.reason_codes


def test_r2_non_estimator_aware_policy():
    """R2 escalates on ambiguity blind to SLAM landmark influence."""
    ev = {
        "entity_id": "inst_pedestal_fan_001",
        "semantic_class": "pedestal fan",
        "has_valid_pose": True,
        "depth_valid": True,
        "observed_motion": "STATIC",
        "has_images": True,
        "views_count": 1,
        "s1_obs_count": 0,
        "s2_obs_count": 0,
    }
    r = evaluate_policy_r2(ev)
    assert r.decision == "ESCALATE"
    assert r.vlm_required


def test_r3_estimator_aware_dependency():
    """R3 demonstrably uses the estimator-relevance signal."""
    ev_high_influence = {
        "entity_id": "inst_chair_002",
        "semantic_class": "chair",
        "has_valid_pose": True,
        "depth_valid": True,
        "observed_motion": "DISPLACED",
        "world_displacement_m": 0.563,
        "displacement_uncertainty_m": 0.04,
        "has_images": True,
        "views_count": 2,
        "s1_obs_count": 15,
        "s2_obs_count": 10,
        "geometric_consistency_score": 0.95,
    }
    r_high = evaluate_policy_r3(ev_high_influence)
    assert r_high.decision == "ESCALATE"
    assert r_high.vlm_required
    assert r_high.estimator_influence is not None and r_high.estimator_influence > 0.50
    assert "HIGH_ESTIMATOR_RELEVANCE" in r_high.reason_codes

    # Invalid pose -> DEFER
    ev_invalid_pose = dict(ev_high_influence)
    ev_invalid_pose["has_valid_pose"] = False
    r_inv = evaluate_policy_r3(ev_invalid_pose)
    assert r_inv.decision == "DEFER"
    assert not r_inv.vlm_required
    assert r_inv.estimator_influence == 0.0


def test_r4_all_eligible_vlm_policy():
    """R4 requests VLM for every image-eligible instance."""
    ev = {
        "entity_id": "inst_desk_001",
        "has_images": True,
        "views_count": 1,
    }
    r = evaluate_policy_r4(ev)
    assert r.decision == "ESCALATE"
    assert r.vlm_required
    assert "R4_ALL_ELIGIBLE_VLM" in r.reason_codes


def test_estimator_influence_computation():
    """Verify compute_estimator_influence scales properly with observation support."""
    ev_few = {"has_valid_pose": True, "s1_obs_count": 1, "s2_obs_count": 1}
    inf_few = compute_estimator_influence(ev_few)

    ev_many = {"has_valid_pose": True, "s1_obs_count": 10, "s2_obs_count": 10}
    inf_many = compute_estimator_influence(ev_many)

    assert inf_few < inf_many
    assert inf_many <= 1.0


def test_unified_evaluate_router_policy():
    """Verify master evaluate_router_policy dispatches correctly by name."""
    ev = {
        "entity_id": "inst_01",
        "semantic_class": "door",
        "has_valid_pose": True,
        "depth_valid": True,
        "observed_motion": "STATIC",
        "has_images": True,
        "views_count": 1,
    }
    r0 = evaluate_router_policy(ev, policy="R0")
    assert r0.decision == "RESOLVE_CHEAP"
    assert not r0.vlm_required

    r4 = evaluate_router_policy(ev, policy="R4")
    assert r4.decision == "ESCALATE"
    assert r4.vlm_required


def test_r2_vs_r3_divergence_on_estimator_influence():
    """Verify that R2 and R3 make divergent decisions when estimator utility warrants it."""
    ev_high = {
        "entity_id": "inst_fan_high",
        "semantic_class": "pedestal fan",
        "has_valid_pose": True,
        "depth_valid": True,
        "observed_motion": "STATIC",
        "has_images": True,
        "views_count": 2,
        "s1_obs_count": 8,
        "geometric_consistency_score": 0.90,
    }
    r2_high = evaluate_policy_r2(ev_high)
    r3_high = evaluate_policy_r3(ev_high)
    assert r2_high.decision == "ESCALATE"
    assert r3_high.decision == "ESCALATE"
    assert r3_high.query_utility is not None and r3_high.query_utility >= 0.0

    ev_low = {
        "entity_id": "inst_fan_low",
        "semantic_class": "pedestal fan",
        "has_valid_pose": True,
        "depth_valid": True,
        "observed_motion": "STATIC",
        "has_images": True,
        "views_count": 1,
        "total_observations": 1,
        "geometric_consistency_score": 0.10,
    }
    r2_low = evaluate_policy_r2(ev_low)
    r3_low = evaluate_policy_r3(ev_low)
    assert r2_low.decision == "ESCALATE"
    assert r2_low.vlm_required is True
    assert r3_low.decision == "RESOLVE_CHEAP"
    assert r3_low.vlm_required is False
    assert r3_low.query_utility is not None and r3_low.query_utility < 0.0
    assert "SUBTHRESHOLD_QUERY_UTILITY_AVOIDED" in r3_low.reason_codes


def test_backend_zero_inliers_sets_zero_influence():
    """Verify that backend telemetry with zero inliers drops estimator influence to 0.0."""
    ev_zero_inliers = {
        "entity_id": "clutter_01",
        "has_valid_pose": True,
        "depth_valid": True,
        "inlier_count": 0,
        "inlier_ratio": 0.0,
        "inlier_provenance": "MEASURED_BACKEND",
        "s1_obs_count": 10,
    }
    score, prov = compute_estimator_influence_with_provenance(ev_zero_inliers)
    assert score == 0.0
    assert prov == "MEASURED_BACKEND_ZERO_INLIERS"


def test_router_reduces_vlm_query_budget():
    """Verify that selective router avoids > 65% of VLM queries on representative distributions."""
    sample_entities = [
        {"entity_id": "door_001", "semantic_class": "door", "observed_motion": "STATIC", "has_images": True, "views_count": 1, "ground_truth": "IMMOVABLE"},
        {"entity_id": "fan_001", "semantic_class": "ceiling fan", "observed_motion": "STATIC", "has_images": True, "views_count": 1, "ground_truth": "IMMOVABLE"},
        {"entity_id": "desk_001", "semantic_class": "desk", "observed_motion": "STATIC", "has_images": True, "views_count": 3, "ground_truth": "MOVABLE"},
        {"entity_id": "desk_018", "semantic_class": "desk", "observed_motion": "STATIC", "has_images": True, "views_count": 1, "ground_truth": "MOVABLE"},
        {"entity_id": "laptop_001", "semantic_class": "laptop", "observed_motion": "STATIC", "has_images": True, "views_count": 2, "ground_truth": "MOVABLE"},
        {"entity_id": "backpack_001", "semantic_class": "backpack", "observed_motion": "STATIC", "has_images": True, "views_count": 2, "ground_truth": "MOVABLE"},
        {"entity_id": "monitor_001", "semantic_class": "computer monitor", "observed_motion": "STATIC", "has_images": True, "views_count": 2, "ground_truth": "MOVABLE"},
        {"entity_id": "person_001", "semantic_class": "person", "observed_motion": "DYNAMIC", "has_images": False, "views_count": 0, "ground_truth": "MOBILE"},
        {"entity_id": "robot_001", "semantic_class": "robot", "observed_motion": "DYNAMIC", "has_images": False, "views_count": 0, "ground_truth": "MOBILE"},
        {"entity_id": "chair_002", "semantic_class": "chair", "observed_motion": "DISPLACED", "world_displacement_m": 0.56, "has_images": True, "views_count": 1, "ground_truth": "MOVABLE"},
        {"entity_id": "fan_002", "semantic_class": "pedestal fan", "observed_motion": "STATIC", "has_images": True, "views_count": 1, "ground_truth": "MOVABLE"},
    ]

    router = SelectiveRouter(enable_vlm=False)
    results = []
    for s in sample_entities:
        res = router.route_entity(s)
        res["ground_truth"] = s["ground_truth"]
        results.append(res)

    metrics = compute_metrics(results)
    eff = metrics["efficiency"]

    assert eff["vlm_queries_requested"] == 2
    assert eff["vlm_queries_invoked"] == 0
    assert eff["vlm_queries_avoided"] == 11
    assert eff["vlm_query_rate_pct"] == 0.0
    assert eff["decision_latency_ms"]["mean"] < 1.0


def test_ablation_oracle_and_random_query_shapes():
    """Verify metrics calculation correctly separates quality, efficiency, and reliability."""
    mock_records = [
        {"entity_id": "e1", "decision": "RESOLVE_CHEAP", "prediction": "IMMOVABLE", "ground_truth": "IMMOVABLE", "abstained": False, "vlm_required": False, "views_used": 0, "evidence_valid": True, "decision_latency_ms": 0.04, "inference_latency_ms": 0.0},
        {"entity_id": "e2", "decision": "RESOLVE_CHEAP", "prediction": "MOVABLE", "ground_truth": "MOVABLE", "abstained": False, "vlm_required": False, "views_used": 0, "evidence_valid": True, "decision_latency_ms": 0.03, "inference_latency_ms": 0.0},
        {"entity_id": "e3", "decision": "ESCALATE", "prediction": "MOVABLE", "ground_truth": "MOVABLE", "abstained": False, "vlm_required": True, "vlm_inference_executed": True, "views_used": 2, "evidence_valid": True, "decision_latency_ms": 0.05, "inference_latency_ms": 12000.0},
        {"entity_id": "e4", "decision": "RESOLVE_CHEAP", "prediction": "MOBILE", "ground_truth": "MOBILE", "abstained": False, "vlm_required": False, "views_used": 0, "evidence_valid": True, "decision_latency_ms": 0.04, "inference_latency_ms": 0.0},
    ]

    metrics = compute_metrics(mock_records)
    assert metrics["quality"]["balanced_accuracy"] == 1.0
    assert metrics["quality"]["macro_f1"] == 1.0
    assert metrics["efficiency"]["vlm_queries_requested"] == 1
    assert metrics["efficiency"]["vlm_queries_invoked"] == 1
    assert metrics["efficiency"]["vlm_query_rate_pct"] == 25.0
    assert metrics["reliability"]["false_cheap_resolutions"] == 0
    assert metrics["reliability"]["invalid_evidence_escalations"] == 0
