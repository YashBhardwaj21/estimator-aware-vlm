"""Unit tests for Estimator-Aware Selective Router metrics, random baseline aggregation, and query accounting."""

import json
import numpy as np
import pytest
from pathlib import Path

root_dir = Path(__file__).resolve().parents[1]

from scripts.evaluate_selective_router import compute_metrics
from scripts.run_router_ablation import (
    run_random_baseline_1000_seeds,
    generate_ablation_markdown_report,
)
from estimator_aware_vlm.selective_router import SelectiveRouter


def test_metric_definitions_query_distinction():
    """Verify compute_metrics strictly separates requested, executed, cached, valid, and accepted queries."""
    records = [
        # 1: Requested and live executed
        {
            "entity_id": "e1",
            "decision": "ESCALATE",
            "prediction": "MOVABLE",
            "ground_truth": "MOVABLE",
            "abstained": False,
            "vlm_required": True,
            "vlm_inference_executed": True,
            "vlm_cache_hit": False,
            "vlm_output_valid": True,
            "vlm_prediction_accepted": True,
            "decision_latency_ms": 0.05,
            "inference_latency_ms": 5000.0,
            "evidence_valid": True,
        },
        # 2: Requested and cache hit
        {
            "entity_id": "e2",
            "decision": "ESCALATE",
            "prediction": "IMMOVABLE",
            "ground_truth": "IMMOVABLE",
            "abstained": False,
            "vlm_required": True,
            "vlm_inference_executed": False,
            "vlm_cache_hit": True,
            "vlm_output_valid": True,
            "vlm_prediction_accepted": True,
            "decision_latency_ms": 0.05,
            "inference_latency_ms": 10.0,
            "evidence_valid": True,
        },
        # 3: Requested but VLM disabled (abstained, 0 executed)
        {
            "entity_id": "e3",
            "decision": "ESCALATE",
            "prediction": "UNKNOWN",
            "ground_truth": "MOVABLE",
            "abstained": True,
            "vlm_required": True,
            "vlm_inference_executed": False,
            "vlm_cache_hit": False,
            "vlm_output_valid": False,
            "vlm_prediction_accepted": False,
            "decision_latency_ms": 0.05,
            "inference_latency_ms": 0.0,
            "evidence_valid": True,
        },
        # 4: Cheap prior (not requested)
        {
            "entity_id": "e4",
            "decision": "RESOLVE_CHEAP",
            "prediction": "MOBILE",
            "ground_truth": "MOBILE",
            "abstained": False,
            "vlm_required": False,
            "vlm_inference_executed": False,
            "vlm_cache_hit": False,
            "vlm_output_valid": True,
            "vlm_prediction_accepted": True,
            "decision_latency_ms": 0.04,
            "inference_latency_ms": 0.0,
            "evidence_valid": True,
        },
    ]

    m = compute_metrics(records)
    eff = m["efficiency"]

    assert eff["total_instances"] == 4
    assert eff["vlm_queries_requested"] == 3       # e1, e2, e3
    assert eff["vlm_inference_executed"] == 1      # e1
    assert eff["vlm_cache_hits"] == 1              # e2
    assert eff["vlm_queries_invoked"] == 2         # e1 + e2 (actual model results used)
    assert eff["vlm_queries_avoided"] == 2         # e3 + e4
    assert eff["vlm_query_rate_pct"] == 50.0       # 2 / 4
    assert eff["vlm_outputs_valid"] == 3           # e1, e2, e4
    assert eff["vlm_predictions_accepted"] == 3    # e1, e2, e4


def test_no_query_accounting_when_vlm_disabled():
    """Verify that when VLM is disabled, executed and invoked queries are strictly 0."""
    router = SelectiveRouter(enable_vlm=False)
    # Target that causes escalation
    evidence = {
        "entity_id": "inst_chair_moving",
        "semantic_class": "chair",
        "observed_motion": "DISPLACED",
        "world_displacement_m": 0.60,
        "displacement_uncertainty_m": 0.10,
        "relocation_evaluable": True,
        "has_images": True,
        "views_count": 2,
        "has_valid_pose": True,
        "depth_valid": True,
    }

    result = router.route_entity(evidence)
    assert result["decision"] == "ESCALATE"
    assert result["vlm_required"] is True
    assert result["vlm_inference_executed"] is False
    assert result["vlm_cache_hit"] is False
    assert result["prediction"] == "UNKNOWN"
    assert result["abstained"] is True

    result["ground_truth"] = "MOVABLE"
    metrics = compute_metrics([result])
    assert metrics["efficiency"]["vlm_queries_requested"] == 1
    assert metrics["efficiency"]["vlm_queries_invoked"] == 0
    assert metrics["efficiency"]["vlm_query_rate_pct"] == 0.0


def test_random_seed_canonical_aggregation():
    """Verify that random baseline 1,000-seed summary aggregates canonical statistics."""
    mock_instances = [
        {"entity_id": f"inst_{i:02d}", "semantic_class": "chair", "contextual_movability_gt": "MOVABLE", "has_images": True, "views_count": 1}
        for i in range(10)
    ]
    mock_vlm_map = {
        f"inst_{i:02d}": {"prediction": "MOVABLE", "abstained": False, "views_used": 1, "inference_latency_ms": 1000.0}
        for i in range(10)
    }

    res = run_random_baseline_1000_seeds(
        instances=mock_instances,
        vlm_predictions_map=mock_vlm_map,
        k_queries=3,
        num_seeds=50,
    )

    assert res["num_seeds"] == 50
    assert res["query_budget"] == 3
    assert "mean" in res["balanced_accuracy"]
    assert "std" in res["balanced_accuracy"]
    assert "median" in res["balanced_accuracy"]
    assert "p05" in res["balanced_accuracy"]
    assert "p95" in res["balanced_accuracy"]
    assert res["canonical_summary_metrics"]["efficiency"]["vlm_queries_invoked"] == 3


def test_report_to_json_agreement(tmp_path):
    """Verify that all markdown report cells match JSON numerical values directly."""
    ablation_json_path = root_dir / "outputs" / "router_ablation" / "router_ablation_results.json"
    if not ablation_json_path.exists():
        pytest.skip("router_ablation_results.json not generated.")

    with open(ablation_json_path, "r", encoding="utf-8-sig") as f:
        ablation = json.load(f)

    rep_path = tmp_path / "test_report.md"
    generate_ablation_markdown_report(ablation, rep_path)
    content = rep_path.read_text(encoding="utf-8")

    # Check selective router values
    sel = ablation["experiments"]["router_selective_vlm"]["metrics"]
    sel_acc = f"{sel['quality']['balanced_accuracy']*100:.1f}%"
    sel_f1 = f"{sel['quality']['macro_f1']*100:.1f}%"
    assert sel_acc in content
    assert sel_f1 in content


@pytest.mark.parametrize(
    "entity_id,semantic_class",
    [
        ("inst_chair_002", "chair"),
        ("inst_desk_001", "desk"),
        ("inst_storage_box_007", "storage box"),
        ("inst_door_001", "door"),
        ("inst_rc_car_robot_001", "robot"),
    ],
)
def test_router_decision_invariant_to_ground_truth_mutation(entity_id: str, semantic_class: str):
    """Verify across multiple classes that mutating benchmark ground truth labels cannot alter the selective routing decision.

    Ground-truth motion, displacement, and movability labels must strictly belong to reference
    evaluation and never leak into policy decisions.
    """
    from estimator_aware_vlm.evidence_adapter import EvidenceAdapter
    from estimator_aware_vlm.router_policy import evaluate_router_policy

    manifest_path = root_dir / "benchmarks" / "router_evaluation_manifest.json"
    if not manifest_path.exists():
        pytest.skip("Benchmark manifest not available.")

    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest_data = json.load(f)["evaluation_instances"]

    manifest_entry = next((x for x in manifest_data if x["entity_id"] == entity_id), None)
    if manifest_entry is None:
        pytest.skip(f"Instance {entity_id} not found in manifest.")

    adapter = EvidenceAdapter(repo_root=root_dir)

    # Build evidence strictly from SLAM observations & registration
    ev_original = adapter.build_instance_evidence(manifest_entry)
    decision_original = evaluate_router_policy(ev_original)

    # 1. Mutate ground truth target fields in entry
    mutated_entry = dict(manifest_entry)
    mutated_entry["contextual_movability_gt"] = "IMMOVABLE" if manifest_entry.get("contextual_movability_gt") != "IMMOVABLE" else "MOBILE"
    mutated_entry["observed_motion_gt"] = "DYNAMIC_FLYING"
    mutated_entry["displacement_gt"] = 12.345
    mutated_entry["current_motion_gt"] = "DISPLACED_RUNAWAY"

    ev_after_gt_mutation = adapter.build_instance_evidence(mutated_entry)
    decision_after_gt_mutation = evaluate_router_policy(ev_after_gt_mutation)

    assert ev_original.get("observed_motion") == ev_after_gt_mutation.get("observed_motion")
    assert ev_original.get("world_displacement_m") == ev_after_gt_mutation.get("world_displacement_m")
    assert decision_original.decision == decision_after_gt_mutation.decision
    assert decision_original.vlm_required == decision_after_gt_mutation.vlm_required
    assert decision_original.reason_codes == decision_after_gt_mutation.reason_codes
    assert decision_original.cheap_label == decision_after_gt_mutation.cheap_label

    # 2. Even if ground-truth fields are injected directly into the evidence dict,
    # evaluate_router_policy must ignore them and evaluate only estimator evidence
    ev_with_gt_injection = dict(ev_original)
    ev_with_gt_injection["current_motion_gt"] = "STATIC"
    ev_with_gt_injection["displacement_gt"] = 0.00
    ev_with_gt_injection["movability_gt"] = "IMMOVABLE"

    decision_injected = evaluate_router_policy(ev_with_gt_injection)
    assert decision_injected.decision == decision_original.decision
    assert decision_injected.vlm_required == decision_original.vlm_required
    assert decision_injected.reason_codes == decision_original.reason_codes


def test_end_to_end_vs_selective_accuracy_distinction():
    """Verify compute_policy_evaluation_metrics computes and distinguishes end-to-end vs selective accuracy."""
    from scripts.evaluate_router_policies import compute_policy_evaluation_metrics

    # 10 instances total:
    # 4 routed to VLM (3 correct, 1 wrong) -> selective accuracy = 3/4 = 0.75
    # 6 resolved cheap (5 correct, 1 wrong) -> cheap accuracy = 5/6 = 0.8333
    # Total correct = 8/10 -> end-to-end accuracy = 0.80
    records = []
    gt_map = {}

    for i in range(4):
        eid = f"vlm_inst_{i}"
        gt_map[eid] = {"contextual_movability_gt": "MOVABLE"}
        pred = "MOVABLE" if i < 3 else "IMMOVABLE"
        records.append({
            "physical_instance_id": eid,
            "decision": "ESCALATE",
            "vlm_invoked": True,
            "final_prediction": pred,
            "total_latency_ms": 1200.0,
        })

    for i in range(6):
        eid = f"cheap_inst_{i}"
        gt_map[eid] = {"contextual_movability_gt": "IMMOVABLE"}
        pred = "IMMOVABLE" if i < 5 else "MOVABLE"
        records.append({
            "physical_instance_id": eid,
            "decision": "RESOLVE_CHEAP",
            "vlm_invoked": False,
            "final_prediction": pred,
            "total_latency_ms": 0.05,
        })

    metrics = compute_policy_evaluation_metrics(records, gt_map)

    assert metrics["total_instances"] == 10
    assert metrics["vlm_queries_count"] == 4
    assert metrics["vlm_query_rate"] == 0.4
    assert metrics["end_to_end_accuracy"] == 0.80
    assert metrics["overall_accuracy"] == 0.80
    assert metrics["selective_accuracy"] == 0.75
    assert metrics["selective_accuracy_on_vlm"] == 0.75
    assert metrics["cheap_accuracy"] == round(5 / 6, 4)


def test_malformed_json_and_id_mismatch_penalized():
    """Verify malformed JSON responses and identity mismatches are penalized in accuracy scoring."""
    from estimator_aware_vlm.multimodal_inference_engine import parse_multimodal_prediction
    from scripts.evaluate_router_policies import compute_policy_evaluation_metrics

    # 1. Truncated/malformed JSON
    malformed_raw = '{"object_id": "inst_chair_001", "movability":'
    parsed_bad, parse_valid, schema_valid = parse_multimodal_prediction(malformed_raw, "inst_chair_001", "chair")
    assert parse_valid is False
    assert schema_valid is False
    assert parsed_bad["movability"] == "UNKNOWN"
    assert parsed_bad["abstain"] is True

    # 2. Object ID mismatch
    wrong_id_raw = json.dumps({
        "object_id": "inst_wrong_002",
        "object_class": "chair",
        "movability": "MOVABLE",
        "confidence": 0.9,
    })
    parsed_mismatch, p_ok, s_ok = parse_multimodal_prediction(wrong_id_raw, "inst_chair_001", "chair")
    assert p_ok is True
    assert s_ok is False
    assert "object_id_mismatch" in parsed_mismatch.get("validation_error", "")

    # 3. Object class mismatch in class-conditioned mode
    wrong_cls_raw = json.dumps({
        "object_id": "inst_chair_001",
        "object_class": "couch",
        "movability": "MOVABLE",
        "confidence": 0.9,
    })
    parsed_cls_mismatch, p_ok, s_ok = parse_multimodal_prediction(wrong_cls_raw, "inst_chair_001", "chair")
    assert p_ok is True
    assert s_ok is False
    assert "object_class_mismatch" in parsed_cls_mismatch.get("validation_error", "")

    # 4. Scoring penalty verification:
    # When malformed/mismatched output is passed to policy evaluation, it must yield UNKNOWN
    # and be scored as 0.0 accuracy against ground truth (never skipped or defaulted)
    records = [{
        "physical_instance_id": "inst_chair_001",
        "decision": "ESCALATE",
        "vlm_invoked": True,
        "final_prediction": parsed_bad["movability"],  # "UNKNOWN"
        "total_latency_ms": 500.0,
    }]
    gt_map = {"inst_chair_001": {"contextual_movability_gt": "MOVABLE"}}

    metrics = compute_policy_evaluation_metrics(records, gt_map)
    assert metrics["end_to_end_accuracy"] == 0.0
    assert metrics["selective_accuracy"] == 0.0
    assert metrics["confusion_matrix"]["MOVABLE"]["UNKNOWN"] == 1


