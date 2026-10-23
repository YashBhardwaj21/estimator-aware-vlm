"""Unit and Mock Integration Tests for Axis 3: Downstream SLAM Landmark Lifecycle.

Verifies:
1. determine_downstream_landmark_action accurately maps predictions to operational SLAM actions.
2. Pose history decoupling is correctly activated for physically relocated objects.
3. Landmark retention, corruption risk, and false invalidation rates are mathematically accurate.
4. Trajectory ATE/RPE are strictly marked as NOT_EVALUATED with scientific justification.
5. evaluate_downstream_lifecycle executes end-to-end and outputs valid summary JSON.
"""

from __future__ import annotations

import json
from pathlib import Path
import pytest

from scripts.evaluate_downstream_lifecycle import (
    determine_downstream_landmark_action,
    evaluate_policy_lifecycle,
    evaluate_downstream_lifecycle,
)

repo_root = Path(__file__).resolve().parents[1]


def test_determine_downstream_landmark_action():
    # 1. Stationary architectural object
    act1, reason1, decouple1 = determine_downstream_landmark_action("IMMOVABLE", "STATIC", 0.0, 0.02)
    assert act1 == "RETAIN_LANDMARK"
    assert decouple1 is False

    # 2. Stationary movable furniture
    act2, reason2, decouple2 = determine_downstream_landmark_action("MOVABLE", "STATIC", 0.0, 0.02)
    assert act2 == "RETAIN_LANDMARK"
    assert decouple2 is False

    # 3. Displaced chair
    act3, reason3, decouple3 = determine_downstream_landmark_action("MOVABLE", "DISPLACED", 0.616, 0.067)
    assert act3 == "INVALIDATE_RELOCATED"
    assert decouple3 is True
    assert "PURGED_RELOCATED" in reason3

    # 4. Mobile agent
    act4, reason4, decouple4 = determine_downstream_landmark_action("MOBILE", "DYNAMIC", 1.85, 0.15)
    assert act4 == "INVALIDATE_DYNAMIC_AGENT"
    assert decouple4 is False


def test_evaluate_policy_lifecycle_metrics():
    inputs_map = {
        "inst_1": {"physical_instance_id": "inst_1", "semantic_class": "door", "estimator_evidence": {"observed_motion": "STATIC", "measured_displacement_m": 0.0, "displacement_uncertainty_m": 0.02}},
        "inst_2": {"physical_instance_id": "inst_2", "semantic_class": "chair", "estimator_evidence": {"observed_motion": "DISPLACED", "measured_displacement_m": 0.55, "displacement_uncertainty_m": 0.05}},
        "inst_3": {"physical_instance_id": "inst_3", "semantic_class": "person", "estimator_evidence": {"observed_motion": "DYNAMIC", "measured_displacement_m": 1.5, "displacement_uncertainty_m": 0.10}},
    }

    gt_map = {
        "inst_1": {"slam_landmark_validity_gt": "RETAIN_LANDMARK"},
        "inst_2": {"slam_landmark_validity_gt": "INVALIDATE_RELOCATED"},
        "inst_3": {"slam_landmark_validity_gt": "INVALIDATE_DYNAMIC_AGENT"},
    }

    # Case A: Perfect policy
    perfect_records = [
        {"physical_instance_id": "inst_1", "semantic_class": "door", "final_prediction": "IMMOVABLE", "decision": "RESOLVE_CHEAP"},
        {"physical_instance_id": "inst_2", "semantic_class": "chair", "final_prediction": "MOVABLE", "decision": "ESCALATE"},
        {"physical_instance_id": "inst_3", "semantic_class": "person", "final_prediction": "MOBILE", "decision": "RESOLVE_CHEAP"},
    ]

    metrics = evaluate_policy_lifecycle("PERFECT_POLICY", perfect_records, inputs_map, gt_map)
    assert metrics["overall_landmark_action_accuracy"] == 1.0
    assert metrics["landmark_retention"]["precision"] == 1.0
    assert metrics["landmark_retention"]["recall"] == 1.0
    assert metrics["landmark_retention"]["false_positives_corruption_risk"] == 0
    assert metrics["landmark_retention"]["false_invalidation_rate"] == 0.0
    assert metrics["relocation_handling"]["precision"] == 1.0
    assert metrics["relocation_handling"]["decoupled_pose_count"] == 1
    assert metrics["trajectory_metrics"]["trajectory_ate_m"] == "NOT_EVALUATED"


def test_unit_test_synthetic_lifecycle_policy(tmp_path):
    """UNIT_TEST_SYNTHETIC_LIFECYCLE_POLICY: Test full downstream lifecycle evaluation against synthetic mock decisions."""
    mock_run_dir = tmp_path / "mock_router_run"
    mock_run_dir.mkdir(parents=True)

    inputs_path = repo_root / "benchmarks" / "paper_benchmark_inputs.json"
    with open(inputs_path, "r", encoding="utf-8") as f:
        instances = json.load(f)["instances"]

    gt_path = repo_root / "benchmarks" / "paper_benchmark_ground_truth.json"
    with open(gt_path, "r", encoding="utf-8") as f:
        gt_map = json.load(f)["labels"]

    # Write a synthetic mock router decision file
    decision_file = mock_run_dir / "router_decisions_b6_estimator_aware_router.jsonl"
    with open(decision_file, "w", encoding="utf-8") as f:
        for inst in instances:
            eid = inst["physical_instance_id"]
            gt_mov = gt_map.get(eid, {}).get("contextual_movability_gt", "MOVABLE")
            # Synthesize escalate for ambiguous classes or measured displaced instances
            est = inst.get("estimator_evidence", {})
            is_displaced = est.get("observed_motion") == "DISPLACED"
            scls = inst["semantic_class"]
            is_ambiguous = scls in ["chair", "desk", "storage box", "pedestal fan", "whiteboard"]
            should_escalate = is_displaced or is_ambiguous

            rec = {
                "physical_instance_id": eid,
                "semantic_class": scls,
                "policy_key": "B6_ESTIMATOR_AWARE_ROUTER",
                "decision": "ESCALATE" if should_escalate else "RESOLVE_CHEAP",
                "final_prediction": gt_mov,
                "vlm_invoked": should_escalate,
            }
            f.write(json.dumps(rec) + "\n")

    summary = evaluate_downstream_lifecycle(router_run_dir=mock_run_dir)
    assert "policies" in summary
    assert "B6_ESTIMATOR_AWARE_ROUTER" in summary["policies"]

    p_summary = summary["policies"]["B6_ESTIMATOR_AWARE_ROUTER"]
    assert p_summary["total_instances_evaluated"] == len(instances)
    assert p_summary["overall_landmark_action_accuracy"] == 1.0
    assert p_summary["landmark_retention"]["precision"] == 1.0
    assert p_summary["relocation_handling"]["decoupled_pose_count"] == 3
    assert p_summary["trajectory_metrics"]["trajectory_ate_m"] == "NOT_EVALUATED"

    summary_file = mock_run_dir / "downstream_lifecycle_summary.json"
    assert summary_file.exists()
