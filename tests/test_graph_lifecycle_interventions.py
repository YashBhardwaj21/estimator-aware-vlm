"""Unit tests for Scene-Graph Lifecycle Intervention Evaluator."""

import json
from pathlib import Path

import pytest
import numpy as np

from estimator_aware_vlm.interventions import evaluate_lifecycle_interventions


def test_evaluate_lifecycle_interventions(tmp_path: Path):
    # Create mock ground truth
    mock_gt = {
        "physical_objects": [
            {
                "object_instance_id": "door_1",
                "semantic_class": "door",
                "movability_gt": "IMMOVABLE",
                "current_motion_gt": "STATIC",
                "displacement_gt": 0.0,
                "slam_invalidated": False,
                "sessions_observed": ["s1", "s2"],
            },
            {
                "object_instance_id": "chair_moving",
                "semantic_class": "chair",
                "movability_gt": "MOVABLE",
                "current_motion_gt": "MOVING",
                "displacement_gt": 0.50,
                "slam_invalidated": True,
                "sessions_observed": ["s1", "s2"],
            },
        ]
    }
    gt_file = tmp_path / "movability_gt.json"
    with open(gt_file, "w", encoding="utf-8") as f:
        json.dump(mock_gt, f)

    mock_audit = {
        "rotation_deg": 0.0,
        "translation_m": [0.0, 0.0],
        "anchors": [
            {"id1": "door_1", "id2": "door_1", "residual_m": 0.05, "is_robust_inlier": True}
        ],
    }
    audit_file = tmp_path / "audit.json"
    with open(audit_file, "w", encoding="utf-8") as f:
        json.dump(mock_audit, f)

    out_dir = tmp_path / "out"
    report = evaluate_lifecycle_interventions(gt_file, audit_file, out_dir)

    assert report["total_verified_physical_instances"] == 2
    assert report["static_instances_count"] == 1
    assert report["relocated_instances_count"] == 1

    c0 = report["counterfactual_arms"]["arm_c0_keep"]
    assert c0["total_stale_merges"] == 1
    assert c0["total_correct_merges"] == 1
    assert c0["relocated_subset_displacement_rmse_m"] == 0.50

    c4 = report["counterfactual_arms"]["arm_c4_graph_temporary_exclude"]
    assert c4["excluded_entities_count"] == 1
    assert c4["held_out_static_anchor_rms_m"] == 0.05

    # Check generated files
    assert (out_dir / "lifecycle_interventions.json").exists()
    assert (out_dir / "lifecycle_interventions_per_instance.csv").exists()
    assert (out_dir / "LIFECYCLE_INTERVENTION_REPORT.md").exists()
