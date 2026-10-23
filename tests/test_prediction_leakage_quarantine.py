"""Unit tests for Prediction Pipeline Leakage Quarantine and Clean Evaluation Interface.

Covers:
- Task 2.1: Quarantine ground truth from prediction pipeline.
- Task 2.2: Establish clean prediction vs evaluation interface.
"""

import json
from pathlib import Path
import pytest

from scripts.run_full_experiment import run_full_experiment
from estimator_aware_vlm.scene_graph_evaluation import evaluate_scene_graph


@pytest.fixture
def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


@pytest.fixture
def fixture_dir(repo_root) -> Path:
    return repo_root / "tests" / "fixtures" / "real_pipeline_fixture"


def test_pure_prediction_scene_graph_contains_no_gt_leakage(repo_root, fixture_dir, tmp_path):
    """Assert generated scene graph contains zero ground truth fields or forced overrides."""
    obs_file = fixture_dir / "fixture_observations.jsonl"
    reg_file = fixture_dir / "fixture_registration_audit.json"
    ann_file = fixture_dir / "fixture_identity_annotations.json"
    out_dir = tmp_path / "leakage_quarantine_run"

    manifest = run_full_experiment(
        s1_session="recording_01",
        s2_session="recording_02",
        obs_path=obs_file,
        reg_path=reg_file,
        annotations_path=ann_file,
        out_dir=out_dir,
    )

    assert manifest is not None
    assert "downstream_evaluation" in manifest
    assert "downstream_evaluation_summary_json" in manifest["output_artifacts"]

    sg_file = out_dir / "scene_graph.json"
    assert sg_file.exists()

    with open(sg_file, "r", encoding="utf-8") as f:
        sg = json.load(f)

    # Check all nodes in canonical scene graph
    for node_id, node in sg["nodes"].items():
        assert "gt_instance_id" not in node, f"Node {node_id} contains leaked 'gt_instance_id'"
        assert "movability_gt" not in node, f"Node {node_id} contains leaked 'movability_gt'"
        assert "current_motion_gt" not in node, f"Node {node_id} contains leaked 'current_motion_gt'"

        # Verify explicit predicted action fields exist
        assert "predicted_landmark_action" in node
        assert node["predicted_landmark_action"] in ["RETAIN", "INVALIDATE"]
        assert "landmark_action_reason" in node

    # Check policy scene graphs (R0 through R4)
    for p in ["r0", "r1", "r2", "r3", "r4"]:
        p_file = out_dir / f"scene_graph_{p}.json"
        assert p_file.exists()
        with open(p_file, "r", encoding="utf-8") as f:
            p_sg = json.load(f)
        for node_id, node in p_sg["nodes"].items():
            assert "gt_instance_id" not in node
            assert "movability_gt" not in node
            assert "current_motion_gt" not in node


def test_clean_evaluation_interface_metrics(repo_root, fixture_dir):
    """Assert evaluate_scene_graph independently computes retention and relocation rates."""
    # Construct a synthetic purely predicted scene graph with 2 nodes:
    # 1. desk (stationary) -> correctly retained
    # 2. chair (relocated) -> invalidated
    synthetic_sg = {
        "nodes": {
            "phys_door_001": {
                "entity_id": "phys_door_001",
                "semantic_class": "door",
                "predicted_movability": "IMMOVABLE",
                "observed_motion": "STATIC",
                "predicted_landmark_action": "RETAIN",
                "slam_invalidated": False,
                "source_track_ids": ["door_001"],
                "source_sessions": ["recording_01"],
            },
            "phys_chair_002": {
                "entity_id": "phys_chair_002",
                "semantic_class": "chair",
                "predicted_movability": "MOVABLE",
                "observed_motion": "DISPLACED",
                "predicted_landmark_action": "INVALIDATE",
                "slam_invalidated": True,
                "source_track_ids": ["chair_002"],
                "source_sessions": ["recording_01", "recording_02"],
            },
        }
    }

    c_events = repo_root / "benchmarks" / "controlled_scene_events.json"
    ann_file = fixture_dir / "fixture_identity_annotations.json"
    mov_file = repo_root / "benchmarks" / "movability_ground_truth.json"

    eval_res = evaluate_scene_graph(
        scene_graph=synthetic_sg,
        controlled_events_path=c_events,
        identity_annotations_path=ann_file,
        movability_gt_path=mov_file,
        policy_name="SYNTHETIC_TEST",
    )

    assert eval_res["evaluation_contract"] == "PURE_PREDICTION_EVALUATION_NO_LEAKAGE"
    assert eval_res["total_nodes_predicted"] == 2
    assert eval_res["nodes_matched_to_ground_truth"] >= 1

    rel = eval_res["landmark_reliability"]
    assert "stationary_retention_rate_pct" in rel
    assert "false_invalidation_rate_pct" in rel
    assert "relocation_invalidation_rate_pct" in rel


def test_policy_specific_scene_graphs_decoupling(repo_root, fixture_dir, tmp_path):
    """Verify distinct policy behavior across R0, R1, R2, R3, R4."""
    obs_file = fixture_dir / "fixture_observations.jsonl"
    reg_file = fixture_dir / "fixture_registration_audit.json"
    ann_file = fixture_dir / "fixture_identity_annotations.json"
    out_dir = tmp_path / "policy_decoupling_run"

    manifest = run_full_experiment(
        s1_session="recording_01",
        s2_session="recording_02",
        obs_path=obs_file,
        reg_path=reg_file,
        annotations_path=ann_file,
        out_dir=out_dir,
    )

    eval_summary_path = out_dir / "downstream_evaluation_summary.json"
    assert eval_summary_path.exists()
    with open(eval_summary_path, "r", encoding="utf-8") as f:
        eval_summary = json.load(f)

    for p in ["R0", "R1", "R2", "R3", "R4"]:
        assert p in eval_summary
        p_rel = eval_summary[p]["landmark_reliability"]
        assert "stationary_retention_rate_pct" in p_rel
        assert "false_invalidation_rate_pct" in p_rel
        assert "relocation_invalidation_rate_pct" in p_rel

    # Verify R0 produces 0 landmark invalidations
    with open(out_dir / "scene_graph_r0.json", "r", encoding="utf-8") as f:
        sg_r0 = json.load(f)
    for n in sg_r0["nodes"].values():
        assert n.get("predicted_landmark_action") == "RETAIN"
        assert n.get("slam_invalidated") is False

    # Verify R3 attaches router decisions
    with open(out_dir / "scene_graph_r3.json", "r", encoding="utf-8") as f:
        sg_r3 = json.load(f)
    for n in sg_r3["nodes"].values():
        assert "predicted_landmark_action" in n
        assert "predicted_movability" in n

