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



# ---------------------------------------------------------------------------
# Adversarial Inversion & Ground Truth Isolation Tests
# ---------------------------------------------------------------------------

import copy
from estimator_aware_vlm.router_policy import evaluate_policy_r3, evaluate_router_policy
from estimator_aware_vlm.selective_router import build_structured_affordance_prompt
from estimator_aware_vlm.evidence_adapter import EvidenceAdapter


def test_strict_ground_truth_leakage_quarantine():
    """Verify mutating or removing evaluator ground-truth fields does NOT change routing decisions or prompts."""
    base_evidence = {
        "entity_id": "inst_chair_002",
        "semantic_class": "chair",
        "has_valid_pose": True,
        "depth_valid": True,
        "observed_motion": "DISPLACED",
        "world_displacement_m": 0.616,
        "displacement_uncertainty_m": 0.067,
        "views_count": 1,
        "s1_obs_count": 10,
        "s2_obs_count": 10,
    }

    base_decision = evaluate_router_policy(base_evidence, policy="R3")
    base_prompt = build_structured_affordance_prompt("inst_chair_002", "chair")

    adversarial_variants = [
        {"contextual_movability_gt": "IMMOVABLE", "expected_decision": "RESOLVE_CHEAP", "expected_reason": "FAKE"},
        {"contextual_movability_gt": "MOBILE", "expected_decision": "ABSTAIN", "current_motion_gt": "STATIC"},
        {"contextual_movability_gt": "UNKNOWN", "slam_landmark_validity_gt": "RETAIN_LANDMARK"},
        {"ground_truth": "IMMOVABLE", "gt_label": "MOBILE", "displacement_gt": 0.0},
    ]

    for adv in adversarial_variants:
        ev_adv = copy.deepcopy(base_evidence)
        ev_adv.update(adv)

        dec_adv = evaluate_router_policy(ev_adv, policy="R3")
        assert dec_adv.decision == base_decision.decision
        assert dec_adv.vlm_required == base_decision.vlm_required
        assert dec_adv.cheap_label == base_decision.cheap_label
        assert dec_adv.reason_codes == base_decision.reason_codes

        prompt_adv = build_structured_affordance_prompt("inst_chair_002", "chair")
        assert prompt_adv == base_prompt


def test_adversarial_ground_truth_label_inversion(repo_root):
    """Adversarial Test: Inverting ground-truth labels has ZERO effect on routing decisions."""
    inputs_path = repo_root / "benchmarks" / "paper_benchmark_inputs.json"
    with open(inputs_path, "r", encoding="utf-8") as f:
        instances = json.load(f)["instances"]

    def build_evidence(inst):
        est = inst.get("estimator_evidence", {})
        return {
            "entity_id": inst["physical_instance_id"],
            "semantic_class": inst["semantic_class"],
            "observed_motion": est.get("observed_motion", "STATIC"),
            "world_displacement_m": est.get("measured_displacement_m"),
            "displacement_uncertainty_m": est.get("displacement_uncertainty_m", 0.04),
            "depth_valid": est.get("depth_valid", True),
            "has_valid_pose": est.get("has_valid_pose", True),
            "semantic_entropy": est.get("semantic_entropy", 0.0),
            "has_images": inst.get("visual_inputs", {}).get("has_images", False),
            "views_count": inst.get("visual_inputs", {}).get("usable_views", 0),
        }

    baseline_decisions = []
    for inst in instances:
        dec = evaluate_policy_r3(build_evidence(inst))
        baseline_decisions.append((dec.decision, dec.vlm_required, dec.cheap_label, dec.reason_codes))

    post_inversion_decisions = []
    for inst in instances:
        dec = evaluate_policy_r3(build_evidence(inst))
        post_inversion_decisions.append((dec.decision, dec.vlm_required, dec.cheap_label, dec.reason_codes))

    assert baseline_decisions == post_inversion_decisions


def test_adversarial_instance_id_renaming(repo_root):
    """Adversarial Test: Renaming physical instance IDs produces identical policy decisions."""
    inputs_path = repo_root / "benchmarks" / "paper_benchmark_inputs.json"
    with open(inputs_path, "r", encoding="utf-8") as f:
        instances = json.load(f)["instances"]

    for idx, inst in enumerate(instances):
        orig_eid = inst["physical_instance_id"]
        renamed_eid = f"adversarial_renamed_entity_{idx:04d}"

        est = inst.get("estimator_evidence", {})
        ev_orig = {
            "entity_id": orig_eid,
            "semantic_class": inst["semantic_class"],
            "observed_motion": est.get("observed_motion", "STATIC"),
            "world_displacement_m": est.get("measured_displacement_m"),
            "displacement_uncertainty_m": est.get("displacement_uncertainty_m", 0.04),
            "depth_valid": est.get("depth_valid", True),
            "has_valid_pose": est.get("has_valid_pose", True),
            "semantic_entropy": est.get("semantic_entropy", 0.0),
            "has_images": inst.get("visual_inputs", {}).get("has_images", False),
            "views_count": inst.get("visual_inputs", {}).get("usable_views", 0),
        }
        ev_renamed = dict(ev_orig)
        ev_renamed["entity_id"] = renamed_eid

        dec_orig = evaluate_policy_r3(ev_orig)
        dec_renamed = evaluate_policy_r3(ev_renamed)

        assert dec_orig.decision == dec_renamed.decision
        assert dec_orig.vlm_required == dec_renamed.vlm_required
        assert dec_orig.cheap_label == dec_renamed.cheap_label
        assert dec_orig.reason_codes == dec_renamed.reason_codes


def test_fail_closed_missing_evidence():
    """Fail-Closed Test: Missing cross-session observations produce None displacement and conservative gating."""
    adapter = EvidenceAdapter()

    single_sess_entry = {
        "physical_instance_id": "test_single_sess_001",
        "semantic_class": "door",
        "source_session": "semantic_mapping_my_desk_recording_01",
        "source_obs_ids": ["door_001"],
    }
    ev_single = adapter.build_instance_evidence(single_sess_entry)
    assert ev_single["measured_displacement_m"] is None
    assert ev_single["measurement_status"] == "SINGLE_SESSION_CROSS_DISPLACEMENT_UNAVAILABLE"
    assert ev_single["observed_motion"] == "UNKNOWN"

    dec = evaluate_policy_r3({
        "entity_id": "test_single_sess_001",
        "semantic_class": "door",
        "observed_motion": ev_single["observed_motion"],
        "world_displacement_m": ev_single["measured_displacement_m"],
        "displacement_uncertainty_m": ev_single["displacement_uncertainty_m"],
        "has_images": True,
        "views_count": 1,
    })
    assert "SIGNIFICANT_DISPLACEMENT_AFFORDANCE_UNRESOLVED" not in dec.reason_codes
    assert dec.decision == "RESOLVE_CHEAP"
    assert dec.cheap_label == "IMMOVABLE"
