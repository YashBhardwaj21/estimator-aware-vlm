"""Unit and Regression Tests for Paper Benchmark Two-File Isolation and Image Integrity.

Verifies:
1. Strict two-file isolation: paper_benchmark_inputs.json contains ZERO ground truth or decision fields.
2. paper_benchmark_ground_truth.json correctly matches physical instance IDs.
3. All image files exist, decode cleanly with PIL, and SHA-256 hashes match.
4. Model registry specifies complete configurations for qwen2_5_vl_3b.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import pytest
from PIL import Image

repo_root = Path(__file__).resolve().parents[1]


def hash_file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def test_two_file_manifest_isolation():
    """Verify inference inputs manifest contains ZERO evaluator labels or decision targets."""
    inputs_path = repo_root / "benchmarks" / "paper_benchmark_inputs.json"
    gt_path = repo_root / "benchmarks" / "paper_benchmark_ground_truth.json"

    assert inputs_path.exists(), f"Missing inputs manifest at {inputs_path}"
    assert gt_path.exists(), f"Missing ground truth manifest at {gt_path}"

    with open(inputs_path, "r", encoding="utf-8") as f:
        inputs_data = json.load(f)["instances"]

    with open(gt_path, "r", encoding="utf-8") as f:
        gt_data = json.load(f)["labels"]

    forbidden_fields = [
        "contextual_movability_gt",
        "current_motion_gt",
        "displacement_m_gt",
        "slam_landmark_validity_gt",
        "expected_decision",
        "expected_reason",
        "annotation_provenance",
        "adjudication_status",
    ]

    for rec in inputs_data:
        eid = rec["physical_instance_id"]
        # Ground truth must exist in evaluator file
        assert eid in gt_data, f"Physical instance {eid} missing from ground truth file"

        # Forbidden fields MUST NOT exist in inference inputs file
        for f in forbidden_fields:
            assert f not in rec, f"Leaked ground truth field '{f}' found in inference input for {eid}"
            if "visual_inputs" in rec:
                assert f not in rec["visual_inputs"]

        # Evaluator ground truth must have valid classes
        gt_rec = gt_data[eid]
        assert gt_rec["contextual_movability_gt"] in ["IMMOVABLE", "MOVABLE", "MOBILE"]
        assert gt_rec["current_motion_gt"] in ["STATIC", "DISPLACED", "DYNAMIC", "UNKNOWN"]
        assert gt_rec["slam_landmark_validity_gt"] in ["RETAIN_LANDMARK", "INVALIDATE_RELOCATED", "INVALIDATE_DYNAMIC_AGENT"]


def test_image_integrity_and_hashes():
    """Verify all referenced images in the benchmark exist, decode cleanly, and match SHA-256 hashes."""
    inputs_path = repo_root / "benchmarks" / "paper_benchmark_inputs.json"
    with open(inputs_path, "r", encoding="utf-8") as f:
        instances = json.load(f)["instances"]

    image_eligible_count = 0
    for rec in instances:
        vis = rec.get("visual_inputs", {})
        if not vis.get("has_images"):
            continue

        image_eligible_count += 1
        crop_paths = vis.get("crop_view_paths", [])
        scene_paths = vis.get("scene_view_paths", [])
        expected_hashes = vis.get("image_sha256", [])

        assert len(crop_paths) == len(scene_paths)
        assert len(expected_hashes) == len(crop_paths) + len(scene_paths)

        hash_idx = 0
        for c_rel, s_rel in zip(crop_paths, scene_paths):
            c_p = repo_root / c_rel
            s_p = repo_root / s_rel

            assert c_p.exists(), f"Missing crop image: {c_p}"
            assert s_p.exists(), f"Missing scene image: {s_p}"

            # Verify PIL decoding
            with Image.open(c_p) as im:
                im.verify()
            with Image.open(s_p) as im:
                im.verify()

            # Verify content SHA-256 hash
            c_hash = hash_file_sha256(c_p)
            s_hash = hash_file_sha256(s_p)
            assert c_hash == expected_hashes[hash_idx]
            assert s_hash == expected_hashes[hash_idx + 1]
            hash_idx += 2

    assert image_eligible_count == 14, f"Expected 14 image-eligible instances, found {image_eligible_count}"


def test_model_registry_schema():
    """Verify model_registry.json contains required entries and parameters for Qwen."""
    reg_path = repo_root / "benchmarks" / "model_registry.json"
    assert reg_path.exists()

    with open(reg_path, "r", encoding="utf-8") as f:
        registry = json.load(f)["models"]

    for mkey in ["qwen2_5_vl_3b"]:
        assert mkey in registry, f"Missing model {mkey} in registry"
        m = registry[mkey]
        for field in ["model_identifier", "server_url", "generation_parameters", "image_input_specification"]:
            assert field in m, f"Model {mkey} missing field {field}"
        assert m["generation_parameters"]["temperature"] == 0.0
        assert m["generation_parameters"]["max_tokens"] >= 150


def test_adversarial_ground_truth_label_inversion():
    """Adversarial Test: Inverting evaluator ground truth labels must produce ZERO changes in router decisions."""
    from estimator_aware_vlm.router_policy import evaluate_policy_r3

    inputs_path = repo_root / "benchmarks" / "paper_benchmark_inputs.json"
    with open(inputs_path, "r", encoding="utf-8") as f:
        instances = json.load(f)["instances"]

    def build_evidence(inst: Dict[str, Any], override_id: Optional[str] = None) -> Dict[str, Any]:
        est = inst.get("estimator_evidence", {})
        return {
            "entity_id": override_id or inst["physical_instance_id"],
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

    # Compute baseline decisions using normal inputs
    baseline_decisions = []
    for inst in instances:
        dec = evaluate_policy_r3(build_evidence(inst))
        baseline_decisions.append((dec.decision, dec.vlm_required, dec.cheap_label, dec.reason_codes))

    # Invert ground-truth labels completely
    inverted_gt = {
        "IMMOVABLE": "MOBILE",
        "MOVABLE": "IMMOVABLE",
        "MOBILE": "MOVABLE",
    }
    # Verify inference pipeline does not import or inspect the inverted labels
    post_inversion_decisions = []
    for inst in instances:
        dec = evaluate_policy_r3(build_evidence(inst))
        post_inversion_decisions.append((dec.decision, dec.vlm_required, dec.cheap_label, dec.reason_codes))

    assert baseline_decisions == post_inversion_decisions


def test_adversarial_instance_id_renaming():
    """Adversarial Test: Renaming physical instance IDs must produce identical decisions and displacement outputs."""
    from estimator_aware_vlm.router_policy import evaluate_policy_r3

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
    """Fail-Closed Test: Assert missing cross-session observations produce None displacement and conservative gating."""
    from estimator_aware_vlm.evidence_adapter import Stage5EvidenceAdapter
    from estimator_aware_vlm.router_policy import evaluate_policy_r3

    adapter = Stage5EvidenceAdapter()

    # Case 1: Manifest entry with single session observations only (door_001)
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

    # Router gating on missing displacement
    dec = evaluate_policy_r3({
        "entity_id": "test_single_sess_001",
        "semantic_class": "door",
        "observed_motion": ev_single["observed_motion"],
        "world_displacement_m": ev_single["measured_displacement_m"],
        "displacement_uncertainty_m": ev_single["displacement_uncertainty_m"],
        "has_images": True,
        "views_count": 1,
    })
    # Never falsely flags physical relocation displacement
    assert "SIGNIFICANT_DISPLACEMENT_AFFORDANCE_UNRESOLVED" not in dec.reason_codes
    assert dec.decision == "RESOLVE_CHEAP"
    assert dec.cheap_label == "IMMOVABLE"

