"""Regression tests for benchmark manifest integrity, image validation, and leakage prevention."""

import hashlib
import json
from pathlib import Path
from PIL import Image
import pytest

repo_root = Path(__file__).resolve().parent.parent.parent

EXPANDED_INPUTS = repo_root / "vlm" / "benchmarks" / "paper_benchmark_inputs_expanded.json"
EXPANDED_GT = repo_root / "vlm" / "benchmarks" / "paper_benchmark_ground_truth_expanded.json"
ROUTER_INPUTS = repo_root / "vlm" / "benchmarks" / "router_benchmark_inputs_expanded.json"
ROUTER_GT = repo_root / "vlm" / "benchmarks" / "router_benchmark_ground_truth_expanded.json"

LEAKAGE_KEYS = {
    "movability",
    "movability_gt",
    "contextual_movability_gt",
    "expected_prediction",
    "expected_movability",
    "ground_truth",
    "label",
    "is_movable",
    "is_static",
}


def test_expanded_visual_manifests_exist():
    assert EXPANDED_INPUTS.exists(), "Expanded visual inputs manifest missing"
    assert EXPANDED_GT.exists(), "Expanded visual ground truth manifest missing"


def test_visual_manifest_zero_ground_truth_leakage():
    with open(EXPANDED_INPUTS, "r", encoding="utf-8") as f:
        data = json.load(f)

    for inst in data.get("instances", []):
        for k in inst.keys():
            assert k.lower() not in LEAKAGE_KEYS, f"Ground truth leakage key '{k}' found in input instance {inst.get('physical_instance_id')}"
        vis = inst.get("visual_inputs", {})
        for k in vis.keys():
            assert k.lower() not in LEAKAGE_KEYS, f"Ground truth leakage key '{k}' found in visual inputs of {inst.get('physical_instance_id')}"


def test_visual_manifest_unique_physical_instances():
    with open(EXPANDED_INPUTS, "r", encoding="utf-8") as f:
        data = json.load(f)
    ids = [inst["physical_instance_id"] for inst in data.get("instances", [])]
    assert len(ids) == len(set(ids)), f"Duplicate physical instance IDs in visual inputs: {len(ids)} vs {len(set(ids))}"


def test_visual_inputs_ground_truth_one_to_one_pairing():
    with open(EXPANDED_INPUTS, "r", encoding="utf-8") as f:
        in_ids = {inst["physical_instance_id"] for inst in json.load(f).get("instances", [])}
    with open(EXPANDED_GT, "r", encoding="utf-8") as f:
        gt_ids = set(json.load(f).get("labels", {}).keys())

    assert in_ids == gt_ids, f"Mismatch between visual inputs and ground truth IDs! Diff: {in_ids ^ gt_ids}"


def test_all_visual_manifest_images_exist_and_hashes_match():
    with open(EXPANDED_INPUTS, "r", encoding="utf-8") as f:
        data = json.load(f)

    for inst in data.get("instances", []):
        vis = inst.get("visual_inputs", {})
        assert vis.get("has_images") is True
        crops = vis.get("crop_view_paths", [])
        scenes = vis.get("scene_view_paths", [])
        hashes = vis.get("image_sha256", [])

        assert len(crops) >= 1, f"Instance {inst['physical_instance_id']} has no crop paths"
        all_paths = crops + scenes

        assert len(all_paths) == len(hashes), f"Hash count mismatch for {inst['physical_instance_id']}"

        for p_str, expected_hash in zip(all_paths, hashes):
            full_p = repo_root / p_str
            assert full_p.exists(), f"Image path does not exist on disk: {full_p}"

            # Verify decodability
            with Image.open(full_p) as im:
                im.verify()

            # Verify SHA-256
            hasher = hashlib.sha256()
            with open(full_p, "rb") as f:
                while chunk := f.read(65536):
                    hasher.update(chunk)
            actual_hash = hasher.hexdigest()
            assert actual_hash == expected_hash, f"SHA-256 mismatch for {full_p}: expected {expected_hash}, got {actual_hash}"


def test_router_manifest_integrity():
    assert ROUTER_INPUTS.exists(), "Expanded router inputs manifest missing"
    assert ROUTER_GT.exists(), "Expanded router ground truth manifest missing"

    with open(ROUTER_INPUTS, "r", encoding="utf-8") as f:
        in_data = json.load(f)
    with open(ROUTER_GT, "r", encoding="utf-8") as f:
        gt_data = json.load(f)

    in_ids = {inst["physical_instance_id"] for inst in in_data.get("instances", [])}
    gt_ids = set(gt_data.get("labels", {}).keys())
    assert in_ids == gt_ids, f"Router inputs vs GT ID mismatch: {in_ids ^ gt_ids}"

    # Verify estimator evidence structure
    for inst in in_data.get("instances", []):
        assert "estimator_evidence" in inst, f"Missing estimator_evidence in router instance {inst['physical_instance_id']}"
        est = inst["estimator_evidence"]
        assert "observed_motion" in est
        assert "has_valid_pose" in est
