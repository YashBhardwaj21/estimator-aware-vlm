"""Regression tests for benchmark manifest integrity, image validation, and leakage prevention."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from PIL import Image
import pytest

from estimator_aware_vlm.common.paths import get_repo_root

repo_root = get_repo_root()

CANONICAL_MANIFEST = repo_root / "benchmarks" / "canonical_benchmark_manifest.json"
EXPANDED_INPUTS = repo_root / "benchmarks" / "paper_benchmark_inputs_expanded.json"
EXPANDED_GT = repo_root / "benchmarks" / "paper_benchmark_ground_truth_expanded.json"
ROUTER_INPUTS = repo_root / "benchmarks" / "router_benchmark_inputs_expanded.json"
ROUTER_GT = repo_root / "benchmarks" / "router_benchmark_ground_truth_expanded.json"

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


def test_manifest_files_exist():
    assert CANONICAL_MANIFEST.exists(), "Canonical benchmark manifest missing"
    assert EXPANDED_INPUTS.exists(), "Expanded visual inputs manifest missing"
    assert EXPANDED_GT.exists(), "Expanded visual ground truth manifest missing"
    assert ROUTER_INPUTS.exists(), "Expanded router inputs manifest missing"
    assert ROUTER_GT.exists(), "Expanded router ground truth manifest missing"


def test_canonical_manifest_populations():
    """Verify population counts and memberships in canonical benchmark manifest."""
    with open(CANONICAL_MANIFEST, "r", encoding="utf-8") as f:
        data = json.load(f)

    pop_defs = data.get("population_definitions", {})
    assert "pop_52_ground_truth" in pop_defs
    assert "pop_42_image_eligible" in pop_defs
    assert "pop_53_router_evaluation" in pop_defs

    assert pop_defs["pop_52_ground_truth"]["count"] == 52
    assert pop_defs["pop_42_image_eligible"]["count"] == 42
    assert pop_defs["pop_53_router_evaluation"]["count"] == 53

    entities = data.get("entities", [])
    assert len(entities) == 52

    pop_42_count = sum(1 for e in entities if e.get("evaluation_memberships", {}).get("pop_42_image_eligible"))
    assert pop_42_count == 42, f"Expected 42 image eligible entities, found {pop_42_count}"

    # Verify all crops point to benchmarks/assets/multiview_samples/
    for e in entities:
        if e.get("visual_evidence", {}).get("has_images"):
            crops_dir = e["visual_evidence"].get("crops_dir")
            assert crops_dir is not None
            assert "benchmarks/assets/multiview_samples" in crops_dir
            for img in e["visual_evidence"].get("images", []):
                rel_p = img.get("relative_path", "")
                assert "benchmarks/assets/multiview_samples" in rel_p
                full_p = repo_root / rel_p
                assert full_p.exists(), f"Asset image missing: {full_p}"


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
    with open(ROUTER_INPUTS, "r", encoding="utf-8") as f:
        in_data = json.load(f)
    with open(ROUTER_GT, "r", encoding="utf-8") as f:
        gt_data = json.load(f)

    # Formally verify N=53 router population
    assert in_data.get("total_instances") == 53
    assert gt_data.get("total_instances") == 53

    in_instances = in_data.get("instances", [])
    assert len(in_instances) == 53

    in_ids = {inst["physical_instance_id"] for inst in in_instances}
    gt_ids = set(gt_data.get("labels", {}).keys())
    assert len(gt_ids) == 53
    assert in_ids == gt_ids, f"Router inputs vs GT ID mismatch: {in_ids ^ gt_ids}"

    # Verify inst_whiteboard_001 is included in the 53 instances
    assert "inst_whiteboard_001" in in_ids

    # Verify estimator evidence structure
    for inst in in_instances:
        assert "estimator_evidence" in inst, f"Missing estimator_evidence in router instance {inst['physical_instance_id']}"
        est = inst["estimator_evidence"]
        assert "observed_motion" in est
        assert "has_valid_pose" in est
