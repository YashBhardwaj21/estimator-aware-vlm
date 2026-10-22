#!/usr/bin/env python3
"""Unit and integration tests for the Ground-Truth Movability Benchmark."""

import json
from pathlib import Path
import pytest


@pytest.fixture
def benchmark_paths():
    repo_root = Path(__file__).resolve().parent.parent.parent
    benchmarks_dir = repo_root / "vlm" / "benchmarks"
    return {
        "gt_json": benchmarks_dir / "movability_ground_truth.json",
        "events_json": benchmarks_dir / "controlled_scene_events.json",
        "metrics_json": benchmarks_dir / "benchmark_summary_metrics.json",
        "protocol_md": benchmarks_dir / "GROUND_TRUTH_PROTOCOL.md",
        "report_md": benchmarks_dir / "BENCHMARK_REPORT.md",
    }


def test_benchmark_files_exist(benchmark_paths):
    """Verify all benchmark artifacts are created and non-empty."""
    for name, path in benchmark_paths.items():
        assert path.exists(), f"Benchmark file {name} ({path}) does not exist"
        assert path.stat().st_size > 0, f"Benchmark file {name} is empty"


def test_required_fields_present_in_all_instances(benchmark_paths):
    """Ensure every physical object has all required ground-truth schema fields."""
    with open(benchmark_paths["gt_json"], "r", encoding="utf-8") as f:
        data = json.load(f)

    physical_objects = data.get("physical_objects", [])
    assert len(physical_objects) > 0, "No physical objects found in benchmark"

    required_fields = [
        "object_instance_id",
        "semantic_class",
        "movability_gt",
        "current_motion_gt",
        "displacement_gt",
        "scene_event",
        "verification_method",
        "slam_invalidated",
        "associated_tracks",
        "centroid_floor_m",
        "dimensions_m",
    ]

    for obj in physical_objects:
        for field in required_fields:
            assert field in obj, f"Object {obj.get('object_instance_id')} missing field: {field}"
            assert obj[field] is not None, f"Object {obj.get('object_instance_id')} has null field: {field}"


def test_rudt_taxonomy_categories(benchmark_paths):
    """Ensure movability categories strictly follow Rüdt et al. standard."""
    with open(benchmark_paths["gt_json"], "r", encoding="utf-8") as f:
        data = json.load(f)

    valid_movability = {"IMMOVABLE", "MOVABLE", "MOBILE", "UNKNOWN"}
    valid_motion = {"STATIC", "MOVING", "UNCERTAIN"}

    category_counts = {"IMMOVABLE": 0, "MOVABLE": 0, "MOBILE": 0, "UNKNOWN": 0}

    for obj in data["physical_objects"]:
        mov = obj["movability_gt"]
        mot = obj["current_motion_gt"]
        assert mov in valid_movability, f"Invalid movability category: {mov}"
        assert mot in valid_motion, f"Invalid motion state: {mot}"
        category_counts[mov] += 1

    # Acceptance gate: verify instances exist in each relevant category
    assert category_counts["IMMOVABLE"] > 0, "Missing IMMOVABLE instances"
    assert category_counts["MOVABLE"] > 0, "Missing MOVABLE instances"
    assert category_counts["MOBILE"] > 0, "Missing MOBILE instances"
    assert category_counts["UNKNOWN"] == 0, "Ground truth benchmark should have 0 UNKNOWN annotations"


def test_disentangled_affordance_vs_motion(benchmark_paths):
    """Verify that affordance and motion are disentangled."""
    with open(benchmark_paths["gt_json"], "r", encoding="utf-8") as f:
        data = json.load(f)

    objects_by_id = {obj["object_instance_id"]: obj for obj in data["physical_objects"]}

    # 1. Stationary desk/chair: MOVABLE affordance, but STATIC motion and False slam_invalidated
    desk = objects_by_id["inst_desk_001"]
    assert desk["movability_gt"] == "MOVABLE"
    assert desk["current_motion_gt"] == "STATIC"
    assert desk["displacement_gt"] == 0.0
    assert desk["slam_invalidated"] is False

    # 2. Relocated chair: MOVABLE affordance, but MOVING motion and True slam_invalidated
    moved_chair = objects_by_id["inst_chair_002"]
    assert moved_chair["movability_gt"] == "MOVABLE"
    assert moved_chair["current_motion_gt"] == "MOVING"
    assert moved_chair["displacement_gt"] > 0.30
    assert moved_chair["slam_invalidated"] is True

    # 3. Mobile robot: MOBILE affordance, MOVING motion, and True slam_invalidated
    robot = objects_by_id["inst_rc_car_robot_001"]
    assert robot["movability_gt"] == "MOBILE"
    assert robot["current_motion_gt"] == "MOVING"
    assert robot["slam_invalidated"] is True

    # 4. Architectural door: IMMOVABLE affordance, STATIC motion, and False slam_invalidated
    door = objects_by_id["inst_door_001"]
    assert door["movability_gt"] == "IMMOVABLE"
    assert door["current_motion_gt"] == "STATIC"
    assert door["displacement_gt"] == 0.0
    assert door["slam_invalidated"] is False


def test_track_deduplication_acceptance_gate(benchmark_paths):
    """Verify raw tracks are clustered into physical instances, not repeated tracks."""
    with open(benchmark_paths["gt_json"], "r", encoding="utf-8") as f:
        data = json.load(f)

    stats = data["statistics"]
    total_physical = stats["total_verified_physical_instances"]
    raw_tracks = stats["raw_tracks_resolved"]

    # Physical instances must be strictly less than raw causal tracks
    assert total_physical < raw_tracks
    assert raw_tracks == 78
    assert total_physical >= 40, "Must have at least 40 verified physical instances"


def test_inter_annotator_agreement_and_uncertainty(benchmark_paths):
    """Verify annotator agreement and measurement uncertainty metrics."""
    with open(benchmark_paths["metrics_json"], "r", encoding="utf-8") as f:
        metrics = json.load(f)

    agreement = metrics["inter_annotator_agreement"]
    assert agreement["raw_agreement_pct"] >= 95.0
    assert agreement["cohens_kappa"] >= 0.85  # Near-perfect agreement

    unc = metrics["displacement_measurement_uncertainty"]
    assert unc["tape_measure_sigma_m"] == 0.020
    assert unc["se2_registration_residual_sigma_m"] == 0.1198
    assert unc["combined_uncertainty_sigma_m"] == 0.1215
    assert agreement["discrepancies_resolved"] == 2


def test_controlled_events_consistency(benchmark_paths):
    """Verify that controlled scene events match recorded displacements."""
    with open(benchmark_paths["events_json"], "r", encoding="utf-8") as f:
        event_log = json.load(f)

    events = event_log["events"]
    assert len(events) >= 5

    relocation_events = [e for e in events if e.get("event_type") == "PHYSICAL_RELOCATION"]
    assert len(relocation_events) == 3

    target_ids = {e["target_object_id"] for e in relocation_events}
    assert "inst_chair_002" in target_ids
    assert "inst_chair_003" in target_ids
    assert "inst_storage_box_007" in target_ids

    for r_evt in relocation_events:
        assert r_evt["measured_displacement_m"] > 0.30
        assert r_evt["slam_invalidated"] is True
