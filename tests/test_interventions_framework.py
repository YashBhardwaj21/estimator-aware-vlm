"""Unit tests for Counterfactual Interventions Schema, Planner, Manifest, and Evaluator."""

import json
from pathlib import Path

import pytest
import numpy as np

from estimator_aware_vlm.datasets.base import Trajectory, TrajectoryPose, TrajectoryRole
from estimator_aware_vlm.interventions.evaluation import evaluate_intervention_trajectory
from estimator_aware_vlm.interventions.manifest import InterventionManifest
from estimator_aware_vlm.interventions.planner import create_intervention_plan
from estimator_aware_vlm.interventions.schema import (
    InterventionAction,
    InterventionMetricRecord,
    InterventionRunResult,
    InterventionSpec,
    InterventionStatus,
    InterventionTargetType,
)


def test_intervention_action_targets():
    assert InterventionAction.KEEP.target_type == InterventionTargetType.NONE
    assert InterventionAction.IMAGE_POLYGON_MASK.target_type == InterventionTargetType.IMAGE_SPACE
    assert InterventionAction.IMAGE_BBOX_MASK.target_type == InterventionTargetType.IMAGE_SPACE
    assert InterventionAction.GRAPH_MARK_SUSPECT.target_type == InterventionTargetType.SCENE_GRAPH
    assert InterventionAction.GRAPH_TEMPORARY_EXCLUDE.target_type == InterventionTargetType.SCENE_GRAPH
    assert InterventionAction.LANDMARK_DOWNWEIGHT.target_type == InterventionTargetType.SLAM_LANDMARK
    assert InterventionAction.LANDMARK_TEMPORARY_EXCLUDE.target_type == InterventionTargetType.SLAM_LANDMARK
    assert InterventionAction.DUAL_INTEGRATION.target_type == InterventionTargetType.SLAM_LANDMARK


def test_intervention_planner():
    candidates = ["monitor_01", "person_01"]
    actions = [InterventionAction.KEEP, InterventionAction.IMAGE_POLYGON_MASK, InterventionAction.IMAGE_BBOX_MASK]
    plan = create_intervention_plan(
        experiment_run_id="run_test",
        dataset_id="tum",
        sequence_id="fr3_walking_xyz",
        candidate_tracks=candidates,
        actions=actions,
        repeats=1,
    )
    # KEEP (1) + candidates (2) * actions (2) = 5 specs
    assert len(plan) == 5
    ids = [s.intervention_id for s in plan]
    assert "tum_fr3_walking_xyz_baseline_KEEP_r01" in ids
    assert "tum_fr3_walking_xyz_monitor_01_IMAGE_POLYGON_MASK_r01" in ids
    assert "tum_fr3_walking_xyz_monitor_01_IMAGE_BBOX_MASK_r01" in ids
    assert "tum_fr3_walking_xyz_person_01_IMAGE_POLYGON_MASK_r01" in ids
    assert "tum_fr3_walking_xyz_person_01_IMAGE_BBOX_MASK_r01" in ids


def test_intervention_manifest(tmp_path: Path):
    spec = InterventionSpec(
        experiment_run_id="run_test",
        intervention_id="tum_fr3_walking_xyz_monitor_01_IMAGE_POLYGON_MASK_r01",
        dataset_id="tum",
        sequence_id="fr3_walking_xyz",
        track_id="monitor_01",
        action=InterventionAction.IMAGE_POLYGON_MASK,
        repeat_index=1,
    )
    manifest = InterventionManifest.create("run_test", [spec])
    assert manifest.total_planned == 1
    assert manifest.total_executed == 0

    res = InterventionRunResult(
        intervention_id=spec.intervention_id,
        spec=spec,
        status=InterventionStatus.SUCCESS,
        trajectory_path="/path/to/traj.txt",
        execution_time_sec=12.5,
    )
    manifest.record_result(res)
    assert manifest.total_executed == 1
    assert manifest.total_success == 1
    assert manifest.total_failed == 0

    out_file = tmp_path / "manifest.json"
    manifest.save(out_file)
    assert out_file.exists()

    loaded = InterventionManifest.load(out_file)
    assert loaded.total_planned == 1
    assert loaded.total_success == 1
    assert loaded.specs[spec.intervention_id]["track_id"] == "monitor_01"


def test_evaluate_intervention_missing_groundtruth(tmp_path: Path):
    spec = InterventionSpec(
        experiment_run_id="run_test",
        intervention_id="d455_desk_laptop_01_IMAGE_POLYGON_MASK_r01",
        dataset_id="d455",
        sequence_id="my_desk_recording_01",
        track_id="laptop_01",
        action=InterventionAction.IMAGE_POLYGON_MASK,
    )
    # Ground truth is None (as in D455)
    rec, paired = evaluate_intervention_trajectory(
        spec=spec,
        intervention_traj_path=None,
        baseline_traj_path=None,
        groundtruth_path=None,
        status=InterventionStatus.SUCCESS,
    )
    assert rec.ate_rmse_common_m is None
    assert rec.delta_ate_rmse_m is None
    assert "Ground truth unavailable" in rec.notes
    assert paired is None

