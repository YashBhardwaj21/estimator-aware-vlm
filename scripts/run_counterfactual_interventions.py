"""Automated Counterfactual SLAM Intervention Benchmark Runner.

Executes controlled counterfactual interventions across TUM RGB-D, OpenLORIS, and D455:
- C0: KEEP (unmodified baseline)
- C1: IMAGE_POLYGON_MASK (pixel-level instance segmentation contour)
- C2: IMAGE_BBOX_MASK (bounding box suppression)
- C3: GRAPH_MARK_SUSPECT (scene-graph lifecycle: mark uncertain)
- C4: GRAPH_TEMPORARY_EXCLUDE (scene-graph lifecycle: temporarily exclude)
- C5: LANDMARK_DOWNWEIGHT (native SLAM optimizer weighting - NOT_IMPLEMENTED diagnostic)
- C6: LANDMARK_TEMPORARY_EXCLUDE (native SLAM landmark exclusion - NOT_IMPLEMENTED diagnostic)
- C7: DUAL_INTEGRATION (joint image/landmark suppression - NOT_IMPLEMENTED diagnostic)

Strictly evaluates trajectory outcomes on common ground-truth timestamp support:
    Delta_i(a) = L(C0) - L(C_i=a)
Upholds zero-fabrication: D455 ATE metrics remain strictly null/None.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

# Ensure src is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from estimator_aware_vlm.common.paths import get_repo_root
from estimator_aware_vlm.datasets import (
    TrajectoryRole,
    get_dataset_adapter,
    load_dataset_config_from_yaml,
    parse_tum_ground_truth,
    parse_tum_trajectory,
)
from estimator_aware_vlm.interventions.evaluation import (
    evaluate_intervention_trajectory,
)
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
from estimator_aware_vlm.slam import (
    EmptyTrajectoryError,
    ORBSLAM3Runner,
    SLAMExecutionError,
    SLAMRuntimeConfig,
)

# Import masking engine from run_pass_b_slam
from run_pass_b_slam import generate_pass_b_masked_associations


ACTION_NAME_MAP: Dict[str, InterventionAction] = {
    "C0": InterventionAction.KEEP,
    "KEEP": InterventionAction.KEEP,
    "C1": InterventionAction.IMAGE_POLYGON_MASK,
    "IMAGE_POLYGON_MASK": InterventionAction.IMAGE_POLYGON_MASK,
    "C2": InterventionAction.IMAGE_BBOX_MASK,
    "IMAGE_BBOX_MASK": InterventionAction.IMAGE_BBOX_MASK,
    "C3": InterventionAction.GRAPH_MARK_SUSPECT,
    "GRAPH_MARK_SUSPECT": InterventionAction.GRAPH_MARK_SUSPECT,
    "C4": InterventionAction.GRAPH_TEMPORARY_EXCLUDE,
    "GRAPH_TEMPORARY_EXCLUDE": InterventionAction.GRAPH_TEMPORARY_EXCLUDE,
    "C5": InterventionAction.LANDMARK_DOWNWEIGHT,
    "LANDMARK_DOWNWEIGHT": InterventionAction.LANDMARK_DOWNWEIGHT,
    "C6": InterventionAction.LANDMARK_TEMPORARY_EXCLUDE,
    "LANDMARK_TEMPORARY_EXCLUDE": InterventionAction.LANDMARK_TEMPORARY_EXCLUDE,
    "C7": InterventionAction.DUAL_INTEGRATION,
    "DUAL_INTEGRATION": InterventionAction.DUAL_INTEGRATION,
}


def parse_action_arg(action_str: str) -> InterventionAction:
    cleaned = action_str.strip().upper()
    if cleaned in ACTION_NAME_MAP:
        return ACTION_NAME_MAP[cleaned]
    raise ValueError(f"Unknown intervention action: {action_str}. Valid choices: {list(ACTION_NAME_MAP.keys())}")


def execute_single_intervention(
    spec: InterventionSpec,
    output_dir: Path,
    runtime_config: SLAMRuntimeConfig,
    skip_slam_execution: bool = False,
) -> Tuple[InterventionRunResult, InterventionMetricRecord]:
    """Execute a single intervention run end-to-end with fail-closed checks."""
    output_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    # 1. Handle Native Landmark Actions (C5, C6, C7)
    if spec.target_type == InterventionTargetType.SLAM_LANDMARK:
        msg = (
            f"Action {spec.action.value} requires native ORB-SLAM3 C++ MapPoint optimizer kernel "
            f"integration, which is unverified/unavailable in external binary. Recorded as NOT_IMPLEMENTED."
        )
        print(f"  [NOT_IMPLEMENTED] {spec.intervention_id}: {msg}")
        result = InterventionRunResult(
            intervention_id=spec.intervention_id,
            spec=spec,
            status=InterventionStatus.NOT_IMPLEMENTED,
            error_message=msg,
            execution_time_sec=0.0,
        )
        record = InterventionMetricRecord(
            intervention_id=spec.intervention_id,
            dataset_id=spec.dataset_id,
            sequence_id=spec.sequence_id,
            track_id=spec.track_id,
            action=spec.action.value,
            repeat_index=spec.repeat_index,
            status=InterventionStatus.NOT_IMPLEMENTED.value,
            notes=msg,
        )
        return result, record

    # 2. Handle Scene Graph Actions (C3, C4)
    if spec.target_type == InterventionTargetType.SCENE_GRAPH:
        msg = f"Scene-graph lifecycle intervention {spec.action.value} is handled via scene-graph evaluator."
        result = InterventionRunResult(
            intervention_id=spec.intervention_id,
            spec=spec,
            status=InterventionStatus.SUCCESS,
            execution_time_sec=0.0,
            runtime_metadata={"lifecycle_action": spec.action.value},
        )
        record = InterventionMetricRecord(
            intervention_id=spec.intervention_id,
            dataset_id=spec.dataset_id,
            sequence_id=spec.sequence_id,
            track_id=spec.track_id,
            action=spec.action.value,
            repeat_index=spec.repeat_index,
            status=InterventionStatus.SUCCESS.value,
            notes=msg,
        )
        return result, record

    # 3. Handle Image-Space / Baseline Actions (C0, C1, C2)
    cfg_path = Path(spec.config_path) if spec.config_path else None
    if cfg_path is None or not cfg_path.exists():
        raise FileNotFoundError(f"Dataset config path not found: {spec.config_path}")
    cfg = load_dataset_config_from_yaml(cfg_path)
    adapter = get_dataset_adapter(cfg)

    obs_path = Path(spec.observations_path) if spec.observations_path else None
    gt_path = Path(spec.groundtruth_path) if spec.groundtruth_path else None
    base_traj_path = Path(spec.baseline_trajectory_path) if spec.baseline_trajectory_path else None

    # Determine mask settings
    mask_type = "polygon"
    if spec.action == InterventionAction.IMAGE_BBOX_MASK:
        mask_type = "bbox"

    masked_track_ids: Set[str] = set()
    if spec.action in (InterventionAction.IMAGE_POLYGON_MASK, InterventionAction.IMAGE_BBOX_MASK):
        if spec.track_id:
            masked_track_ids.add(spec.track_id)

    # Output associations and causality
    causality_report_path = output_dir / "mask_causality_report.json"
    dummy_routing = output_dir / "intervention_routing_context.json"
    with open(dummy_routing, "w", encoding="utf-8") as f:
        json.dump(
            {
                "intervention_id": spec.intervention_id,
                "action": spec.action.value,
                "target_track_id": spec.track_id,
                "mask_type": mask_type,
            },
            f,
            indent=2,
        )

    print(f"\n--- Running Intervention: {spec.intervention_id} ---")
    print(f"  Target: {spec.track_id or 'BASELINE (KEEP)'} | Action: {spec.action.value} | Mask: {mask_type}")

    if spec.action == InterventionAction.KEEP:
        # C0 uses raw associations directly
        assoc_file = output_dir / "associations_raw.txt"
        if not assoc_file.exists():
            adapter.export_associations(assoc_file)
    else:
        if obs_path is None or not obs_path.exists():
            raise FileNotFoundError(f"Observations file not found: {obs_path}")
        assoc_file = generate_pass_b_masked_associations(
            raw_seq_dir=cfg.sequence_dir,
            output_dir=output_dir,
            masked_track_ids=masked_track_ids,
            observations_path=obs_path,
            routing_decisions_path=dummy_routing,
            causality_report_path=causality_report_path,
            dataset_cfg=cfg,
            mask_type=mask_type,
        )

    traj_file = output_dir / "CameraTrajectory.txt"
    eval_json = output_dir / "evaluation_metrics.json"

    # Fast-path: Check for already completed run
    if traj_file.exists() and eval_json.exists() and not skip_slam_execution:
        try:
            with open(eval_json, "r", encoding="utf-8") as f:
                cached_metrics = json.load(f)
            if cached_metrics.get("status") == InterventionStatus.SUCCESS.value:
                print(f"  [CACHE] Found completed run with valid trajectory. Using cached metrics.")
                record, _ = evaluate_intervention_trajectory(
                    spec=spec,
                    intervention_traj_path=traj_file,
                    baseline_traj_path=base_traj_path,
                    groundtruth_path=gt_path,
                    association_tolerance_sec=cfg.association_tolerance_sec,
                    execution_time_sec=cached_metrics.get("execution_time_sec", 0.0),
                    status=InterventionStatus.SUCCESS,
                )
                run_res = InterventionRunResult(
                    intervention_id=spec.intervention_id,
                    spec=spec,
                    status=InterventionStatus.SUCCESS,
                    trajectory_path=str(traj_file),
                    metrics=cached_metrics,
                    execution_time_sec=cached_metrics.get("execution_time_sec", 0.0),
                )
                return run_res, record
        except Exception as exc:
            print(f"  Notice: Could not load cache: {exc}")

    if skip_slam_execution:
        print("  [SKIP] Skipping SLAM execution per CLI flag.")
        exec_time = time.time() - t0
        run_res = InterventionRunResult(
            intervention_id=spec.intervention_id,
            spec=spec,
            status=InterventionStatus.SUCCESS if traj_file.exists() else InterventionStatus.PENDING,
            trajectory_path=str(traj_file) if traj_file.exists() else None,
            execution_time_sec=exec_time,
        )
        rec, _ = evaluate_intervention_trajectory(
            spec=spec,
            intervention_traj_path=traj_file if traj_file.exists() else None,
            baseline_traj_path=base_traj_path,
            groundtruth_path=gt_path,
            association_tolerance_sec=cfg.association_tolerance_sec,
            execution_time_sec=exec_time,
            status=InterventionStatus.SUCCESS if traj_file.exists() else InterventionStatus.PENDING,
        )
        return run_res, rec

    # Execute SLAM
    runner = ORBSLAM3Runner(config=runtime_config)
    try:
        t_slam_start = time.time()
        produced_traj = runner.run(
            sequence_dir=cfg.sequence_dir,
            associations_file=assoc_file,
            output_dir=output_dir,
            settings_path=runtime_config.settings_path,
        )
        slam_dur = time.time() - t_slam_start
        print(f"  SLAM completed in {slam_dur:.2f}s -> {produced_traj.name}")

        status = InterventionStatus.SUCCESS
        err_msg = None
    except Exception as exc:
        print(f"  [ERROR] SLAM execution failed: {exc}")
        status = InterventionStatus.FAILED
        err_msg = str(exc)
        produced_traj = None

    exec_time = time.time() - t0

    # Evaluate trajectory
    record, paired_res = evaluate_intervention_trajectory(
        spec=spec,
        intervention_traj_path=produced_traj,
        baseline_traj_path=base_traj_path,
        groundtruth_path=gt_path,
        association_tolerance_sec=cfg.association_tolerance_sec,
        execution_time_sec=exec_time,
        status=status,
        notes=err_msg or "",
    )

    metrics_dict = record.to_dict()
    if paired_res:
        metrics_dict["paired_details"] = paired_res.to_dict()

    eval_json = output_dir / "evaluation_metrics.json"
    with open(eval_json, "w", encoding="utf-8") as f:
        json.dump(metrics_dict, f, indent=2)

    run_res = InterventionRunResult(
        intervention_id=spec.intervention_id,
        spec=spec,
        status=status,
        trajectory_path=str(produced_traj) if produced_traj else None,
        metrics=metrics_dict,
        error_message=err_msg,
        execution_time_sec=exec_time,
    )

    if record.ate_rmse_common_m is not None and record.ate_baseline_rmse_common_m is not None:
        print(f"  Common ATE (Baseline C0): {record.ate_baseline_rmse_common_m:.4f} m")
        print(f"  Common ATE (Intervention): {record.ate_rmse_common_m:.4f} m")
        print(f"  Delta ATE:                {record.delta_ate_rmse_m:+.4f} m ({record.percentage_ate_improvement:+.2f}%)")
        print(f"  Statistical Sig:          p={record.paired_ttest_pvalue:.4e} (sig={record.is_statistically_significant})")

    return run_res, record


def export_summary_tables(
    results_dir: Path,
    records: List[InterventionMetricRecord],
) -> Tuple[Path, Path]:
    """Export summary results to CSV and JSON."""
    csv_path = results_dir / "counterfactual_results.csv"
    json_path = results_dir / "counterfactual_results.json"

    fieldnames = [
        "intervention_id",
        "dataset_id",
        "sequence_id",
        "track_id",
        "action",
        "repeat_index",
        "status",
        "ate_baseline_rmse_common_m",
        "ate_rmse_common_m",
        "delta_ate_rmse_m",
        "percentage_ate_improvement",
        "ate_rmse_independent_m",
        "rpe_trans_rmse_m",
        "rpe_rot_rmse_deg",
        "tracking_coverage",
        "common_poses_count",
        "jaccard_support_index",
        "paired_ttest_pvalue",
        "paired_wilcoxon_pvalue",
        "is_statistically_significant",
        "execution_time_sec",
        "notes",
    ]

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in records:
            row = r.to_dict()
            writer.writerow({k: row.get(k) for k in fieldnames})

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump([r.to_dict() for r in records], f, indent=2)

    return csv_path, json_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Reproducible Counterfactual SLAM Intervention Benchmark.")
    parser.add_argument("--run-id", type=str, default="run_20261031_counterfactual", help="Benchmark run ID.")
    parser.add_argument("--results-dir", type=str, default=None, help="Root results directory.")
    parser.add_argument("--datasets", type=str, default="tum,openloris", help="Datasets to evaluate (tum, openloris, d455).")
    parser.add_argument("--actions", type=str, default="C0,C1,C2", help="Comma-separated actions (C0..C7).")
    parser.add_argument("--target-tracks", type=str, default=None, help="Comma-separated track IDs.")
    parser.add_argument("--repeats", type=int, default=1, help="Repetitions per arm.")
    parser.add_argument("--smoke-test", action="store_true", help="Run minimal smoke test pilot.")
    parser.add_argument("--skip-slam-execution", action="store_true", help="Prepare masks without running SLAM.")
    parser.add_argument("--timeout", type=int, default=900, help="SLAM timeout seconds.")
    args = parser.parse_args()

    repo_root = get_repo_root()
    results_dir = (
        Path(args.results_dir).resolve()
        if args.results_dir
        else repo_root / "results" / "counterfactual_interventions" / args.run_id
    )
    results_dir.mkdir(parents=True, exist_ok=True)

    requested_datasets = [d.strip().lower() for d in args.datasets.split(",") if d.strip()]
    requested_actions = [parse_action_arg(a) for a in args.actions.split(",") if a.strip()]

    filter_tracks = [t.strip() for t in args.target_tracks.split(",") if t.strip()] if args.target_tracks else None

    # Load Manifest
    manifest_path = results_dir / "intervention_manifest.json"
    if manifest_path.exists():
        manifest = InterventionManifest.load(manifest_path)
    else:
        manifest = InterventionManifest.create(experiment_run_id=args.run_id, specs=[])

    all_specs: List[InterventionSpec] = []
    metric_records: List[InterventionMetricRecord] = []

    # 1. Dataset Configurations and Inputs
    # TUM
    tum_cfg_path = repo_root / "configs" / "datasets" / "tum_rgbd_fr3_walking_xyz.yaml"
    tum_gt_path = repo_root / "data" / "raw" / "rgbd_dataset_freiburg3_walking_xyz" / "groundtruth.txt"
    tum_base_traj = repo_root / "outputs" / "baseline_slam" / "CameraTrajectory.txt"
    tum_obs_path = repo_root / "outputs" / "perception_tracks" / "per_frame_instance_observations.jsonl"
    tum_summary_path = repo_root / "outputs" / "perception_tracks" / "tracks_summary.json"

    # OpenLORIS
    loris_cfg_path = repo_root / "configs" / "datasets" / "openloris_corridor1_4.yaml"
    loris_gt_path = repo_root / "data" / "raw" / "OpenLORIS-Scene_corridor1-4" / "groundtruth.txt"
    loris_base_traj = repo_root / "outputs" / "baseline_slam_OpenLORIS-Scene_corridor1-4" / "CameraTrajectory.txt"
    loris_obs_path = repo_root / "outputs" / "perception_tracks_OpenLORIS-Scene_corridor1-4" / "per_frame_instance_observations.jsonl"
    loris_summary_path = repo_root / "outputs" / "perception_tracks_OpenLORIS-Scene_corridor1-4" / "tracks_summary.json"

    # D455
    d455_cfg_path = repo_root / "configs" / "datasets" / "realsense_d455_recording_01.yaml"
    d455_gt_path = None  # Strictly uncalibrated
    d455_base_traj = None
    d455_obs_path = None

    # Resolve candidate tracks
    dataset_tracks_map: Dict[str, List[str]] = {}

    if "tum" in requested_datasets:
        if args.smoke_test:
            dataset_tracks_map["tum"] = ["monitor_01"]
        elif filter_tracks:
            dataset_tracks_map["tum"] = filter_tracks
        else:
            if tum_summary_path.exists():
                with open(tum_summary_path, "r", encoding="utf-8") as f:
                    tum_sum = json.load(f)
                dataset_tracks_map["tum"] = [t["track_id"] for t in tum_sum]
            else:
                dataset_tracks_map["tum"] = ["monitor_01", "chair_01", "person_01"]

    if "openloris" in requested_datasets:
        if args.smoke_test:
            dataset_tracks_map["openloris"] = ["person_01"]
        elif filter_tracks:
            dataset_tracks_map["openloris"] = [t for t in filter_tracks if "person" in t or t == "person_01"]
        else:
            if loris_summary_path.exists():
                with open(loris_summary_path, "r", encoding="utf-8") as f:
                    loris_sum = json.load(f)
                dataset_tracks_map["openloris"] = [t["track_id"] for t in loris_sum]
            else:
                dataset_tracks_map["openloris"] = ["person_01"]

    # Plan Interventions
    if "tum" in requested_datasets:
        tum_plan = create_intervention_plan(
            experiment_run_id=args.run_id,
            dataset_id="tum",
            sequence_id="fr3_walking_xyz",
            candidate_tracks=dataset_tracks_map.get("tum", []),
            actions=requested_actions,
            repeats=args.repeats,
            config_path=str(tum_cfg_path),
            groundtruth_path=str(tum_gt_path),
            baseline_trajectory_path=str(tum_base_traj),
            observations_path=str(tum_obs_path),
        )
        all_specs.extend(tum_plan)

    if "openloris" in requested_datasets:
        loris_plan = create_intervention_plan(
            experiment_run_id=args.run_id,
            dataset_id="openloris",
            sequence_id="corridor1-4",
            candidate_tracks=dataset_tracks_map.get("openloris", []),
            actions=requested_actions,
            repeats=args.repeats,
            config_path=str(loris_cfg_path),
            groundtruth_path=str(loris_gt_path),
            baseline_trajectory_path=str(loris_base_traj),
            observations_path=str(loris_obs_path),
        )
        all_specs.extend(loris_plan)

    print(f"\n================================================================================")
    print(f"      COUNTERFACTUAL BENCHMARK EXECUTION: {len(all_specs)} PLANNED INTERVENTIONS       ")
    print(f"================================================================================")
    for idx, spec in enumerate(all_specs):
        print(f"  {idx+1:02d}. [{spec.dataset_id.upper()}] {spec.intervention_id} ({spec.action.value})")

    # Update manifest specs
    for s in all_specs:
        manifest.specs[s.intervention_id] = s.to_dict()
    manifest.total_planned = len(manifest.specs)
    manifest.save(manifest_path)

    # Execute Specs
    for idx, spec in enumerate(all_specs):
        out_sub_dir = results_dir / spec.dataset_id / "interventions" / spec.intervention_id
        
        # Configure runtime
        settings_override = None
        if "openloris" in spec.dataset_id.lower() or "corridor" in spec.sequence_id.lower():
            settings_override = str(repo_root / "configs" / "sensors" / "OpenLORIS_D400.yaml")
        runtime_config = SLAMRuntimeConfig.from_env_and_args(
            settings_path=settings_override,
            timeout_sec=args.timeout,
            sequence_name=spec.sequence_id,
        )

        run_result, record = execute_single_intervention(
            spec=spec,
            output_dir=out_sub_dir,
            runtime_config=runtime_config,
            skip_slam_execution=args.skip_slam_execution,
        )

        manifest.record_result(run_result)
        manifest.save(manifest_path)
        metric_records.append(record)

    # Export Summary CSV and JSON
    csv_out, json_out = export_summary_tables(results_dir, metric_records)
    print("\n================================================================================")
    print("                    COUNTERFACTUAL BENCHMARK COMPLETE                          ")
    print("================================================================================")
    print(f"  Manifest: {manifest_path}")
    print(f"  CSV:      {csv_out}")
    print(f"  JSON:     {json_out}\n")


if __name__ == "__main__":
    main()
