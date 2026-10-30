"""Run Baseline ORB-SLAM3 without dynamic filtering (Pass A) with isolated execution.

Executes ORB-SLAM3 in an isolated experiment directory, captures the estimated camera trajectory,
enforces fail-closed error detection on stale trajectories or crashes, evaluates tracking
performance against motion capture ground truth using SE(3) alignment, and exports canonical
evaluation metrics and cryptographic run manifests.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from pathlib import Path
from typing import Optional, Tuple

from estimator_aware_vlm.common.paths import get_repo_root
from estimator_aware_vlm.datasets import (
    DatasetConfig,
    Trajectory,
    TrajectoryPose,
    TrajectoryRole,
    get_dataset_adapter,
    load_dataset_config_from_yaml,
    parse_tum_ground_truth,
)
from estimator_aware_vlm.evaluation import evaluate_trajectory
from estimator_aware_vlm.slam import (
    EmptyTrajectoryError,
    ORBSLAM3Runner,
    SLAMExecutionError,
    SLAMRuntimeConfig,
    create_experiment_dir,
    export_run_manifest,
)


def parse_estimated_trajectory(file_path: Path) -> Trajectory:
    """Parse ORB-SLAM3 CameraTrajectory.txt into a Trajectory with TrajectoryRole.ESTIMATED."""
    if not file_path.exists():
        raise FileNotFoundError(f"Trajectory file not found: {file_path}")

    poses = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) >= 8:
                ts = float(parts[0])
                tx, ty, tz = float(parts[1]), float(parts[2]), float(parts[3])
                qx, qy, qz, qw = float(parts[4]), float(parts[5]), float(parts[6]), float(parts[7])
                poses.append(
                    TrajectoryPose(
                        timestamp=ts,
                        position=(tx, ty, tz),
                        orientation=(qx, qy, qz, qw),
                        role=TrajectoryRole.ESTIMATED,
                    )
                )

    poses.sort(key=lambda p: p.timestamp)
    if not poses:
        raise EmptyTrajectoryError(f"Trajectory at {file_path} contains 0 valid pose lines.")
    return Trajectory(poses=poses, role=TrajectoryRole.ESTIMATED, name="orbslam3_baseline_pass_a")


def run_baseline_slam(
    dataset_config_path: Optional[Path] = None,
    output_dir: Optional[Path] = None,
    experiment_id: Optional[str] = None,
    runtime_config: Optional[SLAMRuntimeConfig] = None,
) -> Tuple[Path, DatasetConfig, Path]:
    """Execute baseline SLAM with path isolation and fail-closed integrity.

    Returns:
        Tuple of (traj_file, dataset_cfg, output_dir).
    """
    repo_root = get_repo_root()

    if dataset_config_path is None:
        dataset_config_path = repo_root / "configs" / "datasets" / "tum_rgbd_fr3_walking_xyz.yaml"

    cfg = load_dataset_config_from_yaml(dataset_config_path)

    # 1. Resolve isolated output directory
    if output_dir is None:
        prefix = f"baseline_{cfg.sequence_name}"
        output_dir = create_experiment_dir(experiment_id=experiment_id, prefix=prefix)
    else:
        output_dir = Path(output_dir).resolve()
        output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n[Baseline SLAM] Running on sequence: {cfg.sequence_name}")
    print(f"[Baseline SLAM] Isolated Run Directory: {output_dir}")

    # 2. Generate associations inside the isolated output directory (NEVER in raw directory)
    assoc_file = output_dir / "associations.txt"
    if not assoc_file.exists():
        print(f"[Baseline SLAM] Generating associations in experiment dir: {assoc_file.name}")
        adapter = get_dataset_adapter(cfg)
        adapter.export_associations(assoc_file)

    # 3. Resolve SLAM runtime configuration
    if runtime_config is None:
        runtime_config = SLAMRuntimeConfig.from_env_and_args(sequence_name=cfg.sequence_name)

    # 4. Execute ORB-SLAM3 Runner with fail-closed checks
    runner = ORBSLAM3Runner(config=runtime_config)
    traj_file = runner.run(
        sequence_dir=cfg.sequence_dir,
        associations_file=assoc_file,
        output_dir=output_dir,
        settings_path=runtime_config.settings_path,
    )

    return traj_file, cfg, output_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="Execute Baseline ORB-SLAM3 in isolated experiment directory.")
    parser.add_argument("--config", type=str, default=None, help="Path to dataset YAML config.")
    parser.add_argument("--output-dir", type=str, default=None, help="Explicit output directory.")
    parser.add_argument("--experiment-id", type=str, default=None, help="Unique ID for experiment.")
    parser.add_argument("--settings", type=str, default=None, help="Custom ORB-SLAM3 settings file.")
    parser.add_argument("--orb-binary", type=str, default=None, help="Path to rgbd_tum executable.")
    parser.add_argument("--orb-vocab", type=str, default=None, help="Path to ORBvoc.txt vocabulary file.")
    parser.add_argument("--wsl-distro", type=str, default=None, help="WSL distribution name.")
    parser.add_argument("--wsl-user", type=str, default=None, help="WSL username.")
    parser.add_argument("--timeout", type=int, default=900, help="Process timeout in seconds.")
    args = parser.parse_args()

    repo_root = get_repo_root()
    cfg_path = Path(args.config) if args.config else (repo_root / "configs" / "datasets" / "tum_rgbd_fr3_walking_xyz.yaml")
    out_dir = Path(args.output_dir) if args.output_dir else None

    # Resolve runtime config from args and environment
    runtime_config = SLAMRuntimeConfig.from_env_and_args(
        orb_binary=args.orb_binary,
        orb_vocab=args.orb_vocab,
        settings_path=args.settings,
        wsl_distro=args.wsl_distro,
        wsl_user=args.wsl_user,
        timeout_sec=args.timeout,
    )

    # 1. Run Baseline SLAM
    traj_file, cfg, exp_dir = run_baseline_slam(
        dataset_config_path=cfg_path,
        output_dir=out_dir,
        experiment_id=args.experiment_id,
        runtime_config=runtime_config,
    )
    print(f"[Baseline SLAM] Generated trajectory: {traj_file}")

    # 2. Ingest Estimated Trajectory
    est_trajectory = parse_estimated_trajectory(traj_file)
    print(f"[Baseline SLAM] Parsed {len(est_trajectory)} estimated camera poses.")

    # 3. Ingest Ground Truth Trajectory via dataset adapter
    adapter = get_dataset_adapter(cfg)
    gt_trajectory = adapter.get_ground_truth_trajectory()
    print(f"[Baseline SLAM] Parsed {len(gt_trajectory)} reference ground-truth poses from {adapter.__class__.__name__}.")

    # 4. Evaluate Trajectory Metrics with SE(3) Umeyama Alignment
    result = evaluate_trajectory(
        estimated=est_trajectory,
        ground_truth=gt_trajectory,
        max_dt_sec=cfg.association_tolerance_sec,
        sequence_name=cfg.sequence_name,
    )

    # 5. Export JSON Report
    metrics_dict = result.to_dict()
    report_file = exp_dir / "baseline_evaluation_metrics.json"
    with open(report_file, "w", encoding="utf-8") as f:
        json.dump(metrics_dict, f, indent=2)

    # 6. Export Cryptographic Run Manifest
    input_files = {
        "dataset_config": cfg_path,
        "groundtruth": gt_file,
    }
    if runtime_config.settings_path and Path(runtime_config.settings_path).exists():
        input_files["camera_settings"] = Path(runtime_config.settings_path)

    output_artifacts = {
        "associations": exp_dir / "associations.txt",
        "trajectory": traj_file,
        "execution_log": exp_dir / "orbslam3_execution.log",
        "metrics_report": report_file,
    }

    manifest_path = export_run_manifest(
        experiment_dir=exp_dir,
        run_id=exp_dir.name,
        pipeline_module="Baseline SLAM Subsystem",
        config={
            "sequence_name": cfg.sequence_name,
            "orb_binary": runtime_config.orb_binary,
            "orb_vocab": runtime_config.orb_vocab,
            "settings_path": runtime_config.settings_path,
            "wsl_distro": runtime_config.wsl_distro,
            "wsl_user": runtime_config.wsl_user,
            "timeout_sec": runtime_config.timeout_sec,
        },
        input_files=input_files,
        output_artifacts=output_artifacts,
        metrics=metrics_dict,
    )

    # 7. Maintain canonical output symlink/copy for legacy compatibility
    suffix = "" if "freiburg3" in cfg.sequence_name else f"_{cfg.sequence_name}"
    legacy_dir = repo_root / "outputs" / f"baseline_slam{suffix}"
    try:
        legacy_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(traj_file, legacy_dir / traj_file.name)
        shutil.copy2(report_file, legacy_dir / report_file.name)
        shutil.copy2(manifest_path, legacy_dir / manifest_path.name)
    except Exception as exc:
        print(f"Notice: Legacy mirror update skipped: {exc}")

    print("\n==================================================")
    print("       BASELINE SLAM EVALUATION RESULTS (PASS A)   ")
    print("==================================================")
    print(f"Sequence:           {result.sequence_name}")
    print(f"Tracking Coverage:  {result.tracking_coverage * 100:.2f}% ({result.num_matched_poses} / {result.num_ground_truth_poses} poses)")
    print(f"ATE RMSE:           {result.ate.rmse_m:.4f} m ({result.ate.rmse_m * 100:.2f} cm)")
    print(f"ATE Mean:           {result.ate.mean_m:.4f} m")
    print(f"ATE Median:         {result.ate.median_m:.4f} m")
    print(f"ATE Std:            {result.ate.std_m:.4f} m")
    print(f"RPE Trans RMSE:     {result.rpe.trans_rmse_m:.4f} m/frame")
    print(f"RPE Rot RMSE:       {result.rpe.rot_rmse_deg:.4f} deg/frame")
    print(f"Experiment Dir:     {exp_dir}")
    print(f"Report:             {report_file}")
    print(f"Manifest:           {manifest_path}")
    print("==================================================\n")


if __name__ == "__main__":
    main()
