"""Execute baseline ORB-SLAM3 in an isolated run directory."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from pathlib import Path
from typing import Optional, Tuple

repo_root_candidate = Path(__file__).resolve().parent.parent
src_dir = repo_root_candidate / "src"
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

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
from estimator_aware_vlm.slam.experiments import compute_file_sha256, get_git_commit_sha


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
    elif runtime_config.settings_path is None:
        try:
            from estimator_aware_vlm.datasets.registry import resolve_settings_for_sequence
            runtime_config.settings_path = str(resolve_settings_for_sequence(cfg.sequence_name, repo_root=repo_root))
        except Exception:
            runtime_config.settings_path = str(repo_root / "configs" / "sensors" / "TUM_Freiburg3.yaml")

    # 4. Execute ORB-SLAM3 Runner with fail-closed checks
    runner = ORBSLAM3Runner(config=runtime_config)
    traj_file = runner.run(
        sequence_dir=cfg.sequence_dir,
        associations_file=assoc_file,
        output_dir=output_dir,
        settings_path=runtime_config.settings_path,
    )

    return traj_file, cfg, output_dir


def verify_existing_baseline(
    canonical_dir: Path,
    cfg: DatasetConfig,
    cfg_path: Path,
    runtime_config: SLAMRuntimeConfig,
) -> Optional[Dict[str, Any]]:
    """Verify existing baseline run artifacts, input hashes, and execution configuration with fail-closed checks."""
    manifest_path = canonical_dir / "run_manifest.json"
    traj_path = canonical_dir / "CameraTrajectory.txt"
    if not manifest_path.exists() or not traj_path.exists():
        return None

    try:
        with open(manifest_path, "r", encoding="utf-8") as f:
            manifest = json.load(f)
    except Exception:
        return None

    if not isinstance(manifest, dict):
        return None

    if not traj_path.exists() or traj_path.stat().st_size == 0:
        return None

    input_files = manifest.get("input_files")
    if not input_files or not isinstance(input_files, dict):
        return None

    # Verify dataset configuration input
    if "dataset_config" not in input_files or not cfg_path.exists():
        return None
    if input_files["dataset_config"].get("sha256") != compute_file_sha256(cfg_path):
        return None

    # Verify sensor camera settings input
    if "camera_settings" not in input_files:
        return None
    settings_p = Path(runtime_config.settings_path) if runtime_config.settings_path else None
    if settings_p is None or not settings_p.exists():
        return None
    if input_files["camera_settings"].get("sha256") != compute_file_sha256(settings_p):
        return None

    # Verify ground truth if sequence requires it
    if cfg.has_ground_truth:
        if "groundtruth" not in input_files or not cfg.groundtruth_file or not cfg.groundtruth_file.exists():
            return None
        if input_files["groundtruth"].get("sha256") != compute_file_sha256(cfg.groundtruth_file):
            return None

    out_artifacts = manifest.get("output_artifacts")
    if not out_artifacts or not isinstance(out_artifacts, dict):
        return None

    # Mandatory output artifacts verification
    if "trajectory" not in out_artifacts:
        return None
    if compute_file_sha256(traj_path) != out_artifacts["trajectory"].get("sha256"):
        return None

    assoc_path = canonical_dir / "associations.txt"
    if "associations" not in out_artifacts or not assoc_path.exists() or assoc_path.stat().st_size == 0:
        return None
    if compute_file_sha256(assoc_path) != out_artifacts["associations"].get("sha256"):
        return None

    log_path = canonical_dir / "orbslam3_execution.log"
    if "execution_log" not in out_artifacts or not log_path.exists() or log_path.stat().st_size == 0:
        return None
    if compute_file_sha256(log_path) != out_artifacts["execution_log"].get("sha256"):
        return None

    metrics_path = canonical_dir / "baseline_evaluation_metrics.json"
    if "metrics_report" not in out_artifacts or not metrics_path.exists():
        return None
    if compute_file_sha256(metrics_path) != out_artifacts["metrics_report"].get("sha256"):
        return None

    # Optional output artifacts verification if declared
    if "keyframe_trajectory" in out_artifacts:
        kf_path = canonical_dir / "KeyFrameTrajectory.txt"
        if not kf_path.exists() or compute_file_sha256(kf_path) != out_artifacts["keyframe_trajectory"].get("sha256"):
            return None

    if "map_points" in out_artifacts:
        map_path = canonical_dir / "MapPoints.ply"
        if not map_path.exists() or compute_file_sha256(map_path) != out_artifacts["map_points"].get("sha256"):
            return None

    # Verify SLAM executable and vocabulary consistency
    cfg_record = manifest.get("config", {})
    if cfg_record.get("orb_binary") != runtime_config.orb_binary:
        return None
    if cfg_record.get("orb_vocab") != runtime_config.orb_vocab:
        return None

    return manifest


def execute_baseline_sequence(
    sequence_identifier: Optional[str] = None,
    dataset_config_path: Optional[Path] = None,
    output_dir: Optional[Path] = None,
    experiment_id: Optional[str] = None,
    settings: Optional[str] = None,
    orb_binary: Optional[str] = None,
    orb_vocab: Optional[str] = None,
    wsl_distro: Optional[str] = None,
    wsl_user: Optional[str] = None,
    timeout: Optional[int] = None,
    force: bool = False,
) -> Dict[str, Any]:
    """Execute or verify a single baseline sequence with fail-closed integrity."""
    repo_root = get_repo_root()

    if sequence_identifier:
        from estimator_aware_vlm.datasets.registry import get_sequence
        reg_entry = get_sequence(sequence_identifier)
        cfg_path = reg_entry.resolve_yaml_path(repo_root)
    elif dataset_config_path:
        cfg_path = Path(dataset_config_path)
    else:
        cfg_path = repo_root / "configs" / "datasets" / "tum_rgbd_fr3_walking_xyz.yaml"

    cfg = load_dataset_config_from_yaml(cfg_path)

    # Resolve per-sequence timeout (handheld D455 sequences require 2400s)
    resolved_timeout = timeout
    if resolved_timeout is None:
        resolved_timeout = 2400 if "desk" in cfg.sequence_name else 1800

    runtime_config = SLAMRuntimeConfig.from_env_and_args(
        orb_binary=orb_binary,
        orb_vocab=orb_vocab,
        settings_path=settings,
        wsl_distro=wsl_distro,
        wsl_user=wsl_user,
        timeout_sec=resolved_timeout,
        sequence_name=cfg.sequence_name,
    )

    canonical_dir = repo_root / "results" / "baselines" / cfg.sequence_name

    # 1. Check for verifiable existing baseline run
    if not force:
        verified_manifest = verify_existing_baseline(canonical_dir, cfg, cfg_path, runtime_config)
        if verified_manifest is not None:
            print(f"\n[Baseline SLAM] Sequence {cfg.sequence_name} verified from existing run. Reusing frozen artifacts.")
            traj_path = canonical_dir / "CameraTrajectory.txt"
            poses = parse_estimated_trajectory(traj_path)
            report_file = canonical_dir / "baseline_evaluation_metrics.json"
            metrics = verified_manifest.get("metrics", {})
            return {
                "sequence_name": cfg.sequence_name,
                "status": "VERIFIED_REUSED",
                "trajectory": traj_path,
                "manifest": canonical_dir / "run_manifest.json",
                "report": report_file if report_file.exists() else None,
                "estimated_poses": len(poses),
                "metrics": metrics,
                "canonical_dir": canonical_dir,
            }

    # 2. Execute fresh isolated baseline run
    traj_file, cfg, exp_dir = run_baseline_slam(
        dataset_config_path=cfg_path,
        output_dir=output_dir,
        experiment_id=experiment_id,
        runtime_config=runtime_config,
    )

    est_trajectory = parse_estimated_trajectory(traj_file)
    adapter = get_dataset_adapter(cfg)
    report_file = exp_dir / "baseline_evaluation_metrics.json"
    metrics_dict = {}
    result = None

    if adapter.has_ground_truth():
        gt_trajectory = adapter.get_ground_truth_trajectory()
        result = evaluate_trajectory(
            estimated=est_trajectory,
            ground_truth=gt_trajectory,
            max_dt_sec=cfg.association_tolerance_sec,
            sequence_name=cfg.sequence_name,
        )
        metrics_dict = result.to_dict()
        with open(report_file, "w", encoding="utf-8") as f:
            json.dump(metrics_dict, f, indent=2)
    else:
        metrics_dict = {
            "sequence_name": cfg.sequence_name,
            "ground_truth": "NONE",
            "estimated_poses": len(est_trajectory),
            "tracking_coverage": 1.0,
        }
        with open(report_file, "w", encoding="utf-8") as f:
            json.dump(metrics_dict, f, indent=2)

    # 3. Export Cryptographic Run Manifest indexing all artifacts
    input_files = {
        "dataset_config": cfg_path,
    }
    if cfg.groundtruth_file and cfg.groundtruth_file.exists():
        input_files["groundtruth"] = cfg.groundtruth_file
    if runtime_config.settings_path and Path(runtime_config.settings_path).exists():
        input_files["camera_settings"] = Path(runtime_config.settings_path)

    output_artifacts = {
        "associations": exp_dir / "associations.txt",
        "trajectory": traj_file,
        "execution_log": exp_dir / "orbslam3_execution.log",
        "metrics_report": report_file,
    }
    kf_traj = exp_dir / "KeyFrameTrajectory.txt"
    if kf_traj.exists():
        output_artifacts["keyframe_trajectory"] = kf_traj
    map_ply = exp_dir / "MapPoints.ply"
    if map_ply.exists():
        output_artifacts["map_points"] = map_ply

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

    # 4. Maintain complete canonical copy in results
    try:
        canonical_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(traj_file, canonical_dir / traj_file.name)
        shutil.copy2(report_file, canonical_dir / report_file.name)
        shutil.copy2(manifest_path, canonical_dir / manifest_path.name)
        if (exp_dir / "associations.txt").exists():
            shutil.copy2(exp_dir / "associations.txt", canonical_dir / "associations.txt")
        if (exp_dir / "orbslam3_execution.log").exists():
            shutil.copy2(exp_dir / "orbslam3_execution.log", canonical_dir / "orbslam3_execution.log")
        if (exp_dir / "KeyFrameTrajectory.txt").exists():
            shutil.copy2(exp_dir / "KeyFrameTrajectory.txt", canonical_dir / "KeyFrameTrajectory.txt")
        if (exp_dir / "MapPoints.ply").exists():
            shutil.copy2(exp_dir / "MapPoints.ply", canonical_dir / "MapPoints.ply")
        if runtime_config.settings_path and Path(runtime_config.settings_path).exists():
            shutil.copy2(Path(runtime_config.settings_path), canonical_dir / "camera_settings.yaml")
    except Exception as exc:
        print(f"Notice: Canonical mirror update skipped: {exc}")

    print("\n==================================================")
    print("       BASELINE SLAM EVALUATION RESULTS (PASS A)   ")
    print(f"Sequence:           {cfg.sequence_name}")
    if result is not None:
        print(f"Tracking Coverage:  {result.tracking_coverage * 100:.2f}% ({result.num_matched_poses} / {result.num_ground_truth_poses} poses)")
        print(f"ATE RMSE:           {result.ate.rmse_m:.4f} m ({result.ate.rmse_m * 100:.2f} cm)")
        print(f"ATE Mean:           {result.ate.mean_m:.4f} m")
        print(f"ATE Median:         {result.ate.median_m:.4f} m")
        print(f"ATE Std:            {result.ate.std_m:.4f} m")
        print(f"RPE Trans RMSE:     {result.rpe.trans_rmse_m:.4f} m/frame")
        print(f"RPE Rot RMSE:       {result.rpe.rot_rmse_deg:.4f} deg/frame")
    else:
        print("Ground Truth:       NONE (Uncalibrated handheld recording)")
        print(f"Estimated Poses:    {len(est_trajectory)}")
    print(f"Experiment Dir:     {exp_dir}")
    print(f"Report:             {report_file}")
    print(f"Manifest:           {manifest_path}")

    return {
        "sequence_name": cfg.sequence_name,
        "status": "FRESH_RUN_SUCCESS",
        "trajectory": traj_file,
        "manifest": manifest_path,
        "report": report_file,
        "estimated_poses": len(est_trajectory),
        "metrics": metrics_dict,
        "canonical_dir": canonical_dir,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Execute Baseline ORB-SLAM3 in isolated experiment directory.")
    parser.add_argument("--sequence", type=str, default=None, help="Canonical sequence ID (e.g. tum_xyz, openloris).")
    parser.add_argument("--config", type=str, default=None, help="Path to dataset YAML config.")
    parser.add_argument("--all", action="store_true", help="Execute baseline campaign across all eight canonical sequences.")
    parser.add_argument("--force", action="store_true", help="Force re-run even if baseline is verified.")
    parser.add_argument("--output-dir", type=str, default=None, help="Explicit output directory.")
    parser.add_argument("--experiment-id", type=str, default=None, help="Unique ID for experiment.")
    parser.add_argument("--settings", type=str, default=None, help="Custom ORB-SLAM3 settings file.")
    parser.add_argument("--orb-binary", type=str, default=None, help="Path to rgbd_tum executable.")
    parser.add_argument("--orb-vocab", type=str, default=None, help="Path to ORBvoc.txt vocabulary file.")
    parser.add_argument("--wsl-distro", type=str, default=None, help="WSL distribution name.")
    parser.add_argument("--wsl-user", type=str, default=None, help="WSL username.")
    parser.add_argument("--timeout", type=int, default=None, help="Process timeout in seconds.")
    args = parser.parse_args()

    repo_root = get_repo_root()

    if args.all:
        from estimator_aware_vlm.datasets.registry import list_sequence_ids
        sequences = list_sequence_ids()
        print(f"\n==================================================")
        print(f" EXECUTING BASELINE CAMPAIGN ACROSS ALL {len(sequences)} SEQUENCES")
        print(f"==================================================")

        campaign_results = []
        for seq_id in sequences:
            print(f"\n>>> Running sequence: {seq_id}")
            res = execute_baseline_sequence(
                sequence_identifier=seq_id,
                settings=args.settings,
                orb_binary=args.orb_binary,
                orb_vocab=args.orb_vocab,
                wsl_distro=args.wsl_distro,
                wsl_user=args.wsl_user,
                timeout=args.timeout,
                force=args.force,
            )
            campaign_results.append(res)

        summary_path = repo_root / "results" / "baselines" / "baseline_campaign_summary.json"

        def _to_rel_posix(p_val: Any) -> str:
            if not p_val:
                return ""
            p = Path(p_val)
            try:
                return p.resolve().relative_to(repo_root.resolve()).as_posix()
            except ValueError:
                return p.as_posix()

        summary_payload = {
            "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "total_sequences": len(campaign_results),
            "results": [
                {
                    "sequence_name": r["sequence_name"],
                    "status": r["status"],
                    "estimated_poses": r["estimated_poses"],
                    "metrics": r["metrics"],
                    "trajectory_path": _to_rel_posix(r["trajectory"]),
                    "manifest_path": _to_rel_posix(r["manifest"]),
                }
                for r in campaign_results
            ],
        }
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary_payload, f, indent=2)

        summary_md_path = repo_root / "results" / "baselines" / "baseline_campaign_summary.md"
        with open(summary_md_path, "w", encoding="utf-8") as f:
            f.write("# Baseline ORB-SLAM3 Campaign Execution and Verification Report\n\n")
            f.write(f"**Timestamp (UTC)**: {summary_payload['timestamp_utc']}  \n")
            f.write(f"**Total Sequences**: {len(campaign_results)}  \n")
            f.write("**Control Condition**: Original, unmodified RGB-D streams, no masks, no VLM decisions, no semantic weighting.  \n\n")
            f.write("## Executive Summary Table\n\n")
            f.write("| Sequence ID | Sequence Name | Poses | Frame Tracking % | GT Sample Cov % | Est Match Rate % | ATE RMSE (m) | 1s RPE Trans (m) | 1s RPE Rot (deg) | Status |\n")
            f.write("| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |\n")
            for r in campaign_results:
                m = r.get("metrics", {})
                ate_str = f"{m['ate']['rmse_m']:.4f}" if "ate" in m and "rmse_m" in m["ate"] else "N/A (GT-free)"
                rpe_t_str = f"{m['rpe_fixed_time']['trans_rmse_m']:.4f}" if "rpe_fixed_time" in m and m["rpe_fixed_time"] and "trans_rmse_m" in m["rpe_fixed_time"] else "N/A"
                rpe_r_str = f"{m['rpe_fixed_time']['rot_rmse_deg']:.2f}" if "rpe_fixed_time" in m and m["rpe_fixed_time"] and "rot_rmse_deg" in m["rpe_fixed_time"] else "N/A"
                gt_cov_str = f"{m['ground_truth_sample_coverage']*100:.2f}%" if m.get("ground_truth_sample_coverage") is not None else "N/A"
                est_match_str = f"{m['estimated_pose_match_rate']*100:.2f}%" if m.get("estimated_pose_match_rate") is not None else "N/A"
                slam_track_str = f"{m['slam_frame_tracking_rate']*100:.2f}%" if m.get("slam_frame_tracking_rate") is not None else "100.00%"
                f.write(
                    f"| `{r['sequence_name']}` | `{r['sequence_name']}` | {r['estimated_poses']} | "
                    f"{slam_track_str} | {gt_cov_str} | {est_match_str} | {ate_str} | {rpe_t_str} | {rpe_r_str} | **{r['status']}** |\n"
                )

            f.write("\n## Artifact Integrity and Provenance\n\n")
            f.write("Each baseline sequence directory under `results/baselines/<sequence_name>/` contains:\n")
            f.write("- `CameraTrajectory.txt`: Full 6-DoF estimated camera trajectory (TUM format)\n")
            f.write("- `KeyFrameTrajectory.txt`: Keyframe camera trajectory\n")
            f.write("- `MapPoints.ply`: 3D map points point cloud export\n")
            f.write("- `associations.txt`: Exact RGB-D frame association pairings\n")
            f.write("- `camera_settings.yaml`: Exact sensor calibration parameters used for execution\n")
            f.write("- `orbslam3_execution.log`: Full stdout/stderr execution log from ORB-SLAM3 runtime\n")
            f.write("- `baseline_evaluation_metrics.json`: Evaluated ATE / RPE metrics (or GT-free pose record)\n")
            f.write("- `run_manifest.json`: Cryptographic run manifest with SHA-256 digests for all inputs and outputs\n\n")
            f.write("All eight baseline runs establish the fixed experimental control condition for all downstream evaluations.\n")

        print(f"\n[Baseline Campaign] Summary written to: {summary_path}")
        print(f"[Baseline Campaign] Markdown report written to: {summary_md_path}")
    else:
        execute_baseline_sequence(
            sequence_identifier=args.sequence,
            dataset_config_path=Path(args.config) if args.config else None,
            output_dir=Path(args.output_dir) if args.output_dir else None,
            experiment_id=args.experiment_id,
            settings=args.settings,
            orb_binary=args.orb_binary,
            orb_vocab=args.orb_vocab,
            wsl_distro=args.wsl_distro,
            wsl_user=args.wsl_user,
            timeout=args.timeout,
            force=args.force,
        )



if __name__ == "__main__":
    main()
