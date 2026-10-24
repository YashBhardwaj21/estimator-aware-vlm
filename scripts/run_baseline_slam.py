"""Run Baseline ORB-SLAM3 on TUM fr3_walking_xyz without dynamic filtering (Pass A).

Executes ORB-SLAM3 in headless RGB-D mode, captures the estimated camera trajectory,
evaluates tracking performance against motion capture ground truth using SE(3) alignment,
and exports canonical evaluation metrics.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional

from estimator_aware_vlm.common.paths import get_repo_root
from estimator_aware_vlm.datasets import (
    DatasetConfig,
    TUMRGBDAdapter,
    Trajectory,
    TrajectoryPose,
    TrajectoryRole,
    parse_tum_ground_truth,
)
from estimator_aware_vlm.evaluation import evaluate_trajectory


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
    return Trajectory(poses=poses, role=TrajectoryRole.ESTIMATED, name="orbslam3_baseline_pass_a")


def run_baseline_slam(
    dataset_config_path: Optional[Path] = None,
    output_dir: Optional[Path] = None,
    settings_path: Optional[Path] = None,
    timeout_sec: int = 900,
) -> Tuple[Path, DatasetConfig]:
    repo_root = get_repo_root()
    from estimator_aware_vlm.datasets import load_dataset_config_from_yaml

    if dataset_config_path is None:
        dataset_config_path = repo_root / "configs" / "datasets" / "tum_rgbd_fr3_walking_xyz.yaml"

    cfg = load_dataset_config_from_yaml(dataset_config_path)
    seq_dir = cfg.sequence_dir
    assoc_file = seq_dir / "associations.txt"

    if output_dir is None:
        suffix = "" if "freiburg3" in cfg.sequence_name else f"_{cfg.sequence_name}"
        output_dir = repo_root / "outputs" / f"baseline_slam{suffix}"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Ensure associations file is generated
    if not assoc_file.exists():
        print(f"Generating associations file at: {assoc_file}")
        adapter = TUMRGBDAdapter(cfg)
        adapter.export_associations(assoc_file)

    def to_wsl_path(win_path: Path) -> str:
        s = str(win_path.resolve()).replace("\\", "/")
        if len(s) >= 2 and s[1] == ":":
            drive = s[0].lower()
            return f"/mnt/{drive}{s[2:]}"
        return s

    wsl_seq_dir = to_wsl_path(seq_dir)
    wsl_assoc_file = to_wsl_path(assoc_file)
    wsl_out_dir = to_wsl_path(output_dir)

    orb_binary = "/home/saturn/ORB_SLAM3/Examples/RGB-D/rgbd_tum"
    orb_vocab = "/home/saturn/ORB_SLAM3/Vocabulary/ORBvoc.txt"

    if settings_path is None:
        if "openloris" in cfg.sequence_name.lower():
            settings_path = repo_root / "configs" / "sensors" / "OpenLORIS_D400.yaml"
        else:
            settings_path = Path("/home/saturn/ORB_SLAM3/Examples/RGB-D/TUM3.yaml")

    wsl_settings = to_wsl_path(settings_path) if settings_path.is_absolute() else str(settings_path)

    cmd = (
        f"cd '{wsl_out_dir}' && "
        f"'{orb_binary}' '{orb_vocab}' '{wsl_settings}' '{wsl_seq_dir}' '{wsl_assoc_file}' 0"
    )

    print(f"Executing ORB-SLAM3 Baseline in WSL...")
    print(f"Sequence:          {cfg.sequence_name}")
    print(f"Working Directory: {output_dir}")
    print(f"Command: {cmd}")

    import platform
    t0 = time.time()
    if platform.system() == "Windows":
        runner_cmd = ["wsl", "-d", "Ubuntu-22.04", "-u", "saturn", "--", "bash", "-c", cmd]
    else:
        runner_cmd = ["bash", "-c", cmd]

    res = subprocess.run(
        runner_cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        timeout=timeout_sec,
    )
    duration = time.time() - t0
    print(f"Execution completed in {duration:.2f}s with return code {res.returncode}")

    log_file = output_dir / "orbslam3_execution.log"
    with open(log_file, "w", encoding="utf-8") as f:
        f.write(res.stdout)

    traj_file = output_dir / "CameraTrajectory.txt"
    if not traj_file.exists():
        alt_traj = output_dir / "KeyFrameTrajectory.txt"
        if alt_traj.exists():
            print(f"Notice: CameraTrajectory.txt missing, found KeyFrameTrajectory.txt")
            traj_file = alt_traj
        else:
            raise RuntimeError(
                f"ORB-SLAM3 failed to produce trajectory file. See log: {log_file}\n"
                f"Tail of stdout:\n{res.stdout[-1500:] if res.stdout else 'None'}"
            )

    return traj_file, cfg


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description="Execute Baseline ORB-SLAM3 on specified dataset profile.")
    parser.add_argument("--config", type=str, default=None, help="Path to dataset YAML config.")
    parser.add_argument("--output-dir", type=str, default=None, help="Custom output directory.")
    parser.add_argument("--settings", type=str, default=None, help="Custom ORB-SLAM3 settings file.")
    args = parser.parse_args()

    repo_root = get_repo_root()
    cfg_path = Path(args.config) if args.config else (repo_root / "configs" / "datasets" / "tum_rgbd_fr3_walking_xyz.yaml")
    out_dir = Path(args.output_dir) if args.output_dir else None
    settings_path = Path(args.settings) if args.settings else None

    # 1. Run Baseline SLAM
    traj_file, cfg = run_baseline_slam(dataset_config_path=cfg_path, output_dir=out_dir, settings_path=settings_path)
    print(f"Generated trajectory: {traj_file}")

    # 2. Ingest Estimated Trajectory
    est_trajectory = parse_estimated_trajectory(traj_file)
    print(f"Parsed {len(est_trajectory)} estimated camera poses.")

    # 3. Ingest Ground Truth Trajectory
    gt_file = cfg.groundtruth_file or (cfg.sequence_dir / "groundtruth.txt")
    gt_trajectory = parse_tum_ground_truth(gt_file)
    print(f"Parsed {len(gt_trajectory)} reference ground-truth poses.")

    # 4. Evaluate Trajectory Metrics with SE(3) Umeyama Alignment
    result = evaluate_trajectory(
        estimated=est_trajectory,
        ground_truth=gt_trajectory,
        max_dt_sec=cfg.association_tolerance_sec,
        sequence_name=cfg.sequence_name,
    )

    # 5. Export JSON Report
    target_out_dir = traj_file.parent
    report_file = target_out_dir / "baseline_evaluation_metrics.json"
    with open(report_file, "w", encoding="utf-8") as f:
        json.dump(result.to_dict(), f, indent=2)

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
    print(f"Report exported to: {report_file}")
    print("==================================================\n")


if __name__ == "__main__":
    main()
