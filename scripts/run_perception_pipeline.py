"""Run Multi-View Perception and Estimator Evidence Extraction on fr3_walking_xyz.

Ingests synchronized frames and Pass A SLAM poses, performs camera-compensated
3D tracking, classifies observed physical motion, and harvests multi-view composite crops.
"""

from __future__ import annotations

import json
from pathlib import Path

from estimator_aware_vlm.common.paths import get_repo_root
from estimator_aware_vlm.datasets import DatasetConfig, TUMRGBDAdapter, parse_tum_trajectory
from estimator_aware_vlm.perception import SequenceTracker


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description="Run 3D Perception & Tracking Pipeline on specified dataset profile.")
    parser.add_argument("--config", type=str, default=None, help="Path to dataset YAML config.")
    parser.add_argument("--trajectory", type=str, default=None, help="Path to estimated CameraTrajectory.txt.")
    parser.add_argument("--output-dir", type=str, default=None, help="Base output directory.")
    args = parser.parse_args()

    repo_root = get_repo_root()
    from estimator_aware_vlm.datasets import load_dataset_config_from_yaml

    cfg_path = Path(args.config) if args.config else (repo_root / "configs" / "datasets" / "tum_rgbd_fr3_walking_xyz.yaml")
    cfg = load_dataset_config_from_yaml(cfg_path)
    seq_dir = cfg.sequence_dir

    suffix = "" if "freiburg3" in cfg.sequence_name else f"_{cfg.sequence_name}"
    if args.trajectory:
        traj_path = Path(args.trajectory)
    else:
        traj_path = repo_root / "outputs" / f"baseline_slam{suffix}" / "CameraTrajectory.txt"

    if not traj_path.exists():
        alt_traj = traj_path.parent / "KeyFrameTrajectory.txt"
        if alt_traj.exists():
            traj_path = alt_traj
        else:
            raise FileNotFoundError(
                f"Pass A baseline trajectory not found at: {traj_path}. "
                f"Please run scripts/run_baseline_slam.py --config {cfg_path} first."
            )

    # 1. Ingest Dataset
    print(f"Loading sequence: {cfg.sequence_name}...")
    adapter = TUMRGBDAdapter(cfg)
    frames = list(adapter.get_frames())
    print(f"Loaded {len(frames)} synchronized RGB-D frames.")

    from estimator_aware_vlm.datasets import TrajectoryRole
    est_trajectory = parse_tum_trajectory(traj_path, role=TrajectoryRole.ESTIMATED, name=f"pass_a_{cfg.sequence_name}")
    print(f"Loaded {len(est_trajectory)} estimated camera poses.")

    # 3. Track Sequence
    print("Executing 3D Spatio-Temporal Tracking and Motion Estimation...")
    tracker = SequenceTracker(
        intrinsics=adapter.get_camera_intrinsics(),
        min_conf_threshold=0.45,
        association_dist_threshold_m=0.85,
        min_track_observations=5,
    )
    tracks = tracker.track_sequence(frames, est_trajectory, max_dt_sec=cfg.association_tolerance_sec * 1.5, stride=4)
    print(f"Discovered {len(tracks)} consistent multi-frame candidate tracks.")

    # 4. Materialize Multi-View Composite Crops
    crops_dir = (Path(args.output_dir) if args.output_dir else repo_root / "outputs") / f"perception_crops{suffix}"
    print(f"Materializing multi-view composite crops in: {crops_dir}...")
    tracker.materialize_multiview_crops(tracks, crops_dir, max_views=3)

    # 5. Export Summary
    tracks_out_dir = (Path(args.output_dir) if args.output_dir else repo_root / "outputs") / f"perception_tracks{suffix}"
    tracks_out_dir.mkdir(parents=True, exist_ok=True)
    summary_path = tracks_out_dir / "tracks_summary.json"
    
    summary_data = [t.to_dict() for t in tracks]
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary_data, f, indent=2)

    # 6. Export Deterministic Per-Frame Observations for Causal Masking
    per_frame_obs_path = tracks_out_dir / "per_frame_instance_observations.jsonl"
    tracker.export_per_frame_observations(tracks, per_frame_obs_path)
    print(f"Exported deterministic per-frame instance observations to: {per_frame_obs_path}")

    print("\n==================================================================================")
    print(f"       TRACKED OBJECT CANDIDATES & ESTIMATOR EVIDENCE ({cfg.sequence_name})       ")
    print("==================================================================================")
    print(f"{'Track ID':<12} | {'Obs Count':<10} | {'Displacement (m)':<18} | {'Uncertainty (m)':<16} | {'Observed Motion':<16} | {'Crops'}")
    print("-" * 88)
    for t in tracks:
        print(f"{t.track_id:<12} | {t.observation_count:<10} | {t.measured_displacement_m:<18.4f} | {t.displacement_uncertainty_m:<16.4f} | {t.observed_motion.value:<16} | {len(t.crop_paths)}")
    print("==================================================================================")
    print(f"Exported track telemetry to: {summary_path}\n")


if __name__ == "__main__":
    main()
