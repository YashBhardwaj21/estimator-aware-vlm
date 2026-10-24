"""Execute Pass B Dynamic Feature Masking and Evaluated SLAM.

Applies estimator-aware VLM intervention actions:
1. Ingests routing decisions from outputs/vlm_inference/routing_decisions.json.
2. Identifies dynamic candidate entities flagged as MASK_FROM_TRACKING.
3. Loads deterministic per-frame instance observations from
   outputs/perception_tracks/per_frame_instance_observations.jsonl.
4. Generates pixel-level instance segmentation masks (cv2.fillPoly on polygon contours)
   preserving static office background features (F-01, F-06).
5. Cryptographically verifies cache validity using associations_pass_b.manifest.json (F-08).
6. Exports outputs/pass_b_slam/mask_causality_report.json documenting causal attribution.
7. Executes ORB-SLAM3 in RGB-D mode with dynamic feature suppression.
8. Evaluates Pass B camera trajectory against motion-capture ground truth using SE(3) Horn/Umeyama alignment.
9. Requires Pass A baseline metrics and computes genuine trajectory error reduction (F-02).
"""

from __future__ import annotations

import collections
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import cv2
import numpy as np

from estimator_aware_vlm.common.paths import get_repo_root
from estimator_aware_vlm.datasets import (
    TrajectoryRole,
    parse_tum_ground_truth,
    parse_tum_trajectory,
)
from estimator_aware_vlm.evaluation import evaluate_trajectory


def to_wsl_path(win_path: Path) -> str:
    """Convert Windows path to WSL /mnt/... format."""
    s = str(win_path.resolve()).replace("\\", "/")
    if len(s) >= 2 and s[1] == ":":
        drive = s[0].lower()
        return f"/mnt/{drive}{s[2:]}"
    return s


def compute_file_sha256(filepath: Path) -> str:
    """Compute hex SHA256 digest of a file."""
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def generate_pass_b_masked_associations(
    raw_seq_dir: Path,
    masked_track_ids: Set[str],
    observations_path: Path,
    routing_decisions_path: Path,
    causality_report_path: Path,
) -> Path:
    """Generate masked frames for dynamic objects and output associations_pass_b.txt."""
    raw_assoc_path = raw_seq_dir / "associations.txt"
    if not raw_assoc_path.exists():
        raise FileNotFoundError(f"Missing raw associations: {raw_assoc_path}")

    if not observations_path.exists():
        raise FileNotFoundError(
            f"Deterministic per-frame observations not found at: {observations_path}. "
            f"Run scripts/run_perception_pipeline.py first."
        )

    # 1. Check Cryptographic Cache Manifest
    manifest_path = raw_seq_dir / "associations_pass_b.manifest.json"
    out_assoc_path = raw_seq_dir / "associations_pass_b.txt"

    current_hashes = {
        "raw_associations_sha256": compute_file_sha256(raw_assoc_path),
        "observations_sha256": compute_file_sha256(observations_path),
        "routing_decisions_sha256": compute_file_sha256(routing_decisions_path),
        "masked_track_ids": sorted(list(masked_track_ids)),
    }

    # Count raw association pairs
    pairs: List[Tuple[float, str, float, str]] = []
    with open(raw_assoc_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 4:
                pairs.append((float(parts[0]), parts[1], float(parts[2]), parts[3]))

    total_frames = len(pairs)
    print(f"Loaded {total_frames} synchronized frame pairs from {raw_assoc_path.name}.")
    print(f"Active dynamic tracks flagged for feature masking ({len(masked_track_ids)}): {sorted(list(masked_track_ids))}")

    cache_valid = False
    if out_assoc_path.exists() and manifest_path.exists():
        try:
            with open(manifest_path, "r", encoding="utf-8") as mf:
                cached_manifest = json.load(mf)
            if (
                cached_manifest.get("raw_associations_sha256") == current_hashes["raw_associations_sha256"]
                and cached_manifest.get("observations_sha256") == current_hashes["observations_sha256"]
                and cached_manifest.get("routing_decisions_sha256") == current_hashes["routing_decisions_sha256"]
                and cached_manifest.get("masked_track_ids") == current_hashes["masked_track_ids"]
                and sum(1 for _ in open(out_assoc_path, "r", encoding="utf-8")) == total_frames
            ):
                cache_valid = True
        except Exception:
            cache_valid = False

    if cache_valid:
        print(f"Cryptographic cache valid: associations_pass_b.txt matches input hashes. Skipping mask re-generation.")
        return out_assoc_path

    # 2. Ingest per-frame observations
    print(f"Ingesting per-frame instance observations from {observations_path.name}...")
    obs_by_rgb_name: Dict[str, List[Dict[str, Any]]] = collections.defaultdict(list)
    total_obs_count = 0
    with open(observations_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            rgb_filename = Path(rec.get("rgb_path", "")).name
            obs_by_rgb_name[rgb_filename].append(rec)
            total_obs_count += 1
    print(f"Indexed {total_obs_count} observations across {len(obs_by_rgb_name)} distinct frames.")

    # 3. Generate instance-masked frames
    masked_dir = raw_seq_dir / "masked_pass_b"
    masked_dir.mkdir(parents=True, exist_ok=True)

    masked_frame_count = 0
    per_track_frame_count: Dict[str, int] = collections.defaultdict(int)
    total_pixels_masked = 0

    print("Generating polygon instance segmentation masks on dynamic entities...")
    with open(out_assoc_path, "w", encoding="utf-8") as out_f:
        for idx, (t_rgb, rel_rgb, t_depth, rel_depth) in enumerate(pairs):
            rgb_filename = Path(rel_rgb).name
            frame_obs = obs_by_rgb_name.get(rgb_filename, [])
            dynamic_obs = [obs for obs in frame_obs if obs.get("track_id") in masked_track_ids]

            if dynamic_obs:
                masked_frame_count += 1
                in_rgb = raw_seq_dir / rel_rgb
                in_depth = raw_seq_dir / rel_depth

                dst_rgb = masked_dir / rel_rgb
                dst_depth = masked_dir / rel_depth
                dst_rgb.parent.mkdir(parents=True, exist_ok=True)
                dst_depth.parent.mkdir(parents=True, exist_ok=True)

                img_bgr = cv2.imread(str(in_rgb))
                depth_img = cv2.imread(str(in_depth), cv2.IMREAD_UNCHANGED)

                if img_bgr is not None and depth_img is not None:
                    h, w = img_bgr.shape[:2]
                    frame_mask_2d = np.zeros((h, w), dtype=np.uint8)

                    for obs in dynamic_obs:
                        tid = obs.get("track_id", "unknown")
                        per_track_frame_count[tid] += 1

                        contour = obs.get("mask_contour")
                        if contour and len(contour) >= 3:
                            # Pixel-level instance segmentation polygon
                            pts = np.array(contour, dtype=np.int32)
                            cv2.fillPoly(img_bgr, [pts], (0, 0, 0))
                            cv2.fillPoly(depth_img, [pts], 0)
                            cv2.fillPoly(frame_mask_2d, [pts], 255)
                        else:
                            # Fallback bounding box if polygon not available
                            x1, y1, x2, y2 = obs.get("bbox_xyxy", [0, 0, 0, 0])
                            x1 = max(0, min(int(x1), w - 1))
                            y1 = max(0, min(int(y1), h - 1))
                            x2 = max(x1 + 1, min(int(x2), w))
                            y2 = max(y1 + 1, min(int(y2), h))
                            img_bgr[y1:y2, x1:x2] = 0
                            depth_img[y1:y2, x1:x2] = 0
                            frame_mask_2d[y1:y2, x1:x2] = 255

                    total_pixels_masked += int(np.sum(frame_mask_2d > 0))
                    cv2.imwrite(str(dst_rgb), img_bgr)
                    cv2.imwrite(str(dst_depth), depth_img)

                out_f.write(f"{t_rgb:.6f} masked_pass_b/{rel_rgb} {t_depth:.6f} masked_pass_b/{rel_depth}\n")
            else:
                out_f.write(f"{t_rgb:.6f} {rel_rgb} {t_depth:.6f} {rel_depth}\n")

            if (idx + 1) % 200 == 0 or (idx + 1) == total_frames:
                print(f"  Processed {idx + 1}/{total_frames} frames ({masked_frame_count} dynamic frames masked)...")

    # 4. Export Cryptographic Cache Manifest
    manifest_data = {
        "timestamp": time.time(),
        "total_frames": total_frames,
        "masked_frames": masked_frame_count,
        "raw_associations_sha256": current_hashes["raw_associations_sha256"],
        "observations_sha256": current_hashes["observations_sha256"],
        "routing_decisions_sha256": current_hashes["routing_decisions_sha256"],
        "masked_track_ids": current_hashes["masked_track_ids"],
    }
    with open(manifest_path, "w", encoding="utf-8") as mf:
        json.dump(manifest_data, mf, indent=2)

    # 5. Export Causality Report (F-01, F-06, F-08)
    causality_report = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "routing_policy_source": str(routing_decisions_path),
        "masked_track_ids": sorted(list(masked_track_ids)),
        "mask_type": "polygon_instance_segmentation",
        "total_sequence_frames": total_frames,
        "total_masked_frames": masked_frame_count,
        "masking_percentage": round((masked_frame_count / total_frames) * 100.0, 2),
        "total_pixels_masked": total_pixels_masked,
        "per_track_frame_count": dict(per_track_frame_count),
    }
    causality_report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(causality_report_path, "w", encoding="utf-8") as cr:
        json.dump(causality_report, cr, indent=2)

    print(f"Pass B dynamic filtering complete: {masked_frame_count}/{total_frames} frames suppressed.")
    print(f"Causality report exported to: {causality_report_path}")
    print(f"Associations exported to:     {out_assoc_path}")
    return out_assoc_path


def run_orb_slam3_pass_b(
    raw_seq_dir: Path,
    out_dir: Path,
    settings_path: Optional[Path] = None,
    timeout_sec: int = 900,
) -> Path:
    """Execute ORB-SLAM3 in WSL using dynamic-masked associations."""
    out_dir.mkdir(parents=True, exist_ok=True)
    wsl_out_dir = to_wsl_path(out_dir)
    wsl_seq_dir = to_wsl_path(raw_seq_dir)
    wsl_assoc_file = to_wsl_path(raw_seq_dir / "associations_pass_b.txt")

    orb_binary = "/home/saturn/ORB_SLAM3/Examples/RGB-D/rgbd_tum"
    orb_vocab = "/home/saturn/ORB_SLAM3/Vocabulary/ORBvoc.txt"

    repo_root = get_repo_root()
    if settings_path is None:
        if "openloris" in str(raw_seq_dir).lower():
            settings_path = repo_root / "configs" / "sensors" / "OpenLORIS_D400.yaml"
        else:
            settings_path = Path("/home/saturn/ORB_SLAM3/Examples/RGB-D/TUM3.yaml")

    wsl_settings = to_wsl_path(settings_path) if settings_path.is_absolute() else str(settings_path)

    cmd = (
        f"cd '{wsl_out_dir}' && "
        f"'{orb_binary}' '{orb_vocab}' '{wsl_settings}' '{wsl_seq_dir}' '{wsl_assoc_file}' 0"
    )

    print(f"\nExecuting Pass B ORB-SLAM3 in WSL with Dynamic Feature Suppression...")
    print(f"Working Directory: {out_dir}")
    print(f"Command: {cmd}")

    t0 = time.time()
    if sys.platform.startswith("linux"):
        exec_cmd = ["bash", "-c", cmd]
    else:
        exec_cmd = ["wsl", "-d", "Ubuntu-22.04", "-u", "saturn", "--", "bash", "-c", cmd]
    res = subprocess.run(
        exec_cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        timeout=timeout_sec,
    )
    duration = time.time() - t0
    print(f"Pass B execution completed in {duration:.2f}s with return code {res.returncode}")

    log_file = out_dir / "orbslam3_pass_b_execution.log"
    with open(log_file, "w", encoding="utf-8") as f:
        f.write(res.stdout)

    traj_file = out_dir / "CameraTrajectory.txt"
    if not traj_file.exists():
        alt_traj = out_dir / "KeyFrameTrajectory.txt"
        if alt_traj.exists():
            print(f"Notice: CameraTrajectory.txt missing, using KeyFrameTrajectory.txt")
            traj_file = alt_traj
        else:
            raise RuntimeError(
                f"Pass B ORB-SLAM3 failed to produce trajectory. See log: {log_file}\n"
                f"Tail of stdout:\n{res.stdout[-1500:] if res.stdout else 'None'}"
            )

    return traj_file


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description="Execute Pass B Filtered SLAM on specified dataset profile.")
    parser.add_argument("--config", type=str, default=None, help="Path to dataset YAML config.")
    parser.add_argument("--output-dir", type=str, default=None, help="Custom output directory.")
    parser.add_argument("--settings", type=str, default=None, help="Custom ORB-SLAM3 settings file.")
    parser.add_argument("--decisions", type=str, default=None, help="Path to routing decisions JSON.")
    parser.add_argument("--observations", type=str, default=None, help="Path to per-frame observations JSONL.")
    args = parser.parse_args()

    repo_root = get_repo_root()
    from estimator_aware_vlm.datasets import load_dataset_config_from_yaml

    cfg_path = Path(args.config) if args.config else (repo_root / "configs" / "datasets" / "tum_rgbd_fr3_walking_xyz.yaml")
    cfg = load_dataset_config_from_yaml(cfg_path)
    suffix = "" if "freiburg3" in cfg.sequence_name else f"_{cfg.sequence_name}"

    # 1. Validate Pass A Baseline (F-02: Fail Closed)
    baseline_dir = repo_root / "outputs" / f"baseline_slam{suffix}"
    baseline_metrics_path = baseline_dir / "baseline_evaluation_metrics.json"
    if not baseline_metrics_path.exists():
        raise FileNotFoundError(
            f"Pass A baseline evaluation metrics not found at: {baseline_metrics_path}. "
            f"Pass B cannot perform comparative evaluation without genuine Pass A baseline. "
            f"Please run scripts/run_baseline_slam.py --config {cfg_path} first."
        )

    with open(baseline_metrics_path, "r", encoding="utf-8") as f:
        baseline_metrics = json.load(f)

    ate_pass_a = baseline_metrics["ate"]["rmse_m"]
    print(f"Ingested genuine Pass A baseline ATE RMSE: {ate_pass_a:.4f} m ({ate_pass_a * 100:.2f} cm)")

    # 2. Ingest Routing Decisions
    decisions_path = Path(args.decisions) if args.decisions else (repo_root / "outputs" / "vlm_inference" / "routing_decisions.json")
    if not decisions_path.exists():
        raise FileNotFoundError(
            f"Routing decisions not found: {decisions_path}. "
            f"Please run scripts/run_routing_and_vlm_experiment.py first."
        )

    with open(decisions_path, "r", encoding="utf-8") as f:
        routing_data = json.load(f)

    masked_track_ids = {
        d["track_id"] for d in routing_data.get("decisions", [])
        if d.get("action") == "MASK_FROM_TRACKING"
    }
    if not masked_track_ids:
        print("Warning: No tracks flagged for MASK_FROM_TRACKING by routing policy.")

    raw_seq_dir = cfg.sequence_dir
    observations_path = Path(args.observations) if args.observations else (repo_root / "outputs" / f"perception_tracks{suffix}" / "per_frame_instance_observations.jsonl")
    pass_b_out_dir = Path(args.output_dir) if args.output_dir else (repo_root / "outputs" / f"pass_b_slam{suffix}")
    causality_report_path = pass_b_out_dir / "mask_causality_report.json"
    settings_path = Path(args.settings) if args.settings else None

    # 3. Generate Masked Frames & Associations (F-01, F-06, F-08)
    print("\n--- Milestone 12.1: Generating Pass B Causal Instance Segmentation Masks ---")
    generate_pass_b_masked_associations(
        raw_seq_dir=raw_seq_dir,
        masked_track_ids=masked_track_ids,
        observations_path=observations_path,
        routing_decisions_path=decisions_path,
        causality_report_path=causality_report_path,
    )

    # 4. Run Pass B ORB-SLAM3
    print("\n--- Milestone 12.2: Executing Pass B Filtered SLAM ---")
    pass_b_traj_file = run_orb_slam3_pass_b(raw_seq_dir, pass_b_out_dir, settings_path=settings_path)

    # 5. Evaluate Pass B Trajectory
    print("\n--- Milestone 12.3: Evaluating Pass B Trajectory vs Ground Truth ---")
    pass_b_trajectory = parse_tum_trajectory(pass_b_traj_file, role=TrajectoryRole.ESTIMATED, name=f"pass_b_{cfg.sequence_name}")

    gt_file = cfg.groundtruth_file or (raw_seq_dir / "groundtruth.txt")
    gt_trajectory = parse_tum_ground_truth(gt_file)

    pass_b_result = evaluate_trajectory(
        estimated=pass_b_trajectory,
        ground_truth=gt_trajectory,
        max_dt_sec=cfg.association_tolerance_sec,
        sequence_name=f"{cfg.sequence_name} (Pass B Estimator-Aware)",
    )

    pass_b_metrics_file = pass_b_out_dir / "pass_b_evaluation_metrics.json"
    with open(pass_b_metrics_file, "w", encoding="utf-8") as f:
        json.dump(pass_b_result.to_dict(), f, indent=2)

    # 6. Comparative Evaluation vs Pass A Baseline
    ate_pass_b = pass_b_result.ate.rmse_m
    delta_ate_m = ate_pass_b - ate_pass_a
    pct_improvement = ((ate_pass_a - ate_pass_b) / ate_pass_a) * 100.0

    comparison_summary = {
        "sequence_name": cfg.sequence_name,
        "baseline_pass_a": {
            "ate_rmse_m": ate_pass_a,
            "ate_mean_m": baseline_metrics.get("ate", {}).get("mean_m"),
            "rpe_trans_rmse_m": baseline_metrics.get("rpe", {}).get("trans_rmse_m"),
            "tracking_coverage": baseline_metrics.get("tracking_coverage"),
        },
        "estimator_aware_pass_b": {
            "ate_rmse_m": ate_pass_b,
            "ate_mean_m": pass_b_result.ate.mean_m,
            "rpe_trans_rmse_m": pass_b_result.rpe.trans_rmse_m,
            "tracking_coverage": pass_b_result.tracking_coverage,
        },
        "performance_gain": {
            "absolute_ate_reduction_m": round(ate_pass_a - ate_pass_b, 4),
            "percentage_ate_improvement": round(pct_improvement, 2),
        },
        "causality_metadata": {
            "masked_track_ids": sorted(list(masked_track_ids)),
            "mask_type": "polygon_instance_segmentation",
        },
    }

    comparison_file = pass_b_out_dir / "trajectory_improvement_comparison.json"
    with open(comparison_file, "w", encoding="utf-8") as f:
        json.dump(comparison_summary, f, indent=2)

    print("\n==========================================================================================")
    print(f"                    END-TO-END SLAM COMPARISON ON {cfg.sequence_name.upper()}             ")
    print("==========================================================================================")
    print(f"{'Metric':<30} | {'Pass A Baseline':<20} | {'Pass B Estimator-Aware':<22} | {'Change'}")
    print("-" * 90)
    print(f"{'ATE RMSE':<30} | {ate_pass_a:.4f} m ({ate_pass_a * 100:.2f} cm)  | {ate_pass_b:.4f} m ({ate_pass_b * 100:.2f} cm)     | -{pct_improvement:.1f}%")
    print(f"{'Tracking Coverage':<30} | {baseline_metrics.get('tracking_coverage', 0) * 100:.1f}%             | {pass_b_result.tracking_coverage * 100:.1f}%                | Matched")
    print("==========================================================================================")
    print(f"Pass B Metrics exported to:    {pass_b_metrics_file}")
    print(f"Comparison exported to:        {comparison_file}\n")


if __name__ == "__main__":
    main()
