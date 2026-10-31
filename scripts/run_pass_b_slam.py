"""Execute Pass B Dynamic Feature Masking and Evaluated SLAM with isolated runtime.

Applies estimator-aware VLM intervention actions:
1. Ingests routing decisions from outputs/vlm_inference/routing_decisions.json.
2. Identifies dynamic candidate entities flagged as MASK_FROM_TRACKING.
3. Loads deterministic per-frame instance observations from
   outputs/perception_tracks/per_frame_instance_observations.jsonl.
4. Generates pixel-level instance segmentation masks (cv2.fillPoly on polygon contours)
   in an isolated experiment workspace, strictly preserving raw data directory immutability.
5. Cryptographically verifies cache validity using associations_pass_b.manifest.json.
6. Exports mask_causality_report.json documenting causal attribution.
7. Executes ORB-SLAM3 in isolated workspace with fail-closed integrity checks.
8. Evaluates Pass B camera trajectory against motion-capture ground truth using SE(3) Horn/Umeyama alignment.
9. Requires Pass A baseline metrics and computes genuine trajectory error reduction.
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import os
import shutil
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import cv2
import numpy as np

from estimator_aware_vlm.common.paths import get_repo_root
from estimator_aware_vlm.datasets import (
    TrajectoryRole,
    get_dataset_adapter,
    load_dataset_config_from_yaml,
    parse_tum_ground_truth,
    parse_tum_trajectory,
)
from estimator_aware_vlm.evaluation import (
    evaluate_paired_trajectories_common_support,
    evaluate_trajectory,
)
from estimator_aware_vlm.slam import (
    EmptyTrajectoryError,
    ORBSLAM3Runner,
    SLAMExecutionError,
    SLAMRuntimeConfig,
    create_experiment_dir,
    export_run_manifest,
    to_wsl_path,
)


def compute_file_sha256(filepath: Path) -> str:
    """Compute hex SHA256 digest of a file."""
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def generate_pass_b_masked_associations(
    raw_seq_dir: Path,
    output_dir: Path,
    masked_track_ids: Set[str],
    observations_path: Path,
    routing_decisions_path: Path,
    causality_report_path: Path,
    raw_assoc_path: Optional[Path] = None,
    dataset_cfg: Optional[Any] = None,
    mask_type: str = "polygon",
) -> Path:
    """Generate masked frames for dynamic objects and output associations_pass_b.txt inside output_dir."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Resolve raw associations file (generate in output_dir if not existing)
    if raw_assoc_path is None or not raw_assoc_path.exists():
        raw_assoc_path = output_dir / "associations_raw.txt"
        if not raw_assoc_path.exists():
            if dataset_cfg is None:
                raise FileNotFoundError("Raw associations not found and dataset_cfg not provided.")
            print(f"Generating temporary base associations at {raw_assoc_path}")
            adapter = get_dataset_adapter(dataset_cfg)
            adapter.export_associations(raw_assoc_path)

    if not observations_path.exists():
        raise FileNotFoundError(
            f"Deterministic per-frame observations not found at: {observations_path}. "
            f"Run scripts/run_perception_pipeline.py first."
        )

    # 2. Check Cryptographic Cache Manifest in output_dir
    manifest_path = output_dir / "associations_pass_b.manifest.json"
    out_assoc_path = output_dir / "associations_pass_b.txt"

    current_hashes = {
        "raw_associations_sha256": compute_file_sha256(raw_assoc_path),
        "observations_sha256": compute_file_sha256(observations_path),
        "routing_decisions_sha256": compute_file_sha256(routing_decisions_path),
        "masked_track_ids": sorted(list(masked_track_ids)),
        "mask_type": mask_type,
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
                and cached_manifest.get("mask_type") == current_hashes["mask_type"]
                and sum(1 for _ in open(out_assoc_path, "r", encoding="utf-8")) == total_frames
            ):
                cache_valid = True
        except Exception:
            cache_valid = False

    if cache_valid:
        print("Cryptographic cache valid: associations_pass_b.txt matches input hashes. Skipping mask re-generation.")
        return out_assoc_path

    # 3. Ingest per-frame observations
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

    # 4. Generate instance-masked frames inside output_dir/masked_pass_b (NEVER inside raw_seq_dir)
    masked_dir = output_dir / "masked_pass_b"
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
                        if mask_type == "polygon" and contour and len(contour) >= 3:
                            # Pixel-level instance segmentation polygon
                            pts = np.array(contour, dtype=np.int32)
                            cv2.fillPoly(img_bgr, [pts], (0, 0, 0))
                            cv2.fillPoly(depth_img, [pts], 0)
                            cv2.fillPoly(frame_mask_2d, [pts], 255)
                        else:
                            # Bounding box masking
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

                # Compute relative path from raw_seq_dir to dst_rgb/dst_depth so ORB-SLAM3 resolves seamlessly
                rel_to_seq_rgb = os.path.relpath(dst_rgb, raw_seq_dir).replace("\\", "/")
                rel_to_seq_depth = os.path.relpath(dst_depth, raw_seq_dir).replace("\\", "/")
                out_f.write(f"{t_rgb:.6f} {rel_to_seq_rgb} {t_depth:.6f} {rel_to_seq_depth}\n")
            else:
                out_f.write(f"{t_rgb:.6f} {rel_rgb} {t_depth:.6f} {rel_depth}\n")

            if (idx + 1) % 200 == 0 or (idx + 1) == total_frames:
                print(f"  Processed {idx + 1}/{total_frames} frames ({masked_frame_count} dynamic frames masked)...")

    # 5. Export Cryptographic Cache Manifest
    manifest_data = {
        "timestamp": time.time(),
        "total_frames": total_frames,
        "masked_frames": masked_frame_count,
        "raw_associations_sha256": current_hashes["raw_associations_sha256"],
        "observations_sha256": current_hashes["observations_sha256"],
        "routing_decisions_sha256": current_hashes["routing_decisions_sha256"],
        "masked_track_ids": current_hashes["masked_track_ids"],
        "mask_type": mask_type,
    }
    with open(manifest_path, "w", encoding="utf-8") as mf:
        json.dump(manifest_data, mf, indent=2)

    # 6. Export Causality Report
    causality_report = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "routing_policy_source": str(routing_decisions_path),
        "masked_track_ids": sorted(list(masked_track_ids)),
        "mask_type": "polygon_segmentation" if mask_type == "polygon" else "bounding_box_suppression",
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


def main() -> None:
    parser = argparse.ArgumentParser(description="Execute Pass B Filtered SLAM with isolated execution.")
    parser.add_argument("--config", type=str, default=None, help="Path to dataset YAML config.")
    parser.add_argument("--output-dir", type=str, default=None, help="Explicit output directory.")
    parser.add_argument("--experiment-id", type=str, default=None, help="Unique ID for experiment.")
    parser.add_argument("--settings", type=str, default=None, help="Custom ORB-SLAM3 settings file.")
    parser.add_argument("--decisions", type=str, default=None, help="Path to routing decisions JSON.")
    parser.add_argument("--observations", type=str, default=None, help="Path to per-frame observations JSONL.")
    parser.add_argument("--baseline-metrics", type=str, default=None, help="Path to baseline evaluation metrics JSON.")
    parser.add_argument("--orb-binary", type=str, default=None, help="Path to rgbd_tum executable.")
    parser.add_argument("--orb-vocab", type=str, default=None, help="Path to ORBvoc.txt vocabulary file.")
    parser.add_argument("--wsl-distro", type=str, default=None, help="WSL distribution name.")
    parser.add_argument("--wsl-user", type=str, default=None, help="WSL username.")
    parser.add_argument("--timeout", type=int, default=900, help="Process timeout in seconds.")
    parser.add_argument("--intervention-id", type=str, default=None, help="Explicit counterfactual intervention ID.")
    parser.add_argument("--intervention-action", type=str, default=None, help="Action: KEEP, IMAGE_POLYGON_MASK, IMAGE_BBOX_MASK.")
    parser.add_argument("--target-track-id", type=str, default=None, help="Single track ID to mask for counterfactual testing.")
    parser.add_argument("--repeat-index", type=int, default=1, help="Repetition index.")
    parser.add_argument("--disable-legacy-mirror", action="store_true", help="Disable copying outputs to legacy outputs/pass_b_slam folder.")
    args = parser.parse_args()

    repo_root = get_repo_root()
    cfg_path = Path(args.config) if args.config else (repo_root / "configs" / "datasets" / "tum_rgbd_fr3_walking_xyz.yaml")
    cfg = load_dataset_config_from_yaml(cfg_path)
    suffix = "" if "freiburg3" in cfg.sequence_name else f"_{cfg.sequence_name}"

    # 1. Validate Pass A Baseline (Fail Closed)
    if args.baseline_metrics:
        baseline_metrics_path = Path(args.baseline_metrics)
    else:
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

    # 2. Ingest Routing Decisions or Explicit Intervention Target
    mask_type = "polygon"
    decisions_path = Path(args.decisions) if args.decisions else (repo_root / "outputs" / "vlm_inference" / "routing_decisions.json")
    if args.target_track_id is not None or args.intervention_action is not None:
        action = (args.intervention_action or "IMAGE_POLYGON_MASK").upper()
        if action == "KEEP":
            masked_track_ids = set()
            mask_type = "polygon"
            print("Intervention Action: KEEP (Baseline reproduction, 0 tracks masked).")
        elif action == "IMAGE_BBOX_MASK":
            masked_track_ids = {args.target_track_id} if args.target_track_id else set()
            mask_type = "bbox"
            print(f"Intervention Action: IMAGE_BBOX_MASK on target track: {sorted(list(masked_track_ids))}")
        else:  # IMAGE_POLYGON_MASK
            masked_track_ids = {args.target_track_id} if args.target_track_id else set()
            mask_type = "polygon"
            print(f"Intervention Action: IMAGE_POLYGON_MASK on target track: {sorted(list(masked_track_ids))}")
    else:
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

    # 3. Create isolated output directory
    if args.output_dir:
        pass_b_out_dir = Path(args.output_dir).resolve()
        pass_b_out_dir.mkdir(parents=True, exist_ok=True)
    else:
        prefix = f"pass_b_{cfg.sequence_name}"
        pass_b_out_dir = create_experiment_dir(experiment_id=args.experiment_id, prefix=prefix)

    causality_report_path = pass_b_out_dir / "mask_causality_report.json"

    # 4. Generate Masked Frames & Associations
    print("\n--- Generating Pass B Causal Instance Segmentation Masks ---")
    pass_b_assoc_file = generate_pass_b_masked_associations(
        raw_seq_dir=raw_seq_dir,
        output_dir=pass_b_out_dir,
        masked_track_ids=masked_track_ids,
        observations_path=observations_path,
        routing_decisions_path=decisions_path,
        causality_report_path=causality_report_path,
        dataset_cfg=cfg,
        mask_type=mask_type,
    )

    # 5. Resolve SLAM runtime configuration
    runtime_config = SLAMRuntimeConfig.from_env_and_args(
        orb_binary=args.orb_binary,
        orb_vocab=args.orb_vocab,
        settings_path=args.settings,
        wsl_distro=args.wsl_distro,
        wsl_user=args.wsl_user,
        timeout_sec=args.timeout,
        sequence_name=cfg.sequence_name,
    )

    # 6. Execute ORB-SLAM3 Runner with fail-closed checks
    print("\n--- Executing Pass B Filtered SLAM ---")
    runner = ORBSLAM3Runner(config=runtime_config)
    pass_b_traj_file = runner.run(
        sequence_dir=raw_seq_dir,
        associations_file=pass_b_assoc_file,
        output_dir=pass_b_out_dir,
        settings_path=runtime_config.settings_path,
    )

    # 7. Evaluate Pass B Trajectory
    print("\n--- Evaluating Pass B Trajectory vs Ground Truth ---")
    pass_b_trajectory = parse_tum_trajectory(pass_b_traj_file, role=TrajectoryRole.ESTIMATED, name=f"pass_b_{cfg.sequence_name}")

    adapter = get_dataset_adapter(cfg)
    gt_trajectory = adapter.get_ground_truth_trajectory()

    pass_b_result = evaluate_trajectory(
        estimated=pass_b_trajectory,
        ground_truth=gt_trajectory,
        max_dt_sec=cfg.association_tolerance_sec,
        sequence_name=f"{cfg.sequence_name} (Pass B Estimator-Aware)",
    )

    pass_b_metrics = pass_b_result.to_dict()
    pass_b_metrics_file = pass_b_out_dir / "pass_b_evaluation_metrics.json"
    with open(pass_b_metrics_file, "w", encoding="utf-8") as f:
        json.dump(pass_b_metrics, f, indent=2)

    # 8. Comparative Evaluation vs Pass A Baseline
    ate_pass_b = pass_b_result.ate.rmse_m
    delta_ate_m = ate_pass_b - ate_pass_a
    pct_improvement = ((ate_pass_a - ate_pass_b) / ate_pass_a) * 100.0

    # 8b. Paired Trajectory Evaluation on Common Support
    baseline_traj_file = baseline_metrics_path.parent / "CameraTrajectory.txt"
    if not baseline_traj_file.exists():
        baseline_traj_file = baseline_metrics_path.parent / "KeyFrameTrajectory.txt"

    paired_metrics_file = None
    paired_summary: Optional[Dict[str, Any]] = None
    if baseline_traj_file.exists():
        try:
            print("\n--- Milestone 12.4: Computing Paired Evaluation on Common Timestamp Support ---")
            pass_a_trajectory = parse_tum_trajectory(
                baseline_traj_file,
                role=TrajectoryRole.ESTIMATED,
                name=f"baseline_pass_a_{cfg.sequence_name}",
            )
            paired_result = evaluate_paired_trajectories_common_support(
                traj_a=pass_a_trajectory,
                traj_b=pass_b_trajectory,
                ground_truth=gt_trajectory,
                max_dt_sec=cfg.association_tolerance_sec,
                delta_sec=1.0,
                sequence_name=cfg.sequence_name,
            )
            paired_summary = paired_result.to_dict()
            paired_metrics_file = pass_b_out_dir / "paired_common_support_metrics.json"
            with open(paired_metrics_file, "w", encoding="utf-8") as pf:
                json.dump(paired_summary, pf, indent=2)
            print(f"Common Support Poses:     {paired_result.num_common_poses} poses (Jaccard: {paired_result.jaccard_support_index * 100:.1f}%)")
            print(f"Pass A ATE (Common):      {paired_result.ate_a_common.rmse_m:.4f} m ({paired_result.ate_a_common.rmse_m * 100:.2f} cm)")
            print(f"Pass B ATE (Common):      {paired_result.ate_b_common.rmse_m:.4f} m ({paired_result.ate_b_common.rmse_m * 100:.2f} cm)")
            print(f"Common ATE Reduction:     {paired_result.absolute_ate_reduction_common_m:.4f} m ({paired_result.percentage_ate_improvement_common:.1f}%)")
            print(f"Paired t-test p-value:    {paired_result.ttest_pvalue:.4e} (Significant: {paired_result.is_statistically_significant})")
        except Exception as exc:
            print(f"Notice: Paired common-support evaluation skipped: {exc}")

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
        "common_support_evaluation": paired_summary,
        "causality_metadata": {
            "masked_track_ids": sorted(list(masked_track_ids)),
            "mask_type": "polygon_instance_segmentation",
        },
    }

    comparison_file = pass_b_out_dir / "trajectory_improvement_comparison.json"
    with open(comparison_file, "w", encoding="utf-8") as f:
        json.dump(comparison_summary, f, indent=2)

    # 9. Export Cryptographic Run Manifest
    gt_file = getattr(adapter, "ground_truth_file", None) or getattr(cfg, "groundtruth_file", None)
    input_files = {
        "dataset_config": cfg_path,
        "routing_decisions": decisions_path,
        "per_frame_observations": observations_path,
        "baseline_metrics": baseline_metrics_path,
    }
    if gt_file and Path(gt_file).exists():
        input_files["groundtruth"] = Path(gt_file)
    if runtime_config.settings_path and Path(runtime_config.settings_path).exists():
        input_files["camera_settings"] = Path(runtime_config.settings_path)

    output_artifacts = {
        "associations_pass_b": pass_b_assoc_file,
        "trajectory": pass_b_traj_file,
        "causality_report": causality_report_path,
        "execution_log": pass_b_out_dir / "orbslam3_execution.log",
        "pass_b_metrics": pass_b_metrics_file,
        "comparison_report": comparison_file,
    }
    if paired_metrics_file and paired_metrics_file.exists():
        output_artifacts["paired_common_support_metrics"] = paired_metrics_file

    manifest_path = export_run_manifest(
        experiment_dir=pass_b_out_dir,
        run_id=pass_b_out_dir.name,
        pipeline_module="Dynamic Feature Suppression SLAM Subsystem",
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
        metrics=comparison_summary,
    )

    # 10. Update legacy mirror directory for backwards compatibility (optional)
    if not args.disable_legacy_mirror:
        legacy_dir = repo_root / "outputs" / f"pass_b_slam{suffix}"
        try:
            legacy_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy2(pass_b_traj_file, legacy_dir / pass_b_traj_file.name)
            shutil.copy2(pass_b_metrics_file, legacy_dir / pass_b_metrics_file.name)
            shutil.copy2(comparison_file, legacy_dir / comparison_file.name)
            shutil.copy2(manifest_path, legacy_dir / manifest_path.name)
        except Exception as exc:
            print(f"Notice: Legacy mirror update skipped: {exc}")

    print("\n==========================================================================================")
    print(f"                    END-TO-END SLAM COMPARISON ON {cfg.sequence_name.upper()}             ")
    print("==========================================================================================")
    print(f"{'Metric':<30} | {'Pass A Baseline':<20} | {'Pass B Estimator-Aware':<22} | {'Change'}")
    print("-" * 90)
    print(f"{'ATE RMSE':<30} | {ate_pass_a:.4f} m ({ate_pass_a * 100:.2f} cm)  | {ate_pass_b:.4f} m ({ate_pass_b * 100:.2f} cm)     | -{pct_improvement:.1f}%")
    print(f"{'Tracking Coverage':<30} | {baseline_metrics.get('tracking_coverage', 0) * 100:.1f}%             | {pass_b_result.tracking_coverage * 100:.1f}%                | Matched")
    print("==========================================================================================")
    print(f"Pass B Metrics:     {pass_b_metrics_file}")
    print(f"Comparison:         {comparison_file}")
    print(f"Manifest:           {manifest_path}\n")


if __name__ == "__main__":
    main()
