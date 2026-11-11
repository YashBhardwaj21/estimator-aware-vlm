#!/usr/bin/env python3
"""Evaluate baseline ORB-SLAM3 trajectory and mapping quality across all sequences.

Generates:
  1. results/baselines/baseline_comparison.csv
  2. results/baselines/baseline_metrics.json
  3. results/baselines/plots/*.png

Separates Ground-Truth supported evaluations from GT-free RealSense D455 sessions.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict
import json
import math
from pathlib import Path
import sys
from typing import Any, Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

repo_root = Path(__file__).resolve().parent.parent
src_dir = repo_root / "src"
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

from estimator_aware_vlm.common.paths import get_repo_root
from estimator_aware_vlm.datasets.base import (
    Trajectory,
    TrajectoryPose,
    TrajectoryRole,
)
from estimator_aware_vlm.datasets.factory import get_dataset_adapter
from estimator_aware_vlm.datasets.registry import list_sequences
from estimator_aware_vlm.evaluation import (
    align_trajectories_se3,
    compute_ate,
    compute_frame_tracking_rate,
    compute_map_point_stats,
    compute_rpe,
    compute_trajectory_continuity,
    evaluate_trajectory,
    load_cross_session_anchors,
    match_trajectories_by_timestamp,
    parse_native_execution_log,
)
from estimator_aware_vlm.evaluation.baseline_evaluator import parse_estimated_trajectory


def parse_input_timestamps_from_associations(assoc_path: Path) -> List[float]:
    """Parse input RGB timestamps from an associations file."""
    if not assoc_path.exists():
        return []
    timestamps: List[float] = []
    with open(assoc_path, "r", encoding="utf-8") as f:
        for line in f:
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            parts = stripped.split()
            if len(parts) >= 2:
                try:
                    timestamps.append(float(parts[0]))
                except ValueError:
                    continue
    return sorted(timestamps)


def plot_ground_truth_trajectory(
    seq_name: str,
    aligned_est_pos: np.ndarray,
    gt_pos: np.ndarray,
    errors_m: np.ndarray,
    ate_rmse: float,
    output_png: Path,
) -> None:
    """Generate high-resolution top-down and 3D trajectory plot for GT-supported sequence."""
    output_png.parent.mkdir(parents=True, exist_ok=True)

    fig, (ax_top, ax_err) = plt.subplots(1, 2, figsize=(14, 6), dpi=200, gridspec_kw={"width_ratios": [2.5, 1]})

    # Determine dominant horizontal axes (for TUM/Bonn camera frame, X=right, Z=forward)
    # Check if trajectory is TUM/Bonn camera frame or robot frame
    x_gt, y_gt, z_gt = gt_pos[:, 0], gt_pos[:, 1], gt_pos[:, 2]
    x_est, y_est, z_est = aligned_est_pos[:, 0], aligned_est_pos[:, 1], aligned_est_pos[:, 2]

    # Use X vs Z as horizontal top-down plane (or X vs Y if Z variance is lower)
    var_y = np.var(z_gt)
    var_z = np.var(y_gt)
    h_label_x = "X Position (m)"
    if var_y >= var_z:
        h_label_y = "Z Position (m)"
        plot_gt_h = z_gt
        plot_est_h = z_est
    else:
        h_label_y = "Y Position (m)"
        plot_gt_h = y_gt
        plot_est_h = y_est

    ax_top.plot(x_gt, plot_gt_h, color="#2C3E50", linestyle="--", linewidth=1.5, label="Ground Truth (MoCap / LiDAR)", alpha=0.8)
    scatter = ax_top.scatter(x_est, plot_est_h, c=errors_m, cmap="plasma", s=12, label="Estimated Trajectory (ORB-SLAM3)", zorder=3)
    ax_top.scatter(x_est[0], plot_est_h[0], color="#27AE60", s=70, marker="o", edgecolors="black", label="Start", zorder=4)
    ax_top.scatter(x_est[-1], plot_est_h[-1], color="#C0392B", s=80, marker="X", edgecolors="black", label="End", zorder=4)

    ax_top.set_title(f"{seq_name}\nTop-Down Camera Trajectory (SE(3) Aligned, ATE RMSE: {ate_rmse:.4f} m)", fontsize=11, fontweight="bold")
    ax_top.set_xlabel(h_label_x, fontsize=10)
    ax_top.set_ylabel(h_label_y, fontsize=10)
    ax_top.grid(True, linestyle=":", alpha=0.6)
    ax_top.axis("equal")
    ax_top.legend(loc="upper right", fontsize=8)

    cbar = plt.colorbar(scatter, ax=ax_top, pad=0.02)
    cbar.set_label("ATE Position Residual (m)", fontsize=9)

    # Error distribution histogram
    ax_err.hist(errors_m, bins=25, color="#2980B9", edgecolor="black", alpha=0.75, density=True)
    ax_err.axvline(float(np.mean(errors_m)), color="#E67E22", linestyle="-", linewidth=1.5, label=f"Mean: {np.mean(errors_m):.3f} m")
    ax_err.axvline(float(np.median(errors_m)), color="#27AE60", linestyle="--", linewidth=1.5, label=f"Median: {np.median(errors_m):.3f} m")
    ax_err.axvline(ate_rmse, color="#C0392B", linestyle="-.", linewidth=1.5, label=f"RMSE: {ate_rmse:.3f} m")
    ax_err.set_title("ATE Error Distribution", fontsize=11, fontweight="bold")
    ax_err.set_xlabel("Absolute Error (m)", fontsize=10)
    ax_err.set_ylabel("Probability Density", fontsize=10)
    ax_err.grid(True, linestyle=":", alpha=0.6)
    ax_err.legend(loc="upper right", fontsize=8)

    plt.tight_layout()
    plt.savefig(output_png)
    plt.close()


def plot_gt_benchmark_panel(
    results_list: List[Dict[str, Any]],
    output_png: Path,
) -> None:
    """Generate combined 6-panel summary figure for all GT benchmark datasets."""
    output_png.parent.mkdir(parents=True, exist_ok=True)

    gt_results = [r for r in results_list if r["ground_truth_available"]]
    if len(gt_results) != 6:
        return

    fig, axes = plt.subplots(2, 3, figsize=(18, 11), dpi=200)
    axes = axes.flatten()

    for idx, res in enumerate(gt_results):
        ax = axes[idx]
        seq_name = res["sequence_name"]
        aligned_est = res["_aligned_est_pos"]
        gt_pos = res["_gt_pos"]
        errors_m = res["_errors_m"]
        ate_rmse = res["ate_rmse_m"]

        ax.plot(gt_pos[:, 0], gt_pos[:, 2], color="#7F8C8D", linestyle="--", linewidth=1.2, label="Ground Truth", alpha=0.8)
        sc = ax.scatter(aligned_est[:, 0], aligned_est[:, 2], c=errors_m, cmap="plasma", s=8, zorder=3)
        ax.scatter(aligned_est[0, 0], aligned_est[0, 2], color="#27AE60", s=45, marker="o", edgecolors="black", label="Start", zorder=4)
        ax.scatter(aligned_est[-1, 0], aligned_est[-1, 2], color="#C0392B", s=55, marker="X", edgecolors="black", label="End", zorder=4)

        ax.set_title(f"{seq_name}\nATE RMSE: {ate_rmse:.4f} m | Trk: {res['frame_tracking_rate_pct']:.1f}%", fontsize=9, fontweight="bold")
        ax.set_xlabel("X (m)", fontsize=8)
        ax.set_ylabel("Z (m)", fontsize=8)
        ax.grid(True, linestyle=":", alpha=0.6)
        ax.axis("equal")
        if idx == 0:
            ax.legend(loc="upper right", fontsize=7)

    plt.tight_layout()
    plt.savefig(output_png)
    plt.close()


def plot_d455_cross_session_alignment(
    session1_traj: Trajectory,
    session2_traj: Trajectory,
    anchor_stats: CrossSessionAnchorStats,
    output_png: Path,
) -> None:
    """Generate bird's-eye view cross-session desk trajectory alignment with anchor residuals."""
    output_png.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 8), dpi=200)

    # Camera top-down trajectories (X vs Z)
    pos1 = session1_traj.positions
    pos2 = session2_traj.positions

    # Plot Session 1 raw trajectory
    ax.plot(pos1[:, 0], pos1[:, 2], color="#2980B9", linewidth=1.5, label="Session 1 Trajectory (Desk Rec 01)", alpha=0.85)
    ax.scatter(pos1[0, 0], pos1[0, 2], color="#27AE60", s=50, marker="o", edgecolors="black", label="Session 1 Start", zorder=3)
    ax.scatter(pos1[-1, 0], pos1[-1, 2], color="#2980B9", s=60, marker="X", edgecolors="black", label="Session 1 End", zorder=3)

    # Apply authentic SE(2) rotation and translation to Session 2 horizontal coordinates
    # R_12 is 2x2 rotation, translation is 2D
    rad = math.radians(anchor_stats.rotation_deg)
    cos_r = math.cos(rad)
    sin_r = math.sin(rad)
    R_mat = np.array([[cos_r, -sin_r], [sin_r, cos_r]], dtype=np.float64)
    t_vec = np.array(anchor_stats.translation_m, dtype=np.float64)

    # Transform Session 2 horizontal coordinates (X, Z) into Session 1 reference frame
    s2_horiz = pos2[:, [0, 2]]
    s2_aligned = (R_mat @ s2_horiz.T).T + t_vec

    ax.plot(s2_aligned[:, 0], s2_aligned[:, 1], color="#E67E22", linewidth=1.5, linestyle="-.", label="Session 2 Aligned Trajectory (Desk Rec 02)", alpha=0.85)
    ax.scatter(s2_aligned[0, 0], s2_aligned[0, 1], color="#F39C12", s=50, marker="o", edgecolors="black", label="Session 2 Start", zorder=3)
    ax.scatter(s2_aligned[-1, 0], s2_aligned[-1, 1], color="#C0392B", s=60, marker="X", edgecolors="black", label="Session 2 End", zorder=3)

    ax.set_title(
        f"RealSense D455 Cross-Session Alignment and Stationary Anchor Verification\n"
        f"SE(2) Rotation: {anchor_stats.rotation_deg:.3f} deg | Translation: [{anchor_stats.translation_m[0]:.3f}, {anchor_stats.translation_m[1]:.3f}] m\n"
        f"Stationary Anchors: {anchor_stats.robust_inliers}/{anchor_stats.total_anchors} Inliers | Inlier RMS Residual: {anchor_stats.inlier_rms_m:.4f} m",
        fontsize=10,
        fontweight="bold",
    )
    ax.set_xlabel("Reference Frame X (m)", fontsize=9)
    ax.set_ylabel("Reference Frame Z (m)", fontsize=9)
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.axis("equal")
    ax.legend(loc="lower right", fontsize=8)

    plt.tight_layout()
    plt.savefig(output_png)
    plt.close()


def evaluate_all_baselines() -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Execute complete evaluation across all eight sequences."""
    repo_root = get_repo_root()
    all_entries = list_sequences()
    baselines_dir = repo_root / "results" / "baselines"
    plots_dir = baselines_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    registration_audit_path = repo_root / "outputs" / "reports" / "registration_audit.json"

    anchor_stats = load_cross_session_anchors(registration_audit_path)

    all_results: List[Dict[str, Any]] = []
    json_ground_truth_evaluated: Dict[str, Any] = {}
    json_ground_truth_free: Dict[str, Any] = {}

    session_trajectories: Dict[str, Trajectory] = {}

    for entry in all_entries:
        seq_name = entry.sequence_name
        seq_dir = baselines_dir / seq_name

        if not seq_dir.exists():
            continue

        traj_path = seq_dir / "CameraTrajectory.txt"
        ply_path = seq_dir / "MapPoints.ply"
        log_path = seq_dir / "orbslam3_execution.log"
        assoc_path = seq_dir / "associations.txt"

        if not traj_path.exists():
            continue

        est_traj = parse_estimated_trajectory(traj_path)
        continuity = compute_trajectory_continuity(est_traj)
        map_stats = compute_map_point_stats(ply_path, continuity.path_length_m)
        log_stats = parse_native_execution_log(log_path)

        input_timestamps = parse_input_timestamps_from_associations(assoc_path)
        tracking_rate, matched_inputs, total_inputs = compute_frame_tracking_rate(
            est_traj.timestamps, input_timestamps
        )

        dataset_cfg = entry.to_dataset_config(repo_root)
        adapter = get_dataset_adapter(dataset_cfg)

        res_row: Dict[str, Any] = {
            "sequence_name": seq_name,
            "dataset_type": str(entry.dataset_type),
            "ground_truth_available": entry.has_ground_truth,
            "estimated_poses": len(est_traj),
            "trajectory_path_length_m": round(continuity.path_length_m, 2),
            "median_dt_s": round(continuity.median_dt_sec, 4),
            "max_tracking_gap_s": round(continuity.max_gap_sec, 4),
            "gaps_exceeding_100ms": continuity.gaps_over_100ms_count,
            "longest_continuous_run": continuity.longest_continuous_run_poses,
            "frame_tracking_rate_pct": round(tracking_rate * 100.0, 2),
            "matched_input_frames": matched_inputs,
            "total_input_frames": total_inputs,
            "map_points_count": map_stats.vertex_count,
            "map_point_linear_density_pts_per_m": round(map_stats.linear_density_pts_per_m, 1),
            "map_robust_volume_m3": round(map_stats.robust_volume_m3, 2),
            "native_tracking_failures": log_stats.tracking_failure_events,
            "native_relocalizations": log_stats.relocalization_events,
            "native_maps_created": log_stats.maps_created_count,
            "native_merges": log_stats.map_merges_detected,
            "native_loops": log_stats.loop_closures_detected,
        }

        if entry.has_ground_truth and adapter.has_ground_truth():
            gt_traj = adapter.get_ground_truth_trajectory()
            eval_res = evaluate_trajectory(
                estimated=est_traj,
                ground_truth=gt_traj,
                sequence_name=seq_name,
                input_frame_timestamps=input_timestamps,
            )

            matched_est, matched_gt = match_trajectories_by_timestamp(est_traj, gt_traj)
            est_pts = np.array([p.position for p in matched_est], dtype=np.float64)
            gt_pts = np.array([p.position for p in matched_gt], dtype=np.float64)
            aligned_est, _, _ = align_trajectories_se3(est_pts, gt_pts)
            errors_m = np.linalg.norm(aligned_est - gt_pts, axis=1)

            res_row["ground_truth_poses"] = len(gt_traj)
            res_row["matched_poses"] = eval_res.num_matched_poses
            res_row["ground_truth_sample_coverage_pct"] = round(eval_res.ground_truth_sample_coverage * 100.0, 2)
            res_row["estimated_pose_match_rate_pct"] = round(eval_res.estimated_pose_match_rate * 100.0, 2)
            res_row["ate_rmse_m"] = round(eval_res.ate.rmse_m, 4)
            res_row["ate_mean_m"] = round(eval_res.ate.mean_m, 4)
            res_row["ate_median_m"] = round(eval_res.ate.median_m, 4)
            res_row["ate_std_m"] = round(eval_res.ate.std_m, 4)
            res_row["rpe_trans_rmse_m"] = round(eval_res.rpe.trans_rmse_m, 4)
            res_row["rpe_rot_rmse_deg"] = round(eval_res.rpe.rot_rmse_deg, 2)
            if eval_res.rpe_fixed_time:
                res_row["rpe_1s_trans_rmse_m"] = round(eval_res.rpe_fixed_time.trans_rmse_m, 4)
                res_row["rpe_1s_rot_rmse_deg"] = round(eval_res.rpe_fixed_time.rot_rmse_deg, 2)
            else:
                res_row["rpe_1s_trans_rmse_m"] = "N/A"
                res_row["rpe_1s_rot_rmse_deg"] = "N/A"

            res_row["_aligned_est_pos"] = aligned_est
            res_row["_gt_pos"] = gt_pts
            res_row["_errors_m"] = errors_m

            # Plot individual sequence
            plot_ground_truth_trajectory(
                seq_name=seq_name,
                aligned_est_pos=aligned_est,
                gt_pos=gt_pts,
                errors_m=errors_m,
                ate_rmse=eval_res.ate.rmse_m,
                output_png=plots_dir / f"{seq_name}_trajectory.png",
            )

            json_ground_truth_evaluated[seq_name] = {
                "sequence_name": seq_name,
                "dataset_type": str(entry.dataset_type),
                "trajectory_poses": len(est_traj),
                "ground_truth_poses": len(gt_traj),
                "matched_poses": eval_res.num_matched_poses,
                "frame_tracking_rate": tracking_rate,
                "ground_truth_sample_coverage": eval_res.ground_truth_sample_coverage,
                "estimated_pose_match_rate": eval_res.estimated_pose_match_rate,
                "ate": asdict(eval_res.ate),
                "rpe_frame_delta": asdict(eval_res.rpe),
                "rpe_fixed_time_1s": asdict(eval_res.rpe_fixed_time) if eval_res.rpe_fixed_time else None,
                "continuity": continuity.to_dict(),
                "map_quality": map_stats.to_dict(),
                "native_log": log_stats.to_dict(),
            }
        else:
            session_trajectories[seq_name] = est_traj

            res_row["ground_truth_poses"] = "N/A"
            res_row["matched_poses"] = "N/A"
            res_row["ground_truth_sample_coverage_pct"] = "N/A"
            res_row["estimated_pose_match_rate_pct"] = "N/A"
            res_row["ate_rmse_m"] = "N/A"
            res_row["ate_mean_m"] = "N/A"
            res_row["ate_median_m"] = "N/A"
            res_row["ate_std_m"] = "N/A"
            res_row["rpe_trans_rmse_m"] = "N/A"
            res_row["rpe_rot_rmse_deg"] = "N/A"
            res_row["rpe_1s_trans_rmse_m"] = "N/A"
            res_row["rpe_1s_rot_rmse_deg"] = "N/A"

            cross_session_inlier_rms = anchor_stats.inlier_rms_m if anchor_stats else "N/A"
            res_row["cross_session_inlier_anchor_rms_m"] = round(cross_session_inlier_rms, 4) if isinstance(cross_session_inlier_rms, float) else "N/A"

            json_ground_truth_free[seq_name] = {
                "sequence_name": seq_name,
                "dataset_type": str(entry.dataset_type),
                "trajectory_poses": len(est_traj),
                "frame_tracking_rate": tracking_rate,
                "matched_input_frames": matched_inputs,
                "total_input_frames": total_inputs,
                "continuity": continuity.to_dict(),
                "map_quality": map_stats.to_dict(),
                "native_log": log_stats.to_dict(),
                "cross_session_registration": anchor_stats.to_dict() if anchor_stats else None,
            }

        all_results.append(res_row)

    # Combined panel plot for GT sequences
    plot_gt_benchmark_panel(all_results, plots_dir / "gt_benchmark_trajectories.png")

    # Cross-session plot for RealSense D455 sequences
    if "my_desk_recording_01" in session_trajectories and "my_desk_recording_02" in session_trajectories and anchor_stats:
        plot_d455_cross_session_alignment(
            session1_traj=session_trajectories["my_desk_recording_01"],
            session2_traj=session_trajectories["my_desk_recording_02"],
            anchor_stats=anchor_stats,
            output_png=plots_dir / "d455_cross_session_alignment.png",
        )

    # Assemble baseline_metrics.json
    final_json = {
        "evaluation_title": "Baseline ORB-SLAM3 Trajectory and Mapping Quality Evaluation",
        "total_sequences": len(all_results),
        "ground_truth_evaluated": json_ground_truth_evaluated,
        "ground_truth_free": json_ground_truth_free,
    }

    # Save CSV
    csv_path = baselines_dir / "baseline_comparison.csv"
    csv_columns = [
        "sequence_name",
        "dataset_type",
        "ground_truth_available",
        "estimated_poses",
        "ground_truth_poses",
        "matched_poses",
        "frame_tracking_rate_pct",
        "ground_truth_sample_coverage_pct",
        "estimated_pose_match_rate_pct",
        "ate_rmse_m",
        "ate_mean_m",
        "ate_median_m",
        "ate_std_m",
        "rpe_trans_rmse_m",
        "rpe_rot_rmse_deg",
        "rpe_1s_trans_rmse_m",
        "rpe_1s_rot_rmse_deg",
        "trajectory_path_length_m",
        "median_dt_s",
        "max_tracking_gap_s",
        "gaps_exceeding_100ms",
        "longest_continuous_run",
        "map_points_count",
        "map_point_linear_density_pts_per_m",
        "map_robust_volume_m3",
        "native_tracking_failures",
        "native_relocalizations",
        "native_maps_created",
        "native_merges",
        "native_loops",
    ]

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=csv_columns, extrasaction="ignore")
        writer.writeheader()
        for row in all_results:
            writer.writerow(row)

    # Save JSON
    json_path = baselines_dir / "baseline_metrics.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(final_json, f, indent=2)

    return all_results, final_json


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate baseline trajectory and mapping quality across all sequences.")
    parser.parse_args()

    print("Evaluating baseline trajectory and mapping quality across all 8 sequences...")
    results, _ = evaluate_all_baselines()

    repo_root = get_repo_root()
    print(f"\nCompleted evaluation for {len(results)} sequences.")
    print(f"Deliverables generated:")
    print(f"  - CSV:  results/baselines/baseline_comparison.csv")
    print(f"  - JSON: results/baselines/baseline_metrics.json")
    print(f"  - Plots: results/baselines/plots/")

    # Print summary table to console
    print("\nSummary Table:")
    print(f"{'Sequence':<40} {'Poses':<8} {'Track %':<10} {'GT Cov %':<10} {'ATE RMSE':<10} {'1s RPE':<10} {'Map Pts':<10}")
    print("-" * 102)
    for r in results:
        ate_str = f"{r['ate_rmse_m']} m" if r["ate_rmse_m"] != "N/A" else "GT-Free"
        rpe_str = f"{r['rpe_1s_trans_rmse_m']} m" if r["rpe_1s_trans_rmse_m"] != "N/A" else "N/A"
        gt_cov_str = f"{r['ground_truth_sample_coverage_pct']}%" if r["ground_truth_sample_coverage_pct"] != "N/A" else "N/A"
        print(f"{r['sequence_name']:<40} {r['estimated_poses']:<8} {r['frame_tracking_rate_pct']:<10} {gt_cov_str:<10} {ate_str:<10} {rpe_str:<10} {r['map_points_count']:<10}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
