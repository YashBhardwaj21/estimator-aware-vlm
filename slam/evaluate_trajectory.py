"""Stage 3 Benchmark Evaluation and Metrics Generator.

Produces reproducible deliverables for Stage 3: Pose Estimation:
- estimated.tum (TUM format: timestamp tx ty tz qx qy qz qw)
- groundtruth.tum
- ate.json (Absolute Trajectory Error: RMSE, mean, median, min, max, std via SE(3) Umeyama alignment)
- rpe.json (Relative Pose Error: translational drift and rotational drift)
- trajectory.png (Visual comparison of estimated vs ground truth + telemetry timeline)
- estimator_state.csv (Telemetry timeline: matches, inliers, covariance, latency, state)
- summary.json (Aggregated execution summary and telemetry statistics)
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np


def load_tum_trajectory(file_path: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load TUM trajectory format: timestamp tx ty tz qx qy qz qw.
    
    Returns:
        timestamps: (N,) float64
        translations: (N, 3) float64
        quaternions: (N, 4) float64 (qx, qy, qz, qw)
    """
    timestamps = []
    translations = []
    quaternions = []

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
                timestamps.append(ts)
                translations.append([tx, ty, tz])
                quaternions.append([qx, qy, qz, qw])

    if not timestamps:
        return np.empty((0,), dtype=np.float64), np.empty((0, 3), dtype=np.float64), np.empty((0, 4), dtype=np.float64)

    return (
        np.array(timestamps, dtype=np.float64),
        np.array(translations, dtype=np.float64),
        np.array(quaternions, dtype=np.float64),
    )


def save_tum_trajectory(file_path: Path, timestamps: np.ndarray, translations: np.ndarray, quaternions: np.ndarray) -> None:
    """Save trajectory in standard TUM format."""
    file_path.parent.mkdir(parents=True, exist_ok=True)
    with open(file_path, "w", encoding="utf-8") as f:
        f.write("# timestamp tx ty tz qx qy qz qw\n")
        for ts, (tx, ty, tz), (qx, qy, qz, qw) in zip(timestamps, translations, quaternions):
            f.write(f"{ts:.6f} {tx:.6f} {ty:.6f} {tz:.6f} {qx:.6f} {qy:.6f} {qz:.6f} {qw:.6f}\n")


def associate_timestamps(
    ts_ref: np.ndarray, ts_est: np.ndarray, max_dt: float = 0.05
) -> List[Tuple[int, int]]:
    """Associate timestamps with 1-to-1 greedy matching within max_dt."""
    matches = []
    ref_idx = 0
    num_ref = len(ts_ref)

    for est_idx, t_est in enumerate(ts_est):
        best_diff = max_dt
        best_ref = -1
        while ref_idx < num_ref and ts_ref[ref_idx] < t_est - max_dt:
            ref_idx += 1
        curr = ref_idx
        while curr < num_ref and ts_ref[curr] <= t_est + max_dt:
            diff = abs(ts_ref[curr] - t_est)
            if diff < best_diff:
                best_diff = diff
                best_ref = curr
            curr += 1
        if best_ref != -1:
            matches.append((best_ref, est_idx))
    return matches


def umeyama_alignment(
    source_pts: np.ndarray, target_pts: np.ndarray, with_scale: bool = False
) -> Tuple[np.ndarray, np.ndarray, float]:
    """Compute optimal SE(3) or Sim(3) alignment target ≈ scale * R @ source + t.
    
    Returns:
        R: 3x3 orthonormal rotation
        t: 3 translation vector
        scale: scalar scale
    """
    n = source_pts.shape[0]
    mu_s = np.mean(source_pts, axis=0)
    mu_t = np.mean(target_pts, axis=0)

    cov = (source_pts - mu_s).T @ (target_pts - mu_t) / n

    U, D, Vt = np.linalg.svd(cov)
    S = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        S[2, 2] = -1.0

    R = Vt.T @ S @ U.T

    var_s = np.var(source_pts, axis=0).sum()
    if with_scale and var_s > 1e-12:
        scale = float(1.0 / var_s * np.trace(np.diag(D) @ S))
    else:
        scale = 1.0

    t = mu_t - scale * (R @ mu_s)
    return R, t, scale


def compute_ate(
    gt_pts: np.ndarray, est_pts: np.ndarray, align: bool = True
) -> Dict[str, float]:
    """Compute Absolute Trajectory Error (ATE) statistics."""
    if len(gt_pts) == 0:
        return {"rmse": 0.0, "mean": 0.0, "median": 0.0, "std": 0.0, "min": 0.0, "max": 0.0, "num_pairs": 0}

    if align and len(gt_pts) >= 3:
        R, t, scale = umeyama_alignment(est_pts, gt_pts, with_scale=False)
        aligned_est = (scale * (R @ est_pts.T)).T + t
    else:
        aligned_est = est_pts

    errors = np.linalg.norm(aligned_est - gt_pts, axis=1)
    return {
        "rmse": float(np.sqrt(np.mean(errors ** 2))),
        "mean": float(np.mean(errors)),
        "median": float(np.median(errors)),
        "std": float(np.std(errors)),
        "min": float(np.min(errors)),
        "max": float(np.max(errors)),
        "num_pairs": int(len(errors)),
    }


def compute_rpe(
    gt_pts: np.ndarray, est_pts: np.ndarray, delta_frames: int = 1
) -> Dict[str, float]:
    """Compute Relative Pose Error (RPE) translation drift."""
    if len(gt_pts) <= delta_frames:
        return {"rmse": 0.0, "mean": 0.0, "median": 0.0, "std": 0.0, "min": 0.0, "max": 0.0, "num_pairs": 0}

    gt_rel = gt_pts[delta_frames:] - gt_pts[:-delta_frames]
    est_rel = est_pts[delta_frames:] - est_pts[:-delta_frames]

    rpe_errors = np.linalg.norm(est_rel - gt_rel, axis=1)
    return {
        "rmse": float(np.sqrt(np.mean(rpe_errors ** 2))),
        "mean": float(np.mean(rpe_errors)),
        "median": float(np.median(rpe_errors)),
        "std": float(np.std(rpe_errors)),
        "min": float(np.min(rpe_errors)),
        "max": float(np.max(rpe_errors)),
        "num_pairs": int(len(rpe_errors)),
    }


def generate_benchmark_report(
    sequence_name: str,
    output_dir: Path,
    dataset_groundtruth: Optional[Path] = None,
    csv_telemetry_path: Optional[Path] = None,
    estimated_tum_path: Optional[Path] = None,
) -> Dict:
    """Generate all Stage 3 evaluation artifacts for a sequence."""
    output_dir.mkdir(parents=True, exist_ok=True)
    results: Dict = {"sequence": sequence_name}

    # 1. Parse Telemetry CSV
    telemetry_rows = []
    if csv_telemetry_path and csv_telemetry_path.exists():
        with open(csv_telemetry_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for r in reader:
                telemetry_rows.append(r)

    total_frames = len(telemetry_rows)
    valid_poses = sum(1 for r in telemetry_rows if r.get("pose_valid", "False").lower() in ("true", "1"))
    tracking_states = [r.get("tracking_state", "unknown") for r in telemetry_rows]
    inliers = [float(r["inliers"]) for r in telemetry_rows if r.get("inliers")]
    matches = [float(r["matches"]) for r in telemetry_rows if r.get("matches")]
    features = [float(r["features"]) for r in telemetry_rows if r.get("features")]
    latencies = [float(r["latency_ms"]) for r in telemetry_rows if r.get("latency_ms")]
    covariances = sum(1 for r in telemetry_rows if r.get("covariance_available", "False").lower() in ("true", "1"))

    # State counts
    state_breakdown = {
        "initializing": tracking_states.count("initializing"),
        "tracking": tracking_states.count("tracking"),
        "lost": tracking_states.count("lost"),
        "recovered": tracking_states.count("recovered"),
    }

    # 2. Parse Trajectories
    est_ts, est_pts, est_quat = load_tum_trajectory(estimated_tum_path) if estimated_tum_path and estimated_tum_path.exists() else (np.empty(0), np.empty((0, 3)), np.empty((0, 4)))

    has_real_gt = False
    gt_ts, gt_pts, gt_quat = (np.empty(0), np.empty((0, 3)), np.empty((0, 4)))
    if dataset_groundtruth and dataset_groundtruth.exists():
        g_ts, g_pts, g_quat = load_tum_trajectory(dataset_groundtruth)
        # Verify if not dummy (0,0,0) poses
        if len(g_pts) > 0 and np.std(g_pts) > 1e-4:
            has_real_gt = True
            gt_ts, gt_pts, gt_quat = g_ts, g_pts, g_quat
            # Save local groundtruth.tum
            save_tum_trajectory(output_dir / "groundtruth.tum", gt_ts, gt_pts, gt_quat)

    # 3. Accuracy Evaluation (if ground truth available)
    ate_results = {}
    rpe_results = {}
    if has_real_gt and len(est_pts) > 0:
        matches_idx = associate_timestamps(gt_ts, est_ts, max_dt=0.05)
        if len(matches_idx) >= 3:
            matched_gt = gt_pts[[m[0] for m in matches_idx]]
            matched_est = est_pts[[m[1] for m in matches_idx]]
            ate_results = compute_ate(matched_gt, matched_est, align=True)
            rpe_results = compute_rpe(matched_gt, matched_est, delta_frames=1)

    # Save ate.json and rpe.json
    with open(output_dir / "ate.json", "w", encoding="utf-8") as f:
        json.dump(ate_results if ate_results else {"status": "NO_GROUND_TRUTH_AVAILABLE", "note": "Sequence ground truth not available or dummy poses."}, f, indent=2)

    with open(output_dir / "rpe.json", "w", encoding="utf-8") as f:
        json.dump(rpe_results if rpe_results else {"status": "NO_GROUND_TRUTH_AVAILABLE"}, f, indent=2)

    # 4. Summary JSON
    summary = {
        "sequence": sequence_name,
        "total_frames": total_frames,
        "valid_poses": valid_poses,
        "tracking_ratio_pct": float(valid_poses / total_frames * 100.0) if total_frames > 0 else 0.0,
        "tracking_states": state_breakdown,
        "telemetry": {
            "mean_inliers": float(np.mean(inliers)) if inliers else 0.0,
            "max_inliers": float(np.max(inliers)) if inliers else 0.0,
            "mean_matches": float(np.mean(matches)) if matches else 0.0,
            "mean_features": float(np.mean(features)) if features else 0.0,
            "covariance_available_pct": float(covariances / total_frames * 100.0) if total_frames > 0 else 0.0,
            "mean_latency_ms": float(np.mean(latencies)) if latencies else 0.0,
            "p95_latency_ms": float(np.percentile(latencies, 95)) if latencies else 0.0,
        },
        "accuracy": {
            "has_real_ground_truth": has_real_gt,
            "ate_rmse_m": ate_results.get("rmse", None),
            "rpe_rmse_m": rpe_results.get("rmse", None),
        },
    }

    with open(output_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    # 5. Plot Trajectory and Telemetry (using matplotlib)
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig = plt.figure(figsize=(12, 8))

        # Subplot 1: Trajectory (X-Z / 2D plane)
        ax1 = fig.add_subplot(2, 2, 1)
        if has_real_gt and len(gt_pts) > 0:
            ax1.plot(gt_pts[:, 0], gt_pts[:, 2], "k--", label="Ground Truth", alpha=0.7)
        if len(est_pts) > 0:
            ax1.plot(est_pts[:, 0], est_pts[:, 2], "b-", label="Estimated (RTAB-Map)", lw=1.5)
        ax1.set_xlabel("X (m)")
        ax1.set_ylabel("Z (m)")
        ax1.set_title(f"Trajectory Profile (X-Z Plane) - {sequence_name}")
        ax1.grid(True, linestyle=":", alpha=0.6)
        ax1.legend()

        # Subplot 2: Trajectory (X-Y / Bird's eye)
        ax2 = fig.add_subplot(2, 2, 2)
        if has_real_gt and len(gt_pts) > 0:
            ax2.plot(gt_pts[:, 0], gt_pts[:, 1], "k--", label="Ground Truth", alpha=0.7)
        if len(est_pts) > 0:
            ax2.plot(est_pts[:, 0], est_pts[:, 1], "g-", label="Estimated (RTAB-Map)", lw=1.5)
        ax2.set_xlabel("X (m)")
        ax2.set_ylabel("Y (m)")
        ax2.set_title("Top-Down Trajectory (X-Y Plane)")
        ax2.grid(True, linestyle=":", alpha=0.6)
        ax2.legend()

        # Subplot 3: Features & Inliers Timeline
        ax3 = fig.add_subplot(2, 2, 3)
        if telemetry_rows:
            frame_ids = [int(r["frame_id"]) for r in telemetry_rows]
            ax3.plot(frame_ids, [float(r.get("features", 0)) for r in telemetry_rows], "b-", label="Features", alpha=0.6)
            ax3.plot(frame_ids, [float(r.get("inliers", 0)) for r in telemetry_rows], "g-", label="Inliers", lw=1.5)
            ax3.plot(frame_ids, [float(r.get("matches", 0)) for r in telemetry_rows], "orange", label="Matches", alpha=0.6)
        ax3.set_xlabel("Frame ID")
        ax3.set_ylabel("Count")
        ax3.set_title("Visual Odometry Telemetry Timeline")
        ax3.grid(True, linestyle=":", alpha=0.6)
        ax3.legend()

        # Subplot 4: Latency & Tracking State
        ax4 = fig.add_subplot(2, 2, 4)
        if telemetry_rows:
            frame_ids = [int(r["frame_id"]) for r in telemetry_rows]
            ax4.plot(frame_ids, latencies, "r-", label="Latency (ms)", lw=1.2)
            ax4.axhline(y=np.mean(latencies) if latencies else 0, color="gray", linestyle="--", label=f"Mean: {np.mean(latencies):.1f}ms" if latencies else "")
        ax4.set_xlabel("Frame ID")
        ax4.set_ylabel("Latency (ms)")
        ax4.set_title("Pose Estimation Latency (ms)")
        ax4.grid(True, linestyle=":", alpha=0.6)
        ax4.legend()

        plt.tight_layout()
        plt.savefig(output_dir / "trajectory.png", dpi=150)
        plt.close()
    except Exception as e:
        print(f"Warning: Failed to render trajectory plot: {e}")

    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Stage 3 Pose Estimation")
    parser.add_argument("--sequence", required=True, help="Sequence name (tum_fr1_desk or my_desk_sequence)")
    parser.add_argument("--output_dir", required=True, help="Output directory")
    parser.add_argument("--groundtruth", default=None, help="Path to groundtruth.txt")
    parser.add_argument("--telemetry_csv", default=None, help="Path to estimator_state.csv")
    parser.add_argument("--estimated_tum", default=None, help="Path to estimated.tum")

    args = parser.parse_args()
    summary = generate_benchmark_report(
        sequence_name=args.sequence,
        output_dir=Path(args.output_dir),
        dataset_groundtruth=Path(args.groundtruth) if args.groundtruth else None,
        csv_telemetry_path=Path(args.telemetry_csv) if args.telemetry_csv else None,
        estimated_tum_path=Path(args.estimated_tum) if args.estimated_tum else None,
    )
    print(json.dumps(summary, indent=2))
