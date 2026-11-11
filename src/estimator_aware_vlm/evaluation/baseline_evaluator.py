"""Baseline Trajectory and Mapping Quality Evaluation.

Evaluates ORB-SLAM3 baseline execution across all eight benchmark sequences,
providing rigorous separation between ground-truth-evaluated sequences and
ground-truth-free RealSense D455 sequences. Computes ATE, RPE, tracking
continuity, sparse map density, native telemetry event statistics, and
cross-session registration residuals.
"""

from __future__ import annotations

import csv
from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import numpy as np

from estimator_aware_vlm.common.paths import get_repo_root
from estimator_aware_vlm.datasets.base import (
    Trajectory,
    TrajectoryPose,
    TrajectoryRole,
)
from estimator_aware_vlm.evaluation.trajectory_metrics import (
    ATEStats,
    RPEStats,
    align_trajectories_se3,
    compute_ate,
    compute_frame_tracking_rate,
    compute_rpe,
    match_trajectories_by_timestamp,
)


@dataclass(frozen=True)
class TrajectoryContinuityStats:
    """Continuity and temporal sampling metrics for estimated trajectories."""
    total_poses: int
    duration_sec: float
    median_dt_sec: float
    mean_dt_sec: float
    max_gap_sec: float
    gaps_over_100ms_count: int
    longest_continuous_run_poses: int
    path_length_m: float

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class MapQualityStats:
    """Sparse map point metrics and spatial bounding volumes."""
    vertex_count: int
    robust_extents_m: Tuple[float, float, float]
    robust_volume_m3: float
    linear_density_pts_per_m: float
    volumetric_density_pts_per_m3: float

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class NativeLogStats:
    """Parsed runtime events from native ORB-SLAM3 execution log."""
    tracking_failure_events: int
    relocalization_events: int
    maps_created_count: int
    map_merges_detected: int
    loop_closures_detected: int
    mean_tracking_time_ms: Optional[float]
    median_tracking_time_ms: Optional[float]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class CrossSessionAnchorStats:
    """Rigid cross-session alignment and stationary anchor consistency."""
    reference_session: str
    aligned_session: str
    rotation_deg: float
    translation_m: Tuple[float, float]
    total_anchors: int
    robust_inliers: int
    inlier_rms_m: float
    all_anchor_rms_m: float
    anchor_residuals: Dict[str, float]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def parse_estimated_trajectory(file_path: Path) -> Trajectory:
    """Parse ORB-SLAM3 CameraTrajectory.txt into a Trajectory."""
    if not file_path.exists():
        raise FileNotFoundError(f"Trajectory file not found: {file_path}")

    poses: List[TrajectoryPose] = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            parts = stripped.split()
            if len(parts) >= 8:
                poses.append(
                    TrajectoryPose(
                        timestamp=float(parts[0]),
                        position=(float(parts[1]), float(parts[2]), float(parts[3])),
                        orientation=(float(parts[4]), float(parts[5]), float(parts[6]), float(parts[7])),
                        role=TrajectoryRole.ESTIMATED,
                    )
                )

    poses.sort(key=lambda p: p.timestamp)
    if not poses:
        raise ValueError(f"Trajectory at {file_path} contains 0 valid poses.")
    return Trajectory(poses=poses, role=TrajectoryRole.ESTIMATED, name="orbslam3_baseline")


def compute_trajectory_continuity(trajectory: Trajectory) -> TrajectoryContinuityStats:
    """Compute exact timestamp continuity, gaps, and total 3D path length."""
    if len(trajectory) == 0:
        return TrajectoryContinuityStats(0, 0.0, 0.0, 0.0, 0.0, 0, 0, 0.0)

    timestamps = np.asarray(trajectory.timestamps, dtype=np.float64)
    positions = trajectory.positions

    if len(timestamps) == 1:
        return TrajectoryContinuityStats(1, 0.0, 0.0, 0.0, 0.0, 0, 1, 0.0)

    dts = np.diff(timestamps)
    duration = float(timestamps[-1] - timestamps[0])
    median_dt = float(np.median(dts))
    mean_dt = float(np.mean(dts))
    max_gap = float(np.max(dts))
    gaps_over_100ms = int(np.sum(dts > 0.1))

    # Longest continuous run where dt <= 0.1s
    longest_run = 1
    current_run = 1
    for dt in dts:
        if dt <= 0.1:
            current_run += 1
            if current_run > longest_run:
                longest_run = current_run
        else:
            current_run = 1

    # 3D Path length
    pos_diffs = np.diff(positions, axis=0)
    path_len = float(np.sum(np.linalg.norm(pos_diffs, axis=1)))

    return TrajectoryContinuityStats(
        total_poses=len(trajectory),
        duration_sec=duration,
        median_dt_sec=median_dt,
        mean_dt_sec=mean_dt,
        max_gap_sec=max_gap,
        gaps_over_100ms_count=gaps_over_100ms,
        longest_continuous_run_poses=longest_run,
        path_length_m=path_len,
    )


def compute_map_point_stats(
    ply_path: Path,
    path_length_m: float,
) -> MapQualityStats:
    """Parse ASCII MapPoints.ply and compute robust spatial extents and point density."""
    if not ply_path.exists():
        return MapQualityStats(0, (0.0, 0.0, 0.0), 0.0, 0.0, 0.0)

    points: List[List[float]] = []
    in_header = True
    is_ascii = False

    with open(ply_path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            stripped = line.strip()
            if in_header:
                if stripped.startswith("format ") and "ascii 1.0" in stripped:
                    is_ascii = True
                elif stripped == "end_header":
                    in_header = False
                continue

            if not stripped:
                continue
            parts = stripped.split()
            if len(parts) >= 3:
                try:
                    x, y, z = float(parts[0]), float(parts[1]), float(parts[2])
                    if math.isfinite(x) and math.isfinite(y) and math.isfinite(z):
                        points.append([x, y, z])
                except ValueError:
                    continue

    if not points or not is_ascii:
        return MapQualityStats(0, (0.0, 0.0, 0.0), 0.0, 0.0, 0.0)

    pts = np.asarray(points, dtype=np.float64)
    total_pts = len(pts)

    # 1st to 99th percentile robust bounding box
    p1 = np.percentile(pts, 1, axis=0)
    p99 = np.percentile(pts, 99, axis=0)
    rob_extents = np.maximum(0.0, p99 - p1)
    rob_volume = float(np.prod(rob_extents))

    linear_density = float(total_pts / path_length_m) if path_length_m > 0 else 0.0
    volumetric_density = float(total_pts / rob_volume) if rob_volume > 0 else 0.0

    return MapQualityStats(
        vertex_count=total_pts,
        robust_extents_m=(float(rob_extents[0]), float(rob_extents[1]), float(rob_extents[2])),
        robust_volume_m3=rob_volume,
        linear_density_pts_per_m=linear_density,
        volumetric_density_pts_per_m3=volumetric_density,
    )


def parse_native_execution_log(log_path: Path) -> NativeLogStats:
    """Parse ORB-SLAM3 log for tracking loss, relocalizations, map merges, and timings."""
    if not log_path.exists():
        return NativeLogStats(0, 0, 0, 0, 0, None, None)

    fails = 0
    relocs = 0
    maps_created = 0
    merges = 0
    loops = 0
    mean_time: Optional[float] = None
    median_time: Optional[float] = None

    with open(log_path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            stripped = line.strip()
            if "Fail to track local map!" in stripped:
                fails += 1
            elif "Relocalized!!" in stripped:
                relocs += 1
            elif "New Map created" in stripped:
                maps_created += 1
            elif "Merge detected" in stripped:
                merges += 1
            elif "Loop detected" in stripped:
                loops += 1
            elif "mean tracking time:" in stripped:
                try:
                    mean_time = float(stripped.split(":")[-1].strip()) * 1000.0  # convert to ms
                except ValueError:
                    pass
            elif "median tracking time:" in stripped:
                try:
                    median_time = float(stripped.split(":")[-1].strip()) * 1000.0  # convert to ms
                except ValueError:
                    pass

    return NativeLogStats(
        tracking_failure_events=fails,
        relocalization_events=relocs,
        maps_created_count=maps_created,
        map_merges_detected=merges,
        loop_closures_detected=loops,
        mean_tracking_time_ms=mean_time,
        median_tracking_time_ms=median_time,
    )


def load_cross_session_anchors(report_path: Path) -> Optional[CrossSessionAnchorStats]:
    """Load authentic cross-session registration and anchor residual audit."""
    if not report_path.exists():
        return None

    try:
        with open(report_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        anchors = data.get("anchors", [])
        total_anchors = len(anchors)
        inliers = [a for a in anchors if a.get("is_robust_inlier", False)]
        inlier_count = len(inliers)

        inlier_residuals = [float(a["residual_m"]) for a in inliers]
        all_residuals = [float(a["residual_m"]) for a in anchors]

        inlier_rms = float(np.sqrt(np.mean(np.square(inlier_residuals)))) if inlier_residuals else 0.0
        all_rms = float(np.sqrt(np.mean(np.square(all_residuals)))) if all_residuals else 0.0

        residual_map = {f"{a.get('id1', 'id')}_{a.get('id2', 'id')}": float(a.get("residual_m", 0.0)) for a in anchors}

        t_trans = data.get("translation_m", [0.0, 0.0])

        return CrossSessionAnchorStats(
            reference_session=str(data.get("reference_session", "session_1")),
            aligned_session=str(data.get("aligned_session", "session_2")),
            rotation_deg=float(data.get("rotation_deg", 0.0)),
            translation_m=(float(t_trans[0]), float(t_trans[1])),
            total_anchors=total_anchors,
            robust_inliers=inlier_count,
            inlier_rms_m=inlier_rms,
            all_anchor_rms_m=all_rms,
            anchor_residuals=residual_map,
        )
    except Exception:
        return None
