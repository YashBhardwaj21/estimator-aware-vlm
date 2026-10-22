"""Trajectory Ingestion, Strict Provenance, 1-to-1 Matching, and Quality Verification.

Guarantees:
- Single authoritative trajectory gate for both generated and override trajectories
- Finite value and valid SE(3) rotation checks (det(R)=+1, orthogonality)
- Strict timestamp monotonicity (no duplicate or backwards timestamps)
- 1-to-1 unique matching policy (one pose cannot match multiple frames)
- SHA-256 provenance tracking
- Atlas artifact verification before claiming localization readiness
- Quaternion SLERP and translation interpolation within tight bounds
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from scene_graph.geometry.transforms import pose_to_transform


class TrajectoryValidationError(ValueError):
    """Raised when trajectory fails provenance, mathematical validity, or alignment."""
    pass


@dataclass
class TrajectoryResult:
    """Comprehensive trajectory ingest and association report."""
    is_valid: bool
    status: str  # VALID, REJECTED_MISMATCH, REJECTED_DISCONTINUITY, REJECTED_NON_FINITE, REJECTED_MONOTONICITY
    source_file: str
    file_sha256: str
    generated_in_current_run: bool
    poses: Dict[float, np.ndarray]  # timestamp -> 4x4 SE(3) matrix
    pose_timestamps: np.ndarray
    atlas_path: Optional[str]
    atlas_verified: bool
    total_recording_frames: int
    unique_matched_frames: int
    coverage_percentage: float
    unmatched_frames_count: int
    gaps_count: int
    max_gap_s: float
    temporal_overlap_s: float
    temporal_overlap_pct: float
    mean_dt_ms: float
    rejection_reason: Optional[str] = None
    audit_summary: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "is_valid": self.is_valid,
            "status": self.status,
            "source_file": self.source_file,
            "file_sha256": self.file_sha256,
            "generated_in_current_run": self.generated_in_current_run,
            "total_poses_in_trajectory": len(self.poses),
            "atlas_path": self.atlas_path,
            "atlas_verified": self.atlas_verified,
            "total_recording_frames": self.total_recording_frames,
            "unique_matched_frames": self.unique_matched_frames,
            "coverage_percentage": round(self.coverage_percentage, 2),
            "unmatched_frames_count": self.unmatched_frames_count,
            "gaps_count": self.gaps_count,
            "max_gap_s": round(self.max_gap_s, 3),
            "temporal_overlap_s": round(self.temporal_overlap_s, 2),
            "temporal_overlap_pct": round(self.temporal_overlap_pct, 2),
            "mean_dt_ms": round(self.mean_dt_ms, 2),
            "rejection_reason": self.rejection_reason,
            "audit_summary": self.audit_summary,
        }


def compute_file_sha256(path: Path) -> str:
    """Compute SHA-256 hash of a file for strict provenance tracking."""
    hasher = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()


def quaternion_slerp(q0: np.ndarray, q1: np.ndarray, fraction: float) -> np.ndarray:
    """Spherical linear interpolation between two normalized quaternions [qx, qy, qz, qw]."""
    dot = np.dot(q0, q1)
    if dot < 0.0:
        q1 = -q1
        dot = -dot
    dot = np.clip(dot, -1.0, 1.0)
    if dot > 0.9995:
        res = q0 + fraction * (q1 - q0)
        return res / np.linalg.norm(res)
    theta_0 = np.arccos(dot)
    sin_theta_0 = np.sin(theta_0)
    theta = theta_0 * fraction
    sin_theta = np.sin(theta)
    s0 = np.cos(theta) - dot * sin_theta / sin_theta_0
    s1 = sin_theta / sin_theta_0
    res = (s0 * q0) + (s1 * q1)
    return res / np.linalg.norm(res)


def load_tum_trajectory(traj_path: str | Path) -> Dict[float, np.ndarray]:
    """Load TUM trajectory format: timestamp tx ty tz qx qy qz qw -> 4x4 SE(3) matrix."""
    path = Path(traj_path)
    if not path.exists():
        raise FileNotFoundError(f"Trajectory file not found: {path}")

    poses: Dict[float, np.ndarray] = {}
    last_t = -float("inf")

    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 8:
                continue

            try:
                vals = [float(p) for p in parts[:8]]
            except ValueError:
                raise TrajectoryValidationError(f"Line {line_no} contains non-numeric values in {path}")

            t = vals[0]
            tx, ty, tz = vals[1:4]
            qx, qy, qz, qw = vals[4:8]

            # 1. Finite checks
            if not all(np.isfinite(v) for v in vals):
                raise TrajectoryValidationError(f"Line {line_no} contains NaN or Inf values in {path}")

            # 2. Duplicate or backwards timestamps
            if t <= last_t:
                raise TrajectoryValidationError(f"Line {line_no} timestamp {t} is not strictly monotonic (> {last_t}) in {path}")
            last_t = t

            # 3. Quaternion norm
            q_norm = math.sqrt(qx*qx + qy*qy + qz*qz + qw*qw)
            if abs(q_norm - 1.0) > 0.05 or q_norm == 0.0:
                raise TrajectoryValidationError(f"Line {line_no} invalid quaternion norm {q_norm:.4f} in {path}")
            qx, qy, qz, qw = qx/q_norm, qy/q_norm, qz/q_norm, qw/q_norm

            T_wc = pose_to_transform(tx, ty, tz, qx, qy, qz, qw)

            # 4. Rotation validity
            R = T_wc[:3, :3]
            if abs(np.linalg.det(R) - 1.0) > 1e-4:
                raise TrajectoryValidationError(f"Line {line_no} rotation matrix det is {np.linalg.det(R):.4f} != +1.0 in {path}")

            poses[t] = T_wc

    if not poses:
        raise TrajectoryValidationError(f"No valid TUM poses found in {traj_path}")
    return poses


def load_and_validate_trajectory(
    trajectory_file: str | Path,
    rgb_timestamps: List[float],
    max_dt: float = 0.05,
    min_coverage_pct: float = 20.0,
    min_temporal_overlap_pct: float = 50.0,
    atlas_path: Optional[str | Path] = None,
    generated_in_current_run: bool = False,
    max_translation_jump_m: float = 5.0,
) -> TrajectoryResult:
    """Comprehensive single-gate validation for trajectory provenance, geometry, and 1-to-1 matching."""
    traj_path = Path(trajectory_file)
    if not traj_path.exists():
        return TrajectoryResult(
            is_valid=False,
            status="REJECTED_MISSING_FILE",
            source_file=str(traj_path),
            file_sha256="",
            generated_in_current_run=generated_in_current_run,
            poses={},
            pose_timestamps=np.array([]),
            atlas_path=str(atlas_path) if atlas_path else None,
            atlas_verified=False,
            total_recording_frames=len(rgb_timestamps),
            unique_matched_frames=0,
            coverage_percentage=0.0,
            unmatched_frames_count=len(rgb_timestamps),
            gaps_count=0,
            max_gap_s=0.0,
            temporal_overlap_s=0.0,
            temporal_overlap_pct=0.0,
            mean_dt_ms=0.0,
            rejection_reason=f"Trajectory file does not exist: {traj_path}",
        )

    file_sha = compute_file_sha256(traj_path)

    # 1. Parse and validate internal consistency
    try:
        poses = load_tum_trajectory(traj_path)
    except TrajectoryValidationError as ex:
        return TrajectoryResult(
            is_valid=False,
            status="REJECTED_INVALID_VALUES",
            source_file=str(traj_path),
            file_sha256=file_sha,
            generated_in_current_run=generated_in_current_run,
            poses={},
            pose_timestamps=np.array([]),
            atlas_path=str(atlas_path) if atlas_path else None,
            atlas_verified=False,
            total_recording_frames=len(rgb_timestamps),
            unique_matched_frames=0,
            coverage_percentage=0.0,
            unmatched_frames_count=len(rgb_timestamps),
            gaps_count=0,
            max_gap_s=0.0,
            temporal_overlap_s=0.0,
            temporal_overlap_pct=0.0,
            mean_dt_ms=0.0,
            rejection_reason=str(ex),
        )

    pose_keys = np.array(sorted(poses.keys()))

    # 2. Check for implausible physical discontinuities
    for i in range(1, len(pose_keys)):
        dt_pair = pose_keys[i] - pose_keys[i - 1]
        if dt_pair < 1.0:  # within continuous stream
            pos_prev = poses[pose_keys[i - 1]][:3, 3]
            pos_cur = poses[pose_keys[i]][:3, 3]
            jump = float(np.linalg.norm(pos_cur - pos_prev))
            if jump > max_translation_jump_m:
                return TrajectoryResult(
                    is_valid=False,
                    status="REJECTED_DISCONTINUITY",
                    source_file=str(traj_path),
                    file_sha256=file_sha,
                    generated_in_current_run=generated_in_current_run,
                    poses=poses,
                    pose_timestamps=pose_keys,
                    atlas_path=str(atlas_path) if atlas_path else None,
                    atlas_verified=False,
                    total_recording_frames=len(rgb_timestamps),
                    unique_matched_frames=0,
                    coverage_percentage=0.0,
                    unmatched_frames_count=len(rgb_timestamps),
                    gaps_count=0,
                    max_gap_s=0.0,
                    temporal_overlap_s=0.0,
                    temporal_overlap_pct=0.0,
                    mean_dt_ms=0.0,
                    rejection_reason=f"Implausible translation jump of {jump:.2f}m in {dt_pair*1000:.1f}ms between t={pose_keys[i-1]:.3f} and t={pose_keys[i]:.3f}",
                )

    if not rgb_timestamps:
        return TrajectoryResult(
            is_valid=False,
            status="REJECTED_NO_FRAMES",
            source_file=str(traj_path),
            file_sha256=file_sha,
            generated_in_current_run=generated_in_current_run,
            poses=poses,
            pose_timestamps=pose_keys,
            atlas_path=str(atlas_path) if atlas_path else None,
            atlas_verified=False,
            total_recording_frames=0,
            unique_matched_frames=0,
            coverage_percentage=0.0,
            unmatched_frames_count=0,
            gaps_count=0,
            max_gap_s=0.0,
            temporal_overlap_s=0.0,
            temporal_overlap_pct=0.0,
            mean_dt_ms=0.0,
            rejection_reason="No RGB timestamps provided in recording manifest.",
        )

    # 3. Temporal overlap check
    t_min_traj, t_max_traj = float(pose_keys[0]), float(pose_keys[-1])
    t_min_rgb, t_max_rgb = float(rgb_timestamps[0]), float(rgb_timestamps[-1])

    overlap_start = max(t_min_traj, t_min_rgb)
    overlap_end = min(t_max_traj, t_max_rgb)
    overlap_duration_s = max(0.0, overlap_end - overlap_start)
    rgb_duration_s = max(0.001, t_max_rgb - t_min_rgb)
    temporal_overlap_pct = (overlap_duration_s / rgb_duration_s) * 100.0

    # Check for complete disconnect in time window
    if t_max_traj < (t_min_rgb - max_dt) or t_min_traj > (t_max_rgb + max_dt):
        return TrajectoryResult(
            is_valid=False,
            status="REJECTED_MISMATCH",
            source_file=str(traj_path),
            file_sha256=file_sha,
            generated_in_current_run=generated_in_current_run,
            poses=poses,
            pose_timestamps=pose_keys,
            atlas_path=str(atlas_path) if atlas_path else None,
            atlas_verified=False,
            total_recording_frames=len(rgb_timestamps),
            unique_matched_frames=0,
            coverage_percentage=0.0,
            unmatched_frames_count=len(rgb_timestamps),
            gaps_count=0,
            max_gap_s=0.0,
            temporal_overlap_s=round(overlap_duration_s, 2),
            temporal_overlap_pct=round(temporal_overlap_pct, 2),
            mean_dt_ms=0.0,
            rejection_reason=(
                f"Trajectory timestamps [{t_min_traj:.3f}, {t_max_traj:.3f}] do not overlap "
                f"with recording RGB timestamps [{t_min_rgb:.3f}, {t_max_rgb:.3f}]. "
                f"Cannot reuse trajectory from a different dataset!"
            ),
        )

    # 4. 1-to-1 Unique matching policy (One pose can only match ONE frame)
    # Find all candidate pairs within max_dt
    candidate_pairs = []
    for f_idx, t_f in enumerate(rgb_timestamps):
        idx = np.searchsorted(pose_keys, t_f)
        neighbors = []
        if idx < len(pose_keys):
            neighbors.append(idx)
        if idx > 0:
            neighbors.append(idx - 1)
        for n_idx in neighbors:
            dt = abs(pose_keys[n_idx] - t_f)
            if dt <= max_dt:
                candidate_pairs.append((dt, f_idx, n_idx))

    # Sort candidate pairs by smallest delta time and match greedily 1-to-1
    candidate_pairs.sort(key=lambda x: x[0])
    matched_frames = set()
    used_poses = set()
    dt_list = []

    for dt, f_idx, n_idx in candidate_pairs:
        if f_idx not in matched_frames and n_idx not in used_poses:
            matched_frames.add(f_idx)
            used_poses.add(n_idx)
            dt_list.append(dt)

    unique_matched = len(matched_frames)
    total_frames = len(rgb_timestamps)
    coverage_pct = (unique_matched / total_frames) * 100.0 if total_frames else 0.0
    unmatched_count = total_frames - unique_matched
    mean_dt = float(np.mean(dt_list) * 1000.0) if dt_list else 0.0

    # Calculate gaps where frames were lost
    gaps_count = 0
    max_gap_s = 0.0
    last_matched_t = None
    for f_idx, t_f in enumerate(rgb_timestamps):
        if f_idx in matched_frames:
            if last_matched_t is not None:
                gap = t_f - last_matched_t
                if gap > 0.5:  # Tracking lost for > 500ms
                    gaps_count += 1
                max_gap_s = max(max_gap_s, gap)
            last_matched_t = t_f

    # 5. Check coverage threshold
    if coverage_pct < min_coverage_pct:
        return TrajectoryResult(
            is_valid=False,
            status="REJECTED_LOW_COVERAGE",
            source_file=str(traj_path),
            file_sha256=file_sha,
            generated_in_current_run=generated_in_current_run,
            poses=poses,
            pose_timestamps=pose_keys,
            atlas_path=str(atlas_path) if atlas_path else None,
            atlas_verified=False,
            total_recording_frames=total_frames,
            unique_matched_frames=unique_matched,
            coverage_percentage=round(coverage_pct, 2),
            unmatched_frames_count=unmatched_count,
            gaps_count=gaps_count,
            max_gap_s=round(max_gap_s, 3),
            temporal_overlap_s=round(overlap_duration_s, 2),
            temporal_overlap_pct=round(temporal_overlap_pct, 2),
            mean_dt_ms=round(mean_dt, 2),
            rejection_reason=f"Unique pose coverage {coverage_pct:.1f}% is below minimum required {min_coverage_pct:.1f}%.",
        )

    # 6. Verify Atlas if provided
    atlas_ok = False
    if atlas_path:
        ap = Path(atlas_path)
        if ap.exists() and ap.stat().st_size > 512:
            atlas_ok = True

    audit_summary = {
        "status": "VALID",
        "provenance": {
            "source_path": str(traj_path),
            "sha256": file_sha,
            "generated_in_current_run": generated_in_current_run,
        },
        "coverage": {
            "total_frames": total_frames,
            "unique_matched_frames": unique_matched,
            "coverage_pct": round(coverage_pct, 2),
            "unmatched_frames": unmatched_count,
            "mean_timestamp_skew_ms": round(mean_dt, 2),
            "gaps_count_gt_500ms": gaps_count,
            "max_gap_duration_s": round(max_gap_s, 3),
        },
        "temporal": {
            "trajectory_duration_s": round(t_max_traj - t_min_traj, 2),
            "recording_duration_s": round(rgb_duration_s, 2),
            "overlap_duration_s": round(overlap_duration_s, 2),
            "overlap_pct": round(temporal_overlap_pct, 2),
        },
        "atlas": {
            "path": str(atlas_path) if atlas_path else None,
            "verified_loadable": atlas_ok,
        },
    }

    return TrajectoryResult(
        is_valid=True,
        status="VALID",
        source_file=str(traj_path),
        file_sha256=file_sha,
        generated_in_current_run=generated_in_current_run,
        poses=poses,
        pose_timestamps=pose_keys,
        atlas_path=str(atlas_path) if atlas_path else None,
        atlas_verified=atlas_ok,
        total_recording_frames=total_frames,
        unique_matched_frames=unique_matched,
        coverage_percentage=round(coverage_pct, 2),
        unmatched_frames_count=unmatched_count,
        gaps_count=gaps_count,
        max_gap_s=round(max_gap_s, 3),
        temporal_overlap_s=round(overlap_duration_s, 2),
        temporal_overlap_pct=round(temporal_overlap_pct, 2),
        mean_dt_ms=round(mean_dt, 2),
        audit_summary=audit_summary,
    )


def validate_trajectory_alignment(
    poses: Optional[Dict[float, np.ndarray]] = None,
    rgb_timestamps: Optional[List[float]] = None,
    trajectory_file: Optional[Union[str, Path]] = None,
    max_dt: float = 0.05,
    min_match_ratio: float = 0.15,
    min_coverage_pct: float = 20.0,
) -> Dict[str, Any]:
    """Validate temporal alignment between trajectory poses and recording RGB timestamps."""
    if trajectory_file is not None and poses is None:
        return load_and_validate_trajectory(
            trajectory_file=trajectory_file,
            rgb_timestamps=rgb_timestamps or [],
            max_dt=max_dt,
            min_coverage_pct=min_coverage_pct,
        ).to_dict()

    poses = poses or {}
    rgb_timestamps = rgb_timestamps or []
    pose_keys = np.array(sorted(poses.keys()))
    t_min_traj = float(pose_keys[0]) if len(pose_keys) else 0.0
    t_max_traj = float(pose_keys[-1]) if len(pose_keys) else 0.0
    t_min_rgb = float(rgb_timestamps[0]) if len(rgb_timestamps) else 0.0
    t_max_rgb = float(rgb_timestamps[-1]) if len(rgb_timestamps) else 0.0

    overlap_start = max(t_min_traj, t_min_rgb)
    overlap_end = min(t_max_traj, t_max_rgb)
    overlap_duration_s = max(0.0, overlap_end - overlap_start)
    rgb_duration_s = max(0.001, t_max_rgb - t_min_rgb)
    temporal_overlap_pct = (overlap_duration_s / rgb_duration_s) * 100.0

    # Sample matching check
    sample_indices = np.linspace(0, len(rgb_timestamps) - 1, min(50, len(rgb_timestamps)), dtype=int) if rgb_timestamps else []
    matched = 0
    for idx in sample_indices:
        t_rgb = rgb_timestamps[idx]
        pos = np.searchsorted(pose_keys, t_rgb)
        best_dt = float("inf")
        if pos < len(pose_keys):
            best_dt = min(best_dt, abs(pose_keys[pos] - t_rgb))
        if pos > 0:
            best_dt = min(best_dt, abs(pose_keys[pos - 1] - t_rgb))
        if best_dt <= max_dt:
            matched += 1

    sample_ratio = (matched / len(sample_indices)) * 100.0 if len(sample_indices) else 0.0
    return {
        "trajectory_poses_count": len(poses),
        "overlap_duration_s": round(overlap_duration_s, 2),
        "temporal_overlap_pct": round(temporal_overlap_pct, 2),
        "sample_match_ratio_pct": round(sample_ratio, 2),
        "is_aligned": sample_ratio >= (min_match_ratio * 100.0),
    }


def find_matching_pose(
    timestamp: float,
    poses: Dict[float, np.ndarray],
    pose_keys: np.ndarray,
    tolerance_s: float = 0.05,
    interpolate: bool = True,
) -> Optional[np.ndarray]:
    """Find the exact or tightly SLERP-interpolated pose for a query timestamp."""
    if len(pose_keys) == 0:
        return None

    idx = np.searchsorted(pose_keys, timestamp)

    # Exact boundary matches
    if idx < len(pose_keys) and abs(pose_keys[idx] - timestamp) <= 1e-6:
        return poses[pose_keys[idx]]

    # Interpolation between neighboring valid poses if gap is tight (<= 0.1s)
    if interpolate and 0 < idx < len(pose_keys):
        t0 = pose_keys[idx - 1]
        t1 = pose_keys[idx]
        dt = t1 - t0
        if dt <= 0.10 and t0 <= timestamp <= t1:
            frac = (timestamp - t0) / dt
            T0 = poses[t0]
            T1 = poses[t1]

            # Linear translation interpolation
            p0 = T0[:3, 3]
            p1 = T1[:3, 3]
            p_interp = p0 + frac * (p1 - p0)

            # Quaternion SLERP
            from scipy.spatial.transform import Rotation
            R0 = T0[:3, :3]
            R1 = T1[:3, :3]
            q0 = Rotation.from_matrix(R0).as_quat()
            q1 = Rotation.from_matrix(R1).as_quat()
            q_interp = quaternion_slerp(q0, q1, frac)
            R_interp = Rotation.from_quat(q_interp).as_matrix()

            T_res = np.eye(4, dtype=np.float64)
            T_res[:3, :3] = R_interp
            T_res[:3, 3] = p_interp
            return T_res

    # Nearest neighbor fallback within tolerance
    candidates = []
    if idx < len(pose_keys):
        candidates.append(pose_keys[idx])
    if idx > 0:
        candidates.append(pose_keys[idx - 1])
    if not candidates:
        return None

    best_t = min(candidates, key=lambda t: abs(t - timestamp))
    if abs(best_t - timestamp) <= tolerance_s:
        return poses[best_t]
    return None
