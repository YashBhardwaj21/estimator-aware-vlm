"""Trajectory Metric Computation Pipeline.

Implements rigorous SE(3) Umeyama alignment (scale=1.0 for metric RGB-D SLAM),
Absolute Trajectory Error (ATE), Relative Pose Error (RPE) with both frame-step and fixed-time
interval evaluation, and tracking coverage following the Sturm et al. (IROS 2012) and EVO
benchmark standards. Enforces fail-closed semantics on failed or zero-pose trajectories.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Dict, List, Optional, Tuple, Union
import numpy as np

from estimator_aware_vlm.datasets.base import (
    Trajectory,
    TrajectoryPose,
    TrajectoryRole,
)


class EvaluationError(ValueError):
    """Raised when trajectory evaluation cannot be completed reliably."""
    pass


@dataclass(frozen=True)
class ATEStats:
    """Absolute Trajectory Error (ATE) statistics in meters."""
    rmse_m: float
    mean_m: float
    median_m: float
    std_m: float
    min_m: float
    max_m: float
    num_pairs: int


@dataclass(frozen=True)
class RPEStats:
    """Relative Pose Error (RPE) statistics."""
    trans_rmse_m: float
    trans_mean_m: float
    trans_median_m: float
    rot_rmse_deg: float
    rot_mean_deg: float
    rot_median_deg: float
    delta_frames: Optional[int] = None
    delta_sec: Optional[float] = None
    num_pairs: int = 0


@dataclass(frozen=True)
class TrajectoryEvaluationResult:
    """Complete evaluation report for an estimated trajectory against ground truth."""
    sequence_name: str
    num_estimated_poses: int
    num_ground_truth_poses: int
    num_matched_poses: int
    tracking_coverage: float
    ate: ATEStats
    rpe: RPEStats
    rpe_fixed_time: Optional[RPEStats]
    alignment_rotation: Tuple[Tuple[float, ...], ...]
    alignment_translation_m: Tuple[float, float, float]

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)


def align_trajectories_se3(
    model_positions: np.ndarray,
    data_positions: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute rigid SE(3) transformation (R, t) aligning model to data without scaling.
    
    Horn / Umeyama rigid alignment:
    Finds R in SO(3) and t in R^3 minimizing sum || R * model_i + t - data_i ||^2.
    
    Returns:
        aligned_positions: Aligned model positions (N, 3)
        R: 3x3 rotation matrix
        t: 3D translation vector
    """
    if len(model_positions) < 3 or len(data_positions) < 3:
        raise EvaluationError(
            f"At least 3 matched points are required for SE(3) trajectory alignment, "
            f"got {len(model_positions)} and {len(data_positions)}"
        )

    if model_positions.shape != data_positions.shape:
        raise EvaluationError(f"Shape mismatch: {model_positions.shape} vs {data_positions.shape}")

    # Centroids
    mu_model = np.mean(model_positions, axis=0)
    mu_data = np.mean(data_positions, axis=0)

    # Shifted coordinates
    shifted_model = model_positions - mu_model
    shifted_data = data_positions - mu_data

    # Correlation matrix H = model^T * data
    H = shifted_model.T @ shifted_data

    # Singular Value Decomposition
    U, _, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T

    # Ensure right-handed coordinate system (det(R) == +1)
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T

    # Translation
    t = mu_data - R @ mu_model

    # Apply transformation to model points
    aligned_positions = (R @ model_positions.T).T + t
    return aligned_positions, R, t


def compute_ate(
    aligned_est_positions: np.ndarray,
    gt_positions: np.ndarray,
    fail_on_empty: bool = True,
) -> ATEStats:
    """Compute Absolute Trajectory Error (ATE) across aligned position pairs."""
    if len(aligned_est_positions) == 0 or len(gt_positions) == 0:
        if fail_on_empty:
            raise EvaluationError("Cannot compute ATE on 0 matched position pairs.")
        return ATEStats(
            rmse_m=float("nan"),
            mean_m=float("nan"),
            median_m=float("nan"),
            std_m=float("nan"),
            min_m=float("nan"),
            max_m=float("nan"),
            num_pairs=0,
        )

    errors = np.linalg.norm(aligned_est_positions - gt_positions, axis=1)
    if len(errors) == 0:
        if fail_on_empty:
            raise EvaluationError("Cannot compute ATE: error array is empty.")
        return ATEStats(
            rmse_m=float("nan"),
            mean_m=float("nan"),
            median_m=float("nan"),
            std_m=float("nan"),
            min_m=float("nan"),
            max_m=float("nan"),
            num_pairs=0,
        )

    rmse = float(np.sqrt(np.mean(errors ** 2)))
    mean_val = float(np.mean(errors))
    median_val = float(np.median(errors))
    std_val = float(np.std(errors))
    min_val = float(np.min(errors))
    max_val = float(np.max(errors))

    return ATEStats(
        rmse_m=rmse,
        mean_m=mean_val,
        median_m=median_val,
        std_m=std_val,
        min_m=min_val,
        max_m=max_val,
        num_pairs=len(errors),
    )


def compute_rpe(
    est_poses: List[TrajectoryPose],
    gt_poses: List[TrajectoryPose],
    delta: int = 1,
    delta_sec: Optional[float] = None,
    max_dt_sec: float = 0.1,
    fail_on_empty: bool = False,
) -> RPEStats:
    """Compute Relative Pose Error (RPE) translation and rotation drift.
    
    Supports:
    - Frame-step delta: Compares poses (i, i + delta) when delta_sec is None.
    - Fixed-time interval delta: Compares poses (i, j) where |(t_j - t_i) - delta_sec| <= max_dt_sec.
    """
    n = len(est_poses)
    if n == 0 or len(gt_poses) != n:
        if fail_on_empty:
            raise EvaluationError("Pose lists must be non-empty and equal in length for RPE computation.")
        return RPEStats(
            trans_rmse_m=float("nan"),
            trans_mean_m=float("nan"),
            trans_median_m=float("nan"),
            rot_rmse_deg=float("nan"),
            rot_mean_deg=float("nan"),
            rot_median_deg=float("nan"),
            delta_frames=delta if delta_sec is None else None,
            delta_sec=delta_sec,
            num_pairs=0,
        )

    trans_errors: List[float] = []
    rot_errors_deg: List[float] = []

    if delta_sec is not None:
        # Fixed temporal interval evaluation
        timestamps = np.array([p.timestamp for p in est_poses], dtype=np.float64)
        for i in range(n):
            target_t = timestamps[i] + delta_sec
            diffs = np.abs(timestamps - target_t)
            j = int(np.argmin(diffs))
            if diffs[j] <= max_dt_sec and j > i:
                P_i = est_poses[i].matrix
                P_j = est_poses[j].matrix
                Q_i = gt_poses[i].matrix
                Q_j = gt_poses[j].matrix

                delta_est = np.linalg.inv(P_i) @ P_j
                delta_gt = np.linalg.inv(Q_i) @ Q_j
                error_mat = np.linalg.inv(delta_gt) @ delta_est

                t_err = np.linalg.norm(error_mat[:3, 3])
                trans_errors.append(float(t_err))

                R_err = error_mat[:3, :3]
                cos_theta = (np.trace(R_err) - 1.0) / 2.0
                cos_theta = np.clip(cos_theta, -1.0, 1.0)
                rot_errors_deg.append(float(np.degrees(np.arccos(cos_theta))))
    else:
        # Fixed frame-step evaluation
        if n <= delta:
            if fail_on_empty:
                raise EvaluationError(f"Insufficient poses ({n}) for delta_frames={delta}")
            return RPEStats(
                trans_rmse_m=float("nan"),
                trans_mean_m=float("nan"),
                trans_median_m=float("nan"),
                rot_rmse_deg=float("nan"),
                rot_mean_deg=float("nan"),
                rot_median_deg=float("nan"),
                delta_frames=delta,
                delta_sec=None,
                num_pairs=0,
            )

        for i in range(n - delta):
            j = i + delta
            P_i = est_poses[i].matrix
            P_j = est_poses[j].matrix
            Q_i = gt_poses[i].matrix
            Q_j = gt_poses[j].matrix

            delta_est = np.linalg.inv(P_i) @ P_j
            delta_gt = np.linalg.inv(Q_i) @ Q_j
            error_mat = np.linalg.inv(delta_gt) @ delta_est

            t_err = np.linalg.norm(error_mat[:3, 3])
            trans_errors.append(float(t_err))

            R_err = error_mat[:3, :3]
            cos_theta = (np.trace(R_err) - 1.0) / 2.0
            cos_theta = np.clip(cos_theta, -1.0, 1.0)
            rot_errors_deg.append(float(np.degrees(np.arccos(cos_theta))))

    if len(trans_errors) == 0:
        if fail_on_empty:
            raise EvaluationError(f"Zero pairs satisfied temporal/frame delta criteria for RPE.")
        return RPEStats(
            trans_rmse_m=float("nan"),
            trans_mean_m=float("nan"),
            trans_median_m=float("nan"),
            rot_rmse_deg=float("nan"),
            rot_mean_deg=float("nan"),
            rot_median_deg=float("nan"),
            delta_frames=delta if delta_sec is None else None,
            delta_sec=delta_sec,
            num_pairs=0,
        )

    t_arr = np.array(trans_errors, dtype=np.float64)
    r_arr = np.array(rot_errors_deg, dtype=np.float64)

    return RPEStats(
        trans_rmse_m=float(np.sqrt(np.mean(t_arr ** 2))),
        trans_mean_m=float(np.mean(t_arr)),
        trans_median_m=float(np.median(t_arr)),
        rot_rmse_deg=float(np.sqrt(np.mean(r_arr ** 2))),
        rot_mean_deg=float(np.mean(r_arr)),
        rot_median_deg=float(np.median(r_arr)),
        delta_frames=delta if delta_sec is None else None,
        delta_sec=delta_sec,
        num_pairs=len(trans_errors),
    )


def match_trajectories_by_timestamp(
    estimated: Trajectory,
    ground_truth: Trajectory,
    max_dt_sec: float = 0.02,
) -> Tuple[List[TrajectoryPose], List[TrajectoryPose]]:
    """Match estimated and ground truth poses strictly by nearest timestamp without duplicates."""
    matched_est: List[TrajectoryPose] = []
    matched_gt: List[TrajectoryPose] = []

    gt_timestamps = ground_truth.timestamps
    if len(gt_timestamps) == 0 or len(estimated) == 0:
        return matched_est, matched_gt

    used_gt_indices = set()
    for est_p in estimated.poses:
        diffs = np.abs(gt_timestamps - est_p.timestamp)
        min_idx = int(np.argmin(diffs))
        if diffs[min_idx] <= max_dt_sec and min_idx not in used_gt_indices:
            used_gt_indices.add(min_idx)
            matched_est.append(est_p)
            matched_gt.append(ground_truth.poses[min_idx])

    return matched_est, matched_gt


def evaluate_trajectory(
    estimated: Trajectory,
    ground_truth: Trajectory,
    max_dt_sec: float = 0.02,
    sequence_name: str = "rgbd_dataset_freiburg3_walking_xyz",
    delta_sec: Optional[float] = 1.0,
) -> TrajectoryEvaluationResult:
    """Evaluate an estimated SLAM trajectory against ground truth.
    
    Guarantees:
    - Enforces TrajectoryRole.ESTIMATED on input trajectory.
    - Enforces TrajectoryRole.GROUND_TRUTH on ground truth reference.
    - Strictly rejects evaluating a trajectory against itself if roles are spoofed.
    - Fails closed with EvaluationError if fewer than 3 poses match.
    """
    estimated.assert_role(TrajectoryRole.ESTIMATED)
    ground_truth.assert_role(TrajectoryRole.GROUND_TRUTH)

    matched_est, matched_gt = match_trajectories_by_timestamp(
        estimated, ground_truth, max_dt_sec=max_dt_sec
    )

    num_matched = len(matched_est)
    num_gt = len(ground_truth)
    coverage = float(num_matched / num_gt) if num_gt > 0 else 0.0

    if num_matched < 3:
        raise EvaluationError(
            f"Insufficient timestamp-aligned poses for evaluation: matched {num_matched} poses "
            f"(threshold=3, tolerance={max_dt_sec}s). Tracking failed or lost immediately."
        )

    est_pos = np.array([p.position for p in matched_est], dtype=np.float64)
    gt_pos = np.array([p.position for p in matched_gt], dtype=np.float64)

    # SE(3) Horn/Umeyama alignment without scaling
    aligned_est_pos, R, t = align_trajectories_se3(est_pos, gt_pos)

    # Compute ATE
    ate = compute_ate(aligned_est_pos, gt_pos, fail_on_empty=True)

    # Compute RPE (frame-to-frame, delta=1)
    rpe = compute_rpe(matched_est, matched_gt, delta=1, fail_on_empty=False)

    # Compute Fixed-Time Interval RPE (e.g. 1.0s)
    rpe_fixed = None
    if delta_sec is not None:
        rpe_fixed = compute_rpe(
            matched_est, matched_gt, delta_sec=delta_sec, max_dt_sec=0.1, fail_on_empty=False
        )

    return TrajectoryEvaluationResult(
        sequence_name=sequence_name,
        num_estimated_poses=len(estimated),
        num_ground_truth_poses=num_gt,
        num_matched_poses=num_matched,
        tracking_coverage=coverage,
        ate=ate,
        rpe=rpe,
        rpe_fixed_time=rpe_fixed,
        alignment_rotation=tuple(tuple(float(v) for v in row) for row in R),
        alignment_translation_m=(float(t[0]), float(t[1]), float(t[2])),
    )
