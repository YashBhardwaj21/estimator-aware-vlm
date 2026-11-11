"""RGB-D Object Geometry Extraction.

Converts 2D instance segmentation masks and calibrated depth maps into
metric camera-frame and world-frame 3D observations:
- Inverse pinhole back-projection: P_cam = Z * K^{-1} * [u, v, 1]^T
- World coordinate transformation: P_world = R_wc * P_cam + t_wc (using ESTIMATED SLAM poses only)
- Mask-level depth sampling, robust dispersion (MAD, P10/P90), and boundary discontinuity
- 3D centroid covariance and spatial bounding dimensions (dx, dy, dz)
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np

from estimator_aware_vlm.datasets.base import (
    CameraIntrinsics,
    TrajectoryPose,
    TrajectoryRole,
)


@dataclass(frozen=True)
class ObjectGeometryResult:
    """Rich 3D geometric measurements derived from a single instance mask and depth."""
    # Camera-frame 3D centroid: [X, Y, Z] in meters
    centroid_camera: Tuple[float, float, float]
    # World-frame 3D centroid: [X, Y, Z] in meters (None if estimator pose unavailable)
    centroid_world: Optional[Tuple[float, float, float]]
    # World position validity flag (True only when valid ESTIMATED pose is applied)
    world_position_valid: bool
    # Coordinate-frame pose provenance: 'ESTIMATED_SLAM' or 'ESTIMATOR_LOST'
    world_pose_provenance: str

    # 3D bounding dimensions: [dx, dy, dz] in meters
    dimensions_3d_m: Tuple[float, float, float]
    # 3x3 centroid covariance matrix flattened to 9 floats (or empty if < 3 points)
    centroid_covariance_camera: List[float]

    # Robust depth statistics
    depth_median_m: float
    depth_mad_m: float
    depth_p10_m: float
    depth_p90_m: float
    valid_depth_fraction: float
    valid_points_count: int

    # Boundary depth discontinuity across mask edge in meters
    boundary_depth_discontinuity_m: float
    # Overall depth confidence in [0.0, 1.0]
    depth_confidence: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "centroid_camera": [round(float(x), 4) for x in self.centroid_camera],
            "centroid_world": [round(float(x), 4) for x in self.centroid_world]
            if self.centroid_world is not None else None,
            "world_position_valid": bool(self.world_position_valid),
            "world_pose_provenance": self.world_pose_provenance,
            "dimensions_3d_m": [round(float(x), 4) for x in self.dimensions_3d_m],
            "centroid_covariance_camera": [round(float(x), 6) for x in self.centroid_covariance_camera],
            "depth_median_m": round(float(self.depth_median_m), 4),
            "depth_mad_m": round(float(self.depth_mad_m), 4),
            "depth_p10_m": round(float(self.depth_p10_m), 4),
            "depth_p90_m": round(float(self.depth_p90_m), 4),
            "valid_depth_fraction": round(float(self.valid_depth_fraction), 4),
            "valid_points_count": int(self.valid_points_count),
            "boundary_depth_discontinuity_m": round(float(self.boundary_depth_discontinuity_m), 4),
            "depth_confidence": round(float(self.depth_confidence), 4),
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> ObjectGeometryResult:
        cw = d.get("centroid_world")
        return cls(
            centroid_camera=tuple(float(x) for x in d["centroid_camera"]),
            centroid_world=tuple(float(x) for x in cw) if cw is not None else None,
            world_position_valid=bool(d.get("world_position_valid", False)),
            world_pose_provenance=str(d.get("world_pose_provenance", "UNKNOWN")),
            dimensions_3d_m=tuple(float(x) for x in d.get("dimensions_3d_m", (0.0, 0.0, 0.0))),
            centroid_covariance_camera=[float(x) for x in d.get("centroid_covariance_camera", [])],
            depth_median_m=float(d.get("depth_median_m", 0.0)),
            depth_mad_m=float(d.get("depth_mad_m", 0.0)),
            depth_p10_m=float(d.get("depth_p10_m", 0.0)),
            depth_p90_m=float(d.get("depth_p90_m", 0.0)),
            valid_depth_fraction=float(d.get("valid_depth_fraction", 0.0)),
            valid_points_count=int(d.get("valid_points_count", 0)),
            boundary_depth_discontinuity_m=float(d.get("boundary_depth_discontinuity_m", 0.0)),
            depth_confidence=float(d.get("depth_confidence", 0.0)),
        )


def compute_boundary_depth_discontinuity(
    depth_m_full: np.ndarray,
    mask_binary: np.ndarray,
    kernel_size: int = 5,
    min_depth_m: float = 0.25,
    max_depth_m: float = 8.0,
) -> float:
    """Measure the median depth step across the object boundary.

    Compares the interior border of the mask with an exterior background ring.
    Returns the absolute depth difference in meters.
    """
    if not np.any(mask_binary):
        return 0.0

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
    # Interior boundary: mask minus eroded mask
    eroded = cv2.erode(mask_binary, kernel, iterations=1)
    inner_boundary = (mask_binary > 0) & (eroded == 0)

    # Exterior boundary: dilated mask minus original mask
    dilated = cv2.dilate(mask_binary, kernel, iterations=1)
    outer_boundary = (dilated > 0) & (mask_binary == 0)

    inner_depths = depth_m_full[inner_boundary]
    outer_depths = depth_m_full[outer_boundary]

    valid_inner = inner_depths[(inner_depths >= min_depth_m) & (inner_depths <= max_depth_m)]
    valid_outer = outer_depths[(outer_depths >= min_depth_m) & (outer_depths <= max_depth_m)]

    if len(valid_inner) == 0 or len(valid_outer) == 0:
        return 0.0

    median_inner = float(np.median(valid_inner))
    median_outer = float(np.median(valid_outer))
    return float(abs(median_outer - median_inner))


def extract_object_geometry(
    depth_img_raw: np.ndarray,
    depth_scale: float,
    bbox_xyxy: Tuple[int, int, int, int],
    intrinsics: CameraIntrinsics,
    mask_contour: Optional[List[List[int]]] = None,
    camera_pose: Optional[TrajectoryPose] = None,
    min_depth_m: float = 0.25,
    max_depth_m: float = 8.0,
) -> Optional[ObjectGeometryResult]:
    """Convert a single 2D detection and depth map into accurate 3D geometry.

    Parameters:
        depth_img_raw: Raw uint16/float depth map.
        depth_scale: Multiplier to convert raw integer depth to meters.
        bbox_xyxy: 2D bounding box (x1, y1, x2, y2) in pixel coordinates.
        intrinsics: Calibrated camera intrinsic parameters.
        mask_contour: Optional polygon contour vertices [[u1, v1], [u2, v2], ...].
        camera_pose: Optional estimated camera pose (strictly ESTIMATED role).
        min_depth_m: Minimum valid depth in meters.
        max_depth_m: Maximum valid depth in meters.

    Returns:
        ObjectGeometryResult with camera-frame and world-frame 3D coordinates and statistics,
        or None if insufficient valid depth exists.
    """
    img_h, img_w = depth_img_raw.shape[:2]
    x1, y1, x2, y2 = bbox_xyxy
    x1 = max(0, min(int(x1), img_w - 1))
    y1 = max(0, min(int(y1), img_h - 1))
    x2 = max(x1 + 1, min(int(x2), img_w))
    y2 = max(y1 + 1, min(int(y2), img_h))

    bw = x2 - x1
    bh = y2 - y1
    if bw <= 0 or bh <= 0:
        return None

    # Construct binary mask over bounding box ROI
    mask_roi = np.zeros((bh, bw), dtype=np.uint8)
    has_valid_polygon = False

    if mask_contour is not None and len(mask_contour) >= 3:
        # Shift contour coordinates to local ROI frame
        local_contour = np.array(
            [[int(pt[0]) - x1, int(pt[1]) - y1] for pt in mask_contour],
            dtype=np.int32,
        )
        cv2.fillPoly(mask_roi, [local_contour], 1)
        if np.any(mask_roi > 0):
            has_valid_polygon = True

    if not has_valid_polygon:
        # Fallback to full bounding box ROI
        mask_roi.fill(1)

    total_mask_pixels = int(np.count_nonzero(mask_roi))
    if total_mask_pixels == 0:
        return None

    # Extract depth within bounding box ROI
    depth_roi_raw = depth_img_raw[y1:y2, x1:x2]
    depth_roi_m = depth_roi_raw.astype(np.float64) * depth_scale

    # Sample strictly within mask and valid depth range
    in_mask = mask_roi > 0
    in_range = in_mask & (depth_roi_m >= min_depth_m) & (depth_roi_m <= max_depth_m)
    valid_count = int(np.count_nonzero(in_range))

    valid_depth_fraction = float(valid_count / max(1, total_mask_pixels))
    if valid_count < 3:
        # Insufficient depth points for 3D reconstruction
        return None

    # Metric depths for valid points
    valid_depths = depth_roi_m[in_range]
    depth_median = float(np.median(valid_depths))
    depth_mad = float(np.median(np.abs(valid_depths - depth_median)))
    depth_p10 = float(np.percentile(valid_depths, 10))
    depth_p90 = float(np.percentile(valid_depths, 90))

    # Grid of pixel coordinates (u, v) in full image space for valid mask points
    v_indices, u_indices = np.where(in_range)
    u_full = (u_indices + x1).astype(np.float64)
    v_full = (v_indices + y1).astype(np.float64)

    # 4. Inverse pinhole back-projection: P_cam = Z * K^{-1} [u, v, 1]^T
    x_cam = (u_full - intrinsics.cx) * valid_depths / intrinsics.fx
    y_cam = (v_full - intrinsics.cy) * valid_depths / intrinsics.fy
    z_cam = valid_depths
    pts_cam = np.column_stack((x_cam, y_cam, z_cam))  # Shape (N, 3)

    # 5. Object centroid in camera coordinates (coordinate-wise median for outlier robustness)
    c_x = float(np.median(x_cam))
    c_y = float(np.median(y_cam))
    c_z = float(np.median(z_cam))
    centroid_cam = (c_x, c_y, c_z)

    # Centroid covariance matrix in camera frame (3x3)
    if valid_count >= 3:
        cov_mat = np.cov(pts_cam, rowvar=False)  # Shape (3, 3)
        cov_flat = [float(x) for x in cov_mat.flatten()]
    else:
        cov_flat = [0.0] * 9

    # 6. Estimate 3D bounding dimensions (2nd to 98th percentile span to discard noise outliers)
    dx = float(np.percentile(x_cam, 98) - np.percentile(x_cam, 2))
    dy = float(np.percentile(y_cam, 98) - np.percentile(y_cam, 2))
    dz = float(np.percentile(z_cam, 98) - np.percentile(z_cam, 2))
    dimensions_3d = (max(0.01, dx), max(0.01, dy), max(0.01, dz))

    # Boundary depth discontinuity
    # Construct full image binary mask for boundary analysis
    full_mask = np.zeros((img_h, img_w), dtype=np.uint8)
    full_mask[y1:y2, x1:x2] = mask_roi
    depth_full_m = depth_img_raw.astype(np.float64) * depth_scale
    boundary_disc = compute_boundary_depth_discontinuity(
        depth_m_full=depth_full_m,
        mask_binary=full_mask,
        min_depth_m=min_depth_m,
        max_depth_m=max_depth_m,
    )

    # Depth confidence: composite metric in [0.0, 1.0]
    # Rewards high valid fraction, low MAD dispersion, and sufficient point support
    s_valid = min(1.0, valid_depth_fraction / 0.65)
    s_disp = math.exp(-min(5.0, depth_mad / 0.25))
    s_count = min(1.0, valid_count / 100.0)
    depth_confidence = float(s_valid * 0.4 + s_disp * 0.4 + s_count * 0.2)

    # 7. World coordinate transformation: P_world = R_wc * P_cam + t_wc
    centroid_world: Optional[Tuple[float, float, float]] = None
    world_valid = False
    world_provenance = "ESTIMATOR_LOST"

    if camera_pose is not None:
        # Enforce role integrity: ground truth must NOT enter perception features
        camera_pose.assert_role(TrajectoryRole.ESTIMATED)
        T_wc = camera_pose.matrix
        R_wc = T_wc[:3, :3]
        t_wc = T_wc[:3, 3]

        p_cam_arr = np.array([c_x, c_y, c_z], dtype=np.float64)
        p_w = R_wc @ p_cam_arr + t_wc
        if np.all(np.isfinite(p_w)):
            centroid_world = (float(p_w[0]), float(p_w[1]), float(p_w[2]))
            world_valid = True
            world_provenance = "ESTIMATED_SLAM"

    return ObjectGeometryResult(
        centroid_camera=centroid_cam,
        centroid_world=centroid_world,
        world_position_valid=world_valid,
        world_pose_provenance=world_provenance,
        dimensions_3d_m=dimensions_3d,
        centroid_covariance_camera=cov_flat,
        depth_median_m=depth_median,
        depth_mad_m=depth_mad,
        depth_p10_m=depth_p10,
        depth_p90_m=depth_p90,
        valid_depth_fraction=valid_depth_fraction,
        valid_points_count=valid_count,
        boundary_depth_discontinuity_m=boundary_disc,
        depth_confidence=depth_confidence,
    )

