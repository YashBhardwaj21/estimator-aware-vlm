"""Object geometry utilities for point clouds."""

import numpy as np

from scene_graph.geometry.camera import CameraIntrinsics
from scene_graph.geometry.transforms import transform_points
from scene_graph.geometry.noise_model import DepthNoiseModel, IsotropicNoiseModel


from dataclasses import dataclass
from typing import Optional, Tuple, Dict
from enum import Enum

class GeometryStatus(str, Enum):
    VALID = "VALID"
    INSUFFICIENT_DEPTH = "INSUFFICIENT_DEPTH"
    INVALID_GEOMETRY = "INVALID_GEOMETRY"
    NO_DEPTH = "NO_DEPTH"
    NO_POSE = "NO_POSE"
    STALE_POSE = "STALE_POSE"


@dataclass
class ObjectGeometry:
    """Cached per-observation geometry."""
    points_camera: Optional[np.ndarray] = None
    points_world: Optional[np.ndarray] = None
    points_world_sampled: Optional[np.ndarray] = None
    centroid_camera: Optional[np.ndarray] = None
    centroid_world: Optional[np.ndarray] = None
    position_covariance_world: Optional[np.ndarray] = None
    point_covariance_world: Optional[np.ndarray] = None
    bbox_min_world: Optional[np.ndarray] = None
    bbox_max_world: Optional[np.ndarray] = None
    obb_center_world: Optional[np.ndarray] = None
    obb_axes_world: Optional[np.ndarray] = None
    obb_extents_world: Optional[np.ndarray] = None
    geometry_quality: float = 0.0
    depth_stats: Optional[Dict[str, float]] = None
    valid_point_count: int = 0
    status: GeometryStatus = GeometryStatus.VALID
    coordinate_frame: str = "camera"
    localization_status: str = "unlocalized"
    pose_timestamp: Optional[float] = None


def compute_object_geometry(
    mask: np.ndarray, 
    depth_m: np.ndarray, 
    intrinsics: CameraIntrinsics, 
    pose: Optional[np.ndarray] = None,
    min_valid_points: int = 30,
    mad_k: float = 3.5,
    voxel_size_m: float = 0.005,
    measurement_noise_std_m: float = 0.01,
    noise_model: Optional[DepthNoiseModel] = None,
    min_depth_m: float = 0.10,
    max_depth_m: float = 10.0,
    spatial_outlier_sigma: float = 3.0,
) -> ObjectGeometry | None:
    """Compute robust 3D geometry for an object, mitigating background leakage.
    
    Args:
        mask: 2D boolean mask.
        depth_m: 2D depth in meters.
        intrinsics: Camera parameters.
        pose: 4x4 SE(3) world_T_camera matrix.
        min_valid_points: Minimum number of valid points required.
        mad_k: Number of robust sigmas for MAD outlier rejection.
        voxel_size_m: Voxel size for downsampling.
        measurement_noise_std_m: Sensor measurement noise standard deviation in meters.
        noise_model: Optional sensor noise model for covariance estimation.
        min_depth_m: Minimum valid depth in meters (near clipping).
        max_depth_m: Maximum valid depth in meters (far clipping).
        spatial_outlier_sigma: Number of robust sigmas for 3D spatial outlier rejection.
        
    Returns:
        ObjectGeometry (or None if totally invalid input)
    """
    if mask.ndim != 2 or depth_m.ndim != 2:
        raise ValueError("Mask and depth must be 2D arrays")
    if mask.shape != depth_m.shape:
        raise ValueError(f"Shape mismatch: mask {mask.shape} != depth {depth_m.shape}")
        
    coord_frame = "world" if pose is not None else "camera"
    loc_status = "localized" if pose is not None else "unlocalized"

    # Valid depth mask (positive, finite, and strictly within sensor range)
    valid_depth = (
        (depth_m >= min_depth_m)
        & (depth_m <= max_depth_m)
        & (depth_m > 0.0)
        & np.isfinite(depth_m)
    )
    combined_mask = (mask > 0) & valid_depth
    
    if np.sum(combined_mask) < min_valid_points:
        return ObjectGeometry(
            status=GeometryStatus.INSUFFICIENT_DEPTH,
            coordinate_frame=coord_frame,
            localization_status=loc_status,
        )
        
    # Extract coordinates
    v, u = np.where(combined_mask)
    z_m = depth_m[v, u]
    
    # Robust depth filtering using Median Absolute Deviation (MAD)
    median = np.median(z_m)
    mad = np.median(np.abs(z_m - median))
    sigma_robust = 1.4826 * mad
    if sigma_robust < 1e-6:
        sigma_robust = 1e-6
    band_mask = np.abs(z_m - median) <= mad_k * sigma_robust
    
    if np.sum(band_mask) < min_valid_points:
        return ObjectGeometry(
            status=GeometryStatus.INSUFFICIENT_DEPTH,
            coordinate_frame=coord_frame,
            localization_status=loc_status,
        )
        
    u_filt = u[band_mask]
    v_filt = v[band_mask]
    z_filt = z_m[band_mask]
    
    # Project to camera coordinates using vectorized projection
    points_camera = intrinsics.pixels_to_camera(u_filt, v_filt, z_filt)
    
    # 3D spatial outlier filtering: reject points whose Euclidean distance
    # from the median centroid exceeds spatial_outlier_sigma * sigma_dist
    if len(points_camera) >= min_valid_points:
        median_pt = np.median(points_camera, axis=0)
        dists = np.linalg.norm(points_camera - median_pt, axis=1)
        med_dist = np.median(dists)
        mad_dist = np.median(np.abs(dists - med_dist))
        sigma_dist = 1.4826 * mad_dist
        if sigma_dist < 1e-6:
            sigma_dist = 1e-6
        spatial_mask = dists <= (med_dist + spatial_outlier_sigma * sigma_dist)
        if np.sum(spatial_mask) >= min_valid_points:
            points_camera = points_camera[spatial_mask]
            z_filt = z_filt[spatial_mask]
        else:
            return ObjectGeometry(
                status=GeometryStatus.INSUFFICIENT_DEPTH,
                coordinate_frame=coord_frame,
                localization_status=loc_status,
            )
    
    center_camera = np.mean(points_camera, axis=0)

    # Compute depth statistics
    depth_stats = {
        "median": float(np.median(z_filt)),
        "p05": float(np.percentile(z_filt, 5)),
        "p25": float(np.percentile(z_filt, 25)),
        "p75": float(np.percentile(z_filt, 75)),
        "p95": float(np.percentile(z_filt, 95))
    }

    if pose is None:
        N = len(points_camera)
        geometry_quality = 1.0 if N > min_valid_points * 2 else float(N) / (min_valid_points * 2)
        return ObjectGeometry(
            points_camera=points_camera,
            points_world=None,
            points_world_sampled=None,
            centroid_camera=center_camera,
            centroid_world=None,
            position_covariance_world=None,
            point_covariance_world=None,
            bbox_min_world=None,
            bbox_max_world=None,
            obb_center_world=None,
            obb_axes_world=None,
            obb_extents_world=None,
            geometry_quality=geometry_quality,
            depth_stats=depth_stats,
            valid_point_count=N,
            status=GeometryStatus.VALID,
            coordinate_frame="camera",
            localization_status="unlocalized",
        )

    # Transform to world coordinates
    points_world = transform_points(pose, points_camera)
    
    # Compute centers (mean, rigid-transform equivariant)
    N = len(points_world)
    center_world = np.mean(points_world, axis=0)
    
    # Covariance
    centered = points_world - center_world
    if N > 1:
        Sigma_P = np.dot(centered.T, centered) / (N - 1)
        Sigma_P = 0.5 * (Sigma_P + Sigma_P.T)
    else:
        Sigma_P = np.zeros((3, 3), dtype=np.float64)
        
    if noise_model is not None:
        Sigma_sensor = noise_model.estimate_covariance(
            points_camera, intrinsics, R_world_camera=pose[:3, :3]
        )
        Sigma_sensor = 0.5 * (Sigma_sensor + Sigma_sensor.T)
    else:
        Sigma_sensor = np.eye(3, dtype=np.float64) * (float(measurement_noise_std_m) ** 2)
    Sigma_c = (Sigma_P / N) + Sigma_sensor
    Sigma_c = 0.5 * (Sigma_c + Sigma_c.T)
    
    # Percentile AABB
    aabb_min = np.percentile(points_world, 2, axis=0)
    aabb_max = np.percentile(points_world, 98, axis=0)
    
    # OBB via PCA
    if N > 2:
        cov = np.dot(centered.T, centered) / N
        eigenvalues, eigenvectors = np.linalg.eigh(cov)
        idx = np.argsort(eigenvalues)[::-1]
        obb_axes = eigenvectors[:, idx]
        
        # Enforce right-handed coordinate frame for OBB axes: det(R) == +1.0
        if np.linalg.det(obb_axes) < 0.0:
            obb_axes[:, 2] *= -1.0
            
        projected = np.dot(centered, obb_axes)
        min_proj = np.min(projected, axis=0)
        max_proj = np.max(projected, axis=0)
        obb_extents = max_proj - min_proj
        obb_center = center_world + np.dot(obb_axes, (max_proj + min_proj) / 2)
    else:
        obb_axes = np.eye(3)
        obb_extents = np.zeros(3)
        obb_center = center_world
    
    # Deterministic voxel downsampling
    from scene_graph.geometry.downsampling import voxel_downsample
    points_world_sampled = voxel_downsample(points_world, voxel_size_m)
    
    # Quality heuristic
    geometry_quality = 1.0 if N > min_valid_points * 2 else float(N) / (min_valid_points * 2)
    
    return ObjectGeometry(
        points_camera=points_camera,
        points_world=points_world,
        points_world_sampled=points_world_sampled,
        centroid_camera=center_camera,
        centroid_world=center_world,
        position_covariance_world=Sigma_c,
        point_covariance_world=Sigma_P,
        bbox_min_world=aabb_min,
        bbox_max_world=aabb_max,
        obb_center_world=obb_center,
        obb_axes_world=obb_axes,
        obb_extents_world=obb_extents,
        geometry_quality=geometry_quality,
        depth_stats=depth_stats,
        valid_point_count=len(points_world),
        status=GeometryStatus.VALID,
        coordinate_frame="world",
        localization_status="localized",
    )
