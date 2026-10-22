"""Rigorous Floor Plane Estimation, Unified Transform Construction, and Dynamic Map Bounds."""

from typing import Dict, List, Optional, Tuple, Any
import numpy as np

from scene_graph.geometry.plane import fit_plane_ransac
from scene_graph.geometry.transforms import transform_points


def construct_floor_transform(n_floor: np.ndarray, d_floor: float) -> Tuple[np.ndarray, np.ndarray]:
    """Construct rigorous 4x4 T_floor_world transform following ROS REP-103 convention.
    
    In floor_map frame:
      - X is forward in floor plane (projection of camera optical axis)
      - Y is left in floor plane
      - Z is perpendicular UP from floor (Z=0 is the floor plane)
    """
    z_axis = n_floor / np.linalg.norm(n_floor)
    v_fwd = np.array([0.0, 0.0, 1.0])
    v_proj = v_fwd - np.dot(v_fwd, z_axis) * z_axis
    if np.linalg.norm(v_proj) < 1e-4:
        v_fwd = np.array([1.0, 0.0, 0.0])
        v_proj = v_fwd - np.dot(v_fwd, z_axis) * z_axis

    x_axis = v_proj / np.linalg.norm(v_proj)
    y_axis = np.cross(z_axis, x_axis)
    y_axis /= np.linalg.norm(y_axis)

    R_floor_world = np.vstack([x_axis, y_axis, z_axis])
    origin_floor_in_world = -d_floor * z_axis

    T_floor_world = np.eye(4, dtype=np.float64)
    T_floor_world[:3, :3] = R_floor_world
    T_floor_world[:3, 3] = -R_floor_world @ origin_floor_in_world

    T_world_floor = np.linalg.inv(T_floor_world)
    return T_floor_world, T_world_floor


def compute_dynamic_map_bounds(
    camera_positions_floor: np.ndarray,
    point_samples_floor: Optional[np.ndarray] = None,
    padding_m: float = 1.2,
    resolution_m: float = 0.05,
    min_span_m: float = 4.0,
) -> Tuple[float, float, float, float]:
    """Derive map bounds (origin_x, origin_y, width, height) dynamically from observed geometry.

    Ensures the occupancy grid adapts to the actual extent of any recording.

    Returns:
        (origin_x_m, origin_y_m, width_m, height_m)
    """
    coords_x = list(camera_positions_floor[:, 0])
    coords_y = list(camera_positions_floor[:, 1])

    if point_samples_floor is not None and len(point_samples_floor) > 0:
        # Include points within robot height band
        valid_z = (point_samples_floor[:, 2] >= 0.05) & (point_samples_floor[:, 2] <= 2.2)
        if np.sum(valid_z) > 10:
            sub = point_samples_floor[valid_z]
            coords_x.extend(sub[:, 0])
            coords_y.extend(sub[:, 1])

    min_x = float(np.min(coords_x)) - padding_m
    max_x = float(np.max(coords_x)) + padding_m
    min_y = float(np.min(coords_y)) - padding_m
    max_y = float(np.max(coords_y)) + padding_m

    # Ensure minimum physical span
    if (max_x - min_x) < min_span_m:
        mid_x = (min_x + max_x) / 2.0
        min_x = mid_x - min_span_m / 2.0
        max_x = mid_x + min_span_m / 2.0

    if (max_y - min_y) < min_span_m:
        mid_y = (min_y + max_y) / 2.0
        min_y = mid_y - min_span_m / 2.0
        max_y = mid_y + min_span_m / 2.0

    # Snap origin DOWN to nearest grid cell
    origin_x = np.floor(min_x / resolution_m) * resolution_m
    origin_y = np.floor(min_y / resolution_m) * resolution_m

    # Snap width and height UP to nearest grid cell
    width = np.ceil((max_x - origin_x) / resolution_m) * resolution_m
    height = np.ceil((max_y - origin_y) / resolution_m) * resolution_m

    return round(float(origin_x), 4), round(float(origin_y), 4), round(float(width), 4), round(float(height), 4)
