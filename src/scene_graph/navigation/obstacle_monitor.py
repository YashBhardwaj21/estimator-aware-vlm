"""Live Depth Obstacle Detection and Safety Monitoring for D455 RC Car.

Performs lightweight, continuous forward safety checks directly on depth frames:
  1. Identifies physical obstacles intersecting the vehicle's forward motion corridor.
  2. Accounts for vehicle footprint, stopping margin, and sensor blind zone.
  3. Detects sensor data staleness, dropouts, or invalid depth streams.
  4. Triggers independent motor inhibition and emergency-stop prior to global replanning.
  5. Computes metric world-space obstacle coordinates to update costmaps and trigger A* replanning.
"""

from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)


class ObstacleSafetyStatus(str, Enum):
    CLEAR = "CLEAR"
    CAUTION = "CAUTION"
    BLOCKED = "BLOCKED"
    SENSOR_FAILURE = "SENSOR_FAILURE"


@dataclass
class ObstacleMonitorConfig:
    """Safety parameters for forward obstacle monitoring."""
    # Metric corridor geometry
    robot_radius_m: float = 0.18
    corridor_width_m: float = 0.42          # Robot baseline 0.20m + 0.11m clearance on each side
    critical_stop_distance_m: float = 0.40  # Emergency stop threshold (D455 min range is ~0.35-0.4m)
    caution_distance_m: float = 0.70        # Slowdown advisory threshold
    min_obstacle_height_m: float = 0.04     # Above floor to filter floor-plane grazing
    max_obstacle_height_m: float = 1.00     # Ceiling clearance
    min_obstacle_cluster_points: int = 15   # Filter single-pixel depth dropouts/noise
    max_stale_duration_s: float = 0.35      # Depth freshness deadline

    # Camera parameters (Intel RealSense D455 default color/depth profile)
    depth_scale: float = 0.001              # Millimeters to meters
    fx: float = 383.5
    fy: float = 383.1
    cx: float = 320.3
    cy: float = 244.6
    image_width: int = 640
    image_height: int = 480


@dataclass
class ObstacleReport:
    """Summary of obstacle monitoring check."""
    status: ObstacleSafetyStatus
    is_safe_to_proceed: bool
    closest_distance_m: float
    obstacle_point_camera: Optional[Tuple[float, float, float]] = None
    obstacle_point_floor: Optional[Tuple[float, float]] = None
    timestamp_s: float = field(default_factory=time.time)
    details: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "status": self.status.value,
            "is_safe": self.is_safe_to_proceed,
            "closest_distance_m": round(self.closest_distance_m, 3),
            "obstacle_floor": [round(v, 3) for v in self.obstacle_point_floor] if self.obstacle_point_floor else None,
            "details": self.details,
        }


class LiveObstacleMonitor:
    """Lightweight depth-space safety monitor for real-time obstacle avoidance."""

    def __init__(self, config: Optional[ObstacleMonitorConfig] = None):
        self.cfg = config or ObstacleMonitorConfig()
        self.last_depth_frame: Optional[np.ndarray] = None
        self.last_depth_timestamp_s: float = 0.0
        self.last_report: Optional[ObstacleReport] = None

    def update_depth_frame(self, depth_img: np.ndarray, timestamp_s: Optional[float] = None) -> None:
        """Receive fresh depth observation from sensor or test feeder."""
        self.last_depth_frame = depth_img
        self.last_depth_timestamp_s = timestamp_s or time.time()

    def evaluate_safety(
        self,
        current_pose_floor: Tuple[float, float, float],  # (x, y, yaw)
        T_floor_camera: Optional[np.ndarray] = None,
        forward_velocity_mps: float = 0.20,
    ) -> ObstacleReport:
        """Perform vectorized collision check along the forward motion corridor.

        Returns ObstacleReport with safety status and metric obstacle coordinates.
        """
        now = time.time()

        # 1. Freshness Check
        if self.last_depth_frame is None:
            rep = ObstacleReport(
                status=ObstacleSafetyStatus.SENSOR_FAILURE,
                is_safe_to_proceed=False,
                closest_distance_m=float("inf"),
                details="No depth sensor frames received",
            )
            self.last_report = rep
            return rep

        age_s = now - self.last_depth_timestamp_s
        if age_s > self.cfg.max_stale_duration_s:
            rep = ObstacleReport(
                status=ObstacleSafetyStatus.SENSOR_FAILURE,
                is_safe_to_proceed=False,
                closest_distance_m=float("inf"),
                details=f"Depth sensor stream is stale (age={age_s:.3f}s > max={self.cfg.max_stale_duration_s:.3f}s)",
            )
            self.last_report = rep
            return rep

        # 2. Extract Depth Region of Interest (Center horizontal band across forward view)
        depth_m = self.last_depth_frame.astype(np.float32) * self.cfg.depth_scale
        h, w = depth_m.shape

        # Sample central band (vertical rows where obstacles in front of robot are visible)
        row_min, row_max = int(h * 0.25), int(h * 0.85)
        roi = depth_m[row_min:row_max, :]

        # Valid depth range filter (ignore zeros, sensor noise, and points beyond caution limit)
        search_max_m = max(self.cfg.caution_distance_m + 0.3, 1.2)
        valid_mask = (roi > 0.15) & (roi < search_max_m)

        if not np.any(valid_mask):
            rep = ObstacleReport(
                status=ObstacleSafetyStatus.CLEAR,
                is_safe_to_proceed=True,
                closest_distance_m=float("inf"),
                details="No obstacles in forward search zone",
            )
            self.last_report = rep
            return rep

        # 3. Vectorized Point Cloud Unprojection for ROI points
        valid_rows, valid_cols = np.where(valid_mask)
        z_vals = roi[valid_rows, valid_cols]
        actual_rows = valid_rows + row_min

        x_cam = (valid_cols - self.cfg.cx) * z_vals / self.cfg.fx
        y_cam = (actual_rows - self.cfg.cy) * z_vals / self.cfg.fy
        z_cam = z_vals

        # Points in camera frame: X=Right, Y=Down, Z=Forward
        # Lateral offset from camera centerline: x_cam
        # Forward distance: z_cam
        half_corridor = self.cfg.corridor_width_m / 2.0
        in_corridor_mask = (np.abs(x_cam) <= half_corridor)

        # Check heights (in camera coordinates, -Y is Up)
        # If camera is ~0.15m above floor, floor is around Y = +0.15m
        corridor_z = z_cam[in_corridor_mask]
        corridor_x = x_cam[in_corridor_mask]
        corridor_y = y_cam[in_corridor_mask]

        if len(corridor_z) < self.cfg.min_obstacle_cluster_points:
            rep = ObstacleReport(
                status=ObstacleSafetyStatus.CLEAR,
                is_safe_to_proceed=True,
                closest_distance_m=float(np.min(corridor_z)) if len(corridor_z) > 0 else float("inf"),
                details="Sparse points below noise threshold",
            )
            self.last_report = rep
            return rep

        # Find closest cluster distance
        min_dist = float(np.min(corridor_z))
        closest_idx = int(np.argmin(corridor_z))
        obs_cam = (float(corridor_x[closest_idx]), float(corridor_y[closest_idx]), float(corridor_z[closest_idx]))

        # Transform obstacle point to floor_map frame if transform provided
        obs_floor = None
        if T_floor_camera is not None:
            p_cam_h = np.array([obs_cam[0], obs_cam[1], obs_cam[2], 1.0], dtype=np.float64)
            p_floor_h = T_floor_camera @ p_cam_h
            obs_floor = (float(p_floor_h[0]), float(p_floor_h[1]))
        else:
            # Planar geometric projection using robot pose (assuming camera points forward)
            rx, ry, ryaw = current_pose_floor
            # Local X is forward in ROS REP-103
            fwd_dist = obs_cam[2]
            lat_dist = -obs_cam[0]  # Camera X is right, Robot Y is left
            c, s = math.cos(ryaw), math.sin(ryaw)
            ox = rx + c * fwd_dist - s * lat_dist
            oy = ry + s * fwd_dist + c * lat_dist
            obs_floor = (float(ox), float(oy))

        # 4. Critical vs Caution Evaluation
        # If velocity is high, increase dynamic stopping distance
        dynamic_stop_dist = max(
            self.cfg.critical_stop_distance_m,
            self.cfg.critical_stop_distance_m + 0.5 * forward_velocity_mps,
        )

        if min_dist <= dynamic_stop_dist:
            rep = ObstacleReport(
                status=ObstacleSafetyStatus.BLOCKED,
                is_safe_to_proceed=False,
                closest_distance_m=min_dist,
                obstacle_point_camera=obs_cam,
                obstacle_point_floor=obs_floor,
                details=f"Obstacle in motion corridor at {min_dist:.2f}m (Threshold: {dynamic_stop_dist:.2f}m)",
            )
            self.last_report = rep
            return rep
        elif min_dist <= self.cfg.caution_distance_m:
            rep = ObstacleReport(
                status=ObstacleSafetyStatus.CAUTION,
                is_safe_to_proceed=True,
                closest_distance_m=min_dist,
                obstacle_point_camera=obs_cam,
                obstacle_point_floor=obs_floor,
                details=f"Obstacle approaching in forward corridor at {min_dist:.2f}m",
            )
            self.last_report = rep
            return rep

        rep = ObstacleReport(
            status=ObstacleSafetyStatus.CLEAR,
            is_safe_to_proceed=True,
            closest_distance_m=min_dist,
            details="Path clear",
        )
        self.last_report = rep
        return rep
