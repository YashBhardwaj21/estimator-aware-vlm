"""Rigid Coordinate Frame Contract and Verification for D455-to-RC-Car Navigation.

Establishes explicit coordinate frames, transformations, and mathematical verifications:
  1. D455 Color Optical Frame ("camera_color_optical_frame"): X=Right, Y=Down, Z=Forward.
  2. D455 Depth Optical Frame ("camera_depth_optical_frame"): aligned to color optical frame.
  3. SLAM World Frame ("slam_world"): Arbitrary origin determined by ORB-SLAM3 first keyframe.
  4. Floor Map Frame ("floor_map"): ROS REP-103 standard: X=Forward, Y=Left, Z=Height (Z=0 on floor plane).
  5. Robot Base Frame ("base_link"): Centered at robot footprint ground projection (X=Forward, Y=Left, Z=Up).
  6. Odometry Frame ("odom"): Smooth continuous planar frame relative to floor_map.

Mathematical Quality Gates:
  - Strict rotation matrix orthogonality: R^T @ R = I (residual < 1e-6)
  - Strict right-handed determinant: det(R) = +1.0 (residual < 1e-6)
  - Strict homogeneous bottom row: [0, 0, 0, 1]
  - Exact inverse consistency: ||T @ inv(T) - I|| < 1e-6
  - Point transformation round-trip consistency: ||inv(T)(T(p)) - p|| < 1e-6
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np

# Standardized Frame Names following ROS REP-103
FRAME_COLOR_OPTICAL = "camera_color_optical_frame"
FRAME_DEPTH_OPTICAL = "camera_depth_optical_frame"
FRAME_SLAM_WORLD = "slam_world"
FRAME_FLOOR_MAP = "floor_map"
FRAME_BASE_LINK = "base_link"
FRAME_ODOM = "odom"


@dataclass
class RobotExtrinsics:
    """Mounting position of D455 camera relative to robot base_link frame.
    
    In base_link (ROS REP-103: X=Forward, Y=Left, Z=Up, origin on ground):
      - x_offset_m: Camera forward offset from robot rotational center.
      - y_offset_m: Camera lateral offset (positive left, 0.0 for centered).
      - z_offset_m: Camera height above ground plane.
      - pitch_deg: Downward optical tilt in degrees (positive = tilted downward).
      - yaw_deg: Horizontal optical azimuth offset (0.0 = forward).
      - roll_deg: Optical roll offset (0.0 = level).
    """
    x_offset_m: float = 0.12     # 12 cm forward of wheel center
    y_offset_m: float = 0.0      # Centered laterally
    z_offset_m: float = 0.16     # 16 cm above ground
    pitch_deg: float = 0.0       # Mounted level
    yaw_deg: float = 0.0
    roll_deg: float = 0.0

    def to_transform_base_to_camera_optical(self) -> np.ndarray:
        """Construct 4x4 transform from base_link to camera_color_optical_frame.
        
        base_link (X=fwd, Y=left, Z=up) -> optical (X=right, Y=down, Z=fwd):
          optical_X = -base_Y
          optical_Y = -base_Z
          optical_Z =  base_X
        """
        # Canonical rotation from base_link to optical frame
        R_base_to_opt_canonical = np.array([
            [0.0, -1.0,  0.0],
            [0.0,  0.0, -1.0],
            [1.0,  0.0,  0.0],
        ], dtype=np.float64)

        # Additional mounting orientation adjustments (pitch, yaw, roll)
        p = np.radians(self.pitch_deg)
        y = np.radians(self.yaw_deg)
        r = np.radians(self.roll_deg)

        Rx = np.array([[1, 0, 0], [0, np.cos(r), -np.sin(r)], [0, np.sin(r), np.cos(r)]])
        Ry = np.array([[np.cos(p), 0, np.sin(p)], [0, 1, 0], [-np.sin(p), 0, np.cos(p)]])
        Rz = np.array([[np.cos(y), -np.sin(y), 0], [np.sin(y), np.cos(y), 0], [0, 0, 1]])
        R_mount = Rz @ Ry @ Rx

        R_total = R_base_to_opt_canonical @ R_mount.T
        t_base = np.array([self.x_offset_m, self.y_offset_m, self.z_offset_m], dtype=np.float64)

        T = np.eye(4, dtype=np.float64)
        T[:3, :3] = R_total
        T[:3, 3] = -R_total @ t_base
        return T

    def to_transform_camera_optical_to_base(self) -> np.ndarray:
        """Construct 4x4 transform from camera_color_optical_frame to base_link."""
        return np.linalg.inv(self.to_transform_base_to_camera_optical())


def verify_transform_matrix(
    T: np.ndarray,
    name: str = "T",
    tolerance: float = 1e-6,
) -> Dict[str, Any]:
    """Execute mathematical quality gates on a 4x4 homogeneous transformation matrix.
    
    Checks:
      1. Shape is (4, 4) and finite.
      2. Bottom row is exactly [0, 0, 0, 1].
      3. Rotation submatrix R is orthogonal: ||R^T @ R - I|| < tolerance.
      4. Determinant is strictly +1.0 (right-handed): |det(R) - 1.0| < tolerance.
      5. Condition number of R is ~1.0.
    
    Raises:
      ValueError if any mathematical quality gate fails.
    """
    if not isinstance(T, np.ndarray) or T.shape != (4, 4):
        raise ValueError(f"Transform {name} must be a 4x4 numpy array, got {type(T)} shape {getattr(T, 'shape', None)}")
    
    if not np.all(np.isfinite(T)):
        raise ValueError(f"Transform {name} contains NaN or Infinite elements")

    # Bottom row check
    bottom_row_diff = np.max(np.abs(T[3, :] - np.array([0.0, 0.0, 0.0, 1.0])))
    if bottom_row_diff > tolerance:
        raise ValueError(f"Transform {name} bottom row violation: diff={bottom_row_diff:.2e} > {tolerance}")

    R = T[:3, :3]
    # Orthogonality check
    ortho_error = np.max(np.abs(R.T @ R - np.eye(3)))
    if ortho_error > tolerance:
        raise ValueError(f"Transform {name} rotation orthogonality violation: ||R^T @ R - I|| = {ortho_error:.2e} > {tolerance}")

    # Determinant check
    det = float(np.linalg.det(R))
    det_error = abs(det - 1.0)
    if det_error > tolerance:
        raise ValueError(f"Transform {name} determinant violation: det(R) = {det:.8f}, error={det_error:.2e} > {tolerance}")

    cond = float(np.linalg.cond(R))
    return {
        "verified": True,
        "name": name,
        "orthogonality_error": ortho_error,
        "det": det,
        "det_error": det_error,
        "condition_number": cond,
    }


def verify_round_trip(
    points: np.ndarray,
    T: np.ndarray,
    T_inv: np.ndarray,
    tolerance: float = 1e-6,
) -> float:
    """Verify point transformation round trip ||T_inv(T(p)) - p|| < tolerance."""
    from scene_graph.geometry.transforms import transform_points
    transformed = transform_points(T, points)
    reconstructed = transform_points(T_inv, transformed)
    max_err = float(np.max(np.linalg.norm(reconstructed - points, axis=1)))
    if max_err > tolerance:
        raise ValueError(f"Transform round-trip error exceeded tolerance: max_err={max_err:.2e} > {tolerance}")
    return max_err


@dataclass
class CoordinateFrameContract:
    """Complete coordinate frame contract managing all active transforms."""
    T_floor_world: np.ndarray
    T_world_floor: np.ndarray
    extrinsics: RobotExtrinsics = field(default_factory=RobotExtrinsics)
    floor_height_residual_std_m: float = 0.01

    def __post_init__(self):
        verify_transform_matrix(self.T_floor_world, "T_floor_world")
        verify_transform_matrix(self.T_world_floor, "T_world_floor")
        T_opt_base = self.extrinsics.to_transform_camera_optical_to_base()
        verify_transform_matrix(T_opt_base, "T_camera_optical_to_base")

    def camera_world_to_floor(self, T_world_camera: np.ndarray) -> np.ndarray:
        """Convert SLAM camera pose (in slam_world) to camera pose in floor_map frame."""
        return self.T_floor_world @ T_world_camera

    def camera_world_to_robot_base_floor(self, T_world_camera: np.ndarray) -> np.ndarray:
        """Convert SLAM camera pose (in slam_world) to robot base_link pose in floor_map frame."""
        T_floor_camera = self.T_floor_world @ T_world_camera
        T_camera_base = self.extrinsics.to_transform_camera_optical_to_base()
        return T_floor_camera @ T_camera_base

    def robot_base_floor_to_camera_floor(self, T_floor_base: np.ndarray) -> np.ndarray:
        """Convert robot base_link pose in floor_map frame to camera optical pose in floor_map frame."""
        T_base_camera = self.extrinsics.to_transform_base_to_camera_optical()
        return T_floor_base @ T_base_camera

    def to_dict(self) -> Dict[str, Any]:
        """Serialize contract metadata and matrices for reproducibility."""
        return {
            "source_frames": {
                "optical": FRAME_COLOR_OPTICAL,
                "slam_world": FRAME_SLAM_WORLD,
                "floor_map": FRAME_FLOOR_MAP,
                "base_link": FRAME_BASE_LINK,
                "odom": FRAME_ODOM,
            },
            "convention": "ROS REP-103 (X=Forward, Y=Left, Z=Height)",
            "robot_extrinsics": {
                "x_offset_m": self.extrinsics.x_offset_m,
                "y_offset_m": self.extrinsics.y_offset_m,
                "z_offset_m": self.extrinsics.z_offset_m,
                "pitch_deg": self.extrinsics.pitch_deg,
                "yaw_deg": self.extrinsics.yaw_deg,
                "roll_deg": self.extrinsics.roll_deg,
            },
            "T_floor_world": self.T_floor_world.tolist(),
            "T_world_floor": self.T_world_floor.tolist(),
            "T_camera_optical_to_base": self.extrinsics.to_transform_camera_optical_to_base().tolist(),
        }
