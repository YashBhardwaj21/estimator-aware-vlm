"""Camera intrinsics and pixel projection."""

from dataclasses import dataclass
from typing import Tuple
import numpy as np


@dataclass(frozen=True)
class CameraIntrinsics:
    """Camera intrinsics (focal length, principal point, and image dimensions)."""
    fx: float
    fy: float
    cx: float
    cy: float
    width: int
    height: int
    distortion: Tuple[float, ...] = (0.0, 0.0, 0.0, 0.0, 0.0)
    distortion_model: str = "plumb_bob"

    def __post_init__(self):
        if self.fx <= 0 or self.fy <= 0:
            raise ValueError("Focal lengths (fx, fy) must be > 0")
        if self.width <= 0 or self.height <= 0:
            raise ValueError("Image dimensions (width, height) must be > 0")
        if not np.isfinite(self.cx) or not np.isfinite(self.cy):
            raise ValueError(f"Principal point (cx, cy) must be finite, got ({self.cx}, {self.cy})")
        if self.cx < 0.0 or self.cx > float(self.width) or self.cy < 0.0 or self.cy > float(self.height):
            raise ValueError(
                f"Principal point ({self.cx}, {self.cy}) lies outside image boundaries (0..{self.width}, 0..{self.height})"
            )

    def pixel_to_camera(self, u: float, v: float, depth_m: float) -> tuple[float, float, float]:
        """Project a 2D pixel with depth into 3D camera coordinates."""
        if not np.isfinite(depth_m) or depth_m <= 0:
            raise ValueError(f"Invalid depth: {depth_m}")

        x = (u - self.cx) * depth_m / self.fx
        y = (v - self.cy) * depth_m / self.fy
        z = depth_m
        return x, y, z

    def pixels_to_camera(self, u: np.ndarray, v: np.ndarray, depth_m: np.ndarray) -> np.ndarray:
        """Vectorized projection from 2D pixels with depth into 3D camera coordinates."""
        z = depth_m.astype(np.float64)
        x = (u - self.cx) * z / self.fx
        y = (v - self.cy) * z / self.fy
        return np.column_stack((x, y, z))

    def camera_to_pixel(self, x: float, y: float, z: float) -> tuple[float, float]:
        """Project a 3D point in camera coordinates into 2D pixel coordinates (u, v)."""
        if not np.isfinite(z) or z <= 0.0:
            raise ValueError(f"Invalid camera depth z: {z} (point must be strictly in front of camera)")
        u = (x * self.fx / z) + self.cx
        v = (y * self.fy / z) + self.cy
        return float(u), float(v)

    def cameras_to_pixels(self, points_camera: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Vectorized projection of (N, 3) camera points to (u, v) pixel coordinates."""
        pts = np.asarray(points_camera, dtype=np.float64)
        if pts.ndim != 2 or pts.shape[1] != 3:
            raise ValueError(f"points_camera must be shape (N, 3), got {pts.shape}")
        z = pts[:, 2]
        if np.any(z <= 0.0) or not np.all(np.isfinite(z)):
            raise ValueError("All points must have finite positive depth z > 0")
        u = (pts[:, 0] * self.fx / z) + self.cx
        v = (pts[:, 1] * self.fy / z) + self.cy
        return u, v

    def is_in_frustum(
        self,
        point_camera: np.ndarray,
        margin_px: float = 0.0,
        min_depth_m: float = 0.10,
        max_depth_m: float = 10.0,
    ) -> bool:
        """Checks if a 3D point in camera coordinates is within the camera viewing frustum."""
        pt = np.asarray(point_camera, dtype=np.float64)
        if pt.shape != (3,):
            raise ValueError(f"point_camera must have shape (3,), got {pt.shape}")
        x, y, z = pt[0], pt[1], pt[2]
        if not np.isfinite(z) or z < min_depth_m or z > max_depth_m:
            return False
        u = (x * self.fx / z) + self.cx
        v = (y * self.fy / z) + self.cy
        return bool(
            margin_px <= u <= (self.width - margin_px)
            and margin_px <= v <= (self.height - margin_px)
        )

    def points_in_frustum(
        self,
        points_camera: np.ndarray,
        margin_px: float = 0.0,
        min_depth_m: float = 0.10,
        max_depth_m: float = 10.0,
    ) -> np.ndarray:
        """Vectorized check returning boolean mask of shape (N,) indicating if points are in frustum."""
        pts = np.asarray(points_camera, dtype=np.float64)
        if pts.ndim != 2 or pts.shape[1] != 3:
            raise ValueError(f"points_camera must be shape (N, 3), got {pts.shape}")
        z = pts[:, 2]
        valid_z = np.isfinite(z) & (z >= min_depth_m) & (z <= max_depth_m)
        mask = np.zeros(len(pts), dtype=bool)
        if not np.any(valid_z):
            return mask
        z_safe = np.where(valid_z, z, 1.0)
        u = (pts[:, 0] * self.fx / z_safe) + self.cx
        v = (pts[:, 1] * self.fy / z_safe) + self.cy
        in_uv = (
            (u >= margin_px)
            & (u <= (self.width - margin_px))
            & (v >= margin_px)
            & (v <= (self.height - margin_px))
        )
        return valid_z & in_uv

    def is_world_point_in_frustum(
        self,
        point_world: np.ndarray,
        world_T_camera: np.ndarray,
        margin_px: float = 0.0,
        min_depth_m: float = 0.10,
        max_depth_m: float = 10.0,
    ) -> bool:
        """Checks if a 3D point in world coordinates is within the camera viewing frustum."""
        from scene_graph.geometry.transforms import invert_se3_transform
        camera_T_world = invert_se3_transform(world_T_camera)
        pt_w = np.asarray(point_world, dtype=np.float64)
        pt_cam = (camera_T_world[:3, :3] @ pt_w) + camera_T_world[:3, 3]
        return bool(
            self.is_in_frustum(
                pt_cam,
                margin_px=margin_px,
                min_depth_m=min_depth_m,
                max_depth_m=max_depth_m,
            )
        )

    def is_box_in_frustum(
        self,
        bbox_min_world: np.ndarray,
        bbox_max_world: np.ndarray,
        world_T_camera: np.ndarray,
        margin_px: float = 0.0,
        min_depth_m: float = 0.10,
        max_depth_m: float = 10.0,
    ) -> bool:
        """Checks if any corner or centroid of an AABB is within the viewing frustum."""
        b_min = np.asarray(bbox_min_world, dtype=np.float64)
        b_max = np.asarray(bbox_max_world, dtype=np.float64)
        corners = np.array([
            [b_min[0], b_min[1], b_min[2]],
            [b_min[0], b_min[1], b_max[2]],
            [b_min[0], b_max[1], b_min[2]],
            [b_min[0], b_max[1], b_max[2]],
            [b_max[0], b_min[0], b_min[2]],
            [b_max[0], b_min[1], b_max[2]],
            [b_max[0], b_max[1], b_min[2]],
            [b_max[0], b_max[1], b_max[2]],
            (b_min + b_max) * 0.5,
        ])
        for pt in corners:
            if self.is_world_point_in_frustum(
                pt,
                world_T_camera,
                margin_px=margin_px,
                min_depth_m=min_depth_m,
                max_depth_m=max_depth_m,
            ):
                return True
        return False


@dataclass(frozen=True)
class DepthModel:
    """Depth scaling model.
    
    Converts raw depth units to meters.
    """
    scale: float
    
    def __post_init__(self):
        if self.scale <= 0:
            raise ValueError("Scale must be > 0")
    
    def depth_to_meters(self, depth_raw: np.ndarray) -> np.ndarray:
        """Convert raw uint16 depth image to metric float depth."""
        return depth_raw.astype(np.float64) / self.scale

