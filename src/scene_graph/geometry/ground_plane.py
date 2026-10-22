"""Persistent ground and structural plane extraction, tracking, and alignment."""

from dataclasses import dataclass, field
from typing import List, Optional, Tuple
import numpy as np

from scene_graph.geometry.camera import CameraIntrinsics
from scene_graph.geometry.plane import fit_plane_ransac, PlaneGeometry
from scene_graph.geometry.spatial_surface import SpatialSurface
from scene_graph.geometry.reference_frame import RelationReferenceFrame, ReferenceFrameType
from scene_graph.geometry.transforms import invert_se3_transform


@dataclass
class GroundPlaneConfig:
    """Configuration for ground plane extraction and tracking."""
    distance_threshold_m: float = 0.04
    max_iterations: int = 150
    min_inliers: int = 60
    min_alignment_cosine: float = 0.85
    min_depth_m: float = 0.20
    max_depth_m: float = 8.0
    subsample_step: int = 16
    ema_alpha: float = 0.30
    max_angle_deg: float = 15.0
    max_offset_m: float = 0.06
    max_age_s: float = 5.0
    random_seed: int = 42


class GroundPlaneExtractor:
    """Extracts dominant horizontal ground/floor plane from RGB-D data."""

    def __init__(self, config: Optional[GroundPlaneConfig] = None):
        self.config = config or GroundPlaneConfig()

    def extract(
        self,
        depth_m: np.ndarray,
        intrinsics: CameraIntrinsics,
        world_T_camera: Optional[np.ndarray] = None,
        expected_up_axis: Optional[np.ndarray] = None,
        exclude_masks: Optional[List[np.ndarray]] = None,
        timestamp: float = 0.0,
    ) -> Optional[SpatialSurface]:
        """Extract dominant horizontal ground plane from depth image."""
        if depth_m is None or depth_m.ndim != 2:
            return None

        h, w = depth_m.shape
        step = max(1, self.config.subsample_step)

        # Build grid of pixel sample coordinates
        u_coords = np.arange(0, w, step)
        v_coords = np.arange(0, h, step)
        grid_u, grid_v = np.meshgrid(u_coords, v_coords)
        flat_u = grid_u.ravel()
        flat_v = grid_v.ravel()

        flat_z = depth_m[flat_v, flat_u].astype(np.float64)

        # Filter by valid depth range
        valid_mask = np.isfinite(flat_z) & (flat_z >= self.config.min_depth_m) & (flat_z <= self.config.max_depth_m)

        # Filter out pixels belonging to foreground object masks
        if exclude_masks:
            for mask in exclude_masks:
                if mask is not None and mask.shape == (h, w):
                    in_obj = mask[flat_v, flat_u] > 0
                    valid_mask = valid_mask & (~in_obj)

        valid_u = flat_u[valid_mask]
        valid_v = flat_v[valid_mask]
        valid_z = flat_z[valid_mask]

        if len(valid_z) < self.config.min_inliers:
            return None

        # Project sample pixels to 3D camera coordinates
        pts_cam = intrinsics.pixels_to_camera(valid_u, valid_v, valid_z)

        # Transform to world coordinates if camera pose is provided
        if world_T_camera is not None:
            R = world_T_camera[:3, :3]
            t = world_T_camera[:3, 3]
            pts_world = (pts_cam @ R.T) + t
        else:
            pts_world = pts_cam

        up_target = (
            np.asarray(expected_up_axis, dtype=np.float64)
            if expected_up_axis is not None
            else np.array([0.0, 0.0, 1.0], dtype=np.float64)
        )
        up_norm = np.linalg.norm(up_target)
        if up_norm > 1e-6:
            up_target = up_target / up_norm

        # Fit plane using RANSAC
        rng = np.random.default_rng(self.config.random_seed)
        plane = fit_plane_ransac(
            pts_world,
            distance_threshold=self.config.distance_threshold_m,
            max_iterations=self.config.max_iterations,
            min_inliers=self.config.min_inliers,
            rng=rng,
        )

        if plane is None or len(plane.inlier_mask) != len(pts_world):
            return None

        inliers = pts_world[plane.inlier_mask]
        if len(inliers) < self.config.min_inliers:
            return None

        normal = np.asarray(plane.normal, dtype=np.float64)
        n_norm = np.linalg.norm(normal)
        if n_norm < 1e-6:
            return None
        normal /= n_norm

        # Gravity alignment check: normal must align with expected up axis
        alignment = float(np.dot(normal, up_target))
        if abs(alignment) < self.config.min_alignment_cosine:
            return None

        # Ensure normal strictly points upward (positive dot product with up_target)
        dist = float(plane.distance)
        if alignment < 0.0:
            normal = -normal
            dist = -dist

        b_min = np.min(inliers, axis=0)
        b_max = np.max(inliers, axis=0)

        return SpatialSurface(
            surface_id="ground_surface",
            normal=normal,
            distance=dist,
            bounds_world=(b_min, b_max),
            last_observed_timestamp=timestamp,
            inlier_count=len(inliers),
            track_id="ground",
            plane_points=inliers,
        )


class GroundPlaneTracker:
    """Maintains a temporally smoothed, persistent ground plane over time."""

    def __init__(self, config: Optional[GroundPlaneConfig] = None):
        self.config = config or GroundPlaneConfig()
        self.extractor = GroundPlaneExtractor(self.config)
        self._tracked_surface: Optional[SpatialSurface] = None
        self._observation_count: int = 0

    @property
    def is_initialized(self) -> bool:
        return self._tracked_surface is not None

    @property
    def current_surface(self) -> Optional[SpatialSurface]:
        return self._tracked_surface

    def update(
        self,
        depth_m: np.ndarray,
        intrinsics: CameraIntrinsics,
        world_T_camera: Optional[np.ndarray] = None,
        expected_up_axis: Optional[np.ndarray] = None,
        exclude_masks: Optional[List[np.ndarray]] = None,
        timestamp: float = 0.0,
    ) -> Optional[SpatialSurface]:
        """Update tracker with new frame observations and return current ground plane."""
        candidate = self.extractor.extract(
            depth_m=depth_m,
            intrinsics=intrinsics,
            world_T_camera=world_T_camera,
            expected_up_axis=expected_up_axis,
            exclude_masks=exclude_masks,
            timestamp=timestamp,
        )

        if candidate is None:
            # Fallback: check if tracked surface is still valid within max_age_s
            if self._tracked_surface is not None:
                if (timestamp - self._tracked_surface.last_observed_timestamp) <= self.config.max_age_s:
                    return self._tracked_surface
                else:
                    self._tracked_surface = None
            return None

        if self._tracked_surface is None:
            # Initialize tracked surface
            self._tracked_surface = candidate
            self._observation_count = 1
            return self._tracked_surface

        # Check coplanarity with existing tracked ground
        if not self._tracked_surface.can_merge(
            candidate,
            max_angle_deg=self.config.max_angle_deg,
            max_offset_m=self.config.max_offset_m,
        ):
            # Potential outlier observation; if tracked surface is stale, re-initialize
            if (timestamp - self._tracked_surface.last_observed_timestamp) > self.config.max_age_s:
                self._tracked_surface = candidate
                self._observation_count = 1
                return self._tracked_surface
            return self._tracked_surface

        # EMA filter on normal vector and plane distance
        alpha = self.config.ema_alpha
        cos_angle = float(np.dot(self._tracked_surface.normal, candidate.normal))
        sign = 1.0 if cos_angle >= 0 else -1.0

        filtered_normal = (1.0 - alpha) * self._tracked_surface.normal + alpha * (candidate.normal * sign)
        norm_val = np.linalg.norm(filtered_normal)
        if norm_val > 1e-6:
            filtered_normal /= norm_val

        filtered_dist = float((1.0 - alpha) * self._tracked_surface.distance + alpha * (candidate.distance * sign))

        b_min = np.minimum(self._tracked_surface.bounds_world[0], candidate.bounds_world[0])
        b_max = np.maximum(self._tracked_surface.bounds_world[1], candidate.bounds_world[1])

        self._observation_count += 1
        self._tracked_surface = SpatialSurface(
            surface_id="ground_surface",
            normal=filtered_normal,
            distance=filtered_dist,
            bounds_world=(b_min, b_max),
            last_observed_timestamp=timestamp,
            inlier_count=self._tracked_surface.inlier_count + candidate.inlier_count,
            track_id="ground",
            plane_points=candidate.plane_points,
        )

        return self._tracked_surface

    def get_ground_plane(self, current_timestamp: float) -> Optional[SpatialSurface]:
        """Retrieve tracked ground plane if within max_age_s."""
        if self._tracked_surface is None:
            return None
        if (current_timestamp - self._tracked_surface.last_observed_timestamp) > self.config.max_age_s:
            return None
        return self._tracked_surface


def create_ground_aligned_reference_frame(
    ground_surface: SpatialSurface,
    initial_heading: Optional[np.ndarray] = None,
) -> RelationReferenceFrame:
    """Construct a canonical RelationReferenceFrame with up-axis aligned to the ground normal."""
    up = ground_surface.normal.copy()
    up /= np.linalg.norm(up)

    heading = (
        np.asarray(initial_heading, dtype=np.float64)
        if initial_heading is not None
        else np.array([1.0, 0.0, 0.0], dtype=np.float64)
    )
    # Project heading onto ground plane: h_proj = h - (h . up) * up
    h_proj = heading - np.dot(heading, up) * up
    h_norm = np.linalg.norm(h_proj)
    if h_norm < 1e-6:
        arbitrary = np.array([0.0, 1.0, 0.0]) if abs(up[1]) < 0.9 else np.array([1.0, 0.0, 0.0])
        h_proj = arbitrary - np.dot(arbitrary, up) * up
        h_norm = np.linalg.norm(h_proj)

    depth = h_proj / h_norm
    right = np.cross(up, depth)
    right /= np.linalg.norm(right)

    # Origin located on the ground plane (closest point to origin is -d * n)
    origin_world = -ground_surface.distance * up

    return RelationReferenceFrame(
        origin_world=origin_world,
        up_axis_world=up,
        horizontal_axis_world=right,
        depth_axis_world=depth,
        frame_type=ReferenceFrameType.WORLD,
    )


def compute_ground_leveling_transform(ground_surface: SpatialSurface) -> np.ndarray:
    """Compute rigid 4x4 SE(3) transform that rotates and shifts world such that the ground plane is at Z = 0 with normal [0, 0, 1]."""
    up = ground_surface.normal.copy()
    up /= np.linalg.norm(up)
    target_up = np.array([0.0, 0.0, 1.0], dtype=np.float64)

    v = np.cross(up, target_up)
    c = float(np.dot(up, target_up))

    if c > 0.9999:
        R = np.eye(3)
    elif c < -0.9999:
        R = np.diag([1.0, -1.0, -1.0])
    else:
        s = float(np.linalg.norm(v))
        vx = np.array([
            [0.0, -v[2], v[1]],
            [v[2], 0.0, -v[0]],
            [-v[1], v[0], 0.0],
        ])
        R = np.eye(3) + vx + (vx @ vx) * ((1.0 - c) / (s ** 2))

    # Point on plane is p0 = -distance * up.
    # In leveled frame, p0_leveled = R @ p0.
    # We want p0_leveled[2] + t[2] = 0.
    p0 = -ground_surface.distance * up
    p0_rot = R @ p0

    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R
    T[:3, 3] = -p0_rot

    return T
