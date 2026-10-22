from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple
import numpy as np

from scene_graph.geometry.plane import PlaneGeometry


@dataclass
class SpatialSurface:
    """Represents a persistent planar support surface in 3D world space."""

    surface_id: str
    normal: np.ndarray
    distance: float
    bounds_world: Tuple[np.ndarray, np.ndarray]
    last_observed_timestamp: float
    inlier_count: int
    track_id: Optional[str] = None
    plane_points: Optional[np.ndarray] = None

    @property
    def plane_equation(self) -> np.ndarray:
        """Hessian normal form [nx, ny, nz, d] where n.x + d = 0."""
        return np.array([self.normal[0], self.normal[1], self.normal[2], self.distance], dtype=np.float64)

    def can_merge(
        self,
        other: "SpatialSurface",
        max_angle_deg: float = 10.0,
        max_offset_m: float = 0.03,
    ) -> bool:
        """Check if two planar surfaces are coplanar and can be merged."""
        cos_thresh = np.cos(np.radians(max_angle_deg))
        cos_angle = float(np.dot(self.normal, other.normal))

        if abs(cos_angle) < cos_thresh:
            return False

        # If opposite orientation, flip other's normal and distance for comparison
        sign = 1.0 if cos_angle >= 0 else -1.0
        other_dist = other.distance * sign

        offset = abs(self.distance - other_dist)
        return offset <= max_offset_m

    def merge(self, other: "SpatialSurface") -> "SpatialSurface":
        """Merge this surface with an adjacent coplanar surface."""
        cos_angle = float(np.dot(self.normal, other.normal))
        sign = 1.0 if cos_angle >= 0 else -1.0

        total_inliers = self.inlier_count + other.inlier_count
        w_self = self.inlier_count / max(total_inliers, 1)
        w_other = other.inlier_count / max(total_inliers, 1)

        merged_normal = w_self * self.normal + w_other * (other.normal * sign)
        merged_normal /= np.linalg.norm(merged_normal)

        merged_distance = float(w_self * self.distance + w_other * (other.distance * sign))

        b_min = np.minimum(self.bounds_world[0], other.bounds_world[0])
        b_max = np.maximum(self.bounds_world[1], other.bounds_world[1])

        merged_points = None
        if self.plane_points is not None and other.plane_points is not None:
            merged_points = np.vstack([self.plane_points, other.plane_points])
        elif self.plane_points is not None:
            merged_points = self.plane_points
        elif other.plane_points is not None:
            merged_points = other.plane_points

        return SpatialSurface(
            surface_id=self.surface_id,
            normal=merged_normal,
            distance=merged_distance,
            bounds_world=(b_min, b_max),
            last_observed_timestamp=max(self.last_observed_timestamp, other.last_observed_timestamp),
            inlier_count=total_inliers,
            track_id=self.track_id or other.track_id,
            plane_points=merged_points,
        )


class SurfaceManager:
    """Tracks and updates persistent spatial surfaces over time."""

    def __init__(self, max_angle_deg: float = 10.0, max_offset_m: float = 0.03):
        self.surfaces: Dict[str, SpatialSurface] = {}
        self.max_angle_deg = max_angle_deg
        self.max_offset_m = max_offset_m
        self._next_id = 1

    def register_or_update(
        self,
        track_id: str,
        normal: np.ndarray,
        distance: float,
        plane_points: np.ndarray,
        timestamp: float,
    ) -> SpatialSurface:
        """Register a new surface or merge with an existing surface for this track."""
        normal = np.asarray(normal, dtype=np.float64)
        norm = np.linalg.norm(normal)
        if norm > 0:
            normal /= norm

        b_min = np.min(plane_points, axis=0) if len(plane_points) > 0 else np.zeros(3)
        b_max = np.max(plane_points, axis=0) if len(plane_points) > 0 else np.zeros(3)

        new_surface = SpatialSurface(
            surface_id=f"surface_{self._next_id:04d}",
            normal=normal,
            distance=float(distance),
            bounds_world=(b_min, b_max),
            last_observed_timestamp=timestamp,
            inlier_count=len(plane_points),
            track_id=track_id,
            plane_points=plane_points,
        )
        self._next_id += 1

        existing = self.surfaces.get(track_id)
        if existing is not None and existing.can_merge(new_surface, self.max_angle_deg, self.max_offset_m):
            merged = existing.merge(new_surface)
            self.surfaces[track_id] = merged
            return merged

        self.surfaces[track_id] = new_surface
        return new_surface

    def get_surface(
        self,
        track_id: str,
        current_timestamp: float,
        max_age_s: float = 2.0,
    ) -> Optional[SpatialSurface]:
        """Retrieve active surface for track_id if within max_age_s."""
        surface = self.surfaces.get(track_id)
        if surface is None:
            return None

        if current_timestamp - surface.last_observed_timestamp > max_age_s:
            return None

        return surface

    def prune(self, current_timestamp: float, max_age_s: float = 5.0) -> None:
        """Evict surfaces not observed for max_age_s."""
        expired = [
            k for k, s in self.surfaces.items()
            if current_timestamp - s.last_observed_timestamp > max_age_s
        ]
        for k in expired:
            del self.surfaces[k]
