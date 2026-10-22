from typing import Optional, Tuple
import numpy as np

from scene_graph.config import SceneGraphConfig
from scene_graph.tracking.track import Track
from scene_graph.relations.context import FrameContext, ObservationGeometry
from scene_graph.relations.inverse_algebra import INVERSE, SYMMETRIC


ALLOWED_PREDICATES = frozenset(INVERSE) | frozenset(SYMMETRIC)


class AdmissibilityFilter:
    """Evaluate whether object pairs are geometrically admissible for relations.

    Admissibility is based on physical geometry and the configured reference frame.
    Semantic detector labels are metadata and do not affect admission.
    """

    def __init__(self, config: SceneGraphConfig):
        self.config = config
        self.contact_tolerance_m = (
            float(config.relations.support.contact_tolerance_m)
            if config.relations and config.relations.support
            else 0.08
        )

    def _get_geometry(
        self,
        track: Track,
        context: Optional[FrameContext] = None,
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:
        """Extract (centroid_world, bbox_min_world, bbox_max_world) for a track."""
        if context is not None and context.observation_geometry:
            geom = context.observation_geometry.get(track.object_id)
            if geom is not None:
                return geom.centroid_world, geom.bbox_min_world, geom.bbox_max_world

        if track.recent_observations:
            obs = track.recent_observations[-1]
            if obs.object_geometry is not None:
                g = obs.object_geometry
                return g.centroid_world, g.bbox_min_world, g.bbox_max_world

        centroid = track.centroid_world
        bbox_min = None
        bbox_max = None
        if centroid is not None and track.size_world is not None:
            half_size = np.asarray(track.size_world) / 2.0
            bbox_min = centroid - half_size
            bbox_max = centroid + half_size

        return centroid, bbox_min, bbox_max

    def _get_up_axis(self, context: Optional[FrameContext]) -> np.ndarray:
        if context is not None and context.reference_frame is not None:
            up = getattr(context.reference_frame, "up_axis_world", None)
            if up is not None:
                return np.asarray(up, dtype=np.float64) / np.linalg.norm(up)
        return np.array([0.0, 0.0, 1.0], dtype=np.float64)

    def is_admissible(
        self,
        predicate: str,
        subject: Track,
        object_: Track,
        context: Optional[FrameContext] = None,
    ) -> bool:
        """Evaluate if relation predicate between subject and object is physically admissible."""
        if predicate not in ALLOWED_PREDICATES:
            raise ValueError(
                f"Predicate '{predicate}' is not part of the "
                f"14-predicate relation contract."
            )

        if subject.object_id == object_.object_id:
            return False

        # Support check: ON (subject is ON object)
        if predicate == "ON":
            return self._is_on_admissible(subject, object_, context)

        # Inverse of ON: UNDER (subject is UNDER object -> object is ON subject)
        if predicate == "UNDER":
            return self._is_on_admissible(object_, subject, context)

        # Containment check: INSIDE (subject is INSIDE object)
        if predicate == "INSIDE":
            return self._is_inside_admissible(subject, object_, context)

        if predicate == "CONTAINING":
            return self._is_inside_admissible(object_, subject, context)

        return True

    def _is_on_admissible(
        self,
        supporter_target: Track,
        supporter_surface: Track,
        context: Optional[FrameContext],
    ) -> bool:
        """Check if supporter_target (A) can physically rest ON supporter_surface (B)."""
        c_A, min_A, max_A = self._get_geometry(supporter_target, context)
        c_B, min_B, max_B = self._get_geometry(supporter_surface, context)

        if c_A is None or c_B is None:
            return False

        up = self._get_up_axis(context)

        # 1. Height check: A's centroid must be above B's centroid
        h_A = float(np.dot(c_A, up))
        h_B = float(np.dot(c_B, up))

        if h_A <= h_B:
            return False

        # 2. Bounding box checks if available
        if min_A is not None and max_A is not None and min_B is not None and max_B is not None:
            # Bottom of A cannot be significantly below bottom of B
            bottom_A = float(np.dot(min_A, up))
            bottom_B = float(np.dot(min_B, up))
            top_B = float(np.dot(max_B, up))

            if bottom_A < bottom_B - 0.05:
                return False

            # Vertical separation clearance check
            gap = bottom_A - top_B
            max_allowed_gap = max(0.60, self.contact_tolerance_m * 5)
            if gap > max_allowed_gap:
                return False

            # Horizontal projection overlap check
            # Find two arbitrary orthonormal axes orthogonal to up
            if abs(up[0]) < 0.9:
                u1 = np.cross(up, [1.0, 0.0, 0.0])
            else:
                u1 = np.cross(up, [0.0, 1.0, 0.0])
            u1 /= np.linalg.norm(u1)
            u2 = np.cross(up, u1)

            # Check overlap on canonical axes if up is close to Z, Y, or X
            for axis_idx in range(3):
                if not np.isclose(abs(up[axis_idx]), 1.0, atol=1e-2):
                    min_val_A, max_val_A = min_A[axis_idx], max_A[axis_idx]
                    min_val_B, max_val_B = min_B[axis_idx], max_B[axis_idx]
                    overlap = min(max_val_A, max_val_B) - max(min_val_A, min_val_B)
                    if overlap < -0.15:  # Margin of 15cm
                        return False

        return True

    def _is_inside_admissible(
        self,
        inner: Track,
        outer: Track,
        context: Optional[FrameContext],
    ) -> bool:
        """Check if inner (A) can physically be INSIDE outer (B)."""
        c_A, min_A, max_A = self._get_geometry(inner, context)
        c_B, min_B, max_B = self._get_geometry(outer, context)

        if min_A is not None and max_A is not None and min_B is not None and max_B is not None:
            vol_A = float(np.prod(np.maximum(max_A - min_A, 1e-4)))
            vol_B = float(np.prod(np.maximum(max_B - min_B, 1e-4)))
            # Outer object must be larger than inner object
            if vol_A > vol_B * 1.5:
                return False

        return True