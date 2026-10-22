from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple
import numpy as np

from scene_graph.config import SceneGraphConfig
from scene_graph.tracking.track import Track, TrackState
from scene_graph.relations.context import FrameContext, ObservationGeometry
from scene_graph.relations.hierarchy import SpatialContext


@dataclass
class CandidateSummary:
    total_objects: int
    potential_pairs: int
    structural_pairs: int
    proximity_pairs: int
    directional_pairs: int
    visibility_pairs: int
    total_candidates: int

    def __str__(self) -> str:
        return (
            f"Objects: {self.total_objects} "
            f"Potential pairs: {self.potential_pairs} "
            f"Structural pairs: {self.structural_pairs} "
            f"Proximity pairs: {self.proximity_pairs} "
            f"Directional pairs: {self.directional_pairs} "
            f"Visibility pairs: {self.visibility_pairs} "
            f"Total candidates: {self.total_candidates}"
        )


class RelationCandidateGenerator:
    """
    Pre-inference typed candidate generator.
    
    Filters pairs BEFORE relation modules execute, enforcing:
    1. Structural candidates: Evaluates objects against candidate anchors/containers.
    2. Proximity candidates: Evaluates strictly sibling objects in the same context; excludes anchor-child pairs.
    3. Directional candidates: Evaluates strictly sibling objects in the same context with orthogonal overlap; excludes anchor-child pairs.
    4. Visibility candidates: Evaluates pairs with 2D camera ray overlap and depth separation.
    """

    def __init__(self, config: Optional[SceneGraphConfig] = None):
        self.config = config
        self.min_orthogonal_overlap_ratio = 0.15
        self.near_threshold = 0.50
        self.scale_factor = 1.2
        self.max_lateral_distance = 0.65

        if config is not None and config.relations is not None:
            if hasattr(config.relations, "directional") and hasattr(config.relations.directional, "min_orthogonal_overlap_ratio"):
                self.min_orthogonal_overlap_ratio = float(config.relations.directional.min_orthogonal_overlap_ratio)
            if hasattr(config.relations, "distance"):
                dist_cfg = config.relations.distance
                if hasattr(dist_cfg, "near_threshold"):
                    self.near_threshold = float(dist_cfg.near_threshold)
                if hasattr(dist_cfg, "scale_factor"):
                    self.scale_factor = float(dist_cfg.scale_factor)

    @staticmethod
    def _get_active_tracks(tracks: List[Track]) -> List[Track]:
        return [
            t for t in tracks
            if t.state in (TrackState.ACTIVE, TrackState.TEMPORARILY_UNOBSERVED)
        ]

    def generate_structural_candidates(
        self,
        tracks: List[Track],
        context: FrameContext,
        contexts: Dict[str, SpatialContext],
    ) -> List[Tuple[Track, Track]]:
        """
        Structural candidates: (manipulable_object, support_surface_anchor).
        Checks vertical clearance and horizontal footprint containment.
        Anchors are support foundations, not objects sitting on other anchors.
        """
        active_tracks = self._get_active_tracks(tracks)
        anchor_track_ids = {
            ctx.anchor_track_id
            for ctx in contexts.values()
            if ctx.anchor_track_id is not None
        }

        track_by_id = {t.object_id: t for t in active_tracks}
        candidates: List[Tuple[Track, Track]] = []

        for sub in active_tracks:
            # Anchors do not sit on other anchors
            if sub.object_id in anchor_track_ids:
                continue

            sub_geom = context.observation_geometry.get(sub.object_id)
            if sub_geom is None:
                continue

            sub_c = sub_geom.centroid_world
            sub_min = sub_geom.bbox_min_world
            sub_max = sub_geom.bbox_max_world
            if sub_min is None or sub_max is None or sub_c is None:
                continue

            for obj_id in anchor_track_ids:
                if obj_id == sub.object_id:
                    continue
                obj = track_by_id.get(obj_id)
                if obj is None:
                    continue
                obj_geom = context.observation_geometry.get(obj_id)
                if obj_geom is None:
                    continue

                obj_min = obj_geom.bbox_min_world
                obj_max = obj_geom.bbox_max_world
                if obj_min is None or obj_max is None:
                    continue

                # 1. Horizontal footprint check (with 10cm margin)
                margin_xy = 0.10
                if not (obj_min[0] - margin_xy <= sub_c[0] <= obj_max[0] + margin_xy and
                        obj_min[1] - margin_xy <= sub_c[1] <= obj_max[1] + margin_xy):
                    continue

                # 2. Vertical relation: subject must be near or above object surface
                dz = sub_min[2] - obj_min[2]
                if dz >= -0.10 and sub_min[2] <= obj_max[2] + 0.35:
                    candidates.append((sub, obj))

        return candidates

    def generate_containment_candidates(
        self,
        tracks: List[Track],
        context: FrameContext,
        contexts: Dict[str, SpatialContext],
    ) -> List[Tuple[Track, Track]]:
        """
        Containment candidates: (inner_object, container).
        An anchor support surface (desk/table) is a support surface, NOT a container.
        Inner object must be geometrically interior to outer container.
        """
        active_tracks = self._get_active_tracks(tracks)
        anchor_track_ids = {
            ctx.anchor_track_id
            for ctx in contexts.values()
            if ctx.anchor_track_id is not None
        }

        candidates: List[Tuple[Track, Track]] = []

        for sub in active_tracks:
            if sub.object_id in anchor_track_ids:
                continue

            sub_geom = context.observation_geometry.get(sub.object_id)
            if sub_geom is None or sub_geom.bbox_min_world is None or sub_geom.bbox_max_world is None:
                continue

            sub_min = sub_geom.bbox_min_world
            sub_max = sub_geom.bbox_max_world

            for obj in active_tracks:
                if obj.object_id == sub.object_id:
                    continue
                # Support surface anchors are not containers
                if obj.object_id in anchor_track_ids:
                    continue

                obj_geom = context.observation_geometry.get(obj.object_id)
                if obj_geom is None or obj_geom.bbox_min_world is None or obj_geom.bbox_max_world is None:
                    continue

                obj_min = obj_geom.bbox_min_world
                obj_max = obj_geom.bbox_max_world

                vol_sub = float(np.prod(np.maximum(sub_max - sub_min, 1e-3)))
                vol_obj = float(np.prod(np.maximum(obj_max - obj_min, 1e-3)))
                if vol_obj < vol_sub * 1.5:
                    continue

                # Subject must not be resting on top of container
                if sub_max[2] > obj_max[2] + 0.02:
                    continue

                if (obj_min[0] - 0.02 <= sub_min[0] and sub_max[0] <= obj_max[0] + 0.02 and
                    obj_min[1] - 0.02 <= sub_min[1] and sub_max[1] <= obj_max[1] + 0.02 and
                    obj_min[2] - 0.02 <= sub_min[2]):
                    candidates.append((sub, obj))

        return candidates

    def generate_proximity_candidates(
        self,
        tracks: List[Track],
        context: FrameContext,
        contexts: Dict[str, SpatialContext],
        membership: Dict[str, str],
    ) -> List[Tuple[Track, Track]]:
        """
        Proximity candidates: Evaluated ONLY between siblings in the same spatial context.
        Anchor-child pairs are explicitly excluded.
        Gated by scale-aware distance threshold.
        """
        active_tracks = self._get_active_tracks(tracks)
        anchor_track_ids = {
            ctx.anchor_track_id
            for ctx in contexts.values()
            if ctx.anchor_track_id is not None
        }

        sibling_tracks = [
            t for t in active_tracks
            if t.object_id not in anchor_track_ids
        ]

        candidates: List[Tuple[Track, Track]] = []
        n = len(sibling_tracks)

        for i in range(n):
            t_a = sibling_tracks[i]
            ctx_a = membership.get(t_a.object_id, "world")
            geom_a = context.observation_geometry.get(t_a.object_id)
            c_a = geom_a.centroid_world if geom_a is not None else t_a.centroid_world
            if c_a is None:
                continue

            scale_a = 0.20
            if geom_a is not None and geom_a.bbox_min_world is not None and geom_a.bbox_max_world is not None:
                scale_a = float(np.mean(geom_a.bbox_max_world - geom_a.bbox_min_world))

            for j in range(i + 1, n):
                t_b = sibling_tracks[j]
                ctx_b = membership.get(t_b.object_id, "world")

                # Must share the same spatial context
                if ctx_a != ctx_b:
                    continue

                geom_b = context.observation_geometry.get(t_b.object_id)
                c_b = geom_b.centroid_world if geom_b is not None else t_b.centroid_world
                if c_b is None:
                    continue

                scale_b = 0.20
                if geom_b is not None and geom_b.bbox_min_world is not None and geom_b.bbox_max_world is not None:
                    scale_b = float(np.mean(geom_b.bbox_max_world - geom_b.bbox_min_world))

                dist = float(np.linalg.norm(c_a - c_b))
                # Size-aware dynamic distance gate:
                # distance must be bounded by scale + contact tolerance, capped at near_threshold
                dynamic_limit = min(self.near_threshold, 0.6 * (scale_a + scale_b) + 0.15)

                if dist <= dynamic_limit:
                    candidates.append((t_a, t_b))

        return candidates

    def generate_directional_candidates(
        self,
        tracks: List[Track],
        context: FrameContext,
        contexts: Dict[str, SpatialContext],
        membership: Dict[str, str],
    ) -> List[Tuple[Track, Track]]:
        """
        Directional candidates: Evaluated ONLY between siblings in the same spatial context.
        Anchor-child pairs are explicitly excluded.
        Requires meaningful orthogonal alignment on BOTH non-query axes (up and depth).
        """
        active_tracks = self._get_active_tracks(tracks)
        anchor_track_ids = {
            ctx.anchor_track_id
            for ctx in contexts.values()
            if ctx.anchor_track_id is not None
        }

        sibling_tracks = [
            t for t in active_tracks
            if t.object_id not in anchor_track_ids
        ]

        candidates: List[Tuple[Track, Track]] = []
        n = len(sibling_tracks)

        rf = context.reference_frame
        up_axis = getattr(rf, "up_axis_world", np.array([0.0, 0.0, 1.0]))
        horiz_axis = getattr(rf, "horizontal_axis_world", np.array([1.0, 0.0, 0.0]))
        depth_axis = getattr(rf, "depth_axis_world", np.array([0.0, 1.0, 0.0]))

        for i in range(n):
            t_a = sibling_tracks[i]
            ctx_a = membership.get(t_a.object_id, "world")
            geom_a = context.observation_geometry.get(t_a.object_id)
            if geom_a is None or geom_a.bbox_min_world is None or geom_a.bbox_max_world is None:
                continue

            for j in range(n):
                if i == j:
                    continue
                t_b = sibling_tracks[j]
                ctx_b = membership.get(t_b.object_id, "world")

                # Must share the same spatial context
                if ctx_a != ctx_b:
                    continue

                geom_b = context.observation_geometry.get(t_b.object_id)
                if geom_b is None or geom_b.bbox_min_world is None or geom_b.bbox_max_world is None:
                    continue

                # Project bounding intervals onto up and depth axes
                a_min, a_max = geom_a.bbox_min_world, geom_a.bbox_max_world
                b_min, b_max = geom_b.bbox_min_world, geom_b.bbox_max_world

                # 1. Overlap along up_axis
                up_a = (float(np.dot(a_min, up_axis)), float(np.dot(a_max, up_axis)))
                up_b = (float(np.dot(b_min, up_axis)), float(np.dot(b_max, up_axis)))
                up_overlap = max(0.0, min(up_a[1], up_b[1]) - max(up_a[0], up_b[0]))
                min_up_span = max(1e-4, min(up_a[1] - up_a[0], up_b[1] - up_b[0]))
                up_ratio = up_overlap / min_up_span

                # 2. Overlap along depth_axis
                dep_a = (float(np.dot(a_min, depth_axis)), float(np.dot(a_max, depth_axis)))
                dep_b = (float(np.dot(b_min, depth_axis)), float(np.dot(b_max, depth_axis)))
                dep_overlap = max(0.0, min(dep_a[1], dep_b[1]) - max(dep_a[0], dep_b[0]))
                min_dep_span = max(1e-4, min(dep_a[1] - dep_a[0], dep_b[1] - dep_b[0]))
                dep_ratio = dep_overlap / min_dep_span

                # Orthogonal alignment: must have sufficient overlap on BOTH orthogonal axes
                # (both table elevation and front/back depth band must align)
                if up_ratio >= self.min_orthogonal_overlap_ratio and dep_ratio >= self.min_orthogonal_overlap_ratio:
                    x_a = float(np.dot(geom_a.centroid_world, horiz_axis))
                    x_b = float(np.dot(geom_b.centroid_world, horiz_axis))
                    lateral_dist = abs(x_a - x_b)

                    # Only emit canonical direction (A is LEFT_OF B if x_a < x_b) and within reasonable local band
                    if x_a < x_b and lateral_dist <= self.max_lateral_distance:
                        candidates.append((t_a, t_b))

        return candidates

    def generate_visibility_candidates(
        self,
        tracks: List[Track],
        context: FrameContext,
    ) -> List[Tuple[Track, Track]]:
        """
        Visibility/Occlusion candidates: Evaluates pairs where 2D projection rays overlap
        and subject is closer to camera than object.
        """
        active_tracks = self._get_active_tracks(tracks)
        candidates: List[Tuple[Track, Track]] = []
        n = len(active_tracks)

        for i in range(n):
            t_a = active_tracks[i]
            geom_a = context.observation_geometry.get(t_a.object_id)
            if geom_a is None or geom_a.points_camera is None or len(geom_a.points_camera) == 0:
                continue

            z_a = float(np.median(geom_a.points_camera[:, 2]))

            for j in range(n):
                if i == j:
                    continue
                t_b = active_tracks[j]
                geom_b = context.observation_geometry.get(t_b.object_id)
                if geom_b is None or geom_b.points_camera is None or len(geom_b.points_camera) == 0:
                    continue

                z_b = float(np.median(geom_b.points_camera[:, 2]))
                # A must be closer to camera than B to occlude B
                if z_a >= z_b - 0.05:
                    continue

                # 1. 2D mask intersection
                if geom_a.mask is not None and geom_b.mask is not None:
                    intersection = np.logical_and(geom_a.mask > 0, geom_b.mask > 0)
                    if np.count_nonzero(intersection) >= 15:
                        candidates.append((t_a, t_b))
                elif geom_a.bbox_min_world is not None and geom_b.bbox_min_world is not None:
                    # If 2D masks not present, check camera ray angular separation
                    c_a = geom_a.centroid_world
                    c_b = geom_b.centroid_world
                    cam_pos = context.world_T_camera[:3, 3] if context.world_T_camera is not None else np.zeros(3)
                    ray_a = (c_a - cam_pos) / max(1e-4, np.linalg.norm(c_a - cam_pos))
                    ray_b = (c_b - cam_pos) / max(1e-4, np.linalg.norm(c_b - cam_pos))
                    angle = np.arccos(np.clip(np.dot(ray_a, ray_b), -1.0, 1.0))
                    # Narrow angular cone (< 6 degrees = 0.10 rad)
                    if angle < 0.10:
                        candidates.append((t_a, t_b))

        return candidates

    def generate_all_candidates(
        self,
        tracks: List[Track],
        context: FrameContext,
        contexts: Dict[str, SpatialContext],
        membership: Dict[str, str],
    ) -> Tuple[Dict[str, List[Tuple[Track, Track]]], CandidateSummary]:
        """
        Generates typed candidates for all relation categories and computes summary metrics.
        """
        active = self._get_active_tracks(tracks)
        n = len(active)
        potential = n * (n - 1)

        structural = self.generate_structural_candidates(tracks, context, contexts)
        containment = self.generate_containment_candidates(tracks, context, contexts)
        proximity = self.generate_proximity_candidates(tracks, context, contexts, membership)
        directional = self.generate_directional_candidates(tracks, context, contexts, membership)
        visibility = self.generate_visibility_candidates(tracks, context)

        total_cand = len(structural) + len(containment) + len(proximity) + len(directional) + len(visibility)

        summary = CandidateSummary(
            total_objects=n,
            potential_pairs=potential,
            structural_pairs=len(structural),
            proximity_pairs=len(proximity),
            directional_pairs=len(directional),
            visibility_pairs=len(visibility),
            total_candidates=total_cand,
        )

        candidates = {
            "structural": structural,
            "containment": containment,
            "proximity": proximity,
            "directional": directional,
            "visibility": visibility,
        }
        return candidates, summary
