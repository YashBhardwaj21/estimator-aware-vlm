from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple, Any
import numpy as np

from scene_graph.config import SpatialHierarchyConfig
from scene_graph.tracking.track import Track, TrackState
from scene_graph.relations.context import FrameContext, ObservationGeometry
from scene_graph.geometry.plane import fit_plane_ransac


@dataclass
class SpatialContext:
    """
    Represents an extensible spatial context abstraction in an N-level hierarchy.
    Examples:
      Level 0: 'world'
      Level 1: 'context_track_0010' (Desk Anchor Surface / Workstation)
      Level 2: 'context_track_0015' (Container/Tray on Desk)
    """
    context_id: str
    parent_context_id: Optional[str] = None
    anchor_track_id: Optional[str] = None
    level: int = 1
    member_track_ids: Set[str] = field(default_factory=set)
    child_context_ids: Set[str] = field(default_factory=set)
    bounding_box_min: Optional[np.ndarray] = None
    bounding_box_max: Optional[np.ndarray] = None
    plane_normal: Optional[np.ndarray] = None
    plane_distance: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class SpatialContextManager:
    """
    Manages spatial contexts with geometry-first anchor discovery.
    
    Principles:
    1. Zero semantic-class gating for anchor detection. Anchors qualify via
       horizontal extent, planar surface fit, temporal stability, and occupancy.
    2. Decouples context membership (computational grouping) from physical
       structural relations (directed graph edges).
    3. Supports arbitrary N-level hierarchical nesting.
    """

    def __init__(self, config: Optional[SpatialHierarchyConfig] = None):
        self.config = config or SpatialHierarchyConfig()
        self._cached_contexts: Dict[str, SpatialContext] = {}

    def compute_anchor_score(
        self,
        track: Track,
        geometry: Optional[ObservationGeometry],
        all_tracks: List[Track],
        geometries: Dict[str, ObservationGeometry],
        up_axis: np.ndarray = np.array([0.0, 0.0, 1.0]),
    ) -> Tuple[float, Dict[str, float]]:
        """
        Computes a continuous physical anchor score in [0, 1].
        
        Score = w_area * S_area + w_plane * S_plane + w_stability * S_stability + w_support * S_support
        Zero class-name dependency.
        """
        if geometry is None:
            return 0.0, {}

        # 1. Horizontal extent area orthogonal to up_axis
        min_pt = geometry.bbox_min_world
        max_pt = geometry.bbox_max_world
        if min_pt is None or max_pt is None:
            return 0.0, {}

        extent = max_pt - min_pt
        # Project dimensions orthogonal to up_axis (assuming up_axis aligned roughly with Z)
        # For general up_axis, area is magnitude of cross product of horizontal dimensions
        if abs(up_axis[2]) > 0.7:
            horizontal_area = float(extent[0] * extent[1])
        elif abs(up_axis[1]) > 0.7:
            horizontal_area = float(extent[0] * extent[2])
        else:
            horizontal_area = float(extent[1] * extent[2])

        min_a = self.config.min_anchor_area_m2
        tgt_a = self.config.target_anchor_area_m2
        if horizontal_area < min_a:
            return 0.0, {
                "area_score": 0.0,
                "plane_score": 0.0,
                "stability_score": 0.0,
                "support_score": 0.0,
                "horizontal_area_m2": horizontal_area,
                "supported_objects_count": 0.0,
            }

        area_score = float(np.clip((horizontal_area - min_a) / max(tgt_a - min_a, 1e-4), 0.0, 1.0))

        # 2. Planar support geometry
        plane_score = 0.0
        points = geometry.points_world_sampled if geometry.points_world_sampled is not None else geometry.points_world
        if points is not None and len(points) >= 10:
            rng = np.random.default_rng(0)
            plane = fit_plane_ransac(
                points,
                distance_threshold=0.05,
                max_iterations=50,
                min_inliers=10,
                rng=rng,
            )
            if plane is not None:
                normal = plane.normal
                normal_norm = np.linalg.norm(normal)
                if normal_norm > 1e-6:
                    normal = normal / normal_norm
                    alignment = float(abs(np.dot(normal, up_axis)))
                    if alignment >= self.config.gravity_alignment_cosine:
                        inlier_ratio = float(len(plane.inlier_mask) / len(points))
                        # Scale by how well it aligns with gravity and fits a plane
                        plane_score = float(np.clip(inlier_ratio * (alignment / 1.0), 0.0, 1.0))

        # 3. Temporal persistence and tracking stability
        obs_ratio = getattr(track, "track_observation_ratio", 1.0)
        obs_count = getattr(track, "observation_count", 1)
        stability_score = float(
            np.clip(obs_ratio * min(1.0, obs_count / float(self.config.min_anchor_observations * 2)), 0.0, 1.0)
        )

        # 4. Occupancy / Contact evidence: check how many other objects have centroids above this surface
        supported_count = 0
        obj_xy_min = min_pt[:2] - 0.05
        obj_xy_max = max_pt[:2] + 0.05
        obj_z_top = max_pt[2]

        for other in all_tracks:
            if other.object_id == track.object_id or other.state == TrackState.CANDIDATE:
                continue
            other_geom = geometries.get(other.object_id)
            c_world = other_geom.centroid_world if other_geom is not None else other.centroid_world
            if c_world is None:
                continue

            # Check if within XY footprint and vertically near or above the surface
            if (obj_xy_min[0] <= c_world[0] <= obj_xy_max[0] and
                obj_xy_min[1] <= c_world[1] <= obj_xy_max[1] and
                c_world[2] >= min_pt[2] - 0.05):
                supported_count += 1

        support_score = float(np.clip(supported_count / 2.0, 0.0, 1.0))

        # An anchor surface MUST have a verified planar support geometry aligned with gravity
        if plane_score <= 0.05:
            return 0.0, {
                "area_score": area_score,
                "plane_score": 0.0,
                "stability_score": stability_score,
                "support_score": support_score,
                "horizontal_area_m2": horizontal_area,
                "supported_objects_count": float(supported_count),
            }

        # Weighted score combination
        total_score = float(
            self.config.weight_area * area_score +
            self.config.weight_plane * plane_score +
            self.config.weight_stability * stability_score +
            self.config.weight_support * support_score
        )

        details = {
            "area_score": area_score,
            "plane_score": plane_score,
            "stability_score": stability_score,
            "support_score": support_score,
            "horizontal_area_m2": horizontal_area,
            "supported_objects_count": float(supported_count),
        }
        return total_score, details

    def discover_contexts(
        self,
        tracks: List[Track],
        frame_context: FrameContext,
    ) -> Dict[str, SpatialContext]:
        """
        Discovers spatial contexts from active tracks and observation geometries.
        Builds an N-level hierarchy rooted at 'world'.
        Suppresses duplicate/fragmented anchors of the same physical surface.
        """
        contexts: Dict[str, SpatialContext] = {
            "world": SpatialContext(
                context_id="world",
                parent_context_id=None,
                anchor_track_id=None,
                level=0,
                metadata={"type": "root_world"},
            )
        }

        up_axis = np.array([0.0, 0.0, 1.0])
        if frame_context.reference_frame is not None:
            rf_up = getattr(frame_context.reference_frame, "up_axis_world", None)
            if rf_up is not None:
                up_axis = np.asarray(rf_up, dtype=np.float64)

        active_tracks = [
            t for t in tracks
            if t.state in (TrackState.ACTIVE, TrackState.TEMPORARILY_UNOBSERVED)
        ]

        # Score all active tracks for anchor potential
        anchor_candidates: List[Tuple[Track, float, Dict[str, float]]] = []
        for track in active_tracks:
            geom = frame_context.observation_geometry.get(track.object_id)
            score, details = self.compute_anchor_score(
                track,
                geom,
                active_tracks,
                frame_context.observation_geometry,
                up_axis=up_axis,
            )
            if score >= self.config.anchor_score_threshold:
                anchor_candidates.append((track, score, details))

        # Sort anchor candidates by continuous score and horizontal extent
        anchor_candidates.sort(
            key=lambda x: (x[1], x[2].get("horizontal_area_m2", 0.0)),
            reverse=True,
        )

        # Non-maximum suppression for overlapping anchors of the same physical support surface
        selected_anchors: List[Tuple[Track, float, Dict[str, float]]] = []
        for cand in anchor_candidates:
            cand_track, cand_score, cand_details = cand
            cand_geom = frame_context.observation_geometry.get(cand_track.object_id)
            if cand_geom is None or cand_geom.bbox_min_world is None or cand_geom.bbox_max_world is None:
                continue

            c_min = cand_geom.bbox_min_world[:2]
            c_max = cand_geom.bbox_max_world[:2]
            c_area = max(1e-4, (c_max[0] - c_min[0]) * (c_max[1] - c_min[1]))

            # Check overlap against already accepted anchors
            duplicate = False
            for sel_track, _, _ in selected_anchors:
                sel_geom = frame_context.observation_geometry.get(sel_track.object_id)
                if sel_geom is None or sel_geom.bbox_min_world is None:
                    continue
                s_min = sel_geom.bbox_min_world[:2]
                s_max = sel_geom.bbox_max_world[:2]

                inter_min = np.maximum(c_min, s_min)
                inter_max = np.minimum(c_max, s_max)
                if np.all(inter_max > inter_min):
                    inter_area = (inter_max[0] - inter_min[0]) * (inter_max[1] - inter_min[1])
                    overlap_ratio = inter_area / min(c_area, max(1e-4, (s_max[0] - s_min[0]) * (s_max[1] - s_min[1])))
                    if overlap_ratio >= 0.50:
                        duplicate = True
                        break

            if not duplicate:
                selected_anchors.append(cand)

        # Build context objects for each discovered anchor
        for anchor_track, score, details in selected_anchors:
            geom = frame_context.observation_geometry.get(anchor_track.object_id)
            cid = f"context_{anchor_track.object_id}"

            # Check if this anchor is inside/supported by an already discovered larger anchor (N-level nesting)
            parent_cid = "world"
            level = 1
            for existing_cid, existing_ctx in list(contexts.items()):
                if existing_cid == "world" or existing_ctx.anchor_track_id is None:
                    continue
                # If current anchor is smaller and fits within existing anchor's bounding box
                if existing_ctx.bounding_box_min is not None and geom is not None:
                    c = geom.centroid_world
                    if (np.all(c >= existing_ctx.bounding_box_min - 0.05) and
                        np.all(c <= existing_ctx.bounding_box_max + 0.05)):
                        parent_cid = existing_cid
                        level = existing_ctx.level + 1
                        existing_ctx.child_context_ids.add(cid)
                        break

            if parent_cid == "world":
                contexts["world"].child_context_ids.add(cid)

            new_ctx = SpatialContext(
                context_id=cid,
                parent_context_id=parent_cid,
                anchor_track_id=anchor_track.object_id,
                level=level,
                bounding_box_min=geom.bbox_min_world.copy() if geom is not None and geom.bbox_min_world is not None else None,
                bounding_box_max=geom.bbox_max_world.copy() if geom is not None and geom.bbox_max_world is not None else None,
                metadata={
                    "anchor_score": score,
                    "anchor_class": anchor_track.class_name,
                    **details,
                },
            )
            contexts[cid] = new_ctx

        self._cached_contexts = contexts
        return contexts

    def assign_context_membership(
        self,
        tracks: List[Track],
        contexts: Dict[str, SpatialContext],
        frame_context: FrameContext,
    ) -> Dict[str, str]:
        """
        Assigns each active track to its primary SpatialContext.
        Returns a mapping of track_id -> context_id.
        Also populates context.member_track_ids.
        
        Important: Context membership is for spatial grouping and candidate pruning.
        It is NOT the complete physical relation model.
        """
        membership: Dict[str, str] = {}
        for ctx in contexts.values():
            ctx.member_track_ids.clear()

        anchor_track_ids = {
            ctx.anchor_track_id: cid
            for cid, ctx in contexts.items()
            if ctx.anchor_track_id is not None
        }

        active_tracks = [
            t for t in tracks
            if t.state in (TrackState.ACTIVE, TrackState.TEMPORARILY_UNOBSERVED)
        ]

        for track in active_tracks:
            # If the track itself is an anchor, it belongs to its parent context
            if track.object_id in anchor_track_ids:
                ctx_id = anchor_track_ids[track.object_id]
                parent_id = contexts[ctx_id].parent_context_id or "world"
                membership[track.object_id] = parent_id
                contexts[parent_id].member_track_ids.add(track.object_id)
                continue

            geom = frame_context.observation_geometry.get(track.object_id)
            c_world = geom.centroid_world if geom is not None else track.centroid_world
            if c_world is None:
                membership[track.object_id] = "world"
                contexts["world"].member_track_ids.add(track.object_id)
                continue

            # Evaluate against all anchor contexts (check deep/higher-level contexts first)
            best_cid = "world"
            best_dist_z = float("inf")

            for cid, ctx in contexts.items():
                if cid == "world" or ctx.bounding_box_min is None or ctx.bounding_box_max is None:
                    continue

                b_min = ctx.bounding_box_min
                b_max = ctx.bounding_box_max

                # Check horizontal footprint overlap
                margin_xy = 0.08
                if (b_min[0] - margin_xy <= c_world[0] <= b_max[0] + margin_xy and
                    b_min[1] - margin_xy <= c_world[1] <= b_max[1] + margin_xy):
                    # Check vertical position: must be near or above the anchor surface
                    dz = c_world[2] - b_min[2]
                    clearance_max = self.config.vertical_clearance_max_m
                    if -0.10 <= dz <= (b_max[2] - b_min[2]) + clearance_max:
                        if dz < best_dist_z:
                            best_dist_z = dz
                            best_cid = cid

            membership[track.object_id] = best_cid
            contexts[best_cid].member_track_ids.add(track.object_id)

        return membership
