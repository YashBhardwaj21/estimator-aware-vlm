from typing import Dict, List, Optional, Tuple

import numpy as np

from scene_graph.config import SceneGraphConfig
from scene_graph.tracking.track import Track
from scene_graph.relations.base import RelationModule
from scene_graph.relations.context import FrameContext
from scene_graph.relations.evidence import RelationEvidence, EvidenceResult, ReferenceFrameType
from scene_graph.geometry.plane import fit_plane_ransac
from scene_graph.geometry.spatial_surface import SurfaceManager


class SupportRelationModule(RelationModule):

    def __init__(self, config: SceneGraphConfig):
        if config is None:
            raise ValueError("SceneGraphConfig is required.")

        if config.relations is None:
            raise ValueError("Relation configuration is required.")

        if config.geometry is None:
            raise ValueError("Geometry configuration is required.")

        support_config = config.relations.support
        plane_config = config.geometry.plane_ransac

        self.plane_residual_m = float(support_config.plane_residual_m)
        self.min_support_overlap = float(support_config.min_support_overlap)
        self.contact_tolerance_m = float(support_config.contact_tolerance_m)
        self.min_contact_density = float(support_config.min_contact_density)
        self.min_plane_points = int(support_config.min_plane_points)
        self.min_plane_alignment_cosine = float(support_config.min_plane_alignment_cosine)

        self.plane_max_iterations = int(plane_config.max_iterations)
        self.plane_min_inliers = int(plane_config.min_inliers)
        self.plane_random_seed = int(plane_config.random_seed)

        if self.plane_residual_m <= 0.0:
            raise ValueError("Support plane_residual_m must be positive.")

        if not 0.0 < self.min_support_overlap <= 1.0:
            raise ValueError("Support min_support_overlap must be in (0, 1].")

        if self.contact_tolerance_m <= 0.0:
            raise ValueError("Support contact_tolerance_m must be positive.")

        if not 0.0 < self.min_contact_density <= 1.0:
            raise ValueError("Support min_contact_density must be in (0, 1].")

        if self.min_plane_points < 3:
            raise ValueError("Support min_plane_points must be at least 3.")

        if not 0.0 < self.min_plane_alignment_cosine <= 1.0:
            raise ValueError("Support min_plane_alignment_cosine must be in (0, 1].")

        if self.plane_max_iterations < 1:
            raise ValueError("Plane RANSAC max_iterations must be positive.")

        if self.plane_min_inliers < 3:
            raise ValueError("Plane RANSAC min_inliers must be at least 3.")

        if self.plane_random_seed < 0:
            raise ValueError("Plane RANSAC random_seed must be non-negative.")

        self.surface_manager = SurfaceManager()
        self._plane_cache: Dict[Tuple[str, int], Tuple[np.ndarray, float, np.ndarray, Any]] = {}
        self._last_frame_index: Optional[int] = None

    def predicates(self) -> List[str]:
        return ["ON"]

    def compute(
        self,
        subject: Track,
        object_: Track,
        context: FrameContext,
    ) -> List[RelationEvidence]:

        if subject.object_id == object_.object_id:
            return []

        subject_geometry = context.observation_geometry.get(subject.object_id)
        support_geometry = context.observation_geometry.get(object_.object_id)

        if subject_geometry is None or support_geometry is None:
            return []

        subject_points = self._get_points(subject_geometry)
        support_points = self._get_points(support_geometry)

        if subject_points is None or support_points is None:
            return []

        if len(subject_points) == 0:
            return []

        up_axis = self._get_axis(context.reference_frame, "up_axis_world")

        if up_axis is None:
            return []

        if self._last_frame_index != context.frame_index:
            self._plane_cache.clear()
            self._last_frame_index = context.frame_index

        cache_key = (object_.object_id, context.frame_index)
        if cache_key in self._plane_cache:
            normal, plane_distance, support_plane_points, plane = self._plane_cache[cache_key]
        else:
            normal = None
            plane_distance = None
            support_plane_points = None
            plane = None

            if support_points is not None and len(support_points) >= self.min_plane_points:
                rng = np.random.default_rng(self.plane_random_seed)

                plane = fit_plane_ransac(
                    support_points,
                    distance_threshold=self.plane_residual_m,
                    max_iterations=self.plane_max_iterations,
                    min_inliers=self.plane_min_inliers,
                    rng=rng,
                )

                if (
                    plane is not None
                    and len(plane.inlier_mask) == len(support_points)
                    and plane.residual_mean <= self.plane_residual_m
                ):
                    cand_plane_points = support_points[plane.inlier_mask]
                    if len(cand_plane_points) >= self.min_plane_points:
                        cand_normal = np.asarray(plane.normal, dtype=np.float64)
                        if cand_normal.shape == (3,) and np.isfinite(cand_normal).all():
                            normal_norm = float(np.linalg.norm(cand_normal))
                            if normal_norm > np.finfo(float).eps:
                                cand_normal /= normal_norm
                                alignment = float(np.dot(cand_normal, up_axis))
                                if abs(alignment) >= self.min_plane_alignment_cosine:
                                    if alignment < 0.0:
                                        normal = -cand_normal
                                        plane_distance = -float(plane.distance)
                                    else:
                                        normal = cand_normal
                                        plane_distance = float(plane.distance)
                                    support_plane_points = cand_plane_points
                                    self.surface_manager.register_or_update(
                                        object_.object_id,
                                        normal,
                                        plane_distance,
                                        support_plane_points,
                                        context.timestamp,
                                    )

            # Fallback to persistent surface if current RANSAC fit was not available
            if normal is None or plane_distance is None or support_plane_points is None:
                surf = self.surface_manager.get_surface(
                    object_.object_id,
                    context.timestamp,
                    max_age_s=2.0,
                )
                if surf is not None and surf.plane_points is not None:
                    normal = surf.normal
                    plane_distance = surf.distance
                    support_plane_points = surf.plane_points

            if normal is not None and plane_distance is not None and support_plane_points is not None:
                self._plane_cache[cache_key] = (normal, plane_distance, support_plane_points, plane)

        if normal is None or plane_distance is None or support_plane_points is None:
            return []



        subject_heights = subject_points @ normal + plane_distance

        if not np.isfinite(subject_heights).all():
            return []

        lower_height = float(np.percentile(subject_heights, 5))
        upper_height = float(np.percentile(subject_heights, 95))

        if lower_height < -self.contact_tolerance_m:
            return []

        contact_mask = (
            (subject_heights >= 0.0)
            & (subject_heights <= self.contact_tolerance_m)
        )

        contact_point_count = int(np.count_nonzero(contact_mask))

        if contact_point_count == 0:
            return []

        contact_density = float(contact_point_count / len(subject_points))

        if contact_density < self.min_contact_density:
            return []

        subject_footprint = self._project_to_plane(
            subject_points,
            normal,
            up_axis,
        )

        support_footprint = self._project_to_plane(
            support_plane_points,
            normal,
            up_axis,
        )

        if subject_footprint is None or support_footprint is None:
            return []

        overlap_ratio = self._footprint_overlap(
            subject_footprint,
            support_footprint,
        )

        if overlap_ratio < self.min_support_overlap:
            return []

        contact_heights = subject_heights[contact_mask]
        contact_distance = float(np.median(np.abs(contact_heights)))

        confidence = self._confidence(
            contact_distance,
            self.contact_tolerance_m,
            overlap_ratio,
            contact_density,
        )

        return [
            RelationEvidence(
                predicate="ON",
                subject_id=subject.object_id,
                object_id=object_.object_id,
                frame_index=context.frame_index,
                timestamp=context.timestamp,
                result=EvidenceResult.SUPPORTED,
                value=contact_distance,
                threshold=self.contact_tolerance_m,
                confidence=confidence,
                reference_frame=ReferenceFrameType.WORLD,
                evidence_type="support_plane_contact",
                details={
                    "contact_distance_m": contact_distance,
                    "contact_density": float(contact_density),
                    "support_overlap": float(overlap_ratio),
                    "plane_residual_mean": float(plane.residual_mean) if plane is not None else 0.0,
                    "plane_residual_std": float(plane.residual_std) if plane is not None else 0.0,
                    "plane_alignment_cosine": float(abs(np.dot(normal, up_axis))),
                    "plane_normal": normal.tolist(),
                    "plane_distance": float(plane_distance),
                    "subject_lower_height_m": lower_height,
                    "subject_upper_height_m": upper_height,
                    "subject_point_count": int(len(subject_points)),
                    "support_point_count": int(len(support_points)) if support_points is not None else 0,
                    "support_plane_point_count": int(len(support_plane_points)),
                    "contact_point_count": contact_point_count,
                    "plane_ransac_max_iterations": self.plane_max_iterations,
                    "plane_ransac_min_inliers": self.plane_min_inliers,
                    "plane_ransac_random_seed": self.plane_random_seed,
                },
            )
        ]

    @staticmethod
    def _get_points(geometry) -> np.ndarray | None:
        points = geometry.points_world_sampled

        if points is None:
            points = geometry.points_world

        if points is None:
            return None

        points = np.asarray(points, dtype=np.float64)

        if points.ndim != 2 or points.shape[1] != 3:
            return None

        if not np.isfinite(points).all():
            return None

        return points

    @staticmethod
    def _get_axis(reference_frame, attribute: str) -> np.ndarray | None:
        if reference_frame is None:
            return None

        axis = getattr(reference_frame, attribute, None)

        if axis is None:
            return None

        axis = np.asarray(axis, dtype=np.float64)

        if axis.shape != (3,) or not np.isfinite(axis).all():
            return None

        norm = float(np.linalg.norm(axis))

        if norm <= np.finfo(float).eps:
            return None

        return axis / norm

    @staticmethod
    def _project_to_plane(
        points: np.ndarray,
        normal: np.ndarray,
        up_axis: np.ndarray,
    ) -> np.ndarray | None:

        plane_u = up_axis - np.dot(up_axis, normal) * normal
        u_norm = float(np.linalg.norm(plane_u))

        if u_norm <= np.finfo(float).eps:
            candidate = np.array([1.0, 0.0, 0.0]) if abs(normal[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
            plane_u = candidate - np.dot(candidate, normal) * normal
            u_norm = float(np.linalg.norm(plane_u))

        if u_norm <= np.finfo(float).eps:
            return None

        plane_u /= u_norm

        plane_v = np.cross(normal, plane_u)
        v_norm = float(np.linalg.norm(plane_v))

        if v_norm <= np.finfo(float).eps:
            return None

        plane_v /= v_norm

        return np.column_stack(
            (
                points @ plane_u,
                points @ plane_v,
            )
        )

    @staticmethod
    def _footprint_overlap(
        subject_points: np.ndarray,
        support_points: np.ndarray,
    ) -> float:

        if len(subject_points) == 0 or len(support_points) == 0:
            return 0.0

        subject_min = np.min(subject_points, axis=0)
        subject_max = np.max(subject_points, axis=0)

        support_min = np.min(support_points, axis=0)
        support_max = np.max(support_points, axis=0)

        overlap_min = np.maximum(subject_min, support_min)
        overlap_max = np.minimum(subject_max, support_max)

        overlap_size = np.maximum(
            overlap_max - overlap_min,
            0.0,
        )

        overlap_area = float(
            overlap_size[0] * overlap_size[1]
        )

        subject_size = np.maximum(
            subject_max - subject_min,
            0.0,
        )

        subject_area = float(
            subject_size[0] * subject_size[1]
        )

        if subject_area <= np.finfo(float).eps:
            return 0.0

        return float(
            np.clip(
                overlap_area / subject_area,
                0.0,
                1.0,
            )
        )

    @staticmethod
    def _confidence(
        contact_distance: float,
        contact_tolerance_m: float,
        overlap_ratio: float,
        contact_density: float,
    ) -> float:

        if contact_tolerance_m <= 0.0:
            return 0.0

        distance_score = float(
            np.clip(
                1.0 - contact_distance / contact_tolerance_m,
                0.0,
                1.0,
            )
        )

        return float(
            np.clip(
                (
                    distance_score
                    + overlap_ratio
                    + contact_density
                ) / 3.0,
                0.0,
                1.0,
            )
        )