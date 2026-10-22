from typing import Dict, List, Tuple

import numpy as np
from scipy.spatial import cKDTree

from scene_graph.config import SceneGraphConfig
from scene_graph.tracking.track import Track
from scene_graph.relations.base import RelationModule
from scene_graph.relations.context import FrameContext
from scene_graph.relations.evidence import (
    RelationEvidence,
    EvidenceResult,
    ReferenceFrameType,
)


class DistanceRelationModule(RelationModule):

    def __init__(self, config: SceneGraphConfig):
        if config is None:
            raise ValueError("SceneGraphConfig is required.")

        if config.relations is None:
            raise ValueError("Relation configuration is required.")

        distance_config = config.relations.distance

        self.near_threshold = float(distance_config.near_threshold)
        self.far_threshold = float(distance_config.far_threshold)

        if self.near_threshold <= 0.0:
            raise ValueError("near_threshold must be positive.")

        if self.far_threshold <= 0.0:
            raise ValueError("far_threshold must be positive.")

        if self.near_threshold >= self.far_threshold:
            raise ValueError("near_threshold must be smaller than far_threshold.")

        self._tree_cache: Dict[Tuple[str, int], cKDTree] = {}
        self._last_frame_index: Optional[int] = None

    def predicates(self) -> List[str]:
        return ["NEAR", "FAR"]

    def compute_pairs(self, pairs: List[Tuple[Track, Track]], context: FrameContext) -> List[RelationEvidence]:
        canonical_pairs = [(subject, object_) for subject, object_ in pairs if subject.object_id < object_.object_id]
        canonical_pairs.sort(key=lambda pair: (pair[0].object_id, pair[1].object_id))

        evidences: List[RelationEvidence] = []

        for subject, object_ in canonical_pairs:
            evidences.extend(self.compute(subject, object_, context))

        return evidences

    def compute(self, subject: Track, object_: Track, context: FrameContext) -> List[RelationEvidence]:
        if subject.object_id == object_.object_id:
            return []

        subject_geometry = context.observation_geometry.get(subject.object_id)
        object_geometry = context.observation_geometry.get(object_.object_id)

        if subject_geometry is None or object_geometry is None:
            return []

        subject_points = self._get_points(subject_geometry)
        object_points = self._get_points(object_geometry)

        if subject_points is None or object_points is None:
            return []

        if len(subject_points) == 0 or len(object_points) == 0:
            return []

        if self._last_frame_index != context.frame_index:
            self._tree_cache.clear()
            self._last_frame_index = context.frame_index

        cache_key = (object_.object_id, context.frame_index)
        tree = self._tree_cache.get(cache_key)
        if tree is None:
            tree = cKDTree(object_points)
            self._tree_cache[cache_key] = tree

        distances, _ = tree.query(subject_points, k=1)

        if distances.size == 0:
            return []

        distance_m = float(np.min(distances))

        if not np.isfinite(distance_m):
            return []

        if distance_m < self.near_threshold:
            predicate = "NEAR"
            threshold = self.near_threshold
            confidence = self._near_confidence(distance_m)

        elif distance_m > self.far_threshold:
            predicate = "FAR"
            threshold = self.far_threshold
            confidence = self._far_confidence(distance_m)

        else:
            return []

        return [
            RelationEvidence(
                predicate=predicate,
                subject_id=subject.object_id,
                object_id=object_.object_id,
                frame_index=context.frame_index,
                timestamp=context.timestamp,
                result=EvidenceResult.SUPPORTED,
                value=distance_m,
                threshold=threshold,
                confidence=confidence,
                reference_frame=ReferenceFrameType.WORLD,
                evidence_type="point_cloud_surface_distance",
                details={
                    "distance_m": distance_m,
                    "near_threshold_m": self.near_threshold,
                    "far_threshold_m": self.far_threshold,
                    "subject_point_count": int(len(subject_points)),
                    "object_point_count": int(len(object_points)),
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

    def _near_confidence(self, distance_m: float) -> float:
        return float(np.clip(1.0 - distance_m / self.near_threshold, 0.0, 1.0))

    def _far_confidence(self, distance_m: float) -> float:
        separation = distance_m - self.far_threshold
        scale = self.far_threshold

        return float(np.clip(separation / scale, 0.0, 1.0))