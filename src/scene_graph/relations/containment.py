import numpy as np
from typing import List

from scene_graph.config import SceneGraphConfig
from scene_graph.tracking.track import Track
from scene_graph.relations.base import RelationModule
from scene_graph.relations.context import FrameContext
from scene_graph.relations.evidence import RelationEvidence, EvidenceResult, ReferenceFrameType


class ContainmentRelationModule(RelationModule):
    """Computes INSIDE relation from subject points relative to a container OBB."""

    def __init__(self, config: SceneGraphConfig):
        if config is None:
            raise ValueError("SceneGraphConfig is required.")
        if config.relations is None:
            raise ValueError("Relation configuration is required.")
        if config.relations.containment is None:
            raise ValueError("Containment relation configuration is required.")

        self.min_containment_ratio = float(config.relations.containment.min_containment_ratio)

        if not 0.0 < self.min_containment_ratio <= 1.0:
            raise ValueError("min_containment_ratio must be in (0, 1].")

    def predicates(self) -> List[str]:
        return ["INSIDE"]

    def compute(self, subject: Track, object_: Track, context: FrameContext) -> List[RelationEvidence]:
        if subject.object_id == object_.object_id:
            return []

        subject_geometry = context.observation_geometry.get(subject.object_id)
        object_geometry = context.observation_geometry.get(object_.object_id)

        if subject_geometry is None or object_geometry is None:
            return []

        points = self._get_points(subject_geometry)
        if points is None or len(points) == 0:
            return []

        obb_center = self._get_vector(object_geometry.obb_center_world)
        obb_axes = self._get_matrix(object_geometry.obb_axes_world)
        obb_extents = self._get_vector(object_geometry.obb_extents_world)

        if obb_center is None or obb_axes is None or obb_extents is None:
            return []

        if np.any(obb_extents <= 0.0):
            return []

        if not np.isfinite(obb_extents).all():
            return []

        if not self._valid_rotation_basis(obb_axes):
            return []

        relative_points = points - obb_center
        local_points = relative_points @ obb_axes

        half_extents = obb_extents / 2.0
        inside_mask = np.all(np.abs(local_points) <= half_extents, axis=1)

        containment_ratio = float(np.mean(inside_mask))

        if containment_ratio >= self.min_containment_ratio:
            result = EvidenceResult.SUPPORTED
            confidence = containment_ratio
        else:
            return []

        return [
            RelationEvidence(
                predicate="INSIDE",
                subject_id=subject.object_id,
                object_id=object_.object_id,
                frame_index=context.frame_index,
                timestamp=context.timestamp,
                result=result,
                value=containment_ratio,
                threshold=self.min_containment_ratio,
                confidence=float(np.clip(confidence, 0.0, 1.0)),
                reference_frame=ReferenceFrameType.WORLD,
                evidence_type="obb_point_containment",
                details={
                    "containment_ratio": containment_ratio,
                    "subject_point_count": int(len(points)),
                    "inside_point_count": int(np.sum(inside_mask)),
                    "container_obb_center_world": obb_center.tolist(),
                    "container_obb_extents_world": obb_extents.tolist(),
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
    def _get_vector(value) -> np.ndarray | None:
        if value is None:
            return None

        vector = np.asarray(value, dtype=np.float64)

        if vector.shape != (3,):
            return None

        if not np.isfinite(vector).all():
            return None

        return vector

    @staticmethod
    def _get_matrix(value) -> np.ndarray | None:
        if value is None:
            return None

        matrix = np.asarray(value, dtype=np.float64)

        if matrix.shape != (3, 3):
            return None

        if not np.isfinite(matrix).all():
            return None

        return matrix

    @staticmethod
    def _valid_rotation_basis(axes: np.ndarray) -> bool:
        if not np.allclose(axes.T @ axes, np.eye(3), atol=1e-5):
            return False

        determinant = float(np.linalg.det(axes))

        return np.isfinite(determinant) and abs(abs(determinant) - 1.0) <= 1e-5