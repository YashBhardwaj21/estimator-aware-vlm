from typing import List

import numpy as np

from scene_graph.config import SceneGraphConfig
from scene_graph.tracking.track import Track
from scene_graph.relations.base import RelationModule
from scene_graph.relations.context import FrameContext
from scene_graph.relations.evidence import RelationEvidence, EvidenceResult, ReferenceFrameType
from scene_graph.geometry.provenance import GeometrySource


class OcclusionRelationModule(RelationModule):

    def __init__(self, config: SceneGraphConfig):
        if config is None:
            raise ValueError("SceneGraphConfig is required.")

        if config.relations is None:
            raise ValueError("Relation configuration is required.")

        occlusion_config = config.relations.occlusion

        self.min_mask_overlap_ratio = float(occlusion_config.min_mask_overlap_ratio)
        self.min_depth_order_ratio = float(occlusion_config.min_depth_order_ratio)
        self.min_valid_depth_samples = int(occlusion_config.min_valid_depth_samples)
        self.depth_margin = float(occlusion_config.depth_margin)

        if not 0.0 < self.min_mask_overlap_ratio <= 1.0:
            raise ValueError("min_mask_overlap_ratio must be in (0, 1].")

        if not 0.0 < self.min_depth_order_ratio <= 1.0:
            raise ValueError("min_depth_order_ratio must be in (0, 1].")

        if self.min_valid_depth_samples < 1:
            raise ValueError("min_valid_depth_samples must be positive.")

        if self.depth_margin <= 0.0:
            raise ValueError("depth_margin must be positive.")

    def predicates(self) -> List[str]:
        return ["OCCLUDING"]

    def compute(
        self,
        subject: Track,
        object_: Track,
        context: FrameContext,
    ) -> List[RelationEvidence]:

        if subject.object_id == object_.object_id:
            return []

        if context.intrinsics is None or context.depth_image is None:
            return []

        subject_geometry = context.observation_geometry.get(subject.object_id)
        object_geometry = context.observation_geometry.get(object_.object_id)

        if subject_geometry is None or object_geometry is None:
            return []

        if (
            subject_geometry.geometry_source != GeometrySource.OBSERVED
            or object_geometry.geometry_source != GeometrySource.OBSERVED
        ):
            return []

        if subject_geometry.mask is None or object_geometry.mask is None:
            return [self._missing_mask(subject, object_, context)]

        subject_mask = np.asarray(subject_geometry.mask, dtype=bool)
        object_mask = np.asarray(object_geometry.mask, dtype=bool)
        depth_image = np.asarray(context.depth_image)

        if subject_mask.ndim != 2 or object_mask.ndim != 2 or depth_image.ndim != 2:
            return []

        if subject_mask.shape != object_mask.shape or subject_mask.shape != depth_image.shape:
            return []

        subject_area = int(np.count_nonzero(subject_mask))
        object_area = int(np.count_nonzero(object_mask))

        if subject_area == 0 or object_area == 0:
            return []

        overlap_mask = subject_mask & object_mask
        overlap_count = int(np.count_nonzero(overlap_mask))

        if overlap_count == 0:
            return []

        overlap_ratio = float(overlap_count / min(subject_area, object_area))

        if overlap_ratio < self.min_mask_overlap_ratio:
            return []

        subject_points_camera = self._get_camera_points(subject_geometry)
        object_points_camera = self._get_camera_points(object_geometry)

        if subject_points_camera is None or object_points_camera is None:
            return [self._insufficient_depth(subject, object_, context, overlap_ratio, 0)]

        subject_depth = self._build_depth_buffer(
            subject_points_camera,
            subject_mask.shape,
            context.intrinsics,
        )

        object_depth = self._build_depth_buffer(
            object_points_camera,
            object_mask.shape,
            context.intrinsics,
        )

        raw_depth = depth_image.astype(np.float64, copy=False)
        valid_sensor_depth = np.isfinite(raw_depth) & (raw_depth > 0.0)

        valid_overlap = overlap_mask & valid_sensor_depth & np.isfinite(subject_depth) & np.isfinite(object_depth)

        valid_depth_samples = int(np.count_nonzero(valid_overlap))

        if valid_depth_samples < self.min_valid_depth_samples:
            return [self._insufficient_depth(subject, object_, context, overlap_ratio, valid_depth_samples)]

        subject_z = subject_depth[valid_overlap]
        object_z = object_depth[valid_overlap]
        sensor_z = raw_depth[valid_overlap]

        subject_closer = subject_z + self.depth_margin < object_z
        object_closer = object_z + self.depth_margin < subject_z

        subject_in_front_count = int(np.count_nonzero(subject_closer))
        object_in_front_count = int(np.count_nonzero(object_closer))

        subject_depth_order_ratio = float(subject_in_front_count / valid_depth_samples)
        object_depth_order_ratio = float(object_in_front_count / valid_depth_samples)

        if subject_depth_order_ratio < self.min_depth_order_ratio:
            return []

        subject_front_difference = (object_z - subject_z)[subject_closer]

        if subject_front_difference.size == 0:
            return []

        median_depth_difference = float(np.median(subject_front_difference))

        sensor_depth_residual = np.abs(
            sensor_z - np.minimum(subject_z, object_z)
        )

        median_sensor_depth_residual = float(np.median(sensor_depth_residual))

        confidence = self._confidence(
            overlap_ratio,
            subject_depth_order_ratio,
            median_depth_difference,
            self.depth_margin,
        )

        return [
            RelationEvidence(
                predicate="OCCLUDING",
                subject_id=subject.object_id,
                object_id=object_.object_id,
                frame_index=context.frame_index,
                timestamp=context.timestamp,
                result=EvidenceResult.SUPPORTED,
                value=subject_depth_order_ratio,
                threshold=self.min_depth_order_ratio,
                confidence=confidence,
                reference_frame=ReferenceFrameType.CAMERA,
                evidence_type="pixel_depth_occlusion",
                details={
                    "mask_overlap_ratio": overlap_ratio,
                    "valid_depth_samples": valid_depth_samples,
                    "subject_in_front_count": subject_in_front_count,
                    "object_in_front_count": object_in_front_count,
                    "subject_depth_order_ratio": subject_depth_order_ratio,
                    "object_depth_order_ratio": object_depth_order_ratio,
                    "depth_margin_m": self.depth_margin,
                    "median_subject_front_depth_difference_m": median_depth_difference,
                    "median_sensor_depth_residual_m": median_sensor_depth_residual,
                },
            )
        ]

    @staticmethod
    def _get_camera_points(geometry) -> np.ndarray | None:
        points = geometry.points_camera

        if points is None:
            return None

        points = np.asarray(points, dtype=np.float64)

        if points.ndim != 2 or points.shape[1] != 3:
            return None

        if not np.isfinite(points).all():
            return None

        points = points[points[:, 2] > 0.0]

        if len(points) == 0:
            return None

        return points

    @staticmethod
    def _build_depth_buffer(points_camera: np.ndarray, image_shape: tuple[int, int], intrinsics) -> np.ndarray:
        height, width = image_shape

        depth_buffer = np.full(
            (height, width),
            np.inf,
            dtype=np.float64,
        )

        z = points_camera[:, 2]

        u = intrinsics.fx * points_camera[:, 0] / z + intrinsics.cx
        v = intrinsics.fy * points_camera[:, 1] / z + intrinsics.cy

        valid = (
            np.isfinite(u)
            & np.isfinite(v)
            & np.isfinite(z)
            & (z > 0.0)
            & (u >= 0.0)
            & (u < width)
            & (v >= 0.0)
            & (v < height)
        )

        if not np.any(valid):
            return depth_buffer

        u_valid = np.rint(u[valid]).astype(np.int64)
        v_valid = np.rint(v[valid]).astype(np.int64)
        z_valid = z[valid]

        in_bounds = (
            (u_valid >= 0)
            & (u_valid < width)
            & (v_valid >= 0)
            & (v_valid < height)
        )

        if not np.any(in_bounds):
            return depth_buffer

        u_valid = u_valid[in_bounds]
        v_valid = v_valid[in_bounds]
        z_valid = z_valid[in_bounds]

        flat_indices = v_valid * width + u_valid

        np.minimum.at(
            depth_buffer.ravel(),
            flat_indices,
            z_valid,
        )

        return depth_buffer

    @staticmethod
    def _confidence(overlap_ratio: float, depth_order_ratio: float, depth_difference: float, depth_margin: float) -> float:
        if depth_margin <= 0.0:
            return 0.0

        depth_score = float(np.clip(depth_difference / depth_margin - 1.0, 0.0, 1.0))

        return float(np.clip((overlap_ratio + depth_order_ratio + depth_score) / 3.0, 0.0, 1.0))

    @staticmethod
    def _missing_mask(subject: Track, object_: Track, context: FrameContext) -> RelationEvidence:
        return RelationEvidence(
            predicate="OCCLUDING",
            subject_id=subject.object_id,
            object_id=object_.object_id,
            frame_index=context.frame_index,
            timestamp=context.timestamp,
            result=EvidenceResult.MISSING_MASK,
            value=None,
            threshold=None,
            confidence=0.0,
            reference_frame=ReferenceFrameType.CAMERA,
            evidence_type="missing_mask",
            details={},
        )

    @staticmethod
    def _insufficient_depth(
        subject: Track,
        object_: Track,
        context: FrameContext,
        overlap_ratio: float,
        valid_depth_samples: int,
    ) -> RelationEvidence:
        return RelationEvidence(
            predicate="OCCLUDING",
            subject_id=subject.object_id,
            object_id=object_.object_id,
            frame_index=context.frame_index,
            timestamp=context.timestamp,
            result=EvidenceResult.INSUFFICIENT_DEPTH,
            value=overlap_ratio,
            threshold=None,
            confidence=0.0,
            reference_frame=ReferenceFrameType.CAMERA,
            evidence_type="insufficient_pixel_depth",
            details={
                "mask_overlap_ratio": float(overlap_ratio),
                "valid_depth_samples": int(valid_depth_samples),
            },
        )