from typing import List

import numpy as np

from scene_graph.config import SceneGraphConfig
from scene_graph.tracking.track import Track
from scene_graph.relations.base import RelationModule
from scene_graph.relations.context import FrameContext
from scene_graph.relations.evidence import (
    RelationEvidence,
    EvidenceResult,
    ReferenceFrameType,
)


class DepthOrderRelationModule(RelationModule):

    def __init__(self, config: SceneGraphConfig):
        if config is None:
            raise ValueError("SceneGraphConfig is required.")

        if config.relations is None:
            raise ValueError("Relation configuration is required.")

        depth_config = config.relations.depth_order

        self.depth_margin = float(depth_config.depth_margin)
        self.uncertainty_sigma = float(depth_config.uncertainty_sigma)

        if self.depth_margin <= 0.0:
            raise ValueError("Depth margin must be positive.")

        if self.uncertainty_sigma <= 0.0:
            raise ValueError("Depth uncertainty_sigma must be positive.")

    def predicates(self) -> List[str]:
        return ["IN_FRONT_OF"]

    def compute(
        self,
        subject: Track,
        object_: Track,
        context: FrameContext,
    ) -> List[RelationEvidence]:

        subject_geometry = context.observation_geometry.get(subject.object_id)
        object_geometry = context.observation_geometry.get(object_.object_id)

        if subject_geometry is None or object_geometry is None:
            return []

        subject_centroid = self._get_centroid(subject, subject_geometry)
        object_centroid = self._get_centroid(object_, object_geometry)

        if subject_centroid is None or object_centroid is None:
            return []

        depth_axis = self._get_axis(
            context.reference_frame,
            "depth_axis_world",
        )

        if depth_axis is None:
            return []

        subject_covariance = self._get_covariance(subject_geometry)
        object_covariance = self._get_covariance(object_geometry)

        uncertainty_std = self._projected_uncertainty(
            axis=depth_axis,
            subject_covariance=subject_covariance,
            object_covariance=object_covariance,
        )

        if uncertainty_std is None:
            return []

        projected_difference = float(
            np.dot(
                object_centroid - subject_centroid,
                depth_axis,
            )
        )

        separation = projected_difference
        required_separation = (
            self.depth_margin
            + self.uncertainty_sigma * uncertainty_std
        )

        if separation > required_separation:
            result = EvidenceResult.SUPPORTED
            confidence = self._confidence(
                abs(separation),
                required_separation,
            )
        elif separation < -required_separation:
            result = EvidenceResult.CONTRADICTED
            confidence = self._confidence(
                abs(separation),
                required_separation,
            )
        else:
            return []

        normalized_separation = (
            separation / uncertainty_std
            if uncertainty_std > np.finfo(float).eps
            else None
        )

        return [
            RelationEvidence(
                predicate="IN_FRONT_OF",
                subject_id=subject.object_id,
                object_id=object_.object_id,
                frame_index=context.frame_index,
                timestamp=context.timestamp,
                result=result,
                value=float(separation),
                threshold=float(required_separation),
                confidence=confidence,
                reference_frame=ReferenceFrameType.CAMERA,
                evidence_type="uncertainty_aware_centroid_depth",
                details={
                    "separation_m": float(separation),
                    "projected_difference_m": float(projected_difference),
                    "depth_margin_m": float(self.depth_margin),
                    "uncertainty_std_m": float(uncertainty_std),
                    "uncertainty_sigma": float(self.uncertainty_sigma),
                    "required_separation_m": float(required_separation),
                    "normalized_separation": (
                        float(normalized_separation)
                        if normalized_separation is not None
                        else None
                    ),
                },
            )
        ]

    @staticmethod
    def _get_centroid(
        track: Track,
        geometry,
    ) -> np.ndarray | None:

        centroid = getattr(
            geometry,
            "centroid_world",
            None,
        )

        if centroid is None:
            centroid = getattr(
                track,
                "centroid_world",
                None,
            )

        if centroid is None:
            return None

        centroid = np.asarray(
            centroid,
            dtype=np.float64,
        )

        if centroid.shape != (3,):
            return None

        if not np.isfinite(centroid).all():
            return None

        return centroid

    @staticmethod
    def _get_covariance(
        geometry,
    ) -> np.ndarray | None:

        covariance = getattr(
            geometry,
            "position_covariance_world",
            None,
        )

        if covariance is None:
            return None

        covariance = np.asarray(
            covariance,
            dtype=np.float64,
        )

        if covariance.shape != (3, 3):
            return None

        if not np.isfinite(covariance).all():
            return None

        covariance = (
            covariance + covariance.T
        ) * 0.5

        eigenvalues = np.linalg.eigvalsh(covariance)

        if not np.isfinite(eigenvalues).all():
            return None

        if np.min(eigenvalues) < -1e-9:
            return None

        return covariance

    @staticmethod
    def _get_axis(
        reference_frame,
        attribute: str,
    ) -> np.ndarray | None:

        if reference_frame is None:
            return None

        axis = getattr(
            reference_frame,
            attribute,
            None,
        )

        if axis is None:
            return None

        axis = np.asarray(
            axis,
            dtype=np.float64,
        )

        if axis.shape != (3,):
            return None

        if not np.isfinite(axis).all():
            return None

        norm = float(np.linalg.norm(axis))

        if norm <= np.finfo(float).eps:
            return None

        return axis / norm

    @staticmethod
    def _projected_uncertainty(
        axis: np.ndarray,
        subject_covariance: np.ndarray | None,
        object_covariance: np.ndarray | None,
    ) -> float | None:

        if (
            subject_covariance is None
            and object_covariance is None
        ):
            return 0.0

        if (
            subject_covariance is None
            or object_covariance is None
        ):
            return None

        covariance = (
            subject_covariance
            + object_covariance
        )

        variance = float(
            axis @ covariance @ axis
        )

        if not np.isfinite(variance):
            return None

        return float(
            np.sqrt(
                max(
                    variance,
                    0.0,
                )
            )
        )

    @staticmethod
    def _confidence(
        separation: float,
        required_separation: float,
    ) -> float:

        if required_separation <= 0.0:
            return 0.0

        return float(
            np.clip(
                (separation - required_separation)
                / required_separation,
                0.0,
                1.0,
            )
        )