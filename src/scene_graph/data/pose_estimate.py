"""Canonical Pose Estimation Contracts: PoseEstimate and EstimatorState.

Defines backend-independent data structures for camera localization and estimator health:
SensorFrame -> PoseEstimator -> PoseEstimate + EstimatorState -> FramePacket.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import numpy as np

from scene_graph.geometry.transforms import validate_se3_transform


class TrackingState(str, Enum):
    """Explicit state of the visual odometry or SLAM pose estimator."""
    INITIALIZING = "initializing"
    TRACKING = "tracking"
    LOST = "lost"
    RECOVERED = "recovered"


@dataclass(frozen=True)
class PoseEstimate:
    """Rigid 6-DoF SE(3) pose estimate of the camera in world coordinates.

    Represents camera spatial location at timestamp t with provenance and covariance.
    """
    world_T_camera: np.ndarray | None
    timestamp: float
    valid: bool
    age: float = 0.0
    source: str = "unknown"
    covariance: np.ndarray | None = None
    frame_id: str = "world"

    # Provenance metadata
    pose_timestamp: float | None = None
    target_timestamp: float | None = None
    lookup_mode: str = "exact"
    transform_source: str = "unknown"
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not np.isfinite(self.timestamp) or self.timestamp < 0.0:
            raise ValueError(f"PoseEstimate.timestamp must be finite and >= 0, got {self.timestamp}")

        if self.target_timestamp is None:
            object.__setattr__(self, "target_timestamp", self.timestamp)

        if self.valid:
            if self.world_T_camera is None:
                raise ValueError("PoseEstimate cannot be valid with world_T_camera=None")
            if not isinstance(self.world_T_camera, np.ndarray) or self.world_T_camera.shape != (4, 4):
                raise ValueError(f"PoseEstimate.world_T_camera must have shape (4, 4), got {getattr(self.world_T_camera, 'shape', type(self.world_T_camera))}")
            validate_se3_transform(self.world_T_camera)

            if self.age < -1e-4:
                raise ValueError(f"PoseEstimate.age cannot be negative for valid causal poses, got {self.age}")

        if self.covariance is not None:
            if not isinstance(self.covariance, np.ndarray):
                raise TypeError(f"PoseEstimate.covariance must be an ndarray, got {type(self.covariance)}")
            if self.covariance.shape == (36,):
                object.__setattr__(self, "covariance", self.covariance.reshape(6, 6))
            elif self.covariance.shape != (6, 6):
                raise ValueError(f"PoseEstimate.covariance must be (6, 6) or (36,), got {self.covariance.shape}")
            if not np.all(np.isfinite(self.covariance)):
                raise ValueError("PoseEstimate.covariance must contain only finite numbers")

    @classmethod
    def invalid(
        cls,
        timestamp: float,
        source: str = "missing_tf",
        target_timestamp: float | None = None,
        transform_source: str = "missing_tf",
        frame_id: str = "world",
    ) -> "PoseEstimate":
        """Factory for an explicitly invalid pose estimate."""
        return cls(
            world_T_camera=None,
            timestamp=timestamp,
            valid=False,
            age=float("inf"),
            source=source,
            covariance=None,
            frame_id=frame_id,
            pose_timestamp=None,
            target_timestamp=target_timestamp or timestamp,
            lookup_mode="none",
            transform_source=transform_source,
        )

    @classmethod
    def identity(
        cls,
        timestamp: float,
        frame_id: str = "camera_color_optical_frame",
        source: str = "camera_local",
    ) -> "PoseEstimate":
        """Factory for a camera-local identity pose estimate."""
        return cls(
            world_T_camera=np.eye(4, dtype=np.float64),
            timestamp=timestamp,
            valid=True,
            age=0.0,
            source=source,
            covariance=None,
            frame_id=frame_id,
            pose_timestamp=timestamp,
            target_timestamp=timestamp,
            lookup_mode="local_identity",
            transform_source="camera_local",
        )

    @property
    def translation(self) -> np.ndarray:
        """Translation vector (tx, ty, tz) in metres."""
        if self.world_T_camera is not None:
            return self.world_T_camera[:3, 3]
        return np.zeros(3, dtype=np.float64)

    @property
    def rotation(self) -> np.ndarray:
        """3x3 rotation matrix."""
        if self.world_T_camera is not None:
            return self.world_T_camera[:3, :3]
        return np.eye(3, dtype=np.float64)

    @property
    def is_finite(self) -> bool:
        """Check if pose matrix contains only finite floats."""
        if self.world_T_camera is not None:
            return bool(np.all(np.isfinite(self.world_T_camera)))
        return True


@dataclass(frozen=True)
class EstimatorState:
    """Telemetry and health state of the visual odometry or SLAM pose estimator.

    Carries backend diagnostic quantities, registration metrics, and tracking states.
    """
    backend_name: str
    tracking_state: TrackingState = TrackingState.INITIALIZING
    matches: int = 0
    inliers: int = 0
    inlier_ratio: float = 0.0
    features: int = 0
    local_map_size: int = 0
    local_key_frames: int = 0
    key_frame_added: bool = False
    registration_time_ms: float = 0.0
    total_estimator_time_ms: float = 0.0
    odometry_lost: bool = False
    current_node_id: int = -1
    loop_closure_id: int = 0
    proximity_detection_id: int = 0
    landmark_id: int = 0
    tracking_recovered: bool = False
    loop_closure_detected: bool = False
    telemetry_timestamp: float | None = None
    telemetry_age: float = 0.0
    covariance_available: bool = False
    covariance_frame: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.tracking_state, TrackingState):
            object.__setattr__(self, "tracking_state", TrackingState(str(self.tracking_state)))

        # Validate inlier ratio consistency
        if self.inlier_ratio == 0.0 and self.features > 0 and self.inliers > 0:
            object.__setattr__(self, "inlier_ratio", float(self.inliers) / float(self.features))

    @property
    def is_healthy(self) -> bool:
        """Returns True if the estimator is actively tracking and not lost."""
        return (self.tracking_state in (TrackingState.TRACKING, TrackingState.RECOVERED)) and not self.odometry_lost

    @classmethod
    def initializing(cls, backend_name: str = "rtabmap") -> "EstimatorState":
        return cls(backend_name=backend_name, tracking_state=TrackingState.INITIALIZING)

    @classmethod
    def lost(cls, backend_name: str = "rtabmap", odometry_lost: bool = True) -> "EstimatorState":
        return cls(backend_name=backend_name, tracking_state=TrackingState.LOST, odometry_lost=odometry_lost)

    @classmethod
    def camera_local(cls) -> "EstimatorState":
        return cls(backend_name="camera_local", tracking_state=TrackingState.TRACKING)
