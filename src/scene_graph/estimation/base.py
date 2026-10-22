"""Base abstraction for pose estimators.

Enforces the backend-independent estimation interface:
SensorFrame -> BasePoseEstimator -> (PoseEstimate, EstimatorState)
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from scene_graph.data.pose_estimate import EstimatorState, PoseEstimate
from scene_graph.data.sensor_frame import SensorFrame


class BasePoseEstimator(ABC):
    """Abstract base class for all camera pose estimators and SLAM backends.

    Implementations must be decoupled from object perception, tracking,
    and downstream graph logic.
    """

    @abstractmethod
    def estimate(self, sensor_frame: SensorFrame) -> tuple[PoseEstimate, EstimatorState]:
        """Estimate 6-DoF camera pose and telemetry for the given acquisition frame.

        Args:
            sensor_frame: Immutable acquisition-level SensorFrame.

        Returns:
            Tuple of (PoseEstimate, EstimatorState).
        """
        pass

    @abstractmethod
    def reset(self) -> None:
        """Reset internal estimator tracking history, caches, and state machines."""
        pass
