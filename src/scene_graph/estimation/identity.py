"""Identity / Camera-Local pose estimator.

Provides transient identity transforms for camera-local tracking fallback.
Crucially marks poses as non-persistent so they never pollute the global world model.
"""

from __future__ import annotations

import numpy as np

from scene_graph.data.pose_estimate import EstimatorState, PoseEstimate, TrackingState
from scene_graph.data.sensor_frame import SensorFrame
from scene_graph.estimation.base import BasePoseEstimator


class IdentityEstimator(BasePoseEstimator):
    """Generates identity poses in the camera's local coordinate frame."""

    def __init__(self, frame_id: str = "camera_color_optical_frame"):
        self.frame_id = frame_id

    def estimate(self, sensor_frame: SensorFrame) -> tuple[PoseEstimate, EstimatorState]:
        ts = sensor_frame.timestamp.value
        pose = PoseEstimate(
            world_T_camera=np.eye(4, dtype=np.float64),
            timestamp=ts,
            valid=True,
            age=0.0,
            source="camera_local",
            covariance=None,
            frame_id=sensor_frame.frame_id or self.frame_id,
            pose_timestamp=ts,
            target_timestamp=ts,
            lookup_mode="local_identity",
            transform_source="camera_local",
        )
        state = EstimatorState(
            backend_name="camera_local",
            tracking_state=TrackingState.TRACKING,
            matches=0,
            inliers=0,
            inlier_ratio=1.0,
            features=0,
            telemetry_timestamp=ts,
            telemetry_age=0.0,
        )
        return pose, state

    def reset(self) -> None:
        pass
