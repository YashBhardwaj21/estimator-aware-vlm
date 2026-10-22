"""TF-based pose estimator extracting camera localization from tf2 buffer.

Preserves the hardened causal lookup policy:
Exact TF lookup -> Causal latest fallback -> Future transform check -> Stale transform check -> SE(3) conversion.
"""

from __future__ import annotations

import time
from typing import Any, Protocol

import numpy as np

from scene_graph.data.pose_estimate import EstimatorState, PoseEstimate, TrackingState
from scene_graph.data.sensor_frame import SensorFrame
from scene_graph.estimation.base import BasePoseEstimator
from scene_graph.geometry.transforms import quaternion_to_matrix


class TFBufferProtocol(Protocol):
    """Structural protocol for tf2_ros.Buffer or test doubles."""
    def lookup_transform(self, target_frame: str, source_frame: str, time: Any, timeout: Any = None) -> Any: ...


def _extract_tf_transform(transform_stamped: Any) -> tuple[float, np.ndarray]:
    """Extract (timestamp, 4x4 SE(3) matrix) from TransformStamped message or mock."""
    if hasattr(transform_stamped, "header"):
        stamp = transform_stamped.header.stamp
        pose_ts = float(stamp.sec) + float(stamp.nanosec) * 1e-9
    else:
        pose_ts = getattr(transform_stamped, "timestamp", 0.0)

    transform = getattr(transform_stamped, "transform", transform_stamped)
    tx = float(transform.translation.x)
    ty = float(transform.translation.y)
    tz = float(transform.translation.z)

    qx = float(transform.rotation.x)
    qy = float(transform.rotation.y)
    qz = float(transform.rotation.z)
    qw = float(transform.rotation.w)

    R = quaternion_to_matrix(qx, qy, qz, qw)
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R
    T[:3, 3] = [tx, ty, tz]
    return pose_ts, T


class TFBufferEstimator(BasePoseEstimator):
    """Extracts globally referenced camera poses from a TF buffer."""

    def __init__(
        self,
        tf_buffer: Any | None = None,
        world_frame: str = "world",
        sensor_frame: str = "camera_optical_frame",
        pose_max_dt: float = 0.05,
        slam_pose_max_age: float = 2.0,
        allow_causal_fallback: bool = True,
        backend_name: str = "tf_buffer",
    ):
        self.tf_buffer = tf_buffer
        self.world_frame = world_frame
        self.sensor_frame = sensor_frame
        self.pose_max_dt = pose_max_dt
        self.slam_pose_max_age = slam_pose_max_age
        self.allow_causal_fallback = allow_causal_fallback
        self.backend_name = backend_name

        self._previous_state: TrackingState = TrackingState.INITIALIZING
        self._last_valid_pose: PoseEstimate | None = None

    def reset(self) -> None:
        self._previous_state = TrackingState.INITIALIZING
        self._last_valid_pose = None

    def estimate(self, sensor_frame: SensorFrame) -> tuple[PoseEstimate, EstimatorState]:
        rgb_ts = sensor_frame.timestamp.value
        child_frame = sensor_frame.frame_id or self.sensor_frame

        if self.tf_buffer is None:
            state = EstimatorState.initializing(backend_name=self.backend_name)
            pose = PoseEstimate.invalid(
                timestamp=rgb_ts,
                source=self.backend_name,
                target_timestamp=rgb_ts,
                transform_source="no_tf_buffer",
                frame_id=self.world_frame,
            )
            return pose, state

        t_start = time.monotonic()
        transform_stamped = None
        lookup_mode = "none"
        exact_succeeded = False
        source_tag = "unknown"

        # Lazy imports for ROS 2 time types if running under rclpy
        try:
            from rclpy.duration import Duration
            from rclpy.time import Time
            target_time = Time(seconds=int(rgb_ts), nanoseconds=int((rgb_ts % 1.0) * 1e9))
            zero_time = Time()
            zero_timeout = Duration(seconds=0)
            configured_timeout = Duration(seconds=self.pose_max_dt)
        except ImportError:
            # Standalone / non-ROS fallback
            target_time = rgb_ts
            zero_time = 0.0
            zero_timeout = 0.0
            configured_timeout = self.pose_max_dt

        target_frames = [self.world_frame]
        if self.world_frame != "odom":
            target_frames.append("odom")

        resolved_target_frame = self.world_frame

        # Step 1: Attempt exact or interpolated transform at timestamp T
        for frame_candidate in target_frames:
            try:
                timeout_to_use = zero_timeout if self.allow_causal_fallback else configured_timeout
                transform_stamped = self.tf_buffer.lookup_transform(
                    frame_candidate,
                    child_frame,
                    target_time,
                    timeout=timeout_to_use,
                )
                pose_ts, T = _extract_tf_transform(transform_stamped)
                raw_age = rgb_ts - pose_ts

                if raw_age < -1e-4:
                    # Future transform returned: strictly reject
                    source_tag = "future_tf_rejected"
                    transform_stamped = None
                elif raw_age > self.pose_max_dt:
                    # Stale exact transform
                    source_tag = "stale_tf"
                    transform_stamped = None
                else:
                    exact_succeeded = True
                    lookup_mode = "exact"
                    source_tag = "tf_exact"
                    resolved_target_frame = frame_candidate
                    break
            except Exception:
                transform_stamped = None

        # Step 2: Causal latest fallback if exact is not ready
        if not exact_succeeded and self.allow_causal_fallback:
            for frame_candidate in target_frames:
                try:
                    transform_stamped = self.tf_buffer.lookup_transform(
                        frame_candidate,
                        child_frame,
                        zero_time,
                        timeout=configured_timeout,
                    )
                    pose_ts, T = _extract_tf_transform(transform_stamped)

                    if pose_ts > rgb_ts + 1e-4:
                        source_tag = "future_tf_rejected"
                        transform_stamped = None
                    else:
                        age = max(0.0, rgb_ts - pose_ts)
                        if self.slam_pose_max_age > 0 and age > self.slam_pose_max_age:
                            source_tag = "stale_slam_tf"
                            transform_stamped = None
                        else:
                            lookup_mode = "causal_latest"
                            source_tag = "slam_causal_tf"
                            resolved_target_frame = frame_candidate
                            break
                except Exception:
                    source_tag = "missing_slam_tf"
                    transform_stamped = None

        latency_ms = (time.monotonic() - t_start) * 1000.0

        # Construct PoseEstimate
        if transform_stamped is not None and lookup_mode in ("exact", "causal_latest"):
            pose_ts, T = _extract_tf_transform(transform_stamped)
            pose_age = max(0.0, rgb_ts - pose_ts)
            pose_estimate = PoseEstimate(
                world_T_camera=T,
                timestamp=rgb_ts,
                valid=True,
                age=pose_age,
                source=self.backend_name,
                covariance=None,
                frame_id=resolved_target_frame,
                pose_timestamp=pose_ts,
                target_timestamp=rgb_ts,
                lookup_mode=lookup_mode,
                transform_source=source_tag,
            )
            # Tracking state machine
            if self._previous_state == TrackingState.LOST:
                current_state = TrackingState.RECOVERED
            elif self._previous_state in (TrackingState.RECOVERED, TrackingState.INITIALIZING):
                current_state = TrackingState.TRACKING
            else:
                current_state = TrackingState.TRACKING

            self._last_valid_pose = pose_estimate
        else:
            pose_estimate = PoseEstimate.invalid(
                timestamp=rgb_ts,
                source=self.backend_name,
                target_timestamp=rgb_ts,
                transform_source=source_tag,
                frame_id=self.world_frame,
            )
            if self._last_valid_pose is None:
                current_state = TrackingState.INITIALIZING
            else:
                current_state = TrackingState.LOST

        self._previous_state = current_state

        estimator_state = EstimatorState(
            backend_name=self.backend_name,
            tracking_state=current_state,
            matches=0,
            inliers=0,
            inlier_ratio=1.0 if pose_estimate.valid else 0.0,
            features=0,
            total_estimator_time_ms=latency_ms,
            odometry_lost=not pose_estimate.valid,
            tracking_recovered=(current_state == TrackingState.RECOVERED),
            telemetry_timestamp=pose_estimate.pose_timestamp,
            telemetry_age=pose_estimate.age if pose_estimate.valid else float("inf"),
            covariance_available=False,
            covariance_frame="",
        )

        return pose_estimate, estimator_state
