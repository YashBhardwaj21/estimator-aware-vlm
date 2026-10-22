"""RTAB-Map SLAM and visual odometry pose estimator.

Combines globally referenced TF transforms with RTAB-Map registration diagnostics:
- /tf: global map/world -> camera SE(3) pose
- /rtabmap/odom: odometry pose, velocity, spatial covariance
- /rtabmap/odom_info: features, matches, inliers, keyframe flags, registration time
- /rtabmap/info: current reference node ID, loop closure ID, proximity detection ID
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Any

import numpy as np

from scene_graph.data.pose_estimate import EstimatorState, PoseEstimate, TrackingState
from scene_graph.data.sensor_frame import SensorFrame
from scene_graph.estimation.tf_buffer import TFBufferEstimator


@dataclass(frozen=True)
class RTABMapOdomTelemetry:
    """Snapshot from /rtabmap/odom."""
    timestamp: float
    covariance: np.ndarray | None  # (6, 6)
    linear_velocity: np.ndarray | None  # (3,)
    angular_velocity: np.ndarray | None  # (3,)
    frame_id: str = "odom"
    transform: np.ndarray | None = None  # (4, 4) SE(3) pose matrix


@dataclass(frozen=True)
class RTABMapOdomInfoTelemetry:
    """Snapshot from /rtabmap/odom_info."""
    timestamp: float
    lost: bool = False
    matches: int = 0
    inliers: int = 0
    features: int = 0
    local_map_size: int = 0
    local_key_frames: int = 0
    key_frame_added: bool = False
    time_estimation_sec: float = 0.0
    covariance: np.ndarray | None = None  # (6, 6)
    distance_travelled: float = 0.0
    gravity_roll_error: float = 0.0
    gravity_pitch_error: float = 0.0


@dataclass(frozen=True)
class RTABMapInfoTelemetry:
    """Snapshot from /rtabmap/info."""
    timestamp: float
    ref_id: int = -1
    loop_closure_id: int = 0
    proximity_detection_id: int = 0
    landmark_id: int = 0
    wm_state_count: int = 0


class RTABMapEstimator(TFBufferEstimator):
    """Estimator integrating RTAB-Map visual odometry, TF, and registration telemetry."""

    def __init__(
        self,
        tf_buffer: Any | None = None,
        world_frame: str = "world",
        sensor_frame: str = "camera_optical_frame",
        pose_max_dt: float = 0.05,
        slam_pose_max_age: float = 0.20,      # Strict timeout: TF older than 200ms is stale
        telemetry_max_age: float = 0.25,      # Telemetry older than 250ms is stale
        allow_causal_fallback: bool = True,
        backend_name: str = "rtabmap",
        buffer_capacity: int = 100,
    ):
        super().__init__(
            tf_buffer=tf_buffer,
            world_frame=world_frame,
            sensor_frame=sensor_frame,
            pose_max_dt=pose_max_dt,
            slam_pose_max_age=slam_pose_max_age,
            allow_causal_fallback=allow_causal_fallback,
            backend_name=backend_name,
        )
        self.telemetry_max_age = telemetry_max_age
        self.odom_queue: deque[RTABMapOdomTelemetry] = deque(maxlen=buffer_capacity)
        self.odom_info_queue: deque[RTABMapOdomInfoTelemetry] = deque(maxlen=buffer_capacity)
        self.info_queue: deque[RTABMapInfoTelemetry] = deque(maxlen=buffer_capacity)

        self._last_processed_loop_id: int = 0
        self._loop_closure_active: bool = False
        self._rtabmap_tracking_state: TrackingState = TrackingState.INITIALIZING

    def reset(self) -> None:
        super().reset()
        self.odom_queue.clear()
        self.odom_info_queue.clear()
        self.info_queue.clear()
        self._last_processed_loop_id = 0
        self._loop_closure_active = False
        self._rtabmap_tracking_state = TrackingState.INITIALIZING
    # Telemetry Ingestion API (called by ROS subscribers or replay adapters)

    def add_odom_sample(
        self,
        timestamp: float,
        covariance: np.ndarray | Sequence[float] | None = None,
        linear_velocity: np.ndarray | None = None,
        angular_velocity: np.ndarray | None = None,
        frame_id: str = "odom",
        transform: np.ndarray | None = None,
    ) -> None:
        """Cache /rtabmap/odom or /odom sample."""
        cov_6x6 = None
        if covariance is not None:
            cov_arr = np.asarray(covariance, dtype=np.float64)
            if cov_arr.shape == (36,):
                cov_6x6 = cov_arr.reshape(6, 6)
            elif cov_arr.shape == (6, 6):
                cov_6x6 = cov_arr

        self.odom_queue.append(
            RTABMapOdomTelemetry(
                timestamp=float(timestamp),
                covariance=cov_6x6,
                linear_velocity=linear_velocity,
                angular_velocity=angular_velocity,
                frame_id=frame_id,
                transform=transform,
            )
        )

    def add_odom_info_sample(
        self,
        timestamp: float,
        lost: bool = False,
        matches: int = 0,
        inliers: int = 0,
        features: int = 0,
        local_map_size: int = 0,
        local_key_frames: int = 0,
        key_frame_added: bool = False,
        time_estimation_sec: float = 0.0,
        covariance: np.ndarray | Sequence[float] | None = None,
        distance_travelled: float = 0.0,
        gravity_roll_error: float = 0.0,
        gravity_pitch_error: float = 0.0,
    ) -> None:
        """Cache /rtabmap/odom_info sample."""
        cov_6x6 = None
        if covariance is not None:
            cov_arr = np.asarray(covariance, dtype=np.float64)
            if cov_arr.shape == (36,):
                cov_6x6 = cov_arr.reshape(6, 6)
            elif cov_arr.shape == (6, 6):
                cov_6x6 = cov_arr

        self.odom_info_queue.append(
            RTABMapOdomInfoTelemetry(
                timestamp=float(timestamp),
                lost=bool(lost),
                matches=int(matches),
                inliers=int(inliers),
                features=int(features),
                local_map_size=int(local_map_size),
                local_key_frames=int(local_key_frames),
                key_frame_added=bool(key_frame_added),
                time_estimation_sec=float(time_estimation_sec),
                covariance=cov_6x6,
                distance_travelled=float(distance_travelled),
                gravity_roll_error=float(gravity_roll_error),
                gravity_pitch_error=float(gravity_pitch_error),
            )
        )

    def add_info_sample(
        self,
        timestamp: float,
        ref_id: int = -1,
        loop_closure_id: int = 0,
        proximity_detection_id: int = 0,
        landmark_id: int = 0,
        wm_state_count: int = 0,
    ) -> None:
        """Cache /rtabmap/info sample."""
        self.info_queue.append(
            RTABMapInfoTelemetry(
                timestamp=float(timestamp),
                ref_id=int(ref_id),
                loop_closure_id=int(loop_closure_id),
                proximity_detection_id=int(proximity_detection_id),
                landmark_id=int(landmark_id),
                wm_state_count=int(wm_state_count),
            )
        )

    def add_odom_msg(self, msg: Any) -> None:
        """Ingest nav_msgs/Odometry message or duck-typed equivalent."""
        stamp = getattr(msg, "header", None)
        if stamp is None or not hasattr(stamp, "stamp"):
            return
        st = stamp.stamp
        ts = float(st.sec) + float(st.nanosec) * 1e-9
        pose_obj = getattr(msg, "pose", None)
        cov = getattr(pose_obj, "covariance", None) if pose_obj else None
        transform = None
        if pose_obj is not None and hasattr(pose_obj, "pose"):
            p = pose_obj.pose
            if hasattr(p, "position") and hasattr(p, "orientation"):
                pos = p.position
                ori = p.orientation
                from scene_graph.geometry.transforms import pose_to_transform
                transform = pose_to_transform(
                    float(pos.x), float(pos.y), float(pos.z),
                    float(ori.x), float(ori.y), float(ori.z), float(ori.w),
                )
        lin_vel = None
        ang_vel = None
        twist_obj = getattr(msg, "twist", None)
        if twist_obj is not None and hasattr(twist_obj, "twist"):
            tw = twist_obj.twist
            if hasattr(tw, "linear") and hasattr(tw, "angular"):
                lin_vel = np.array([tw.linear.x, tw.linear.y, tw.linear.z], dtype=np.float64)
                ang_vel = np.array([tw.angular.x, tw.angular.y, tw.angular.z], dtype=np.float64)
        frame_id = getattr(msg.header, "frame_id", "odom")
        self.add_odom_sample(
            timestamp=ts,
            covariance=cov,
            linear_velocity=lin_vel,
            angular_velocity=ang_vel,
            frame_id=frame_id,
            transform=transform,
        )

    def add_odom_info_msg(self, msg: Any) -> None:
        """Ingest rtabmap_msgs/OdomInfo message or duck-typed equivalent."""
        stamp = getattr(msg, "header", None)
        if stamp is None or not hasattr(stamp, "stamp"):
            return
        st = stamp.stamp
        ts = float(st.sec) + float(st.nanosec) * 1e-9
        self.add_odom_info_sample(
            timestamp=ts,
            lost=getattr(msg, "lost", False),
            matches=getattr(msg, "matches", 0),
            inliers=getattr(msg, "inliers", 0),
            features=getattr(msg, "features", 0),
            local_map_size=getattr(msg, "local_map_size", 0),
            local_key_frames=getattr(msg, "local_key_frames", 0),
            key_frame_added=getattr(msg, "key_frame_added", False),
            time_estimation_sec=getattr(msg, "time_estimation", 0.0),
            covariance=getattr(msg, "covariance", None),
            distance_travelled=getattr(msg, "distance_travelled", 0.0),
        )

    def add_info_msg(self, msg: Any) -> None:
        """Ingest rtabmap_msgs/Info message or duck-typed equivalent."""
        stamp = getattr(msg, "header", None)
        if stamp is None or not hasattr(stamp, "stamp"):
            return
        st = stamp.stamp
        ts = float(st.sec) + float(st.nanosec) * 1e-9
        self.add_info_sample(
            timestamp=ts,
            ref_id=getattr(msg, "ref_id", -1),
            loop_closure_id=getattr(msg, "loop_closure_id", 0),
            proximity_detection_id=getattr(msg, "proximity_detection_id", 0),
            landmark_id=getattr(msg, "landmark_id", 0),
            wm_state_count=getattr(msg, "wm_state_count", 0),
        )

    # Telemetry Matching (Enforces strict causal invariant: telemetry_ts <= frame_ts)

    def _get_causal_odom_info(self, frame_ts: float) -> tuple[RTABMapOdomInfoTelemetry | None, float]:
        """Find the newest causal odom_info record with timestamp <= frame_ts."""
        best: RTABMapOdomInfoTelemetry | None = None
        for item in reversed(self.odom_info_queue):
            if item.timestamp <= frame_ts + 1e-4:
                best = item
                break
        if best is None:
            return None, float("inf")
        age = max(0.0, frame_ts - best.timestamp)
        return best, age

    def _get_causal_odom(self, frame_ts: float) -> tuple[RTABMapOdomTelemetry | None, float]:
        """Find the newest causal odom record with timestamp <= frame_ts."""
        best: RTABMapOdomTelemetry | None = None
        for item in reversed(self.odom_queue):
            if item.timestamp <= frame_ts + 1e-4:
                best = item
                break
        if best is None:
            return None, float("inf")
        age = max(0.0, frame_ts - best.timestamp)
        return best, age

    def _get_causal_info(self, frame_ts: float) -> tuple[RTABMapInfoTelemetry | None, float]:
        """Find the newest causal info record with timestamp <= frame_ts."""
        best: RTABMapInfoTelemetry | None = None
        for item in reversed(self.info_queue):
            if item.timestamp <= frame_ts + 1e-4:
                best = item
                break
        if best is None:
            return None, float("inf")
        age = max(0.0, frame_ts - best.timestamp)
        return best, age

    # Pose and Telemetry Estimation

    def estimate(self, sensor_frame: SensorFrame) -> tuple[PoseEstimate, EstimatorState]:
        rgb_ts = sensor_frame.timestamp.value

        # Step 1: Base TF lookup for world_T_camera
        base_pose, base_state = super().estimate(sensor_frame)

        # Step 2: Causal telemetry lookup
        odom_info, odom_info_age = self._get_causal_odom_info(rgb_ts)
        odom, odom_age = self._get_causal_odom(rgb_ts)
        info, info_age = self._get_causal_info(rgb_ts)

        # Step 3: Determine Odometry Lost & Freshness
        odometry_lost = False
        features = 0
        matches = 0
        inliers = 0
        inlier_ratio = 0.0
        local_map_size = 0
        local_key_frames = 0
        key_frame_added = False
        reg_time_ms = 0.0
        covariance_6x6 = None
        cov_frame = ""
        cov_available = False

        if odom_info is not None and odom_info_age <= self.telemetry_max_age:
            odometry_lost = odom_info.lost
            features = odom_info.features
            matches = odom_info.matches
            inliers = odom_info.inliers
            inlier_ratio = (
                float(inliers) / float(features)
                if features > 0
                else (float(inliers) / float(matches) if matches > 0 else 0.0)
            )
            local_map_size = odom_info.local_map_size
            local_key_frames = odom_info.local_key_frames
            key_frame_added = odom_info.key_frame_added
            reg_time_ms = odom_info.time_estimation_sec * 1000.0

            if odom_info.covariance is not None:
                covariance_6x6 = odom_info.covariance
                cov_frame = self.sensor_frame
                cov_available = True

        if covariance_6x6 is None and odom is not None and odom_age <= self.telemetry_max_age:
            if odom.covariance is not None:
                covariance_6x6 = odom.covariance
                cov_frame = odom.frame_id
                cov_available = True

        # Step 4: Loop Closure & Landmark detection
        current_node_id = -1
        loop_closure_id = 0
        proximity_id = 0
        landmark_id = 0
        loop_closure_detected = False

        if info is not None and info_age <= self.telemetry_max_age:
            current_node_id = info.ref_id
            loop_closure_id = info.loop_closure_id
            proximity_id = info.proximity_detection_id
            landmark_id = info.landmark_id

            if loop_closure_id > 0 and loop_closure_id != self._last_processed_loop_id:
                loop_closure_detected = True
                self._last_processed_loop_id = loop_closure_id

        # Fallback to direct odometry pose if TF buffer has not linked world->camera yet
        if not base_pose.valid and odom is not None and odom.transform is not None and odom_age <= self.slam_pose_max_age and not odometry_lost:
            base_pose = PoseEstimate(
                world_T_camera=odom.transform,
                timestamp=rgb_ts,
                valid=True,
                age=odom_age,
                source=self.backend_name,
                covariance=covariance_6x6,
                frame_id=odom.frame_id,
                pose_timestamp=odom.timestamp,
                target_timestamp=rgb_ts,
                lookup_mode="odom_telemetry",
                transform_source="rtabmap_odom",
            )

        # Step 5: Multi-Signal Tracking State Machine
        # Health decision rule: TF fresh AND odometry not lost -> TRACKING
        pose_is_fresh = base_pose.valid and (base_pose.age <= self.slam_pose_max_age)

        if not pose_is_fresh or odometry_lost:
            # Estimator is lost or initializing
            if self._last_valid_pose is None:
                current_state = TrackingState.INITIALIZING
            else:
                current_state = TrackingState.LOST

            # Hard invariant: If odometry is lost, do not let stale TF masquerade as valid
            if odometry_lost and base_pose.valid:
                base_pose = PoseEstimate.invalid(
                    timestamp=rgb_ts,
                    source=self.backend_name,
                    target_timestamp=rgb_ts,
                    transform_source="rtabmap_odom_lost",
                    frame_id=self.world_frame,
                )
        else:
            if self._rtabmap_tracking_state == TrackingState.LOST:
                current_state = TrackingState.RECOVERED
            elif self._rtabmap_tracking_state in (TrackingState.RECOVERED, TrackingState.INITIALIZING):
                current_state = TrackingState.TRACKING
            else:
                current_state = TrackingState.TRACKING

        self._rtabmap_tracking_state = current_state
        self._previous_state = current_state
        if base_pose.valid:
            self._last_valid_pose = base_pose

        # Update pose estimate with covariance if available
        if base_pose.valid and covariance_6x6 is not None:
            base_pose = PoseEstimate(
                world_T_camera=base_pose.world_T_camera,
                timestamp=base_pose.timestamp,
                valid=True,
                age=base_pose.age,
                source=base_pose.source,
                covariance=covariance_6x6,
                frame_id=base_pose.frame_id,
                pose_timestamp=base_pose.pose_timestamp,
                target_timestamp=base_pose.target_timestamp,
                lookup_mode=base_pose.lookup_mode,
                transform_source=base_pose.transform_source,
                metadata={
                    "key_frame_added": key_frame_added,
                    "inlier_ratio": inlier_ratio,
                    "ref_node_id": current_node_id,
                },
            )

        telemetry_ts = odom_info.timestamp if odom_info is not None else (odom.timestamp if odom is not None else None)
        telemetry_age_val = min(odom_info_age, odom_age) if (odom_info is not None or odom is not None) else float("inf")

        estimator_state = EstimatorState(
            backend_name=self.backend_name,
            tracking_state=current_state,
            matches=matches,
            inliers=inliers,
            inlier_ratio=inlier_ratio,
            features=features,
            local_map_size=local_map_size,
            local_key_frames=local_key_frames,
            key_frame_added=key_frame_added,
            registration_time_ms=reg_time_ms,
            total_estimator_time_ms=base_state.total_estimator_time_ms,
            odometry_lost=odometry_lost,
            current_node_id=current_node_id,
            loop_closure_id=loop_closure_id,
            proximity_detection_id=proximity_id,
            landmark_id=landmark_id,
            tracking_recovered=(current_state == TrackingState.RECOVERED),
            loop_closure_detected=loop_closure_detected,
            telemetry_timestamp=telemetry_ts,
            telemetry_age=telemetry_age_val,
            covariance_available=cov_available,
            covariance_frame=cov_frame,
            metadata={
                "pose_age_ms": base_pose.age * 1000.0 if np.isfinite(base_pose.age) else -1.0,
                "distance_travelled": odom_info.distance_travelled if odom_info else 0.0,
            },
        )

        return base_pose, estimator_state
