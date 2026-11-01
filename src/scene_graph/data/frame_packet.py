from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING

import numpy as np

from scene_graph.geometry.camera import CameraIntrinsics, DepthModel
from scene_graph.geometry.reference_frame import RelationReferenceFrame
from scene_graph.data.pose_estimate import EstimatorState, PoseEstimate

if TYPE_CHECKING:
    from scene_graph.data.sensor_frame import SensorFrame



class LocalizationMode(str, Enum):
    WORLD_MODE = "world_mode"
    CAMERA_LOCAL_MODE = "camera_local_mode"


@dataclass(frozen=True)
class IMUSample:
    """Inertial Measurement Unit sample representing linear acceleration and angular velocity."""
    timestamp: float
    accel: np.ndarray  # (3,) m/s^2
    gyro: np.ndarray   # (3,) rad/s


@dataclass
class FramePacket:
    """A synchronized, per-frame container for downstream scene graph tasks.
    
    Invariant:
    - depth is ALWAYS metric float32 depth in meters (or None).
    - Raw integer sensor depth is converted by the data source adapter before reaching FramePacket.
    - Core geometry never performs raw depth scale conversion.
    """
    frame_index: int            
    timestamp: float            
    rgb: np.ndarray             # HxWx3 uint8 RGB format
    depth: np.ndarray | None # HxW metric float32 depth in meters
    world_T_camera: np.ndarray | None  # (4, 4) pose matrix
    camera_intrinsics: CameraIntrinsics
    depth_model: DepthModel | None = None  # Deprecated: adapters own scale
    relation_frame: RelationReferenceFrame | None = None
    imu_samples: tuple[IMUSample, ...] = ()  # IMU samples in (t_{k-1}, t_k]
    frame_id: str = "camera_color_optical_frame"
    optical_frame_id: str | None = "camera_depth_optical_frame"
    pose_source: str = "unknown"
    metadata: dict = field(default_factory=dict)

    # Pose & Estimator Telemetry Contracts
    pose_estimate: PoseEstimate | None = None
    estimator_state: EstimatorState | None = None

    # Explicit Localization & Timestamp Provenance
    sensor_timestamp: float | None = None
    rgb_timestamp: float | None = None
    depth_timestamp: float | None = None
    pose_timestamp: float | None = None
    pose_age: float = 0.0
    world_frame: str = "world"
    transform_source: str = "unknown"
    transform_valid: bool = True
    localization_mode: LocalizationMode = LocalizationMode.WORLD_MODE

    def __post_init__(self):
        if not np.isfinite(self.timestamp) or self.timestamp < 0.0:
            raise ValueError(f"FramePacket timestamp must be finite and non-negative, got {self.timestamp}")

        if isinstance(self.localization_mode, str) and not isinstance(self.localization_mode, LocalizationMode):
            self.localization_mode = LocalizationMode(self.localization_mode)

        if self.rgb_timestamp is None:
            self.rgb_timestamp = self.timestamp
        if self.sensor_timestamp is None:
            self.sensor_timestamp = self.rgb_timestamp
        if self.depth_timestamp is None and self.depth is not None:
            self.depth_timestamp = self.rgb_timestamp

        if self.pose_estimate is not None:
            if self.world_T_camera is None:
                self.world_T_camera = self.pose_estimate.world_T_camera
            if self.pose_timestamp is None:
                if self.pose_estimate.valid or self.pose_estimate.pose_timestamp is not None:
                    self.pose_timestamp = self.pose_estimate.pose_timestamp
                else:
                    self.pose_timestamp = None
            self.pose_age = self.pose_estimate.age
            self.transform_valid = self.pose_estimate.valid
            eff_source = self.pose_estimate.transform_source if self.pose_estimate.transform_source != "unknown" else self.pose_estimate.source
            if eff_source != "unknown":
                self.transform_source = eff_source
                self.pose_source = eff_source
            self.world_frame = self.pose_estimate.frame_id
        elif self.localization_mode == LocalizationMode.CAMERA_LOCAL_MODE:
            self.world_frame = self.frame_id
            if self.world_T_camera is None:
                self.world_T_camera = np.eye(4, dtype=np.float64)
            if self.pose_timestamp is None:
                self.pose_timestamp = self.rgb_timestamp
            self.transform_source = "camera_local"
            self.pose_source = "camera_local"
            self.transform_valid = True
            self.pose_age = 0.0
            self.pose_estimate = PoseEstimate.identity(
                timestamp=self.rgb_timestamp,
                frame_id=self.world_frame,
                source=self.transform_source,
            )
        else:
            if self.pose_timestamp is not None:
                self.pose_age = abs(self.rgb_timestamp - self.pose_timestamp)
            elif self.world_T_camera is not None:
                self.pose_timestamp = self.rgb_timestamp
                self.pose_age = 0.0
            else:
                self.transform_valid = False
                self.transform_source = "missing_tf" if self.transform_source == "unknown" else self.transform_source

            eff_source = self.transform_source if self.transform_source != "unknown" else self.pose_source
            if self.transform_valid and self.world_T_camera is not None:
                self.pose_estimate = PoseEstimate(
                    world_T_camera=self.world_T_camera,
                    timestamp=self.rgb_timestamp,
                    valid=True,
                    age=self.pose_age,
                    source=eff_source,
                    frame_id=self.world_frame,
                    pose_timestamp=self.pose_timestamp,
                    target_timestamp=self.rgb_timestamp,
                    transform_source=eff_source,
                )
            else:
                self.pose_estimate = PoseEstimate.invalid(
                    timestamp=self.rgb_timestamp,
                    source=eff_source,
                    target_timestamp=self.rgb_timestamp,
                    transform_source=self.transform_source,
                    frame_id=self.world_frame,
                )

        if self.rgb is not None and isinstance(self.rgb, np.ndarray):
            if self.rgb.ndim != 3 or self.rgb.shape[2] != 3:
                raise ValueError(f"FramePacket.rgb must be (H, W, 3), got shape {self.rgb.shape}")
            if self.rgb.dtype != np.uint8:
                raise TypeError(f"FramePacket.rgb must have uint8 dtype, got {self.rgb.dtype}")

        if self.depth is not None and isinstance(self.depth, np.ndarray):
            if np.issubdtype(self.depth.dtype, np.integer):
                raise ValueError(
                    f"FramePacket.depth must be metric depth in meters (float32), got integer dtype {self.depth.dtype}. "
                    "Raw depth conversion belongs in the sensor adapter, not inside the core pipeline."
                )
            if self.depth.ndim != 2:
                raise ValueError(f"FramePacket.depth must be (H, W) 2D array, got shape {self.depth.shape}")
            if self.depth.dtype != np.float32:
                self.depth = self.depth.astype(np.float32)

        if self.world_T_camera is not None and isinstance(self.world_T_camera, np.ndarray) and self.world_T_camera.shape != (4, 4):
            raise ValueError("FramePacket.world_T_camera must be (4, 4) pose matrix")

        if self.metadata is None:
            self.metadata = {}

    @property
    def has_depth(self) -> bool:
        return self.depth is not None

    @property
    def has_pose(self) -> bool:
        return self.world_T_camera is not None

    @property
    def has_imu(self) -> bool:
        return len(self.imu_samples) > 0

    @property
    def is_persistent_world_frame(self) -> bool:
        return (
            self.localization_mode == LocalizationMode.WORLD_MODE
            and self.transform_valid
            and self.has_pose
        )

    @classmethod
    def from_sensor_frame(
        cls,
        sensor_frame: "SensorFrame",
        world_T_camera: np.ndarray | None = None,
        pose_timestamp: float | None = None,
        pose_source: str = "unknown",
        transform_source: str = "unknown",
        transform_valid: bool = True,
        localization_mode: LocalizationMode = LocalizationMode.WORLD_MODE,
        relation_frame: RelationReferenceFrame | None = None,
        world_frame: str = "world",
        pose_estimate: PoseEstimate | None = None,
        estimator_state: EstimatorState | None = None,
    ) -> "FramePacket":
        """Compatibility bridge converting a canonical SensorFrame into FramePacket.

        Preserves all sensor timestamps, intrinsics, and metric depth while attaching
        downstream localization, pose estimates, and reference frame context.
        """
        mapped_imu = tuple(
            IMUSample(timestamp=s.timestamp, accel=s.accel, gyro=s.gyro)
            for s in sensor_frame.imu_samples
        )

        ts_sec = (
            sensor_frame.mapped_ros_timestamp
            if sensor_frame.mapped_ros_timestamp is not None
            else sensor_frame.timestamp.value
        )

        metadata = dict(sensor_frame.metadata)
        metadata["session_id"] = sensor_frame.session_id
        metadata["stream_status"] = sensor_frame.status.value
        metadata["timestamp_domain"] = sensor_frame.timestamp.domain.value
        if sensor_frame.host_capture_timestamp is not None:
            metadata["host_capture_time"] = sensor_frame.host_capture_timestamp.value
        if pose_estimate is not None:
            eff_src = pose_estimate.transform_source if pose_estimate.transform_source != "unknown" else pose_estimate.source
            if eff_src != "unknown":
                pose_source = eff_src
                transform_source = eff_src
        elif transform_source != "unknown" and pose_source == "unknown":
            pose_source = transform_source
        elif pose_source != "unknown" and transform_source == "unknown":
            transform_source = pose_source

        return cls(
            frame_index=sensor_frame.sequence_number,
            timestamp=ts_sec,
            rgb=sensor_frame.rgb,
            depth=sensor_frame.depth,
            world_T_camera=world_T_camera,
            camera_intrinsics=sensor_frame.camera_intrinsics,
            relation_frame=relation_frame,
            imu_samples=mapped_imu,
            frame_id=sensor_frame.frame_id,
            optical_frame_id=sensor_frame.optical_frame_id,
            pose_source=pose_source,
            metadata=metadata,
            pose_estimate=pose_estimate,
            estimator_state=estimator_state,
            sensor_timestamp=sensor_frame.timestamp.value,
            rgb_timestamp=sensor_frame.timestamp.value,
            depth_timestamp=sensor_frame.timestamp.value if sensor_frame.has_depth else None,
            pose_timestamp=pose_timestamp,
            world_frame=world_frame,
            transform_source=transform_source,
            transform_valid=transform_valid,
            localization_mode=localization_mode,
        )


