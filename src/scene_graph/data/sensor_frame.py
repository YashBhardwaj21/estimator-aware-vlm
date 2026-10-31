"""Canonical Sensor Ingestion Contract: SensorFrame.

Represents an immutable, acquisition-level sensor payload. Contains strictly sensor,
calibration, timing, and stream validity data. Contains ZERO downstream concepts
(no world poses, no relation reference frames, no tracking state).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import numpy as np

from scene_graph.data.timestamp import Timestamp, TimestampDomain
from scene_graph.geometry.camera import CameraIntrinsics


class StreamStatus(str, Enum):
    """Operational health status of the sensor streams comprising this frame."""
    OK = "ok"                       # All expected modalities are healthy and synchronized
    DEGRADED = "degraded"           # One non-critical stream is missing or jittered
    DEPTH_DROPPED = "depth_dropped" # Depth stream was lost, desynchronized, or corrupted
    IMU_DROPPED = "imu_dropped"     # Inertial stream is silent or timed out
    CALIBRATION_FALLBACK = "calibration_fallback" # Using fallback calibration instead of firmware


@dataclass(frozen=True)
class IMUSample:
    """Inertial Measurement Unit sample representing linear acceleration and angular velocity.
    
    Contains explicit individual timestamp and clock domain information.
    """
    timestamp: float                    # Timestamp in seconds
    accel: np.ndarray                   # (3,) m/s^2 linear acceleration [ax, ay, az]
    gyro: np.ndarray                    # (3,) rad/s angular velocity [wx, wy, wz]
    domain: TimestampDomain = TimestampDomain.HARDWARE_CLOCK
    sequence: int = -1                  # Hardware IMU sample counter if provided

    def __post_init__(self) -> None:
        if not isinstance(self.domain, TimestampDomain):
            object.__setattr__(self, "domain", TimestampDomain.from_string(str(self.domain)))
        if not isinstance(self.accel, np.ndarray) or self.accel.shape != (3,):
            object.__setattr__(self, "accel", np.asarray(self.accel, dtype=np.float64).reshape(3))
        if not isinstance(self.gyro, np.ndarray) or self.gyro.shape != (3,):
            object.__setattr__(self, "gyro", np.asarray(self.gyro, dtype=np.float64).reshape(3))


@dataclass(frozen=True)
class SensorFrame:
    """Canonical data structure representing an acquisition event.
    
    Invariants:
    - depth is strictly float32 metric depth in meters (or None).
    - Raw integer sensor depth is converted ONCE in sensor adapters before reaching SensorFrame.
    - timestamps preserve exact domain and source provenance.
    - Free of downstream concepts (no world_T_camera, no relation_frame, no track IDs).
    """
    session_id: str                                  # Unique sensor session / run identifier
    sequence_number: int                             # Monotonic frame sequence counter
    timestamp: Timestamp                             # Authoritative acquisition timestamp
    rgb: np.ndarray                                  # (H, W, 3) uint8 RGB format
    camera_intrinsics: CameraIntrinsics              # Pinhole intrinsic calibration
    depth: np.ndarray | None = None               # (H, W) float32 metric depth in meters
    depth_scale: float = 0.001                       # Runtime meters-per-raw-unit (D455=0.001, TUM=0.0002)
    distortion: tuple[float, ...] = (0.0, 0.0, 0.0, 0.0, 0.0)
    distortion_model: str = "plumb_bob"
    imu_samples: tuple[IMUSample, ...] = ()          # Batch of IMU readings collected in (t_{k-1}, t_k]
    
    # Provenance and Arrival Timing
    host_capture_timestamp: Timestamp | None = None # Sender capture completion timestamp
    network_arrival_timestamp: Timestamp | None = None # Receiver socket read timestamp
    mapped_ros_timestamp: float | None = None     # Safe synchronized ROS timestamp (seconds)
    
    # Metadata and Status
    frame_id: str = "camera_color_optical_frame"
    optical_frame_id: str = "camera_depth_optical_frame"
    status: StreamStatus = StreamStatus.OK
    is_aligned_to_color: bool = True
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        # 1. Validate sequence and session
        if self.sequence_number < 0:
            raise ValueError(f"SensorFrame.sequence_number must be >= 0, got {self.sequence_number}")
        if not self.session_id:
            raise ValueError("SensorFrame.session_id must be a non-empty string")

        # 2. Validate timestamp
        if not isinstance(self.timestamp, Timestamp):
            raise TypeError(f"SensorFrame.timestamp must be Timestamp instance, got {type(self.timestamp)}")

        # 3. Validate RGB array
        if self.rgb is None or not isinstance(self.rgb, np.ndarray):
            raise TypeError("SensorFrame.rgb must be a valid NumPy ndarray")
        if self.rgb.ndim != 3 or self.rgb.shape[2] != 3:
            raise ValueError(f"SensorFrame.rgb must have shape (H, W, 3), got {self.rgb.shape}")
        if self.rgb.dtype != np.uint8:
            raise TypeError(f"SensorFrame.rgb must have dtype uint8, got {self.rgb.dtype}")

        # 4. Validate Camera Intrinsics match image dimensions
        if self.camera_intrinsics is not None:
            if self.camera_intrinsics.width != self.rgb.shape[1] or self.camera_intrinsics.height != self.rgb.shape[0]:
                raise ValueError(
                    f"Image shape ({self.rgb.shape[0]}, {self.rgb.shape[1]}) does not match camera intrinsics "
                    f"dimensions ({self.camera_intrinsics.height}, {self.camera_intrinsics.width})"
                )

        # 5. Validate Depth array (Strict Invariant: Float32 Meters)
        if self.depth is not None:
            if not isinstance(self.depth, np.ndarray):
                raise TypeError("SensorFrame.depth must be a valid NumPy ndarray or None")
            if np.issubdtype(self.depth.dtype, np.integer):
                raise ValueError(
                    f"SensorFrame.depth must be metric float32 depth in meters, got integer dtype {self.depth.dtype}. "
                    "Raw depth scaling must occur inside sensor adapters."
                )
            if self.depth.ndim != 2:
                raise ValueError(f"SensorFrame.depth must be (H, W) 2D array, got shape {self.depth.shape}")
            if self.depth.shape != self.rgb.shape[:2]:
                raise ValueError(
                    f"Depth shape {self.depth.shape} does not match RGB shape {self.rgb.shape[:2]}"
                )
            if self.depth.dtype != np.float32:
                object.__setattr__(self, "depth", self.depth.astype(np.float32))

        # 6. Validate depth scale
        if self.depth_scale <= 0.0 or not np.isfinite(self.depth_scale):
            raise ValueError(f"SensorFrame.depth_scale must be positive and finite, got {self.depth_scale}")

        # 7. Ensure IMU samples is a tuple
        if not isinstance(self.imu_samples, tuple):
            object.__setattr__(self, "imu_samples", tuple(self.imu_samples))

    @property
    def has_depth(self) -> bool:
        return self.depth is not None

    @property
    def has_imu(self) -> bool:
        return len(self.imu_samples) > 0

    @property
    def width(self) -> int:
        return int(self.rgb.shape[1])

    @property
    def height(self) -> int:
        return int(self.rgb.shape[0])
