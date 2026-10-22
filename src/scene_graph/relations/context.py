from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np

from scene_graph.geometry.camera import CameraIntrinsics
from scene_graph.geometry.reference_frame import (
    RelationReferenceFrame,
    CameraFrame,
)


from scene_graph.geometry.provenance import GeometrySource


@dataclass
class ObservationGeometry:
    """Cached per-observation. Computed ONCE, consumed by all relation modules."""

    obs_id: str
    track_id: str
    centroid_world: Optional[np.ndarray] = None
    bbox_min_world: Optional[np.ndarray] = None
    bbox_max_world: Optional[np.ndarray] = None
    depth_stats: Optional[Dict[str, float]] = None
    points_world_sampled: Optional[np.ndarray] = None
    points_world: Optional[np.ndarray] = None
    points_camera: Optional[np.ndarray] = None
    centroid_camera: Optional[np.ndarray] = None
    mask: Optional[np.ndarray] = None
    valid_point_count: int = 0
    geometry_source: GeometrySource = GeometrySource.OBSERVED
    position_covariance_world: Optional[np.ndarray] = None
    obb_center_world: Optional[np.ndarray] = None
    obb_axes_world: Optional[np.ndarray] = None
    obb_extents_world: Optional[np.ndarray] = None
    is_in_frustum: bool = True
    coordinate_frame: str = "world"
    localization_status: str = "localized"



@dataclass
class FrameContext:
    frame_index: int
    timestamp: float
    intrinsics: CameraIntrinsics
    world_T_camera: np.ndarray
    reference_frame: RelationReferenceFrame
    camera_frame: CameraFrame
    depth_image: Optional[np.ndarray]
    observation_geometry: Dict[str, ObservationGeometry]