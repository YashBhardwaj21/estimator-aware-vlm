"""Geometry module for camera intrinsics and 3D transforms."""

from scene_graph.geometry.camera import CameraIntrinsics, DepthModel
from scene_graph.geometry.point_cloud import (
    ObjectGeometry,
    compute_object_geometry,
)
from scene_graph.geometry.reference_frame import (
    RelationReferenceFrame,
    compute_alignment_transform,
    compute_alignment_from_normal,
    estimate_support_plane_normal,
)
from scene_graph.geometry.transforms import (
    pose_to_transform,
    quaternion_to_matrix,
    transform_points,
)

from scene_graph.geometry.ground_plane import (
    GroundPlaneConfig,
    GroundPlaneExtractor,
    GroundPlaneTracker,
    create_ground_aligned_reference_frame,
    compute_ground_leveling_transform,
)

__all__ = [
    "CameraIntrinsics",
    "DepthModel",
    "quaternion_to_matrix",
    "pose_to_transform",
    "transform_points",
    "ObjectGeometry",
    "compute_object_geometry",
    "RelationReferenceFrame",
    "estimate_support_plane_normal",
    "compute_alignment_transform",
    "compute_alignment_from_normal",
    "GroundPlaneConfig",
    "GroundPlaneExtractor",
    "GroundPlaneTracker",
    "create_ground_aligned_reference_frame",
    "compute_ground_leveling_transform",
]
