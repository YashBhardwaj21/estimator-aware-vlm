from dataclasses import dataclass
from enum import Enum
from typing import Optional
import numpy as np


class ReferenceFrameType(Enum):
    WORLD = "world"
    CAMERA = "camera"


@dataclass
class RelationReferenceFrame:
    """The canonical reference frame for relation evaluation.
    
    All geometric relation checks (left/right, above/below, front/back) 
    MUST be evaluated in this frame to ensure mathematical consistency.
    """
    origin_world: np.ndarray        # (3,) World coordinate of the origin (e.g. camera center)
    up_axis_world: np.ndarray       # (3,) Unit vector for 'up' (positive Y in camera frame, usually)
    horizontal_axis_world: np.ndarray # (3,) Unit vector for 'right' (positive X in camera frame)
    depth_axis_world: np.ndarray    # (3,) Unit vector for 'depth' (positive Z in camera frame)
    frame_type: ReferenceFrameType = ReferenceFrameType.WORLD
    
    def __post_init__(self):
        # Validate unit vectors
        for axis in [self.up_axis_world, self.horizontal_axis_world, self.depth_axis_world]:
            if not np.all(np.isfinite(axis)):
                raise ValueError("Reference frame axes must be finite")
            norm = np.linalg.norm(axis)
            if not np.isclose(norm, 1.0, atol=1e-4):
                raise ValueError(f"Reference frame axes must be unit vectors, got norm {norm}")
                
        # Validate orthogonality
        if not np.isclose(np.dot(self.up_axis_world, self.horizontal_axis_world), 0.0, atol=1e-4):
            raise ValueError("Up and horizontal axes must be orthogonal")
        if not np.isclose(np.dot(self.up_axis_world, self.depth_axis_world), 0.0, atol=1e-4):
            raise ValueError("Up and depth axes must be orthogonal")
        if not np.isclose(np.dot(self.horizontal_axis_world, self.depth_axis_world), 0.0, atol=1e-4):
            raise ValueError("Horizontal and depth axes must be orthogonal")

    @classmethod
    def create(
        cls,
        mode: str,
        world_T_camera: np.ndarray,
        gravity_vector: Optional[np.ndarray] = None,
        initial_heading: Optional[np.ndarray] = None,
    ) -> "RelationReferenceFrame":
        """Factory creating RelationReferenceFrame for CAMERA, MAP, or GRAVITY modes."""
        mode_lower = str(mode).lower()
        if mode_lower == "camera":
            R = world_T_camera[:3, :3]
            T = world_T_camera[:3, 3]
            cam_x = R[:, 0]
            cam_y = R[:, 1]
            cam_z = R[:, 2]
            return cls(
                origin_world=T,
                up_axis_world=-cam_y / np.linalg.norm(cam_y),
                horizontal_axis_world=cam_x / np.linalg.norm(cam_x),
                depth_axis_world=cam_z / np.linalg.norm(cam_z),
                frame_type=ReferenceFrameType.CAMERA,
            )
        elif mode_lower == "gravity":
            up = gravity_vector if gravity_vector is not None else np.array([0.0, 0.0, 1.0])
            heading = initial_heading if initial_heading is not None else np.array([1.0, 0.0, 0.0])
            frame = cls.from_gravity_and_heading(
                origin_world=world_T_camera[:3, 3],
                up_axis_world=up,
                heading_world=heading,
            )
            object.__setattr__(frame, "frame_type", ReferenceFrameType.WORLD)
            return frame
        else:  # "map" or "world"
            return cls(
                origin_world=np.zeros(3, dtype=np.float64),
                up_axis_world=np.array([0.0, 0.0, 1.0], dtype=np.float64),
                horizontal_axis_world=np.array([1.0, 0.0, 0.0], dtype=np.float64),
                depth_axis_world=np.array([0.0, 1.0, 0.0], dtype=np.float64),
                frame_type=ReferenceFrameType.WORLD,
            )

    @classmethod
    def from_gravity_and_heading(
        cls,
        origin_world: np.ndarray,
        up_axis_world: np.ndarray,
        heading_world: np.ndarray,
    ) -> "RelationReferenceFrame":
        """Construct a fixed global reference frame from gravity and heading.
        
        This normalizes the up vector, projects the heading onto the horizontal
        plane, and derives orthogonal axes to freeze the relation frame for the sequence.
        """
        up = up_axis_world / np.linalg.norm(up_axis_world)
        
        # Project heading onto horizontal plane (orthogonal to up)
        # v_proj = v - (v . up) * up
        heading = heading_world - np.dot(heading_world, up) * up
        if np.linalg.norm(heading) < 1e-6:
            # Fallback if heading is parallel to up
            # Choose an arbitrary orthogonal vector
            arbitrary = np.array([1.0, 0.0, 0.0]) if abs(up[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
            heading = arbitrary - np.dot(arbitrary, up) * up
            
        depth = heading / np.linalg.norm(heading)
        
        # Right is up x depth
        right = np.cross(up, depth)
        right = right / np.linalg.norm(right)
        
        return cls(
            origin_world=origin_world,
            up_axis_world=up,
            horizontal_axis_world=right,
            depth_axis_world=depth
        )


@dataclass
class CameraFrame:
    """A dynamic camera-relative reference frame for occlusion checking."""
    origin_world: np.ndarray        # (3,)
    up_axis_world: np.ndarray       # (3,)
    horizontal_axis_world: np.ndarray # (3,)
    depth_axis_world: np.ndarray    # (3,)
    
    @classmethod
    def from_camera_pose(cls, world_T_camera: np.ndarray) -> "CameraFrame":
        R = world_T_camera[:3, :3]
        T = world_T_camera[:3, 3]
        
        # OpenCV convention: X right, Y down, Z forward
        cam_x = R[:, 0]
        cam_y = R[:, 1]
        cam_z = R[:, 2]
        
        return cls(
            origin_world=T,
            up_axis_world=-cam_y,
            horizontal_axis_world=cam_x,
            depth_axis_world=cam_z
        )


def estimate_support_plane_normal(points: np.ndarray) -> np.ndarray:
    """Estimate the dominant support plane normal from a point cloud.
    
    Uses Principal Component Analysis (SVD) to find the normal vector.
    Assumes `points` is a dense cloud of the primary desk or floor.
    """
    if len(points) < 3:
        return np.array([0.0, 0.0, 1.0])
        
    centroid = np.mean(points, axis=0)
    centered = points - centroid
    
    # SVD on the centered points
    _, _, vh = np.linalg.svd(centered, full_matrices=False)
    
    # The normal is the eigenvector corresponding to the smallest eigenvalue
    normal = vh[2, :]
    
    # Ensure the normal generally points "up" (positive Z)
    if normal[2] < 0:
        normal = -normal
        
    return normal


def compute_alignment_from_normal(normal: np.ndarray) -> np.ndarray:
    """Compute a 4x4 transform that aligns the given normal to [0, 0, 1]."""
    target_up = np.array([0.0, 0.0, 1.0])
    
    normal = normal / np.linalg.norm(normal)
    v = np.cross(normal, target_up)
    c = np.dot(normal, target_up)
    
    if c > 0.999:
        R = np.eye(3)
    elif c < -0.999:
        R = np.diag([1.0, -1.0, -1.0])
    else:
        s = np.linalg.norm(v)
        vx = np.array([
            [0, -v[2], v[1]],
            [v[2], 0, -v[0]],
            [-v[1], v[0], 0]
        ])
        R = np.eye(3) + vx + (vx @ vx) * ((1 - c) / (s ** 2))
        
    T = np.eye(4)
    T[:3, :3] = R
    return T


def compute_alignment_transform(points_world: np.ndarray) -> np.ndarray:
    """Compute a 4x4 transform that aligns the normal of points_world to [0, 0, 1].
    
    This computes an alignment to the dominant support plane (e.g., desk, floor),
    which is useful for defining a stable world orientation for relations.
    """
    normal = estimate_support_plane_normal(points_world)
    return compute_alignment_from_normal(normal)
