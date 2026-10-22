"""SE(3) transform utilities."""

import numpy as np


def quaternion_to_matrix(qx: float, qy: float, qz: float, qw: float) -> np.ndarray:
    """Convert a quaternion (TUM convention) to a 3x3 rotation matrix.
    
    Args:
        qx, qy, qz, qw: Quaternion components.
        
    Returns:
        A 3x3 numpy array representing the rotation matrix.
    """
    q = np.array([qx, qy, qz, qw], dtype=np.float64)
    if not np.all(np.isfinite(q)):
        raise ValueError("Quaternion contains non-finite values (NaN or Inf)")
        
    norm = np.linalg.norm(q)
    if norm < 1e-12:
        raise ValueError("Quaternion norm must be non-zero")
    q = q / norm
    qx, qy, qz, qw = q
    
    R = np.array([
        [
            1.0 - 2.0 * (qy * qy + qz * qz),
            2.0 * (qx * qy - qz * qw),
            2.0 * (qx * qz + qy * qw),
        ],
        [
            2.0 * (qx * qy + qz * qw),
            1.0 - 2.0 * (qx * qx + qz * qz),
            2.0 * (qy * qz - qx * qw),
        ],
        [
            2.0 * (qx * qz - qy * qw),
            2.0 * (qy * qz + qx * qw),
            1.0 - 2.0 * (qx * qx + qy * qy),
        ],
    ], dtype=np.float64)
    
    return R


def matrix_to_quaternion(R: np.ndarray) -> tuple[float, float, float, float]:
    """Convert a 3x3 orthonormal rotation matrix into a unit quaternion (x, y, z, w)."""
    tr = np.trace(R)
    if tr > 0:
        S = np.sqrt(tr + 1.0) * 2.0
        qw = 0.25 * S
        qx = (R[2, 1] - R[1, 2]) / S
        qy = (R[0, 2] - R[2, 0]) / S
        qz = (R[1, 0] - R[0, 1]) / S
    elif (R[0, 0] > R[1, 1]) and (R[0, 0] > R[2, 2]):
        S = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2.0
        qw = (R[2, 1] - R[1, 2]) / S
        qx = 0.25 * S
        qy = (R[0, 1] + R[1, 0]) / S
        qz = (R[0, 2] + R[2, 0]) / S
    elif R[1, 1] > R[2, 2]:
        S = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2.0
        qw = (R[0, 2] - R[2, 0]) / S
        qx = (R[0, 1] + R[1, 0]) / S
        qy = 0.25 * S
        qz = (R[1, 2] + R[2, 1]) / S
    else:
        S = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2.0
        qw = (R[1, 0] - R[0, 1]) / S
        qx = (R[0, 2] + R[2, 0]) / S
        qy = (R[1, 2] + R[2, 1]) / S
        qz = 0.25 * S

    return float(qx), float(qy), float(qz), float(qw)


def pose_to_transform(tx: float, ty: float, tz: float, qx: float, qy: float, qz: float, qw: float) -> np.ndarray:
    """Convert translation and quaternion to 4x4 SE(3) transformation matrix (world_T_camera).
    
    Args:
        tx, ty, tz: Translation components.
        qx, qy, qz, qw: Quaternion components.
        
    Returns:
        A 4x4 numpy array representing the SE(3) transform.
    """
    R = quaternion_to_matrix(qx, qy, qz, qw)
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R
    T[:3, 3] = [tx, ty, tz]
    return T


def validate_se3_transform(T: np.ndarray) -> None:
    """Strictly validate that a matrix is a valid SE(3) transform."""
    if T.shape != (4, 4):
        raise ValueError(f"Transform matrix must be 4x4, got {T.shape}")
        
    if not np.all(np.isfinite(T)):
        raise ValueError("Transform matrix contains non-finite values (NaN or Inf)")
        
    # Check bottom row is [0, 0, 0, 1]
    if not np.allclose(T[3, :], [0.0, 0.0, 0.0, 1.0], atol=1e-5):
        raise ValueError(f"Transform bottom row must be [0, 0, 0, 1], got {T[3, :]}")
        
    R = T[:3, :3]
    
    # Orthonormal check: R.T @ R == I
    RtR = R.T @ R
    if not np.allclose(RtR, np.eye(3), atol=1e-4):
        raise ValueError("Rotation matrix is not orthonormal")
        
    # det(R) == 1
    det = np.linalg.det(R)
    if not np.isclose(det, 1.0, atol=1e-4):
        raise ValueError(f"Rotation matrix determinant must be 1, got {det}")


def transform_points(T: np.ndarray, points_camera: np.ndarray) -> np.ndarray:
    """Transform an Nx3 array of 3D points from camera to world coordinates.
    
    Args:
        T: 4x4 SE(3) transformation matrix (world_T_camera).
        points_camera: Nx3 numpy array of points in camera coordinates.
        
    Returns:
        Nx3 numpy array of points in world coordinates.
    """
    validate_se3_transform(T)
    if points_camera.ndim != 2 or points_camera.shape[1] != 3:
        raise ValueError(f"Points must be Nx3, got {points_camera.shape}")
    if not np.all(np.isfinite(points_camera)):
        raise ValueError("Points contain non-finite values (NaN or Inf)")
        
    if len(points_camera) == 0:
        return np.empty((0, 3), dtype=np.float64)
        
    # N x 4 homogeneous coordinates
    points_h = np.hstack([points_camera, np.ones((len(points_camera), 1), dtype=np.float64)])
    
    # Transform: (T @ points_h.T).T => points_h @ T.T
    points_world_h = points_h @ T.T
    
    # Return Nx3
    return points_world_h[:, :3]


def optical_to_body_rotation() -> np.ndarray:
    """Return standard 3x3 rotation from camera optical (X-right, Y-down, Z-forward)
    to robot body frame (X-forward, Y-left, Z-up) according to ROS REP-103.
    """
    return np.array([
        [0.0,  0.0,  1.0],
        [-1.0, 0.0,  0.0],
        [0.0, -1.0,  0.0],
    ], dtype=np.float64)


def body_to_optical_rotation() -> np.ndarray:
    """Return standard 3x3 rotation from robot body frame to camera optical frame."""
    return optical_to_body_rotation().T


def optical_to_body_transform(t_body: np.ndarray | None = None) -> np.ndarray:
    """Return 4x4 SE(3) transformation matrix (body_T_optical)."""
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = optical_to_body_rotation()
    if t_body is not None:
        T[:3, 3] = np.asarray(t_body, dtype=np.float64)
    return T


def body_to_optical_transform(t_optical: np.ndarray | None = None) -> np.ndarray:
    """Return 4x4 SE(3) transformation matrix (optical_T_body)."""
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = body_to_optical_rotation()
    if t_optical is not None:
        T[:3, 3] = np.asarray(t_optical, dtype=np.float64)
    return T


def invert_se3_transform(T: np.ndarray) -> np.ndarray:
    """Invert a 4x4 SE(3) transformation matrix using closed-form rigid inversion.
    
    Given T = [R, t; 0, 1], T^-1 = [R^T, -R^T * t; 0, 1].
    """
    validate_se3_transform(T)
    R = T[:3, :3]
    t = T[:3, 3]
    T_inv = np.eye(4, dtype=np.float64)
    T_inv[:3, :3] = R.T
    T_inv[:3, 3] = -R.T @ t
    return T_inv


def transform_points_inverse(T: np.ndarray, points_world: np.ndarray) -> np.ndarray:
    """Transform an Nx3 array of 3D points from world back to camera coordinates using T^-1."""
    T_inv = invert_se3_transform(T)
    return transform_points(T_inv, points_world)
