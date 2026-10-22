import numpy as np

def voxel_downsample(points: np.ndarray, voxel_size: float) -> np.ndarray:
    """Deterministically downsample a point cloud using voxel hashing.
    
    Args:
        points: (N, 3) numpy array of 3D points.
        voxel_size: Size of the voxel grid cells.
        
    Returns:
        (M, 3) numpy array of downsampled points.
    """
    if len(points) == 0:
        return points
        
    voxel = np.floor(points / voxel_size).astype(np.int64)

    _, indices = np.unique(
        voxel,
        axis=0,
        return_index=True,
    )

    return points[np.sort(indices)]
