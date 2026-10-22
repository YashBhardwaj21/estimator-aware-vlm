from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class PlaneGeometry:
    normal: np.ndarray
    distance: float
    inlier_mask: np.ndarray
    residual_mean: float
    residual_std: float


def fit_plane_ransac(
    points: np.ndarray,
    distance_threshold: float,
    max_iterations: int,
    min_inliers: int,
    rng: np.random.Generator,
) -> Optional[PlaneGeometry]:

    points = np.asarray(points, dtype=np.float64)

    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("points must have shape (N, 3).")

    if not np.isfinite(points).all():
        raise ValueError("points must contain only finite values.")

    if distance_threshold <= 0.0:
        raise ValueError("distance_threshold must be positive.")

    if max_iterations < 1:
        raise ValueError("max_iterations must be positive.")

    if min_inliers < 3:
        raise ValueError("min_inliers must be at least 3.")

    if not isinstance(rng, np.random.Generator):
        raise TypeError("rng must be a numpy.random.Generator.")

    point_count = len(points)

    if point_count < min_inliers:
        return None

    best_inlier_mask = None
    best_inlier_count = 0
    best_residual_mean = np.inf

    for _ in range(max_iterations):
        sample_indices = rng.choice(point_count, size=3, replace=False)
        p1, p2, p3 = points[sample_indices]

        v1 = p2 - p1
        v2 = p3 - p1
        normal = np.cross(v1, v2)
        normal_norm = float(np.linalg.norm(normal))

        if normal_norm <= np.finfo(float).eps:
            continue

        normal /= normal_norm
        distance = -float(np.dot(normal, p1))

        residuals = np.abs(points @ normal + distance)
        inlier_mask = residuals <= distance_threshold
        inlier_count = int(np.count_nonzero(inlier_mask))

        if inlier_count < min_inliers:
            continue

        inlier_residuals = residuals[inlier_mask]
        residual_mean = float(np.mean(inlier_residuals))

        if (
            inlier_count > best_inlier_count
            or (
                inlier_count == best_inlier_count
                and residual_mean < best_residual_mean
            )
        ):
            best_inlier_count = inlier_count
            best_residual_mean = residual_mean
            best_inlier_mask = inlier_mask

    if best_inlier_mask is None:
        return None

    inlier_points = points[best_inlier_mask]

    if len(inlier_points) < min_inliers:
        return None

    centroid = np.mean(inlier_points, axis=0)
    centered = inlier_points - centroid

    _, singular_values, vh = np.linalg.svd(
        centered,
        full_matrices=False,
    )

    if len(singular_values) < 3:
        return None

    normal = vh[-1]
    normal_norm = float(np.linalg.norm(normal))

    if normal_norm <= np.finfo(float).eps:
        return None

    normal /= normal_norm
    distance = -float(np.dot(normal, centroid))

    final_residuals = np.abs(points @ normal + distance)
    final_inlier_mask = final_residuals <= distance_threshold
    final_inlier_count = int(np.count_nonzero(final_inlier_mask))

    if final_inlier_count < min_inliers:
        return None

    inlier_residuals = final_residuals[final_inlier_mask]

    return PlaneGeometry(
        normal=normal,
        distance=distance,
        inlier_mask=final_inlier_mask,
        residual_mean=float(np.mean(inlier_residuals)),
        residual_std=float(np.std(inlier_residuals)),
    )