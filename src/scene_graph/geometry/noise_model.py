from abc import ABC, abstractmethod
from typing import Callable, Optional
import numpy as np

from scene_graph.geometry.camera import CameraIntrinsics


class DepthNoiseModel(ABC):
    """Abstract base class for camera depth measurement noise models."""

    @abstractmethod
    def estimate_covariance(
        self,
        points_camera: np.ndarray,
        intrinsics: CameraIntrinsics,
        R_world_camera: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Estimate 3x3 sensor measurement covariance matrix.
        
        Args:
            points_camera: (N, 3) points in camera coordinate frame.
            intrinsics: CameraIntrinsics object.
            R_world_camera: Optional 3x3 rotation matrix from camera to world frame.
                If provided, covariance is returned in world frame: R * Sigma_cam * R.T.
                If None, covariance is returned in camera frame.
                
        Returns:
            (3, 3) positive semi-definite covariance matrix.
        """
        pass


class IsotropicNoiseModel(DepthNoiseModel):
    """Uniform isotropic noise model: Sigma = std_m^2 * I_3."""

    def __init__(self, std_m: float = 0.01):
        if std_m <= 0.0:
            raise ValueError(f"std_m must be positive, got {std_m}")
        self.std_m = float(std_m)

    def estimate_covariance(
        self,
        points_camera: np.ndarray,
        intrinsics: CameraIntrinsics,
        R_world_camera: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        var = self.std_m ** 2
        return np.eye(3, dtype=np.float64) * var


class StereoDepthNoiseModel(DepthNoiseModel):
    """Stereo depth measurement noise model.
    
    Axial range uncertainty scales quadratically with depth:
        sigma_z = (z^2 / (f * B)) * sigma_d
    where:
        B = baseline (m)
        f = focal length (pixels)
        sigma_d = subpixel disparity standard deviation (pixels)
        
    Lateral uncertainties scale linearly with depth:
        sigma_x = (z / fx) * sigma_p
        sigma_y = (z / fy) * sigma_p
    where:
        sigma_p = pixel localization noise standard deviation (pixels)
    """

    def __init__(
        self,
        baseline_m: float = 0.095,
        subpixel_disparity_std: float = 0.1,
        pixel_noise_std: float = 0.5,
        min_depth_m: float = 0.1,
        axial_noise_floor_m: float = 0.0,
    ):
        if baseline_m <= 0.0:
            raise ValueError("baseline_m must be positive.")
        if subpixel_disparity_std <= 0.0:
            raise ValueError("subpixel_disparity_std must be positive.")
        if pixel_noise_std <= 0.0:
            raise ValueError("pixel_noise_std must be positive.")
        if axial_noise_floor_m < 0.0:
            raise ValueError("axial_noise_floor_m must be non-negative.")

        self.baseline_m = float(baseline_m)
        self.subpixel_disparity_std = float(subpixel_disparity_std)
        self.pixel_noise_std = float(pixel_noise_std)
        self.min_depth_m = float(min_depth_m)
        self.axial_noise_floor_m = float(axial_noise_floor_m)

    def estimate_covariance(
        self,
        points_camera: np.ndarray,
        intrinsics: CameraIntrinsics,
        R_world_camera: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        if points_camera is None or len(points_camera) == 0:
            z = 1.0
        else:
            z = float(np.median(points_camera[:, 2]))
        
        z = max(z, self.min_depth_m)
        f = float((intrinsics.fx + intrinsics.fy) / 2.0)
        
        sigma_z = (z ** 2 / (f * self.baseline_m)) * self.subpixel_disparity_std + self.axial_noise_floor_m
        sigma_x = (z / intrinsics.fx) * self.pixel_noise_std
        sigma_y = (z / intrinsics.fy) * self.pixel_noise_std
        
        cov_camera = np.diag([sigma_x ** 2, sigma_y ** 2, sigma_z ** 2]).astype(np.float64)
        
        if R_world_camera is not None:
            R = np.asarray(R_world_camera, dtype=np.float64)
            cov_world = R @ cov_camera @ R.T
            return 0.5 * (cov_world + cov_world.T)
            
        return cov_camera


class QuadraticDepthNoiseModel(DepthNoiseModel):
    """Parametric quadratic depth measurement noise model:
    
        sigma_z(z) = alpha * z^2 + sigma_0
        sigma_x(z) = (z / fx) * pixel_noise_std
        sigma_y(z) = (z / fy) * pixel_noise_std
        
    Reflecting empirical range error in RGB-D structured-light / active stereo sensors
    (e.g., Nguyen et al. 2012, Khoshelham & Elberink 2012).
    """

    def __init__(
        self,
        alpha: float = 0.0014,
        sigma_0: float = 0.003,
        pixel_noise_std: float = 0.5,
        min_depth_m: float = 0.1,
    ):
        if alpha < 0.0:
            raise ValueError("alpha must be non-negative.")
        if sigma_0 <= 0.0:
            raise ValueError("sigma_0 must be positive.")
        if pixel_noise_std <= 0.0:
            raise ValueError("pixel_noise_std must be positive.")

        self.alpha = float(alpha)
        self.sigma_0 = float(sigma_0)
        self.pixel_noise_std = float(pixel_noise_std)
        self.min_depth_m = float(min_depth_m)

    def estimate_covariance(
        self,
        points_camera: np.ndarray,
        intrinsics: CameraIntrinsics,
        R_world_camera: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        if points_camera is None or len(points_camera) == 0:
            z = 1.0
        else:
            z = float(np.median(points_camera[:, 2]))

        z = max(z, self.min_depth_m)

        sigma_z = (self.alpha * (z ** 2)) + self.sigma_0
        sigma_x = (z / intrinsics.fx) * self.pixel_noise_std
        sigma_y = (z / intrinsics.fy) * self.pixel_noise_std

        cov_camera = np.diag([sigma_x ** 2, sigma_y ** 2, sigma_z ** 2]).astype(np.float64)

        if R_world_camera is not None:
            R = np.asarray(R_world_camera, dtype=np.float64)
            cov_world = R @ cov_camera @ R.T
            return 0.5 * (cov_world + cov_world.T)

        return cov_camera


class CustomNoiseModel(DepthNoiseModel):
    """User-supplied noise function."""

    def __init__(self, fn: Callable[[np.ndarray, CameraIntrinsics, Optional[np.ndarray]], np.ndarray]):
        if not callable(fn):
            raise TypeError("fn must be callable.")
        self._fn = fn

    def estimate_covariance(
        self,
        points_camera: np.ndarray,
        intrinsics: CameraIntrinsics,
        R_world_camera: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        cov = self._fn(points_camera, intrinsics, R_world_camera)
        return np.asarray(cov, dtype=np.float64)


def create_noise_model_from_config(config) -> DepthNoiseModel:
    """Factory creating the appropriate DepthNoiseModel from SceneGraphConfig."""
    sensor = getattr(config, "sensor", None)
    if sensor is None or getattr(sensor, "measurement_model", "isotropic") == "isotropic":
        std = getattr(sensor, "isotropic_std_m", None)
        if std is None and getattr(config, "geometry", None):
            std = getattr(config.geometry, "measurement_noise_std_m", 0.01)
        if std is None:
            std = 0.01
        return IsotropicNoiseModel(std_m=std)
    elif sensor.measurement_model == "stereo":
        return StereoDepthNoiseModel(
            baseline_m=sensor.baseline_m,
            subpixel_disparity_std=sensor.subpixel_disparity_std,
            pixel_noise_std=getattr(sensor, "pixel_noise_std", 0.5),
            axial_noise_floor_m=getattr(sensor, "axial_noise_floor_m", 0.0),
        )
    elif sensor.measurement_model == "quadratic":
        return QuadraticDepthNoiseModel(
            alpha=getattr(sensor, "quadratic_alpha", 0.0014),
            sigma_0=getattr(sensor, "quadratic_sigma_0", 0.003),
            pixel_noise_std=getattr(sensor, "pixel_noise_std", 0.5),
        )
    else:
        return IsotropicNoiseModel(std_m=0.01)
