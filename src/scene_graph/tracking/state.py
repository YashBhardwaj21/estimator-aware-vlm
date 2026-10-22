import numpy as np


class KalmanState:
    """6D constant-velocity Kalman filter state."""

    STATE_DIMENSION = 6
    POSITION_DIMENSION = 3

    def __init__(self, position: np.ndarray, initial_cov_pos: float | np.ndarray, initial_cov_vel: float):
        position = np.asarray(position, dtype=np.float64)

        if position.shape != (self.POSITION_DIMENSION,):
            raise ValueError("position must have shape (3,).")

        if not np.isfinite(position).all():
            raise ValueError("position must contain only finite values.")

        if not np.isfinite(initial_cov_vel) or initial_cov_vel <= 0.0:
            raise ValueError("initial_cov_vel must be finite and positive.")

        self.x = np.zeros(self.STATE_DIMENSION, dtype=np.float64)
        self.x[:self.POSITION_DIMENSION] = position

        self.P = np.zeros((self.STATE_DIMENSION, self.STATE_DIMENSION), dtype=np.float64)
        if isinstance(initial_cov_pos, (int, float)):
            if not np.isfinite(initial_cov_pos) or initial_cov_pos <= 0.0:
                raise ValueError("initial_cov_pos must be finite and positive.")
            self.P[:3, :3] = np.eye(3, dtype=np.float64) * float(initial_cov_pos)
        else:
            cov_mat = np.asarray(initial_cov_pos, dtype=np.float64)
            if cov_mat.shape != (3, 3):
                raise ValueError("initial_cov_pos as a matrix must have shape (3, 3).")
            self.P[:3, :3] = self._validate_covariance(cov_mat, "initial position covariance")

        self.P[3:, 3:] = np.eye(3, dtype=np.float64) * float(initial_cov_vel)
        self.P = self._symmetrize(self.P)

    def predict(self, dt: float, q_std: float) -> None:
        if not np.isfinite(dt):
            raise ValueError("dt must be finite.")

        if dt < 0.0:
            raise ValueError("dt must be non-negative.")

        if not np.isfinite(q_std) or q_std <= 0.0:
            raise ValueError("q_std must be finite and positive.")

        if dt == 0.0:
            return

        F = np.eye(self.STATE_DIMENSION, dtype=np.float64)
        F[0, 3] = dt
        F[1, 4] = dt
        F[2, 5] = dt

        q_variance = q_std ** 2
        dt2 = dt ** 2
        dt3 = dt ** 3 / 2.0
        dt4 = dt ** 4 / 4.0

        Q = np.zeros((self.STATE_DIMENSION, self.STATE_DIMENSION), dtype=np.float64)
        identity = np.eye(3, dtype=np.float64)

        Q[:3, :3] = identity * dt4 * q_variance
        Q[:3, 3:] = identity * dt3 * q_variance
        Q[3:, :3] = identity * dt3 * q_variance
        Q[3:, 3:] = identity * dt2 * q_variance

        self.x = F @ self.x
        self.P = F @ self.P @ F.T + Q
        self.P = self._validate_covariance(self.P, "state covariance")

        if not np.isfinite(self.x).all():
            raise FloatingPointError("Kalman prediction produced a non-finite state.")

    def update(self, measurement: np.ndarray, R: np.ndarray) -> None:
        measurement = np.asarray(measurement, dtype=np.float64)
        R = np.asarray(R, dtype=np.float64)

        if measurement.shape != (self.POSITION_DIMENSION,):
            raise ValueError("measurement must have shape (3,).")

        if not np.isfinite(measurement).all():
            raise ValueError("measurement must contain only finite values.")

        if R.shape != (3, 3):
            raise ValueError("R must have shape (3, 3).")

        if not np.isfinite(R).all():
            raise ValueError("R must contain only finite values.")

        R = self._validate_covariance(R, "measurement covariance")

        H = np.zeros((3, self.STATE_DIMENSION), dtype=np.float64)
        H[:, :3] = np.eye(3, dtype=np.float64)

        innovation = measurement - H @ self.x
        S = H @ self.P @ H.T + R
        S = self._validate_covariance(S, "innovation covariance")

        PHt = self.P @ H.T

        try:
            K = np.linalg.solve(S, PHt.T).T
        except np.linalg.LinAlgError as exc:
            raise FloatingPointError(
                "Unable to solve Kalman innovation covariance."
            ) from exc

        self.x = self.x + K @ innovation

        I = np.eye(self.STATE_DIMENSION, dtype=np.float64)
        I_KH = I - K @ H

        self.P = I_KH @ self.P @ I_KH.T + K @ R @ K.T
        self.P = self._validate_covariance(self.P, "updated state covariance")

        if not np.isfinite(self.x).all():
            raise FloatingPointError("Kalman update produced a non-finite state.")

    @staticmethod
    def _symmetrize(matrix: np.ndarray) -> np.ndarray:
        return 0.5 * (matrix + matrix.T)

    @classmethod
    def _validate_covariance(cls, covariance: np.ndarray, name: str) -> np.ndarray:
        covariance = cls._symmetrize(covariance)

        if not np.isfinite(covariance).all():
            raise ValueError(f"{name} must contain only finite values.")

        eigenvalues = np.linalg.eigvalsh(covariance)

        if not np.isfinite(eigenvalues).all():
            raise ValueError(f"{name} must have finite eigenvalues.")

        if np.min(eigenvalues) < -1e-10:
            raise ValueError(f"{name} must be positive semidefinite.")

        return covariance

    @property
    def position(self) -> np.ndarray:
        return self.x[:3].copy()

    @property
    def velocity(self) -> np.ndarray:
        return self.x[3:].copy()

    @property
    def position_covariance(self) -> np.ndarray:
        return self.P[:3, :3].copy()