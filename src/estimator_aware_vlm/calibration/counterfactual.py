"""Counterfactual Utility Calibration Pipeline for Estimator-Aware VLM.

Calibrates the routing decision utility function:
    U(q) = delta_L_slam(i) - lambda_cost * C_vlm
against genuine empirical single-track SLAM counterfactual outcomes:
    delta_L_slam(i) = ATE(Baseline) - ATE(Mask only track i)

Replaces heuristic proxy weights with a regularized, cross-validated estimator
evaluated via Leave-One-Out Cross-Validation (LOOCV) without held-out test data leakage.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np


@dataclass
class CounterfactualRecord:
    """Empirical outcome of a single-track masking counterfactual SLAM experiment."""
    track_id: str
    semantic_class: str
    observation_count: int
    lifespan_sec: float
    measured_displacement_m: float
    displacement_uncertainty_m: float
    apparent_camera_frame_displacement_m: float
    camera_ego_motion_displacement_m: float
    empirical_baseline_ate_rmse_m: float
    empirical_single_masked_ate_rmse_m: float
    empirical_delta_L_slam: float  # ATE_baseline - ATE_single_masked
    action_taken: str = "MASK_FROM_TRACKING"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


FEATURE_NAMES = [
    "measured_displacement_m",
    "displacement_uncertainty_m",
    "observation_count_norm",
    "camera_ego_motion_displacement_m",
    "lifespan_sec_norm",
    "displacement_to_ego_ratio",
]


def extract_track_features(track_dict: Dict[str, Any]) -> np.ndarray:
    """Extract standard normalized feature vector phi(i) from a tracked entity."""
    w_disp = float(track_dict.get("measured_displacement_m", track_dict.get("world_displacement_m", 0.0)) or 0.0)
    disp_unc = float(track_dict.get("displacement_uncertainty_m", 0.0) or 0.0)
    obs_cnt = float(track_dict.get("observation_count", track_dict.get("total_observations", 0)) or 0)
    cam_disp = float(track_dict.get("camera_ego_motion_displacement_m", 0.0) or 0.0)
    lifespan = float(track_dict.get("lifespan_sec", 0.0) or 0.0)
    if lifespan <= 0.0 and "first_timestamp" in track_dict and "last_timestamp" in track_dict:
        try:
            lifespan = max(0.0, float(track_dict["last_timestamp"]) - float(track_dict["first_timestamp"]))
        except (ValueError, TypeError):
            lifespan = 0.0

    obs_cnt_norm = min(1.0, obs_cnt / 100.0)
    lifespan_norm = min(1.0, lifespan / 10.0)
    disp_to_ego_ratio = min(5.0, w_disp / (cam_disp + 1e-4))

    return np.array([
        w_disp,
        disp_unc,
        obs_cnt_norm,
        cam_disp,
        lifespan_norm,
        disp_to_ego_ratio,
    ], dtype=np.float64)


@dataclass
class CalibratedUtilityModel:
    """Calibrated predictor of downstream SLAM error reduction delta_L_slam(i)."""
    feature_names: List[str]
    weights: List[float]
    bias: float
    feature_mean: List[float]
    feature_std: List[float]
    loocv_mae: float
    loocv_r2: float
    lambda_cost: float = 0.45
    c_vlm: float = 0.50

    def predict_delta_L(self, features: np.ndarray) -> float:
        """Predict expected SLAM trajectory loss reduction delta_L_slam in meters."""
        feat = np.array(features, dtype=np.float64)
        mean = np.array(self.feature_mean, dtype=np.float64)
        std = np.array(self.feature_std, dtype=np.float64)
        std = np.where(std < 1e-6, 1.0, std)

        x_norm = (feat - mean) / std
        w = np.array(self.weights, dtype=np.float64)
        pred = float(np.dot(w, x_norm) + self.bias)
        # Empirical loss reduction is bounded below by small negative (masking static landmark slightly hurts)
        return max(-0.05, pred)

    def compute_utility(
        self,
        features: np.ndarray,
        lambda_cost: Optional[float] = None,
        c_vlm: Optional[float] = None,
    ) -> float:
        """Compute expected query utility U(i) = delta_L_slam(i) - lambda * C_vlm."""
        lam = self.lambda_cost if lambda_cost is None else lambda_cost
        cost = self.c_vlm if c_vlm is None else c_vlm
        pred_delta_l = self.predict_delta_L(features)
        return float(pred_delta_l - (lam * cost))

    def should_query_vlm(
        self,
        features: np.ndarray,
        utility_threshold: float = 0.0,
        lambda_cost: Optional[float] = None,
    ) -> bool:
        """Return True if expected net utility exceeds decision threshold."""
        u = self.compute_utility(features, lambda_cost=lambda_cost)
        return bool(u >= utility_threshold)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "feature_names": self.feature_names,
            "weights": [round(float(w), 6) for w in self.weights],
            "bias": round(float(self.bias), 6),
            "feature_mean": [round(float(m), 6) for m in self.feature_mean],
            "feature_std": [round(float(s), 6) for s in self.feature_std],
            "loocv_mae": round(float(self.loocv_mae), 6),
            "loocv_r2": round(float(self.loocv_r2), 4),
            "lambda_cost": self.lambda_cost,
            "c_vlm": self.c_vlm,
        }

    def save_json(self, path: Union[str, Path]) -> None:
        p = Path(path).resolve()
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load_json(cls, path: Union[str, Path]) -> CalibratedUtilityModel:
        p = Path(path).resolve()
        with open(p, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls(
            feature_names=data["feature_names"],
            weights=data["weights"],
            bias=data["bias"],
            feature_mean=data["feature_mean"],
            feature_std=data["feature_std"],
            loocv_mae=data["loocv_mae"],
            loocv_r2=data["loocv_r2"],
            lambda_cost=data.get("lambda_cost", 0.45),
            c_vlm=data.get("c_vlm", 0.50),
        )


def fit_calibrated_utility_model(
    records: List[CounterfactualRecord],
    lambda_cost: float = 0.45,
    c_vlm: float = 0.50,
    ridge_alpha: float = 0.50,
) -> CalibratedUtilityModel:
    """Fit a ridge-regularized counterfactual utility model with Leave-One-Out Cross-Validation.

    Guarantees:
    - Normalizes feature matrix via z-score.
    - Fits Ridge regression: w = (X^T X + alpha * I)^{-1} X^T y.
    - Evaluates out-of-fold generalization via LOOCV to strictly prevent held-out leakage.
    """
    if len(records) < 3:
        raise ValueError(f"At least 3 counterfactual records required to fit utility model, got {len(records)}")

    # Extract feature matrix X and target vector y
    X_raw = np.array([extract_track_features(r.to_dict()) for r in records], dtype=np.float64)
    y = np.array([r.empirical_delta_L_slam for r in records], dtype=np.float64)

    N, D = X_raw.shape
    mean = np.mean(X_raw, axis=0)
    std = np.std(X_raw, axis=0)
    std = np.where(std < 1e-6, 1.0, std)

    X_norm = (X_raw - mean) / std

    # Perform Leave-One-Out Cross-Validation (LOOCV)
    oof_predictions = np.zeros(N, dtype=np.float64)
    for i in range(N):
        train_idx = [j for j in range(N) if j != i]
        X_train = X_norm[train_idx]
        y_train = y[train_idx]

        # Centering targets
        y_train_mean = float(np.mean(y_train))
        y_train_centered = y_train - y_train_mean

        # Ridge solution
        A = X_train.T @ X_train + ridge_alpha * np.eye(D)
        b_vec = X_train.T @ y_train_centered
        w_loo = np.linalg.solve(A, b_vec)
        b_loo = y_train_mean

        oof_predictions[i] = float(np.dot(w_loo, X_norm[i]) + b_loo)

    mae = float(np.mean(np.abs(oof_predictions - y)))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    ss_res = float(np.sum((y - oof_predictions) ** 2))
    r2 = 1.0 - (ss_res / ss_tot) if ss_tot > 1e-9 else 1.0

    # Fit final model on all data
    y_mean = float(np.mean(y))
    y_centered = y - y_mean
    A = X_norm.T @ X_norm + ridge_alpha * np.eye(D)
    b_vec = X_norm.T @ y_centered
    w_final = np.linalg.solve(A, b_vec)
    b_final = y_mean

    return CalibratedUtilityModel(
        feature_names=FEATURE_NAMES,
        weights=[float(w) for w in w_final],
        bias=float(b_final),
        feature_mean=[float(m) for m in mean],
        feature_std=[float(s) for s in std],
        loocv_mae=mae,
        loocv_r2=r2,
        lambda_cost=lambda_cost,
        c_vlm=c_vlm,
    )

