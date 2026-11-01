"""Sequence-Level Grouped Validation and Data Sufficiency Gate for EDRR.

Enforces:
1. Group-isolated validation: GroupKFold and Leave-One-Group-Out (LOGO) ensuring all records
   from the same sequence belong strictly to either train or test fold, eliminating intra-sequence leakage.
2. Fold-isolated feature normalization: Feature mean and standard deviation are computed strictly
   on the training fold, never seeing held-out validation sequences.
3. Data Sufficiency Gate:
   Requires at least N >= 30 verified single-candidate records across at least 5 distinct sequences.
   If criteria are unmet, model fitting is refused and status is logged as:
       NOT_TRAINED_INSUFFICIENT_DATA (N=<count>, sequences=<seq_count>)
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np

from estimator_aware_vlm.edrr.dataset import InterventionDataset, SingleCandidateInterventionRecord


MIN_SAMPLE_COUNT_THRESHOLD = 30
MIN_SEQUENCE_COUNT_THRESHOLD = 5


@dataclass
class GroupedValidationResult:
    """Out-of-fold cross-validation performance metrics under grouped splits."""
    status: str
    n_samples: int
    n_sequences: int
    validation_strategy: str
    oof_mae: Optional[float] = None
    oof_rmse: Optional[float] = None
    oof_r2: Optional[float] = None
    fold_metrics: List[Dict[str, Any]] = field(default_factory=list)
    rejection_reason: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "status": self.status,
            "n_samples": self.n_samples,
            "n_sequences": self.n_sequences,
            "validation_strategy": self.validation_strategy,
            "oof_mae": round(self.oof_mae, 6) if self.oof_mae is not None else None,
            "oof_rmse": round(self.oof_rmse, 6) if self.oof_rmse is not None else None,
            "oof_r2": round(self.oof_r2, 4) if self.oof_r2 is not None else None,
            "fold_metrics": self.fold_metrics,
            "rejection_reason": self.rejection_reason,
        }


def check_data_sufficiency(dataset: InterventionDataset) -> Tuple[bool, str]:
    """Verify whether dataset meets the minimum N >= 30 and sequences >= 5 criteria.

    Returns:
        (is_sufficient, status_string)
    """
    valid_records = dataset.filter_valid_trajectory_records()
    n_samples = len(valid_records)
    sequences = sorted(list({r.sequence_id for r in valid_records}))
    n_seq = len(sequences)

    if n_samples < MIN_SAMPLE_COUNT_THRESHOLD or n_seq < MIN_SEQUENCE_COUNT_THRESHOLD:
        status = f"NOT_TRAINED_INSUFFICIENT_DATA (N={n_samples} < {MIN_SAMPLE_COUNT_THRESHOLD}, sequences={n_seq} < {MIN_SEQUENCE_COUNT_THRESHOLD})"
        return False, status

    return True, "DATA_SUFFICIENT"


def extract_record_features(record: SingleCandidateInterventionRecord) -> np.ndarray:
    """Extract standard 6D feature vector from an audited intervention record."""
    w_disp = float(record.measured_displacement_m)
    disp_unc = float(record.displacement_uncertainty_m)
    obs_cnt_norm = min(1.0, float(record.observation_count) / 100.0)
    cam_disp = float(record.camera_ego_motion_displacement_m)
    lifespan_norm = min(1.0, float(record.lifespan_sec) / 10.0)
    disp_to_ego_ratio = min(5.0, w_disp / (cam_disp + 1e-4))

    return np.array([
        w_disp,
        disp_unc,
        obs_cnt_norm,
        cam_disp,
        lifespan_norm,
        disp_to_ego_ratio,
    ], dtype=np.float64)


def run_sequence_level_grouped_validation(
    dataset: InterventionDataset,
    ridge_alpha: float = 0.50,
    enforce_sufficiency_gate: bool = True,
) -> GroupedValidationResult:
    """Execute Leave-One-Group-Out (LOGO) cross-validation grouped by sequence_id.

    Guarantees:
    - Normalization parameters (mean, std) are computed strictly on training sequences.
    - If enforce_sufficiency_gate is True and criteria unmet, refuses to fit and returns gate status.
    """
    valid_records = dataset.filter_valid_trajectory_records()
    n_samples = len(valid_records)
    sequences = sorted(list({r.sequence_id for r in valid_records}))
    n_seq = len(sequences)

    is_sufficient, status_msg = check_data_sufficiency(dataset)
    if not is_sufficient:
        if enforce_sufficiency_gate:
            return GroupedValidationResult(
                status=status_msg,
                n_samples=n_samples,
                n_sequences=n_seq,
                validation_strategy="LEAVE_ONE_GROUP_OUT",
                rejection_reason=(
                    f"Sample count N={n_samples} is below required threshold ({MIN_SAMPLE_COUNT_THRESHOLD}) "
                    f"or sequence count {n_seq} is below ({MIN_SEQUENCE_COUNT_THRESHOLD}). "
                    "Claiming calibrated VOI model on insufficient sample support is strictly prohibited."
                ),
            )

    if n_seq < 2:
        return GroupedValidationResult(
            status="FAILED_SINGLE_SEQUENCE",
            n_samples=n_samples,
            n_sequences=n_seq,
            validation_strategy="LEAVE_ONE_GROUP_OUT",
            rejection_reason="At least 2 distinct sequences required for grouped cross-validation.",
        )

    # Feature matrix X and target vector y
    X_raw = np.array([extract_record_features(r) for r in valid_records], dtype=np.float64)
    y = np.array([r.empirical_delta_ate_m for r in valid_records], dtype=np.float64)
    seq_labels = [r.sequence_id for r in valid_records]

    N, D = X_raw.shape
    oof_predictions = np.zeros(N, dtype=np.float64)
    fold_metrics = []

    for held_out_seq in sequences:
        val_indices = [idx for idx, s in enumerate(seq_labels) if s == held_out_seq]
        train_indices = [idx for idx, s in enumerate(seq_labels) if s != held_out_seq]

        X_train_raw = X_raw[train_indices]
        y_train = y[train_indices]
        X_val_raw = X_raw[val_indices]
        y_val = y[val_indices]

        # Fold-isolated normalization parameters
        train_mean = np.mean(X_train_raw, axis=0)
        train_std = np.std(X_train_raw, axis=0)
        train_std = np.where(train_std < 1e-6, 1.0, train_std)

        X_train = (X_train_raw - train_mean) / train_std
        X_val = (X_val_raw - train_mean) / train_std

        # Ridge regression on training sequences
        y_train_mean = float(np.mean(y_train))
        y_train_centered = y_train - y_train_mean

        A = X_train.T @ X_train + ridge_alpha * np.eye(D)
        b_vec = X_train.T @ y_train_centered
        w_fold = np.linalg.solve(A, b_vec)
        b_fold = y_train_mean

        pred_val = (X_val @ w_fold) + b_fold
        for v_idx, pred in zip(val_indices, pred_val):
            oof_predictions[v_idx] = float(pred)

        fold_mae = float(np.mean(np.abs(pred_val - y_val)))
        fold_metrics.append({
            "held_out_sequence": held_out_seq,
            "test_samples": len(val_indices),
            "mae": round(fold_mae, 6),
        })

    oof_mae = float(np.mean(np.abs(oof_predictions - y)))
    oof_rmse = float(np.sqrt(np.mean((oof_predictions - y) ** 2)))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    ss_res = float(np.sum((y - oof_predictions) ** 2))
    oof_r2 = float(1.0 - (ss_res / ss_tot)) if ss_tot > 1e-9 else 1.0

    return GroupedValidationResult(
        status="VALIDATED_GROUPED_CV",
        n_samples=n_samples,
        n_sequences=n_seq,
        validation_strategy="LEAVE_ONE_GROUP_OUT",
        oof_mae=oof_mae,
        oof_rmse=oof_rmse,
        oof_r2=oof_r2,
        fold_metrics=fold_metrics,
    )

