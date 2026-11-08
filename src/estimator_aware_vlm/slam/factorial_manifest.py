"""Canonical Factorial Experiment Manifest, Invariant Verification, and Evaluation Controller.

Enforces zero-hardcoding, zero-simulation, and fail-closed validation across the 4-arm
factorial SLAM integration matrix (I0, I1, I2, I3).
"""

from __future__ import annotations

import csv
from dataclasses import asdict, dataclass, field
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Set, Tuple, Union
import yaml

from estimator_aware_vlm.audit.anti_fabrication import (
    AntiFabricationValidator,
    IntegrityViolationError,
)
from estimator_aware_vlm.common.paths import get_repo_root
from estimator_aware_vlm.datasets import (
    TrajectoryRole,
    parse_tum_ground_truth,
    parse_tum_trajectory,
)
from estimator_aware_vlm.evaluation import (
    evaluate_paired_trajectories_common_support,
    evaluate_trajectory,
)


class FactorialValidationError(IntegrityViolationError):
    """Raised when an experimental manifest or trajectory mapping violates integrity invariants."""
    pass


class IncompleteFactorialError(FactorialValidationError):
    """Raised when complete factorial evaluation is requested but arms are missing."""
    pass


VALID_MODE_IDS = ("I0", "I1", "I2", "I3")
STATUS_AVAILABLE = "available"
STATUS_NOT_RUN = "not_run"
STATUS_FAILED = "failed"
STATUS_INCOMPLETE = "INCOMPLETE_FACTORIAL_EXPERIMENT"
STATUS_COMPLETE = "COMPLETE_FACTORIAL_EXPERIMENT"

MODE_TYPE_SPECS = {
    "I0": {"type": "baseline", "image_mask": False, "native_weighting": False},
    "I1": {"type": "image_mask_only", "image_mask": True, "native_weighting": False},
    "I2": {"type": "native_weighting_only", "image_mask": False, "native_weighting": True},
    "I3": {"type": "combined", "image_mask": True, "native_weighting": True},
}


@dataclass
class FactorialModeConfig:
    """Declared specification and provenance for a single factorial intervention arm."""
    mode_id: str
    type: str
    image_mask: bool
    native_landmark_weighting: bool
    status: str
    source: Optional[str] = None
    source_run_id: Optional[str] = None
    source_manifest: Optional[str] = None
    trajectory_path: Optional[str] = None
    trajectory_sha256: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class FactorialExperimentConfig:
    """Canonical 4-arm factorial experiment configuration."""
    experiment_id: str
    sequence: str
    description: str
    modes: Dict[str, FactorialModeConfig] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> FactorialExperimentConfig:
        raw_modes = data.get("modes", {})
        modes: Dict[str, FactorialModeConfig] = {}
        for m_id, m_cfg in raw_modes.items():
            modes[m_id] = FactorialModeConfig(
                mode_id=m_id,
                type=m_cfg.get("type", ""),
                image_mask=bool(m_cfg.get("image_mask", False)),
                native_landmark_weighting=bool(m_cfg.get("native_landmark_weighting", False)),
                status=str(m_cfg.get("status", STATUS_NOT_RUN)).lower(),
                source=m_cfg.get("source"),
                source_run_id=m_cfg.get("source_run_id"),
                source_manifest=m_cfg.get("source_manifest"),
                trajectory_path=m_cfg.get("trajectory_path"),
                trajectory_sha256=m_cfg.get("trajectory_sha256"),
            )
        return cls(
            experiment_id=data.get("experiment_id", ""),
            sequence=data.get("sequence", ""),
            description=data.get("description", ""),
            modes=modes,
        )

    @classmethod
    def load_yaml(cls, yaml_path: Union[str, Path]) -> FactorialExperimentConfig:
        p = Path(yaml_path).resolve()
        if not p.is_file():
            raise FactorialValidationError(f"Experiment configuration file not found: {p}")
        with open(p, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        if not isinstance(data, dict):
            raise FactorialValidationError(f"Invalid YAML root structure in {p}: expected mapping")
        return cls.from_dict(data)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "experiment_id": self.experiment_id,
            "sequence": self.sequence,
            "description": self.description,
            "modes": {m_id: m.to_dict() for m_id, m in self.modes.items()},
        }


class FactorialManifestValidator:
    """Strict validator for factorial experiment configuration, artifacts, and mappings."""

    def __init__(self, repo_root: Optional[Path] = None):
        self.repo_root = Path(repo_root).resolve() if repo_root else get_repo_root()
        self.anti_fab = AntiFabricationValidator(strict_leakage_check=True)

    def _resolve_path(self, raw_path: Union[str, Path]) -> Path:
        p = Path(raw_path)
        if p.is_absolute():
            return p.resolve()
        return (self.repo_root / p).resolve()

    def validate_manifest(self, config: FactorialExperimentConfig) -> Dict[str, Any]:
        """Validate all configuration invariants across arms.

        Raises:
            FactorialValidationError on any invariant violation.
        """
        # 1. Check all required modes are defined
        missing_modes = [m_id for m_id in VALID_MODE_IDS if m_id not in config.modes]
        if missing_modes:
            raise FactorialValidationError(
                f"Missing required factorial modes: {missing_modes}. All four {VALID_MODE_IDS} must be declared."
            )

        unexpected_modes = [m_id for m_id in config.modes if m_id not in VALID_MODE_IDS]
        if unexpected_modes:
            raise FactorialValidationError(
                f"Unexpected factorial modes declared: {unexpected_modes}. Only {VALID_MODE_IDS} permitted."
            )

        seen_trajectories: Dict[str, str] = {}  # resolved_path_str -> mode_id
        seen_hashes: Dict[str, str] = {}        # sha256_hex -> mode_id
        seen_source_arms: Dict[str, str] = {}   # source_arm_key -> mode_id
        validation_records: Dict[str, Dict[str, Any]] = {}

        # 2. Validate individual mode specifications
        for m_id in VALID_MODE_IDS:
            mode = config.modes[m_id]
            expected_spec = MODE_TYPE_SPECS[m_id]

            # Invariant: mode type matches
            if mode.type != expected_spec["type"]:
                raise FactorialValidationError(
                    f"Mode {m_id} declared type '{mode.type}' does not match canonical requirement '{expected_spec['type']}'"
                )

            # Invariant: image_mask flag matches
            if mode.image_mask != expected_spec["image_mask"]:
                raise FactorialValidationError(
                    f"Mode {m_id} image_mask={mode.image_mask} violates canonical requirement image_mask={expected_spec['image_mask']}"
                )

            # Invariant: native_landmark_weighting flag matches
            if mode.native_landmark_weighting != expected_spec["native_weighting"]:
                raise FactorialValidationError(
                    f"Mode {m_id} native_landmark_weighting={mode.native_landmark_weighting} violates canonical requirement {expected_spec['native_weighting']}"
                )

            # Invariant: valid status
            if mode.status not in (STATUS_AVAILABLE, STATUS_NOT_RUN, STATUS_FAILED):
                raise FactorialValidationError(
                    f"Mode {m_id} has invalid execution status '{mode.status}'. Must be one of ({STATUS_AVAILABLE}, {STATUS_NOT_RUN}, {STATUS_FAILED})."
                )

            # Invariant: NOT_RUN arms must not reference trajectories
            if mode.status == STATUS_NOT_RUN:
                if mode.trajectory_path:
                    raise FactorialValidationError(
                        f"Mode {m_id} is marked status='{STATUS_NOT_RUN}' but defines trajectory_path='{mode.trajectory_path}'. Must be null."
                    )
                if mode.trajectory_sha256:
                    raise FactorialValidationError(
                        f"Mode {m_id} is marked status='{STATUS_NOT_RUN}' but defines trajectory_sha256='{mode.trajectory_sha256}'. Must be null."
                    )
                validation_records[m_id] = {
                    "status": STATUS_NOT_RUN,
                    "validated": True,
                    "message": "Arm is safely declared NOT_RUN with null trajectory references.",
                }
                continue

            # Invariant: AVAILABLE arms must have valid existing trajectory and provenance
            if mode.status == STATUS_AVAILABLE:
                if not mode.trajectory_path:
                    raise FactorialValidationError(
                        f"Mode {m_id} is marked status='{STATUS_AVAILABLE}' but trajectory_path is missing or null."
                    )
                if not mode.trajectory_sha256:
                    raise FactorialValidationError(
                        f"Mode {m_id} is marked status='{STATUS_AVAILABLE}' but trajectory_sha256 is missing or null."
                    )

                traj_file = self._resolve_path(mode.trajectory_path)
                if not traj_file.is_file():
                    raise FactorialValidationError(
                        f"Mode {m_id} trajectory file does not exist on disk: {traj_file}"
                    )

                # Validate trajectory syntax and poses using anti-fabrication validator
                traj_info = self.anti_fab.validate_trajectory(str(traj_file), min_poses=10)
                actual_sha = traj_info["sha256"].lower()
                declared_sha = mode.trajectory_sha256.strip().lower()

                if actual_sha != declared_sha:
                    raise FactorialValidationError(
                        f"Mode {m_id} trajectory SHA-256 mismatch! Declared: {declared_sha}, Actual on disk: {actual_sha} (path: {traj_file})"
                    )

                # Check duplicate trajectory paths across modes
                traj_str = str(traj_file).lower()
                if traj_str in seen_trajectories:
                    raise FactorialValidationError(
                        f"Duplicate trajectory assignment! Mode {m_id} references {traj_file}, which is already assigned to {seen_trajectories[traj_str]}."
                    )
                seen_trajectories[traj_str] = m_id

                # Check duplicate trajectory hashes across modes
                if actual_sha in seen_hashes:
                    raise FactorialValidationError(
                        f"Duplicate trajectory hash detected! Mode {m_id} has identical trajectory digest to mode {seen_hashes[actual_sha]} (SHA: {actual_sha}). Distinct intervention modes cannot reuse trajectories."
                    )
                seen_hashes[actual_sha] = m_id

                # Check duplicate source arms across modes
                if mode.source:
                    source_key = f"{mode.source_run_id or 'default'}::{mode.source}"
                    if source_key in seen_source_arms:
                        raise FactorialValidationError(
                            f"Duplicate source arm assignment! Mode {m_id} references source '{mode.source}' already used by {seen_source_arms[source_key]}."
                        )
                    seen_source_arms[source_key] = m_id

                # Check source manifest consistency if provided
                if mode.source_manifest:
                    sm_path = self._resolve_path(mode.source_manifest)
                    if not sm_path.is_file():
                        raise FactorialValidationError(
                            f"Mode {m_id} referenced source_manifest not found on disk: {sm_path}"
                        )
                    with open(sm_path, "r", encoding="utf-8") as f:
                        sm_data = json.load(f)

                    arms = sm_data.get("arms", [])
                    matching_arm = next((a for a in arms if a.get("arm_name") == mode.source or a.get("arm_id") == mode.source or a.get("intervention_id", "").endswith(str(mode.source))), None)
                    if not matching_arm:
                        raise FactorialValidationError(
                            f"Mode {m_id} source arm '{mode.source}' not found in source manifest {sm_path}"
                        )

                    # Validate source arm configuration against mode requirements
                    arm_mask_mode = matching_arm.get("mask_mode", "none").lower()
                    if not mode.image_mask and arm_mask_mode != "none":
                        raise FactorialValidationError(
                            f"Mode {m_id} requires image_mask=False, but source arm '{mode.source}' in {sm_path.name} has mask_mode='{arm_mask_mode}'."
                        )
                    if mode.image_mask and arm_mask_mode == "none":
                        raise FactorialValidationError(
                            f"Mode {m_id} requires image_mask=True, but source arm '{mode.source}' in {sm_path.name} has mask_mode='none'."
                        )

                    # Check native weighting provenance
                    # The source manifest from run_20261101_expansion is an image-masking expansion; it contains NO native weighting.
                    has_native = bool(matching_arm.get("native_landmark_weighting", False))
                    if mode.native_landmark_weighting and not has_native:
                        raise FactorialValidationError(
                            f"Mode {m_id} requires native_landmark_weighting=True, but source arm '{mode.source}' in {sm_path.name} does not have native landmark weighting."
                        )

                validation_records[m_id] = {
                    "status": STATUS_AVAILABLE,
                    "validated": True,
                    "trajectory_path": str(traj_file),
                    "trajectory_sha256": actual_sha,
                    "valid_poses": traj_info["valid_poses"],
                    "duration_sec": traj_info["duration_sec"],
                }

        return {
            "validation_status": "PASSED",
            "experiment_id": config.experiment_id,
            "sequence": config.sequence,
            "modes": validation_records,
        }

    def evaluate(
        self,
        config: FactorialExperimentConfig,
        ground_truth_path: Union[str, Path],
        allow_partial: bool = False,
        max_dt_sec: float = 0.02,
    ) -> Dict[str, Any]:
        """Perform verified evaluation of available arms against ground truth.

        Args:
            config: The experiment configuration.
            ground_truth_path: Path to groundtruth.txt.
            allow_partial: If False, raises IncompleteFactorialError if any arm is not AVAILABLE.
            max_dt_sec: Matching tolerance in seconds.

        Returns:
            Dictionary with evaluation results and explicit status.
        """
        # Validate manifest first
        val_res = self.validate_manifest(config)

        gt_file = self._resolve_path(ground_truth_path)
        if not gt_file.is_file():
            raise FactorialValidationError(f"Ground truth file not found: {gt_file}")
        ground_truth = parse_tum_ground_truth(gt_file)

        available_modes = [m_id for m_id in VALID_MODE_IDS if config.modes[m_id].status == STATUS_AVAILABLE]
        pending_modes = [m_id for m_id in VALID_MODE_IDS if config.modes[m_id].status != STATUS_AVAILABLE]

        if pending_modes and not allow_partial:
            raise IncompleteFactorialError(
                f"Complete factorial evaluation failed closed! Missing available arms: {pending_modes}. "
                "The evaluator refuses to claim a completed 4-arm factorial matrix while intervention arms are missing."
            )

        eval_status = STATUS_COMPLETE if not pending_modes else STATUS_INCOMPLETE

        evaluated_modes: List[Dict[str, Any]] = []
        i0_traj = None
        i0_single_eval = None

        # Parse and evaluate available trajectories
        for m_id in available_modes:
            mode = config.modes[m_id]
            traj_path = self._resolve_path(mode.trajectory_path)
            est_traj = parse_tum_trajectory(traj_path, role=TrajectoryRole.ESTIMATED, name=f"{m_id}_{mode.type}")
            single_eval = evaluate_trajectory(est_traj, ground_truth, max_dt_sec=max_dt_sec)

            if m_id == "I0":
                i0_traj = est_traj
                i0_single_eval = single_eval

            row: Dict[str, Any] = {
                "mode_id": m_id,
                "type": mode.type,
                "status": mode.status,
                "image_mask": mode.image_mask,
                "native_landmark_weighting": mode.native_landmark_weighting,
                "source": mode.source,
                "trajectory_path": str(traj_path.relative_to(self.repo_root)),
                "trajectory_sha256": mode.trajectory_sha256,
                "estimated_poses": len(est_traj),
                "ate_rmse_m": round(single_eval.ate.rmse_m, 4),
                "ate_mean_m": round(single_eval.ate.mean_m, 4),
                "ate_median_m": round(single_eval.ate.median_m, 4),
                "ate_std_m": round(single_eval.ate.std_m, 4),
                "tracking_coverage_pct": round(single_eval.tracking_coverage * 100.0, 2),
                "delta_ate_vs_i0_m": None,
                "pct_improvement_vs_i0": None,
                "common_support_poses": None,
                "common_ate_rmse_m": None,
                "common_delta_ate_m": None,
                "common_pct_improvement": None,
                "note": None,
            }

            if i0_traj is not None and i0_single_eval is not None:
                paired = evaluate_paired_trajectories_common_support(i0_traj, est_traj, ground_truth, max_dt_sec=max_dt_sec)
                delta_ate = round(i0_single_eval.ate.rmse_m - single_eval.ate.rmse_m, 4)
                pct_imp = round(delta_ate / i0_single_eval.ate.rmse_m * 100.0, 2)
                row["delta_ate_vs_i0_m"] = delta_ate
                row["pct_improvement_vs_i0"] = pct_imp
                row["common_support_poses"] = paired.num_common_poses
                row["common_ate_rmse_m"] = round(paired.ate_b_common.rmse_m, 4)
                row["common_delta_ate_m"] = round(paired.absolute_ate_reduction_common_m, 4)
                row["common_pct_improvement"] = round(paired.percentage_ate_improvement_common, 2)

            evaluated_modes.append(row)

        # Include pending modes in output table to ensure transparent disclosure
        for m_id in pending_modes:
            mode = config.modes[m_id]
            evaluated_modes.append({
                "mode_id": m_id,
                "type": mode.type,
                "status": mode.status.upper(),
                "image_mask": mode.image_mask,
                "native_landmark_weighting": mode.native_landmark_weighting,
                "source": None,
                "trajectory_path": None,
                "trajectory_sha256": None,
                "estimated_poses": None,
                "ate_rmse_m": None,
                "ate_mean_m": None,
                "ate_median_m": None,
                "ate_std_m": None,
                "tracking_coverage_pct": None,
                "delta_ate_vs_i0_m": None,
                "pct_improvement_vs_i0": None,
                "common_support_poses": None,
                "common_ate_rmse_m": None,
                "common_delta_ate_m": None,
                "common_pct_improvement": None,
                "note": "PENDING_NATIVE_EXECUTION (Awaiting genuine ORB-SLAM3 integration and telemetry in Gates 2-3)",
            })

        return {
            "status": eval_status,
            "experiment_id": config.experiment_id,
            "sequence": config.sequence,
            "description": config.description,
            "available_modes": available_modes,
            "pending_modes": pending_modes,
            "is_complete_factorial": (len(pending_modes) == 0),
            "manifest_validation": val_res,
            "evaluated_modes": evaluated_modes,
        }
