"""
Automated Anti-Fabrication and Experimental Integrity Guardrails.
Enforces zero-hardcoding, zero-simulation, and fail-closed validation
across SLAM execution, VLM inference, and metric calculations.
"""

import os
import math
import hashlib
from typing import Dict, Any, List, Optional, Set

class IntegrityViolationError(Exception):
    """Raised when an empirical integrity or provenance violation is detected."""
    pass

class AntiFabricationValidator:
    """Fail-closed validator for experimental artifacts and metrics."""

    def __init__(self, strict_leakage_check: bool = True):
        self.strict_leakage_check = strict_leakage_check
        self.seen_trajectory_hashes: Dict[str, str] = {} # hash -> candidate_id
        self.seen_candidate_ids: Set[str] = set()

    @staticmethod
    def compute_sha256(file_path: str) -> str:
        if not os.path.isfile(file_path):
            raise IntegrityViolationError(f"Artifact not found on disk: {file_path}")
        h = hashlib.sha256()
        with open(file_path, "rb") as f:
            while chunk := f.read(65536):
                h.update(chunk)
        return h.hexdigest()

    def validate_trajectory(self, traj_path: str, min_poses: int = 10) -> Dict[str, Any]:
        """Validates that a trajectory file exists, is non-empty, and has valid numeric poses."""
        if not os.path.isfile(traj_path):
            raise IntegrityViolationError(f"Missing trajectory artifact: {traj_path}")
        
        file_hash = self.compute_sha256(traj_path)
        valid_poses = 0
        timestamps = []
        
        with open(traj_path, "r", encoding="utf-8") as f:
            for line_idx, line in enumerate(f, 1):
                parts = line.strip().split()
                if not parts:
                    continue
                if len(parts) < 8:
                    raise IntegrityViolationError(f"Malformed trajectory line {line_idx} in {traj_path}: expected >= 8 elements")
                
                try:
                    ts = float(parts[0])
                    tx, ty, tz = float(parts[1]), float(parts[2]), float(parts[3])
                    qx, qy, qz, qw = float(parts[4]), float(parts[5]), float(parts[6]), float(parts[7])
                except ValueError as e:
                    raise IntegrityViolationError(f"Non-numeric pose on line {line_idx} in {traj_path}: {e}")
                
                # Check for NaNs or Infs
                if any(math.isnan(v) or math.isinf(v) for v in [ts, tx, ty, tz, qx, qy, qz, qw]):
                    raise IntegrityViolationError(f"Non-finite value on line {line_idx} in {traj_path}")
                
                # Quaternion normalization check
                q_norm = math.sqrt(qx*qx + qy*qy + qz*qz + qw*qw)
                if abs(q_norm - 1.0) > 0.05:
                    raise IntegrityViolationError(f"Unnormalized quaternion ({q_norm:.4f}) on line {line_idx} in {traj_path}")
                
                timestamps.append(ts)
                valid_poses += 1

        if valid_poses < min_poses:
            raise IntegrityViolationError(f"Insufficient valid poses ({valid_poses} < {min_poses}) in {traj_path}")
        
        # Check timestamp monotonicity
        for i in range(1, len(timestamps)):
            if timestamps[i] < timestamps[i-1]:
                raise IntegrityViolationError(f"Negative timestamp step at pose {i} in {traj_path}: {timestamps[i]} < {timestamps[i-1]}")

        return {
            "valid_poses": valid_poses,
            "duration_sec": round(timestamps[-1] - timestamps[0], 3),
            "sha256": file_hash
        }

    @staticmethod
    def validate_metric_bounds(
        ate_rmse: Optional[float],
        ate_mean: Optional[float] = None,
        rpe_trans: Optional[float] = None,
        rpe_rot: Optional[float] = None
    ) -> None:
        """Enforces mathematical validity of error metrics (must be non-negative and finite)."""
        metrics = {
            "ate_rmse": ate_rmse,
            "ate_mean": ate_mean,
            "rpe_trans": rpe_trans,
            "rpe_rot": rpe_rot
        }
        for name, val in metrics.items():
            if val is None:
                continue
            if math.isnan(val) or math.isinf(val):
                raise IntegrityViolationError(f"Metric {name} is non-finite: {val}")
            if val < 0.0:
                raise IntegrityViolationError(f"FATAL: Metric {name} cannot be negative! Received: {val}")

    def register_intervention(self, candidate_id: str, traj_hash: str, baseline_hash: str) -> None:
        """Enforces that intervention trajectories are unique and non-identical to baseline."""
        if candidate_id in self.seen_candidate_ids:
            raise IntegrityViolationError(f"Duplicate intervention candidate ID registered: {candidate_id}")
        
        if traj_hash == baseline_hash:
            raise IntegrityViolationError(
                f"Treatment trajectory hash is identical to baseline hash for {candidate_id}! "
                f"No actual SLAM intervention occurred ({traj_hash[:16]})."
            )
            
        if traj_hash in self.seen_trajectory_hashes:
            prev_cand = self.seen_trajectory_hashes[traj_hash]
            raise IntegrityViolationError(
                f"Duplicate treatment trajectory hash detected! Candidate {candidate_id} shares "
                f"identical trajectory with candidate {prev_cand} ({traj_hash[:16]}). "
                "Duplicating trajectories across candidates is strictly prohibited."
            )
            
        self.seen_candidate_ids.add(candidate_id)
        self.seen_trajectory_hashes[traj_hash] = candidate_id

    def validate_vlm_prompt_cleanliness(self, prompt_text: str) -> None:
        """Enforces zero telemetry or ground-truth leakage in visual affordance prompts."""
        if not self.strict_leakage_check:
            return
            
        leakage_keywords = [
            "measured displacement",
            "measured 3d landmark",
            "displacement:",
            "ground truth",
            "groundtruth",
            "future position",
            "baseline ate"
        ]
        lower_prompt = prompt_text.lower()
        for kw in leakage_keywords:
            if kw in lower_prompt:
                raise IntegrityViolationError(
                    f"Telemetry or ground-truth leakage detected in VLM prompt: found '{kw}'. "
                    "VLM affordance queries must rely exclusively on visual crops and semantic category."
                )

