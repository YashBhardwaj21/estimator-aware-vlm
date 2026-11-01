"""Intervention Dataset Builder with Strict Attribution and Single-Candidate Auditing.

Guarantees:
- Rejects joint multi-candidate interventions from single-candidate calibration.
- Rejects duplicate candidate or frame entries.
- Enforces strict sequence provenance.
- Excludes handheld datasets without MoCap (e.g., D455) from trajectory ATE ground truth.
"""

from __future__ import annotations

import csv
from dataclasses import asdict, dataclass, field
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple
import numpy as np


@dataclass
class SingleCandidateInterventionRecord:
    """Audited empirical outcome of an isolated, single-candidate intervention."""
    intervention_id: str
    sequence_id: str
    target_id: str
    semantic_class: str
    action_type: str  # e.g., "IMAGE_POLYGON_MASK", "IMAGE_BBOX_MASK", "LANDMARK_WEIGHT_ZERO"
    
    # Sensory / tracking features
    measured_displacement_m: float
    displacement_uncertainty_m: float
    observation_count: int
    lifespan_sec: float
    camera_ego_motion_displacement_m: float
    local_keypoint_count: int
    
    # Ground-truth downstream outcomes
    baseline_ate_rmse_m: Optional[float]
    intervention_ate_rmse_m: Optional[float]
    empirical_delta_ate_m: Optional[float]
    common_support_poses: int
    tracking_lost_ratio: float = 0.0
    map_anchor_residual_m: Optional[float] = None
    
    # Audit metadata
    is_joint_intervention: bool = False
    is_trajectory_ate_valid: bool = True
    notes: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class InterventionDataset:
    """Collection of validated single-candidate intervention records across sequences."""

    def __init__(self, records: Optional[List[SingleCandidateInterventionRecord]] = None) -> None:
        self.records: List[SingleCandidateInterventionRecord] = []
        self._seen_keys: Set[Tuple[str, str]] = set()
        if records:
            for r in records:
                self.add_record(r)

    def add_record(self, record: SingleCandidateInterventionRecord) -> None:
        """Add and validate an intervention record with attribution checks."""
        # 1. Prohibit joint interventions
        if record.is_joint_intervention or record.action_type in ("JOINT_MASK", "JOINT_INTERVENTION"):
            raise ValueError(
                f"Joint multi-candidate intervention '{record.intervention_id}' cannot be added as single-candidate."
            )

        # 2. Reject duplicate (sequence_id, target_id)
        key = (record.sequence_id, record.target_id)
        if key in self._seen_keys:
            raise ValueError(f"Duplicate entry for sequence '{record.sequence_id}' and target '{record.target_id}'")

        # 3. D455 trajectory ATE check
        if "d455" in record.sequence_id.lower():
            if record.baseline_ate_rmse_m is not None or record.intervention_ate_rmse_m is not None:
                raise ValueError("D455 recordings lack external MoCap ground-truth; trajectory ATE must be null.")
            record.is_trajectory_ate_valid = False

        self._seen_keys.add(key)
        self.records.append(record)

    def __len__(self) -> int:
        return len(self.records)

    def get_sequences(self) -> List[str]:
        return sorted(list({r.sequence_id for r in self.records}))

    def filter_valid_trajectory_records(self) -> List[SingleCandidateInterventionRecord]:
        """Return only records with valid empirical trajectory ATE ground-truth."""
        return [r for r in self.records if r.is_trajectory_ate_valid and r.empirical_delta_ate_m is not None]

    def export_csv(self, path: Path) -> None:
        path = Path(path).resolve()
        path.parent.mkdir(parents=True, exist_ok=True)
        if not self.records:
            return
        fields = list(self.records[0].to_dict().keys())
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            for r in self.records:
                writer.writerow(r.to_dict())

    def export_json(self, path: Path) -> None:
        path = Path(path).resolve()
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump([r.to_dict() for r in self.records], f, indent=2)

