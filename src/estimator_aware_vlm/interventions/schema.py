"""Schema and Data Models for Counterfactual SLAM Interventions.

Defines standardized data models for:
1. Intervention actions (C0..C7) spanning image masking, scene graph, and landmark levels.
2. Intervention specifications (InterventionSpec) detailing target track, dataset, and params.
3. Execution results (InterventionRunResult) and comparative metrics (InterventionMetricRecord).
"""

from __future__ import annotations

import enum
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional


class InterventionAction(str, enum.Enum):
    """Controlled counterfactual intervention action identifiers."""

    # C0: Unchanged baseline (all observations retained)
    KEEP = "KEEP"

    # C1: Suppress image-space features inside segmentation polygon contour
    IMAGE_POLYGON_MASK = "IMAGE_POLYGON_MASK"

    # C2: Suppress image-space features inside object bounding box
    IMAGE_BBOX_MASK = "IMAGE_BBOX_MASK"

    # C3: Retain observation history in scene graph but mark entity suspect/uncertain
    GRAPH_MARK_SUSPECT = "GRAPH_MARK_SUSPECT"

    # C4: Temporarily exclude entity from active scene-graph representation
    GRAPH_TEMPORARY_EXCLUDE = "GRAPH_TEMPORARY_EXCLUDE"

    # C5: Downweight landmark contribution in SLAM bundle adjustment (backend optimizer)
    LANDMARK_DOWNWEIGHT = "LANDMARK_DOWNWEIGHT"

    # C6: Temporarily exclude associated SLAM landmarks with reactivatable status
    LANDMARK_TEMPORARY_EXCLUDE = "LANDMARK_TEMPORARY_EXCLUDE"

    # C7: Joint image-space masking and landmark-level suppression
    DUAL_INTEGRATION = "DUAL_INTEGRATION"

    @property
    def target_type(self) -> InterventionTargetType:
        if self in (InterventionAction.KEEP,):
            return InterventionTargetType.NONE
        elif self in (InterventionAction.IMAGE_POLYGON_MASK, InterventionAction.IMAGE_BBOX_MASK):
            return InterventionTargetType.IMAGE_SPACE
        elif self in (InterventionAction.GRAPH_MARK_SUSPECT, InterventionAction.GRAPH_TEMPORARY_EXCLUDE):
            return InterventionTargetType.SCENE_GRAPH
        elif self in (InterventionAction.LANDMARK_DOWNWEIGHT, InterventionAction.LANDMARK_TEMPORARY_EXCLUDE, InterventionAction.DUAL_INTEGRATION):
            return InterventionTargetType.SLAM_LANDMARK
        return InterventionTargetType.NONE


class InterventionTargetType(str, enum.Enum):
    """Subsystem target layer for intervention action."""
    NONE = "none"
    IMAGE_SPACE = "image_space"
    SCENE_GRAPH = "scene_graph"
    SLAM_LANDMARK = "slam_landmark"


class InterventionStatus(str, enum.Enum):
    """Execution status of an intervention experiment."""
    PENDING = "PENDING"
    RUNNING = "RUNNING"
    SUCCESS = "SUCCESS"
    FAILED = "FAILED"
    NOT_IMPLEMENTED = "NOT_IMPLEMENTED"


@dataclass
class InterventionSpec:
    """Deterministic specification for a single counterfactual intervention run."""

    experiment_run_id: str
    intervention_id: str
    dataset_id: str  # "tum", "openloris", "d455"
    sequence_id: str  # "fr3_walking_xyz", "corridor1-4", "my_desk_recording_01", etc.
    track_id: Optional[str]  # e.g. "monitor_01", None for baseline C0
    action: InterventionAction
    repeat_index: int = 1
    target_type: InterventionTargetType = InterventionTargetType.IMAGE_SPACE
    parameters: Dict[str, Any] = field(default_factory=dict)
    config_path: Optional[str] = None
    groundtruth_path: Optional[str] = None
    baseline_trajectory_path: Optional[str] = None
    observations_path: Optional[str] = None
    created_at: float = field(default_factory=time.time)

    @classmethod
    def generate_id(
        cls,
        dataset_id: str,
        sequence_id: str,
        track_id: Optional[str],
        action: InterventionAction,
        repeat_index: int = 1,
    ) -> str:
        """Generate a canonical, reproducible identifier."""
        target_str = track_id if track_id else "baseline"
        clean_seq = sequence_id.replace("/", "_").replace("\\", "_").replace(" ", "_")
        return f"{dataset_id}_{clean_seq}_{target_str}_{action.value}_r{repeat_index:02d}"

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["action"] = self.action.value
        data["target_type"] = self.target_type.value
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> InterventionSpec:
        data = dict(data)
        data["action"] = InterventionAction(data["action"])
        data["target_type"] = InterventionTargetType(data.get("target_type", data["action"].target_type))
        return cls(**data)


@dataclass
class InterventionRunResult:
    """Execution outcome for a counterfactual intervention."""

    intervention_id: str
    spec: InterventionSpec
    status: InterventionStatus
    trajectory_path: Optional[str] = None
    metrics: Optional[Dict[str, Any]] = None
    error_message: Optional[str] = None
    execution_time_sec: float = 0.0
    runtime_metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "intervention_id": self.intervention_id,
            "spec": self.spec.to_dict(),
            "status": self.status.value,
            "trajectory_path": self.trajectory_path,
            "metrics": self.metrics,
            "error_message": self.error_message,
            "execution_time_sec": self.execution_time_sec,
            "runtime_metadata": self.runtime_metadata,
        }


@dataclass
class InterventionMetricRecord:
    """Summary metric record for comparative reporting and statistical analysis."""

    intervention_id: str
    dataset_id: str
    sequence_id: str
    track_id: Optional[str]
    action: str
    repeat_index: int
    status: str
    ate_rmse_common_m: Optional[float] = None
    ate_baseline_rmse_common_m: Optional[float] = None
    delta_ate_rmse_m: Optional[float] = None  # L(C0) - L(C_i=a) -> positive = error reduction
    percentage_ate_improvement: Optional[float] = None
    ate_rmse_independent_m: Optional[float] = None
    rpe_trans_rmse_m: Optional[float] = None
    rpe_rot_rmse_deg: Optional[float] = None
    tracking_coverage: Optional[float] = None
    common_poses_count: Optional[int] = None
    jaccard_support_index: Optional[float] = None
    paired_ttest_pvalue: Optional[float] = None
    paired_wilcoxon_pvalue: Optional[float] = None
    is_statistically_significant: Optional[bool] = None
    execution_time_sec: float = 0.0
    notes: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

