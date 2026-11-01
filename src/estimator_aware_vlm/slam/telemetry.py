"""Native ORB-SLAM3 Telemetry Record Schemas, Importer, and Exporter.

Implements native telemetry parsing, validation, and serialization across:
- frame_feature_associations.jsonl
- landmark_support.jsonl
- optimizer_observations.jsonl
- optimizer_summary.json
- telemetry_manifest.json
"""

from __future__ import annotations

import csv
from dataclasses import asdict, dataclass, field
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union


@dataclass
class FrameFeatureAssociationRecord:
    """Telemetry record associating a 2D keypoint with a 3D landmark and semantic object."""
    frame_id: int
    timestamp: float
    keypoint_index: int
    u: float
    v: float
    map_point_id: Optional[int]
    associated_object_id: Optional[str] = None
    semantic_class: Optional[str] = None
    is_in_dynamic_mask: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class LandmarkSupportRecord:
    """Telemetry record summarizing 3D landmark lifecycle state and observation support."""
    landmark_id: int
    x: float
    y: float
    z: float
    total_observations: int
    semantic_weight: float
    dynamic_state: str  # ACTIVE, TEMPORARILY_EXCLUDED, REACTIVATION_PENDING, REACTIVATED, PERMANENTLY_BAD
    associated_object_id: Optional[str] = None
    semantic_class: Optional[str] = None
    consecutive_inliers: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class OptimizerObservationRecord:
    """Telemetry record from g2o optimizer iteration (PoseOptimization or LocalBA)."""
    optimization_type: str  # "POSE_OPTIMIZATION" or "LOCAL_BUNDLE_ADJUSTMENT"
    frame_id: int
    iteration: int
    keypoint_index: int
    map_point_id: int
    residual_u: float
    residual_v: float
    residual_norm: float
    chi2_raw: float
    semantic_weight: float
    chi2_weighted: float
    is_tracking_inlier: bool
    is_optimizer_inlier: bool

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class OptimizerSummary:
    """Aggregated statistics across SLAM backend optimization sessions."""
    total_frames_optimized: int
    total_edge_observations: int
    total_optimizer_inliers: int
    inlier_ratio: float
    mean_residual_norm: float
    mean_chi2_weighted: float
    temporarily_excluded_count: int
    reactivated_count: int

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_frames_optimized": self.total_frames_optimized,
            "total_edge_observations": self.total_edge_observations,
            "total_optimizer_inliers": self.total_optimizer_inliers,
            "inlier_ratio": round(self.inlier_ratio, 4),
            "mean_residual_norm": round(self.mean_residual_norm, 4),
            "mean_chi2_weighted": round(self.mean_chi2_weighted, 4),
            "temporarily_excluded_count": self.temporarily_excluded_count,
            "reactivated_count": self.reactivated_count,
        }


@dataclass
class TelemetryManifest:
    """Manifest verifying provenance and checksums of exported SLAM telemetry files."""
    schema_version: str = "1.0.0"
    sequence_id: str = ""
    run_id: str = ""
    integration_mode: str = ""  # I0, I1, I2, I3
    orb_slam3_commit: str = "4452a3c4ab75b1cde34e5505a36ec3f9edcdc4c4"
    files: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def save_json(self, path: Union[str, Path]) -> None:
        p = Path(path).resolve()
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)


def compute_file_sha256(path: Union[str, Path]) -> str:
    """Compute SHA-256 hash of a file on disk."""
    p = Path(path).resolve()
    h = hashlib.sha256()
    with open(p, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()

