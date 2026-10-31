"""Common Observation & Evidence Record Schema.

Defines the unified, zero-fabrication observation record schema across all three
experimental datasets (TUM RGB-D, OpenLORIS, RealSense D455).

Schema fields adhere strictly to:
- dataset: str
- sequence: str
- frame_id: int
- timestamp: float
- camera_pose: Optional[List[float]] (7-element [tx, ty, tz, qx, qy, qz, qw])
- pose_source: Optional[str] ("ORB_SLAM3", "GROUND_TRUTH", etc.)
- track_id: str
- object_class: str
- bbox: Optional[List[float]] ([xmin, ymin, xmax, ymax])
- mask_path: Optional[str]
- crop_path: Optional[str]
- depth_mean_m: Optional[float]
- depth_variance: Optional[float]
- observed_displacement_m: Optional[float]
- displacement_uncertainty_m: Optional[float]
- landmark_observation_count: Optional[int]
- reprojection_error: Optional[float]
- ground_truth_motion_state: Optional[str] ("STATIC", "MOVING", None)
- ground_truth_affordance: Optional[str] ("MOVABLE", "IMMOVABLE", "MOBILE", None)

Missing fields remain explicitly None / null. No synthetic defaults or heuristic
fabrication is permitted.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Union


@dataclass
class CommonObservationRecord:
    dataset: str
    sequence: str
    frame_id: int
    timestamp: float
    camera_pose: Optional[List[float]]
    pose_source: Optional[str]
    track_id: str
    object_class: str
    bbox: Optional[List[float]]
    mask_path: Optional[str] = None
    crop_path: Optional[str] = None
    depth_mean_m: Optional[float] = None
    depth_variance: Optional[float] = None
    observed_displacement_m: Optional[float] = None
    displacement_uncertainty_m: Optional[float] = None
    landmark_observation_count: Optional[int] = None
    reprojection_error: Optional[float] = None
    ground_truth_motion_state: Optional[str] = None
    ground_truth_affordance: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert record to dictionary matching canonical schema."""
        return {
            "dataset": self.dataset,
            "sequence": self.sequence,
            "frame_id": int(self.frame_id),
            "timestamp": float(self.timestamp),
            "camera_pose": [float(v) for v in self.camera_pose] if self.camera_pose is not None else None,
            "pose_source": self.pose_source,
            "track_id": self.track_id,
            "object_class": self.object_class,
            "bbox": [float(b) for b in self.bbox] if self.bbox is not None else None,
            "mask_path": self.mask_path,
            "crop_path": self.crop_path,
            "depth_mean_m": float(self.depth_mean_m) if self.depth_mean_m is not None else None,
            "depth_variance": float(self.depth_variance) if self.depth_variance is not None else None,
            "observed_displacement_m": float(self.observed_displacement_m) if self.observed_displacement_m is not None else None,
            "displacement_uncertainty_m": float(self.displacement_uncertainty_m) if self.displacement_uncertainty_m is not None else None,
            "landmark_observation_count": int(self.landmark_observation_count) if self.landmark_observation_count is not None else None,
            "reprojection_error": float(self.reprojection_error) if self.reprojection_error is not None else None,
            "ground_truth_motion_state": self.ground_truth_motion_state,
            "ground_truth_affordance": self.ground_truth_affordance,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> CommonObservationRecord:
        """Create a CommonObservationRecord from a dictionary."""
        return cls(
            dataset=data["dataset"],
            sequence=data["sequence"],
            frame_id=data["frame_id"],
            timestamp=data["timestamp"],
            camera_pose=data.get("camera_pose"),
            pose_source=data.get("pose_source"),
            track_id=data["track_id"],
            object_class=data["object_class"],
            bbox=data.get("bbox"),
            mask_path=data.get("mask_path"),
            crop_path=data.get("crop_path"),
            depth_mean_m=data.get("depth_mean_m"),
            depth_variance=data.get("depth_variance"),
            observed_displacement_m=data.get("observed_displacement_m"),
            displacement_uncertainty_m=data.get("displacement_uncertainty_m"),
            landmark_observation_count=data.get("landmark_observation_count"),
            reprojection_error=data.get("reprojection_error"),
            ground_truth_motion_state=data.get("ground_truth_motion_state"),
            ground_truth_affordance=data.get("ground_truth_affordance"),
        )

