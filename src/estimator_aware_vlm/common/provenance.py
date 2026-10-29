"""Evidence Provenance and Telemetry Audit Framework.

Classifies all perception and SLAM fields into three strict, auditable categories:
1. MEASURED_BACKEND: Directly acquired from sensor hardware (timestamps, depth, images)
   or state estimator backend outputs (trajectory poses, graph residuals, inlier sets).
2. DERIVED: Deterministic mathematical transformations of measured backend data without
   heuristic assumptions (e.g. SE(3) camera ego-motion compensation, metric Euclidean distance,
   temporal duration, sample standard deviation).
3. PROXY: Approximations, 2D projections, or heuristic thresholds where direct 3D bundle-adjustment
   telemetry is unavailable (e.g. 2D bounding box depth projection, heuristic motion state thresholds,
   object detector classification votes).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import Enum
from typing import Any, Dict, Optional


class EvidenceProvenance(str, Enum):
    """Provenance level of perception and estimator data fields."""
    MEASURED_BACKEND = "MEASURED_BACKEND"
    DERIVED = "DERIVED"
    PROXY = "PROXY"


@dataclass(frozen=True)
class FieldProvenanceMeta:
    """Provenance metadata for an auditable field."""
    field_name: str
    provenance: EvidenceProvenance
    source: str
    description: str

    def to_dict(self) -> Dict[str, str]:
        return {
            "field_name": self.field_name,
            "provenance": self.provenance.value,
            "source": self.source,
            "description": self.description,
        }


# Canonical provenance mappings for entity tracking & SLAM evidence
OBSERVATION_FIELD_PROVENANCE: Dict[str, FieldProvenanceMeta] = {
    "frame_index": FieldProvenanceMeta("frame_index", EvidenceProvenance.MEASURED_BACKEND, "sensor_stream", "Frame sequence number in video stream"),
    "timestamp": FieldProvenanceMeta("timestamp", EvidenceProvenance.MEASURED_BACKEND, "sensor_hardware", "Monotonic camera hardware acquisition timestamp"),
    "median_depth_m": FieldProvenanceMeta("median_depth_m", EvidenceProvenance.MEASURED_BACKEND, "depth_sensor", "Direct sensor depth map measurement at object centroid"),
    "camera_position_world": FieldProvenanceMeta("camera_position_world", EvidenceProvenance.MEASURED_BACKEND, "slam_trajectory", "Estimated camera translation from Pass A SLAM"),
    "centroid_camera": FieldProvenanceMeta("centroid_camera", EvidenceProvenance.DERIVED, "pinhole_backprojection", "Pinhole inverse projection: K^{-1} [u, v, 1]^T * Z"),
    "centroid_world": FieldProvenanceMeta("centroid_world", EvidenceProvenance.DERIVED, "se3_compensation", "Camera ego-motion compensated 3D point: R_wc * P_cam + t_wc"),
    "confidence": FieldProvenanceMeta("confidence", EvidenceProvenance.PROXY, "yolo_detector", "2D object detector bounding box confidence score"),
    "bbox_xyxy": FieldProvenanceMeta("bbox_xyxy", EvidenceProvenance.PROXY, "yolo_detector", "2D pixel coordinate bounding box"),
    "mask_contour": FieldProvenanceMeta("mask_contour", EvidenceProvenance.PROXY, "yolo_segmentation", "2D pixel polygon instance segmentation contour"),
    "semantic_class": FieldProvenanceMeta("semantic_class", EvidenceProvenance.PROXY, "yolo_detector", "Predicted COCO class mapped to indoor ontology"),
}

TRACK_FIELD_PROVENANCE: Dict[str, FieldProvenanceMeta] = {
    "first_timestamp": FieldProvenanceMeta("first_timestamp", EvidenceProvenance.MEASURED_BACKEND, "sensor_hardware", "Timestamp of initial observation"),
    "last_timestamp": FieldProvenanceMeta("last_timestamp", EvidenceProvenance.MEASURED_BACKEND, "sensor_hardware", "Timestamp of final observation"),
    "track_id": FieldProvenanceMeta("track_id", EvidenceProvenance.DERIVED, "causal_tracker", "Deterministic 3D Euclidean nearest-neighbor tracklet association"),
    "observation_count": FieldProvenanceMeta("observation_count", EvidenceProvenance.DERIVED, "causal_tracker", "Total count of valid observations aggregated in tracklet"),
    "lifespan_sec": FieldProvenanceMeta("lifespan_sec", EvidenceProvenance.DERIVED, "causal_tracker", "Elapsed time delta between last and first observation"),
    "camera_ego_motion_displacement_m": FieldProvenanceMeta("camera_ego_motion_displacement_m", EvidenceProvenance.DERIVED, "slam_trajectory", "Norm of camera translation delta during tracking window"),
    "apparent_camera_frame_displacement_m": FieldProvenanceMeta("apparent_camera_frame_displacement_m", EvidenceProvenance.DERIVED, "sensor_depth", "Norm of camera-frame centroid delta (corrupted by ego-motion)"),
    "world_displacement_m": FieldProvenanceMeta("world_displacement_m", EvidenceProvenance.DERIVED, "se3_compensation", "Norm of camera-compensated 3D world centroid displacement"),
    "measured_displacement_m": FieldProvenanceMeta("measured_displacement_m", EvidenceProvenance.DERIVED, "se3_compensation", "Effective 3D world displacement (max of end-to-end and excursion)"),
    "displacement_uncertainty_m": FieldProvenanceMeta("displacement_uncertainty_m", EvidenceProvenance.DERIVED, "statistics", "Spatial position dispersion standard deviation around mean world centroid"),
    "observed_motion": FieldProvenanceMeta("observed_motion", EvidenceProvenance.PROXY, "heuristic_thresholds", "Three-tier motion classification (STATIC < 0.12m, DISPLACED < 0.35m, MOVING >= 0.35m)"),
    "semantic_class": FieldProvenanceMeta("semantic_class", EvidenceProvenance.PROXY, "majority_vote", "Temporal majority vote of 2D object detector class labels"),
    "inlier_ratio": FieldProvenanceMeta("inlier_ratio", EvidenceProvenance.PROXY, "bbox_depth_projection", "Fraction of candidate 3D points projecting inside 2D bounding box"),
}


def get_track_provenance_manifest() -> Dict[str, Dict[str, str]]:
    """Return dictionary of all track field provenance metadata."""
    return {k: meta.to_dict() for k, meta in TRACK_FIELD_PROVENANCE.items()}


def get_observation_provenance_manifest() -> Dict[str, Dict[str, str]]:
    """Return dictionary of all observation field provenance metadata."""
    return {k: meta.to_dict() for k, meta in OBSERVATION_FIELD_PROVENANCE.items()}

