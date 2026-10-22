"""Stage 5: Source Record Validation and Schema Enforcement.

Provides fail-closed, strict validation for raw object observation records before
they enter geometric, temporal, semantic, and fused evidence extraction.

Guarantees:
1. No silent line dropping: non_empty_lines == accepted_records + rejected_records.
2. Type and numerical safety: validates finite numbers, bounds, and array shapes.
3. Strict session-scoped identity: enforces unique (source_session, obs_id) composite keys.
4. Modality-specific validity flags: geometry_valid, temporal_valid, semantic_valid.
5. Rejection audit trail: records line number, obs_id, reason, and field diagnostics.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple


@dataclass
class RejectionRecord:
    """Audit record for a rejected source observation line."""
    line_number: int
    raw_preview: str
    raw_sha256: str
    obs_id: Optional[str]
    source_session: Optional[str]
    reason: str
    diagnostic_details: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "line_number": self.line_number,
            "raw_preview": self.raw_preview,
            "raw_sha256": self.raw_sha256,
            "obs_id": self.obs_id,
            "source_session": self.source_session,
            "reason": self.reason,
            "diagnostic_details": self.diagnostic_details,
        }


@dataclass
class ValidatedModalityStatus:
    """Detailed modality readiness flags computed during source validation."""
    has_valid_identity: bool = False
    has_valid_geometry_inputs: bool = False
    has_valid_temporal_inputs: bool = False
    has_valid_semantic_inputs: bool = False
    has_valid_pose: bool = False
    has_valid_depth: bool = False
    has_valid_floor_centroid: bool = False
    has_valid_bbox: bool = False
    missing_fields: List[str] = field(default_factory=list)
    diagnostics: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "has_valid_identity": self.has_valid_identity,
            "has_valid_geometry_inputs": self.has_valid_geometry_inputs,
            "has_valid_temporal_inputs": self.has_valid_temporal_inputs,
            "has_valid_semantic_inputs": self.has_valid_semantic_inputs,
            "has_valid_pose": self.has_valid_pose,
            "has_valid_depth": self.has_valid_depth,
            "has_valid_floor_centroid": self.has_valid_floor_centroid,
            "has_valid_bbox": self.has_valid_bbox,
            "missing_fields": self.missing_fields,
            "diagnostics": self.diagnostics,
        }


def is_finite_number(val: Any) -> bool:
    """Return True if val is an int or float and is finite (not NaN, not Inf)."""
    if isinstance(val, (int, float)):
        return not (math.isnan(val) or math.isinf(val))
    return False


def is_finite_vector(vec: Any, expected_dim: Optional[int] = None) -> bool:
    """Return True if vec is a list or tuple of finite numbers with expected dimension."""
    if not isinstance(vec, (list, tuple)):
        return False
    if expected_dim is not None and len(vec) != expected_dim:
        return False
    return all(is_finite_number(x) for x in vec)


class SourceRecordValidator:
    """Stateful validator for streaming observation JSONL records."""

    def __init__(self, strict_mode: bool = True):
        self.strict_mode = strict_mode
        self.seen_composite_ids: Set[Tuple[str, str]] = set()
        self.total_lines_read = 0
        self.total_non_empty_lines = 0
        self.total_accepted = 0
        self.total_rejected = 0
        self.rejections: List[RejectionRecord] = []

    def validate_line(
        self,
        raw_line: str,
        line_number: int,
    ) -> Tuple[Optional[Dict[str, Any]], Optional[ValidatedModalityStatus], Optional[RejectionRecord]]:
        """Validate a single raw JSONL line.

        Returns:
            (parsed_record, modality_status, rejection_record)
            If rejected, parsed_record and modality_status are None.
            If accepted, rejection_record is None.
        """
        self.total_lines_read += 1
        stripped = raw_line.strip()
        if not stripped:
            return None, None, None

        self.total_non_empty_lines += 1
        line_hash = hashlib.sha256(raw_line.encode("utf-8")).hexdigest()
        preview = stripped[:120] + ("..." if len(stripped) > 120 else "")

        # 1. JSON Parse
        try:
            record = json.loads(stripped)
        except json.JSONDecodeError as err:
            rejection = RejectionRecord(
                line_number=line_number,
                raw_preview=preview,
                raw_sha256=line_hash,
                obs_id=None,
                source_session=None,
                reason="JSON_DECODE_ERROR",
                diagnostic_details=f"Invalid JSON at line {line_number}: {err}",
            )
            self.total_rejected += 1
            self.rejections.append(rejection)
            if self.strict_mode:
                raise ValueError(f"Strict mode rejected malformed JSON at line {line_number}: {err}") from err
            return None, None, rejection

        # 2. Type validation: must be a JSON object (dict)
        if not isinstance(record, dict):
            rejection = RejectionRecord(
                line_number=line_number,
                raw_preview=preview,
                raw_sha256=line_hash,
                obs_id=None,
                source_session=None,
                reason="INVALID_RECORD_TYPE",
                diagnostic_details=f"Line {line_number} contains {type(record).__name__}, expected JSON object (dict).",
            )
            self.total_rejected += 1
            self.rejections.append(rejection)
            if self.strict_mode:
                raise TypeError(f"Line {line_number} must be a JSON object, got {type(record).__name__}")
            return None, None, rejection

        # 3. Core Identity & Timestamp Validation
        obs_id = record.get("obs_id")
        if not isinstance(obs_id, str) or not obs_id.strip():
            rejection = RejectionRecord(
                line_number=line_number,
                raw_preview=preview,
                raw_sha256=line_hash,
                obs_id=None,
                source_session=record.get("source_session"),
                reason="MISSING_OR_EMPTY_OBS_ID",
                diagnostic_details=f"Missing or non-string obs_id at line {line_number}: {obs_id!r}",
            )
            self.total_rejected += 1
            self.rejections.append(rejection)
            if self.strict_mode:
                raise ValueError(f"Line {line_number} missing valid obs_id: {obs_id!r}")
            return None, None, rejection
        obs_id = obs_id.strip()

        source_session = record.get("source_session")
        if not isinstance(source_session, str) or not source_session.strip():
            rejection = RejectionRecord(
                line_number=line_number,
                raw_preview=preview,
                raw_sha256=line_hash,
                obs_id=obs_id,
                source_session=None,
                reason="MISSING_OR_EMPTY_SOURCE_SESSION",
                diagnostic_details=f"Missing or non-string source_session at line {line_number}: {source_session!r}",
            )
            self.total_rejected += 1
            self.rejections.append(rejection)
            if self.strict_mode:
                raise ValueError(f"Line {line_number} (obs_id={obs_id}) missing valid source_session.")
            return None, None, rejection
        source_session = source_session.strip()

        # Enforce unique composite key (source_session, obs_id)
        comp_key = (source_session, obs_id)
        if comp_key in self.seen_composite_ids:
            rejection = RejectionRecord(
                line_number=line_number,
                raw_preview=preview,
                raw_sha256=line_hash,
                obs_id=obs_id,
                source_session=source_session,
                reason="DUPLICATE_COMPOSITE_ID",
                diagnostic_details=f"Duplicate observation ID '{obs_id}' in session '{source_session}' at line {line_number}.",
            )
            self.total_rejected += 1
            self.rejections.append(rejection)
            if self.strict_mode:
                raise ValueError(f"Duplicate (session, obs_id) pair {comp_key} at line {line_number}.")
            return None, None, rejection

        # Validate frame_index
        frame_idx = record.get("frame_index")
        if not isinstance(frame_idx, int) or frame_idx < 0:
            rejection = RejectionRecord(
                line_number=line_number,
                raw_preview=preview,
                raw_sha256=line_hash,
                obs_id=obs_id,
                source_session=source_session,
                reason="INVALID_FRAME_INDEX",
                diagnostic_details=f"Invalid frame_index {frame_idx!r} at line {line_number}. Must be integer >= 0.",
            )
            self.total_rejected += 1
            self.rejections.append(rejection)
            if self.strict_mode:
                raise ValueError(f"Invalid frame_index at line {line_number}: {frame_idx!r}")
            return None, None, rejection

        # Validate timestamp
        ts = record.get("timestamp")
        if not is_finite_number(ts) or ts < 0.0:
            rejection = RejectionRecord(
                line_number=line_number,
                raw_preview=preview,
                raw_sha256=line_hash,
                obs_id=obs_id,
                source_session=source_session,
                reason="INVALID_TIMESTAMP",
                diagnostic_details=f"Invalid timestamp {ts!r} at line {line_number}. Must be finite float >= 0.",
            )
            self.total_rejected += 1
            self.rejections.append(rejection)
            if self.strict_mode:
                raise ValueError(f"Invalid timestamp at line {line_number}: {ts!r}")
            return None, None, rejection

        # 4. Modality Status & Optional Field Checks
        mod_status = ValidatedModalityStatus(has_valid_identity=True)

        # Check camera pose: 7 elements [tx, ty, tz, qx, qy, qz, qw]
        cam_pose = record.get("camera_pose")
        if cam_pose is not None:
            if is_finite_vector(cam_pose, 7):
                qx, qy, qz, qw = cam_pose[3], cam_pose[4], cam_pose[5], cam_pose[6]
                q_norm = math.sqrt(qx * qx + qy * qy + qz * qz + qw * qw)
                if 0.90 <= q_norm <= 1.10:
                    mod_status.has_valid_pose = True
                else:
                    mod_status.diagnostics.append(f"Invalid quaternion norm {q_norm:.4f} in camera_pose.")
            else:
                mod_status.diagnostics.append(f"Malformed camera_pose array: {cam_pose!r}")
        else:
            mod_status.missing_fields.append("camera_pose")

        # Check depth (centroid_camera[2])
        c_cam = record.get("centroid_camera")
        if c_cam is not None and is_finite_vector(c_cam, 3):
            z = float(c_cam[2])
            if 0.15 <= z <= 6.0:
                mod_status.has_valid_depth = True
            else:
                mod_status.diagnostics.append(f"Depth {z:.3f}m out of sensor operating range [0.15, 6.0]m.")
        else:
            mod_status.missing_fields.append("centroid_camera")

        # Check floor centroid
        c_floor = record.get("centroid_floor")
        if c_floor is not None and is_finite_vector(c_floor, 3):
            mod_status.has_valid_floor_centroid = True
        else:
            mod_status.missing_fields.append("centroid_floor")

        # Check bbox_xyxy: [x1, y1, x2, y2]
        bbox = record.get("bbox_xyxy")
        if bbox is not None:
            if is_finite_vector(bbox, 4):
                x1, y1, x2, y2 = bbox
                if x1 < x2 and y1 < y2:
                    mod_status.has_valid_bbox = True
                else:
                    mod_status.diagnostics.append(f"Inverted bbox coordinates: [{x1}, {y1}, {x2}, {y2}].")
            else:
                mod_status.diagnostics.append(f"Malformed bbox_xyxy: {bbox!r}")
        else:
            mod_status.missing_fields.append("bbox_xyxy")

        # Check confidence
        conf = record.get("confidence")
        if conf is not None:
            if not is_finite_number(conf) or not (0.0 <= conf <= 1.0):
                mod_status.diagnostics.append(f"Confidence {conf!r} out of range [0.0, 1.0].")

        # Check class name
        cname = record.get("class_name")
        if cname is not None and isinstance(cname, str) and cname.strip():
            mod_status.has_valid_semantic_inputs = True
        else:
            mod_status.missing_fields.append("class_name")

        # Geometry branch input validity
        mod_status.has_valid_geometry_inputs = bool(
            mod_status.has_valid_depth
            or (mod_status.has_valid_pose and mod_status.has_valid_bbox and mod_status.has_valid_floor_centroid)
        )

        # Temporal branch input validity
        mod_status.has_valid_temporal_inputs = bool(
            mod_status.has_valid_floor_centroid and record.get("local_track_id")
        )

        self.seen_composite_ids.add(comp_key)
        self.total_accepted += 1
        return record, mod_status, None

    def reconcile_summary(self) -> Dict[str, Any]:
        """Verify and return exact reconciliation counters."""
        total_accounted = self.total_accepted + self.total_rejected
        is_consistent = (total_accounted == self.total_non_empty_lines)
        return {
            "total_lines_read": self.total_lines_read,
            "total_non_empty_lines": self.total_non_empty_lines,
            "total_accepted": self.total_accepted,
            "total_rejected": self.total_rejected,
            "reconciliation_passed": is_consistent,
            "unique_composite_ids_count": len(self.seen_composite_ids),
            "rejections": [r.to_dict() for r in self.rejections],
        }
