"""Stage 5: Temporal Evidence Extractor for Non-VLM Pre-Routing Pipeline.

Quantifies temporal stability and motion:
1. Multi-scale world-relative displacement:
   - step_displacement_m: displacement since previous observation.
   - net_displacement_m: net displacement from initial track centroid.
   - cumulative_path_length_m: total integrated path traversed over track lifetime.
   (Input centroid_floor is expressed in floor-fixed world coordinates).
2. Honest handling of missing inputs:
   - Missing centroids or poses yield explicit invalid evidence states (UNCERTAIN motion).
3. Verified motion states: STATIC, MOVING, DISPLACED, TRANSIENT, UNCERTAIN.
4. Observation gap events and reappearance detection.
5. Deterministic session-scoped track state keys (zero cross-session track collisions).
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Literal, Optional, Tuple
import numpy as np


# Configuration Constants
MOTION_NOISE_FLOOR_M = 0.04      # Configured baseline measurement noise (1-sigma, 40mm)
MOTION_GATE_3SIGMA_M = 0.12      # 3-sigma gate (120mm)
DISPLACEMENT_THRESHOLD_M = 0.25  # Minimal physical relocation threshold (250mm)
STEP_MOTION_THRESHOLD_M = 0.05   # Step displacement threshold for active instantaneous movement (50mm)
CUMULATIVE_PATH_THRESHOLD_M = 0.50 # Cumulative path threshold for move-and-return relocation detection
FRAME_GAP_THRESHOLD = 12         # Sampling gap frame threshold (~0.4s at 30Hz)
TIME_GAP_THRESHOLD_S = 0.45      # Sampling gap timestamp threshold (seconds)


@dataclass
class TemporalEvidenceRecord:
    """Timestamp-aligned temporal tracking and motion evidence record."""
    obs_id: str
    frame_index: int
    timestamp: float
    entity_id: str
    track_id: str
    source_session: str
    current_motion: Literal["STATIC", "MOVING", "DISPLACED", "TRANSIENT", "UNCERTAIN"]
    world_displacement_m: float                  # Backwards-compatible alias for net displacement
    net_displacement_m: float                    # Distance from initial track centroid
    step_displacement_m: float                   # Distance from immediately preceding observation
    cumulative_path_length_m: float              # Integrated path length along track trajectory
    world_displacement_sigma_m: float            # Configured measurement uncertainty
    camera_ego_displacement_m: float
    ego_motion_compensation_ratio: float
    motion_significance_ratio: float
    visibility_duration_s: float
    observation_count: int
    observation_gap_events_count: int
    frames_since_last_seen: int
    is_reappearance_after_gap: bool
    temporal_persistence_score: float
    position_change_since_last_step_m: float     # Backwards-compatible alias for step displacement
    has_valid_evidence: bool
    missing_reason: Optional[str] = None
    temporal_diagnostics: List[str] = None

    # Backwards-compatibility properties
    @property
    def occlusion_events_count(self) -> int:
        return self.observation_gap_events_count

    @property
    def is_reappearance_after_occlusion(self) -> bool:
        return self.is_reappearance_after_gap

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["occlusion_events_count"] = self.observation_gap_events_count
        d["is_reappearance_after_occlusion"] = self.is_reappearance_after_gap
        return d


class TemporalEvidenceExtractor:
    """Tracks temporal continuity and estimates world-relative motion with session-scoped keys."""

    def __init__(self):
        # Key: (session, track_id) -> state dict
        self._tracks: Dict[Tuple[str, str], Dict[str, Any]] = {}

    def extract_observation_temporal(
        self,
        obs: Dict[str, Any],
    ) -> TemporalEvidenceRecord:
        """Extract temporal and motion evidence for a timestamped observation."""
        obs_id = str(obs.get("obs_id", "")).strip()
        frame_idx = int(obs.get("frame_index", 0))
        ts = float(obs.get("timestamp", 0.0))
        entity_id = str(obs.get("global_entity_id", "")).strip()
        track_id = str(obs.get("local_track_id", "")).strip()
        session = str(obs.get("source_session", "")).strip()

        # Session-qualified track key (never bare track_id)
        sess_key = session if session else "__unspecified_session__"
        if track_id:
            track_key = (sess_key, track_id)
        elif entity_id:
            track_key = (sess_key, entity_id)
        else:
            # Isolated key for un-tracked observations
            track_key = (sess_key, f"__isolated_{obs_id}__")

        diagnostics: List[str] = []

        # 1. Honest check of 3D floor centroid
        c_floor = obs.get("centroid_floor")
        if (
            c_floor is None
            or not isinstance(c_floor, (list, tuple))
            or len(c_floor) < 3
            or any(x is None or not (isinstance(x, (int, float)) and math.isfinite(x)) for x in c_floor[:3])
        ):
            return TemporalEvidenceRecord(
                obs_id=obs_id,
                frame_index=frame_idx,
                timestamp=ts,
                entity_id=entity_id,
                track_id=track_id,
                source_session=session,
                current_motion="UNCERTAIN",
                world_displacement_m=0.0,
                net_displacement_m=0.0,
                step_displacement_m=0.0,
                cumulative_path_length_m=0.0,
                world_displacement_sigma_m=MOTION_NOISE_FLOOR_M,
                camera_ego_displacement_m=0.0,
                ego_motion_compensation_ratio=1.0,
                motion_significance_ratio=0.0,
                visibility_duration_s=0.0,
                observation_count=0,
                observation_gap_events_count=0,
                frames_since_last_seen=0,
                is_reappearance_after_gap=False,
                temporal_persistence_score=0.0,
                position_change_since_last_step_m=0.0,
                has_valid_evidence=False,
                missing_reason="MISSING_CENTROID_FLOOR",
                temporal_diagnostics=["Missing or non-finite centroid_floor in observation."],
            )

        curr_centroid = np.array(c_floor[:3], dtype=np.float64)

        # 2. Camera pose check
        cam_pose = obs.get("camera_pose")
        has_pose = False
        curr_cam_pos = np.zeros(3, dtype=np.float64)
        if cam_pose is not None and isinstance(cam_pose, (list, tuple)) and len(cam_pose) >= 7:
            if all(isinstance(v, (int, float)) and math.isfinite(v) for v in cam_pose[:7]):
                qx, qy, qz, qw = cam_pose[3], cam_pose[4], cam_pose[5], cam_pose[6]
                q_norm = math.sqrt(qx * qx + qy * qy + qz * qz + qw * qw)
                if 0.90 <= q_norm <= 1.10:
                    has_pose = True
                    curr_cam_pos = np.array(cam_pose[:3], dtype=np.float64)

        if not has_pose:
            diagnostics.append("Camera pose missing or unnormalized; camera ego-motion unavailable.")

        # 3. Track History Management
        if track_key not in self._tracks:
            # First observation of this track
            self._tracks[track_key] = {
                "initial_centroid": curr_centroid.copy(),
                "last_centroid": curr_centroid.copy(),
                "initial_cam_pos": curr_cam_pos.copy() if has_pose else None,
                "last_cam_pos": curr_cam_pos.copy() if has_pose else None,
                "initial_ts": ts,
                "last_ts": ts,
                "initial_frame": frame_idx,
                "last_frame": frame_idx,
                "observation_count": 1,
                "gap_events": 0,
                "cumulative_path": 0.0,
            }
            return TemporalEvidenceRecord(
                obs_id=obs_id,
                frame_index=frame_idx,
                timestamp=ts,
                entity_id=entity_id,
                track_id=track_id,
                source_session=session,
                current_motion="TRANSIENT",
                world_displacement_m=0.0,
                net_displacement_m=0.0,
                step_displacement_m=0.0,
                cumulative_path_length_m=0.0,
                world_displacement_sigma_m=MOTION_NOISE_FLOOR_M,
                camera_ego_displacement_m=0.0,
                ego_motion_compensation_ratio=1.0,
                motion_significance_ratio=0.0,
                visibility_duration_s=0.0,
                observation_count=1,
                observation_gap_events_count=0,
                frames_since_last_seen=0,
                is_reappearance_after_gap=False,
                temporal_persistence_score=0.15,
                position_change_since_last_step_m=0.0,
                has_valid_evidence=True,
                missing_reason=None if has_pose else "MISSING_CAMERA_POSE",
                temporal_diagnostics=diagnostics,
            )

        trk = self._tracks[track_key]
        prev_centroid = trk["last_centroid"]
        prev_ts = trk["last_ts"]
        prev_frame = trk["last_frame"]

        # Validate non-decreasing time and frame index
        if ts < prev_ts or frame_idx < prev_frame:
            diagnostics.append(
                f"Non-monotonic timestamp/frame detected: (ts={ts:.3f} < prev={prev_ts:.3f}) "
                f"or (frame={frame_idx} < prev={prev_frame})."
            )

        frame_gap = max(0, frame_idx - prev_frame)
        time_gap = max(0.0, ts - prev_ts)

        # 4. Gap and Reappearance Detection
        is_reappearance = False
        if frame_gap > FRAME_GAP_THRESHOLD or time_gap > TIME_GAP_THRESHOLD_S:
            trk["gap_events"] += 1
            is_reappearance = True

        # 5. Multi-Scale Displacement Calculations
        step_disp = float(np.linalg.norm(curr_centroid - prev_centroid))
        net_disp = float(np.linalg.norm(curr_centroid - trk["initial_centroid"]))
        trk["cumulative_path"] += step_disp
        cum_path = float(trk["cumulative_path"])

        trk["observation_count"] += 1
        n_obs = trk["observation_count"]

        # Camera ego-displacement
        cam_disp = 0.0
        if has_pose:
            if trk["initial_cam_pos"] is not None:
                cam_disp = float(np.linalg.norm(curr_cam_pos - trk["initial_cam_pos"]))
            else:
                trk["initial_cam_pos"] = curr_cam_pos.copy()

        # Motion classification
        combined_sigma = MOTION_NOISE_FLOOR_M
        motion_sig_ratio = net_disp / max(0.005, combined_sigma)

        if n_obs < 3:
            current_motion = "TRANSIENT"
        elif step_disp > STEP_MOTION_THRESHOLD_M:
            current_motion = "MOVING"
        elif net_disp > DISPLACEMENT_THRESHOLD_M:
            current_motion = "DISPLACED"
        elif cum_path > CUMULATIVE_PATH_THRESHOLD_M and step_disp <= STEP_MOTION_THRESHOLD_M:
            # Handles move-and-return trajectory: net displacement is small but cumulative motion is large
            current_motion = "DISPLACED"
        else:
            current_motion = "STATIC"

        # Diagnostic ego-motion compensation ratio
        ego_comp = (cam_disp / (net_disp + 1e-4)) if cam_disp > 0.05 else 1.0

        # Visibility and persistence score
        vis_duration = max(0.0, float(ts - trk["initial_ts"]))
        persistence_score = 1.0 - math.exp(-n_obs / 8.0)

        # Update last states
        trk["last_centroid"] = curr_centroid.copy()
        if has_pose:
            trk["last_cam_pos"] = curr_cam_pos.copy()
        trk["last_ts"] = max(ts, prev_ts)
        trk["last_frame"] = max(frame_idx, prev_frame)

        return TemporalEvidenceRecord(
            obs_id=obs_id,
            frame_index=frame_idx,
            timestamp=ts,
            entity_id=entity_id,
            track_id=track_id,
            source_session=session,
            current_motion=current_motion,
            world_displacement_m=round(net_disp, 4),
            net_displacement_m=round(net_disp, 4),
            step_displacement_m=round(step_disp, 4),
            cumulative_path_length_m=round(cum_path, 4),
            world_displacement_sigma_m=round(combined_sigma, 4),
            camera_ego_displacement_m=round(cam_disp, 4),
            ego_motion_compensation_ratio=round(min(500.0, ego_comp), 2),
            motion_significance_ratio=round(motion_sig_ratio, 3),
            visibility_duration_s=round(vis_duration, 3),
            observation_count=n_obs,
            observation_gap_events_count=trk["gap_events"],
            frames_since_last_seen=frame_gap,
            is_reappearance_after_gap=is_reappearance,
            temporal_persistence_score=round(persistence_score, 4),
            position_change_since_last_step_m=round(step_disp, 4),
            has_valid_evidence=True,
            missing_reason=None if has_pose else "MISSING_CAMERA_POSE",
            temporal_diagnostics=diagnostics,
        )
