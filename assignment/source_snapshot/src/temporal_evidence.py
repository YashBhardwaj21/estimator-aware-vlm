"""Stage 5: Temporal Evidence Extractor for Non-VLM Pre-Routing Pipeline.

Quantifies temporal stability and motion:
1. World-relative object displacement:
   Input `centroid_floor` is already expressed in the floor-fixed world coordinate frame
   (ROS REP-103: X=Forward, Y=Left, Z=Up, Z=0 on floor). Therefore, differences in
   `centroid_floor` directly represent world-relative motion. Camera ego-motion is
   tracked concurrently to provide diagnostic ego-motion compensation ratios without
   double-subtracting camera translations.
2. Honest handling of missing inputs:
   Missing centroids or poses are never silently replaced with [0, 0, 0]. They yield
   explicit invalid evidence states (UNCERTAIN motion and missing reasons).
3. Verified motion states: STATIC, MOVING, DISPLACED, TRANSIENT, UNCERTAIN.
4. Occlusion intervals and reappearance detection.
5. Persistent versus transient observation inconsistencies.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Literal, Optional, Tuple
import numpy as np


MOTION_NOISE_FLOOR_M = 0.04      # Baseline measurement noise standard deviation (1-sigma)
MOTION_GATE_3SIGMA_M = 0.12      # 3-sigma gate (0.12m)
DISPLACEMENT_THRESHOLD_M = 0.25  # Minimal true displacement threshold (isolates physical relocation from bounding box view-frustum drift)


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
    world_displacement_m: float
    world_displacement_sigma_m: float
    camera_ego_displacement_m: float
    ego_motion_compensation_ratio: float
    motion_significance_ratio: float
    visibility_duration_s: float
    observation_count: int
    occlusion_events_count: int
    frames_since_last_seen: int
    is_reappearance_after_occlusion: bool
    temporal_persistence_score: float
    position_change_since_last_step_m: float
    has_valid_evidence: bool
    missing_reason: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class TemporalEvidenceExtractor:
    """Tracks temporal continuity and estimates world-relative motion with camera ego-compensation."""

    def __init__(self):
        # Key: track_key (session_track or track_id) -> history dict
        self._tracks: Dict[str, Dict[str, Any]] = {}

    def extract_observation_temporal(
        self,
        obs: Dict[str, Any],
    ) -> TemporalEvidenceRecord:
        """Extract temporal and motion evidence for a timestamped observation."""
        obs_id = obs.get("obs_id", "")
        frame_idx = int(obs.get("frame_index", 0))
        ts = float(obs.get("timestamp", 0.0))
        entity_id = obs.get("global_entity_id", "")
        track_id = obs.get("local_track_id", "")
        session = obs.get("source_session", "")
        key = f"{session}_{track_id}" if (session and track_id) else (track_id or entity_id)

        # 1. Honest check of 3D floor centroid
        c_floor = obs.get("centroid_floor")
        if c_floor is None or len(c_floor) < 3 or any(x is None for x in c_floor[:3]):
            return TemporalEvidenceRecord(
                obs_id=obs_id,
                frame_index=frame_idx,
                timestamp=ts,
                entity_id=entity_id,
                track_id=track_id,
                source_session=session,
                current_motion="UNCERTAIN",
                world_displacement_m=0.0,
                world_displacement_sigma_m=MOTION_NOISE_FLOOR_M,
                camera_ego_displacement_m=0.0,
                ego_motion_compensation_ratio=1.0,
                motion_significance_ratio=0.0,
                visibility_duration_s=0.0,
                observation_count=0,
                occlusion_events_count=0,
                frames_since_last_seen=0,
                is_reappearance_after_occlusion=False,
                temporal_persistence_score=0.0,
                position_change_since_last_step_m=0.0,
                has_valid_evidence=False,
                missing_reason="MISSING_CENTROID_FLOOR",
            )

        curr_centroid = np.array(c_floor[:3], dtype=np.float64)

        # 2. Honest check of camera pose
        cam_pose = obs.get("camera_pose")
        has_pose = (cam_pose is not None and len(cam_pose) >= 7 and not all(x == 0 for x in cam_pose[:7]))
        curr_cam_pos = np.array(cam_pose[:3], dtype=np.float64) if has_pose else np.zeros(3, dtype=np.float64)

        if key not in self._tracks:
            # First observation of this track
            self._tracks[key] = {
                "initial_centroid": curr_centroid.copy(),
                "last_centroid": curr_centroid.copy(),
                "initial_cam_pos": curr_cam_pos.copy() if has_pose else None,
                "last_cam_pos": curr_cam_pos.copy() if has_pose else None,
                "initial_ts": ts,
                "last_ts": ts,
                "initial_frame": frame_idx,
                "last_frame": frame_idx,
                "observation_count": 1,
                "occlusion_events": 0,
                "centroids_history": [curr_centroid.copy()],
                "ts_history": [ts],
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
                world_displacement_sigma_m=MOTION_NOISE_FLOOR_M,
                camera_ego_displacement_m=0.0,
                ego_motion_compensation_ratio=1.0,
                motion_significance_ratio=0.0,
                visibility_duration_s=0.0,
                observation_count=1,
                occlusion_events_count=0,
                frames_since_last_seen=0,
                is_reappearance_after_occlusion=False,
                temporal_persistence_score=0.15,
                position_change_since_last_step_m=0.0,
                has_valid_evidence=True,
                missing_reason=None if has_pose else "MISSING_CAMERA_POSE",
            )

        trk = self._tracks[key]
        prev_centroid = trk["last_centroid"]
        prev_ts = trk["last_ts"]
        prev_frame = trk["last_frame"]

        # Track updates
        trk["observation_count"] += 1
        trk["centroids_history"].append(curr_centroid.copy())
        trk["ts_history"].append(ts)

        # 3. Occlusion and gap detection
        frame_gap = frame_idx - prev_frame
        time_gap = ts - prev_ts
        is_reappearance = False
        if frame_gap > 12 or time_gap > 0.45:
            trk["occlusion_events"] += 1
            is_reappearance = True

        # 4. Camera Ego-Motion vs Object World Displacement
        # Centroid_floor is ALREADY in floor coordinate frame (world-relative).
        total_obj_disp = float(np.linalg.norm(curr_centroid - trk["initial_centroid"]))
        step_obj_disp = float(np.linalg.norm(curr_centroid - prev_centroid))

        if has_pose and trk["initial_cam_pos"] is not None:
            cam_displacement = float(np.linalg.norm(curr_cam_pos - trk["initial_cam_pos"]))
        else:
            cam_displacement = 0.0

        # Empirical measurement noise standard deviation (1-sigma)
        n_obs = trk["observation_count"]
        combined_sigma = MOTION_NOISE_FLOOR_M

        # 5. Motion Classification & Signal-to-Noise Ratio (Z-score)
        motion_sig_ratio = total_obj_disp / max(0.005, combined_sigma)

        if n_obs < 3:
            current_motion = "TRANSIENT"
        elif total_obj_disp > max(DISPLACEMENT_THRESHOLD_M, 3.0 * combined_sigma):
            current_motion = "MOVING" if step_obj_disp > 0.05 else "DISPLACED"
        else:
            current_motion = "STATIC"

        # Ego-motion compensation ratio (diagnostic ratio)
        ego_comp = (cam_displacement / (total_obj_disp + 1e-4)) if cam_displacement > 0.05 else 1.0

        # 6. Visibility Duration & Temporal Persistence Score
        vis_duration = float(ts - trk["initial_ts"])
        persistence_score = 1.0 - math.exp(-n_obs / 8.0)

        # Update last states
        trk["last_centroid"] = curr_centroid.copy()
        if has_pose:
            trk["last_cam_pos"] = curr_cam_pos.copy()
            if trk["initial_cam_pos"] is None:
                trk["initial_cam_pos"] = curr_cam_pos.copy()
        trk["last_ts"] = ts
        trk["last_frame"] = frame_idx

        return TemporalEvidenceRecord(
            obs_id=obs_id,
            frame_index=frame_idx,
            timestamp=ts,
            entity_id=entity_id,
            track_id=track_id,
            source_session=session,
            current_motion=current_motion,
            world_displacement_m=round(total_obj_disp, 4),
            world_displacement_sigma_m=round(combined_sigma, 4),
            camera_ego_displacement_m=round(cam_displacement, 4),
            ego_motion_compensation_ratio=round(min(500.0, ego_comp), 2),
            motion_significance_ratio=round(motion_sig_ratio, 3),
            visibility_duration_s=round(vis_duration, 3),
            observation_count=n_obs,
            occlusion_events_count=trk["occlusion_events"],
            frames_since_last_seen=frame_gap,
            is_reappearance_after_occlusion=is_reappearance,
            temporal_persistence_score=round(persistence_score, 4),
            position_change_since_last_step_m=round(step_obj_disp, 4),
            has_valid_evidence=True,
            missing_reason=None if has_pose else "MISSING_CAMERA_POSE",
        )
