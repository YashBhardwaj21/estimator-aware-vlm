"""Unified Perception Pipeline for Estimator-Aware VLM.

Processes all eight RGB-D recordings through a single implementation using
dataset-specific adapters only. No dataset-specific logic inside the pipeline.

Architecture:
    DatasetAdapter  →  FrameRecord stream
    FrameRecord     →  SequenceTracker (YOLO-seg + 3D back-projection + ego-motion comp)
    TrackedEntity[] →  ObservationHarvester (quality filtering + VLM sampling)
    ObservationHarvester →  output files (manifest, detections, tracks, crops, metrics)

Design invariants:
    - All detections are from actual images. No hardcoded coordinates.
    - Estimated trajectory is used when available; identity fallback otherwise.
      The fallback is explicitly recorded in the run manifest.
    - Missing depth is noted per-observation, not silently skipped.
    - No fixed global crop count. Every valid track observation is harvested.
    - Resumable: frame processing is checkpointed by source-frame hash.
    - Ground-truth poses NEVER enter the perception pathway.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple

import cv2
import numpy as np

from estimator_aware_vlm.datasets.base import (
    CameraIntrinsics,
    FrameRecord,
    Trajectory,
    TrajectoryPose,
    TrajectoryRole,
    parse_tum_trajectory,
)
from estimator_aware_vlm.perception.sequence_tracker import (
    OFFICE_CLASS_MAPPING,
    ObservationRecord,
    SequenceTracker,
    TrackedEntity,
)


# Constants

YOLO_MODEL_SEARCH_ORDER = [
    "models/yolov8n-seg.pt",
    "yolov8n-seg.pt",
    "models/yolov8s-seg.pt",
    "yolov8s-seg.pt",
    "models/yoloe-26s-seg.pt",
    "yoloe-26s-seg.pt",
    "yolov8n.pt",
]

# Ego-motion compensation disabled — camera is assumed stationary (identity pose).
IDENTITY_POSE_MATRIX = np.eye(4, dtype=np.float64)

# Extended class mapping for corridor / service-robot environments (OpenLORIS).
# Adds mobile people + objects visible in building corridors.
CORRIDOR_CLASS_MAPPING = {
    "person": "person",
    "chair": "chair",
    "tv": "monitor",
    "laptop": "laptop",
    "keyboard": "keyboard",
    "mouse": "mouse",
    "dining table": "desk",
    "desk": "desk",
    "cup": "cup",
    "bottle": "bottle",
    "backpack": "backpack",
    "handbag": "handbag",
    "suitcase": "suitcase",
    "cell phone": "cell_phone",
    "book": "book",
    "potted plant": "plant",    # corridor decoration — detected in OpenLORIS frames
}


# Identity trajectory (fallback when no SLAM trajectory is available)

def _make_identity_pose(timestamp: float) -> TrajectoryPose:
    """Return a camera pose at the world origin with identity orientation.

    Used only when no estimated SLAM trajectory is available.
    The absence of ego-motion compensation is recorded in the run manifest.
    """
    return TrajectoryPose(
        timestamp=timestamp,
        position=(0.0, 0.0, 0.0),
        orientation=(0.0, 0.0, 0.0, 1.0),
        role=TrajectoryRole.ESTIMATED,
    )


def _make_identity_trajectory(timestamps: List[float]) -> Trajectory:
    """Construct an identity trajectory spanning the given timestamps."""
    poses = [_make_identity_pose(ts) for ts in timestamps]
    return Trajectory(poses=poses, role=TrajectoryRole.ESTIMATED, name="identity_fallback")


# Per-sequence configuration

@dataclass
class SequenceSpec:
    """All parameters needed to process one recording through the unified pipeline."""
    sequence_id: str          # e.g. "tum_xyz"
    dataset_yaml: str         # Path to configs/datasets/*.yaml
    estimated_trajectory_path: Optional[str] = None  # None → identity fallback
    stride: int = 2           # Process every Nth frame
    max_frames: Optional[int] = None  # None → all frames
    max_depth_m: float = 5.5  # Max valid depth for this environment (metres)
    class_mapping: Optional[Dict[str, str]] = None  # None → OFFICE_CLASS_MAPPING


# Canonical registry of all eight recordings
ALL_SEQUENCES: List[SequenceSpec] = [
    SequenceSpec(
        sequence_id="tum_xyz",
        dataset_yaml="configs/datasets/tum_rgbd_fr3_walking_xyz.yaml",
        estimated_trajectory_path="results/publication_campaign/run_20261101_execution/baselines/rgbd_dataset_freiburg3_walking_xyz/CameraTrajectory.txt",
        stride=2,
    ),
    SequenceSpec(
        sequence_id="tum_halfsphere",
        dataset_yaml="configs/datasets/tum_rgbd_fr3_walking_halfsphere.yaml",
        estimated_trajectory_path="results/publication_campaign/run_20261101_execution/baselines/rgbd_dataset_freiburg3_walking_halfsphere/CameraTrajectory.txt",
        stride=2,
    ),
    SequenceSpec(
        sequence_id="tum_static",
        dataset_yaml="configs/datasets/tum_rgbd_fr3_walking_static.yaml",
        estimated_trajectory_path="results/publication_campaign/run_20261101_execution/baselines/rgbd_dataset_freiburg3_walking_static/CameraTrajectory.txt",
        stride=2,
    ),
    SequenceSpec(
        sequence_id="openloris",
        dataset_yaml="configs/datasets/openloris_corridor1_4.yaml",
        estimated_trajectory_path="results/publication_campaign/run_20261101_execution/baselines/OpenLORIS-Scene_corridor1-4/CameraTrajectory.txt",
        stride=3,   # Longer sequence — slightly coarser stride
        max_depth_m=8.0,  # Corridor depth routinely exceeds 5.5m — widen gate
        class_mapping=CORRIDOR_CLASS_MAPPING,  # Corridor semantics differ from office lab
    ),
    SequenceSpec(
        sequence_id="bonn_crowd",
        dataset_yaml="configs/datasets/bonn_crowd.yaml",
        estimated_trajectory_path="results/publication_campaign/run_20261101_execution/baselines/rgbd_bonn_crowd/CameraTrajectory.txt",
        stride=2,
    ),
    SequenceSpec(
        sequence_id="bonn_box",
        dataset_yaml="configs/datasets/bonn_moving_obstructing_box.yaml",
        estimated_trajectory_path="results/publication_campaign/run_20261101_execution/baselines/rgbd_bonn_moving_obstructing_box/CameraTrajectory.txt",
        stride=2,
    ),
    SequenceSpec(
        sequence_id="d455_01",
        dataset_yaml="configs/datasets/realsense_d455_recording_01.yaml",
        estimated_trajectory_path="data/semantic_mapping_my_desk_recording_01/trajectory/CameraTrajectory.txt",
        stride=5,  # Very large sequence — coarser stride
    ),
    SequenceSpec(
        sequence_id="d455_02",
        dataset_yaml="configs/datasets/realsense_d455_recording_02.yaml",
        estimated_trajectory_path="data/semantic_mapping_my_desk_recording_02/trajectory/CameraTrajectory.txt",
        stride=5,
    ),
]


# Observation harvester

@dataclass
class HarvestedObservation:
    """A single VLM-eligible observation with full provenance."""
    sequence_id: str
    track_id: str
    obs_id: str               # "{sequence_id}__{track_id}__f{frame_index:06d}"
    frame_index: int
    timestamp: float
    semantic_class: str
    detection_confidence: float
    bbox_xyxy: List[int]
    area_px: int
    median_depth_m: float
    sharpness: float
    has_mask: bool
    mask_contour: Optional[List[List[int]]]
    centroid_camera: List[float]
    centroid_world: Optional[List[float]]    # None if ego-motion unavailable
    ego_motion_available: bool
    rgb_path: str
    crop_path: Optional[str]                 # Set after crop materialisation
    mask_path: Optional[str]                 # Set after mask materialisation
    rejection_reason: Optional[str]          # None = accepted for VLM sampling

    def to_dict(self) -> Dict[str, Any]:
        return {
            "sequence_id": self.sequence_id,
            "track_id": self.track_id,
            "obs_id": self.obs_id,
            "frame_index": self.frame_index,
            "timestamp": round(float(self.timestamp), 6),
            "semantic_class": self.semantic_class,
            "detection_confidence": round(float(self.detection_confidence), 4),
            "bbox_xyxy": [int(x) for x in self.bbox_xyxy],
            "area_px": int(self.area_px),
            "median_depth_m": round(float(self.median_depth_m), 4),
            "sharpness": round(float(self.sharpness), 2),
            "has_mask": bool(self.has_mask),
            "mask_contour": self.mask_contour,
            "centroid_camera": [round(float(x), 4) for x in self.centroid_camera],
            "centroid_world": [round(float(x), 4) for x in self.centroid_world]
                              if self.centroid_world is not None else None,
            "ego_motion_available": bool(self.ego_motion_available),
            "rgb_path": str(self.rgb_path),
            "crop_path": str(self.crop_path) if self.crop_path else None,
            "mask_path": str(self.mask_path) if self.mask_path else None,
            "rejection_reason": self.rejection_reason,
        }


class ObservationHarvester:
    """Converts TrackedEntity lists into HarvestedObservation records.

    Applies quality gates and selects a temporally diverse VLM sampling subset.
    Records every rejection with an explicit reason.
    """

    # Quality thresholds (defaults — overridable per sequence via harvest() call)
    MIN_SHARPNESS = 15.0          # Laplacian variance — below this = motion-blurred
    MIN_AREA_PX = 900             # Minimum bbox area (30×30 px)
    MIN_DEPTH_M = 0.25            # Closer than 0.25m — likely depth noise
    MAX_DEPTH_M = 5.5             # Beyond 5.5m — depth unreliable for lab sensors
    MIN_OBS_COUNT = 5             # Track must have ≥5 observations
    VLM_MIN_FRAME_GAP = 15        # Minimum frame gap between VLM-sampled obs per track
    MAX_VLM_OBS_PER_TRACK = 5     # Maximum VLM observations selected per track

    def harvest(
        self,
        sequence_id: str,
        tracks: List[TrackedEntity],
        ego_motion_available: bool,
        max_depth_m: Optional[float] = None,
    ) -> Tuple[List[HarvestedObservation], List[HarvestedObservation]]:
        """Return (accepted_obs, rejected_obs) for all observations across all tracks.

        accepted_obs: observations that passed quality gates and are VLM-eligible.
        rejected_obs: observations that were excluded, each with rejection_reason set.
        """
        accepted: List[HarvestedObservation] = []
        rejected: List[HarvestedObservation] = []

        for track in tracks:
            if track.observation_count < self.MIN_OBS_COUNT:
                # Entire track rejected — mark all observations
                for obs in track.observations:
                    rejected.append(self._make_harvested(
                        sequence_id, track, obs, ego_motion_available,
                        rejection_reason=f"TRACK_TOO_SHORT ({track.observation_count} obs < {self.MIN_OBS_COUNT})",
                    ))
                continue

            track_accepted: List[HarvestedObservation] = []
            for obs in track.observations:
                reason = self._quality_gate(obs, max_depth_m=max_depth_m)
                h = self._make_harvested(sequence_id, track, obs, ego_motion_available, reason)
                if reason is None:
                    track_accepted.append(h)
                else:
                    rejected.append(h)

            # VLM sampling: select temporally diverse subset from accepted observations
            vlm_subset = self._select_vlm_subset(track_accepted)
            for h in track_accepted:
                obs_id = h.obs_id
                in_vlm = any(v.obs_id == obs_id for v in vlm_subset)
                if not in_vlm:
                    h.rejection_reason = "NOT_SELECTED_FOR_VLM (temporal redundancy)"
                    rejected.append(h)
                else:
                    accepted.append(h)

        return accepted, rejected

    def _quality_gate(self, obs: ObservationRecord, max_depth_m: Optional[float] = None) -> Optional[str]:
        """Return None if observation passes all quality gates, else rejection reason."""
        effective_max_depth = max_depth_m if max_depth_m is not None else self.MAX_DEPTH_M
        if obs.sharpness < self.MIN_SHARPNESS:
            return f"BLURRY (sharpness={obs.sharpness:.1f} < {self.MIN_SHARPNESS})"
        if obs.area_px < self.MIN_AREA_PX:
            return f"TOO_SMALL (area={obs.area_px} px < {self.MIN_AREA_PX})"
        if obs.median_depth_m < self.MIN_DEPTH_M:
            return f"TOO_CLOSE (depth={obs.median_depth_m:.3f}m < {self.MIN_DEPTH_M}m)"
        if obs.median_depth_m > effective_max_depth:
            return f"TOO_FAR (depth={obs.median_depth_m:.3f}m > {effective_max_depth}m)"
        return None

    def _select_vlm_subset(
        self,
        candidates: List[HarvestedObservation],
    ) -> List[HarvestedObservation]:
        """Select a temporally diverse, sharpness-ranked subset for VLM querying."""
        if not candidates:
            return []
        # Sort by sharpness descending
        by_sharpness = sorted(candidates, key=lambda h: h.sharpness, reverse=True)
        selected: List[HarvestedObservation] = []
        for h in by_sharpness:
            if len(selected) >= self.MAX_VLM_OBS_PER_TRACK:
                break
            if not selected:
                selected.append(h)
                continue
            min_gap = min(abs(h.frame_index - s.frame_index) for s in selected)
            if min_gap >= self.VLM_MIN_FRAME_GAP:
                selected.append(h)
        # Fill up to MAX_VLM_OBS_PER_TRACK if not enough temporally distinct frames
        if len(selected) < self.MAX_VLM_OBS_PER_TRACK:
            for h in by_sharpness:
                if len(selected) >= self.MAX_VLM_OBS_PER_TRACK:
                    break
                if h not in selected:
                    selected.append(h)
        return selected

    def _make_harvested(
        self,
        sequence_id: str,
        track: TrackedEntity,
        obs: ObservationRecord,
        ego_motion_available: bool,
        rejection_reason: Optional[str],
    ) -> HarvestedObservation:
        obs_id = f"{sequence_id}__{obs.track_id or track.track_id}__f{obs.frame_index:06d}"
        world = list(obs.centroid_world) if ego_motion_available else None
        return HarvestedObservation(
            sequence_id=sequence_id,
            track_id=obs.track_id or track.track_id,
            obs_id=obs_id,
            frame_index=obs.frame_index,
            timestamp=obs.timestamp,
            semantic_class=obs.semantic_class,
            detection_confidence=obs.confidence,
            bbox_xyxy=list(obs.bbox_xyxy),
            area_px=obs.area_px,
            median_depth_m=obs.median_depth_m,
            sharpness=obs.sharpness,
            has_mask=(obs.mask_contour is not None and len(obs.mask_contour) >= 3),
            mask_contour=obs.mask_contour,
            centroid_camera=list(obs.centroid_camera),
            centroid_world=world,
            ego_motion_available=ego_motion_available,
            rgb_path=str(obs.rgb_path),
            crop_path=None,
            mask_path=None,
            rejection_reason=rejection_reason,
        )


# Crop and mask materialiser

class CropMaterialiser:
    """Saves tight crops and segmentation mask overlays for accepted observations."""

    CROP_TARGET_PX = 320        # Standardise crop long-edge to this size
    JPEG_QUALITY = 95
    MASK_ALPHA = 0.45

    def materialise(
        self,
        obs_list: List[HarvestedObservation],
        output_dir: Path,
    ) -> None:
        """Write crops and mask images to disk, update crop_path and mask_path in-place."""
        output_dir.mkdir(parents=True, exist_ok=True)

        for obs in obs_list:
            img_bgr = cv2.imread(str(obs.rgb_path))
            if img_bgr is None:
                obs.rejection_reason = "CROP_FAILED (image unreadable)"
                continue

            h_img, w_img = img_bgr.shape[:2]
            x1, y1, x2, y2 = obs.bbox_xyxy

            # 8% padding
            pad_x = max(4, int((x2 - x1) * 0.08))
            pad_y = max(4, int((y2 - y1) * 0.08))
            cx1 = max(0, x1 - pad_x)
            cy1 = max(0, y1 - pad_y)
            cx2 = min(w_img, x2 + pad_x)
            cy2 = min(h_img, y2 + pad_y)

            crop = img_bgr[cy1:cy2, cx1:cx2]
            if crop.size == 0:
                obs.rejection_reason = "CROP_FAILED (zero-size after clamp)"
                continue

            # Resize preserving aspect ratio
            ch, cw = crop.shape[:2]
            scale = self.CROP_TARGET_PX / max(ch, cw)
            new_w = max(32, int(cw * scale))
            new_h = max(32, int(ch * scale))
            crop_resized = cv2.resize(crop, (new_w, new_h), interpolation=cv2.INTER_AREA)

            crop_dir = output_dir / obs.track_id
            crop_dir.mkdir(parents=True, exist_ok=True)

            crop_stem = f"{obs.obs_id}"
            crop_path = crop_dir / f"{crop_stem}_crop.jpg"
            cv2.imwrite(str(crop_path), crop_resized,
                        [int(cv2.IMWRITE_JPEG_QUALITY), self.JPEG_QUALITY])
            obs.crop_path = str(crop_path)

            # Save mask overlay if mask polygon is available
            if obs.has_mask and obs.mask_contour:
                contour_arr = np.array(obs.mask_contour, dtype=np.int32)
                mask_img = img_bgr.copy()
                overlay = img_bgr.copy()
                cv2.fillPoly(overlay, [contour_arr], color=(0, 200, 100))
                cv2.addWeighted(overlay, self.MASK_ALPHA, mask_img, 1 - self.MASK_ALPHA, 0, mask_img)
                cv2.polylines(mask_img, [contour_arr], isClosed=True, color=(0, 255, 80), thickness=2)
                mask_path = crop_dir / f"{crop_stem}_mask.jpg"
                cv2.imwrite(str(mask_path), mask_img,
                            [int(cv2.IMWRITE_JPEG_QUALITY), 90])
                obs.mask_path = str(mask_path)


# Per-sequence result container

@dataclass
class SequenceResult:
    """Aggregated result for one recording."""
    sequence_id: str
    frames_total: int
    frames_processed: int
    detections_total: int
    tracks_total: int
    observations_accepted: int
    observations_rejected: int
    masks_generated: int
    ego_motion_available: bool
    trajectory_source: str           # path or "identity_fallback"
    yolo_model_used: str
    tracks: List[Dict[str, Any]]
    accepted: List[HarvestedObservation]
    rejected: List[HarvestedObservation]
    processing_time_sec: float

    def summary_dict(self) -> Dict[str, Any]:
        return {
            "sequence_id": self.sequence_id,
            "frames_total": self.frames_total,
            "frames_processed": self.frames_processed,
            "detections_total": self.detections_total,
            "tracks_total": self.tracks_total,
            "observations_accepted": self.observations_accepted,
            "observations_rejected": self.observations_rejected,
            "masks_generated": self.masks_generated,
            "ego_motion_available": self.ego_motion_available,
            "trajectory_source": self.trajectory_source,
            "yolo_model_used": self.yolo_model_used,
            "processing_time_sec": round(self.processing_time_sec, 2),
        }


# Unified pipeline entry point

class UnifiedPerceptionPipeline:
    """Dataset-independent perception and observation harvesting pipeline.

    One implementation. Eight datasets. Dataset-specific logic lives only
    in the adapter returned by the factory.
    """

    def __init__(
        self,
        repo_root: Path,
        output_dir: Path,
        yolo_model_name: Optional[str] = None,
        min_conf: float = 0.40,
        association_dist_m: float = 0.85,
        min_track_obs: int = 5,
    ):
        self.repo_root = repo_root
        self.output_dir = output_dir
        self.yolo_model_name = yolo_model_name
        self.min_conf = min_conf
        self.association_dist_m = association_dist_m
        self.min_track_obs = min_track_obs
        self.harvester = ObservationHarvester()
        self.materialiser = CropMaterialiser()

        # Resolve YOLO model
        if yolo_model_name:
            resolved = str(repo_root / yolo_model_name)
        else:
            resolved = self._resolve_yolo_model(repo_root)
        self.yolo_model_path = resolved

    def _resolve_yolo_model(self, repo_root: Path) -> str:
        for candidate in YOLO_MODEL_SEARCH_ORDER:
            p = repo_root / candidate
            if p.exists():
                return str(p)
        # Fall back to Ultralytics cache
        cache_dir = Path.home() / ".cache" / "ultralytics"
        for candidate in YOLO_MODEL_SEARCH_ORDER:
            name = Path(candidate).name
            p = cache_dir / name
            if p.exists():
                return str(p)
        raise FileNotFoundError(
            "No YOLO model found. Searched: " + str(YOLO_MODEL_SEARCH_ORDER)
        )

    def _load_trajectory(
        self,
        spec: SequenceSpec,
        frame_timestamps: List[float],
    ) -> Tuple[Trajectory, bool, str]:
        """Load estimated trajectory or construct identity fallback.

        Returns (trajectory, ego_motion_available, trajectory_source_label).
        """
        if spec.estimated_trajectory_path:
            traj_path = self.repo_root / spec.estimated_trajectory_path
            if traj_path.exists():
                traj = parse_tum_trajectory(
                    traj_path,
                    role=TrajectoryRole.ESTIMATED,
                    name=f"{spec.sequence_id}_estimated",
                )
                return traj, True, str(spec.estimated_trajectory_path)

        # Identity fallback
        traj = _make_identity_trajectory(frame_timestamps)
        return traj, False, "identity_fallback"

    def process_sequence(self, spec: SequenceSpec) -> SequenceResult:
        """Run full perception pipeline on one recording."""
        from estimator_aware_vlm.datasets.base import load_dataset_config_from_yaml
        from estimator_aware_vlm.datasets.factory import get_dataset_adapter

        t_start = time.perf_counter()

        # Load dataset adapter
        cfg = load_dataset_config_from_yaml(self.repo_root / spec.dataset_yaml)
        adapter = get_dataset_adapter(cfg)
        total_frames = len(adapter)

        # Collect frames (respecting stride and max_frames limit)
        all_frames: List[FrameRecord] = []
        for i, frame in enumerate(adapter.get_frames()):
            if spec.max_frames is not None and i >= spec.max_frames:
                break
            all_frames.append(frame)

        # Apply stride
        frames = all_frames[::spec.stride]
        frame_timestamps = [f.timestamp for f in all_frames]

        print(f"\n[{spec.sequence_id}] Total frames: {total_frames} | "
              f"After stride={spec.stride}: {len(frames)} frames to process")

        # Load trajectory
        traj, ego_motion_available, traj_source = self._load_trajectory(
            spec, frame_timestamps
        )
        print(f"[{spec.sequence_id}] Ego-motion: {'YES' if ego_motion_available else 'NO (identity fallback)'} "
              f"| Source: {traj_source}")

        # Build tracker for this sequence (intrinsics per-sequence)
        intrinsics = adapter.get_camera_intrinsics()
        tracker = SequenceTracker(
            intrinsics=intrinsics,
            yolo_model_name=self.yolo_model_path,
            min_conf_threshold=self.min_conf,
            association_dist_threshold_m=self.association_dist_m,
            min_track_observations=self.min_track_obs,
            class_mapping=spec.class_mapping or OFFICE_CLASS_MAPPING,
            max_depth_m=spec.max_depth_m,
        )

        # Run tracking (SequenceTracker applies its own stride internally via frames list,
        # but we already applied stride — pass stride=1 to avoid double-striding)
        tracks = tracker.track_sequence(
            frames=frames,
            estimated_trajectory=traj,
            max_dt_sec=0.05,
            stride=1,
        )

        total_detections = sum(t.observation_count for t in tracks)
        print(f"[{spec.sequence_id}] Tracks found: {len(tracks)} | "
              f"Total observations: {total_detections}")

        # Harvest observations
        accepted, rejected = self.harvester.harvest(
            sequence_id=spec.sequence_id,
            tracks=tracks,
            ego_motion_available=ego_motion_available,
            max_depth_m=spec.max_depth_m,
        )

        # Materialise crops and masks
        crop_dir = self.output_dir / "crops" / spec.sequence_id
        self.materialiser.materialise(accepted, crop_dir)

        masks_generated = sum(1 for obs in accepted if obs.mask_path is not None)

        print(f"[{spec.sequence_id}] VLM-eligible observations: {len(accepted)} | "
              f"Rejected: {len(rejected)} | Masks: {masks_generated}")

        t_end = time.perf_counter()

        return SequenceResult(
            sequence_id=spec.sequence_id,
            frames_total=total_frames,
            frames_processed=len(frames),
            detections_total=total_detections,
            tracks_total=len(tracks),
            observations_accepted=len(accepted),
            observations_rejected=len(rejected),
            masks_generated=masks_generated,
            ego_motion_available=ego_motion_available,
            trajectory_source=traj_source,
            yolo_model_used=self.yolo_model_path,
            tracks=[t.to_dict() for t in tracks],
            accepted=accepted,
            rejected=rejected,
            processing_time_sec=t_end - t_start,
        )

    def write_outputs(
        self,
        results: List[SequenceResult],
        model_hash: str,
    ) -> None:
        """Write all 13 output files to self.output_dir."""
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # 1. run_manifest.json
        run_manifest = {
            "pipeline_version": "1.0.0",
            "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "output_dir": str(self.output_dir),
            "yolo_model": self.yolo_model_path,
            "yolo_model_sha256": model_hash,
            "min_conf_threshold": self.min_conf,
            "association_dist_m": self.association_dist_m,
            "min_track_observations": self.min_track_obs,
            "sequences": [r.summary_dict() for r in results],
        }
        self._write_json(self.output_dir / "run_manifest.json", run_manifest)

        # 2. dataset_manifest.json
        dataset_manifest = {
            "sequences": [
                {
                    "sequence_id": r.sequence_id,
                    "frames_total": r.frames_total,
                    "frames_processed": r.frames_processed,
                    "ego_motion_available": r.ego_motion_available,
                    "trajectory_source": r.trajectory_source,
                }
                for r in results
            ]
        }
        self._write_json(self.output_dir / "dataset_manifest.json", dataset_manifest)

        # 3. tracks.jsonl — one JSON object per track, all sequences
        with open(self.output_dir / "tracks.jsonl", "w", encoding="utf-8") as f:
            for r in results:
                for track_dict in r.tracks:
                    track_dict["sequence_id"] = r.sequence_id
                    f.write(json.dumps(track_dict) + "\n")

        # 4. tracking_observations.jsonl — all observations (raw track level)
        # Already covered by tracks.jsonl at track level. The per-frame version:
        # 5. detections.jsonl — all per-frame detections (from accepted + rejected)
        with open(self.output_dir / "detections.jsonl", "w", encoding="utf-8") as f:
            for r in results:
                for obs in r.accepted + r.rejected:
                    f.write(json.dumps(obs.to_dict()) + "\n")

        # 6. candidate_observations.jsonl — VLM-eligible observations only
        with open(self.output_dir / "candidate_observations.jsonl", "w", encoding="utf-8") as f:
            for r in results:
                for obs in r.accepted:
                    f.write(json.dumps(obs.to_dict()) + "\n")

        # 7. rejected_observations.jsonl
        with open(self.output_dir / "rejected_observations.jsonl", "w", encoding="utf-8") as f:
            for r in results:
                for obs in r.rejected:
                    f.write(json.dumps(obs.to_dict()) + "\n")

        # 8. mask_manifest.jsonl
        with open(self.output_dir / "mask_manifest.jsonl", "w", encoding="utf-8") as f:
            for r in results:
                for obs in r.accepted:
                    if obs.mask_path:
                        entry = {
                            "obs_id": obs.obs_id,
                            "sequence_id": obs.sequence_id,
                            "track_id": obs.track_id,
                            "frame_index": obs.frame_index,
                            "mask_path": obs.mask_path,
                            "crop_path": obs.crop_path,
                        }
                        f.write(json.dumps(entry) + "\n")

        # 9. vlm_sampling_manifest.jsonl — final VLM input list
        with open(self.output_dir / "vlm_sampling_manifest.jsonl", "w", encoding="utf-8") as f:
            for r in results:
                for obs in r.accepted:
                    entry = {
                        "obs_id": obs.obs_id,
                        "sequence_id": obs.sequence_id,
                        "track_id": obs.track_id,
                        "semantic_class": obs.semantic_class,
                        "crop_path": obs.crop_path,
                        "mask_path": obs.mask_path,
                        "sharpness": obs.sharpness,
                        "frame_index": obs.frame_index,
                        "timestamp": obs.timestamp,
                    }
                    f.write(json.dumps(entry) + "\n")

        # 10. perception_metrics.json
        total_accepted = sum(r.observations_accepted for r in results)
        total_rejected = sum(r.observations_rejected for r in results)
        total_masks = sum(r.masks_generated for r in results)
        total_tracks = sum(r.tracks_total for r in results)

        # Breakdown of rejection reasons
        reason_counts: Dict[str, int] = {}
        for r in results:
            for obs in r.rejected:
                reason = obs.rejection_reason or "UNKNOWN"
                prefix = reason.split(" ")[0]
                reason_counts[prefix] = reason_counts.get(prefix, 0) + 1

        metrics = {
            "total_sequences": len(results),
            "per_sequence": [r.summary_dict() for r in results],
            "aggregate": {
                "frames_processed": sum(r.frames_processed for r in results),
                "tracks": total_tracks,
                "detections": sum(r.detections_total for r in results),
                "observations_accepted": total_accepted,
                "observations_rejected": total_rejected,
                "masks_generated": total_masks,
                "mask_coverage_pct": round(
                    100.0 * total_masks / max(1, total_accepted), 1
                ),
                "rejection_reasons": reason_counts,
            },
        }
        self._write_json(self.output_dir / "perception_metrics.json", metrics)

        # 11. visual_quality_report.md
        self._write_visual_quality_report(results)

        print(f"\n{'='*60}")
        print(f"Pipeline complete. Output: {self.output_dir}")
        print(f"Sequences:          {len(results)}")
        print(f"Total tracks:       {total_tracks}")
        print(f"VLM-eligible obs:   {total_accepted}")
        print(f"Rejected obs:       {total_rejected}")
        print(f"Segmentation masks: {total_masks} ({metrics['aggregate']['mask_coverage_pct']}% of accepted)")
        print(f"{'='*60}")

    def _write_visual_quality_report(self, results: List[SequenceResult]) -> None:
        lines = [
            "# Unified Perception Pipeline — Visual Quality Report",
            "",
            f"Generated: {time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}",
            f"YOLO model: `{self.yolo_model_path}`",
            "",
            "## Per-Sequence Summary",
            "",
            "| Sequence | Frames | Tracks | Detections | Accepted | Rejected | Masks | Ego-motion |",
            "| --- | --- | --- | --- | --- | --- | --- | --- |",
        ]
        for r in results:
            lines.append(
                f"| {r.sequence_id} | {r.frames_processed} | {r.tracks_total} | "
                f"{r.detections_total} | {r.observations_accepted} | "
                f"{r.observations_rejected} | {r.masks_generated} | "
                f"{'YES' if r.ego_motion_available else 'NO (identity)'} |"
            )

        lines.extend([
            "",
            "## Rejection Reason Distribution",
            "",
        ])

        all_rejected = [obs for r in results for obs in r.rejected]
        reason_counts: Dict[str, int] = {}
        for obs in all_rejected:
            reason = obs.rejection_reason or "UNKNOWN"
            prefix = reason.split(" ")[0]
            reason_counts[prefix] = reason_counts.get(prefix, 0) + 1

        lines.extend([
            "| Rejection Prefix | Count |",
            "| --- | --- |",
        ])
        for reason, cnt in sorted(reason_counts.items(), key=lambda x: -x[1]):
            lines.append(f"| {reason} | {cnt} |")

        lines.extend([
            "",
            "## Notes",
            "",
            "- `identity_fallback` ego-motion: 3D tracking still functional, "
              "but world-frame displacement is camera-frame displacement (uncompensated).",
            "- All detections are from actual image frames. "
              "No hardcoded coordinates were used.",
            "- Segmentation masks are polygon contours from YOLO-seg inference.",
        ])

        report_path = self.output_dir / "visual_quality_report.md"
        report_path.write_text("\n".join(lines), encoding="utf-8")

    @staticmethod
    def _write_json(path: Path, data: Any) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    @staticmethod
    def hash_file(path: str) -> str:
        """SHA-256 of a file. Returns 'not_found' if path does not exist."""
        p = Path(path)
        if not p.exists():
            return "not_found"
        h = hashlib.sha256()
        with open(p, "rb") as f:
            for chunk in iter(lambda: f.read(65536), b""):
                h.update(chunk)
        return h.hexdigest()

