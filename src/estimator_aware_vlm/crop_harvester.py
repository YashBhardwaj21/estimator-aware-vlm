"""Multi-View Observation Crop Harvester.

Filters, selects, and extracts high-quality, viewpoint-diverse RGB image crops
for each tracked physical object to feed single-view and multi-view VLM reasoning.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np


@dataclass
class ViewObservation:
    """A single view observation of a physical object."""
    view_index: int
    frame_index: int
    timestamp: float
    bbox_xyxy: List[float]
    detector_confidence: float
    sharpness_score: float
    viewing_angle_deg: float
    camera_pose: List[float]
    crop_rel_path: str
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "view_index": self.view_index,
            "frame_index": self.frame_index,
            "timestamp": round(self.timestamp, 4),
            "bbox_xyxy": [round(float(c), 1) for c in self.bbox_xyxy],
            "detector_confidence": round(self.detector_confidence, 4),
            "sharpness_score": round(self.sharpness_score, 2),
            "viewing_angle_deg": round(self.viewing_angle_deg, 2),
            "camera_pose": [round(float(p), 4) for p in self.camera_pose],
            "crop_rel_path": self.crop_rel_path,
            "metadata": self.metadata,
        }


def compute_image_sharpness(img_gray: np.ndarray) -> float:
    """Compute image focus/sharpness via Laplacian variance."""
    if img_gray.size == 0:
        return 0.0
    return float(cv2.Laplacian(img_gray, cv2.CV_64F).var())


def compute_viewing_angle(cam_pos: np.ndarray, obj_pos: np.ndarray) -> float:
    """Compute horizontal azimuth angle in degrees from camera to object."""
    vec = obj_pos[:2] - cam_pos[:2]
    norm = np.linalg.norm(vec)
    if norm < 1e-4:
        return 0.0
    angle_rad = math.atan2(vec[1], vec[0])
    return math.degrees(angle_rad)


def angle_difference(a1: float, a2: float) -> float:
    """Compute minimal absolute angular difference in degrees (-180 to 180)."""
    diff = (a1 - a2 + 180) % 360 - 180
    return abs(diff)


class MultiViewCropHarvester:
    """Harvester selecting up to K diverse, high-quality viewpoints per physical object."""

    def __init__(
        self,
        output_dir: Path,
        max_views_per_object: int = 3,
        min_angular_separation_deg: float = 12.0,
        context_padding_pct: float = 0.15,
        min_crop_size: int = 24,
    ):
        self.output_dir = Path(output_dir)
        self.crops_dir = self.output_dir / "crops"
        self.crops_dir.mkdir(parents=True, exist_ok=True)
        self.max_views = max_views_per_object
        self.min_ang_sep = min_angular_separation_deg
        self.padding_pct = context_padding_pct
        self.min_crop_size = min_crop_size

    def select_diverse_candidates(
        self, candidates: List[Dict[str, Any]], obj_pos: Optional[np.ndarray]
    ) -> List[Dict[str, Any]]:
        """Greedily select up to max_views that maximize quality and angular diversity."""
        if not candidates:
            return []

        # Sort candidate observations by a composite score of confidence and sharpness
        # Normalize sharpness to [0, 1] range across candidates
        max_sharp = max([c["sharpness"] for c in candidates]) + 1e-5
        for c in candidates:
            norm_sharp = c["sharpness"] / max_sharp
            c["rank_score"] = 0.5 * c["confidence"] + 0.5 * norm_sharp

        candidates_sorted = sorted(candidates, key=lambda x: x["rank_score"], reverse=True)

        selected = [candidates_sorted[0]]

        for cand in candidates_sorted[1:]:
            if len(selected) >= self.max_views:
                break
            # Check angular diversity against all currently selected
            cand_angle = cand["viewing_angle"]
            min_sep = min([angle_difference(cand_angle, s["viewing_angle"]) for s in selected])
            
            # Accept if separated by at least min_ang_sep or if candidates are scarce
            if min_sep >= self.min_ang_sep:
                selected.append(cand)

        # If strict angular diversity didn't find enough views, fill up with next best distinct frames
        if len(selected) < self.max_views:
            selected_frames = set(s["frame_index"] for s in selected)
            for cand in candidates_sorted:
                if len(selected) >= self.max_views:
                    break
                if cand["frame_index"] not in selected_frames:
                    # Ensure frames are separated by at least 10 frames
                    frame_dist = min([abs(cand["frame_index"] - s["frame_index"]) for s in selected])
                    if frame_dist >= 10:
                        selected.append(cand)
                        selected_frames.add(cand["frame_index"])

        return selected

    def harvest_for_sequence(
        self,
        seq_name: str,
        seq_raw_dir: Path,
        mapping_dir: Path,
        ground_truth_objects: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Harvest multi-view crops for all ground truth objects in a sequence."""
        # 1. Load trajectory for camera poses
        traj_path = mapping_dir / "trajectory" / "CameraTrajectory.txt"
        trajectory_poses = {}
        if traj_path.exists():
            with open(traj_path) as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 8:
                        t = float(parts[0])
                        pose = [float(x) for x in parts[1:8]]  # tx, ty, tz, qx, qy, qz, qw
                        trajectory_poses[round(t, 4)] = pose

        # 2. Load timestamp associations
        assoc_path = mapping_dir / "associations.txt"
        frame_rgb_paths = {}
        if assoc_path.exists():
            with open(assoc_path) as f:
                for line in f:
                    if line.startswith("#"):
                        continue
                    parts = line.strip().split()
                    if len(parts) >= 2:
                        t = float(parts[0])
                        frame_rgb_paths[round(t, 4)] = parts[1]

        # 3. Load all detections
        det_path = mapping_dir / "detections.jsonl"
        detections_by_track = {}
        if det_path.exists():
            with open(det_path) as f:
                for line in f:
                    det = json.loads(line)
                    trk_id = det.get("local_track_id")
                    g_id = det.get("global_entity_id")
                    if g_id:
                        detections_by_track.setdefault(g_id, []).append(det)

        results = []

        for obj in ground_truth_objects:
            obj_id = obj["object_id"]
            associated_tracks = obj["associated_track_ids"]
            obj_centroid = np.array(obj["centroid_floor_m"]) if obj.get("centroid_floor_m") else None

            # Collect all candidate detections across associated tracks
            candidates = []
            for trk in associated_tracks:
                for det in detections_by_track.get(trk, []):
                    t = round(det["timestamp"], 4)
                    rgb_rel = frame_rgb_paths.get(t)
                    if not rgb_rel:
                        continue
                    full_rgb_path = seq_raw_dir / rgb_rel
                    if not full_rgb_path.exists():
                        continue

                    box = det["bbox_xyxy"]
                    w = box[2] - box[0]
                    h = box[3] - box[1]
                    if w < self.min_crop_size or h < self.min_crop_size:
                        continue

                    # Camera pose and viewing angle
                    pose = trajectory_poses.get(t, [0.0]*7)
                    cam_pos = np.array(pose[:3])
                    v_angle = compute_viewing_angle(cam_pos, obj_centroid) if obj_centroid is not None else 0.0

                    candidates.append({
                        "det": det,
                        "frame_index": det["frame_index"],
                        "timestamp": t,
                        "confidence": det["confidence"],
                        "bbox_xyxy": box,
                        "rgb_path": full_rgb_path,
                        "camera_pose": pose,
                        "viewing_angle": v_angle,
                        "sharpness": 0.0, # Will measure on candidate pool
                    })

            if not candidates:
                results.append({
                    "object_id": obj_id,
                    "class_name": obj["class_name"],
                    "contextual_mobility_gt": obj["contextual_mobility_gt"],
                    "views_collected": 0,
                    "views": []
                })
                continue

            # Subsample candidate pool to at most 30 to compute sharpness quickly
            if len(candidates) > 30:
                step = len(candidates) // 30
                candidates_subset = candidates[::step]
            else:
                candidates_subset = candidates

            # Measure sharpness of candidates
            valid_candidates = []
            for c in candidates_subset:
                img = cv2.imread(str(c["rgb_path"]))
                if img is None:
                    continue
                h_img, w_img = img.shape[:2]
                bx = c["bbox_xyxy"]
                x1 = max(0, int(bx[0]))
                y1 = max(0, int(bx[1]))
                x2 = min(w_img, int(bx[2]))
                y2 = min(h_img, int(bx[3]))
                crop = img[y1:y2, x1:x2]
                if crop.size == 0:
                    continue
                gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
                c["sharpness"] = compute_image_sharpness(gray)
                c["_cached_img"] = img
                valid_candidates.append(c)

            # Select diverse viewpoints
            selected = self.select_diverse_candidates(valid_candidates, obj_centroid)

            # Save the chosen crops
            obj_views = []
            for v_idx, sel in enumerate(selected):
                img = sel["_cached_img"]
                h_img, w_img = img.shape[:2]
                bx = sel["bbox_xyxy"]
                
                # Apply context margin (padding) so VLM sees affordance (wheels, ground, legs, etc.)
                bw = bx[2] - bx[0]
                bh = bx[3] - bx[1]
                pad_x = bw * self.padding_pct
                pad_y = bh * self.padding_pct
                x1 = max(0, int(bx[0] - pad_x))
                y1 = max(0, int(bx[1] - pad_y))
                x2 = min(w_img, int(bx[2] + pad_x))
                y2 = min(h_img, int(bx[3] + pad_y))
                
                crop_patch = img[y1:y2, x1:x2]
                crop_fname = f"{obj_id}_view_{v_idx}.jpg"
                crop_save_path = self.crops_dir / crop_fname
                cv2.imwrite(str(crop_save_path), crop_patch, [cv2.IMWRITE_JPEG_QUALITY, 92])

                view_obs = ViewObservation(
                    view_index=v_idx,
                    frame_index=sel["frame_index"],
                    timestamp=sel["timestamp"],
                    bbox_xyxy=[x1, y1, x2, y2],
                    detector_confidence=sel["confidence"],
                    sharpness_score=sel["sharpness"],
                    viewing_angle_deg=sel["viewing_angle"],
                    camera_pose=sel["camera_pose"],
                    crop_rel_path=f"crops/{crop_fname}",
                    metadata={"source_sequence": seq_name}
                )
                obj_views.append(view_obs.to_dict())

            results.append({
                "object_id": obj_id,
                "class_name": obj["class_name"],
                "contextual_mobility_gt": obj["contextual_mobility_gt"],
                "observed_motion_gt": obj["observed_motion_gt"],
                "views_collected": len(obj_views),
                "views": obj_views
            })

        return results
