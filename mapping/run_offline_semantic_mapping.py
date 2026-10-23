#!/usr/bin/env python3
"""Generalized Offline Semantic Mapping Pipeline.

Accepts arbitrary RGB-D recordings with dynamic frame discovery,
calibrated camera parameter ingestion, timestamp synchronization,
pose alignment validation, unified floor-frame transformation,
dynamic map bounds derivation, persistent entity tracking,
and metric occupancy grid generation.
"""

import argparse
import datetime
import json
import math
import os
import platform
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

# Add src to path
# sys.path.insert(0, os.path.abspath("src"))

from scene_graph.geometry.camera import CameraIntrinsics
from scene_graph.geometry.transforms import transform_points
from scene_graph.geometry.plane import fit_plane_ransac
from scene_graph.geometry.floor_alignment import construct_floor_transform, compute_dynamic_map_bounds
from scene_graph.data.dataset_discovery import discover_recording, RecordingManifest
from scene_graph.pipeline.trajectory_manager import (
    load_tum_trajectory,
    validate_trajectory_alignment,
    find_matching_pose,
    TrajectoryValidationError,
)
from scene_graph.perception.yoloe_detector import YOLOEDetector, CANONICAL_23_CLASSES
from scene_graph.mapping.occupancy_grid import OccupancyGrid, OccupancyGridConfig
from scene_graph.mapping.global_associator import (
    GlobalObjectAssociator,
    ObjectObservationRecord,
)


class LocalIoUTracker:
    """Frame-to-frame bounding box IoU tracker for assigning sequential local_track_id."""

    def __init__(self, iou_thresh: float = 0.25, max_age: int = 4):
        self.iou_thresh = iou_thresh
        self.max_age = max_age
        self.next_id = 1
        self.active_tracks: Dict[int, dict] = {}

    @staticmethod
    def _box_iou(box1: np.ndarray, box2: np.ndarray) -> float:
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        inter_area = max(0.0, x2 - x1) * max(0.0, y2 - y1)
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        denom = area1 + area2 - inter_area
        return inter_area / denom if denom > 0 else 0.0

    def update(self, frame_idx: int, detections: list) -> List[str]:
        assigned_ids = [""] * len(detections)
        if not detections:
            self.active_tracks = {
                tid: trk for tid, trk in self.active_tracks.items()
                if frame_idx - trk["last_frame"] <= self.max_age
            }
            return assigned_ids

        matched_tracks = set()
        matched_dets = set()

        for d_idx, det in enumerate(detections):
            best_iou = self.iou_thresh
            best_tid = None
            d_box = np.array(det["bbox_xyxy"])
            d_cls = det["class_name"]

            for tid, trk in self.active_tracks.items():
                if tid in matched_tracks:
                    continue
                if trk["class_name"] != d_cls:
                    continue
                iou = self._box_iou(d_box, trk["bbox"])
                if iou >= best_iou:
                    best_iou = iou
                    best_tid = tid

            if best_tid is not None:
                assigned_ids[d_idx] = f"trk_{best_tid:04d}"
                self.active_tracks[best_tid] = {
                    "bbox": d_box,
                    "class_name": d_cls,
                    "last_frame": frame_idx,
                }
                matched_tracks.add(best_tid)
                matched_dets.add(d_idx)

        for d_idx, det in enumerate(detections):
            if d_idx not in matched_dets:
                tid = self.next_id
                self.next_id += 1
                assigned_ids[d_idx] = f"trk_{tid:04d}"
                self.active_tracks[tid] = {
                    "bbox": np.array(det["bbox_xyxy"]),
                    "class_name": det["class_name"],
                    "last_frame": frame_idx,
                }

        self.active_tracks = {
            tid: trk for tid, trk in self.active_tracks.items()
            if frame_idx - trk["last_frame"] <= self.max_age
        }

        return assigned_ids


def sample_frame_indices(
    total_frames: int,
    policy: str = "step",
    step: int = 5,
    target_fps: float = 5.0,
    timestamps: Optional[List[float]] = None,
    max_frames: Optional[int] = None,
) -> List[int]:
    """Select frame indices according to the configured sampling policy."""
    if total_frames <= 0:
        return []

    if policy == "all":
        selected = list(range(total_frames))
    elif policy == "fps" and timestamps and len(timestamps) == total_frames:
        interval = 1.0 / target_fps
        selected = [0]
        last_t = timestamps[0]
        for i in range(1, total_frames):
            if timestamps[i] - last_t >= interval:
                selected.append(i)
                last_t = timestamps[i]
    else:  # Default 'step'
        selected = list(range(0, total_frames, max(1, step)))

    if max_frames and max_frames > 0:
        selected = selected[:max_frames]
    return selected


def run_pipeline(args) -> Dict[str, Any]:
    start_wall_time = time.time()
    dataset_dir = Path(args.dataset_dir).resolve()
    if not dataset_dir.exists():
        raise FileNotFoundError(f"Recording directory does not exist: {dataset_dir}")

    # 1. Output Directory Resolution & Isolation
    if args.output_dir:
        out_dir = Path(args.output_dir).resolve()
    else:
        timestamp_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        out_dir = Path(f"outputs/semantic_mapping_{dataset_dir.name}_{timestamp_str}").resolve()

    if out_dir.exists() and not args.overwrite:
        # Check if directory contains existing run deliverables
        if (out_dir / "semantic_map.json").exists():
            raise FileExistsError(
                f"Output directory {out_dir} already exists. Specify a unique --output-dir "
                f"or pass --overwrite to replace existing deliverables."
            )
    out_dir.mkdir(parents=True, exist_ok=True)
    vis_dir = out_dir / "visualizations"
    vis_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 75)
    print("STARTING GENERALIZED OFFLINE SEMANTIC MAPPING PIPELINE")
    print(f"Recording Directory:  {dataset_dir}")
    print(f"Output Directory:     {out_dir}")
    print(f"Sampling Policy:      {args.sample_policy} (step={args.sample_step}, fps={args.sample_fps})")
    print(f"Map Resolution:       {args.map_resolution} m/cell")
    print(f"Robot Footprint:      radius={args.robot_radius} m, safety_margin={args.safety_margin} m")
    print(f"Pipeline Mode:        {args.mode}")
    print("=" * 75)

    # 2. Dynamic Recording Discovery & Ingestion
    print("\n[Step 1/6] Discovering frames, calibration, and timestamp associations...")
    manifest = discover_recording(
        dataset_dir=dataset_dir,
        intrinsics_file=args.intrinsics_file,
        max_sync_dt=args.sync_max_dt,
        custom_depth_scale=args.depth_scale,
        custom_associations_file=args.associations_file,
    )

    # Apply manual intrinsic CLI overrides if specified
    intr = manifest.intrinsics
    fx = args.fx if args.fx is not None else intr.fx
    fy = args.fy if args.fy is not None else intr.fy
    cx = args.cx if args.cx is not None else intr.cx
    cy = args.cy if args.cy is not None else intr.cy
    w_px = args.width if args.width is not None else intr.width
    h_px = args.height if args.height is not None else intr.height
    intrinsics = CameraIntrinsics(fx=fx, fy=fy, cx=cx, cy=cy, width=w_px, height=h_px)
    depth_scale = args.depth_scale if args.depth_scale is not None else manifest.depth_scale_m_per_unit

    print(f"  Sequence:       {manifest.sequence_name}")
    print(f"  RGB Frames:     {manifest.total_rgb_count}")
    print(f"  Depth Frames:   {manifest.total_depth_count}")
    print(f"  Associated Pairs: {manifest.total_associated_count}")
    print(f"  Duration:       {manifest.duration_s:.1f} s (~{manifest.fps_estimate:.1f} FPS)")
    print(f"  Intrinsics:     fx={intrinsics.fx:.1f}, fy={intrinsics.fy:.1f}, cx={intrinsics.cx:.1f}, cy={intrinsics.cy:.1f} ({intrinsics.width}x{intrinsics.height})")
    print(f"  Depth Scale:    {depth_scale:.6f} m/unit")

    # Save associations in output dir for full reproducibility
    assoc_out_path = out_dir / "associations.txt"
    with open(assoc_out_path, "w", encoding="utf-8") as f:
        f.write("# Generated timestamp associations for sequence " + manifest.sequence_name + "\n")
        f.write("# t_rgb rgb_path t_depth depth_path\n")
        for p in manifest.associated_pairs:
            f.write(f"{p[0]:.6f} {p[1]} {p[2]:.6f} {p[3]}\n")

    # 3. Trajectory Ingestion & Alignment Validation
    print("\n[Step 2/6] Ingesting camera trajectory and validating timestamp alignment...")
    traj_path = None
    if args.trajectory_file:
        traj_path = Path(args.trajectory_file)
    else:
        # Search candidate locations in dataset directory
        candidates = [
            dataset_dir / "CameraTrajectory.txt",
            dataset_dir / "groundtruth.txt",
            dataset_dir / "trajectory.txt",
        ]
        for c in candidates:
            if c.exists():
                traj_path = c
                break

    if traj_path is None or not traj_path.exists():
        raise FileNotFoundError(
            f"No trajectory file specified or discovered for {dataset_dir}. "
            f"Please supply a valid trajectory via --trajectory-file."
        )

    poses = load_tum_trajectory(traj_path)
    pose_keys = np.array(sorted(poses.keys()))
    rgb_timestamps = [p[0] for p in manifest.associated_pairs]

    # Validate provenance and temporal alignment
    traj_val = validate_trajectory_alignment(
        poses=poses,
        rgb_timestamps=rgb_timestamps,
        max_dt=args.pose_tolerance_s,
        min_match_ratio=0.15,
    )
    print(f"  Trajectory source: {traj_path} ({traj_val['trajectory_poses_count']} poses)")
    print(f"  Temporal overlap:  {traj_val['overlap_duration_s']}s ({traj_val['temporal_overlap_pct']}%)")
    print(f"  Match check:       {traj_val['sample_match_ratio_pct']}% of test points aligned")

    # 4. Sampling Selection & Coverage Audit
    selected_indices = sample_frame_indices(
        total_frames=len(manifest.associated_pairs),
        policy=args.sample_policy,
        step=args.sample_step,
        target_fps=args.sample_fps,
        timestamps=rgb_timestamps,
        max_frames=args.max_frames,
    )

    # Check pose availability for selected frames
    posed_selected_indices = []
    for f_idx in selected_indices:
        t_rgb = manifest.associated_pairs[f_idx][0]
        if find_matching_pose(t_rgb, poses, pose_keys, tolerance_s=args.pose_tolerance_s) is not None:
            posed_selected_indices.append(f_idx)

    input_coverage_pct = round(len(selected_indices) / max(1, len(manifest.associated_pairs)) * 100.0, 2)
    pose_coverage_pct = round(len(posed_selected_indices) / max(1, len(selected_indices)) * 100.0, 2)

    print(f"  Selected frames:   {len(selected_indices)} ({input_coverage_pct}% of input sequence)")
    print(f"  Frames with poses: {len(posed_selected_indices)} ({pose_coverage_pct}% pose coverage)")

    if len(posed_selected_indices) < 5:
        raise RuntimeError(f"Insufficient frames with valid camera poses ({len(posed_selected_indices)}) to perform mapping!")

    # 5. Robust Floor Plane Fitting & Unified Transform
    print("\n[Step 3/6] Estimating Floor Plane via RANSAC and Constructing Unified Transform...")
    # Sample points evenly distributed across the trajectory
    num_plane_samples = min(75, len(posed_selected_indices))
    sample_sub_indices = [
        posed_selected_indices[i]
        for i in np.linspace(0, len(posed_selected_indices) - 1, num_plane_samples, dtype=int)
    ]

    world_pts_floor_candidates = []
    for f_idx in sample_sub_indices:
        t_rgb, _, _, depth_rel = manifest.associated_pairs[f_idx]
        T_wc = find_matching_pose(t_rgb, poses, pose_keys, tolerance_s=args.pose_tolerance_s)
        if T_wc is None:
            continue
        depth_img_path = dataset_dir / depth_rel
        depth_raw = cv2.imread(str(depth_img_path), cv2.IMREAD_UNCHANGED)
        if depth_raw is None:
            continue
        depth_m = depth_raw.astype(np.float32) * depth_scale

        step = 14
        h, w = depth_m.shape
        u_grid, v_grid = np.meshgrid(np.arange(0, w, step), np.arange(0, h, step))
        u_f, v_f = u_grid.ravel(), v_grid.ravel()
        z_f = depth_m[v_f, u_f]

        valid = np.isfinite(z_f) & (z_f > 0.3) & (z_f < 3.5)
        pts_cam = intrinsics.pixels_to_camera(u_f[valid], v_f[valid], z_f[valid])
        pts_w = transform_points(T_wc, pts_cam)
        world_pts_floor_candidates.append(pts_w)

    if not world_pts_floor_candidates:
        raise RuntimeError("No valid depth points could be unprojected for floor fitting!")

    accum_world_pts = np.vstack(world_pts_floor_candidates)
    # Heuristic: candidate floor points in lower vertical percentiles
    y_threshold = np.percentile(accum_world_pts[:, 1], 55)
    floor_candidates = accum_world_pts[accum_world_pts[:, 1] > y_threshold]

    plane_fit = fit_plane_ransac(
        floor_candidates,
        distance_threshold=0.035,
        max_iterations=350,
        min_inliers=100,
        rng=np.random.default_rng(42),
    )
    if plane_fit is None:
        raise RuntimeError("RANSAC failed to fit floor plane from depth data!")

    n_floor = plane_fit.normal
    d_floor = plane_fit.distance
    if n_floor[1] > 0:
        n_floor = -n_floor
        d_floor = -d_floor

    inlier_cnt = int(np.sum(plane_fit.inlier_mask))
    inlier_pct = round(inlier_cnt / len(floor_candidates) * 100.0, 2)
    if inlier_pct < 15.0 or plane_fit.residual_mean > 0.08:
        raise RuntimeError(
            f"Floor plane quality gate FAILED: inlier percentage {inlier_pct:.1f}% (< 15%) "
            f"or residual {plane_fit.residual_mean*1000:.1f}mm (> 80mm). "
            f"Floor plane cannot be reliably established."
        )

    # Calculate actual camera height using starting camera center and plane equation: |n . C + d| / ||n||
    first_posed_f = posed_selected_indices[0]
    t_first = manifest.associated_pairs[first_posed_f][0]
    T_wc_0 = find_matching_pose(t_first, poses, pose_keys, tolerance_s=args.pose_tolerance_s)
    if T_wc_0 is not None:
        c_cam_0 = T_wc_0[:3, 3]
        n_unit = n_floor / np.linalg.norm(n_floor)
        cam_start_h = float(abs(np.dot(n_unit, c_cam_0) + d_floor))
    else:
        cam_start_h = abs(float(d_floor))

    # Construct UNIFIED 4x4 T_floor_world transform
    T_floor_world, T_world_floor = construct_floor_transform(n_floor, d_floor)

    floor_plane_info = {
        "frame_id": "floor_map",
        "parent_frame_id": "slam_world",
        "coordinate_convention": "ROS REP-103: X=Forward, Y=Left, Z=Height (Z=0 on floor plane)",
        "upward_normal": [round(float(x), 4) for x in n_floor],
        "plane_distance_d": round(float(d_floor), 4),
        "inlier_count": inlier_cnt,
        "total_candidate_points": len(floor_candidates),
        "inlier_percentage": inlier_pct,
        "residual_mean_m": round(float(plane_fit.residual_mean), 4),
        "residual_std_m": round(float(plane_fit.residual_std), 4),
        "camera_start_height_above_floor_m": round(cam_start_h, 4),
        "T_floor_world": [[round(float(c), 6) for c in row] for row in T_floor_world],
        "T_world_floor": [[round(float(c), 6) for c in row] for row in T_world_floor],
    }
    with open(out_dir / "floor_plane.json", "w", encoding="utf-8") as f:
        json.dump(floor_plane_info, f, indent=2)

    print(f"  Normal (upward):  [{n_floor[0]:.4f}, {n_floor[1]:.4f}, {n_floor[2]:.4f}]")
    print(f"  Residual:         Mean {plane_fit.residual_mean*1000:.1f} mm, Std {plane_fit.residual_std*1000:.1f} mm")
    print(f"  Camera H at start:{cam_start_h:.3f} m")
    print(f"  T_floor_world:    Constructed and saved to floor_plane.json")

    # 6. Dynamic Map Bounds & Occupancy Grid Initialization
    print("\n[Step 4/6] Deriving Dynamic Map Bounds and Initializing Components...")
    # Transform camera trajectory into floor_map frame to calculate exact bounds
    cam_floor_pts = []
    for f_idx in posed_selected_indices:
        t_rgb = manifest.associated_pairs[f_idx][0]
        T_wc = find_matching_pose(t_rgb, poses, pose_keys, tolerance_s=args.pose_tolerance_s)
        if T_wc is not None:
            T_fc = T_floor_world @ T_wc
            cam_floor_pts.append(T_fc[:3, 3])
    cam_floor_pts = np.array(cam_floor_pts)

    if args.map_bounds and args.map_bounds.lower() != "auto":
        parts = [float(p.strip()) for p in args.map_bounds.split(",")]
        if len(parts) != 4:
            raise ValueError("--map-bounds must be 'auto' or 'min_x,min_y,max_x,max_y'")
        origin_x, origin_y = parts[0], parts[1]
        width_m = parts[2] - parts[0]
        height_m = parts[3] - parts[1]
    else:
        # Dynamic derivation from both observed camera positions AND reconstructed geometry
        pts_floor_samples = transform_points(T_floor_world, accum_world_pts[::10])
        combined_geo_pts = np.vstack([cam_floor_pts, pts_floor_samples[:, :3]])
        origin_x, origin_y, width_m, height_m = compute_dynamic_map_bounds(
            camera_positions_floor=combined_geo_pts,
            padding_m=args.map_padding,
            resolution_m=args.map_resolution,
        )

    grid_cfg = OccupancyGridConfig(
        resolution_m=args.map_resolution,
        origin_x_m=origin_x,
        origin_y_m=origin_y,
        width_m=width_m,
        height_m=height_m,
        min_obstacle_height_m=0.08,
        max_obstacle_height_m=1.85,
        floor_band_m=0.06,
        robot_radius_m=args.robot_radius,
        safety_margin_m=args.safety_margin,
        max_ray_range_m=4.0,
    )
    occ_grid = OccupancyGrid(grid_cfg)
    print(f"  Dynamic Grid Bounds: Origin=({origin_x:.2f}, {origin_y:.2f}) m | Dimensions={width_m:.2f} x {height_m:.2f} m ({occ_grid.cols}x{occ_grid.rows} cells)")

    # Detection Vocabulary Setup
    if args.classes.lower() == "all":
        selected_classes = list(CANONICAL_23_CLASSES)
    else:
        selected_classes = [c.strip() for c in args.classes.split(",") if c.strip()]

    detector = YOLOEDetector(
        model_path=args.model_path,
        confidence_threshold=args.confidence_thresh,
        text_prompts=selected_classes,
        device="cpu",
        image_size=480,
    )
    associator = GlobalObjectAssociator(
        association_dist_threshold_m=0.65,
        min_hits_to_confirm=max(3, int(args.min_hits)),
        min_frame_span=max(5, int(args.min_span)),
        max_centroid_variance_m=0.25,
        voxel_size_m=0.015,
    )
    local_tracker = LocalIoUTracker(iou_thresh=0.25, max_age=4)

    # 7. Processing Frames with Full Provenance Tracking
    print("\n[Step 5/6] Processing sequence with strict provenance and floor-frame projection...")
    det_out = open(out_dir / "detections.jsonl", "w", encoding="utf-8")
    obs_out = open(out_dir / "object_observations.jsonl", "w", encoding="utf-8")

    total_raw_detections = 0
    total_valid_obs_count = 0
    total_rejected_depth = 0
    annotated_sample_frames = []
    proc_start_time = time.time()

    for idx, f_idx in enumerate(posed_selected_indices):
        t_rgb, rgb_rel, t_depth, depth_rel = manifest.associated_pairs[f_idx]
        T_wc = find_matching_pose(t_rgb, poses, pose_keys, tolerance_s=args.pose_tolerance_s)
        if T_wc is None:
            continue

        T_floor_c = T_floor_world @ T_wc
        cam_floor_pos = T_floor_c[:3, 3]

        rgb_path = dataset_dir / rgb_rel
        depth_path = dataset_dir / depth_rel
        rgb = cv2.imread(str(rgb_path))
        depth_raw = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED)
        if rgb is None or depth_raw is None:
            continue

        depth_m = depth_raw.astype(np.float32) * depth_scale
        h, w = depth_m.shape

        # A. Detect Objects
        from scene_graph.data.frame_packet import FramePacket
        packet = FramePacket(
            frame_index=f_idx,
            timestamp=t_rgb,
            rgb=rgb,
            depth=depth_m,
            world_T_camera=None,
            camera_intrinsics=intrinsics,
        )
        raw_observations = detector.detect(packet)
        total_raw_detections += len(raw_observations)

        # B. Assign Sequential local_track_id
        det_dicts = [{"class_name": o.class_name, "bbox_xyxy": o.bbox_xyxy} for o in raw_observations]
        local_track_ids = local_tracker.update(f_idx, det_dicts)

        frame_observations: List[ObjectObservationRecord] = []
        raw_det_records = []

        # C. 3D Unprojection into floor_map Frame
        for d_i, obs in enumerate(raw_observations):
            loc_tid = local_track_ids[d_i]
            x1, y1, x2, y2 = [int(v) for v in obs.bbox_xyxy]
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)

            mask = obs.get_mask()
            if mask is None:
                mask = np.zeros((h, w), dtype=np.uint8)
                mask[y1:y2, x1:x2] = 1
            else:
                mask = mask.astype(np.uint8)

            kernel = np.ones((5, 5), np.uint8)
            eroded_mask = cv2.erode(mask, kernel, iterations=1)
            if np.sum(eroded_mask) < 25:
                eroded_mask = mask

            v_coords, u_coords = np.where(eroded_mask > 0)
            z_vals = depth_m[v_coords, u_coords]
            valid_depth = np.isfinite(z_vals) & (z_vals >= 0.20) & (z_vals <= 4.0)

            has_depth = False
            obs_rec = None

            if np.sum(valid_depth) >= 30:
                u_valid, v_valid, z_valid = u_coords[valid_depth], v_coords[valid_depth], z_vals[valid_depth]
                med_z = np.median(z_valid)
                mad_z = np.median(np.abs(z_valid - med_z))
                inliers = np.abs(z_valid - med_z) <= (3.0 * max(1e-4, 1.4826 * mad_z))

                if np.sum(inliers) >= 30:
                    has_depth = True
                    pts_cam = intrinsics.pixels_to_camera(u_valid[inliers], v_valid[inliers], z_valid[inliers])
                    pts_floor = transform_points(T_floor_c, pts_cam)
                    centroid_cam = np.median(pts_cam, axis=0)
                    centroid_floor = np.median(pts_floor, axis=0)

                    obs_rec = ObjectObservationRecord(
                        obs_id=f"obs_{f_idx:04d}_{len(frame_observations):02d}",
                        frame_index=f_idx,
                        timestamp=t_rgb,
                        class_name=obs.class_name,
                        confidence=float(obs.confidence),
                        bbox_xyxy=[float(x) for x in obs.bbox_xyxy],
                        centroid_camera=[float(x) for x in centroid_cam],
                        centroid_floor=[float(x) for x in centroid_floor],
                        points_floor=pts_floor,
                        local_track_id=loc_tid,
                        status="VALID",
                    )
                    frame_observations.append(obs_rec)
                    total_valid_obs_count += 1

            if not has_depth:
                total_rejected_depth += 1

            raw_det_records.append({
                "detection_id": f"det_{f_idx:04d}_{d_i:02d}",
                "frame_index": f_idx,
                "timestamp": t_rgb,
                "class_name": obs.class_name,
                "confidence": round(float(obs.confidence), 4),
                "bbox_xyxy": [round(float(x), 1) for x in obs.bbox_xyxy],
                "local_track_id": loc_tid,
                "has_valid_depth": has_depth,
                "obs_record": obs_rec,
            })

        # D. Associate Observations to Global Entities
        assigned_entities = associator.update(frame_observations)

        # E. Stream Provenance JSONL
        for det_info in raw_det_records:
            obs_rec = det_info["obs_record"]
            gid = assigned_entities.get(obs_rec.obs_id, "") if obs_rec else ""
            det_json = {
                "detection_id": det_info["detection_id"],
                "frame_index": det_info["frame_index"],
                "timestamp": det_info["timestamp"],
                "class_name": det_info["class_name"],
                "confidence": det_info["confidence"],
                "bbox_xyxy": det_info["bbox_xyxy"],
                "local_track_id": det_info["local_track_id"],
                "global_entity_id": gid,
                "has_valid_depth": det_info["has_valid_depth"],
            }
            det_out.write(json.dumps(det_json) + "\n")

            if obs_rec is not None:
                b_min_f = np.min(obs_rec.points_floor, axis=0)
                b_max_f = np.max(obs_rec.points_floor, axis=0)
                obs_json = {
                    "obs_id": obs_rec.obs_id,
                    "frame_index": f_idx,
                    "timestamp": t_rgb,
                    "class_name": obs_rec.class_name,
                    "confidence": round(float(obs_rec.confidence), 4),
                    "local_track_id": obs_rec.local_track_id,
                    "global_entity_id": gid,
                    "points_count": len(obs_rec.points_floor),
                    "centroid_camera": [round(float(x), 4) for x in obs_rec.centroid_camera],
                    "centroid_floor": [round(float(x), 4) for x in obs_rec.centroid_floor],
                    "bbox_min_floor": [round(float(x), 4) for x in b_min_f],
                    "bbox_max_floor": [round(float(x), 4) for x in b_max_f],
                    "status": "VALID",
                }
                obs_out.write(json.dumps(obs_json) + "\n")

        # F. Depth Raycasting (Excluding Dynamic Persons)
        grid_step = 8
        u_sub, v_sub = np.meshgrid(np.arange(0, w, grid_step), np.arange(0, h, grid_step))
        u_flat, v_flat = u_sub.ravel(), v_sub.ravel()
        z_flat = depth_m[v_flat, u_flat]
        valid_scene = np.isfinite(z_flat) & (z_flat >= 0.20) & (z_flat <= 4.0)

        # Suppress dynamic persons
        person_boxes = [o.bbox_xyxy for o in raw_observations if o.class_name == "person"]
        if person_boxes:
            for pb in person_boxes:
                in_p = (u_flat >= pb[0]) & (u_flat <= pb[2]) & (v_flat >= pb[1]) & (v_flat <= pb[3])
                valid_scene = valid_scene & (~in_p)

        pts_cam_sc = intrinsics.pixels_to_camera(u_flat[valid_scene], v_flat[valid_scene], z_flat[valid_scene])
        pts_floor_sc = transform_points(T_floor_c, pts_cam_sc)

        occ_grid.integrate_depth_points(
            camera_floor_x=cam_floor_pos[0],
            camera_floor_y=cam_floor_pos[1],
            points_floor_aligned=pts_floor_sc,
        )

        # G. Save Sample Annotated Frame
        if idx % max(1, len(posed_selected_indices) // 6) == 0 or (len(frame_observations) > 0 and len(annotated_sample_frames) < 4):
            vis_img = rgb.copy()
            for o_rec in frame_observations:
                gid = assigned_entities.get(o_rec.obs_id, "")
                x1, y1, x2, y2 = [int(v) for v in o_rec.bbox_xyxy]
                cv2.rectangle(vis_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(vis_img, f"{gid} [{o_rec.local_track_id}] ({o_rec.confidence:.2f})",
                            (x1, max(18, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 2)
            s_name = f"frame_{f_idx:04d}_annotated.png"
            cv2.imwrite(str(vis_dir / s_name), vis_img)
            annotated_sample_frames.append(s_name)

        if (idx + 1) % 50 == 0 or (idx + 1) == len(posed_selected_indices):
            el = time.time() - proc_start_time
            rate = (idx + 1) / el
            print(f"  Processed {idx + 1}/{len(posed_selected_indices)} frames ({rate:.1f} FPS) | Valid Obs: {total_valid_obs_count} | Confirmed: {len(associator.entities)} | Candidates: {len(associator.unconfirmed_entities)}")

    det_out.close()
    obs_out.close()

    # 8. Geometry Fusion, Confirmation Policy, Spatial Merge, and Canonicalization
    print("\n[Step 6/6] Geometry cleaning, spatial merge, and delivering artifacts...")
    associator.filter_and_fuse_geometries()
    associator.evaluate_confirmation_policy()
    associator.merge_coincident_entities(merge_distance_threshold_m=0.40)
    associator.canonicalize_all_records()

    # Export canonical object_observations.jsonl from final authoritative state (zero dangling IDs)
    with open(out_dir / "object_observations.jsonl", "w", encoding="utf-8") as can_obs_out:
        all_canonical_entities = list(associator.entities.values()) + list(associator.unconfirmed_entities.values())
        for ent in all_canonical_entities:
            for obs_rec in ent.observations:
                b_min_f = np.min(obs_rec.points_floor, axis=0) if (obs_rec.points_floor is not None and len(obs_rec.points_floor) > 0) else [0, 0, 0]
                b_max_f = np.max(obs_rec.points_floor, axis=0) if (obs_rec.points_floor is not None and len(obs_rec.points_floor) > 0) else [0, 0, 0]
                obs_json = {
                    "obs_id": obs_rec.obs_id,
                    "frame_index": obs_rec.frame_index,
                    "timestamp": obs_rec.timestamp,
                    "class_name": obs_rec.class_name,
                    "confidence": round(float(obs_rec.confidence), 4),
                    "local_track_id": obs_rec.local_track_id,
                    "global_entity_id": ent.entity_id,
                    "points_count": len(obs_rec.points_floor) if obs_rec.points_floor is not None else 0,
                    "centroid_camera": [round(float(x), 4) for x in obs_rec.centroid_camera],
                    "centroid_floor": [round(float(x), 4) for x in obs_rec.centroid_floor] if obs_rec.centroid_floor else [0, 0, 0],
                    "bbox_min_floor": [round(float(x), 4) for x in b_min_f],
                    "bbox_max_floor": [round(float(x), 4) for x in b_max_f],
                }
                can_obs_out.write(json.dumps(obs_json) + "\n")

    # Export point clouds and semantic registry
    associator.export_point_clouds(out_dir)
    associator.export_semantic_map(out_dir / "semantic_map.json")

    pairwise_distances = associator.compute_pairwise_distances()
    with open(out_dir / "object_distances.json", "w", encoding="utf-8") as f:
        json.dump(pairwise_distances, f, indent=2)

    # Export raw occupancy map AND inflated navigation costmap
    occ_grid.export_ros_map(
        output_pgm_path=out_dir / "occupancy_map.pgm",
        output_yaml_path=out_dir / "occupancy_map.yaml",
        use_inflation=False,
    )
    occ_grid.export_ros_map(
        output_pgm_path=out_dir / "occupancy_costmap.pgm",
        output_yaml_path=out_dir / "occupancy_costmap.yaml",
        use_inflation=True,
    )

    grid_stats = occ_grid.compute_stats()

    # Save Configuration & Manifests
    effective_config = {
        "dataset_dir": str(dataset_dir),
        "output_dir": str(out_dir),
        "trajectory_file": str(traj_path),
        "sample_policy": args.sample_policy,
        "sample_step": args.sample_step,
        "sample_fps": args.sample_fps,
        "max_frames": args.max_frames,
        "map_resolution": args.map_resolution,
        "map_bounds": [origin_x, origin_y, width_m, height_m],
        "robot_radius": args.robot_radius,
        "safety_margin": args.safety_margin,
        "confidence_threshold": args.confidence_thresh,
        "classes": selected_classes,
        "mode": args.mode,
        "intrinsics": {
            "fx": intrinsics.fx, "fy": intrinsics.fy,
            "cx": intrinsics.cx, "cy": intrinsics.cy,
            "width": intrinsics.width, "height": intrinsics.height,
        },
        "depth_scale_m_per_unit": depth_scale,
    }
    with open(out_dir / "effective_config.json", "w", encoding="utf-8") as f:
        json.dump(effective_config, f, indent=2)

    with open(out_dir / "input_manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest.to_dict(), f, indent=2)

    system_metadata = {
        "platform": platform.platform(),
        "python_version": sys.version,
        "numpy_version": np.__version__,
        "opencv_version": cv2.__version__,
        "timestamp": datetime.datetime.now().isoformat(),
    }
    with open(out_dir / "system_metadata.json", "w", encoding="utf-8") as f:
        json.dump(system_metadata, f, indent=2)

    cmd_str = f"python {' '.join(sys.argv)}"
    with open(out_dir / "reproducible_command.sh", "w", encoding="utf-8") as f:
        f.write("#!/bin/bash\n" + cmd_str + "\n")

    # Audit summary
    multi_view_audit = []
    for eid, entity in associator.entities.items():
        obs_c = np.array([o.centroid_floor for o in entity.observations])
        devs = np.linalg.norm(obs_c - entity.centroid_floor, axis=1) if len(obs_c) >= 2 else np.array([0.0])
        multi_view_audit.append({
            "entity_id": eid,
            "class_name": entity.class_name,
            "observation_count": entity.observation_count,
            "frame_span": entity.frame_span,
            "mean_centroid_deviation_m": round(float(np.mean(devs)), 4),
            "max_centroid_deviation_m": round(float(np.max(devs)), 4),
            "converged_under_tolerance": bool(np.mean(devs) <= 0.25),
        })

    audit_summary = {
        "sequence_name": manifest.sequence_name,
        "coordinate_frame": "floor_map (ROS REP-103)",
        "total_input_frames": len(manifest.associated_pairs),
        "selected_frames": len(selected_indices),
        "processed_frames_with_pose": len(posed_selected_indices),
        "input_frame_coverage_pct": input_coverage_pct,
        "pose_coverage_pct": pose_coverage_pct,
        "raw_detections_count": total_raw_detections,
        "valid_3d_observations_count": total_valid_obs_count,
        "rejected_depth_count": total_rejected_depth,
        "confirmed_entities_count": len(associator.entities),
        "unconfirmed_candidates_count": len(associator.unconfirmed_entities),
        "floor_plane": floor_plane_info,
        "occupancy_grid_stats": grid_stats,
        "multi_view_consistency_audit": multi_view_audit,
    }
    with open(out_dir / "audit_summary.json", "w", encoding="utf-8") as f:
        json.dump(audit_summary, f, indent=2)

    # 9. Clean Visualizations
    traj_floor = cam_floor_pts
    occ_pgm = cv2.imread(str(out_dir / "occupancy_map.pgm"), cv2.IMREAD_GRAYSCALE)
    occ_img = np.flipud(occ_pgm)
    extent = [origin_x, origin_x + width_m, origin_y, origin_y + height_m]

    STYLE = {
        "chair": {"color": "#1E90FF", "border": "#000080", "name": "Chair"},
        "desk": {"color": "#32CD32", "border": "#006400", "name": "Desk/Table"},
        "table": {"color": "#32CD32", "border": "#006400", "name": "Desk/Table"},
        "door": {"color": "#FF4500", "border": "#8B0000", "name": "Door/Cabinet"},
        "storage box": {"color": "#FFA500", "border": "#D2691E", "name": "Storage Box"},
        "person": {"color": "#FF1493", "border": "#C71585", "name": "Person"},
        "backpack": {"color": "#9370DB", "border": "#4B0082", "name": "Backpack"},
        "computer monitor": {"color": "#00CED1", "border": "#008B8B", "name": "Monitor"},
        "whiteboard": {"color": "#F0E68C", "border": "#BDB76B", "name": "Whiteboard"},
        "laptop": {"color": "#00CED1", "border": "#008B8B", "name": "Laptop"},
    }
    DEFAULT_STYLE = {"color": "#A9A9A9", "border": "#696969", "name": "Object"}

    # Figure 1: Clean Floor Plan Overlay
    fig, ax = plt.subplots(figsize=(14, 13), dpi=220)
    ax.imshow(occ_img, cmap='gray', origin='lower', extent=extent, vmin=0, vmax=255, alpha=0.9)
    if len(traj_floor) > 0:
        ax.plot(traj_floor[:, 0], traj_floor[:, 1], color='#00CCCC', linewidth=2.8, label='Robot Trajectory', zorder=4)
        ax.scatter(traj_floor[0, 0], traj_floor[0, 1], color='#00FF00', edgecolors='black', s=140, marker='o', label='Start (0, 0)', zorder=5)
        ax.scatter(traj_floor[-1, 0], traj_floor[-1, 1], color='#FF0055', edgecolors='black', s=140, marker='X', label='End', zorder=5)

    used_classes = set()
    label_boxes = []

    for eid, entity in associator.entities.items():
        st = STYLE.get(entity.class_name, DEFAULT_STYLE)
        used_classes.add(entity.class_name)
        cf = entity.centroid_floor
        dim = entity.dimensions if entity.dimensions is not None else np.array([0.5, 0.5, 0.5])

        w_f = max(0.35, min(2.0, dim[0]))
        l_f = max(0.35, min(2.0, dim[1]))

        rect = patches.FancyBboxPatch(
            (cf[0] - w_f / 2.0, cf[1] - l_f / 2.0), w_f, l_f,
            boxstyle="round,pad=0.03",
            facecolor=st["color"],
            edgecolor=st["border"],
            linewidth=2.0,
            alpha=0.55,
            zorder=6
        )
        ax.add_patch(rect)
        ax.scatter(cf[0], cf[1], color=st["border"], s=40, zorder=7)

        lx, ly = cf[0], cf[1] + l_f / 2.0 + 0.16
        for bx, by in label_boxes:
            if math.hypot(lx - bx, ly - by) < 0.35:
                ly += 0.22
                lx += 0.12
        label_boxes.append((lx, ly))

        ax.text(
            lx, ly, eid,
            fontsize=9.0, weight='bold', color='black',
            ha='center', va='center',
            bbox=dict(boxstyle='round,pad=0.20', facecolor='white', edgecolor=st["border"], alpha=0.92, linewidth=1.4),
            zorder=8, clip_on=True
        )

    ax.set_xlim(origin_x, origin_x + width_m)
    ax.set_ylim(origin_y, origin_y + height_m)
    ax.set_aspect('equal')
    ax.grid(True, linestyle=':', alpha=0.55, color='#777777')
    ax.set_xlabel("Floor Coordinate X (meters - Forward)", fontsize=13, weight='bold')
    ax.set_ylabel("Floor Coordinate Y (meters - Left)", fontsize=13, weight='bold')
    ax.set_title(f"Reconstructed Semantic Floor Plan ({manifest.sequence_name})", fontsize=14, weight='bold', pad=12)

    leg_items = [
        plt.Line2D([0], [0], color='#00CCCC', lw=2.8, label='Robot Trajectory'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='#00FF00', markeredgecolor='k', markersize=10, label='Start (0, 0)'),
        plt.Line2D([0], [0], marker='X', color='w', markerfacecolor='#FF0055', markeredgecolor='k', markersize=10, label='End'),
    ]
    for cls in sorted(used_classes):
        st = STYLE.get(cls, DEFAULT_STYLE)
        leg_items.append(patches.Patch(facecolor=st["color"], edgecolor=st["border"], alpha=0.6, label=st["name"]))
    ax.legend(handles=leg_items, loc='upper right', framealpha=0.95, fontsize=10, title="Legend")

    floor_plan_png = out_dir / "semantic_floor_plan.png"
    plt.tight_layout()
    plt.savefig(floor_plan_png)
    plt.close()

    # Figure 2: Master 4-Panel Audit Panel
    fig2 = plt.figure(figsize=(19, 13), dpi=200)

    # Panel 1: 3D Scene
    ax1 = fig2.add_subplot(2, 2, 1, projection='3d')
    if len(traj_floor) > 0:
        ax1.plot(traj_floor[:, 0], traj_floor[:, 1], traj_floor[:, 2], color='#000080', linewidth=2.0, label='Camera Path')
        ax1.scatter(traj_floor[0, 0], traj_floor[0, 1], traj_floor[0, 2], color='lime', s=80, label='Start')
        ax1.scatter(traj_floor[-1, 0], traj_floor[-1, 1], traj_floor[-1, 2], color='red', s=80, label='End')

    for eid, entity in associator.entities.items():
        st = STYLE.get(entity.class_name, DEFAULT_STYLE)
        cf = entity.centroid_floor
        ax1.scatter(cf[0], cf[1], cf[2], color=st["color"], edgecolors=st["border"], s=85, alpha=0.9)
        ax1.text(cf[0], cf[1], cf[2] + 0.15, eid, fontsize=8, weight='bold', color='black')

    ax1.set_xlabel("X (Forward, m)")
    ax1.set_ylabel("Y (Left, m)")
    ax1.set_zlabel("Z (Height, m)")
    ax1.set_title("3D Scene in floor_map Frame (ROS REP-103)", fontsize=11, weight='bold')
    ax1.legend(loc='upper right', fontsize=8)

    # Panel 2: Semantic Floor Plan
    ax2 = fig2.add_subplot(2, 2, 2)
    ax2.imshow(occ_img, cmap='gray', origin='lower', extent=extent, vmin=0, vmax=255, alpha=0.85)
    if len(traj_floor) > 0:
        ax2.plot(traj_floor[:, 0], traj_floor[:, 1], color='#00CCCC', linewidth=2.0, label='Trajectory')
    for eid, entity in associator.entities.items():
        st = STYLE.get(entity.class_name, DEFAULT_STYLE)
        cf = entity.centroid_floor
        circ = patches.Circle((cf[0], cf[1]), radius=0.25, facecolor=st["color"], edgecolor=st["border"], alpha=0.6, linewidth=1.5)
        ax2.add_patch(circ)
        ax2.text(cf[0], cf[1] + 0.32, eid, fontsize=8, weight='bold', ha='center', va='center',
                 bbox=dict(boxstyle='square,pad=0.15', facecolor='white', edgecolor=st["border"], alpha=0.85), clip_on=True)
    ax2.set_xlim(origin_x, origin_x + width_m)
    ax2.set_ylim(origin_y, origin_y + height_m)
    ax2.set_aspect('equal')
    ax2.grid(True, linestyle=':', alpha=0.5)
    ax2.set_title("Top-Down Semantic Map", fontsize=11, weight='bold')

    # Panel 3: Raw Metric Occupancy Grid
    ax3 = fig2.add_subplot(2, 2, 3)
    ax3.imshow(occ_img, cmap='gray', origin='lower', extent=extent, vmin=0, vmax=255)
    if len(traj_floor) > 0:
        ax3.plot(traj_floor[:, 0], traj_floor[:, 1], color='cyan', linewidth=2.0, label='Trajectory')
    ax3.set_xlim(origin_x, origin_x + width_m)
    ax3.set_ylim(origin_y, origin_y + height_m)
    ax3.set_aspect('equal')
    ax3.grid(True, linestyle=':', alpha=0.45)
    ax3.set_title("Raw Metric Occupancy Grid (White: Free, Black: Obstacle)", fontsize=11, weight='bold')

    # Panel 4: Physical Clearances Between Key Objects
    ax4 = fig2.add_subplot(2, 2, 4)
    sample_pairs = pairwise_distances[:min(6, len(pairwise_distances))]
    if sample_pairs:
        p_labels = [f"{p['entity_a']}-\n{p['entity_b']}" for p in sample_pairs]
        p_dists = [p['centroid_distance_2d_m'] for p in sample_pairs]
        p_clears = [p['footprint_clearance_m'] for p in sample_pairs]
        p_errs = [p['uncertainty_std_m'] for p in sample_pairs]
        x_idx = np.arange(len(p_labels))
        w_bar = 0.35
        ax4.bar(x_idx - w_bar/2, p_dists, w_bar, label='Centroid Dist (m)', color='#1E90FF')
        ax4.bar(x_idx + w_bar/2, p_clears, w_bar, yerr=p_errs, capsize=4, label='Footprint Clearance (m)', color='#FF8C00')
        ax4.set_xticks(x_idx)
        ax4.set_xticklabels(p_labels, fontsize=8)
        ax4.set_ylabel("Distance (m)", fontsize=10)
        ax4.set_title("Object Clearance with Measurement Uncertainty", fontsize=11, weight='bold')
        ax4.legend(loc='upper right', fontsize=8)
    else:
        ax4.text(0.5, 0.5, "No confirmed pairs", ha='center', va='center')
    ax4.grid(True, linestyle=':', alpha=0.5)

    plt.tight_layout()
    master_png = out_dir / "mapping_visualization.png"
    plt.savefig(master_png)
    plt.close()

    # Markdown Report
    report_md = f"""# Offline Semantic Mapping Audit Report

**Sequence**: `{manifest.sequence_name}`
**Timestamp**: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
**Pipeline Mode**: `{args.mode}`
**Output Path**: `{out_dir}`

---

## 1. Sequence Provenance & Coverage
- **Total RGB Frames**: `{manifest.total_rgb_count}`
- **Total Depth Frames**: `{manifest.total_depth_count}`
- **Synchronized RGB-D Pairs**: `{manifest.total_associated_count}`
- **Sampling Policy**: `{args.sample_policy}` (step={args.sample_step}, fps={args.sample_fps})
- **Sampled Frames Processed**: `{len(posed_selected_indices)}`
- **Input Coverage**: `{input_coverage_pct}%` of all available sequence frames
- **Pose Alignment Coverage**: `{pose_coverage_pct}%` of sampled frames matched to SLAM poses

---

## 2. Floor Coordinate System & Transform
- **Frame Convention**: ROS REP-103 (`floor_map` frame: X=Forward, Y=Left, Z=Height above floor)
- **Fitted Floor Plane**:
  - Upward Normal: `[{n_floor[0]:.4f}, {n_floor[1]:.4f}, {n_floor[2]:.4f}]`
  - Plane Distance $d$: `{d_floor:.4f} m`
  - Inliers: `{inlier_cnt}/{len(floor_candidates)} ({inlier_pct}%)`
  - Residual Mean/Std: `{plane_fit.residual_mean*1000:.1f} mm` / `{plane_fit.residual_std*1000:.1f} mm`
- **Transform**: Full $4 \\times 4$ matrix $T_{{\\text{{floor\\_world}}}}$ stored in `floor_plane.json`.

---

## 3. Dynamic Occupancy Grid & Obstacle Clearances
- **Origin**: `({origin_x:.3f}, {origin_y:.3f}) m`
- **Dimensions**: `{width_m:.2f} x {height_m:.2f} m` @ `{args.map_resolution} m/cell` ({occ_grid.cols} x {occ_grid.rows} cells)
- **Free Cells**: `{grid_stats['free_cells']} ({grid_stats['free_pct']}%)`
- **Occupied Cells**: `{grid_stats['occupied_cells']} ({grid_stats['occupied_pct']}%)`
- **Unknown Cells**: `{grid_stats['unknown_cells']} ({grid_stats['unknown_pct']}%)`
- **Inflated Navigation Occupied Cells**: `{grid_stats['inflated_occupied_cells']} ({grid_stats['inflated_occupied_pct']}%)`
- **Robot Radius / Safety Margin**: `{args.robot_radius} m` / `{args.safety_margin} m`

---

## 4. Semantic Objects Summary
- **Confirmed Static Entities**: `{len(associator.entities)}`
- **Quarantined Candidates**: `{len(associator.unconfirmed_entities)}`

| Entity ID | Class | Observations | Frame Span | Dimensions (WxLxH m) | Spread $\\sigma$ (m) | Status |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
"""
    for eid, e in sorted(associator.entities.items()):
        report_md += f"| `{eid}` | {e.class_name} | {e.observation_count} | {e.frame_span} | `{e.dimensions[0]:.2f}x{e.dimensions[1]:.2f}x{e.dimensions[2]:.2f}` | `{e.centroid_std_floor:.3f}` | **CONFIRMED** |\n"
    for eid, e in sorted(associator.unconfirmed_entities.items()):
        report_md += f"| `{eid}` | {e.class_name} | {e.observation_count} | {e.frame_span} | `{e.dimensions[0]:.2f}x{e.dimensions[1]:.2f}x{e.dimensions[2]:.2f}` | - | QUARANTINED ({e.quarantine_reason}) |\n"

    report_md += f"""
---

## 5. Pairwise Clearances
Pairwise clearances computed with derived measurement uncertainty:
$\\sigma_{{AB}} = \\sqrt{{\\sigma_A^2 + \\sigma_B^2 + 0.015^2}}$
"""
    with open(out_dir / "mapping_audit_report.md", "w", encoding="utf-8") as f:
        f.write(report_md)

    print("\n" + "=" * 75)
    print("MAPPING PIPELINE COMPLETED SUCCESSFULLY!")
    print(f"Total Wall Time:  {time.time() - start_wall_time:.1f} s")
    print(f"Artifacts saved in: {out_dir}")
    print("=" * 75)
    return audit_summary


def main():
    parser = argparse.ArgumentParser(
        description="Generalized Offline Semantic Mapping Pipeline with Dynamic Discovery and Floor Calibration",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # Required or primary arguments
    parser.add_argument("dataset_dir", nargs="?", default="data/raw/d455_rgbd_20261010_123443",
                        help="Path to RGB-D recording directory")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Output directory (defaults to unique timestamped folder under outputs/)")
    parser.add_argument("--overwrite", action="store_true",
                        help="Allow overwriting existing deliverables in output directory")

    # Calibration and trajectory inputs
    parser.add_argument("--trajectory-file", type=str, default=None,
                        help="TUM format trajectory file (validates timestamp overlap with recording)")
    parser.add_argument("--associations-file", type=str, default=None,
                        help="TUM format associations file (if omitted, dynamically computed via DP matching)")
    parser.add_argument("--intrinsics-file", type=str, default=None,
                        help="Path to camera intrinsics JSON or YAML calibration")
    parser.add_argument("--depth-scale", type=float, default=None,
                        help="Depth unit scale in meters/unit (e.g. 0.001 for millimeters)")

    # Explicit camera intrinsics overrides
    parser.add_argument("--fx", type=float, default=None, help="Camera focal length X in pixels")
    parser.add_argument("--fy", type=float, default=None, help="Camera focal length Y in pixels")
    parser.add_argument("--cx", type=float, default=None, help="Camera principal point X in pixels")
    parser.add_argument("--cy", type=float, default=None, help="Camera principal point Y in pixels")
    parser.add_argument("--width", type=int, default=None, help="Image width in pixels")
    parser.add_argument("--height", type=int, default=None, help="Image height in pixels")

    # Sampling policy
    parser.add_argument("--sample-policy", choices=["step", "fps", "all"], default="step",
                        help="Frame subsampling policy")
    parser.add_argument("--sample-step", type=int, default=5,
                        help="Sampling step stride when policy is 'step'")
    parser.add_argument("--sample-fps", type=float, default=5.0,
                        help="Target FPS when policy is 'fps'")
    parser.add_argument("--max-frames", type=int, default=None,
                        help="Maximum frames to process")
    parser.add_argument("--sync-max-dt", type=float, default=0.05,
                        help="Max time tolerance (seconds) for RGB-Depth association")
    parser.add_argument("--pose-tolerance-s", type=float, default=0.05,
                        help="Max time tolerance (seconds) for matching camera poses")

    # Perception & Semantic thresholds
    parser.add_argument("--classes", type=str, default="all",
                        help="Target classes ('all' or comma-separated names)")
    parser.add_argument("--model-path", type=str, default="models/yoloe-26s-seg.pt",
                        help="YOLOE weights file")
    parser.add_argument("--confidence-thresh", type=float, default=0.25,
                        help="Detection confidence threshold")
    parser.add_argument("--min-hits", type=int, default=8,
                        help="Minimum observations required to confirm persistent object")
    parser.add_argument("--min-span", type=int, default=15,
                        help="Minimum frame span required to confirm persistent object")

    # Metric Occupancy & Robot Footprint
    parser.add_argument("--map-resolution", type=float, default=0.05,
                        help="Occupancy grid resolution in meters/cell")
    parser.add_argument("--map-bounds", type=str, default="auto",
                        help="'auto' or 'min_x,min_y,max_x,max_y' for fixed map extent")
    parser.add_argument("--map-padding", type=float, default=1.2,
                        help="Padding in meters around observed camera path for auto map bounds")
    parser.add_argument("--robot-radius", type=float, default=0.18,
                        help="Robot collision radius in meters for navigation costmap")
    parser.add_argument("--safety-margin", type=float, default=0.05,
                        help="Safety inflation clearance in meters")

    # Operating mode
    parser.add_argument("--mode", choices=["fresh", "update"], default="fresh",
                        help="Map lifecycle mode ('fresh' scopes IDs to current run; 'update' requires reference map)")
    parser.add_argument("--reference-map", type=str, default=None,
                        help="Path to reference semantic map when --mode is 'update'")

    args = parser.parse_args()
    try:
        run_pipeline(args)
    except Exception as e:
        print(f"\n[FATAL ERROR]: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
