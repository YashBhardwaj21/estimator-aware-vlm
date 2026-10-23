#!/usr/bin/env python3
"""Offline RGB-D Semantic Mapping and Trajectory Integration Pipeline.

Accepts any arbitrary RealSense D455 RGB-D recording session and automatically:
  1. Validates recording integrity, frame counts, synchronization skew, and D455 profile.
  2. Generates RGB-Depth timestamp associations.
  3. Runs ORB-SLAM3 RGB-D trajectory estimation with persistent Atlas save (if no trajectory provided).
  4. Audits camera trajectory continuity and timestamp matching.
  5. Performs YOLOE segmentation, 3D metric projection, and local tracking.
  6. Maintains persistent global identity with canonical alias table (entity_aliases.json).
  7. Fuses multi-view 3D geometry and estimates floor plane (T_floor_world).
  8. Computes object dimensions, footprints, and pairwise clearances with uncertainty.
  9. Builds metric occupancy grid (occupancy_grid.pgm, occupancy_grid.yaml).
  10. Produces comprehensive visualizations and audit reports in outputs/<run_id>/.

Usage:
  python mapping/run_mapping_pipeline.py data/raw/desk_run_20261009_161552 --output-root outputs
"""

from __future__ import annotations

import argparse
import datetime
import json
import math
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yaml

# Add src and repo root to path
from estimator_aware_vlm.common.paths import get_repo_root
repo_root = get_repo_root()
sys.path.insert(0, str(repo_root / "src"))

from scene_graph.geometry.coordinate_frames import CoordinateFrameContract, RobotExtrinsics
from scene_graph.data.dataset_discovery import discover_recording, validate_recording_session, RecordingManifest
from mapping.run_offline_semantic_mapping import run_pipeline, sample_frame_indices


def generate_orbslam3_yaml(
    manifest: RecordingManifest,
    output_yaml_path: Path,
    atlas_save_path: Optional[Path] = None,
) -> Path:
    """Generate calibrated ORB-SLAM3 configuration YAML for the active stream profile."""
    fx = manifest.intrinsics.fx
    fy = manifest.intrinsics.fy
    cx = manifest.intrinsics.cx
    cy = manifest.intrinsics.cy
    w = manifest.intrinsics.width
    h = manifest.intrinsics.height
    fps = max(10, int(round(manifest.fps_estimate)))
    depth_factor = manifest.depth_units_per_meter

    content = f"""%YAML:1.0

# Auto-generated ORB-SLAM3 Settings for {manifest.sequence_name}
File.version: "1.0"

Camera.type: "PinHole"
Camera1.fx: {fx:.6f}
Camera1.fy: {fy:.6f}
Camera1.cx: {cx:.6f}
Camera1.cy: {cy:.6f}

Camera1.k1: 0.0
Camera1.k2: 0.0
Camera1.p1: 0.0
Camera1.p2: 0.0
Camera1.k3: 0.0

Camera.width: {w}
Camera.height: {h}
Camera.fps: {fps}
Camera.RGB: 0

Stereo.ThDepth: 40.0
Stereo.b: 0.095
RGBD.DepthMapFactor: {depth_factor:.1f}

ORBextractor.nFeatures: 1200
ORBextractor.scaleFactor: 1.2
ORBextractor.nLevels: 8
ORBextractor.iniThFAST: 20
ORBextractor.minThFAST: 7

Viewer.KeyFrameSize: 0.05
Viewer.KeyFrameLineWidth: 1.0
Viewer.GraphLineWidth: 0.9
Viewer.PointSize: 1.0
Viewer.CameraSize: 0.08
Viewer.CameraLineWidth: 3.0
Viewer.ViewpointX: 0.0
Viewer.ViewpointY: -0.7
Viewer.ViewpointZ: -1.8
Viewer.ViewpointF: 500.0
"""
    if atlas_save_path:
        content += f'\nSystem.SaveAtlasToFile: "{atlas_save_path.as_posix()}"\n'

    output_yaml_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_yaml_path, "w", encoding="utf-8") as f:
        f.write(content)
    return output_yaml_path


from scene_graph.pipeline.trajectory_manager import (
    load_and_validate_trajectory,
    TrajectoryResult,
    TrajectoryValidationError,
)


def execute_orbslam3(
    dataset_dir: Path,
    associations_file: Path,
    settings_yaml: Path,
    trajectory_out_dir: Path,
    orbslam3_root: Optional[str] = None,
    viewer: bool = False,
) -> Tuple[Path, Path]:
    """Execute ORB-SLAM3 (headless or with Pangolin GUI) and save trajectories."""
    trajectory_out_dir.mkdir(parents=True, exist_ok=True)
    root_str = orbslam3_root or os.getenv("ORB_SLAM3_ROOT", "") or "/opt/ORB_SLAM3"
    root_dir = Path(root_str)
    vocab_path = root_dir / "Vocabulary" / "ORBvoc.txt"
    exec_path = root_dir / "Examples" / "RGB-D" / "rgbd_tum"

    if not exec_path.exists():
        raise FileNotFoundError(
            f"ORB-SLAM3 executable not found at '{exec_path}'. "
            f"Set ORB_SLAM3_ROOT environment variable, pass --orbslam3-root, "
            f"or supply a verified trajectory with --trajectory-file."
        )

    log_path = trajectory_out_dir / "run.log"
    print(f"Executing ORB-SLAM3 on {dataset_dir} (Pangolin viewer={'ENABLED' if viewer else 'OFF'}, logging to {log_path})...")

    # Command runs in trajectory_out_dir so CameraTrajectory.txt is written there
    cmd = [
        str(exec_path),
        str(vocab_path),
        str(settings_yaml),
        str(dataset_dir),
        str(associations_file),
        "1" if viewer else "0",  # 1 = Pangolin viewer enabled, 0 = headless
    ]

    env = os.environ.copy()
    if viewer:
        # In WSLg, the hardware D3D12 Gallium driver frequently deadlocks on Pangolin OpenGL context creation,
        # leaving a ghost taskbar icon while freezing execution. Software rendering bypasses this cleanly.
        env["LIBGL_ALWAYS_SOFTWARE"] = "1"

    with open(log_path, "w") as log_f:
        res = subprocess.run(cmd, cwd=str(trajectory_out_dir), stdout=log_f, stderr=subprocess.STDOUT, env=env)

    cam_traj = trajectory_out_dir / "CameraTrajectory.txt"
    kf_traj = trajectory_out_dir / "KeyFrameTrajectory.txt"

    if not cam_traj.exists() or cam_traj.stat().st_size < 100:
        raise RuntimeError(f"ORB-SLAM3 execution failed with returncode {res.returncode}. See {log_path}")
    if res.returncode != 0:
        print(f"  [Notice] ORB-SLAM3 exited with code {res.returncode}, but CameraTrajectory.txt successfully written ({cam_traj.stat().st_size} bytes).")

    return cam_traj, kf_traj



def main():
    parser = argparse.ArgumentParser(description="One-command offline mapping pipeline for D455 recordings")
    parser.add_argument("recording_path", type=str, help="Path to raw D455 recording directory")
    parser.add_argument("--output-root", type=str, default="outputs", help="Root directory for outputs")
    parser.add_argument("--run-id", type=str, default=None, help="Explicit run ID / folder name")
    parser.add_argument("--trajectory-file", type=str, default=None, help="Optional existing trajectory file override")
    parser.add_argument("--sample-step", type=int, default=5, help="Sampling step stride (default 5)")
    parser.add_argument("--orbslam3-root", type=str, default=os.getenv("ORB_SLAM3_ROOT", None), help="Path to ORB-SLAM3 installation (default: $ORB_SLAM3_ROOT or /opt/ORB_SLAM3)")
    parser.add_argument("--model-path", type=str, default="models/yoloe-26s-seg.pt", help="YOLOE weights file")
    parser.add_argument("--robot-radius", type=float, default=0.18, help="Robot footprint radius in meters")
    parser.add_argument("--safety-margin", type=float, default=0.05, help="Safety clearance inflation in meters")
    parser.add_argument("--save-atlas", action="store_true", default=False, help="Enable ORB-SLAM3 persistent Atlas serialization (.osa). Disabled by default to avoid Boost serialization crash on map merges.")
    parser.add_argument("--viewer", "--gui", dest="viewer", action="store_true", default=False, help="Display live ORB-SLAM3 tracking in Pangolin GUI window.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing output directory")
    args = parser.parse_args()

    rec_dir = Path(args.recording_path).resolve()
    seq_name = rec_dir.name
    run_id = args.run_id or f"semantic_mapping_{seq_name}"
    out_dir = Path(args.output_root).resolve() / run_id

    print("=" * 76)
    print("STARTING ONE-COMMAND D455 AUTONOMOUS OFFLINE MAPPING PIPELINE")
    print(f"Recording Directory:  {rec_dir}")
    print(f"Output Directory:     {out_dir}")
    print(f"Target Run ID:        {run_id}")
    print("=" * 76)

    # 1. Recording Validation
    print("\n[Step 1/6] Validating Recording Session and Integrity...")
    val_report = validate_recording_session(rec_dir)
    print(f"  Validation Status: {val_report['status']}")
    print(f"  Captured Pairs:    {val_report.get('counts', {}).get('associated_pairs', 0)}")
    print(f"  Depth Valid Pct:   {val_report.get('quality', {}).get('mean_valid_depth_fraction', 0)*100:.1f}%")
    if val_report["status"] != "PASS":
        print(f"[ERROR] Recording validation failed: {val_report.get('errors')}", file=sys.stderr)
        sys.exit(1)

    out_dir.mkdir(parents=True, exist_ok=True)
    traj_dir = out_dir / "trajectory"
    vis_dir = out_dir / "visualizations"
    obj_dir = out_dir / "objects"
    traj_dir.mkdir(parents=True, exist_ok=True)
    vis_dir.mkdir(parents=True, exist_ok=True)
    obj_dir.mkdir(parents=True, exist_ok=True)

    # Copy recording validation into run directory
    with open(out_dir / "recording_validation.json", "w", encoding="utf-8") as f:
        json.dump(val_report, f, indent=2)

    # Discover Manifest
    manifest = discover_recording(rec_dir)
    with open(out_dir / "input_manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest.to_dict(), f, indent=2)

    # 2. ORB-SLAM3 Trajectory Ingestion, Provenance, & Quality Gate
    print("\n[Step 2/6] Acquiring Camera Trajectory & Persistent Atlas Map...")
    assoc_file = rec_dir / "associations.txt"
    if not assoc_file.exists():
        print(f"  Writing {assoc_file.name} ({len(manifest.associated_pairs)} pairs)...")
        with open(assoc_file, "w", encoding="utf-8") as f:
            for p in manifest.associated_pairs:
                f.write(f"{p[0]:.6f} {p[1]} {p[2]:.6f} {p[3]}\n")
    shutil.copy2(assoc_file, out_dir / "associations.txt")
    rgb_timestamps = [p[0] for p in manifest.associated_pairs]
    atlas_save = traj_dir / "atlas.osa"

    if args.trajectory_file and Path(args.trajectory_file).exists():
        raw_traj = Path(args.trajectory_file)
        print(f"  Validating supplied trajectory override: {raw_traj}")
        traj_res = load_and_validate_trajectory(
            trajectory_file=raw_traj,
            rgb_timestamps=rgb_timestamps,
            max_dt=0.05,
            min_coverage_pct=20.0,
            atlas_path=atlas_save if atlas_save.exists() else None,
            generated_in_current_run=False,
        )
        if not traj_res.is_valid:
            raise RuntimeError(f"Trajectory Validation FAILED: {traj_res.rejection_reason}")
        shutil.copy2(raw_traj, traj_dir / "CameraTrajectory.txt")
        cam_traj = traj_dir / "CameraTrajectory.txt"
    else:
        slam_yaml = traj_dir / f"d455_{seq_name}_orbslam3.yaml"
        atlas_save_target = atlas_save if args.save_atlas else None
        generate_orbslam3_yaml(manifest, slam_yaml, atlas_save_path=atlas_save_target)
        cam_traj, kf_traj = execute_orbslam3(
            rec_dir, assoc_file, slam_yaml, traj_dir, orbslam3_root=args.orbslam3_root, viewer=args.viewer
        )
        traj_res = load_and_validate_trajectory(
            trajectory_file=cam_traj,
            rgb_timestamps=rgb_timestamps,
            max_dt=0.05,
            min_coverage_pct=20.0,
            atlas_path=atlas_save if (args.save_atlas and atlas_save.exists()) else None,
            generated_in_current_run=True,
        )
        if not traj_res.is_valid:
            raise RuntimeError(f"Generated SLAM Trajectory FAILED validation: {traj_res.rejection_reason}")

    with open(traj_dir / "trajectory_audit.json", "w", encoding="utf-8") as f:
        json.dump(traj_res.to_dict(), f, indent=2)

    print(f"  Trajectory Status:   {traj_res.status} (SHA256: {traj_res.file_sha256[:12]}...)")
    print(f"  1-to-1 Coverage:     {traj_res.coverage_percentage:.1f}% ({traj_res.unique_matched_frames}/{traj_res.total_recording_frames} frames)")
    print(f"  Mean Timestamp Skew: {traj_res.mean_dt_ms:.2f} ms")
    print(f"  Atlas Loadable:      {'YES' if traj_res.atlas_verified else 'NO (Replay Only)'}")

    # 3. Execute Semantic Mapping Pipeline Core
    print("\n[Step 3/6] Running Semantic Perception, Geometry Fusion, & Floor Alignment...")
    mapping_args = argparse.Namespace(
        dataset_dir=str(rec_dir),
        output_dir=str(out_dir),
        trajectory_file=str(cam_traj),
        intrinsics_file=None,
        depth_scale=None,
        associations_file=None,
        fx=None, fy=None, cx=None, cy=None, width=None, height=None,
        sample_policy="step",
        sample_step=args.sample_step,
        sample_fps=5.0,
        max_frames=None,
        sync_max_dt=0.05,
        pose_tolerance_s=0.05,
        classes="all",
        model_path=args.model_path,
        confidence_thresh=0.25,
        min_hits=8,
        min_span=15,
        map_resolution=0.05,
        map_bounds="auto",
        map_padding=1.2,
        robot_radius=args.robot_radius,
        safety_margin=args.safety_margin,
        mode="fresh",
        reference_map=None,
        overwrite=True,
    )
    run_pipeline(mapping_args)

    # 4. Standardize Map & Occupancy Outputs
    print("\n[Step 4/6] Exporting Standard Occupancy Grids & Maps...")
    if (out_dir / "occupancy_map.pgm").exists():
        shutil.copy2(out_dir / "occupancy_map.pgm", out_dir / "map.pgm")
        shutil.copy2(out_dir / "occupancy_map.yaml", out_dir / "map.yaml")
    if (out_dir / "occupancy_costmap.pgm").exists():
        shutil.copy2(out_dir / "occupancy_costmap.pgm", out_dir / "occupancy_grid.pgm")
        shutil.copy2(out_dir / "occupancy_costmap.yaml", out_dir / "occupancy_grid.yaml")

    # Metadata
    occ_meta = {
        "map_frame": "floor_map",
        "resolution_m": 0.05,
        "robot_footprint_radius_m": args.robot_radius,
        "safety_margin_m": args.safety_margin,
        "occupied_threshold": 100,
        "free_threshold": 50,
        "files": {
            "occupancy_map": "map.yaml",
            "occupancy_grid": "occupancy_grid.yaml",
            "semantic_map": "semantic_map.json",
        }
    }
    with open(out_dir / "occupancy_metadata.json", "w", encoding="utf-8") as f:
        json.dump(occ_meta, f, indent=2)

    # 5. Build Entity Aliases Table
    print("\n[Step 5/6] Building Authoritative Entity Aliases & Canonical Membership...")
    with open(out_dir / "semantic_map.json", "r", encoding="utf-8") as f:
        sem_map = json.load(f)

    aliases = {
        "sequence_name": seq_name,
        "canonical_entities": list(sem_map.get("entities", {}).keys()),
        "alias_mapping": {eid: eid for eid in sem_map.get("entities", {})},
        "merges": [],
        "splits": [],
        "quarantined_candidates": [],
    }
    with open(out_dir / "entity_aliases.json", "w", encoding="utf-8") as f:
        json.dump(aliases, f, indent=2)

    # 6. Generate Configurations & Final Visualizations
    print("\n[Step 6/6] Generating Runtime Configurations & Final Deliverables...")
    loc_cfg = {
        "localization": {
            "map_directory": str(out_dir),
            "floor_transform_file": "floor_plane.json",
            "atlas_file": "trajectory/atlas.osa" if traj_res.atlas_verified else None,
            "status": "READY_FOR_LOCALIZATION" if traj_res.atlas_verified else "REPLAY_ONLY (Atlas not available)",
            "reference_trajectory": "trajectory/CameraTrajectory.txt",
            "trajectory_sha256": traj_res.file_sha256,
            "max_covariance": 0.05,
            "min_inliers_tracked": 15,
        }
    }
    with open(out_dir / "localization_config.yaml", "w", encoding="utf-8") as f:
        yaml.dump(loc_cfg, f, default_flow_style=False)

    fp_cfg = {
        "robot_footprint": {
            "radius_m": args.robot_radius,
            "safety_margin_m": args.safety_margin,
            "track_width_m": 0.20,
            "camera_offset_x_m": 0.12,
            "camera_offset_z_m": 0.16,
        }
    }
    with open(out_dir / "robot_footprint.yaml", "w", encoding="utf-8") as f:
        yaml.dump(fp_cfg, f, default_flow_style=False)

    # Write audit report alias
    if (out_dir / "mapping_audit_report.md").exists():
        shutil.copy2(out_dir / "mapping_audit_report.md", out_dir / "audit_report.md")

    # Copy visualizations into visualizations/ subfolder
    for vis_file in ("semantic_floor_plan.png", "mapping_visualization.png"):
        src_p = out_dir / vis_file
        if src_p.exists():
            shutil.copy2(src_p, vis_dir / vis_file)

    print("\n" + "=" * 76)
    print("ONE-COMMAND OFFLINE MAPPING PIPELINE COMPLETED SUCCESSFULLY!")
    print(f"All deliverables ready at: {out_dir}")
    print("=" * 76)


if __name__ == "__main__":
    main()
