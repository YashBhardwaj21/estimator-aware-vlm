#!/usr/bin/env python3
"""Generate Clean Reporting and Visualizations for Semantic Mapping Run 2."""

import json
import math
import os
import sys
from pathlib import Path
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

# sys.path.insert(0, os.path.abspath("src"))
from scene_graph.geometry.transforms import pose_to_transform


def main():
    run_dir = Path("outputs/semantic_mapping_run2")
    
    with open(run_dir / "floor_plane.json", "r") as f:
        floor_info = json.load(f)
    T_floor_world = np.array(floor_info["T_floor_world"])
    
    with open(run_dir / "semantic_map.json", "r") as f:
        sem_map = json.load(f)
        
    with open(run_dir / "object_distances.json", "r") as f:
        pairwise_distances = json.load(f)
        
    # Read occupancy map
    occ_pgm = cv2.imread(str(run_dir / "occupancy_map.pgm"), cv2.IMREAD_GRAYSCALE)
    occ_img = np.flipud(occ_pgm)
    
    with open(run_dir / "occupancy_map.yaml", "r") as f:
        meta = {}
        for line in f:
            if ":" in line:
                k, v = line.split(":", 1)
                meta[k.strip()] = v.strip()
    res = float(meta["resolution"])
    orig = [float(x) for x in meta["origin"].strip("[]").split(",")]
    rows, cols = occ_img.shape
    extent = [orig[0], orig[0] + cols * res, orig[1], orig[1] + rows * res]
    
    # Load trajectory and transform into floor_map frame
    traj_floor = []
    with open("outputs/orbslam3_d455_trajectory/full_run/CameraTrajectory.txt", "r") as f:
        for line in f:
            parts = [float(x) for x in line.strip().split()]
            if len(parts) >= 8:
                tx, ty, tz = parts[1:4]
                qx, qy, qz, qw = parts[4:8]
                T_wc = pose_to_transform(tx, ty, tz, qx, qy, qz, qw)
                T_fc = T_floor_world @ T_wc
                traj_floor.append(T_fc[:3, 3])
    traj_floor = np.array(traj_floor)
    
    # Occupancy statistics
    # ROS PGM: 254 = Free, 0 = Occupied, 205 = Unknown
    total_cells = rows * cols
    free_cells = int(np.sum(occ_img == 254))
    occ_cells = int(np.sum(occ_img == 0))
    unk_cells = int(np.sum(occ_img == 205))
    grid_stats = {
        "resolution_m": res,
        "width_m": cols * res,
        "height_m": rows * res,
        "total_cells": total_cells,
        "free_cells": free_cells,
        "free_pct": round(free_cells / total_cells * 100.0, 2),
        "occupied_cells": occ_cells,
        "occupied_pct": round(occ_cells / total_cells * 100.0, 2),
        "unknown_cells": unk_cells,
        "unknown_pct": round(unk_cells / total_cells * 100.0, 2),
    }
    
    # Multi-view consistency audit
    multi_view_audit = []
    for eid, e in sem_map["entities"].items():
        std_pos = e.get("centroid_std_floor_m", 0.0)
        multi_view_audit.append({
            "entity_id": eid,
            "class_name": e["class_name"],
            "observation_count": e["observation_count"],
            "frame_span": e["frame_span"],
            "centroid_std_m": std_pos,
            "converged_under_tolerance": bool(std_pos is not None and std_pos <= 0.25),
        })
        
    audit_summary = {
        "dataset_name": "d455_rgbd_20261010_123443",
        "coordinate_frame": "floor_map (ROS REP-103: X=Forward, Y=Left, Z=Height, Z=0 on floor)",
        "processed_frames": 499,
        "sampling_step": 5,
        "raw_detections_count": 2420,
        "valid_3d_observations_count": 2096,
        "confirmed_entities_count": len(sem_map["entities"]),
        "unconfirmed_candidates_count": len(sem_map["unconfirmed_candidates"]),
        "floor_plane": floor_info,
        "occupancy_grid_stats": grid_stats,
        "multi_view_consistency_audit": multi_view_audit,
    }
    with open(run_dir / "audit_summary.json", "w", encoding="utf-8") as f:
        json.dump(audit_summary, f, indent=2)
        
    STYLE = {
        "chair": {"color": "#1E90FF", "border": "#000080", "name": "Chair"},
        "desk": {"color": "#32CD32", "border": "#006400", "name": "Desk/Table"},
        "table": {"color": "#32CD32", "border": "#006400", "name": "Desk/Table"},
        "door": {"color": "#FF4500", "border": "#8B0000", "name": "Door/Cabinet"},
        "storage box": {"color": "#FFA500", "border": "#D2691E", "name": "Storage Box"},
        "backpack": {"color": "#9370DB", "border": "#4B0082", "name": "Backpack"},
        "handbag": {"color": "#BA55D3", "border": "#4B0082", "name": "Backpack"},
        "laptop": {"color": "#00CED1", "border": "#008B8B", "name": "Laptop"},
        "ceiling fan": {"color": "#DAA520", "border": "#B8860B", "name": "Ceiling Fan"},
    }
    DEFAULT_STYLE = {"color": "#A9A9A9", "border": "#696969", "name": "Object"}
    
    # -------------------------------------------------------------
    # FIGURE 1: DEDICATED CLEAN SEMANTIC FLOOR PLAN
    # -------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(14, 13), dpi=250)
    ax.imshow(occ_img, cmap='gray', origin='lower', extent=extent, vmin=0, vmax=255, alpha=0.9)
    ax.plot(traj_floor[:, 0], traj_floor[:, 1], color='#00CCCC', linewidth=2.8, label='Robot Trajectory', zorder=4)
    ax.scatter(traj_floor[0, 0], traj_floor[0, 1], color='#00FF00', edgecolors='black', s=140, marker='o', label='Start (0, 0)', zorder=5)
    ax.scatter(traj_floor[-1, 0], traj_floor[-1, 1], color='#FF0055', edgecolors='black', s=140, marker='X', label='End', zorder=5)

    used_classes = set()
    label_boxes = []

    for eid, e in sem_map["entities"].items():
        cls_name = e["class_name"]
        if cls_name == "ceiling fan":
            continue # Ceiling fan is high overhead, omit from 2D floor obstacle plan
            
        st = STYLE.get(cls_name, DEFAULT_STYLE)
        used_classes.add(cls_name)
        cf = e["centroid_floor"]
        dim = e["dimensions"]

        w_f = max(0.40, min(1.8, dim[0]))
        l_f = max(0.40, min(1.8, dim[1]))

        rect = patches.FancyBboxPatch(
            (cf[0] - w_f / 2.0, cf[1] - l_f / 2.0), w_f, l_f,
            boxstyle="round,pad=0.03",
            facecolor=st["color"],
            edgecolor=st["border"],
            linewidth=2.2,
            alpha=0.52,
            zorder=6
        )
        ax.add_patch(rect)
        ax.scatter(cf[0], cf[1], color=st["border"], s=40, zorder=7)

        lx = cf[0]
        ly = cf[1] + l_f / 2.0 + 0.16
        for bx, by in label_boxes:
            if math.hypot(lx - bx, ly - by) < 0.35:
                ly += 0.22
                lx += 0.12
        label_boxes.append((lx, ly))

        ax.text(
            lx, ly, eid,
            fontsize=9.5, weight='bold', color='black',
            ha='center', va='center',
            bbox=dict(boxstyle='round,pad=0.22', facecolor='white', edgecolor=st["border"], alpha=0.92, linewidth=1.5),
            zorder=8, clip_on=True
        )

    xlim = (-3.8, 4.0)
    ylim = (-2.8, 5.4)
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_aspect('equal')
    ax.grid(True, linestyle=':', alpha=0.55, color='#777777')
    ax.set_xlabel("Floor Coordinate X (meters - Forward)", fontsize=13, weight='bold')
    ax.set_ylabel("Floor Coordinate Y (meters - Left)", fontsize=13, weight='bold')
    ax.set_title("Reconstructed Semantic Floor Plan (Metric Occupancy + Confirmed Objects)", fontsize=15, weight='bold', pad=14)

    legend_elements = [
        plt.Line2D([0], [0], color='#00CCCC', lw=2.8, label='Robot Trajectory'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='#00FF00', markeredgecolor='k', markersize=11, label='Start (0, 0)'),
        plt.Line2D([0], [0], marker='X', color='w', markerfacecolor='#FF0055', markeredgecolor='k', markersize=11, label='End'),
    ]
    for cls in sorted(used_classes):
        st = STYLE.get(cls, DEFAULT_STYLE)
        legend_elements.append(
            patches.Patch(facecolor=st["color"], edgecolor=st["border"], alpha=0.6, label=st["name"])
        )
    ax.legend(handles=legend_elements, loc='upper right', framealpha=0.95, fontsize=10.5, title="Legend", title_fontsize=11.5)

    clean_plan_path = run_dir / "semantic_floor_plan.png"
    plt.tight_layout()
    plt.savefig(clean_plan_path)
    plt.close()
    print(f"Saved clean floor plan to: {clean_plan_path}")

    # -------------------------------------------------------------
    # FIGURE 2: MASTER 4-PANEL VISUALIZATION
    # -------------------------------------------------------------
    fig2 = plt.figure(figsize=(19, 13), dpi=220)

    # Panel 1: 3D Scene in Floor Coordinates
    ax1 = fig2.add_subplot(2, 2, 1, projection='3d')
    ax1.plot(traj_floor[:, 0], traj_floor[:, 1], traj_floor[:, 2], color='#000080', linewidth=2.0, label='Camera Path')
    ax1.scatter(traj_floor[0, 0], traj_floor[0, 1], traj_floor[0, 2], color='lime', s=80, label='Start')
    ax1.scatter(traj_floor[-1, 0], traj_floor[-1, 1], traj_floor[-1, 2], color='red', s=80, label='End')

    for eid, e in sem_map["entities"].items():
        st = STYLE.get(e["class_name"], DEFAULT_STYLE)
        cf = e["centroid_floor"]
        ax1.scatter(cf[0], cf[1], cf[2], color=st["color"], edgecolors=st["border"], s=95, alpha=0.9)
        ax1.text(cf[0], cf[1], cf[2] + 0.15, eid, fontsize=8, weight='bold', color='black')

    ax1.set_xlabel("X (Forward, m)")
    ax1.set_ylabel("Y (Left, m)")
    ax1.set_zlabel("Z (Height, m)")
    ax1.set_title("3D Scene in floor_map Frame (ROS REP-103)", fontsize=11.5, weight='bold')
    ax1.legend(loc='upper right', fontsize=8.5)

    # Panel 2: Semantic Floor Plan
    ax2 = fig2.add_subplot(2, 2, 2)
    ax2.imshow(occ_img, cmap='gray', origin='lower', extent=extent, vmin=0, vmax=255, alpha=0.85)
    ax2.plot(traj_floor[:, 0], traj_floor[:, 1], color='#00CCCC', linewidth=2.2, label='Trajectory')
    ax2.scatter(traj_floor[0, 0], traj_floor[0, 1], color='#00FF00', s=80, marker='o', label='Start')
    ax2.scatter(traj_floor[-1, 0], traj_floor[-1, 1], color='#FF0055', s=80, marker='X', label='End')

    label_boxes2 = []
    for eid, e in sem_map["entities"].items():
        if e["class_name"] == "ceiling fan":
            continue
        st = STYLE.get(e["class_name"], DEFAULT_STYLE)
        cf = e["centroid_floor"]
        circ = patches.Circle((cf[0], cf[1]), radius=0.28, facecolor=st["color"], edgecolor=st["border"], alpha=0.6, linewidth=1.8)
        ax2.add_patch(circ)
        lx, ly = cf[0], cf[1] + 0.35
        for bx, by in label_boxes2:
            if math.hypot(lx - bx, ly - by) < 0.35:
                ly += 0.22
                lx += 0.12
        label_boxes2.append((lx, ly))
        ax2.text(lx, ly, eid, fontsize=8, weight='bold', ha='center', va='center',
                 bbox=dict(boxstyle='square,pad=0.18', facecolor='white', edgecolor=st["border"], alpha=0.88), clip_on=True)

    ax2.set_xlim(xlim)
    ax2.set_ylim(ylim)
    ax2.set_aspect('equal')
    ax2.grid(True, linestyle=':', alpha=0.5)
    ax2.set_title("Top-Down Semantic Floor Plan", fontsize=11.5, weight='bold')
    ax2.legend(loc='lower left', fontsize=8.5)

    # Panel 3: Metric 2D Occupancy Grid
    ax3 = fig2.add_subplot(2, 2, 3)
    ax3.imshow(occ_img, cmap='gray', origin='lower', extent=extent, vmin=0, vmax=255)
    ax3.plot(traj_floor[:, 0], traj_floor[:, 1], color='cyan', linewidth=2.0, label='Robot Trajectory')
    ax3.scatter(traj_floor[0, 0], traj_floor[0, 1], color='lime', s=70, label='Start')
    ax3.scatter(traj_floor[-1, 0], traj_floor[-1, 1], color='magenta', s=70, label='End')
    ax3.set_xlim(xlim)
    ax3.set_ylim(ylim)
    ax3.set_aspect('equal')
    ax3.grid(True, linestyle=':', alpha=0.45)
    ax3.set_title("Raw Metric 2D Occupancy Grid (White: Free, Black: Obstacle)", fontsize=11.5, weight='bold')
    ax3.legend(loc='lower left', fontsize=8.5)

    # Panel 4: Physical Clearances Between Key Furniture
    ax4 = fig2.add_subplot(2, 2, 4)
    sample_pairs = pairwise_distances[:min(6, len(pairwise_distances))]
    p_labels = [f"{p['entity_a']}-\n{p['entity_b']}" for p in sample_pairs]
    p_dists = [p['centroid_distance_2d_m'] for p in sample_pairs]
    p_clears = [p['footprint_clearance_m'] for p in sample_pairs]
    p_errs = [p['uncertainty_std_m'] for p in sample_pairs]

    x_idx = np.arange(len(p_labels))
    w_bar = 0.35
    ax4.bar(x_idx - w_bar/2, p_dists, w_bar, label='Centroid Dist (m)', color='#1E90FF')
    ax4.bar(x_idx + w_bar/2, p_clears, w_bar, yerr=p_errs, capsize=4, label='Footprint Clearance (m)', color='#FF8C00')
    ax4.set_xticks(x_idx)
    ax4.set_xticklabels(p_labels, fontsize=8.5)
    ax4.set_ylabel("Distance (m)", fontsize=10.5)
    ax4.set_title("Furniture Physical Clearance (with Derived Uncertainty)", fontsize=11.5, weight='bold')
    ax4.legend(loc='upper right', fontsize=8.5)
    ax4.grid(True, linestyle=':', alpha=0.5)

    plt.tight_layout()
    master_path = run_dir / "mapping_visualization.png"
    plt.savefig(master_path)
    plt.close()
    print(f"Saved master visualization to: {master_path}")

    # Generate Markdown Audit Report
    report_md = f"""# Offline Semantic Mapping & Geometry Audit Report (Run 2)

**Dataset**: `{audit_summary['dataset_name']}`
**Coordinate Frame**: `{audit_summary['coordinate_frame']}`

---

## 1. Coordinate Provenance & Unified Transform
- **Frame Convention**: ROS REP-103 standard (`floor_map` frame).
  - Horizontal plane: $X$ (forward), $Y$ (left)
  - Vertical: $Z$ (height above floor, $Z=0$ is floor plane)
- **Fitted Floor Plane**:
  - Upward Normal: `[{floor_info['upward_normal'][0]:.4f}, {floor_info['upward_normal'][1]:.4f}, {floor_info['upward_normal'][2]:.4f}]`
  - Plane Offset $d$: `{floor_info['plane_distance_d']:.4f} m`
  - Inliers: `{floor_info['inlier_count']}/{floor_info['total_candidate_points']} ({floor_info['inlier_percentage']}%)`
  - Plane Residual: Mean `{floor_info['residual_mean_m']*1000:.1f} mm`, Std `{floor_info['residual_std_m']*1000:.1f} mm`
  - Camera Height at Origin: `{floor_info['camera_start_height_above_floor_m']:.3f} m`
- **Unified Transform Matrix (`T_floor_world`)**:
  - Saved in `floor_plane.json`. Every camera pose, object point cloud, bounding box, footprint, and grid raycast is transformed through this single matrix.

---

## 2. Perception & Provenance Summary
- **Total Processed Frames**: `499` (sampled every 5 frames)
- **Raw Detections**: `{audit_summary['raw_detections_count']}`
- **Valid 3D Observations**: `{audit_summary['valid_3d_observations_count']}`
- **Sequential Local Tracking**: Every detection and observation records its `local_track_id` and assigned `global_entity_id`.

---

## 3. Persistent Object Registry & Confirmation
- **Confirmed World Objects**: `{len(sem_map['entities'])}`
- **Quarantined Candidates**: `{len(sem_map['unconfirmed_candidates'])}` (filtered out by strict confirmation policy or dynamic flag)

| Entity ID | Class | Observations | Frame Span | Dimensions (WxLxH m) | Centroid Spread $\\sigma$ (m) | Status |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
"""
    for eid, e in sorted(sem_map["entities"].items()):
        report_md += f"| `{eid}` | {e['class_name']} | {e['observation_count']} | {e['frame_span']} | `{e['dimensions'][0]:.2f}x{e['dimensions'][1]:.2f}x{e['dimensions'][2]:.2f}` | `{e.get('centroid_std_floor_m', 0.0):.3f}` | **CONFIRMED** |\n"

    for eid, e in sorted(sem_map["unconfirmed_candidates"].items()):
        report_md += f"| `{eid}` | {e['class_name']} | {e['observation_count']} | {e['frame_span']} | - | - | QUARANTINED ({e.get('quarantine_reason', 'UNCONFIRMED')}) |\n"

    report_md += f"""
---

## 4. Occupancy Grid Statistics
- **Total Cells**: `{grid_stats['total_cells']}` (`{grid_stats['width_m']:.1f} x {grid_stats['height_m']:.1f} m` @ `{grid_stats['resolution_m']} m`)
- **Free Cells**: `{grid_stats['free_cells']} ({grid_stats['free_pct']}%)`
- **Occupied Obstacle Cells**: `{grid_stats['occupied_cells']} ({grid_stats['occupied_pct']}%)`
- **Unknown Cells**: `{grid_stats['unknown_cells']} ({grid_stats['unknown_pct']}%)`
- **Dynamic Objects**: Persons were suppressed from the static occupancy grid.

---

## 5. Pairwise Distances & Derived Uncertainties
- Pairwise distances between all confirmed static objects computed with **measurement-derived uncertainty** $\\sigma = \\sqrt{{\\sigma_A^2 + \\sigma_B^2 + \\sigma_{{\\text{{depth}}}}^2}}$.
"""
    with open(run_dir / "mapping_audit_report.md", "w", encoding="utf-8") as f:
        f.write(report_md)
    print("Report written to:", run_dir / "mapping_audit_report.md")


if __name__ == "__main__":
    main()
