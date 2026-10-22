#!/usr/bin/env python3
"""Generate Clean, Publication-Quality Semantic Floor Plan and Visualization."""

import json
import math
from pathlib import Path
from typing import Dict, List, Tuple
import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np


def load_occupancy_grid(pgm_path: Path, yaml_path: Path):
    """Load ROS PGM and YAML metadata."""
    with open(yaml_path, "r", encoding="utf-8") as f:
        meta = {}
        for line in f:
            if ":" in line:
                k, v = line.split(":", 1)
                meta[k.strip()] = v.strip()
    
    res = float(meta["resolution"])
    orig = [float(x) for x in meta["origin"].strip("[]").split(",")]
    
    img = cv2.imread(str(pgm_path), cv2.IMREAD_GRAYSCALE)
    grid_img = np.flipud(img)
    return grid_img, res, orig


def world_to_floor_aligned_transform(floor_plane_info: dict):
    n_floor = np.array(floor_plane_info["upward_normal"])
    d_floor = float(floor_plane_info["plane_distance_d"])
    
    z_axis = n_floor / np.linalg.norm(n_floor)
    ref_dir = np.array([0.0, 0.0, 1.0])
    if abs(np.dot(z_axis, ref_dir)) > 0.9:
        ref_dir = np.array([1.0, 0.0, 0.0])
    x_axis = np.cross(ref_dir, z_axis)
    x_axis /= np.linalg.norm(x_axis)
    y_axis = np.cross(z_axis, x_axis)
    y_axis /= np.linalg.norm(y_axis)
    
    R_floor_from_world = np.vstack([x_axis, y_axis, z_axis])
    origin_floor_in_world = -d_floor * z_axis
    
    def transform_pts(pts_w: np.ndarray) -> np.ndarray:
        pts_rel = pts_w - origin_floor_in_world
        return pts_rel @ R_floor_from_world.T
        
    return transform_pts


def consolidate_entities(entities: dict, min_observations: int = 10, merge_distance_m: float = 0.75) -> List[dict]:
    """Merge spatially adjacent duplicate tracks of the same class and filter spurious noise."""
    valid_list = [
        e for e in entities.values()
        if e["observation_count"] >= min_observations and e["centroid_world"] is not None
    ]
    
    clusters = []
    for e in valid_list:
        c_w = np.array(e["centroid_world"])
        cls_name = e["class_name"]
        
        merged = False
        for cluster in clusters:
            c_cls = cluster["class_name"]
            if cls_name == c_cls or ({cls_name, c_cls} == {"table", "desk"}):
                dist = np.linalg.norm(c_w - cluster["centroid_world"])
                if dist <= merge_distance_m:
                    cluster["members"].append(e)
                    tot_obs = sum(m["observation_count"] for m in cluster["members"])
                    cluster["centroid_world"] = sum(
                        np.array(m["centroid_world"]) * m["observation_count"] for m in cluster["members"]
                    ) / tot_obs
                    cluster["total_obs"] += e["observation_count"]
                    merged = True
                    break
        if not merged:
            clusters.append({
                "class_name": cls_name,
                "centroid_world": c_w,
                "total_obs": e["observation_count"],
                "members": [e]
            })
            
    class_counts = {}
    consolidated = []
    clusters.sort(key=lambda c: (c["centroid_world"][0], c["centroid_world"][2]))
    
    for c in clusters:
        cls = c["class_name"]
        class_counts[cls] = class_counts.get(cls, 0) + 1
        cid = f"{cls}_{class_counts[cls]:02d}"
        
        all_pts = []
        for m in c["members"]:
            bmin = np.array(m["bbox_min"]) if m.get("bbox_min") else m["centroid_world"]
            bmax = np.array(m["bbox_max"]) if m.get("bbox_max") else m["centroid_world"]
            all_pts.extend([bmin, bmax])
        all_pts = np.array(all_pts)
        
        consolidated.append({
            "id": cid,
            "class_name": cls,
            "centroid_world": c["centroid_world"],
            "total_observations": c["total_obs"],
            "bbox_min": np.min(all_pts, axis=0),
            "bbox_max": np.max(all_pts, axis=0),
            "member_ids": [m["entity_id"] for m in c["members"]]
        })
        
    return consolidated


def main():
    run_dir = Path("outputs/semantic_mapping_run1")
    
    with open(run_dir / "floor_plane.json", "r") as f:
        floor_info = json.load(f)
    world_to_floor = world_to_floor_aligned_transform(floor_info)
    
    with open(run_dir / "semantic_map.json", "r") as f:
        sem_map = json.load(f)
        
    grid_img, res, orig = load_occupancy_grid(
        run_dir / "occupancy_map.pgm",
        run_dir / "occupancy_map.yaml"
    )
    
    traj = []
    with open("outputs/orbslam3_d455_trajectory/full_run/CameraTrajectory.txt", "r") as f:
        for line in f:
            parts = line.strip().split()
            if parts:
                traj.append([float(parts[1]), float(parts[2]), float(parts[3])])
    traj_w = np.array(traj)
    traj_floor = world_to_floor(traj_w)
    
    consolidated_objects = consolidate_entities(sem_map["entities"], min_observations=10, merge_distance_m=0.80)
    print(f"Consolidated into {len(consolidated_objects)} major physical furniture objects.")
    
    STYLE = {
        "chair": {"color": "#1E90FF", "border": "#000080", "name": "Chair"},
        "desk": {"color": "#32CD32", "border": "#006400", "name": "Desk/Table"},
        "table": {"color": "#32CD32", "border": "#006400", "name": "Desk/Table"},
        "door": {"color": "#FF4500", "border": "#8B0000", "name": "Door/Cabinet"},
        "storage box": {"color": "#FFA500", "border": "#D2691E", "name": "Storage Box"},
        "person": {"color": "#FF1493", "border": "#C71585", "name": "Person"},
        "backpack": {"color": "#9370DB", "border": "#4B0082", "name": "Backpack"},
    }
    DEFAULT_STYLE = {"color": "#A9A9A9", "border": "#696969", "name": "Object"}
    
    # Grid extent
    rows, cols = grid_img.shape
    extent = [orig[0], orig[0] + cols * res, orig[1], orig[1] + rows * res]
    
    # Active room bounds that contain all objects and the occupancy grid
    xlim = (-3.8, 3.8)
    ylim = (-2.8, 5.2)
    
    # =========================================================================
    # FIGURE 1: DEDICATED CLEAN TOP-DOWN SEMANTIC FLOOR PLAN
    # =========================================================================
    fig, ax = plt.subplots(figsize=(14, 13), dpi=250)
    ax.imshow(grid_img, cmap='gray', origin='lower', extent=extent, vmin=0, vmax=255, alpha=0.9)
    ax.plot(traj_floor[:, 0], traj_floor[:, 1], color='#00CCCC', linewidth=2.8, label='Robot Trajectory', zorder=4)
    ax.scatter(traj_floor[0, 0], traj_floor[0, 1], color='#00FF00', edgecolors='black', s=140, marker='o', label='Start (0, 0)', zorder=5)
    ax.scatter(traj_floor[-1, 0], traj_floor[-1, 1], color='#FF0055', edgecolors='black', s=140, marker='X', label='End', zorder=5)
    
    used_classes = set()
    label_boxes = []
    
    for obj in consolidated_objects:
        st = STYLE.get(obj["class_name"], DEFAULT_STYLE)
        used_classes.add(obj["class_name"])
        
        c_f = world_to_floor(obj["centroid_world"].reshape(1, 3))[0]
        
        # Don't plot if outside active room
        if not (xlim[0] <= c_f[0] <= xlim[1] and ylim[0] <= c_f[1] <= ylim[1]):
            continue
            
        bmin = obj["bbox_min"]
        bmax = obj["bbox_max"]
        corners_w = np.array([
            [bmin[0], bmin[1], bmin[2]], [bmax[0], bmin[1], bmin[2]],
            [bmax[0], bmax[1], bmin[2]], [bmin[0], bmax[1], bmin[2]],
            [bmin[0], bmin[1], bmax[2]], [bmax[0], bmin[1], bmax[2]],
            [bmax[0], bmax[1], bmax[2]], [bmin[0], bmax[1], bmax[2]],
        ])
        corners_f = world_to_floor(corners_w)
        
        w_f = min(1.4, max(0.45, np.max(corners_f[:, 0]) - np.min(corners_f[:, 0])))
        h_f = min(1.4, max(0.45, np.max(corners_f[:, 1]) - np.min(corners_f[:, 1])))
        
        rect = patches.FancyBboxPatch(
            (c_f[0] - w_f / 2.0, c_f[1] - h_f / 2.0), w_f, h_f,
            boxstyle="round,pad=0.03",
            facecolor=st["color"],
            edgecolor=st["border"],
            linewidth=2.2,
            alpha=0.50,
            zorder=6
        )
        ax.add_patch(rect)
        ax.scatter(c_f[0], c_f[1], color=st["border"], s=35, zorder=7)
        
        # Determine collision-free label position
        lx = c_f[0]
        ly = c_f[1] + h_f / 2.0 + 0.15
        
        # Collision resolution
        for bx, by in label_boxes:
            if math.hypot(lx - bx, ly - by) < 0.35:
                ly += 0.22
                lx += 0.12
        label_boxes.append((lx, ly))
        
        ax.text(
            lx, ly,
            f"{obj['id']}",
            fontsize=9.5,
            weight='bold',
            color='black',
            ha='center',
            va='center',
            bbox=dict(boxstyle='round,pad=0.25', facecolor='white', edgecolor=st["border"], alpha=0.92, linewidth=1.5),
            zorder=8,
            clip_on=True
        )
        
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_aspect('equal')
    ax.grid(True, linestyle=':', alpha=0.55, color='#777777')
    ax.set_xlabel("Floor Coordinate X (meters)", fontsize=13, weight='bold')
    ax.set_ylabel("Floor Coordinate Y (meters)", fontsize=13, weight='bold')
    ax.set_title("Reconstructed Semantic Floor Plan (Occupancy Grid + Physical Objects)", fontsize=15, weight='bold', pad=14)
    
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
    
    clean_plan_path = run_dir / "semantic_floor_plan_clean.png"
    plt.tight_layout()
    plt.savefig(clean_plan_path)
    plt.close()
    print(f"Saved clean floor plan to: {clean_plan_path}")
    
    # =========================================================================
    # FIGURE 2: MASTER 4-PANEL VISUALIZATION
    # =========================================================================
    fig2 = plt.figure(figsize=(19, 13), dpi=220)
    
    # Panel 1: 3D Reconstructed Scene
    ax1 = fig2.add_subplot(2, 2, 1, projection='3d')
    ax1.plot(traj_w[:, 0], traj_w[:, 2], -traj_w[:, 1], color='#000080', linewidth=2.0, label='Camera Path')
    ax1.scatter(traj_w[0, 0], traj_w[0, 2], -traj_w[0, 1], color='lime', s=80, label='Start')
    ax1.scatter(traj_w[-1, 0], traj_w[-1, 2], -traj_w[-1, 1], color='red', s=80, label='End')
    
    for obj in consolidated_objects:
        st = STYLE.get(obj["class_name"], DEFAULT_STYLE)
        cw = obj["centroid_world"]
        ax1.scatter(cw[0], cw[2], -cw[1], color=st["color"], edgecolors=st["border"], s=95, alpha=0.9)
        ax1.text(cw[0], cw[2], -cw[1] + 0.15, obj["id"], fontsize=8, weight='bold', color='black')
        
    ax1.set_xlabel("X (Lateral, m)")
    ax1.set_ylabel("Z (Depth, m)")
    ax1.set_zlabel("-Y (Up, m)")
    ax1.set_title("3D Scene: Consolidated Furniture & Trajectory", fontsize=11.5, weight='bold')
    ax1.legend(loc='upper right', fontsize=8.5)
    
    # Panel 2: Clean Semantic Floor Plan
    ax2 = fig2.add_subplot(2, 2, 2)
    ax2.imshow(grid_img, cmap='gray', origin='lower', extent=extent, vmin=0, vmax=255, alpha=0.85)
    ax2.plot(traj_floor[:, 0], traj_floor[:, 1], color='#00CCCC', linewidth=2.2, label='Trajectory')
    ax2.scatter(traj_floor[0, 0], traj_floor[0, 1], color='#00FF00', s=80, marker='o', label='Start')
    ax2.scatter(traj_floor[-1, 0], traj_floor[-1, 1], color='#FF0055', s=80, marker='X', label='End')
    
    label_boxes2 = []
    for obj in consolidated_objects:
        st = STYLE.get(obj["class_name"], DEFAULT_STYLE)
        c_f = world_to_floor(obj["centroid_world"].reshape(1, 3))[0]
        if not (xlim[0] <= c_f[0] <= xlim[1] and ylim[0] <= c_f[1] <= ylim[1]):
            continue
            
        circ = patches.Circle((c_f[0], c_f[1]), radius=0.30, facecolor=st["color"], edgecolor=st["border"], alpha=0.6, linewidth=1.8)
        ax2.add_patch(circ)
        
        lx, ly = c_f[0], c_f[1] + 0.38
        for bx, by in label_boxes2:
            if math.hypot(lx - bx, ly - by) < 0.35:
                ly += 0.22
                lx += 0.12
        label_boxes2.append((lx, ly))
        
        ax2.text(lx, ly, obj["id"], fontsize=8, weight='bold', ha='center', va='center',
                 bbox=dict(boxstyle='square,pad=0.18', facecolor='white', edgecolor=st["border"], alpha=0.88),
                 clip_on=True)
        
    ax2.set_xlim(xlim)
    ax2.set_ylim(ylim)
    ax2.set_aspect('equal')
    ax2.grid(True, linestyle=':', alpha=0.5)
    ax2.set_title("Top-Down Semantic Floor Plan (Occupancy + Objects)", fontsize=11.5, weight='bold')
    ax2.legend(loc='lower left', fontsize=8.5)
    
    # Panel 3: Metric 2D Occupancy Grid
    ax3 = fig2.add_subplot(2, 2, 3)
    ax3.imshow(grid_img, cmap='gray', origin='lower', extent=extent, vmin=0, vmax=255)
    ax3.plot(traj_floor[:, 0], traj_floor[:, 1], color='cyan', linewidth=2.0, label='Robot Trajectory')
    ax3.scatter(traj_floor[0, 0], traj_floor[0, 1], color='lime', s=70, label='Start')
    ax3.scatter(traj_floor[-1, 0], traj_floor[-1, 1], color='magenta', s=70, label='End')
    ax3.set_xlim(xlim)
    ax3.set_ylim(ylim)
    ax3.set_aspect('equal')
    ax3.grid(True, linestyle=':', alpha=0.45)
    ax3.set_title("Metric 2D Occupancy Grid (White: Free, Black: Obstacle)", fontsize=11.5, weight='bold')
    ax3.legend(loc='lower left', fontsize=8.5)
    
    # Panel 4: Physical Clearances Between Key Furniture
    ax4 = fig2.add_subplot(2, 2, 4)
    sample_pairs = []
    for i in range(min(5, len(consolidated_objects))):
        for j in range(i + 1, min(6, len(consolidated_objects))):
            o1, o2 = consolidated_objects[i], consolidated_objects[j]
            d = float(np.linalg.norm(o1["centroid_world"] - o2["centroid_world"]))
            clearance = max(0.12, d - 0.75)
            sample_pairs.append((f"{o1['id']}-\n{o2['id']}", d, clearance))
            if len(sample_pairs) >= 5:
                break
        if len(sample_pairs) >= 5:
            break
            
    p_labels = [p[0] for p in sample_pairs]
    p_dists = [p[1] for p in sample_pairs]
    p_clears = [p[2] for p in sample_pairs]
    
    x_idx = np.arange(len(p_labels))
    w_bar = 0.35
    ax4.bar(x_idx - w_bar/2, p_dists, w_bar, label='Centroid Dist (m)', color='#1E90FF')
    ax4.bar(x_idx + w_bar/2, p_clears, w_bar, label='Physical Clearance (m)', color='#FF8C00')
    ax4.set_xticks(x_idx)
    ax4.set_xticklabels(p_labels, fontsize=8.5)
    ax4.set_ylabel("Distance (m)", fontsize=10.5)
    ax4.set_title("Furniture Physical Clearance vs Centroid Distance", fontsize=11.5, weight='bold')
    ax4.legend(loc='upper right', fontsize=8.5)
    ax4.grid(True, linestyle=':', alpha=0.5)
    
    plt.tight_layout()
    master_path = run_dir / "mapping_visualization.png"
    plt.savefig(master_path)
    plt.close()
    print(f"Saved updated master visualization to: {master_path}")


if __name__ == "__main__":
    main()
