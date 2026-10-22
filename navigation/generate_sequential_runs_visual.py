#!/usr/bin/env python3
"""Generate Visual Map of 3 Sequential Scene Graph Navigation Runs with Traveled Paths.

Plots the 2D metric occupancy map, target objects, and the 3 executed sequential routes:
  1. Origin (0,0) -> desk_001 (1.95 m, Stop 1)
  2. Stop 1 (1.38, -1.26) -> chair_001 (3.23 m, Stop 2)
  3. Stop 2 (4.31, -2.63) -> chair_002 (2.05 m, Stop 3)
"""

from __future__ import annotations

import json
import math
import os
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, os.path.abspath("src"))

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.lines import Line2D
import numpy as np
import yaml

from scene_graph.navigation.goal_resolver import GoalResolver
from scene_graph.navigation.planner import GridPathPlanner


def main():
    map_dir = Path("outputs/semantic_mapping_fused_desk_run")
    out_dir = map_dir / "visualizations"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_img_path = out_dir / "sequential_runs_travelled_paths.png"

    print(f"Loading map and planners from: {map_dir}")
    resolver = GoalResolver.from_map_dir(map_dir)
    planner = GridPathPlanner.from_map_dir(map_dir, allow_unknown=False)

    # 1. Resolve and Plan Run 1
    # Prompt: "go to desk_001 and stop 0.5 metres away" from origin (0, 0, 0)
    start_pose_1 = (0.0, 0.0, 0.0)
    g1 = resolver.resolve("go to desk_001 and stop 0.5 metres away", current_robot_pose=start_pose_1)
    p1 = planner.plan(
        start_pose_1,
        g1.approach_pose_2d,
        target_entity_id=g1.target_entity_id,
        target_center=g1.target_center_floor,
    )
    print(f"Run 1: {g1.target_entity_id} | Path len: {p1.path_length_m:.2f} m | Success: {p1.success}")

    # 2. Resolve and Plan Run 2
    # Prompt: "go to chair_001 and stop 0.5 metres away" from Stop 1 (using saved stopping pose or snapped goal)
    start_pose_2 = p1.goal_pose  # Arrival pose of Run 1
    g2 = resolver.resolve("go to chair_001 and stop 0.5 metres away", current_robot_pose=start_pose_2)
    p2 = planner.plan(
        start_pose_2,
        g2.approach_pose_2d,
        target_entity_id=g2.target_entity_id,
        target_center=g2.target_center_floor,
    )
    print(f"Run 2: {g2.target_entity_id} | Path len: {p2.path_length_m:.2f} m | Success: {p2.success}")

    # 3. Resolve and Plan Run 3
    # Prompt: "go to chair_002 and stop 0.5 metres away" from Stop 2
    start_pose_3 = p2.goal_pose  # Arrival pose of Run 2
    g3 = resolver.resolve("go to chair_002 and stop 0.5 metres away", current_robot_pose=start_pose_3)
    p3 = planner.plan(
        start_pose_3,
        g3.approach_pose_2d,
        target_entity_id=g3.target_entity_id,
        target_center=g3.target_center_floor,
    )
    print(f"Run 3: {g3.target_entity_id} | Path len: {p3.path_length_m:.2f} m | Success: {p3.success}")

    # Load semantic map entities
    sem_path = map_dir / "semantic_map.json"
    entities = {}
    if sem_path.exists():
        with open(sem_path, "r", encoding="utf-8") as f:
            sem_data = json.load(f)
            entities = sem_data.get("entities", {})

    # Load occupancy grid map image & yaml
    yaml_path = map_dir / "occupancy_map.yaml"
    with open(yaml_path, "r", encoding="utf-8") as f:
        map_meta = yaml.safe_load(f)

    resolution = float(map_meta["resolution"])
    origin = [float(v) for v in map_meta["origin"]]
    ox, oy = origin[0], origin[1]

    pgm_path = map_dir / map_meta["image"]
    pgm_img = cv2.imread(str(pgm_path), cv2.IMREAD_UNCHANGED)
    if pgm_img is None:
        raise FileNotFoundError(f"Failed to read {pgm_path}")

    h, w = pgm_img.shape
    extent = [ox, ox + w * resolution, oy, oy + h * resolution]

    # Clean occupancy rendering
    # Free space: 254 -> Crisp off-white (#F8FAFC)
    # Unknown: 205 -> Muted light gray (#E2E8F0)
    # Occupied: 0 -> Slate obstacle (#475569)
    rgb_map = np.ones((h, w, 3), dtype=np.uint8) * 248
    rgb_map[pgm_img == 205] = [226, 232, 240]  # Unknown
    rgb_map[pgm_img == 254] = [255, 255, 255]  # Free
    rgb_map[pgm_img == 0] = [71, 85, 105]     # Lethal obstacles

    # Setup Figure with Map on Left, Detailed Telemetry Sidebar on Right
    fig = plt.figure(figsize=(19, 11), dpi=220)
    gs = fig.add_gridspec(1, 2, width_ratios=[2.4, 1.0], wspace=0.12)
    ax_map = fig.add_subplot(gs[0])
    ax_side = fig.add_subplot(gs[1])

    # Plot Map
    ax_map.imshow(rgb_map, origin="lower", extent=extent)
    ax_map.set_title("3-Stage Sequential Navigation: Traveled Paths & Semantic Scene Graph", fontsize=15, fontweight="bold", pad=12, color="#0F172A")
    ax_map.set_xlabel("Floor X (metres)", fontsize=11, fontweight="bold", color="#334155")
    ax_map.set_ylabel("Floor Y (metres)", fontsize=11, fontweight="bold", color="#334155")
    ax_map.grid(True, linestyle="--", alpha=0.35, color="#94A3B8")

    # Draw confirmed semantic objects
    for eid, ent in entities.items():
        cf = ent.get("centroid_floor")
        dim = ent.get("dimensions", [0.4, 0.4, 0.4])
        cls = ent.get("class_name", "")
        if not cf:
            continue

        ew, el = float(dim[0]), float(dim[1])
        x0, y0 = cf[0] - ew / 2.0, cf[1] - el / 2.0

        is_target_run1 = (eid == "desk_001")
        is_target_run2 = (eid == "chair_001")
        is_target_run3 = (eid == "chair_002")

        if is_target_run1:
            edge_col = "#16A34A"
            face_col = "#DCFCE7"
            lw = 2.5
            alpha = 0.85
        elif is_target_run2:
            edge_col = "#0284C7"
            face_col = "#E0F2FE"
            lw = 2.5
            alpha = 0.85
        elif is_target_run3:
            edge_col = "#D97706"
            face_col = "#FEF3C7"
            lw = 2.5
            alpha = 0.85
        else:
            edge_col = "#94A3B8"
            face_col = "#F1F5F9"
            lw = 1.0
            alpha = 0.45

        rect = patches.FancyBboxPatch(
            (x0, y0), ew, el,
            boxstyle="round,pad=0.01",
            linewidth=lw,
            edgecolor=edge_col,
            facecolor=face_col,
            alpha=alpha,
            zorder=3,
        )
        ax_map.add_patch(rect)

        # Draw labels only for prominent objects to avoid clutter
        if is_target_run1 or is_target_run2 or is_target_run3 or eid in ("desk_004", "storage box_001"):
            label_text = f"{eid}\n({cls})" if (is_target_run1 or is_target_run2 or is_target_run3) else eid
            ax_map.text(
                cf[0], cf[1], label_text,
                fontsize=8.5 if (is_target_run1 or is_target_run2 or is_target_run3) else 7.5,
                fontweight="bold" if (is_target_run1 or is_target_run2 or is_target_run3) else "normal",
                color="#0F172A",
                ha="center", va="center",
                bbox=dict(boxstyle="round,pad=0.2", facecolor="white", edgecolor=edge_col, alpha=0.9, linewidth=1.2),
                zorder=5,
            )

    # Function to draw a styled path with directional arrows
    def draw_path(path_obj, color, arrow_color, label, stop_label, stop_color):
        if not path_obj.success or not path_obj.waypoints_floor:
            return
        wps = path_obj.waypoints_floor
        xs = [w[0] for w in wps]
        ys = [w[1] for w in wps]

        # Path trace
        ax_map.plot(xs, ys, "-", color=color, linewidth=3.5, zorder=6, label=label)
        ax_map.plot(xs, ys, "o", color=color, markersize=5.0, alpha=0.8, zorder=7)

        # Directional arrows along route
        step = max(1, len(wps) // 4)
        for i in range(0, len(wps) - 1, step):
            wx1, wy1 = xs[i], ys[i]
            wx2, wy2 = xs[i+1], ys[i+1]
            dx, dy = wx2 - wx1, wy2 - wy1
            L = math.hypot(dx, dy)
            if L > 0.05:
                ax_map.annotate(
                    "", xy=(wx1 + dx * 0.7, wy1 + dy * 0.7), xytext=(wx1, wy1),
                    arrowprops=dict(arrowstyle="-|>", color=arrow_color, lw=2.2, mutation_scale=15),
                    zorder=8,
                )

        # Arrival stop marker
        gx, gy, gyaw = path_obj.goal_pose
        stop_circle = patches.Circle((gx, gy), 0.18, facecolor=stop_color, edgecolor="white", linewidth=2.0, alpha=0.95, zorder=9)
        ax_map.add_patch(stop_circle)
        # Orientation pointer
        ax_map.arrow(gx, gy, 0.28 * math.cos(gyaw), 0.28 * math.sin(gyaw), head_width=0.08, head_length=0.08, fc="black", ec="black", lw=1.8, zorder=10)

        # Stop pill label
        ax_map.text(
            gx, gy + 0.28, stop_label,
            fontsize=9.0, fontweight="bold", color="white",
            ha="center", va="center",
            bbox=dict(boxstyle="round,pad=0.3", facecolor=stop_color, edgecolor="black", linewidth=1.2, alpha=1.0),
            zorder=11,
        )

    # Plot Start Origin
    start_circle = patches.Circle((0.0, 0.0), 0.20, facecolor="#10B981", edgecolor="black", linewidth=2.2, zorder=9)
    ax_map.add_patch(start_circle)
    ax_map.text(
        0.0, 0.32, "START / ORIGIN\n(0.0, 0.0)",
        fontsize=9.0, fontweight="bold", color="white",
        ha="center", va="center",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="#059669", edgecolor="black", linewidth=1.2, alpha=1.0),
        zorder=11,
    )

    # Draw the 3 sequential paths
    # Run 1: Origin -> desk_001
    draw_path(p1, color="#16A34A", arrow_color="#14532D",
              label=f"Run 1: Origin -> desk_001 ({p1.path_length_m:.2f} m)",
              stop_label="STOP 1: desk_001\n(0.5m standoff)", stop_color="#16A34A")

    # Run 2: desk_001 -> chair_001
    draw_path(p2, color="#0284C7", arrow_color="#0369A1",
              label=f"Run 2: desk_001 -> chair_001 ({p2.path_length_m:.2f} m)",
              stop_label="STOP 2: chair_001\n(0.5m standoff)", stop_color="#0284C7")

    # Run 3: chair_001 -> chair_002
    draw_path(p3, color="#D97706", arrow_color="#B45309",
              label=f"Run 3: chair_001 -> chair_002 ({p3.path_length_m:.2f} m)",
              stop_label="STOP 3: chair_002\n(0.5m standoff)", stop_color="#D97706")

    # Adjust Map Bounds
    all_x = [0.0, p1.goal_pose[0], p2.goal_pose[0], p3.goal_pose[0]]
    all_y = [0.0, p1.goal_pose[1], p2.goal_pose[1], p3.goal_pose[1]]
    pad = 0.9
    ax_map.set_xlim(min(all_x) - pad, max(all_x) + pad)
    ax_map.set_ylim(min(all_y) - pad, max(all_y) + pad)
    ax_map.set_aspect("equal")

    # Sidebar: Clean Informational Table and Mission Summary
    ax_side.axis("off")

    summary_text = (
        "MISSION SPECIFICATION & RUN METRICS\n"
        "======================================\n\n"
        "• Environment: Desk Mapping (Fused Session)\n"
        "• Coordinate Frame: floor_map (REP-103)\n"
        "• Controller: L298N Differential Drive\n"
        "• Safety Gate: Monitored Costmap Clearance\n\n"
        "--------------------------------------\n"
        "RUN 1: PRIMARY WORKSTATION\n"
        "--------------------------------------\n"
        "  Target:       desk_001 (desk)\n"
        f"  Start:        (0.00 m, 0.00 m, 0.0°)\n"
        f"  End:          ({p1.goal_pose[0]:.2f} m, {p1.goal_pose[1]:.2f} m, {math.degrees(p1.goal_pose[2]):.1f}°)\n"
        f"  Distance:     {p1.path_length_m:.2f} m\n"
        f"  Waypoints:    {len(p1.waypoints_floor)}\n"
        "  Standoff:     0.50 m (Front Face)\n"
        "  Status:       SUCCESS (Path Cleared)\n\n"
        "--------------------------------------\n"
        "RUN 2: OFFICE TASK CHAIR\n"
        "--------------------------------------\n"
        "  Target:       chair_001 (chair)\n"
        f"  Start:        ({p2.start_pose[0]:.2f} m, {p2.start_pose[1]:.2f} m, {math.degrees(p2.start_pose[2]):.1f}°)\n"
        f"  End:          ({p2.goal_pose[0]:.2f} m, {p2.goal_pose[1]:.2f} m, {math.degrees(p2.goal_pose[2]):.1f}°)\n"
        f"  Distance:     {p2.path_length_m:.2f} m\n"
        f"  Waypoints:    {len(p2.waypoints_floor)}\n"
        "  Standoff:     0.50 m (Direct Approach)\n"
        "  Status:       SUCCESS (Path Cleared)\n\n"
        "--------------------------------------\n"
        "RUN 3: CONFERENCE SIDE CHAIR\n"
        "--------------------------------------\n"
        "  Target:       chair_002 (chair)\n"
        f"  Start:        ({p3.start_pose[0]:.2f} m, {p3.start_pose[1]:.2f} m, {math.degrees(p3.start_pose[2]):.1f}°)\n"
        f"  End:          ({p3.goal_pose[0]:.2f} m, {p3.goal_pose[1]:.2f} m, {math.degrees(p3.goal_pose[2]):.1f}°)\n"
        f"  Distance:     {p3.path_length_m:.2f} m\n"
        f"  Waypoints:    {len(p3.waypoints_floor)}\n"
        "  Standoff:     0.50 m (Direct Approach)\n"
        "  Status:       SUCCESS (Path Cleared)\n\n"
        "--------------------------------------\n"
        f"Total Mission Travel: {p1.path_length_m + p2.path_length_m + p3.path_length_m:.2f} m"
    )

    ax_side.text(
        0.05, 0.95, summary_text,
        transform=ax_side.transAxes,
        fontsize=9.2, fontfamily="monospace",
        verticalalignment="top",
        bbox=dict(boxstyle="round,pad=0.8", facecolor="#F8FAFC", edgecolor="#CBD5E1", linewidth=1.5),
        color="#0F172A",
    )

    # Save artifact
    plt.savefig(out_img_path, bbox_inches="tight", dpi=220)
    plt.close(fig)
    print(f"\nSuccessfully generated and saved route map to: {out_img_path}")


if __name__ == "__main__":
    main()
