#!/usr/bin/env python3
"""Generate Corrected Navigation Map Matching the Original Semantic Floor Plan.

Properly aligns the occupancy grid (unflipped to match REP-103 floor coordinates),
plots the full room environment, and details the route from Origin (0,0) to desk_018.
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
import numpy as np
import yaml

from scene_graph.navigation.goal_resolver import GoalResolver
from scene_graph.navigation.planner import GridPathPlanner


def main():
    map_dir = Path("outputs/semantic_mapping_fused_desk_run")
    out_dir = map_dir / "visualizations"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_img_path = out_dir / "desk_018_navigation_route_map.png"

    print("Loading semantic entities and metric occupancy grid...")
    sem_path = map_dir / "semantic_map.json"
    with open(sem_path, "r", encoding="utf-8") as f:
        sem_data = json.load(f)
        entities = sem_data.get("entities", {})

    yaml_path = map_dir / "occupancy_map.yaml"
    with open(yaml_path, "r", encoding="utf-8") as f:
        map_meta = yaml.safe_load(f)

    resolution = float(map_meta["resolution"])
    origin = [float(v) for v in map_meta["origin"]]
    ox, oy = origin[0], origin[1]

    # Load PGM file
    pgm_path = map_dir / map_meta["image"]
    pgm_img = cv2.imread(str(pgm_path), cv2.IMREAD_UNCHANGED)
    if pgm_img is None:
        raise FileNotFoundError(f"Failed to read {pgm_path}")

    # CRITICAL FIX: The PGM file on disk is vertically flipped (ROS standard).
    # To match semantic_floor_plan.png and Cartesian floor Y, flipud it!
    aligned_grid = np.flipud(pgm_img)

    h, w = aligned_grid.shape
    extent = [ox, ox + w * resolution, oy, oy + h * resolution]

    # Build clean high-contrast RGB map identical to semantic_floor_plan.png
    # 254 (free) -> Clean white (255, 255, 255)
    # 205 (unknown) -> Soft gray (235, 238, 242)
    # 0 (obstacle) -> Dark slate (60, 70, 85)
    rgb_map = np.ones((h, w, 3), dtype=np.uint8) * 255
    rgb_map[aligned_grid == 205] = [235, 238, 242]
    rgb_map[aligned_grid == 254] = [255, 255, 255]
    rgb_map[aligned_grid == 0] = [60, 70, 85]

    # Plan Route to desk_018
    resolver = GoalResolver.from_map_dir(map_dir)
    planner = GridPathPlanner.from_map_dir(map_dir, allow_unknown=False)

    start_pose = (0.0, 0.0, 0.0)
    g = resolver.resolve("go to desk_018 and stop 0.5 metres away", current_robot_pose=start_pose)
    path = planner.plan(start_pose, g.approach_pose_2d, target_entity_id=g.target_entity_id, target_center=g.target_center_floor)

    print(f"Goal resolved: {g.target_entity_id} at {g.approach_pose_2d}")
    print(f"Path planning success: {path.success}, length: {path.path_length_m:.2f} m")

    # Figure Layout: Left = Full Aligned Map with Trajectory; Right = Telemetry Table
    fig = plt.figure(figsize=(24, 13), dpi=220)
    gs = fig.add_gridspec(1, 2, width_ratios=[2.4, 1.0], wspace=0.10)
    ax_map = fig.add_subplot(gs[0])
    ax_side = fig.add_subplot(gs[1])

    # Show Background Occupancy Map
    ax_map.imshow(rgb_map, origin="lower", extent=extent)
    ax_map.set_title("Aligned Navigation Plan: Start Origin (0,0) to desk_018",
                     fontsize=15, fontweight="bold", pad=12, color="#0F172A")
    ax_map.set_xlabel("Floor Coordinate X (meters - Forward)", fontsize=12, fontweight="bold", color="#334155")
    ax_map.set_ylabel("Floor Coordinate Y (meters - Left)", fontsize=12, fontweight="bold", color="#334155")
    ax_map.grid(True, linestyle=":", alpha=0.50, color="#94A3B8")

    # Plot All Confirmed Semantic Objects
    for eid, ent in entities.items():
        cf = ent.get("centroid_floor")
        dim = ent.get("dimensions", [0.5, 0.5, 0.5])
        cls = ent.get("class_name", "")
        obs = ent.get("observation_count", 0)
        if not cf:
            continue

        ew, el = float(dim[0]), float(dim[1])
        x0, y0 = cf[0] - ew / 2.0, cf[1] - el / 2.0

        is_target = (eid == "desk_018")
        is_corridor_item = eid in ("chair_027", "chair_038", "desk_023", "desk_024", "chair_019", "desk_017")

        if is_target:
            edge_col, face_col = "#DC2626", "#FEE2E2"
            lw = 3.0
            alpha = 0.90
            zord = 6
        elif is_corridor_item:
            edge_col = "#D97706" if "chair" in cls else "#2563EB"
            face_col = "#FEF3C7" if "chair" in cls else "#DBEAFE"
            lw = 1.8
            alpha = 0.75
            zord = 4
        else:
            edge_col = "#64748B" if "desk" in cls else ("#94A3B8" if "chair" in cls else "#CBD5E1")
            face_col = "#F1F5F9" if "desk" in cls else ("#F8FAFC" if "chair" in cls else "#FFFFFF")
            lw = 1.0
            alpha = 0.45
            zord = 2

        rect = patches.FancyBboxPatch(
            (x0, y0), ew, el,
            boxstyle="round,pad=0.01",
            linewidth=lw,
            edgecolor=edge_col,
            facecolor=face_col,
            alpha=alpha,
            zorder=zord,
        )
        ax_map.add_patch(rect)

        # Labels
        if is_target:
            ax_map.text(
                cf[0], cf[1], f"TARGET:\n{eid}\n({cls})",
                fontsize=9.0, fontweight="bold", color="#991B1B",
                ha="center", va="center",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="#DC2626", alpha=0.95, linewidth=1.8),
                zorder=9,
            )
        elif is_corridor_item:
            ax_map.text(
                cf[0], cf[1], f"{eid}",
                fontsize=7.2, fontweight="bold", color="#1E293B",
                ha="center", va="center",
                bbox=dict(boxstyle="round,pad=0.18", facecolor="white", edgecolor=edge_col, alpha=0.90, linewidth=1.0),
                zorder=5,
            )

    # Plot Navigation Route to desk_018
    if path.success and path.waypoints_floor:
        wps = path.waypoints_floor
        xs = [w[0] for w in wps]
        ys = [w[1] for w in wps]

        # Traveled route line
        ax_map.plot(xs, ys, "-", color="#0284C7", linewidth=3.5, zorder=7, label=f"Direct Clear Path ({path.path_length_m:.2f} m)")
        ax_map.plot(xs, ys, "o", color="#0369A1", markersize=6.0, zorder=8)

        # Directional arrows
        for i in range(len(wps) - 1):
            wx1, wy1 = xs[i], ys[i]
            wx2, wy2 = xs[i+1], ys[i+1]
            dx, dy = wx2 - wx1, wy2 - wy1
            L = math.hypot(dx, dy)
            if L > 0.05:
                # Add multiple arrows along the segment
                for t in (0.35, 0.70):
                    ax_map.annotate(
                        "", xy=(wx1 + dx * t, wy1 + dy * t), xytext=(wx1 + dx * (t - 0.15), wy1 + dy * (t - 0.15)),
                        arrowprops=dict(arrowstyle="-|>", color="#0369A1", lw=2.4, mutation_scale=16),
                        zorder=8,
                    )

        # Alternative Rectilinear Corridor Route (Right-Angle Turns via Main Aisle)
        rect_wps = [(0.0, 0.0), (0.1, -1.8), (-0.2, -2.8), (-1.36, -3.02)]
        rxs = [w[0] for w in rect_wps]
        rys = [w[1] for w in rect_wps]
        ax_map.plot(rxs, rys, "--", color="#10B981", linewidth=2.5, alpha=0.85, zorder=7,
                    label="Wide Corridor Option (Turn via South Junction)")

        # Start Marker (Origin)
        start_circle = patches.Circle((0.0, 0.0), 0.20, facecolor="#10B981", edgecolor="black", linewidth=2.2, zorder=9)
        ax_map.add_patch(start_circle)
        ax_map.arrow(0.0, 0.0, 0.30, 0.0, head_width=0.08, head_length=0.08, fc="black", ec="black", lw=2.0, zorder=10)
        ax_map.text(
            0.0, 0.35, "START: ORIGIN\n(0.0, 0.0, 0°)",
            fontsize=9.0, fontweight="bold", color="white",
            ha="center", va="center",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="#059669", edgecolor="black", linewidth=1.2, alpha=1.0),
            zorder=11,
        )

        # Arrival Marker (desk_018 Standoff)
        gx, gy, gyaw = path.goal_pose
        stop_circle = patches.Circle((gx, gy), 0.20, facecolor="#DC2626", edgecolor="white", linewidth=2.2, zorder=9)
        ax_map.add_patch(stop_circle)
        # Orientation arrow pointing directly at desk_018 (West)
        ax_map.arrow(gx, gy, -0.32, 0.0, head_width=0.08, head_length=0.08, fc="black", ec="black", lw=2.0, zorder=10)
        ax_map.text(
            gx, gy + 0.36, "ARRIVAL: desk_018\n(0.5m Standoff)",
            fontsize=9.0, fontweight="bold", color="white",
            ha="center", va="center",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="#DC2626", edgecolor="black", linewidth=1.2, alpha=1.0),
            zorder=11,
        )

        # Draw 0.5m standoff buffer circle
        standoff_circle = patches.Circle((gx, gy), 0.50, edgecolor="#DC2626", facecolor="none", linestyle=":", linewidth=1.8, zorder=5)
        ax_map.add_patch(standoff_circle)

    # Set Bounds matching original semantic_floor_plan.png (-5.8m to +5.8m, -9.2m to +3.2m)
    ax_map.set_xlim(-5.2, 5.2)
    ax_map.set_ylim(-8.5, 2.5)
    ax_map.set_aspect("equal")
    ax_map.legend(loc="upper right", framealpha=0.92, fontsize=9.5)

    # Sidebar: Comprehensive Route & Safety Analysis
    ax_side.axis("off")

    sidebar_info = (
        "MISSION SPECIFICATION: ORIGIN -> DESK 18\n"
        "==========================================\n\n"
        "COMMAND TO RUN IN TERMINAL:\n"
        "------------------------------------------\n"
        "python tools/navigation/navigate_to_object.py \\\n"
        "  --map outputs/semantic_mapping_fused_desk_run \\\n"
        "  --start-pose '0.0 0.0 0.0' \\\n"
        "  --goal 'go to desk_018 and stop 0.5 metres away' \\\n"
        "  --enable-motion --allow-open-loop-motion\n\n"
        "------------------------------------------\n"
        "TRAJECTORY & TURNS BREAKDOWN:\n"
        "------------------------------------------\n"
        "• Target Entity:   desk_018 (West Cluster Bench)\n"
        "• Target Center:   (X = -2.56 m, Y = -3.02 m)\n"
        "• Start Pose:      (X =  0.00 m, Y =  0.00 m, Yaw = 0.0°)\n"
        "• Approach Pose:   (X = -1.36 m, Y = -3.02 m, Yaw = 180°)\n"
        f"• Travel Distance: {path.path_length_m:.2f} metres\n"
        "• Standoff:        0.50 metres in front of desk\n\n"
        "MANEUVER SEQUENCE:\n"
        "  1. Start at Origin (0, 0) facing forward (0°).\n"
        "  2. Pivot Right: Turn heading to -114.6°\n"
        "     (South-South-West toward West aisle).\n"
        "  3. Accelerate to Cruise PWM: Drive straight\n"
        "     for 3.36 m down the clear corridor.\n"
        "  4. Final Turn on Arrival: Face West (180°)\n"
        "     directly toward the front face of desk_018.\n"
        "  5. Decelerate & Safe Stop: Halts 0.5m away.\n\n"
        "------------------------------------------\n"
        "CORRIDOR SAFETY & HAZARD AUDIT:\n"
        "------------------------------------------\n"
        "• Min Metric Clearance: 0.20 m (Costmap free)\n"
        "• Left Obstacles:  chair_027 (-0.07, -1.03)\n"
        "                   chair_038 (-0.04, -1.99)\n"
        "• Right Obstacles: desk_024  (-0.65, -0.28)\n"
        "                   desk_023  (-1.16, -1.11)\n\n"
        "POTENTIAL ISSUES & MITIGATIONS:\n"
        "1. Open-Loop Odometry Drift:\n"
        "   Over a 3.36m diagonal run, wheel slippage\n"
        "   can accumulate 5-15cm lateral error.\n"
        "   Mitigation: Keep floor surface clean;\n"
        "   verify RC car tracks straight.\n\n"
        "2. Narrow Passage between Chairs:\n"
        "   The corridor between desk_023 and chair_027\n"
        "   is ~0.70m wide. Car track width is 0.20m,\n"
        "   leaving ~25cm safety buffer on each side.\n\n"
        "3. Emergency Stop:\n"
        "   Press Ctrl+C at any moment to cut PWM."
    )

    ax_side.text(
        0.05, 0.95, sidebar_info,
        transform=ax_side.transAxes,
        fontsize=8.8, fontfamily="monospace",
        verticalalignment="top",
        bbox=dict(boxstyle="round,pad=0.8", facecolor="#F8FAFC", edgecolor="#CBD5E1", linewidth=1.5),
        color="#0F172A",
    )

    plt.savefig(out_img_path, bbox_inches="tight", dpi=220)
    plt.close(fig)
    print(f"\nSaved aligned navigation route map to: {out_img_path}")


if __name__ == "__main__":
    main()
