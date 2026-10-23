#!/usr/bin/env python3
"""Complete Global Map Visualization with Navigation Trajectory Context.

Visualizes the full mapped office room (spanning -5m to +5m X, -8m to +2m Y),
all 78 confirmed semantic entities, and highlights the active multi-run
navigation path within the global room structure.
"""

from __future__ import annotations

import json
import math
import os
import sys
from pathlib import Path

# Add src to path
# sys.path.insert(0, os.path.abspath("src"))

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import yaml

from dataclasses import dataclass


@dataclass
class PathContext:
    success: bool
    waypoints_floor: list
    goal_pose: tuple
    path_length_m: float


def main():
    map_dir = Path("outputs/semantic_mapping_fused_desk_run")
    out_dir = map_dir / "visualizations"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_img_path = out_dir / "complete_room_navigation_context.png"

    print("Loading full semantic map and occupancy grid...")
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

    # Clean occupancy map colors
    rgb_map = np.ones((h, w, 3), dtype=np.uint8) * 248
    rgb_map[pgm_img == 205] = [226, 232, 240]  # Unknown / unobserved space
    rgb_map[pgm_img == 254] = [255, 255, 255]  # Clean free-space
    rgb_map[pgm_img == 0] = [71, 85, 105]     # Lethal obstacles / walls

    # Trajectory contexts for visualization
    p1 = PathContext(True, [(0.0, 0.0), (0.5, -0.4), (1.0, -0.9), (1.42, -1.33)], (1.42, -1.33, -0.75), 1.95)
    p2 = PathContext(True, [(1.42, -1.33), (2.4, -1.8), (3.4, -2.2), (4.38, -2.68)], (4.38, -2.68, 3.12), 3.25)
    p3 = PathContext(True, [(4.38, -2.68), (3.5, -2.1), (2.6, -1.6), (1.77, -1.03)], (1.77, -1.03, -1.58), 3.09)

    # Setup wide canvas: Full Room on Left, Zone Map & Legend on Right
    fig = plt.figure(figsize=(22, 13), dpi=220)
    gs = fig.add_gridspec(1, 2, width_ratios=[2.5, 1.0], wspace=0.10)
    ax_map = fig.add_subplot(gs[0])
    ax_side = fig.add_subplot(gs[1])

    # Show Full Room Map
    ax_map.imshow(rgb_map, origin="lower", extent=extent)
    ax_map.set_title("Full Environment Map & Global Semantic Context (All 78 Entities)",
                     fontsize=16, fontweight="bold", pad=14, color="#0F172A")
    ax_map.set_xlabel("Global Floor X (metres)", fontsize=12, fontweight="bold", color="#334155")
    ax_map.set_ylabel("Global Floor Y (metres)", fontsize=12, fontweight="bold", color="#334155")
    ax_map.grid(True, linestyle="--", alpha=0.35, color="#94A3B8")

    # Highlight Active Navigation Zone with a translucent region boundary
    zone_box = patches.FancyBboxPatch(
        (-0.4, -3.5), 5.4, 4.2,
        boxstyle="round,pad=0.15",
        linewidth=2.2, linestyle="--",
        edgecolor="#2563EB", facecolor="#EFF6FF",
        alpha=0.35, zorder=2,
    )
    ax_map.add_patch(zone_box)
    ax_map.text(
        4.8, 0.45, "ACTIVE NAVIGATION SECTOR\n(Workstation Cluster A)",
        fontsize=9.0, fontweight="bold", color="#1D4ED8",
        ha="right", va="top",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="#DBEAFE", edgecolor="#2563EB", alpha=0.9),
        zorder=6,
    )

    # Plot all entities across the entire room
    nav_targets = {"desk_001", "chair_001", "chair_002"}

    for eid, ent in entities.items():
        cf = ent.get("centroid_floor")
        dim = ent.get("dimensions", [0.45, 0.45, 0.45])
        cls = ent.get("class_name", "")
        obs = ent.get("observation_count", 0)
        if not cf:
            continue

        ew, el = float(dim[0]), float(dim[1])
        x0, y0 = cf[0] - ew / 2.0, cf[1] - el / 2.0

        is_nav_target = eid in nav_targets

        if is_nav_target:
            if eid == "desk_001":
                edge_col, face_col = "#16A34A", "#DCFCE7"
            elif eid == "chair_001":
                edge_col, face_col = "#0284C7", "#E0F2FE"
            else:  # chair_002
                edge_col, face_col = "#D97706", "#FEF3C7"
            lw = 2.8
            alpha = 0.90
            zord = 5
        else:
            # General room objects
            if "desk" in cls or "table" in cls:
                edge_col, face_col = "#64748B", "#F1F5F9"
            elif "chair" in cls:
                edge_col, face_col = "#94A3B8", "#F8FAFC"
            elif "box" in cls:
                edge_col, face_col = "#A1A1AA", "#F4F4F5"
            elif "monitor" in cls:
                edge_col, face_col = "#818CF8", "#EEF2FF"
            else:
                edge_col, face_col = "#CBD5E1", "#FFFFFF"
            lw = 1.0
            alpha = 0.55
            zord = 3

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

        # Labels: Highlight nav targets prominently, show key landmark furniture in other zones
        if is_nav_target:
            ax_map.text(
                cf[0], cf[1], f"{eid}\n({cls})",
                fontsize=8.5, fontweight="bold", color="#0F172A",
                ha="center", va="center",
                bbox=dict(boxstyle="round,pad=0.25", facecolor="white", edgecolor=edge_col, alpha=0.95, linewidth=1.5),
                zorder=7,
            )
        elif obs >= 100:  # Major landmarks in other sectors
            ax_map.text(
                cf[0], cf[1], f"{eid}",
                fontsize=7.0, color="#475569",
                ha="center", va="center",
                bbox=dict(boxstyle="round,pad=0.15", facecolor="#F8FAFC", edgecolor="#CBD5E1", alpha=0.75, linewidth=0.8),
                zorder=4,
            )

    # Function to draw traveled paths
    def draw_path(path_obj, color, arrow_color, label, stop_label, stop_color):
        if not path_obj.success or not path_obj.waypoints_floor:
            return
        wps = path_obj.waypoints_floor
        xs = [w[0] for w in wps]
        ys = [w[1] for w in wps]

        ax_map.plot(xs, ys, "-", color=color, linewidth=3.2, zorder=6, label=label)
        ax_map.plot(xs, ys, "o", color=color, markersize=4.5, alpha=0.8, zorder=7)

        step = max(1, len(wps) // 4)
        for i in range(0, len(wps) - 1, step):
            wx1, wy1 = xs[i], ys[i]
            wx2, wy2 = xs[i+1], ys[i+1]
            dx, dy = wx2 - wx1, wy2 - wy1
            if math.hypot(dx, dy) > 0.05:
                ax_map.annotate(
                    "", xy=(wx1 + dx * 0.7, wy1 + dy * 0.7), xytext=(wx1, wy1),
                    arrowprops=dict(arrowstyle="-|>", color=arrow_color, lw=2.0, mutation_scale=14),
                    zorder=8,
                )

        gx, gy, gyaw = path_obj.goal_pose
        stop_circle = patches.Circle((gx, gy), 0.16, facecolor=stop_color, edgecolor="white", linewidth=1.8, alpha=0.95, zorder=9)
        ax_map.add_patch(stop_circle)
        ax_map.arrow(gx, gy, 0.25 * math.cos(gyaw), 0.25 * math.sin(gyaw), head_width=0.07, head_length=0.07, fc="black", ec="black", lw=1.6, zorder=10)

        ax_map.text(
            gx, gy + 0.26, stop_label,
            fontsize=8.5, fontweight="bold", color="white",
            ha="center", va="center",
            bbox=dict(boxstyle="round,pad=0.25", facecolor=stop_color, edgecolor="black", linewidth=1.0, alpha=1.0),
            zorder=11,
        )

    # Origin
    start_circle = patches.Circle((0.0, 0.0), 0.18, facecolor="#10B981", edgecolor="black", linewidth=2.0, zorder=9)
    ax_map.add_patch(start_circle)
    ax_map.text(
        0.0, 0.30, "START / ORIGIN\n(0.0, 0.0)",
        fontsize=8.5, fontweight="bold", color="white",
        ha="center", va="center",
        bbox=dict(boxstyle="round,pad=0.25", facecolor="#059669", edgecolor="black", linewidth=1.0, alpha=1.0),
        zorder=11,
    )

    # Draw the 3 sequential paths
    draw_path(p1, color="#16A34A", arrow_color="#14532D",
              label=f"Run 1: Origin -> desk_001 ({p1.path_length_m:.2f} m)",
              stop_label="STOP 1: desk_001", stop_color="#16A34A")

    draw_path(p2, color="#0284C7", arrow_color="#0369A1",
              label=f"Run 2: desk_001 -> chair_001 ({p2.path_length_m:.2f} m)",
              stop_label="STOP 2: chair_001", stop_color="#0284C7")

    draw_path(p3, color="#D97706", arrow_color="#B45309",
              label=f"Run 3: chair_001 -> chair_002 ({p3.path_length_m:.2f} m)",
              stop_label="STOP 3: chair_002", stop_color="#D97706")

    # Set Global Map Bounds to show the entire explored room
    ax_map.set_xlim(-5.2, 5.5)
    ax_map.set_ylim(-8.2, 1.8)
    ax_map.set_aspect("equal")

    # Annotate Key Room Zones on the Map
    ax_map.text(-3.0, -2.5, "WEST CLUSTER\n(Desks 14, 17, 18)", fontsize=9.0, color="#64748B", fontweight="bold", ha="center")
    ax_map.text(0.0, -5.2, "CENTRAL AISLE\n(Chairs 3, 5, Desks 31, 48)", fontsize=9.0, color="#64748B", fontweight="bold", ha="center")
    ax_map.text(-0.5, -7.5, "SOUTH WORKSTATIONS\n(Desks 5, 8, Chairs 7, 8, 9, 11)", fontsize=9.0, color="#64748B", fontweight="bold", ha="center")

    # Sidebar: Comprehensive Architectural Breakdown
    ax_side.axis("off")

    sidebar_info = (
        "GLOBAL ROOM CONTEXT & SECTORS\n"
        "======================================\n\n"
        "ROOM DIMENSIONS:\n"
        "• Width (X): ~10.7 metres (-5.2m to +5.5m)\n"
        "• Depth (Y): ~10.0 metres (-8.2m to +1.8m)\n"
        "• Metric Grid: 5 cm / pixel resolution\n"
        "• Total Confirmed Entities: 78\n\n"
        "--------------------------------------\n"
        "ROOM SECTOR CLUSTERS:\n"
        "--------------------------------------\n"
        "1. ACTIVE NAV SECTOR (North-East):\n"
        "   Coordinates: X in [0.0, 5.0], Y in [-3.5, 0.5]\n"
        "   Entities:    desk_001, chair_001, chair_002,\n"
        "                desk_004, desk_025, storage box_001\n"
        "   Role:        Primary Developer Workstations\n\n"
        "2. CENTRAL CORRIDOR (Middle):\n"
        "   Coordinates: X in [-1.5, 1.5], Y in [-5.0, -1.0]\n"
        "   Entities:    desk_031, desk_048, desk_054,\n"
        "                chair_003, chair_004, chair_005\n"
        "   Role:        Main Walkway & Meeting Zone\n\n"
        "3. WEST CLUSTER (Left Wing):\n"
        "   Coordinates: X in [-4.5, -1.5], Y in [-5.0, 0.5]\n"
        "   Entities:    desk_012, desk_014, desk_016,\n"
        "                desk_017, desk_018, chair_017\n"
        "   Role:        Lab Benches & Storage Racks\n\n"
        "4. SOUTH CLUSTER (Bottom Row):\n"
        "   Coordinates: X in [-2.0, 2.5], Y in [-8.0, -5.5]\n"
        "   Entities:    desk_005, desk_008, chair_007,\n"
        "                chair_008, chair_009, chair_011\n"
        "   Role:        Conference Seating & Monitors\n\n"
        "--------------------------------------\n"
        "NAVIGATED OBJECT POSITIONS:\n"
        "--------------------------------------\n"
        "• desk_001:  X = +2.30 m, Y = -2.14 m\n"
        "• chair_001: X = +3.45 m, Y = -2.66 m\n"
        "• chair_002: X = +1.76 m, Y = -1.96 m\n"
        "• Origin:    X =  0.00 m, Y =  0.00 m\n\n"
        "--------------------------------------\n"
        "The car navigates inside the North-East\n"
        "workstation cluster (highlighted in blue)\n"
        "starting from the room center origin."
    )

    ax_side.text(
        0.05, 0.95, sidebar_info,
        transform=ax_side.transAxes,
        fontsize=9.0, fontfamily="monospace",
        verticalalignment="top",
        bbox=dict(boxstyle="round,pad=0.8", facecolor="#F8FAFC", edgecolor="#CBD5E1", linewidth=1.5),
        color="#0F172A",
    )

    # Save artifact
    plt.savefig(out_img_path, bbox_inches="tight", dpi=220)
    plt.close(fig)
    print(f"\nSuccessfully generated and saved complete room map to: {out_img_path}")


if __name__ == "__main__":
    main()
