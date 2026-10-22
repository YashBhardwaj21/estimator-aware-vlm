"""A* Grid Path Planner and Route Visualizer for Differential-Drive Robot Navigation.

Finds collision-free paths from current robot pose to goal pose in the floor-aligned costmap.
Generates publication-quality route visualizations and checks safety clearances.
"""

from __future__ import annotations

import heapq
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np


@dataclass
class PlannedPath:
    """Complete planned path and trajectory metrics."""
    success: bool
    waypoints_floor: List[Tuple[float, float, float]] = field(default_factory=list)  # [(x, y, yaw)]
    path_length_m: float = 0.0
    turn_points_count: int = 0
    start_pose: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    goal_pose: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    original_goal_pose: Optional[Tuple[float, float, float]] = None
    endpoint_adjustment_m: float = 0.0
    min_clearance_m: float = 0.0
    is_validated: bool = False
    target_entity_id: Optional[str] = None
    rejection_reason: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "success": self.success,
            "path_length_m": round(self.path_length_m, 3),
            "turn_points_count": self.turn_points_count,
            "waypoints_count": len(self.waypoints_floor),
            "start_pose": {"x": self.start_pose[0], "y": self.start_pose[1], "yaw_rad": self.start_pose[2]},
            "goal_pose": {"x": self.goal_pose[0], "y": self.goal_pose[1], "yaw_rad": self.goal_pose[2]},
            "original_goal_pose": {
                "x": self.original_goal_pose[0],
                "y": self.original_goal_pose[1],
                "yaw_rad": self.original_goal_pose[2],
            } if self.original_goal_pose else None,
            "endpoint_adjustment_m": round(self.endpoint_adjustment_m, 4),
            "min_clearance_m": round(self.min_clearance_m, 4),
            "is_validated": self.is_validated,
            "target_entity_id": self.target_entity_id,
            "waypoints_floor": [
                {"x": round(w[0], 3), "y": round(w[1], 3), "yaw_rad": round(w[2], 3)}
                for w in self.waypoints_floor
            ],
            "rejection_reason": self.rejection_reason,
        }


class GridPathPlanner:
    """A* Path Planner on 2D costmaps following ROS REP-103 convention."""

    def __init__(
        self,
        costmap_grid: np.ndarray,
        resolution_m: float,
        origin_xy_m: Tuple[float, float],
        occupied_threshold: int = 100,
        raw_map: Optional[np.ndarray] = None,
        map_dir: Optional[Path] = None,
    ):
        """
        costmap_grid: 2D numpy array where:
          0 = free space
          255 (or > occupied_threshold) = lethal obstacle / inflated collision zone
        origin_xy_m: (min_x_m, min_y_m) world coordinate of grid[0, 0]
        """
        self.grid = costmap_grid
        self.raw_map = raw_map
        self.height, self.width = costmap_grid.shape
        self.res = resolution_m
        self.ox, self.oy = origin_xy_m
        self.occ_thresh = occupied_threshold
        self.map_dir = map_dir

    @classmethod
    def from_map_dir(cls, map_dir: Path, allow_unknown: bool = False) -> "GridPathPlanner":
        map_dir = Path(map_dir)
        # Prefer inflated navigation costmap if present
        costmap_p = map_dir / "occupancy_costmap.pgm"
        meta_p = map_dir / "occupancy_costmap.yaml"
        if not costmap_p.exists():
            costmap_p = map_dir / "navigation_costmap.pgm"
            meta_p = map_dir / "navigation_costmap.yaml"
        if not costmap_p.exists():
            costmap_p = map_dir / "occupancy_map.pgm"
            meta_p = map_dir / "occupancy_map.yaml"
        if not costmap_p.exists():
            costmap_p = map_dir / "map.pgm"
            meta_p = map_dir / "map.yaml"

        if not costmap_p.exists():
            raise FileNotFoundError(f"No costmap or occupancy map found in {map_dir}")

        img = cv2.imread(str(costmap_p), cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise ValueError(f"Failed to read costmap image from {costmap_p}")

        # Parse YAML
        import yaml
        with open(meta_p, "r", encoding="utf-8") as f:
            ydata = yaml.safe_load(f)
        res = float(ydata.get("resolution", 0.05))
        origin = ydata.get("origin", [0.0, 0.0, 0.0])
        ox, oy = float(origin[0]), float(origin[1])

        # In ROS PGM, images are stored with origin at bottom-left, so rows are flipped.
        # Flip vertically so row 0 corresponds to y_min (origin_y)
        img_aligned = np.flipud(img)

        # Base raw map for visualization
        base_map_p = map_dir / "map.pgm"
        raw_base = None
        if base_map_p.exists():
            b_img = cv2.imread(str(base_map_p), cv2.IMREAD_GRAYSCALE)
            if b_img is not None:
                raw_base = np.flipud(b_img)
        if raw_base is None:
            raw_base = img_aligned.copy()

        # In ROS PGM: 0=occupied (black), 205=unknown (grey), 254=free (white)
        # Safe Navigation: Unknown cells must be blocked by default unless explicit exploration mode
        obstacle_mask = np.zeros_like(img_aligned, dtype=np.uint8)
        if allow_unknown:
            # Only lethal obstacles blocked
            obstacle_mask[img_aligned < 128] = 255
        else:
            # Both occupied (<128) and unknown (128-249) blocked for safe navigation
            obstacle_mask[img_aligned < 250] = 255

        return cls(obstacle_mask, res, (ox, oy), raw_map=raw_base, map_dir=map_dir)

    def world_to_grid(self, x: float, y: float) -> Tuple[int, int]:
        gx = int(math.floor((x - self.ox) / self.res))
        gy = int(math.floor((y - self.oy) / self.res))
        return gx, gy

    def grid_to_world(self, gx: int, gy: int) -> Tuple[float, float]:
        x = self.ox + (gx + 0.5) * self.res
        y = self.oy + (gy + 0.5) * self.res
        return x, y

    def is_free(self, gx: int, gy: int) -> bool:
        if 0 <= gx < self.width and 0 <= gy < self.height:
            return self.grid[gy, gx] < self.occ_thresh
        return False

    def find_nearest_free_cell(self, gx: int, gy: int, max_radius: int = 5) -> Optional[Tuple[int, int]]:
        """If goal is slightly inside an inflated cell border, find closest reachable free cell."""
        if self.is_free(gx, gy):
            return gx, gy
        for r in range(1, max_radius + 1):
            for dx in range(-r, r + 1):
                for dy in range(-r, r + 1):
                    if abs(dx) == r or abs(dy) == r:
                        nx, ny = gx + dx, gy + dy
                        if self.is_free(nx, ny):
                            return nx, ny
    def inject_obstacle(
        self,
        world_x: float,
        world_y: float,
        radius_m: float = 0.20,
        inflation_radius_m: float = 0.15,
    ) -> int:
        """Dynamically insert an observed obstacle into the costmap grid for real-time replanning."""
        cx, cy = self.world_to_grid(world_x, world_y)
        inf_cells = max(1, int(math.ceil((radius_m + inflation_radius_m) / self.res)))

        cells_modified = 0
        for dy in range(-inf_cells, inf_cells + 1):
            for dx in range(-inf_cells, inf_cells + 1):
                dist_m = math.hypot(dx, dy) * self.res
                gx, gy = cx + dx, cy + dy
                if 0 <= gx < self.width and 0 <= gy < self.height:
                    if dist_m <= radius_m:
                        self.grid[gy, gx] = 255  # Lethal obstacle
                        cells_modified += 1
                    elif dist_m <= (radius_m + inflation_radius_m):
                        self.grid[gy, gx] = max(int(self.grid[gy, gx]), self.occ_thresh + 10)  # Inflated buffer
                        cells_modified += 1
        return cells_modified

    def plan(
        self,
        start_pose: Tuple[float, float, float],
        goal_pose: Tuple[float, float, float],
        target_entity_id: Optional[str] = None,
        allow_goal_snap: bool = False,
        max_endpoint_adjustment_m: float = 0.10,
        robot_radius_m: Optional[float] = None,
        target_center: Optional[Tuple[float, float, float]] = None,
    ) -> PlannedPath:
        """Execute A* search on costmap with strict endpoint adjustment limits and footprint validation."""
        sx, sy, syaw = start_pose
        gx, gy, gyaw = goal_pose
        original_goal_pose = goal_pose

        sgx, sgy = self.world_to_grid(sx, sy)
        ggx, ggy = self.world_to_grid(gx, gy)
        snap_dist = 0.0

        # Check and resolve start cell
        if not self.is_free(sgx, sgy):
            free_start = self.find_nearest_free_cell(sgx, sgy, max_radius=3)
            if not free_start:
                return PlannedPath(
                    success=False,
                    start_pose=start_pose,
                    goal_pose=goal_pose,
                    original_goal_pose=original_goal_pose,
                    rejection_reason=f"Start position ({sx:.2f}, {sy:.2f}) is in an impassable collision zone",
                )
            sgx, sgy = free_start

        # Check and resolve goal cell with strict adjustment tolerance
        if not self.is_free(ggx, ggy):
            if allow_goal_snap:
                free_goal = self.find_nearest_free_cell(ggx, ggy, max_radius=5)
                if not free_goal:
                    return PlannedPath(
                        success=False,
                        start_pose=start_pose,
                        goal_pose=goal_pose,
                        original_goal_pose=original_goal_pose,
                        rejection_reason=f"Goal position ({gx:.2f}, {gy:.2f}) is blocked by obstacles or outside navigable area",
                    )
                fw_x, fw_y = self.grid_to_world(*free_goal)
                snap_dist = math.hypot(fw_x - gx, fw_y - gy)
                if snap_dist > max_endpoint_adjustment_m:
                    return PlannedPath(
                        success=False,
                        start_pose=start_pose,
                        goal_pose=goal_pose,
                        original_goal_pose=original_goal_pose,
                        endpoint_adjustment_m=snap_dist,
                        rejection_reason=(
                            f"Goal snap adjustment ({snap_dist:.3f} m) exceeds maximum allowed "
                            f"tolerance ({max_endpoint_adjustment_m:.3f} m). Silent goal shifting is prohibited."
                        ),
                    )
                ggx, ggy = free_goal
            else:
                return PlannedPath(
                    success=False,
                    start_pose=start_pose,
                    goal_pose=goal_pose,
                    original_goal_pose=original_goal_pose,
                    rejection_reason=f"Goal position ({gx:.2f}, {gy:.2f}) is blocked by obstacles or unobserved cell. Silent goal shifting is prohibited.",
                )

        # 8-connected A* search
        actions = [
            (1, 0, 1.0), (-1, 0, 1.0), (0, 1, 1.0), (0, -1, 1.0),
            (1, 1, math.sqrt(2)), (-1, 1, math.sqrt(2)), (1, -1, math.sqrt(2)), (-1, -1, math.sqrt(2))
        ]

        open_set = []
        heapq.heappush(open_set, (0.0, 0.0, (sgx, sgy)))
        came_from: Dict[Tuple[int, int], Tuple[int, int]] = {}
        g_cost: Dict[Tuple[int, int], float] = {(sgx, sgy): 0.0}

        def heuristic(cell_a, cell_b):
            dx = abs(cell_a[0] - cell_b[0])
            dy = abs(cell_a[1] - cell_b[1])
            return (dx + dy) + (math.sqrt(2) - 2.0) * min(dx, dy)

        reached = False
        while open_set:
            _, current_g, current = heapq.heappop(open_set)
            if current == (ggx, ggy):
                reached = True
                break

            for adx, ady, step_cost in actions:
                nx, ny = current[0] + adx, current[1] + ady
                if not self.is_free(nx, ny):
                    continue

                # Prevent diagonal corner cutting
                if abs(adx) == 1 and abs(ady) == 1:
                    if not self.is_free(current[0] + adx, current[1]) or not self.is_free(current[0], current[1] + ady):
                        continue

                tentative_g = current_g + step_cost
                if (nx, ny) not in g_cost or tentative_g < g_cost[(nx, ny)]:
                    g_cost[(nx, ny)] = tentative_g
                    priority = tentative_g + heuristic((nx, ny), (ggx, ggy))
                    heapq.heappush(open_set, (priority, tentative_g, (nx, ny)))
                    came_from[(nx, ny)] = current

        if not reached:
            return PlannedPath(
                success=False,
                start_pose=start_pose,
                goal_pose=goal_pose,
                original_goal_pose=original_goal_pose,
                endpoint_adjustment_m=snap_dist,
                rejection_reason="No collision-free path exists between current robot pose and target goal",
            )

        # Reconstruct path
        path_cells = [(ggx, ggy)]
        curr = (ggx, ggy)
        while curr in came_from:
            curr = came_from[curr]
            path_cells.append(curr)
        path_cells.reverse()

        # Convert to world coordinates
        raw_pts = [self.grid_to_world(c[0], c[1]) for c in path_cells]
        simplified_pts = self._simplify_path(raw_pts)

        # Validate path footprint
        valid, min_clearance, reason = self.validate_path_footprint(simplified_pts, robot_radius_m=robot_radius_m)
        if not valid:
            valid_dense, min_clearance_dense, reason_dense = self.validate_path_footprint(raw_pts, robot_radius_m=robot_radius_m)
            if valid_dense:
                selected_pts = raw_pts
                min_clearance = min_clearance_dense
            else:
                return PlannedPath(
                    success=False,
                    start_pose=start_pose,
                    goal_pose=goal_pose,
                    original_goal_pose=original_goal_pose,
                    endpoint_adjustment_m=snap_dist,
                    rejection_reason=f"Path footprint validation failed: {reason}",
                )
        else:
            selected_pts = simplified_pts

        # Compute headings for each waypoint
        waypoints = []
        total_len = 0.0
        for i in range(len(selected_pts)):
            px, py = selected_pts[i]
            if i < len(selected_pts) - 1:
                nx, ny = selected_pts[i + 1]
                yaw = math.atan2(ny - py, nx - px)
                total_len += math.hypot(nx - px, ny - py)
            else:
                if target_center is not None:
                    yaw = math.atan2(target_center[1] - py, target_center[0] - px)
                else:
                    yaw = gyaw
            waypoints.append((px, py, yaw))

        final_endpoint = (waypoints[-1][0], waypoints[-1][1], waypoints[-1][2])
        if snap_dist == 0.0:
            snap_dist = math.hypot(final_endpoint[0] - original_goal_pose[0], final_endpoint[1] - original_goal_pose[1])

        return PlannedPath(
            success=True,
            waypoints_floor=waypoints,
            path_length_m=total_len,
            turn_points_count=max(0, len(selected_pts) - 2),
            start_pose=start_pose,
            goal_pose=final_endpoint,
            original_goal_pose=original_goal_pose,
            endpoint_adjustment_m=snap_dist,
            min_clearance_m=min_clearance,
            is_validated=True,
            target_entity_id=target_entity_id,
        )

    def validate_path_footprint(
        self,
        waypoints: List[Tuple[float, float]],
        robot_radius_m: Optional[float] = None,
        sample_step_m: float = 0.025,
    ) -> Tuple[bool, float, Optional[str]]:
        """Independently validate continuous path segments for collision-free robot footprint clearance."""
        if not waypoints:
            return False, 0.0, "Empty waypoints"

        r_cells = int(math.ceil(robot_radius_m / self.res)) if (robot_radius_m and robot_radius_m > 0) else 0
        min_clearance = float("inf")

        for i in range(len(waypoints) - 1):
            p1 = waypoints[i]
            p2 = waypoints[i + 1]
            seg_len = math.hypot(p2[0] - p1[0], p2[1] - p1[1])
            num_steps = max(2, int(math.ceil(seg_len / sample_step_m)))

            for s in range(num_steps + 1):
                t = s / float(num_steps)
                px = p1[0] + t * (p2[0] - p1[0])
                py = p1[1] + t * (p2[1] - p1[1])
                gx, gy = self.world_to_grid(px, py)

                if not (0 <= gx < self.width and 0 <= gy < self.height):
                    return False, 0.0, f"Path point ({px:.2f}, {py:.2f}) is outside map boundaries"

                if not self.is_free(gx, gy):
                    return False, 0.0, f"Path centerline at ({px:.2f}, {py:.2f}) intersects obstacle cell"

                if r_cells > 0:
                    for dx in range(-r_cells, r_cells + 1):
                        for dy in range(-r_cells, r_cells + 1):
                            if dx * dx + dy * dy <= (robot_radius_m / self.res) ** 2:
                                nx, ny = gx + dx, gy + dy
                                if not (0 <= nx < self.width and 0 <= ny < self.height):
                                    return False, 0.0, f"Robot footprint at ({px:.2f}, {py:.2f}) exceeds map boundary"
                                if self.grid[ny, nx] >= self.occ_thresh:
                                    return False, 0.0, f"Robot footprint at ({px:.2f}, {py:.2f}) intersects obstacle at grid ({nx}, {ny})"

                # Approximate obstacle distance for clearance metric
                dist_cells = max(1, r_cells)
                found = False
                for search_r in range(1, 15):
                    for dx in range(-search_r, search_r + 1):
                        for dy in range(-search_r, search_r + 1):
                            if abs(dx) == search_r or abs(dy) == search_r:
                                nx, ny = gx + dx, gy + dy
                                if not (0 <= nx < self.width and 0 <= ny < self.height) or self.grid[ny, nx] >= self.occ_thresh:
                                    dist_cells = search_r
                                    found = True
                                    break
                        if found:
                            break
                    if found:
                        break

                clearance_at_p = dist_cells * self.res
                min_clearance = min(min_clearance, clearance_at_p)

        return True, round(min_clearance, 3), None

    def _simplify_path(self, pts: List[Tuple[float, float]], tolerance_m: float = 0.08) -> List[Tuple[float, float]]:
        """Line-of-sight path simplification to eliminate zig-zags."""
        if len(pts) <= 2:
            return pts

        simplified = [pts[0]]
        current_idx = 0

        while current_idx < len(pts) - 1:
            furthest_idx = current_idx + 1
            for test_idx in range(current_idx + 2, len(pts)):
                if self._line_of_sight(pts[current_idx], pts[test_idx]):
                    furthest_idx = test_idx
                else:
                    break
            simplified.append(pts[furthest_idx])
            current_idx = furthest_idx

        return simplified

    def _line_of_sight(self, p1: Tuple[float, float], p2: Tuple[float, float]) -> bool:
        """Sample line in continuous Euclidean coordinates at half-cell resolution to ensure strict clearance."""
        dist = math.hypot(p2[0] - p1[0], p2[1] - p1[1])
        steps = max(1, int(math.ceil(dist / (self.res * 0.5))))
        for step in range(steps + 1):
            px = p1[0] + (p2[0] - p1[0]) * (step / steps)
            py = p1[1] + (p2[1] - p1[1]) * (step / steps)
            gx, gy = self.world_to_grid(px, py)
            if not self.is_free(gx, gy):
                return False
        return True

    def visualize_planned_route(
        self,
        planned_path: PlannedPath,
        output_image_path: Path,
        target_center: Optional[Tuple[float, float, float]] = None,
        target_bbox: Optional[Tuple[float, float, float, float]] = None,
        target_radius_m: float = 0.4,
        robot_radius_m: float = 0.18,
    ) -> None:
        """Render publication-grade visualization of costmap, robot footprint, and planned route."""
        fig, ax = plt.subplots(figsize=(11, 11), dpi=220)

        # Plot realistic 3-state costmap background:
        # Free (254) -> Pure White
        # Unknown (205) -> Clean Neutral Light Gray
        # Lethal Obstacles (0) -> Crisp Dark Slate
        # Inflated Safety Margin -> Soft Warning Tint
        extent = [self.ox, self.ox + self.width * self.res, self.oy, self.oy + self.height * self.res]

        ref_map = self.raw_map if self.raw_map is not None else self.grid

        # Clean noise: filter out isolated 1-15 cell obstacle noise and stray free-space specks
        raw_obs = (ref_map <= 50).astype(np.uint8)
        num_labels_o, labels_o, stats_o, _ = cv2.connectedComponentsWithStats(raw_obs, connectivity=8)
        clean_obs = np.zeros_like(raw_obs)
        for i in range(1, num_labels_o):
            if stats_o[i, cv2.CC_STAT_AREA] >= 16:
                clean_obs[labels_o == i] = 1

        raw_free = (ref_map >= 240).astype(np.uint8)
        num_labels_f, labels_f, stats_f, _ = cv2.connectedComponentsWithStats(raw_free, connectivity=8)
        clean_free = np.zeros_like(raw_free)
        for i in range(1, num_labels_f):
            if stats_f[i, cv2.CC_STAT_AREA] >= 20:
                clean_free[labels_f == i] = 1

        # Background: unknown space = soft neutral light slate gray (#F1F5F9)
        vis_rgb = np.full((self.height, self.width, 3), 241, dtype=np.uint8)

        # 1. Mapped Free Space -> Pure White (#FFFFFF)
        is_free = (clean_free == 1)
        vis_rgb[is_free] = [255, 255, 255]

        # 2. Inflated Safety Margin -> Soft warning buffer (#FEE2E2)
        is_inflated = is_free & (self.grid >= self.occ_thresh)
        vis_rgb[is_inflated] = [254, 226, 226]

        # 3. Lethal Obstacles -> Structured slate (#64748B) - Clean physical structures, not pitch black!
        is_obstacle = (clean_obs == 1)
        vis_rgb[is_obstacle] = [100, 116, 139]

        ax.imshow(vis_rgb, origin="lower", extent=extent)

        # Draw confirmed entities in the visible area to give semantic clarity
        if self.map_dir and (self.map_dir / "semantic_map.json").exists():
            try:
                import json
                with open(self.map_dir / "semantic_map.json", "r", encoding="utf-8") as sf:
                    sdata = json.load(sf)
                for eid, ent in sdata.get("entities", {}).items():
                    if eid == planned_path.target_entity_id:
                        continue  # Target is drawn with special prominent styling
                    cf = ent.get("centroid_floor")
                    dim = ent.get("dimensions", [0.5, 0.5, 0.5])
                    if not cf:
                        continue
                    ew, el = float(dim[0]), float(dim[1])
                    cls = ent.get("class_name", "")
                    col = "#94A3B8" if ("desk" in cls or "table" in cls) else ("#60A5FA" if "chair" in cls else "#A3E635")
                    epatch = patches.FancyBboxPatch(
                        (cf[0] - ew / 2.0, cf[1] - el / 2.0), ew, el,
                        boxstyle="round,pad=0.01",
                        linewidth=1.2, edgecolor=col, facecolor=col,
                        alpha=0.30, zorder=2,
                    )
                    ax.add_patch(epatch)
            except Exception:
                pass

        # Draw tight actual target footprint
        if target_bbox is not None:
            min_x, min_y, max_x, max_y = target_bbox
            w = max_x - min_x
            h = max_y - min_y
            rect = patches.FancyBboxPatch(
                (min_x, min_y), w, h,
                boxstyle="round,pad=0.02",
                linewidth=2.4, edgecolor="#7C3AED", facecolor="#A78BFA",
                alpha=0.60, zorder=4, label=f"Target Footprint ({planned_path.target_entity_id or 'Target'})"
            )
            ax.add_patch(rect)
        elif target_center is not None:
            tcx, tcy = target_center[0], target_center[1]
            obj_circle = patches.Circle(
                (tcx, tcy), target_radius_m,
                color="#8B5CF6", alpha=0.50, zorder=4, label=f"Target Object ({planned_path.target_entity_id or 'Goal'})"
            )
            ax.add_patch(obj_circle)

        if target_center is not None:
            ax.plot(target_center[0], target_center[1], "x", color="purple", markersize=10, mew=2.5, label="Target Center")

        if planned_path.success and planned_path.waypoints_floor:
            xs = [w[0] for w in planned_path.waypoints_floor]
            ys = [w[1] for w in planned_path.waypoints_floor]

            # Route line & waypoints
            ax.plot(xs, ys, "-", color="tab:cyan", linewidth=3.0, label=f"Planned Route ({planned_path.path_length_m:.2f} m)")
            ax.plot(xs, ys, "o", color="tab:blue", markersize=5, label="Turn Waypoints")

            # Start robot footprint
            sx, sy, syaw = planned_path.start_pose
            start_circle = patches.Circle((sx, sy), robot_radius_m, color="tab:green", alpha=0.5, label="Start Robot Pose")
            ax.add_patch(start_circle)
            ax.arrow(sx, sy, 0.3 * math.cos(syaw), 0.3 * math.sin(syaw), head_width=0.08, color="darkgreen")

            # Original requested approach goal (if different from snapped endpoint)
            orig_goal = planned_path.original_goal_pose
            final_goal = planned_path.goal_pose
            if orig_goal is not None and math.hypot(orig_goal[0] - final_goal[0], orig_goal[1] - final_goal[1]) > 0.03:
                ox, oy, oyaw = orig_goal
                orig_circle = patches.Circle((ox, oy), robot_radius_m, color="tab:pink", alpha=0.45, linestyle="--", fill=False, linewidth=2.0, label="Requested Goal")
                ax.add_patch(orig_circle)
                ax.plot([ox, final_goal[0]], [oy, final_goal[1]], "k--", linewidth=1.5, label=f"Endpoint Adj ({planned_path.endpoint_adjustment_m:.2f} m)")

            # Final planned endpoint and footprint
            gx, gy, gyaw = final_goal
            goal_circle = patches.Circle((gx, gy), robot_radius_m, color="tab:red", alpha=0.5, label="Planned Endpoint")
            ax.add_patch(goal_circle)
            ax.arrow(gx, gy, 0.3 * math.cos(gyaw), 0.3 * math.sin(gyaw), head_width=0.08, color="darkred")

            # Add technical parameters card
            info_text = (
                f"Path Length: {planned_path.path_length_m:.2f} m\n"
                f"Endpoint Adj: {planned_path.endpoint_adjustment_m:.3f} m\n"
                f"Min Clearance: {planned_path.min_clearance_m:.2f} m\n"
                f"Validation: {'PASSED' if planned_path.is_validated else 'UNCHECKED'}"
            )
            ax.text(
                0.02, 0.02, info_text, transform=ax.transAxes,
                fontsize=9, verticalalignment="bottom",
                bbox=dict(boxstyle="round,pad=0.5", facecolor="white", alpha=0.85, edgecolor="gray")
            )

        ax.set_title(f"Planned Autonomous Navigation Route — Goal: {planned_path.target_entity_id or 'Goal Pose'}", fontsize=13, weight="bold")
        ax.set_xlabel("Floor Coordinate X (meters - Forward)", fontsize=11)
        ax.set_ylabel("Floor Coordinate Y (meters - Left)", fontsize=11)
        ax.legend(loc="upper right", framealpha=0.9, fontsize=8)
        ax.grid(True, linestyle="--", alpha=0.5)
        ax.axis("equal")

        # Auto-tighten viewport around start pose, route, and target object
        focus_x = [planned_path.start_pose[0], planned_path.goal_pose[0]]
        focus_y = [planned_path.start_pose[1], planned_path.goal_pose[1]]
        if planned_path.waypoints_floor:
            focus_x.extend([w[0] for w in planned_path.waypoints_floor])
            focus_y.extend([w[1] for w in planned_path.waypoints_floor])
        if target_center is not None:
            focus_x.append(target_center[0])
            focus_y.append(target_center[1])
        if target_bbox is not None:
            focus_x.extend([target_bbox[0], target_bbox[2]])
            focus_y.extend([target_bbox[1], target_bbox[3]])

        pad_m = 1.4
        min_fx, max_fx = min(focus_x) - pad_m, max(focus_x) + pad_m
        min_fy, max_fy = min(focus_y) - pad_m, max(focus_y) + pad_m
        span = max(max_fx - min_fx, max_fy - min_fy, 4.2)
        cx_v = (min_fx + max_fx) / 2.0
        cy_v = (min_fy + max_fy) / 2.0
        ax.set_xlim(cx_v - span / 2.0, cx_v + span / 2.0)
        ax.set_ylim(cy_v - span / 2.0, cy_v + span / 2.0)

        output_image_path = Path(output_image_path)
        output_image_path.parent.mkdir(parents=True, exist_ok=True)
        plt.tight_layout()
        plt.savefig(str(output_image_path))
        plt.close(fig)
