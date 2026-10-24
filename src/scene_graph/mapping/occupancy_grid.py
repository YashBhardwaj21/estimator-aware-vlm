"""Metric 2D Occupancy Grid Mapping with Raycasting and Inflation."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple, Optional
import numpy as np


@dataclass
class OccupancyGridConfig:
    resolution_m: float = 0.05        # 5 cm per cell
    origin_x_m: float = -3.5           # Grid origin X in floor-aligned coordinates
    origin_y_m: float = -3.5           # Grid origin Y in floor-aligned coordinates
    width_m: float = 8.0               # Grid width in meters
    height_m: float = 8.0              # Grid height in meters
    min_obstacle_height_m: float = 0.08  # Obstacles must be at least 8 cm above floor
    max_obstacle_height_m: float = 1.80  # Max obstacle height (clearance)
    floor_band_m: float = 0.06           # Depth points within +/- 6 cm of floor are floor returns
    robot_radius_m: float = 0.15         # Robot collision radius
    safety_margin_m: float = 0.05        # Additional clearance inflation
    max_ray_range_m: float = 4.0         # Max depth range for raycasting
    l_occ: float = 0.85                  # Log-odds increment on hit (P ~ 0.70)
    l_free: float = -0.40                # Log-odds decrement on free/miss (P ~ 0.40)
    l_min: float = -5.0                  # Lower log-odds saturation bound
    l_max: float = 5.0                   # Upper log-odds saturation bound
    l_occ_thresh: float = 0.85           # Log-odds threshold for OCCUPIED state (P >= 0.70)
    l_free_thresh: float = -0.40         # Log-odds threshold for FREE state (P <= 0.40)


class OccupancyGrid:
    """A metric 2D occupancy grid maintaining Free (0), Occupied (100), and Unknown (-1) states."""

    UNKNOWN = -1
    FREE = 0
    OCCUPIED = 100

    def __init__(self, config: Optional[OccupancyGridConfig] = None):
        self.config = config or OccupancyGridConfig()
        self.resolution = self.config.resolution_m
        self.origin_x = self.config.origin_x_m
        self.origin_y = self.config.origin_y_m

        self.cols = int(math.ceil(self.config.width_m / self.resolution))
        self.rows = int(math.ceil(self.config.height_m / self.resolution))

        # Grid representation: int8 array
        # -1 = Unknown, 0 = Free, 100 = Occupied
        self.grid = np.full((self.rows, self.cols), self.UNKNOWN, dtype=np.int8)

        # Continuous bounded log-odds representation for inverse sensor model
        self.log_odds = np.zeros((self.rows, self.cols), dtype=np.float32)

        # Hit and miss count accumulators for robust thresholding
        self.hits = np.zeros((self.rows, self.cols), dtype=np.int32)
        self.misses = np.zeros((self.rows, self.cols), dtype=np.int32)

    def world_to_grid(self, x: float, y: float) -> Tuple[int, int]:
        """Convert continuous 2D floor coordinates (x, y) to grid cell indices (col, row)."""
        col = int(math.floor((x - self.origin_x) / self.resolution))
        row = int(math.floor((y - self.origin_y) / self.resolution))
        return col, row

    def grid_to_world(self, col: int, row: int) -> Tuple[float, float]:
        """Convert grid cell indices (col, row) to cell center in world coordinates."""
        x = self.origin_x + (col + 0.5) * self.resolution
        y = self.origin_y + (row + 0.5) * self.resolution
        return x, y

    def is_inside(self, col: int, row: int) -> bool:
        """Check whether (col, row) falls inside grid bounds."""
        return 0 <= col < self.cols and 0 <= row < self.rows

    def update_ray(self, start_x: float, start_y: float, end_x: float, end_y: float, is_hit: bool):
        """Bresenham raycasting with log-odds inverse sensor model.
        
        Cells along the ray are decremented by l_free. Endpoints are incremented by l_occ if hit.
        Cells previously OCCUPIED can be cleared back to FREE if observed empty, preventing ghost maps.
        """
        c0, r0 = self.world_to_grid(start_x, start_y)
        c1, r1 = self.world_to_grid(end_x, end_y)

        # Bresenham line generation
        dx = abs(c1 - c0)
        dy = abs(r1 - r0)
        sx = 1 if c0 < c1 else -1
        sy = 1 if r0 < r1 else -1
        err = dx - dy

        curr_c, curr_r = c0, r0
        while True:
            if curr_c == c1 and curr_r == r1:
                break
            if self.is_inside(curr_c, curr_r):
                self.misses[curr_r, curr_c] += 1
                new_l = max(self.config.l_min, self.log_odds[curr_r, curr_c] + self.config.l_free)
                self.log_odds[curr_r, curr_c] = new_l
                if new_l <= self.config.l_free_thresh:
                    self.grid[curr_r, curr_c] = self.FREE
                elif new_l >= self.config.l_occ_thresh:
                    self.grid[curr_r, curr_c] = self.OCCUPIED
                else:
                    self.grid[curr_r, curr_c] = self.UNKNOWN

            e2 = 2 * err
            if e2 > -dy:
                err -= dy
                curr_c += sx
            if e2 < dx:
                err += dx
                curr_r += sy

        # Endpoint handling
        if self.is_inside(c1, r1):
            if is_hit:
                self.hits[r1, c1] += 1
                new_l = min(self.config.l_max, self.log_odds[r1, c1] + self.config.l_occ)
                self.log_odds[r1, c1] = new_l
                if new_l >= self.config.l_occ_thresh:
                    self.grid[r1, c1] = self.OCCUPIED
                elif new_l <= self.config.l_free_thresh:
                    self.grid[r1, c1] = self.FREE
                else:
                    self.grid[r1, c1] = self.UNKNOWN
            else:
                self.misses[r1, c1] += 1
                new_l = max(self.config.l_min, self.log_odds[r1, c1] + self.config.l_free)
                self.log_odds[r1, c1] = new_l
                if new_l <= self.config.l_free_thresh:
                    self.grid[r1, c1] = self.FREE
                elif new_l >= self.config.l_occ_thresh:
                    self.grid[r1, c1] = self.OCCUPIED
                else:
                    self.grid[r1, c1] = self.UNKNOWN

    def fuse_log_odds(self, other: OccupancyGrid) -> None:
        """Probabilistically fuse another aligned occupancy grid using log-odds addition."""
        if self.rows != other.rows or self.cols != other.cols:
            raise ValueError(
                f"Grid dimension mismatch for fusion: ({self.rows}, {self.cols}) vs ({other.rows}, {other.cols})"
            )
        self.log_odds = np.clip(
            self.log_odds + other.log_odds,
            self.config.l_min,
            self.config.l_max,
        )
        self.hits += other.hits
        self.misses += other.misses

        self.grid.fill(self.UNKNOWN)
        self.grid[self.log_odds <= self.config.l_free_thresh] = self.FREE
        self.grid[self.log_odds >= self.config.l_occ_thresh] = self.OCCUPIED

    def integrate_depth_points(
        self,
        camera_floor_x: float,
        camera_floor_y: float,
        points_floor_aligned: np.ndarray,
    ):
        """Integrate 3D points expressed in floor-aligned coordinates.
        
        Floor-aligned convention:
        - X, Y are on the horizontal floor plane.
        - Z is perpendicular height above the floor (Z=0 is floor surface).
        """
        if points_floor_aligned.size == 0:
            return

        cam_x = camera_floor_x
        cam_y = camera_floor_y

        for pt in points_floor_aligned:
            px, py, pz = pt[0], pt[1], pt[2]
            dist = math.hypot(px - cam_x, py - cam_y)
            if dist > self.config.max_ray_range_m or dist < 0.15:
                continue

            # Check if point is floor return
            if abs(pz) <= self.config.floor_band_m:
                # Floor return clears free space up to contact point
                self.update_ray(cam_x, cam_y, px, py, is_hit=False)
            elif self.config.min_obstacle_height_m <= pz <= self.config.max_obstacle_height_m:
                # Obstacle hit
                self.update_ray(cam_x, cam_y, px, py, is_hit=True)

    def get_inflated_grid(self) -> np.ndarray:
        """Create an inflated occupancy grid using robot radius + safety margin."""
        inflated = self.grid.copy()
        inflation_radius = self.config.robot_radius_m + self.config.safety_margin_m
        kernel_radius_cells = int(math.ceil(inflation_radius / self.resolution))

        occ_rows, occ_cols = np.where(self.grid == self.OCCUPIED)
        if len(occ_rows) == 0:
            return inflated

        # Circular kernel
        y_idx, x_idx = np.ogrid[-kernel_radius_cells:kernel_radius_cells + 1, -kernel_radius_cells:kernel_radius_cells + 1]
        mask = (x_idx ** 2 + y_idx ** 2) <= (inflation_radius / self.resolution) ** 2

        k_h, k_w = mask.shape
        half_h, half_w = k_h // 2, k_w // 2

        for r, c in zip(occ_rows, occ_cols):
            r_min = max(0, r - half_h)
            r_max = min(self.rows, r + half_h + 1)
            c_min = max(0, c - half_w)
            c_max = min(self.cols, c + half_w + 1)

            kr_min = half_h - (r - r_min)
            kr_max = half_h + (r_max - r)
            kc_min = half_w - (c - c_min)
            kc_max = half_w + (c_max - c)

            sub_mask = mask[kr_min:kr_max, kc_min:kc_max]
            # Inflate cells: if currently FREE or UNKNOWN, make OCCUPIED
            sub_grid = inflated[r_min:r_max, c_min:c_max]
            sub_grid[sub_mask] = self.OCCUPIED

        return inflated

    def export_ros_map(self, output_pgm_path: Path, output_yaml_path: Path, use_inflation: bool = False):
        """Export map to standard ROS PGM and YAML files."""
        grid_data = self.get_inflated_grid() if use_inflation else self.grid

        # Standard ROS mapping:
        # Occupied (100) -> 0 (Black)
        # Free (0) -> 254 (White)
        # Unknown (-1) -> 205 (Grey)
        image_data = np.full((self.rows, self.cols), 205, dtype=np.uint8)
        image_data[grid_data == self.FREE] = 254
        image_data[grid_data == self.OCCUPIED] = 0

        # ROS map images have origin at bottom-left, so flip vertically
        image_flipped = np.flipud(image_data)

        # Write PGM (P5 binary)
        with open(output_pgm_path, "wb") as f:
            header = f"P5\n{self.cols} {self.rows}\n255\n"
            f.write(header.encode("ascii"))
            f.write(image_flipped.tobytes())

        # Write YAML metadata
        yaml_content = f"""image: {output_pgm_path.name}
resolution: {self.resolution:.4f}
origin: [{self.origin_x:.4f}, {self.origin_y:.4f}, 0.0]
negate: 0
occupied_thresh: 0.65
free_thresh: 0.196
"""
        with open(output_yaml_path, "w", encoding="utf-8") as f:
            f.write(yaml_content)

    def compute_stats(self) -> dict:
        """Compute mapping cell statistics."""
        total = self.rows * self.cols
        free = int(np.sum(self.grid == self.FREE))
        occ = int(np.sum(self.grid == self.OCCUPIED))
        unk = int(np.sum(self.grid == self.UNKNOWN))

        inflated = self.get_inflated_grid()
        inflated_occ = int(np.sum(inflated == self.OCCUPIED))
        inflated_free = int(np.sum(inflated == self.FREE))

        return {
            "total_cells": total,
            "free_cells": free,
            "occupied_cells": occ,
            "unknown_cells": unk,
            "free_pct": round(free / total * 100.0, 2),
            "occupied_pct": round(occ / total * 100.0, 2),
            "unknown_pct": round(unk / total * 100.0, 2),
            "inflated_occupied_cells": inflated_occ,
            "inflated_occupied_pct": round(inflated_occ / total * 100.0, 2),
            "inflated_free_cells": inflated_free,
            "inflated_free_pct": round(inflated_free / total * 100.0, 2),
            "resolution_m": self.resolution,
            "width_m": round(self.cols * self.resolution, 4),
            "height_m": round(self.rows * self.resolution, 4),
        }

