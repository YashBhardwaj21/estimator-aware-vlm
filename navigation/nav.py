#!/usr/bin/env python3
"""Lightweight Autonomous Navigation for Raspberry Pi 5 + Intel RealSense D455 + L298N 4WD.

Key Design:
  1. A* Path Planning on 2D metric occupancy map with obstacle inflation.
  2. D455 Hardware Gyro Closed-Loop: Real-time heading tracking and drift-free active steering.
  3. D455 Depth Obstacle Avoidance: Direct ROI depth check (< 0.40m halt & A* replan) - zero CPU bloat.
  4. Dead-Reckoning Position Tracking: Clean integration of speed and gyro heading.
  5. Minimal Bandwidth & CPU: Depth + Gyro only (no heavy visual odometry or image streaming).
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
import heapq
import json
import math
import os
import sys
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import cv2
import numpy as np

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

try:
    from gpiozero import DigitalOutputDevice, PWMOutputDevice
    HAS_GPIO = True
except ImportError:
    HAS_GPIO = False

# L298N GPIO Pinout (Raspberry Pi 5 BCM numbering)
IN1_PIN = 17
IN2_PIN = 27
IN3_PIN = 22
IN4_PIN = 23
ENA_PIN = 18
ENB_PIN = 19


@dataclass
class MotionResult:
    """Structured contract returned by every low-level movement operation."""
    success: bool
    halt_reason: str            # "completed", "obstacle", "timeout", "stopped", "gyro_divergence", "sensor_stale"
    final_pose: tuple[float, float, float]       # (x_m, y_m, yaw_rad)
    estimated_displacement: tuple[float, float]  # (dx_m, dy_m)
    elapsed_s: float
    depth_m: Optional[float] = None
    depth_age_s: Optional[float] = None
    actual_turn_rad: float = 0.0
    details: str = ""

    def __iter__(self):
        # Backward compatibility tuple unpacking: ((dx, dy), halt_reason, last_depth)
        yield self.estimated_displacement
        yield self.halt_reason
        yield self.depth_m

    def __getitem__(self, index):
        return tuple(self)[index]

    def __float__(self) -> float:
        return float(self.actual_turn_rad)

    def __add__(self, other) -> float:
        return self.actual_turn_rad + float(other)

    def __radd__(self, other) -> float:
        return float(other) + self.actual_turn_rad

    def __sub__(self, other) -> float:
        return self.actual_turn_rad - float(other)

    def __rsub__(self, other) -> float:
        return float(other) - self.actual_turn_rad


@dataclass
class NavConfig:
    """Instance-specific validated configuration for locomotion, sensors, and safety bounds."""
    cruise_pwm: float = 0.58          # Forward rolling duty cycle (smooth, non-aggressive)
    turn_pwm: float = 0.65            # Skid-steer breakaway & turn duty cycle
    speed_mps: float = 0.18           # Calibrated linear speed in m/s (~18 cm/s at 58% PWM)
    turn_radps: float = 0.90          # Turn rate in rad/s (~51 deg/s at 65% PWM)
    obstacle_halt_dist_m: float = 0.40# Stop threshold ahead. D455 cannot measure below ~0.4m, so never set lower.
    obstacle_slow_dist_m: float = 0.80# Start slowing down when obstacle is closer than this
    backup_dist_m: float = 0.12       # Reversal clearance after obstacle halt
    robot_radius_m: float = 0.08      # Robot radius for inflation
    obstacle_inflation_m: float = 0.05# Extra margin on top of robot radius for grid obstacle dilation
    dynamic_obstacle_radius_m: float = 0.10  # Physical radius assumed for a sensed obstacle
    depth_hfov_deg: float = 87.0      # D455 depth horizontal field of view
    max_depth_age_s: float = 0.60     # Freshness deadline for depth frames
    max_replans: int = 8              # Bounded replanning limit before safe abort
    breakaway_burst_s: float = 0.07   # High-torque burst to overcome floor scrub
    breakaway_pwm: float = 0.78       # Peak PWM during breakaway burst
    cam_inverted: bool = True         # RealSense mounted upside-down
    motors_inverted: bool = False     # Motor wiring polarity
    flip_gyro: bool = True            # Gyro polarity inversion (default True: sign = +1.0 for inverted D455)
    pivot_tolerance_deg: float = 3.5  # Alignment angular tolerance
    waypoint_tolerance_m: float = 0.08# Waypoint arrival tolerance
    goal_tolerance_m: float = 0.12    # Final destination arrival tolerance
    goal_obstacle_tolerance_m: float = 0.40  # Obstacle halt this close to goal counts as arrived (targets sit next to furniture)
    dynamic_obstacle_ttl_s: float = 20.0  # Dynamic obstacle expiration TTL in seconds
    localize: bool = True             # Correct dead-reckoning drift by matching depth scans to the map
    loc_max_shift_m: float = 0.30     # Largest position correction searched per localization
    loc_max_rot_deg: float = 10.0     # Largest heading correction searched per localization
    loc_min_score: float = 0.45       # Match quality (0..1) required to accept a correction
    loc_min_gain: float = 0.08        # Required score improvement over the current pose


DEFAULT_CONFIG = NavConfig()

# Legacy module constants for backward compatibility
CRUISE_PWM = DEFAULT_CONFIG.cruise_pwm
TURN_PWM = DEFAULT_CONFIG.turn_pwm
SPEED_MPS = DEFAULT_CONFIG.speed_mps
TURN_RADPS = DEFAULT_CONFIG.turn_radps


def speed_for_pwm(pwm: float) -> float:
    """Estimate linear speed for a cruise PWM by scaling the calibrated default."""
    return max(0.06, DEFAULT_CONFIG.speed_mps * pwm / DEFAULT_CONFIG.cruise_pwm)


def find_fused_dir() -> Path:
    candidates = [
        Path("outputs/semantic_mapping_fused_desk_run"),
        Path("semantic_mapping_fused_desk_run"),
        Path("outputs/outputs/semantic_mapping_fused_desk_run"),
        Path.home() / "perception/outputs/semantic_mapping_fused_desk_run",
        Path.home() / "perception/semantic_mapping_fused_desk_run",
    ]
    for c in candidates:
        if (c / "occupancy_map.yaml").exists() or (c / "map.yaml").exists():
            return c
    for p in Path(".").glob("**/occupancy_map.yaml"):
        return p.parent
    for p in Path(".").glob("**/map.yaml"):
        return p.parent
    return Path("outputs/semantic_mapping_fused_desk_run")


STATE_FILE = Path("robot_state.json")
FUSED_DIR = find_fused_dir()
FUSED_STATE_FILE = FUSED_DIR / "robot_navigation_state.json"

# Facility Landmark Database with Standoff Approach Poses
TARGET_DATABASE = {
    "desk_001": {
        "aliases": ["desk 1", "desk 01", "desk001", "main desk", "table 1", "primary desk"],
        "approach": (2.00, -1.25, math.radians(-90.0)),
        "center": (2.30, -2.14),
        "desc": "Primary workstation desk",
    },
    "chair_002": {
        "aliases": ["chair 2", "chair 02", "chair002", "side chair", "desk chair", "yellow chair", "conference chair"],
        "approach": (1.85, -1.25, math.radians(-90.0)),
        "center": (1.76, -1.96),
        "desc": "Desk 1 yellow conference chair",
    },
    "chair_003": {
        "aliases": ["chair 3", "chair 03", "chair003", "meeting chair", "central chair"],
        "approach": (1.13, -2.20, math.radians(-90.0)),
        "center": (1.13, -2.92),
        "desc": "Central meeting chair",
    },
    "box_001": {
        "aliases": ["box 1", "box 01", "storage box 1", "box", "storage box_001"],
        "approach": (1.66, -1.33, math.radians(-68.6)),
        "center": (1.65, -2.89),
        "desc": "Storage box by desk 1",
    },
    "chair_005": {
        "aliases": ["chair 5", "chair 05", "chair005", "south chair"],
        "approach": (0.61, -3.22, math.radians(-76.8)),
        "center": (0.66, -3.87),
        "desc": "South meeting chair",
    },
    "desk_005": {
        "aliases": ["desk 5", "desk 05", "desk005", "south bench", "lab bench"],
        "approach": (-0.16, -4.50, math.radians(-90.0)),
        "center": (-0.14, -5.72),
        "desc": "South laboratory bench",
    },
    "desk_018": {
        "aliases": ["desk 18", "desk18", "table 18", "bench 18", "west desk", "west bench"],
        "approach": (-1.36, -3.02, math.pi),
        "center": (-2.56, -3.02),
        "desc": "West workstation bench",
    },
    "desk_017": {
        "aliases": ["desk 17", "desk17", "table 17", "northwest bench"],
        "approach": (-1.17, -2.30, math.radians(80.3)),
        "center": (-1.88, -1.53),
        "desc": "Northwest center bench",
    },
    "desk_023": {
        "aliases": ["desk 23", "desk23", "table 23", "west corridor desk"],
        "approach": (-1.08, -2.05, math.radians(78.3)),
        "center": (-1.16, -1.11),
        "desc": "West corridor workstation",
    },
    "desk_024": {
        "aliases": ["desk 24", "desk24", "table 24", "north desk"],
        "approach": (-0.38, -0.80, math.radians(83.7)),
        "center": (-0.65, -0.28),
        "desc": "North corridor desk",
    },
    "origin": {
        "aliases": ["origin", "start", "home", "base", "0 0", "start origin"],
        "approach": (0.0, 0.0, 0.0),
        "center": (0.0, 0.0),
        "desc": "Facility entrance origin",
    },
}

TOUR_SEQUENCE = [
    ("desk_001", "Primary Workstation Desk"),
    ("chair_002", "Conference Side Chair"),
    ("chair_003", "Central Meeting Chair"),
    ("box_001", "Storage Box by Desk 1"),
    ("chair_005", "South Meeting Chair"),
    ("desk_005", "South Laboratory Bench"),
    ("desk_018", "West Workstation Bench"),
    ("desk_017", "Northwest Center Bench"),
    ("desk_023", "West Corridor Workstation"),
    ("desk_024", "North Corridor Desk"),
]


def normalize_angle(angle: float) -> float:
    """Normalize angle to [-pi, +pi]."""
    while angle > math.pi:
        angle -= 2.0 * math.pi
    while angle < -math.pi:
        angle += 2.0 * math.pi
    return angle


def load_state() -> tuple[float, float, float]:
    """Load last known robot pose (x, y, yaw_rad)."""
    if STATE_FILE.exists():
        try:
            with open(STATE_FILE, "r", encoding="utf-8") as f:
                d = json.load(f)
                return float(d.get("x", 0.0)), float(d.get("y", 0.0)), float(d.get("yaw", 0.0))
        except Exception:
            pass
    return (0.0, 0.0, 0.0)


def save_state(x: float, y: float, yaw: float, last_target: str):
    """Save persistent robot pose."""
    d = {
        "x": round(x, 3),
        "y": round(y, 3),
        "yaw": round(yaw, 4),
        "yaw_deg": round(math.degrees(yaw), 1),
        "last_target": last_target,
        "updated_at": time.time(),
    }
    with open(STATE_FILE, "w", encoding="utf-8") as f:
        json.dump(d, f, indent=2)

    if FUSED_DIR.exists():
        try:
            with open(FUSED_STATE_FILE, "w", encoding="utf-8") as f:
                json.dump({"current_pose": d}, f, indent=2)
        except Exception:
            pass


def resolve_target(query: str, semantic_entities: Dict[str, dict] = None) -> tuple[Optional[str], Optional[dict]]:
    """Match query string against target database."""
    if not query:
        return None, None
    q = query.strip().lower()
    for prefix in ["go to ", "move to ", "navigate to ", "drive to ", "find ", "head to "]:
        if q.startswith(prefix):
            q = q[len(prefix):].strip()

    # Pass 1: exact matches
    for tid, info in TARGET_DATABASE.items():
        if q == tid.lower() or q == tid.lower().replace("_", "") or q == tid.lower().replace("_", " "):
            return tid, info
        for alias in info.get("aliases", []):
            if q == alias.lower():
                return tid, info

    # Pass 2: token match
    q_tokens = set(q.replace("_", " ").split())
    for tid, info in TARGET_DATABASE.items():
        if q_tokens == set(tid.lower().replace("_", " ").split()):
            return tid, info
        for alias in info.get("aliases", []):
            if q_tokens == set(alias.lower().split()):
                return tid, info

    # Pass 3: number extraction (e.g. "desk 18")
    import re
    q_nums = re.findall(r"\d+", q)
    if q_nums:
        for tid, info in TARGET_DATABASE.items():
            t_nums = re.findall(r"\d+", tid)
            if [str(int(n)) for n in q_nums] == [str(int(n)) for n in t_nums]:
                for word in ["desk", "chair", "table", "box", "bench"]:
                    if word in q and (word in tid or any(word in a for a in info.get("aliases", []))):
                        return tid, info

    # Pass 4: dynamic semantic map fallback
    if semantic_entities:
        for eid, ent in semantic_entities.items():
            if q == eid.lower() or q == eid.lower().replace("_", " "):
                cf = ent.get("centroid_floor", [0, 0])
                cls_name = ent.get("class_name", "object")
                info = {
                    "aliases": [eid],
                    "approach": (cf[0] + 0.50, cf[1], math.pi),
                    "center": (cf[0], cf[1]),
                    "desc": f"Scene object: {eid} ({cls_name})",
                }
                return eid, info

    return None, None


class FastAStarPlanner:
    """8-connected grid A* path planner on 2D metric occupancy grid."""

    def __init__(self, map_dir: Path):
        self.map_dir = Path(map_dir)
        yaml_p = self.map_dir / "occupancy_map.yaml"
        if not yaml_p.exists():
            yaml_p = self.map_dir / "map.yaml"
        if not yaml_p.exists():
            for p in self.map_dir.glob("**/*map*.yaml"):
                yaml_p = p
                self.map_dir = p.parent
                break
        if not yaml_p.exists():
            for p in Path(".").glob("**/occupancy_map.yaml"):
                yaml_p = p
                self.map_dir = p.parent
                break

        import yaml
        with open(yaml_p, "r", encoding="utf-8") as f:
            meta = yaml.safe_load(f)

        self.res = float(meta["resolution"])
        orig = [float(v) for v in meta["origin"]]
        self.ox, self.oy = orig[0], orig[1]

        img_name = meta["image"]
        pgm_p = self.map_dir / img_name
        if not pgm_p.exists():
            cand = self.map_dir / (Path(img_name).stem + ".png")
            if cand.exists():
                pgm_p = cand
            else:
                for p in self.map_dir.glob("*.pgm"):
                    pgm_p = p
                    break

        raw = cv2.imread(str(pgm_p), cv2.IMREAD_UNCHANGED)
        if raw is None:
            raise FileNotFoundError(f"Could not read map image at '{pgm_p}'")
        self.grid = np.flipud(raw)

        self.h, self.w = self.grid.shape
        self.extent = [self.ox, self.ox + self.w * self.res, self.oy, self.oy + self.h * self.res]

        # Inflate obstacles by robot radius + safety margin so planned paths keep the chassis off walls
        self.inflation_radius_m = DEFAULT_CONFIG.robot_radius_m + DEFAULT_CONFIG.obstacle_inflation_m
        r_cells = max(1, int(round(self.inflation_radius_m / self.res)))
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r_cells + 1, 2 * r_cells + 1))
        obs_mask = (self.grid <= 64).astype(np.uint8)
        self.inflated_obs = cv2.dilate(obs_mask, kernel)

        self.raw_grid = self.grid.copy()
        self.raw_inflated_obs = self.inflated_obs.copy()
        # Distance (m) from every cell to the nearest static obstacle, for scan matching
        # Only obstacle SURFACES (obstacle cells touching free space) attract scan points, so a scan
        # cannot be shifted "inside" a thick wall and still look like a perfect match.
        obs = (self.raw_grid <= 64).astype(np.uint8)
        free = (self.raw_grid >= 250).astype(np.uint8)
        surface = obs & cv2.dilate(free, np.ones((3, 3), np.uint8))
        self.obs_dist_m = cv2.distanceTransform((1 - surface).astype(np.uint8), cv2.DIST_L2, 5).astype(np.float32) * self.res
        self.dynamic_obstacles: list[dict] = []

    def scan_score(self, pts_world: np.ndarray, sigma: float = 0.06) -> np.ndarray:
        """Score scan points against the static map. pts_world: (..., N, 2). Returns mean likelihood (...)."""
        gx = np.floor((pts_world[..., 0] - self.ox) / self.res).astype(np.int32)
        gy = np.floor((pts_world[..., 1] - self.oy) / self.res).astype(np.int32)
        inside = (gx >= 0) & (gx < self.w) & (gy >= 0) & (gy < self.h)
        d = np.full(gx.shape, 1.0, dtype=np.float32)
        d[inside] = self.obs_dist_m[gy[inside], gx[inside]]
        return np.exp(-(d * d) / (2.0 * sigma * sigma)).mean(axis=-1)

    def match_scan(self, pose: tuple[float, float, float], pts_robot: np.ndarray,
                   max_shift: float = 0.30, max_rot_deg: float = 10.0):
        """Correlative scan matching around `pose`. pts_robot: (N, 2) [forward, left] in metres.
        Small corrections are preferred (prior penalty) and ambiguous matches, e.g. a single
        flat wall the scan can slide along, are flagged.
        Returns (best_pose, best_score, current_score, ambiguous)."""
        x0, y0, th0 = pose
        f, l = pts_robot[:, 0], pts_robot[:, 1]

        def _grid(cx, cy, cth, shifts, rots):
            ths = cth + rots
            c, s_ = np.cos(ths), np.sin(ths)
            rx = c[:, None] * f[None, :] - s_[:, None] * l[None, :]   # (R, N)
            ry = s_[:, None] * f[None, :] + c[:, None] * l[None, :]
            dx, dy = np.meshgrid(shifts, shifts, indexing="ij")
            dx, dy = dx.ravel(), dy.ravel()                            # (S,)
            wx = cx + dx[None, :, None] + rx[:, None, :]               # (R, S, N)
            wy = cy + dy[None, :, None] + ry[:, None, :]
            sc = self.scan_score(np.stack([wx, wy], axis=-1))          # (R, S)
            px = (cx + dx)[None, :].repeat(len(ths), 0)
            py = (cy + dy)[None, :].repeat(len(ths), 0)
            pth = ths[:, None].repeat(len(dx), 1)
            return sc, px, py, pth

        def _penalised(sc, px, py, pth):
            dxy = np.hypot(px - x0, py - y0) / max(1e-3, max_shift)
            dth = np.abs(np.vectorize(normalize_angle)(pth - th0)) / max(1e-3, math.radians(max_rot_deg))
            return sc - 0.15 * dxy ** 2 - 0.15 * dth ** 2

        cur_score = float(_grid(x0, y0, th0, np.zeros(1), np.zeros(1))[0][0, 0])
        step = max(self.res, 0.05)
        sc, px, py, pth = _grid(x0, y0, th0,
                                np.arange(-max_shift, max_shift + 1e-6, step),
                                np.radians(np.arange(-max_rot_deg, max_rot_deg + 1e-6, 1.0)))
        pen = _penalised(sc, px, py, pth)
        k = int(np.argmax(pen))
        bx, by, bth = px.flat[k], py.flat[k], pth.flat[k]

        # Ambiguity: other clearly different poses that explain the scan almost as well
        near_best = pen >= pen.flat[k] - 0.03
        spread_xy = float(np.max(np.hypot(px[near_best] - bx, py[near_best] - by)))
        spread_th = float(np.max(np.abs(np.vectorize(normalize_angle)(pth[near_best] - bth))))
        ambiguous = spread_xy > 0.12 or spread_th > math.radians(3.0)

        # Fine refinement around the best coarse pose
        sc2, px2, py2, pth2 = _grid(bx, by, bth, np.arange(-step, step + 1e-6, 0.01),
                                    np.radians(np.arange(-1.0, 1.0 + 1e-6, 0.25)))
        pen2 = _penalised(sc2, px2, py2, pth2)
        k2 = int(np.argmax(pen2))
        best = (float(px2.flat[k2]), float(py2.flat[k2]), normalize_angle(float(pth2.flat[k2])))
        return best, float(sc2.flat[k2]), cur_score, ambiguous

    def cleanup_expired_dynamic_obstacles(self, now: Optional[float] = None) -> bool:
        """Prune dynamic obstacles that exceeded TTL, restoring static map and reapplying active ones."""
        if not self.dynamic_obstacles:
            return False
        cur_t = now if now is not None else time.time()
        active = [obs for obs in self.dynamic_obstacles if obs.get("expires_at", float("inf")) > cur_t]
        if len(active) == len(self.dynamic_obstacles):
            return False
        # Reset to pristine static map and re-rasterize active ones
        base_grid = getattr(self, "raw_grid", getattr(self, "static_grid", None))
        base_inflated = getattr(self, "raw_inflated_obs", getattr(self, "inflated_obs", None))
        if base_grid is not None:
            self.grid = base_grid.copy()
        if base_inflated is not None:
            self.inflated_obs = base_inflated.copy()
        self.dynamic_obstacles = []
        for obs in active:
            self._apply_dynamic_obstacle(obs["x"], obs["y"], obs["radius_m"])
            self.dynamic_obstacles.append(obs)
        return True

    def point_in_dynamic_obstacle(self, x: float, y: float) -> bool:
        """True if (x, y) lies inside the inflated footprint of any active dynamic obstacle."""
        for obs in self.dynamic_obstacles:
            if math.hypot(x - obs["x"], y - obs["y"]) <= obs["radius_m"] + self.inflation_radius_m:
                return True
        return False

    def reset_dynamic_obstacles(self):
        """Reset occupancy grid to static map and clear all dynamic obstacles."""
        base_grid = getattr(self, "raw_grid", getattr(self, "static_grid", None))
        base_inflated = getattr(self, "raw_inflated_obs", getattr(self, "inflated_obs", None))
        if base_grid is not None:
            self.grid = base_grid.copy()
        if base_inflated is not None:
            self.inflated_obs = base_inflated.copy()
        self.dynamic_obstacles = []

    def world_to_grid(self, x: float, y: float) -> tuple[int, int]:
        return int(math.floor((x - self.ox) / self.res)), int(math.floor((y - self.oy) / self.res))

    def grid_to_world(self, gx: int, gy: int) -> tuple[float, float]:
        return self.ox + (gx + 0.5) * self.res, self.oy + (gy + 0.5) * self.res

    def is_free(self, gx: int, gy: int, check_inflation: bool = True) -> bool:
        if 0 <= gx < self.w and 0 <= gy < self.h:
            # In ROS occupancy grids, pixel 254 is free space.
            # Pixels <= 64 are occupied obstacles/walls; pixels ~205 are unmapped unknown space.
            if self.grid[gy, gx] < 250:
                return False
            if check_inflation and self.inflated_obs[gy, gx] > 0:
                return False
            return True
        return False

    def find_nearest_free(self, gx: int, gy: int, max_r: int = 8, check_inflation: bool = True) -> tuple[int, int]:
        if self.is_free(gx, gy, check_inflation=check_inflation):
            return gx, gy
        for r in range(1, max_r + 1):
            for dx in range(-r, r + 1):
                for dy in range(-r, r + 1):
                    if abs(dx) == r or abs(dy) == r:
                        nx, ny = gx + dx, gy + dy
                        if self.is_free(nx, ny, check_inflation=check_inflation):
                            return nx, ny
        # Fallback to uninflated cell if check_inflation was True
        if check_inflation:
            for r in range(0, max_r + 1):
                for dx in range(-r, r + 1):
                    for dy in range(-r, r + 1):
                        nx, ny = gx + dx, gy + dy
                        if self.is_free(nx, ny, check_inflation=False):
                            return nx, ny
        return gx, gy

    def _apply_dynamic_obstacle(self, x: float, y: float, radius_m: float = 0.10):
        """Rasterize dynamic obstacle into occupancy grid (hard core + robot-radius inflation ring)."""
        gx, gy = self.world_to_grid(x, y)
        cell_r = max(1, int(math.ceil(radius_m / self.res)))
        infl_r = cell_r + max(1, int(math.ceil(self.inflation_radius_m / self.res)))
        for dx in range(-infl_r, infl_r + 1):
            for dy in range(-infl_r, infl_r + 1):
                d2 = dx * dx + dy * dy
                if d2 <= infl_r * infl_r:
                    nx, ny = gx + dx, gy + dy
                    if 0 <= nx < self.w and 0 <= ny < self.h:
                        self.inflated_obs[ny, nx] = 1
                        if d2 <= cell_r * cell_r:
                            self.grid[ny, nx] = 0

    def add_dynamic_obstacle(self, x: float, y: float, radius_m: float = 0.10, ttl_s: float = 60.0):
        """Mark a sensed obstacle into occupancy grid for A* replanning with expiration TTL."""
        self._apply_dynamic_obstacle(x, y, radius_m)
        self.dynamic_obstacles.append({
            "x": x,
            "y": y,
            "radius_m": radius_m,
            "created_at": time.time(),
            "expires_at": time.time() + ttl_s,
        })

    def line_of_sight(self, p1: tuple[float, float], p2: tuple[float, float], clearance_m: float = 0.05, check_inflation: bool = True) -> bool:
        """Check collision-free line-of-sight between two points."""
        dist = math.hypot(p2[0] - p1[0], p2[1] - p1[1])
        if dist < 0.03:
            return True
        steps = max(2, int(math.ceil(dist / (self.res * 0.5))))
        clearance_cells = max(1, int(round(clearance_m / self.res)))
        for s in range(steps + 1):
            alpha = s / steps
            x = p1[0] + alpha * (p2[0] - p1[0])
            y = p1[1] + alpha * (p2[1] - p1[1])
            gx, gy = self.world_to_grid(x, y)
            for cdx in range(-clearance_cells, clearance_cells + 1):
                for cdy in range(-clearance_cells, clearance_cells + 1):
                    if cdx * cdx + cdy * cdy <= clearance_cells * clearance_cells:
                        nx, ny = gx + cdx, gy + cdy
                        if 0 <= nx < self.w and 0 <= ny < self.h:
                            if self.grid[ny, nx] < 250 or (check_inflation and self.inflated_obs[ny, nx] > 0):
                                return False
                        else:
                            return False
        return True

    def smooth_path(self, waypoints: list[tuple[float, float]]) -> list[tuple[float, float]]:
        """Shortcut waypoints using collision-free line-of-sight."""
        # If A* only succeeded in the tight (uninflated) corridor, smooth with the same clearance,
        # otherwise no shortcut passes and the path stays a staircase of 10 cm legs.
        infl = getattr(self, "last_plan_inflated", True)
        if not waypoints or len(waypoints) < 2:
            return waypoints
        if len(waypoints) == 2:
            if self.line_of_sight(waypoints[0], waypoints[1], check_inflation=infl):
                return waypoints
            return waypoints

        smoothed = [waypoints[0]]
        i = 0
        while i < len(waypoints) - 1:
            best_j = i + 1
            for j in range(len(waypoints) - 1, i + 1, -1):
                if self.line_of_sight(smoothed[-1], waypoints[j], check_inflation=infl):
                    best_j = j
                    break
            smoothed.append(waypoints[best_j])
            i = best_j

        filtered = self.filter_micro_waypoints(smoothed)
        # Safety verification: ensure no segment in filtered path violates clearance.
        # Fall back to the line-of-sight shortcut path (not the raw grid staircase).
        for k in range(len(filtered) - 1):
            if not self.line_of_sight(filtered[k], filtered[k + 1], check_inflation=infl):
                return smoothed
        return filtered

    def filter_micro_waypoints(self, waypoints: list[tuple[float, float]], min_seg_m: float = 0.35) -> list[tuple[float, float]]:
        """Filter out micro-segments and merge near-collinear legs."""
        if len(waypoints) <= 2:
            return waypoints

        merged = [waypoints[0]]
        for i in range(1, len(waypoints) - 1):
            p_prev = merged[-1]
            p_curr = waypoints[i]
            p_next = waypoints[i + 1]
            b1 = math.atan2(p_curr[1] - p_prev[1], p_curr[0] - p_prev[0])
            b2 = math.atan2(p_next[1] - p_curr[1], p_next[0] - p_curr[0])
            if abs(normalize_angle(b2 - b1)) > math.radians(15.0):
                merged.append(p_curr)
        merged.append(waypoints[-1])

        filtered = [merged[0]]
        for i in range(1, len(merged) - 1):
            prev = filtered[-1]
            curr = merged[i]
            if math.hypot(curr[0] - prev[0], curr[1] - prev[1]) >= min_seg_m:
                filtered.append(curr)
        filtered.append(merged[-1])
        return filtered

    def plan(
        self,
        start_xy: tuple[float, float],
        goal_xy: tuple[float, float],
        check_inflation: bool = True,
    ) -> list[tuple[float, float]]:
        """A* algorithm: Computes shortest collision-free grid path from start to goal."""
        sx, sy = start_xy
        gx, gy = goal_xy

        sgx_raw, sgy_raw = self.world_to_grid(sx, sy)
        ggx_raw, ggy_raw = self.world_to_grid(gx, gy)

        if not (0 <= sgx_raw < self.w and 0 <= sgy_raw < self.h):
            print(f"  [A* ERROR] Start position ({sx:.2f}, {sy:.2f}) is outside map bounds!")
            return []
        if not (0 <= ggx_raw < self.w and 0 <= ggy_raw < self.h):
            print(f"  [A* ERROR] Goal position ({gx:.2f}, {gy:.2f}) is outside map bounds!")
            return []

        try:
            sgx, sgy = self.find_nearest_free(sgx_raw, sgy_raw, check_inflation=check_inflation)
            ggx, ggy = self.find_nearest_free(ggx_raw, ggy_raw, check_inflation=check_inflation)
        except TypeError:
            sgx, sgy = self.find_nearest_free(sgx_raw, sgy_raw)
            ggx, ggy = self.find_nearest_free(ggx_raw, ggy_raw)

        snap_dist_start = math.hypot((sgx - sgx_raw) * self.res, (sgy - sgy_raw) * self.res)
        snap_dist_goal = math.hypot((ggx - ggx_raw) * self.res, (ggy - ggy_raw) * self.res)
        max_snap_allowed = 0.35
        if snap_dist_start > max_snap_allowed:
            print(f"  [A* ERROR] Start ({sx:.2f}, {sy:.2f}) inside obstacle and snapped {snap_dist_start:.2f}m (> {max_snap_allowed:.2f}m).")
            return []
        if snap_dist_goal > max_snap_allowed:
            print(f"  [A* ERROR] Goal ({gx:.2f}, {gy:.2f}) inside obstacle and snapped {snap_dist_goal:.2f}m (> {max_snap_allowed:.2f}m).")
            return []

        if not self.is_free(sgx, sgy, check_inflation=False) or not self.is_free(ggx, ggy, check_inflation=False):
            print(f"  [A* ERROR] Start or goal is inside an unnavigable obstacle.")
            return []

        actions = [
            (1, 0, 1.0), (-1, 0, 1.0), (0, 1, 1.0), (0, -1, 1.0),
            (1, 1, 1.414), (-1, 1, 1.414), (1, -1, 1.414), (-1, -1, 1.414),
        ]
        open_set = []
        heapq.heappush(open_set, (0.0, 0.0, (sgx, sgy)))
        came_from = {}
        g_cost = {(sgx, sgy): 0.0}

        def h_cost(a, b):
            dx, dy = abs(a[0] - b[0]), abs(a[1] - b[1])
            return (dx + dy) + (1.414 - 2.0) * min(dx, dy)

        reached = False
        while open_set:
            _, cur_g, current = heapq.heappop(open_set)
            if current == (ggx, ggy):
                reached = True
                break
            for adx, ady, step_c in actions:
                nx, ny = current[0] + adx, current[1] + ady
                if not self.is_free(nx, ny, check_inflation=check_inflation):
                    continue
                # Corner-cutting check: do not cut diagonally between blocked orthogonal cells
                if adx != 0 and ady != 0:
                    if not self.is_free(current[0] + adx, current[1], check_inflation=check_inflation) or not self.is_free(current[0], current[1] + ady, check_inflation=check_inflation):
                        continue

                tentative = cur_g + step_c
                if (nx, ny) not in g_cost or tentative < g_cost[(nx, ny)]:
                    g_cost[(nx, ny)] = tentative
                    pri = tentative + h_cost((nx, ny), (ggx, ggy))
                    heapq.heappush(open_set, (pri, tentative, (nx, ny)))
                    came_from[(nx, ny)] = current

        if not reached:
            if check_inflation:
                print(f"  [A* NOTICE] Safety corridor tight; searching tighter collision-free corridor...")
                try:
                    return self.plan(start_xy, goal_xy, check_inflation=False)
                except TypeError:
                    return FastAStarPlanner.plan(self, start_xy, goal_xy, check_inflation=False)
            print(f"  [A* ERROR] No collision-free route found from ({sx:.2f}, {sy:.2f}) to ({gx:.2f}, {gy:.2f})!")
            return []

        self.last_plan_inflated = check_inflation
        cells = [(ggx, ggy)]
        cur = (ggx, ggy)
        while cur in came_from:
            cur = came_from[cur]
            cells.append(cur)
        cells.reverse()

        pts = [self.grid_to_world(c[0], c[1]) for c in cells]
        if len(pts) <= 2:
            return pts

        # Collinear pruning
        pruned = [pts[0]]
        for i in range(1, len(pts) - 1):
            prev = pruned[-1]
            curr = pts[i]
            nxt = pts[i + 1]
            a1 = math.atan2(curr[1] - prev[1], curr[0] - prev[0])
            a2 = math.atan2(nxt[1] - curr[1], nxt[0] - curr[0])
            if abs(normalize_angle(a1 - a2)) > 0.05:
                pruned.append(curr)
        pruned.append(pts[-1])

        return pruned


class MotorDriver:
    """Hardware driver for L298N 4WD chassis + Intel RealSense D455 Gyro & Depth.

    Low CPU & Low Bandwidth:
      - Only reads RealSense hardware gyro and depth frame (no heavy color stream or visual odometry).
      - Tracks real physical yaw using calibrated gyro integration.
      - Measures forward obstacle distance using fast depth ROI slice.
    """

    def __init__(
        self,
        cam_inverted: bool = True,
        motors_inverted: bool = False,
        flip_gyro: bool = False,
        cruise_pwm: Optional[float] = None,
        turn_pwm: Optional[float] = None,
        config: Optional[NavConfig] = None,
    ):
        self.in1 = self.in2 = self.in3 = self.in4 = self.ena = self.enb = None
        self.pipe = None

        # Build or use provided configuration instance
        if config is not None:
            self.config = config
        else:
            self.config = NavConfig(
                cam_inverted=cam_inverted,
                motors_inverted=motors_inverted,
                flip_gyro=flip_gyro,
            )
            if cruise_pwm is not None:
                self.config.cruise_pwm = cruise_pwm
                self.config.speed_mps = speed_for_pwm(cruise_pwm)
            if turn_pwm is not None:
                self.config.turn_pwm = turn_pwm
                self.config.turn_radps = max(0.35, turn_pwm * 1.55)

        self.cam_inverted = self.config.cam_inverted
        self.motors_inverted = self.config.motors_inverted
        self.flip_gyro = self.config.flip_gyro
        self.cruise_pwm = self.config.cruise_pwm
        self.turn_pwm = self.config.turn_pwm
        self.speed_mps = self.config.speed_mps
        self.turn_radps = self.config.turn_radps

        # Inverted front D455: local +Y points UP. In BMI085 IMU coordinate frame,
        # counter-clockwise (turn left) produces negative Y gyro rate.
        # Thus, inverted requires -1.0 multiplier so Turning Left is ALWAYS POSITIVE (+).
        base_sign = -1.0 if self.cam_inverted else 1.0
        self.gyro_sign: float = -base_sign if self.flip_gyro else base_sign

        self.has_camera = False
        self.has_hardware_gyro = False
        self.last_depth: Optional[float] = None
        self.last_depth_ts: float = 0.0
        self.last_depth_age_s: Optional[float] = None
        self.last_color_frame = None
        self.is_moving: bool = False
        self.gyro_bias_y: float = 0.0
        self.last_yaw_rate: float = 0.0
        self.current_yaw_rad: float = 0.0
        self.last_frame_time: float = time.time()
        self.last_gyro_ts: float = 0.0
        self.stop_flag = False
        self.live_pose: Tuple[float, float, float] = (0.0, 0.0, 0.0)
        self.active_waypoints: list[tuple[float, float]] = []
        self.remaining_dist_m: float = 0.0
        self.visual_sectors: dict[str, Optional[float]] = {"left": None, "center": None, "right": None}
        self.depth_frame_ts: float = 0.0       # last time ANY depth frame arrived (freshness)
        self.depth_frame_count: int = 0        # number of depth frames processed (for debouncing)
        self.too_close: bool = False           # obstacle inside D455 minimum range (depth reads as zeros)
        self._last_near_ts: float = 0.0        # last time a valid near (< 0.8m) reading was seen
        self.last_scan: Optional[np.ndarray] = None  # (N, 2) [forward, left] points from latest depth frame
        self._depth_intr = None                # (fx, ppx) of the (flipped) depth image
        self._cam_lock = threading.Lock()

        # 1. Initialize L298N GPIO
        if HAS_GPIO:
            try:
                self.in1 = DigitalOutputDevice(IN1_PIN)
                self.in2 = DigitalOutputDevice(IN2_PIN)
                self.in3 = DigitalOutputDevice(IN3_PIN)
                self.in4 = DigitalOutputDevice(IN4_PIN)
                self.ena = PWMOutputDevice(ENA_PIN, frequency=1000, initial_value=0)
                self.enb = PWMOutputDevice(ENB_PIN, frequency=1000, initial_value=0)
                print(f"  [HARDWARE] L298N GPIO Ready (IN: {IN1_PIN},{IN2_PIN},{IN3_PIN},{IN4_PIN} | PWM: {ENA_PIN},{ENB_PIN})")
            except Exception as e:
                print(f"  [WARNING] GPIO initialization failed: {e}")
                self.in1 = self.in2 = self.in3 = self.in4 = self.ena = self.enb = None
        else:
            print("  [SIMULATION] Running in mock/dry-run mode (no physical GPIO devices).")

        # 2. Automatic Linux IIO sensor device permissions for RealSense IMU
        if sys.platform.startswith("linux"):
            try:
                import subprocess
                subprocess.run(
                    ["sudo", "chmod", "-R", "a+rw", "/sys/bus/iio/devices/"],
                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=1.0
                )
                subprocess.run(
                    "sudo chmod -R a+rw /dev/iio:device* /dev/hidraw* 2>/dev/null",
                    shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=1.0
                )
            except Exception:
                pass

        # 3. Initialize RealSense D455 with Multi-Tier Fallback and Auto Hardware Reset
        try:
            import pyrealsense2 as rs
            ctx = rs.context()
            devs = ctx.query_devices()

            if len(devs) == 0:
                # Brief settling wait in case USB is still enumerating
                time.sleep(0.8)
                devs = ctx.query_devices()

            if len(devs) > 0:
                dev = devs[0]
                name = dev.get_info(rs.camera_info.name) if dev.supports(rs.camera_info.name) else "RealSense D455"
                serial = dev.get_info(rs.camera_info.serial_number) if dev.supports(rs.camera_info.serial_number) else None
                usb = dev.get_info(rs.camera_info.usb_type_descriptor) if dev.supports(rs.camera_info.usb_type_descriptor) else "Unknown"
                print(f"  [D455 HARDWARE] Found {name} (Serial: {serial} | USB: {usb})")

                # Candidate streaming configurations for D455 in order of priority:
                # Tests explicit 200Hz motion format as well as native auto-negotiation
                stream_profiles = [
                    ("640x480 RGB+Depth@15fps + IMU", lambda c: (
                        c.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 15),
                        c.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 15),
                        c.enable_stream(rs.stream.gyro),
                        c.enable_stream(rs.stream.accel),
                    )),
                    ("640x480 RGB+Depth@15fps (Visual Only)", lambda c: (
                        c.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 15),
                        c.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 15),
                    )),
                    ("640x480 Depth@15fps + IMU (200Hz)", lambda c: (
                        c.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 15),
                        c.enable_stream(rs.stream.gyro, rs.format.motion_xyz32f, 200),
                        c.enable_stream(rs.stream.accel, rs.format.motion_xyz32f, 200),
                    )),
                    ("640x480 Depth@15fps + IMU (Auto)", lambda c: (
                        c.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 15),
                        c.enable_stream(rs.stream.gyro),
                        c.enable_stream(rs.stream.accel),
                    )),
                    ("480x270 Depth@15fps + IMU (200Hz)", lambda c: (
                        c.enable_stream(rs.stream.depth, 480, 270, rs.format.z16, 15),
                        c.enable_stream(rs.stream.gyro, rs.format.motion_xyz32f, 200),
                        c.enable_stream(rs.stream.accel, rs.format.motion_xyz32f, 200),
                    )),
                    ("480x270 Depth@15fps + IMU (Auto)", lambda c: (
                        c.enable_stream(rs.stream.depth, 480, 270, rs.format.z16, 15),
                        c.enable_stream(rs.stream.gyro),
                        c.enable_stream(rs.stream.accel),
                    )),
                    ("Depth + Gyro Only (200Hz)", lambda c: (
                        c.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 15),
                        c.enable_stream(rs.stream.gyro, rs.format.motion_xyz32f, 200),
                    )),
                    ("IMU Only (Gyro + Accel @ 200Hz)", lambda c: (
                        c.enable_stream(rs.stream.gyro, rs.format.motion_xyz32f, 200),
                        c.enable_stream(rs.stream.accel, rs.format.motion_xyz32f, 200),
                    )),
                    ("IMU Only (Gyro + Accel Auto)", lambda c: (
                        c.enable_stream(rs.stream.gyro),
                        c.enable_stream(rs.stream.accel),
                    )),
                    ("640x480 Depth@15fps (Depth Only)", lambda c: (
                        c.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 15),
                    )),
                    ("Auto Depth Only", lambda c: (
                        c.enable_stream(rs.stream.depth),
                    )),
                ]

                pipe_started = False
                for desc, setup_fn in stream_profiles:
                    try:
                        self.pipe = rs.pipeline()
                        cfg = rs.config()
                        if serial:
                            cfg.enable_device(serial)
                        setup_fn(cfg)
                        self.pipe.start(cfg)
                        pipe_started = True
                        if "IMU" in desc or "Gyro" in desc:
                            self.has_hardware_gyro = True
                        print(f"  [D455 SENSORS OK] Stream started successfully: {desc}")
                        break
                    except Exception as err:
                        if "IMU" in desc or "Gyro" in desc:
                            err_str = str(err).splitlines()[0] if str(err) else "Unknown"
                            print(f"  [D455 PROBE] Profile '{desc}' unavailable: {err_str}")
                        try:
                            if self.pipe:
                                self.pipe.stop()
                        except Exception:
                            pass
                        self.pipe = None
                        continue

                # If initial attempts failed, USB endpoint might be locked: attempt hardware reset
                if not pipe_started:
                    print("  [D455 RESET] Camera endpoint locked. Performing hardware USB reset...")
                    try:
                        dev.hardware_reset()
                        time.sleep(2.5)  # Wait for D455 to re-enumerate
                        ctx = rs.context()
                        devs = ctx.query_devices()
                        if len(devs) > 0:
                            dev = devs[0]
                            serial = dev.get_info(rs.camera_info.serial_number) if dev.supports(rs.camera_info.serial_number) else None
                            self.pipe = rs.pipeline()
                            cfg = rs.config()
                            if serial:
                                cfg.enable_device(serial)
                            cfg.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 15)
                            cfg.enable_stream(rs.stream.gyro)
                            cfg.enable_stream(rs.stream.accel)
                            self.pipe.start(cfg)
                            pipe_started = True
                            self.has_hardware_gyro = True
                            print("  [D455 SENSORS OK] Recovered and started after hardware reset!")
                    except Exception as reset_err:
                        print(f"  [D455 RESET FAILED] {reset_err}")

                if pipe_started and self.pipe:
                    self.has_camera = True
                    # Warm-up stream and sample stationary gyro bias
                    gyro_samples = []
                    if self.has_hardware_gyro:
                        for _ in range(25):
                            try:
                                frames = self.pipe.wait_for_frames(timeout_ms=1200)
                                gyro_f = frames.first_or_default(rs.stream.gyro)
                                if gyro_f:
                                    gyro_samples.append(self.gyro_sign * gyro_f.as_motion_frame().get_motion_data().y)
                            except Exception:
                                time.sleep(0.04)
                                continue

                        if len(gyro_samples) >= 6:
                            s_mean = float(np.mean(gyro_samples))
                            s_std = float(np.std(gyro_samples))
                            if s_std < 0.03 and abs(s_mean) < 0.05:
                                self.gyro_bias_y = s_mean
                                print(f"  [D455 GYRO] Calibrated stationary bias = {self.gyro_bias_y:+.4f} rad/s")
                            else:
                                self.gyro_bias_y = 0.0
                                print(f"  [D455 GYRO] Bias unstable ({s_mean:+.4f} ± {s_std:.4f} rad/s). Using zero-bias.")

                        print(f"  [D455 SENSORS READY] Depth & Gyro telemetry ONLINE (sign={self.gyro_sign:+.1f}).")
                    else:
                        print(f"  [D455 DEPTH ONLY] Depth obstacle avoidance ONLINE | Hardware Gyro OFFLINE.")
                        print(f"  [D455 NOTICE] To unlock 200Hz Gyroscope: run with 'sudo -E python3 nav.py' or check Linux permissions.")
                else:
                    print("  [D455 SENSORS ERROR] Could not start any RealSense stream profile.")
                    self.pipe = None
                    self.has_camera = False
            else:
                print("  [D455 SENSORS ERROR] No RealSense camera found on USB bus.")
                print("                   Ensure D455 is firmly connected to a blue USB 3.0 port.")
                self.pipe = None
                self.has_camera = False
        except Exception as e:
            print(f"  [D455 SENSORS ERROR] RealSense startup failed: {e}")
            self.pipe = None
            self.has_camera = False

    def read_motion_and_depth(self) -> tuple[float, float, float, Optional[float]]:
        """Compatibility helper for live visualizers: returns (x, y, yaw_rad, depth_m)."""
        yaw, depth = self.read_sensors()
        return (self.live_pose[0], self.live_pose[1], yaw, depth)

    def read_sensors(self) -> tuple[float, Optional[float]]:
        """Read synchronized gyro and depth from D455.
        Drains all queued motion and depth frames without blocking (only waits briefly when
        the queue is empty) so the control loops run fast and every 200Hz gyro sample is integrated.
        Returns: (current_yaw_rad, forward_obstacle_depth_m). Depth None means path clear.
        """
        if not self.pipe:
            now = time.time()
            self.last_depth_age_s = (now - self.depth_frame_ts) if self.depth_frame_ts > 0.0 else None
            return (self.current_yaw_rad, self.last_depth)

        with self._cam_lock:
            try:
                import pyrealsense2 as rs
                now = time.time()

                # Collect all queued framesets / motion frames without blocking
                frames_list = []
                for _ in range(80):
                    try:
                        polled = self.pipe.poll_for_frames()
                        if isinstance(polled, tuple):
                            if len(polled) == 2 and polled[0] and polled[1]:
                                frames_list.append(polled[1])
                            else:
                                break
                        elif polled:
                            frames_list.append(polled)
                        else:
                            break
                    except Exception:
                        break
                if not frames_list:
                    try:
                        f0 = self.pipe.wait_for_frames(timeout_ms=40)
                        if f0:
                            frames_list.append(f0)
                    except Exception:
                        pass

                for frames in frames_list:
                    # 1. Process Gyro Motion Frame
                    gyro_f = None
                    if hasattr(frames, "first_or_default"):
                        gyro_f = frames.first_or_default(rs.stream.gyro)
                    elif hasattr(frames, "is_motion_frame") and frames.is_motion_frame():
                        if frames.get_profile().stream_type() == rs.stream.gyro:
                            gyro_f = frames.as_motion_frame()

                    if gyro_f:
                        try:
                            g_data = gyro_f.as_motion_frame().get_motion_data()
                            raw_rate_y = self.gyro_sign * g_data.y
                            calibrated_rate = raw_rate_y - self.gyro_bias_y
                            self.last_yaw_rate = calibrated_rate

                            # Accurate dt per gyro sample using hardware timestamp
                            ts_sec = gyro_f.get_timestamp() * 1e-3
                            if self.last_gyro_ts > 0.0:
                                dt = ts_sec - self.last_gyro_ts
                                if dt <= 0.0 or dt > 0.5:
                                    dt = 0.005
                            else:
                                dt = 0.005
                            self.last_gyro_ts = ts_sec

                            if abs(calibrated_rate) > 0.005:
                                self.current_yaw_rad = normalize_angle(self.current_yaw_rad + calibrated_rate * dt)
                        except Exception:
                            pass

                    # 2. Process Depth Frame
                    depth_f = None
                    if hasattr(frames, "get_depth_frame"):
                        depth_f = frames.get_depth_frame()
                    elif hasattr(frames, "is_depth_frame") and frames.is_depth_frame():
                        depth_f = frames.as_depth_frame()

                    if depth_f:
                        d_img = np.asanyarray(depth_f.get_data())
                        if self.cam_inverted:
                            # Sensor mounted upside down: flip horizontally and vertically (-1)
                            d_img = cv2.flip(d_img, -1)

                        h, w = d_img.shape
                        y1, y2 = int(h * 0.20), int(h * 0.50)  # above horizon: keeps the floor out
                        x1, x2 = int(w * 0.22), int(w * 0.78)

                        roi = d_img[y1:y2, x1:x2]
                        valid_mask = (roi > 60) & (roi < 4000)
                        valid = roi[valid_mask]
                        # A frame arrived: depth is fresh even if nothing is in range (= path clear)
                        self.depth_frame_ts = now
                        self.depth_frame_count += 1

                        # Pseudo laser scan for localization: nearest valid depth per column in the band
                        try:
                            if self._depth_intr is None:
                                intr = depth_f.get_profile().as_video_stream_profile().get_intrinsics()
                                ppx = (w - 1 - intr.ppx) if self.cam_inverted else intr.ppx
                                self._depth_intr = (float(intr.fx), float(ppx))
                            fx, ppx = self._depth_intr
                            cols = np.arange(4, w - 4, 6)
                            band = d_img[y1:y2, cols].astype(np.float32)
                            band[(band < 450) | (band > 4000)] = np.inf
                            z = band.min(axis=0) * 0.001
                            ok = np.isfinite(z)
                            if np.count_nonzero(ok) >= 20:
                                zf = z[ok]
                                left = -(cols[ok] - ppx) * zf / fx
                                self.last_scan = np.stack([zf, left], axis=1)
                            else:
                                self.last_scan = None
                        except Exception:
                            self.last_scan = None
                        if len(valid) >= 40:
                            # Low percentile so thin objects (chair legs, poles) are not averaged away
                            depth_val = float(np.percentile(valid, 10)) * 0.001
                        else:
                            depth_val = None
                        if depth_val is not None and depth_val < 0.80:
                            self._last_near_ts = now

                        # D455 returns 0 for anything closer than its minimum range (~0.4m).
                        # If the forward ROI goes mostly invalid right after we saw something near,
                        # the obstacle is now too close to measure: treat it as blocked.
                        invalid_frac = 1.0 - float(np.count_nonzero(valid_mask)) / max(1, valid_mask.size)
                        self.too_close = invalid_frac > 0.70 and (now - self._last_near_ts) < 1.5
                        if self.too_close:
                            depth_val = 0.20
                        self.last_depth = depth_val
                        self.last_depth_ts = now

                        # Fast 3-Sector Visual Guidance (<1ms slice)
                        r_l = d_img[y1:y2, int(w * 0.08):int(w * 0.36)]
                        r_c = d_img[y1:y2, int(w * 0.36):int(w * 0.64)]
                        r_r = d_img[y1:y2, int(w * 0.64):int(w * 0.92)]

                        def _eval_sec(r_sub):
                            v = r_sub[(r_sub > 60) & (r_sub < 4000)]
                            return float(np.percentile(v, 10)) * 0.001 if len(v) >= 25 else None

                        self.visual_sectors = {
                            "left": _eval_sec(r_l),
                            "center": _eval_sec(r_c),
                            "right": _eval_sec(r_r),
                        }

                    # 3. Process Color Frame (or generate colormapped depth visualizer)
                    color_f = None
                    if hasattr(frames, "get_color_frame"):
                        color_f = frames.get_color_frame()
                    elif hasattr(frames, "first_or_default"):
                        color_f = frames.first_or_default(rs.stream.color)

                    if color_f:
                        c_img = np.asanyarray(color_f.get_data())
                        if self.cam_inverted:
                            c_img = cv2.flip(c_img, -1)
                        self.last_color_frame = c_img
                    elif depth_f:
                        # Fallback: Colorize depth frame into a Turbo colormap with ROI box
                        d_vis = cv2.convertScaleAbs(d_img, alpha=0.035)
                        d_vis_color = cv2.applyColorMap(d_vis, cv2.COLORMAP_TURBO)
                        cv2.rectangle(d_vis_color, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        self.last_color_frame = d_vis_color

                self.last_frame_time = now
                self.last_depth_age_s = (now - self.depth_frame_ts) if self.depth_frame_ts > 0.0 else None

                self.live_pose = (self.live_pose[0], self.live_pose[1], self.current_yaw_rad)
                return (self.current_yaw_rad, self.last_depth)
            except Exception:
                now = time.time()
                self.last_depth_age_s = (now - self.depth_frame_ts) if self.depth_frame_ts > 0.0 else None
                return (self.current_yaw_rad, self.last_depth)

    def get_yaw_rad(self) -> float:
        return self.current_yaw_rad

    def set_yaw_rad(self, yaw_rad: float):
        self.current_yaw_rad = yaw_rad

    def get_yaw_rate(self) -> float:
        return getattr(self, "last_yaw_rate", 0.0)

    def get_forward_depth(self) -> Optional[float]:
        return self.last_depth

    def stop(self):
        """Operator/emergency stop: halt motors AND abort the active mission."""
        self.stop_flag = True
        self._halt_motors()

    def clear_stop(self):
        """Re-arm the driver for a new mission after an operator stop."""
        self.stop_flag = False

    def _halt_motors(self):
        """Immediately stop all 4 motors and zero PWM (does not abort the mission)."""
        self.is_moving = False
        if self.ena and self.enb:
            self.ena.off()
            self.enb.off()
            self.in1.off()
            self.in2.off()
            self.in3.off()
            self.in4.off()
        time.sleep(0.04)

    def drive_raw(
        self,
        left_forward: bool,
        right_forward: bool,
        speed: Optional[float] = None,
        speed_right: Optional[float] = None,
    ):
        """Low-level differential drive with standard L298N wiring:
        Left motor:  IN1=1, IN2=0 (forward), IN1=0, IN2=1 (reverse)
        Right motor: IN3=1, IN4=0 (forward), IN3=0, IN4=1 (reverse)
        """
        act_left = speed if speed is not None else self.cruise_pwm
        act_right = speed_right if speed_right is not None else act_left

        lf = not left_forward if self.motors_inverted else left_forward
        rf = not right_forward if self.motors_inverted else right_forward

        if self.ena and self.enb:
            self.in1.value = int(lf)
            self.in2.value = int(not lf)
            self.in3.value = int(rf)
            self.in4.value = int(not rf)
            self.ena.value = max(0.0, min(1.0, act_left))
            self.enb.value = max(0.0, min(1.0, act_right))

    def pivot(self, angle_rad: float) -> MotionResult:
        """Pivot in place tracking real physical rotation from D455 hardware gyro.
        Uses closed-loop heading feedback on the UNWRAPPED rotation (safe for turns near 180 deg),
        slows down near the target to limit overshoot, and detects runaway divergence (safety halt).
        """
        start_yaw = self.current_yaw_rad
        target_yaw = normalize_angle(start_yaw + angle_rad)
        is_left = angle_rad > 0
        dir_str = "LEFT" if is_left else "RIGHT"
        target_deg = math.degrees(abs(angle_rad))

        if abs(angle_rad) < math.radians(self.config.pivot_tolerance_deg):
            return MotionResult(
                success=True,
                halt_reason="completed",
                final_pose=self.live_pose,
                estimated_displacement=(0.0, 0.0),
                elapsed_s=0.0,
                depth_m=self.last_depth,
                depth_age_s=self.last_depth_age_s,
                actual_turn_rad=0.0,
                details="Angle within tolerance",
            )

        if self.stop_flag:
            return MotionResult(False, "stopped", self.live_pose, (0.0, 0.0), 0.0, self.last_depth, self.last_depth_age_s, 0.0, "Aborted before pivot")

        self.is_moving = True
        cur_pwm = max(0.48, min(1.0, self.turn_pwm))
        slow_pwm = max(0.48, cur_pwm * 0.80)
        print(f"\n  [PIVOT {dir_str}] Target: {target_deg:.1f}° (Base PWM = {cur_pwm*100:.0f}%)")

        if not self.has_hardware_gyro:
            dur = target_deg / math.degrees(self.turn_radps)
            print(f"  [DEAD-RECKONING PIVOT] Turning for {dur:.2f}s at {math.degrees(self.turn_radps):.1f}°/s...")
            if self.ena:
                self.drive_raw(not is_left, is_left, cur_pwm, cur_pwm)
                time.sleep(dur)
                self._halt_motors()
            self.is_moving = False
            self.current_yaw_rad = target_yaw
            self.live_pose = (self.live_pose[0], self.live_pose[1], self.current_yaw_rad)
            print(f"  [PIVOT DONE] Headed to {math.degrees(self.current_yaw_rad):+.1f}°")
            return MotionResult(
                success=True,
                halt_reason="completed",
                final_pose=self.live_pose,
                estimated_displacement=(0.0, 0.0),
                elapsed_s=dur,
                depth_m=self.last_depth,
                depth_age_s=self.last_depth_age_s,
                actual_turn_rad=angle_rad,
                details="Dead reckoning pivot completed",
            )

        t_start = time.time()
        t_timeout = t_start + min(10.0, 2.0 + target_deg / 20.0)
        boost = 0.0

        # Breakaway torque burst to overcome skid-steer floor friction
        if self.ena:
            self.drive_raw(not is_left, is_left, self.config.breakaway_pwm, self.config.breakaway_pwm)
            time.sleep(self.config.breakaway_burst_s)
            self.drive_raw(not is_left, is_left, cur_pwm, cur_pwm)

        runaway_ticks = 0
        halt_reason = "completed"
        success = True
        rotated = 0.0                 # unwrapped rotation since start (rad)
        prev_yaw = start_yaw

        try:
            while time.time() < t_timeout:
                if self.stop_flag:
                    halt_reason = "stopped"
                    success = False
                    break
                self.read_sensors()

                current_yaw = self.current_yaw_rad
                rotated += normalize_angle(current_yaw - prev_yaw)
                prev_yaw = current_yaw
                remaining_rad = angle_rad - rotated
                remaining_deg = math.degrees(abs(remaining_rad))

                # Stop when aligned within configured angular tolerance
                if remaining_deg <= self.config.pivot_tolerance_deg:
                    break

                elapsed = time.time() - t_start

                # --- DIVERGENCE DETECTION & SAFETY HALT ---
                # If chassis rotates clearly in the wrong direction, stop immediately.
                commanded_dir = 1.0 if is_left else -1.0
                if elapsed > 0.35 and commanded_dir * rotated < -0.15:
                    runaway_ticks += 1
                    if runaway_ticks >= 4:
                        print(f"\n  [GYRO FAULT] Rotation contradicts commanded direction! Immediate safety halt.")
                        print(f"  [GYRO TIP] Current polarity: sign={self.gyro_sign:+.1f}. If robot turned physically left but yaw decreased, pass --flip-gyro or click 'Invert Gyro' on Web UI.")
                        halt_reason = "gyro_divergence"
                        success = False
                        break
                else:
                    runaway_ticks = 0

                # Closed-loop direction: steer towards target, slower in the last 20 deg
                need_left = remaining_rad > 0
                # Stall compensation: if turning slower than ~20 deg/s, add PWM (up to +0.30)
                if elapsed > 0.4 and abs(self.last_yaw_rate) < math.radians(20.0):
                    boost = min(0.30, boost + 0.01)
                elif abs(self.last_yaw_rate) > math.radians(45.0):
                    boost = max(0.0, boost - 0.01)
                pwm = min(1.0, (slow_pwm if remaining_deg < 20.0 else cur_pwm) + boost)
                if self.ena:
                    self.drive_raw(not need_left, need_left, pwm, pwm)

                sys.stdout.write(
                    f"\r  [PIVOT HUD {dir_str}] Target: {math.degrees(target_yaw):+5.1f}° | "
                    f"Current: {math.degrees(current_yaw):+5.1f}° | Remaining: {remaining_deg:4.1f}°"
                )
                sys.stdout.flush()
                time.sleep(0.01)
        finally:
            self._halt_motors()
            time.sleep(0.10)  # Settle chassis before final reading
            print()

        final_yaw, final_depth = self.read_sensors()
        rotated += normalize_angle(final_yaw - prev_yaw)
        self.live_pose = (self.live_pose[0], self.live_pose[1], final_yaw)
        elapsed_total = time.time() - t_start

        if halt_reason == "completed" and time.time() >= t_timeout:
            if math.degrees(abs(angle_rad - rotated)) > self.config.pivot_tolerance_deg * 2:
                halt_reason = "timeout"
                success = False

        status_tag = "OK" if success else "HALT"
        print(f"  [PIVOT DONE - {status_tag}] Turned {math.degrees(abs(rotated)):.1f}° to Heading {math.degrees(final_yaw):+.1f}° ({halt_reason})")
        return MotionResult(
            success=success,
            halt_reason=halt_reason,
            final_pose=self.live_pose,
            estimated_displacement=(0.0, 0.0),
            elapsed_s=elapsed_total,
            depth_m=final_depth,
            depth_age_s=self.last_depth_age_s,
            actual_turn_rad=rotated,
            details=f"Target {target_deg:.1f}deg, actual {math.degrees(abs(rotated)):.1f}deg",
        )

    def drive_straight(
        self,
        distance_m: float,
        bearing: Optional[float] = None,
        start_p: Optional[tuple[float, float]] = None,
        leg_idx: int = 1,
        total_legs: int = 1,
    ) -> MotionResult:
        """Drive to the point `distance_m` along `bearing` from `start_p`.
        Steers toward that goal point (not just a fixed heading) so lateral drift is corrected,
        slows down near obstacles and halts at the configured obstacle distance.
        """
        if distance_m <= 0.015:
            return MotionResult(
                success=True,
                halt_reason="completed",
                final_pose=self.live_pose,
                estimated_displacement=(0.0, 0.0),
                elapsed_s=0.0,
                depth_m=self.last_depth,
                depth_age_s=self.last_depth_age_s,
                actual_turn_rad=0.0,
                details="Leg distance negligible",
            )

        if self.stop_flag:
            return MotionResult(False, "stopped", self.live_pose, (0.0, 0.0), 0.0, self.last_depth, self.last_depth_age_s, 0.0, "Aborted before leg")

        leg_heading = bearing if bearing is not None else self.current_yaw_rad
        print(f"\n  [LEG {leg_idx:02d}/{total_legs:02d}] Distance: {distance_m:.2f} m | Heading: {math.degrees(leg_heading):+.1f}°")

        self.is_moving = True
        cur_x, cur_y = start_p if start_p else (self.live_pose[0], self.live_pose[1])
        start_x, start_y = cur_x, cur_y
        ux, uy = math.cos(leg_heading), math.sin(leg_heading)
        goal_x, goal_y = start_x + distance_m * ux, start_y + distance_m * uy
        start_yaw = self.current_yaw_rad
        halt_reason = "completed"
        last_depth = self.last_depth
        along = 0.0
        obs_hits, obs_frame = 0, -1

        t_start = time.time()
        t_max_timeout = t_start + max(distance_m / max(0.06, self.speed_mps) * 2.5, 4.0)
        cur_drive_pwm = max(0.42, min(1.0, self.cruise_pwm))
        min_pwm = 0.42

        # Check for an obstacle before even starting to move
        _, d0 = self.read_sensors()
        if d0 is not None and d0 <= self.config.obstacle_halt_dist_m:
            self.is_moving = False
            print(f"  [OBSTACLE DETECTED] Obstacle at {d0:.2f} m ahead before leg start.")
            return MotionResult(False, "obstacle", self.live_pose, (0.0, 0.0), 0.0, d0, self.last_depth_age_s, 0.0, "Blocked at leg start")

        # High-torque burst to overcome floor friction
        burst_s = self.config.breakaway_burst_s
        burst_pwm = self.config.breakaway_pwm
        if self.ena:
            self.drive_raw(True, True, burst_pwm, burst_pwm)
            time.sleep(burst_s)
        # Wheels are still accelerating during the burst: count it at cruise speed
        burst_dist = self.speed_mps * burst_s
        cur_x += burst_dist * math.cos(self.current_yaw_rad)
        cur_y += burst_dist * math.sin(self.current_yaw_rad)
        self.live_pose = (cur_x, cur_y, self.current_yaw_rad)
        t_last = time.time()

        try:
            while time.time() < t_max_timeout:
                if self.stop_flag:
                    halt_reason = "stopped"
                    break

                # Progress along the leg (projection on leg direction)
                along = (cur_x - start_x) * ux + (cur_y - start_y) * uy
                remaining = distance_m - along
                if remaining <= 0.0:
                    break

                # 1. Read sensors (Gyro + Obstacle Depth)
                current_yaw, measured_depth = self.read_sensors()
                now = time.time()
                dt = max(0.0, min(0.25, now - t_last))
                t_last = now
                if measured_depth is not None:
                    last_depth = measured_depth

                # 2. SENSOR STALENESS SAFETY HALT (no depth frames at all)
                if self.has_camera and (self.last_depth_age_s is not None and self.last_depth_age_s > self.config.max_depth_age_s):
                    print(f"\n  [SAFETY HALT] No depth frames for {self.last_depth_age_s:.2f}s! Halting.")
                    halt_reason = "sensor_stale"
                    break

                # 3. OBSTACLE AVOIDANCE: stop if obstacle directly ahead within halt distance
                if measured_depth is not None and measured_depth <= self.config.obstacle_halt_dist_m:
                    if self.depth_frame_count != obs_frame:
                        obs_frame = self.depth_frame_count
                        obs_hits += 1
                else:
                    obs_hits = 0
                if obs_hits >= 2:
                    tag = " (inside camera min range)" if self.too_close else ""
                    print(f"\n  [OBSTACLE DETECTED] Obstacle at {measured_depth:.2f} m directly ahead{tag}! Halting.")
                    halt_reason = "obstacle"
                    break

                # 4. Steering: aim at the leg goal point to cancel lateral drift
                if remaining > 0.15:
                    target_heading = math.atan2(goal_y - cur_y, goal_x - cur_x)
                else:
                    target_heading = leg_heading
                heading_err = normalize_angle(target_heading - current_yaw)
                if abs(heading_err) > math.radians(25.0) and remaining > 0.10:
                    print(f"\n  [MISALIGNED] Heading off by {math.degrees(heading_err):+.1f}°. Re-aligning.")
                    halt_reason = "misaligned"
                    break
                trim = max(-0.18, min(0.18, 0.60 * heading_err))

                # Side clearance: steer away from the closer side if something is near
                sec = self.visual_sectors or {}
                d_l = sec.get("left")
                d_r = sec.get("right")
                side_thresh = 0.55
                if d_l is not None and d_l < side_thresh and (d_r is None or d_r > d_l + 0.10):
                    trim -= min(0.08, 0.08 * (side_thresh - d_l) / 0.15)  # turn right, away from left
                elif d_r is not None and d_r < side_thresh and (d_l is None or d_l > d_r + 0.10):
                    trim += min(0.08, 0.08 * (side_thresh - d_r) / 0.15)  # turn left, away from right
                trim = max(-0.20, min(0.20, trim))

                # Slow down as an obstacle approaches, and for the final few cm of the leg
                base_pwm = cur_drive_pwm
                if measured_depth is not None and measured_depth < self.config.obstacle_slow_dist_m:
                    span = max(0.05, self.config.obstacle_slow_dist_m - self.config.obstacle_halt_dist_m)
                    frac = max(0.0, (measured_depth - self.config.obstacle_halt_dist_m) / span)
                    base_pwm = min_pwm + (cur_drive_pwm - min_pwm) * frac
                if remaining < 0.12:
                    base_pwm = min(base_pwm, max(min_pwm, cur_drive_pwm * 0.85))

                left_pwm = max(min_pwm - 0.02, min(1.0, base_pwm - trim))
                right_pwm = max(min_pwm - 0.02, min(1.0, base_pwm + trim))

                if self.ena:
                    self.drive_raw(True, True, speed=left_pwm, speed_right=right_pwm)

                # 5. Integrate displacement with PWM-proportional calibrated velocity
                effective_speed = self.speed_mps * ((left_pwm + right_pwm) / (2.0 * cur_drive_pwm))
                step_d = effective_speed * dt
                cur_x += step_d * math.cos(current_yaw)
                cur_y += step_d * math.sin(current_yaw)
                self.live_pose = (cur_x, cur_y, current_yaw)

                depth_str = f"{measured_depth:4.2f}m" if measured_depth is not None else "clear"
                sys.stdout.write(
                    f"\r  [DRIVING] Leg {leg_idx}/{total_legs} ({max(0.0, along):4.2f}/{distance_m:4.2f}m) | "
                    f"Pose: ({cur_x:+5.2f}, {cur_y:+5.2f}) | Yaw: {math.degrees(current_yaw):+5.1f}° | Depth: {depth_str}"
                )
                sys.stdout.flush()
                time.sleep(0.01)
        finally:
            self._halt_motors()
            time.sleep(0.05)  # Settle chassis
            print()

        final_yaw, final_depth = self.read_sensors()
        if final_depth is not None:
            last_depth = final_depth
        self.live_pose = (cur_x, cur_y, final_yaw)
        dx = cur_x - start_x
        dy = cur_y - start_y
        elapsed_total = time.time() - t_start
        along = dx * ux + dy * uy

        if halt_reason == "completed" and along < distance_m * 0.90:
            halt_reason = "timeout"

        return MotionResult(
            success=(halt_reason == "completed"),
            halt_reason=halt_reason,
            final_pose=self.live_pose,
            estimated_displacement=(dx, dy),
            elapsed_s=elapsed_total,
            depth_m=last_depth,
            depth_age_s=self.last_depth_age_s,
            actual_turn_rad=normalize_angle(final_yaw - start_yaw),
            details=f"Travelled {along:.2f}/{distance_m:.2f}m",
        )

    def backup(self, distance_m: float = 0.12, heading_rad: Optional[float] = None) -> MotionResult:
        """Safely reverse away from an obstacle to create clearance for pivoting."""
        print(f"\n  [OBSTACLE CLEARANCE] Backing up {distance_m:.2f} m...")
        t_start = time.time()
        t_timeout = t_start + max(1.2, (distance_m / max(0.06, self.speed_mps)) * 2.0 + 0.4)
        accumulated_back = 0.0
        reverse_speed = self.speed_mps * 0.90
        cur_drive_pwm = max(0.42, min(1.0, self.cruise_pwm))
        self.is_moving = True

        # Breakaway burst in reverse to break tire scrub
        burst_s = self.config.breakaway_burst_s
        burst_pwm = self.config.breakaway_pwm
        if self.ena:
            self.drive_raw(False, False, burst_pwm, burst_pwm)
            time.sleep(burst_s)
            self.drive_raw(False, False, cur_drive_pwm, cur_drive_pwm)
        accumulated_back += reverse_speed * burst_s
        t_last = time.time()

        try:
            while accumulated_back < distance_m and time.time() < t_timeout:
                if self.stop_flag:
                    break
                self.read_sensors()
                now = time.time()
                dt = max(0.0, min(0.25, now - t_last))
                t_last = now
                accumulated_back += reverse_speed * dt
                time.sleep(0.02)
        finally:
            self._halt_motors()
            time.sleep(0.05)

        cur_yaw, final_depth = self.read_sensors()
        eff_heading = heading_rad if heading_rad is not None else cur_yaw
        dx = -accumulated_back * math.cos(eff_heading)
        dy = -accumulated_back * math.sin(eff_heading)
        new_x = self.live_pose[0] + dx
        new_y = self.live_pose[1] + dy
        self.live_pose = (new_x, new_y, cur_yaw)

        return MotionResult(
            success=True,
            halt_reason="completed",
            final_pose=self.live_pose,
            estimated_displacement=(dx, dy),
            elapsed_s=time.time() - t_start,
            depth_m=final_depth,
            depth_age_s=self.last_depth_age_s,
            actual_turn_rad=0.0,
            details=f"Reversed {accumulated_back:.2f}m",
        )

    def cleanup(self):
        self.stop()
        if self.pipe:
            try:
                self.pipe.stop()
            except Exception:
                pass
        if self.ena:
            for dev in [self.ena, self.enb, self.in1, self.in2, self.in3, self.in4]:
                if dev:
                    try:
                        dev.close()
                    except Exception:
                        pass


def _align_heading(driver: MotorDriver, pose: tuple[float, float, float], turn_ang: float) -> tuple[float, float, float]:
    """Pivot by turn_ang with gyro-divergence recovery. Returns the new (x, y, heading)."""
    robot_x, robot_y, heading = pose
    if abs(turn_ang) <= math.radians(driver.config.pivot_tolerance_deg):
        return pose
    target_heading = normalize_angle(heading + turn_ang)
    pivot_res = driver.pivot(turn_ang)
    if not pivot_res.success and pivot_res.halt_reason == "gyro_divergence":
        old_sign = driver.gyro_sign
        driver.gyro_sign = -driver.gyro_sign
        driver.config.flip_gyro = not driver.config.flip_gyro
        print(
            f"  [GYRO AUTO-CORRECT] Divergence detected. Inverting polarity while stationary "
            f"({old_sign:+.1f} -> {driver.gyro_sign:+.1f}). Retrying alignment..."
        )
        _, _, heading = pivot_res.final_pose
        pivot_res = driver.pivot(normalize_angle(target_heading - heading))
        if not pivot_res.success and pivot_res.halt_reason == "gyro_divergence":
            print("  [PIVOT DEAD RECKONING] Completing turn using calibrated time-based rotation...")
            _, _, heading = pivot_res.final_pose
            retry = normalize_angle(target_heading - heading)
            is_left_turn = retry > 0
            pwm = driver.config.turn_pwm
            driver.drive_raw(not is_left_turn, is_left_turn, pwm, pwm)
            time.sleep(abs(retry) / max(0.1, driver.turn_radps))
            driver._halt_motors()
            driver.current_yaw_rad = target_heading
            driver.live_pose = (robot_x, robot_y, target_heading)
            print(f"  [PIVOT RECOVERY DONE] Aligned to {math.degrees(target_heading):+.1f} deg. Continuing mission.")
            return (robot_x, robot_y, target_heading)
    robot_x, robot_y, heading = pivot_res.final_pose
    driver.live_pose = (robot_x, robot_y, heading)
    return (robot_x, robot_y, heading)


def _mark_sensed_obstacles(driver: MotorDriver, planner: FastAStarPlanner, robot_x: float, robot_y: float,
                           heading: float, center_dist: Optional[float]) -> list[tuple[float, float]]:
    """Project the obstacle(s) seen by the depth sectors into the planner grid. Returns marked points."""
    cfg = driver.config
    ttl = cfg.dynamic_obstacle_ttl_s
    radius = cfg.dynamic_obstacle_radius_m
    # Sector centres sit at 22% / 50% / 78% of the image width
    side_off = math.radians((0.50 - 0.22) * cfg.depth_hfov_deg)
    sec = driver.visual_sectors or {}
    marks = []
    c_d = center_dist if center_dist is not None else sec.get("center")
    if c_d is None or driver.too_close:
        c_d = min(c_d, 0.30) if c_d is not None else 0.30
    marks.append((c_d, 0.0))
    for key, off in (("left", side_off), ("right", -side_off)):
        d = sec.get(key)
        if d is not None and d < cfg.obstacle_halt_dist_m + 0.10:
            marks.append((d, off))
    pts = []
    for d, off in marks:
        # Place the obstacle centre a little beyond the measured surface
        r = d + radius * 0.5
        ox = robot_x + r * math.cos(heading + off)
        oy = robot_y + r * math.sin(heading + off)
        planner.add_dynamic_obstacle(ox, oy, radius_m=radius, ttl_s=ttl)
        pts.append((ox, oy))
    return pts


def localize_pose(driver: MotorDriver, planner: FastAStarPlanner,
                  pose: tuple[float, float, float], tag: str = "") -> tuple[float, float, float]:
    """While stationary, match a few depth scans to the static map and correct drift if the match is good."""
    cfg = driver.config
    if not cfg.localize or not driver.has_camera or not hasattr(planner, "match_scan"):
        return pose
    scans, seen, t_end = [], -1, time.time() + 0.45
    while time.time() < t_end:
        driver.read_sensors()
        if driver.last_scan is not None and driver.depth_frame_count != seen:
            seen = driver.depth_frame_count
            scans.append(driver.last_scan)
        time.sleep(0.03)
    if not scans:
        return pose
    pts = np.concatenate(scans, axis=0)
    if len(pts) < 40:
        return pose
    best, best_sc, cur_sc, ambiguous = planner.match_scan(pose, pts, cfg.loc_max_shift_m, cfg.loc_max_rot_deg)
    dxy = math.hypot(best[0] - pose[0], best[1] - pose[1])
    dth = math.degrees(normalize_angle(best[2] - pose[2]))
    if cur_sc >= 0.80:
        print(f"  [LOCALIZE{(' ' + tag) if tag else ''}] Pose matches map well (match {cur_sc:.2f}). No correction.")
        return pose
    if ambiguous:
        print(f"  [LOCALIZE{(' ' + tag) if tag else ''}] View is ambiguous (e.g. one flat wall). Pose kept.")
        return pose
    if best_sc >= cfg.loc_min_score and best_sc - cur_sc >= cfg.loc_min_gain and (dxy > 0.02 or abs(dth) > 0.5):
        print(f"  [LOCALIZE{(' ' + tag) if tag else ''}] Corrected pose by {dxy:.2f} m, {dth:+.1f}° "
              f"(match {cur_sc:.2f} -> {best_sc:.2f}) -> ({best[0]:+.2f}, {best[1]:+.2f}, {math.degrees(best[2]):+.1f}°)")
        driver.set_yaw_rad(best[2])
        driver.live_pose = best
        return best
    print(f"  [LOCALIZE{(' ' + tag) if tag else ''}] Pose kept (match {cur_sc:.2f}, best {best_sc:.2f}).")
    return pose


def navigate_to_target(
    tid: str,
    tinfo: dict,
    cur_pose: tuple[float, float, float],
    sem_entities: dict,
    driver: MotorDriver,
    planner: Optional[FastAStarPlanner] = None,
    dry_run: bool = False,
    step_prefix: str = "",
) -> tuple[float, float, float]:
    """Execute autonomous A* navigation to the target with real-time D455 obstacle avoidance."""
    cur_x, cur_y, cur_yaw = cur_pose
    target_x, target_y, target_yaw = tinfo["approach"]
    cfg = driver.config

    if planner is None:
        planner = FastAStarPlanner(FUSED_DIR)

    # New mission: re-arm after any previous operator stop
    driver.clear_stop()
    planner.cleanup_expired_dynamic_obstacles()

    # 1. ALWAYS compute collision-free path with A* algorithm on occupancy grid
    print("=" * 64)
    print(f"MISSION: Navigate to '{tid}' ({tinfo.get('desc', tid)})")
    print("=" * 64)
    print(f"  Start:  ({cur_x:+.2f} m, {cur_y:+.2f} m, {math.degrees(cur_yaw):+.1f}°)")
    print(f"  Target: ({target_x:+.2f} m, {target_y:+.2f} m, {math.degrees(target_yaw):+.1f}°)")
    print(f"  [A* PLANNER] Searching optimal collision-free path on occupancy map...")

    raw_path = planner.plan((cur_x, cur_y), (target_x, target_y))
    if not raw_path:
        print(f"  [A* HALT] No collision-free route could be found from ({cur_x:.2f}, {cur_y:.2f}) to ({target_x:.2f}, {target_y:.2f})!")
        driver._halt_motors()
        save_state(cur_x, cur_y, cur_yaw, "blocked_no_path")
        return (cur_x, cur_y, cur_yaw)

    waypoints = planner.smooth_path(raw_path) or raw_path
    # Make the final waypoint the exact approach pose (A* ends on the nearest grid cell centre)
    if math.hypot(waypoints[-1][0] - target_x, waypoints[-1][1] - target_y) < 0.20:
        waypoints = list(waypoints[:-1]) + [(target_x, target_y)]

    def _publish_path(wps):
        driver.active_waypoints = [(p[0], p[1]) for p in wps]
        dist = sum(math.hypot(wps[i+1][0] - wps[i][0], wps[i+1][1] - wps[i][1]) for i in range(len(wps) - 1))
        driver.remaining_dist_m = dist
        return dist

    total_dist = _publish_path(waypoints)
    print(f"  [A* PATH FOUND] {total_dist:.2f} m across {len(waypoints)} waypoints:")
    for w_i, wp in enumerate(waypoints):
        print(f"    • WP {w_i:02d}: ({wp[0]:+.2f} m, {wp[1]:+.2f} m)")
    print("-" * 64)

    if dry_run:
        print("[DRY-RUN MODE] Simulated arrival without physical actuation.")
        driver.live_pose = (target_x, target_y, target_yaw)
        save_state(target_x, target_y, target_yaw, tid)
        return (target_x, target_y, target_yaw)

    # 2. Hardware Waypoint Execution
    robot_x, robot_y, heading = cur_x, cur_y, cur_yaw
    if driver.has_hardware_gyro:
        driver.set_yaw_rad(cur_yaw)

    driver.live_pose = (robot_x, robot_y, heading)

    # Correct accumulated drift before driving; replan if the pose moved noticeably
    loc = localize_pose(driver, planner, (robot_x, robot_y, heading), "start")
    if math.hypot(loc[0] - robot_x, loc[1] - robot_y) > 0.08:
        robot_x, robot_y, heading = loc
        new_plan = planner.plan((robot_x, robot_y), (target_x, target_y))
        if len(new_plan) >= 2:
            waypoints = planner.smooth_path(new_plan) or new_plan
            if math.hypot(waypoints[-1][0] - target_x, waypoints[-1][1] - target_y) < 0.20:
                waypoints = list(waypoints[:-1]) + [(target_x, target_y)]
            _publish_path(waypoints)
            print(f"  [REPLANNED FROM CORRECTED POSE] {len(waypoints)} waypoints.")
    else:
        robot_x, robot_y, heading = loc
    reached_target = False
    obstacle_halted = False
    user_stopped = False
    replan_count = 0
    stale_count = 0
    misalign_count = 0
    max_replans = cfg.max_replans
    wp_idx = 0

    def _dist_goal():
        return math.hypot(robot_x - target_x, robot_y - target_y)

    while wp_idx < len(waypoints) - 1:
        if driver.stop_flag:
            print("\n[STOPPED] Mission aborted by user.")
            user_stopped = True
            break

        # Expire temporary dynamic obstacles according to TTL
        planner.cleanup_expired_dynamic_obstacles()

        p_next = waypoints[wp_idx + 1]
        seg_dist = math.hypot(p_next[0] - robot_x, p_next[1] - robot_y)
        if seg_dist < cfg.waypoint_tolerance_m:
            wp_idx += 1
            continue

        # A. Turn to face waypoint
        bearing = math.atan2(p_next[1] - robot_y, p_next[0] - robot_x)
        for _ in range(3):
            turn = normalize_angle(bearing - heading)
            if abs(turn) <= math.radians(cfg.pivot_tolerance_deg * 2):
                break
            robot_x, robot_y, heading = _align_heading(driver, (robot_x, robot_y, heading), turn)
            if driver.stop_flag:
                break
        if driver.stop_flag:
            user_stopped = True
            break

        # Recalculate geometric bearing to waypoint after pivot settling
        bearing = math.atan2(p_next[1] - robot_y, p_next[0] - robot_x)
        seg_dist = math.hypot(p_next[0] - robot_x, p_next[1] - robot_y)

        # B. Drive towards waypoint (steers onto the waypoint, corrects drift)
        m_res = driver.drive_straight(
            distance_m=seg_dist,
            bearing=bearing,
            start_p=(robot_x, robot_y),
            leg_idx=wp_idx + 1,
            total_legs=len(waypoints) - 1,
        )

        # Authoritative state update: always update pose from movement result
        robot_x, robot_y, heading = m_res.final_pose
        driver.live_pose = (robot_x, robot_y, heading)
        if m_res.halt_reason in ("completed", "obstacle", "timeout") and m_res.elapsed_s > 0.5:
            robot_x, robot_y, heading = localize_pose(driver, planner, (robot_x, robot_y, heading))
        driver.remaining_dist_m = max(0.0, _dist_goal())

        if m_res.halt_reason == "stopped":
            user_stopped = True
            break
        if m_res.halt_reason == "misaligned":
            misalign_count += 1
            if misalign_count <= 10:
                continue

        # Check target arrival
        if _dist_goal() <= cfg.goal_tolerance_m:
            reached_target = True
            break

        # Lost depth frames: pause and retry the same waypoint a few times
        if m_res.halt_reason == "sensor_stale":
            stale_count += 1
            if stale_count > 3:
                print(f"\n  [SAFETY HALT] Depth camera keeps dropping out. Halting navigation.")
                obstacle_halted = True
                break
            print(f"  [SENSOR WAIT] Waiting for depth stream to recover ({stale_count}/3)...")
            time.sleep(0.6)
            continue

        # C. Dynamic Obstacle Avoidance & A* Re-routing
        if m_res.halt_reason == "obstacle":
            # Targets are parked next to furniture: an obstacle stop close to the goal is an arrival
            if _dist_goal() <= cfg.goal_obstacle_tolerance_m:
                print(f"\n  [GOAL NEAR OBSTACLE] Stopped {_dist_goal():.2f} m from goal in front of an object. Treating as arrived.")
                reached_target = True
                break

            obs_dist = m_res.depth_m if m_res.depth_m is not None else cfg.obstacle_halt_dist_m

            # The obstacle IS the target furniture: goal is (almost) straight ahead and not much
            # farther than the obstacle itself -> we are as close as we can get.
            goal_brg = math.atan2(target_y - robot_y, target_x - robot_x)
            if (_dist_goal() <= obs_dist + 0.60 and _dist_goal() <= 1.0
                    and abs(normalize_angle(goal_brg - heading)) < math.radians(35.0)):
                print(f"\n  [GOAL BEHIND OBJECT] Target furniture reached ({_dist_goal():.2f} m from approach point). Stopping here.")
                reached_target = True
                break

            # People / moving things: wait up to 3 s for the way to clear before detouring
            print(f"  [WAITING] Obstacle ahead. Waiting up to 3 s for it to move...")
            t_wait = time.time() + 3.0
            cleared = False
            while time.time() < t_wait and not driver.stop_flag:
                _, d_now = driver.read_sensors()
                if d_now is None or d_now > cfg.obstacle_slow_dist_m:
                    cleared = True
                    break
                time.sleep(0.1)
            if cleared:
                print(f"  [PATH CLEARED] Obstacle moved away. Continuing on the same route.")
                continue

            marked = _mark_sensed_obstacles(driver, planner, robot_x, robot_y, heading, obs_dist)
            pts_str = ", ".join(f"({p[0]:+.2f}, {p[1]:+.2f})" for p in marked)
            print(f"\n  [OBSTACLE MAPPED] {len(marked)} obstacle point(s) at {pts_str} (heading {math.degrees(heading):+.1f}°)")

            # Back up to create turning clearance away from obstacle
            back_res = driver.backup(distance_m=cfg.backup_dist_m, heading_rad=heading)
            robot_x, robot_y, heading = back_res.final_pose
            driver.live_pose = (robot_x, robot_y, heading)
            robot_x, robot_y, heading = localize_pose(driver, planner, (robot_x, robot_y, heading), "after backup")
            replan_count += 1

            if replan_count > max_replans:
                print(f"\n  [OBSTACLE HALT] Exceeded maximum avoidance attempts ({max_replans}). Path blocked!")
                obstacle_halted = True
                break

            print(f"\n  [A* REPLANNING] Computing detour around obstacle (Attempt {replan_count}/{max_replans})...")
            new_plan = planner.plan((robot_x, robot_y), (target_x, target_y))

            if len(new_plan) < 2:
                print(f"  [PATH TEMPORARILY BLOCKED] Waiting 2.0s for obstruction to clear...")
                time.sleep(2.0)
                _, d_now = driver.read_sensors()
                if d_now is None or d_now > cfg.obstacle_slow_dist_m:
                    # Whatever blocked us has moved: forget the sensed obstacles
                    print(f"  [PATH CLEARED] Obstacle no longer visible. Clearing temporary obstacles.")
                    planner.reset_dynamic_obstacles()
                new_plan = planner.plan((robot_x, robot_y), (target_x, target_y))

            if len(new_plan) < 2:
                print(f"  [ERROR] No navigable detour found to target '{tid}'! Halting safely.")
                obstacle_halted = True
                break

            waypoints = planner.smooth_path(new_plan) or new_plan
            if math.hypot(waypoints[-1][0] - target_x, waypoints[-1][1] - target_y) < 0.20:
                waypoints = list(waypoints[:-1]) + [(target_x, target_y)]
            total_dist = _publish_path(waypoints)
            # A detour much longer than the straight distance usually means the sensed obstacle
            # is sealing off the goal area. Near the goal, stop instead of wandering off.
            if total_dist > 2.5 * _dist_goal() + 1.0:
                if _dist_goal() <= 1.0:
                    print(f"\n  [GOAL AREA BLOCKED] Detour {total_dist:.2f} m for a goal {_dist_goal():.2f} m away. Stopping here.")
                    reached_target = _dist_goal() <= cfg.goal_obstacle_tolerance_m + 0.20
                    obstacle_halted = not reached_target
                    break
                print(f"  [LONG DETOUR] {total_dist:.2f} m. Forgetting old obstacle marks and replanning once...")
                planner.reset_dynamic_obstacles()
                _mark_sensed_obstacles(driver, planner, robot_x, robot_y, heading, obs_dist)
                new_plan = planner.plan((robot_x, robot_y), (target_x, target_y))
                if len(new_plan) >= 2:
                    waypoints = planner.smooth_path(new_plan) or new_plan
                    if math.hypot(waypoints[-1][0] - target_x, waypoints[-1][1] - target_y) < 0.20:
                        waypoints = list(waypoints[:-1]) + [(target_x, target_y)]
                    total_dist = _publish_path(waypoints)
            print(f"  [DETOUR PATH ACTIVE] New path generated: {total_dist:.2f} m across {len(waypoints)} waypoints.")
            wp_idx = 0
            continue

        wp_idx += 1

    # 2b. Final approach correction: if the last leg came up short, drive straight onto the goal
    if not (reached_target or obstacle_halted or user_stopped):
        for _ in range(2):
            d_goal = _dist_goal()
            if d_goal <= cfg.goal_tolerance_m:
                break
            if d_goal > 0.6 or not planner.line_of_sight((robot_x, robot_y), (target_x, target_y), check_inflation=False):
                break
            bearing = math.atan2(target_y - robot_y, target_x - robot_x)
            robot_x, robot_y, heading = _align_heading(driver, (robot_x, robot_y, heading), normalize_angle(bearing - heading))
            bearing = math.atan2(target_y - robot_y, target_x - robot_x)
            m_res = driver.drive_straight(_dist_goal(), bearing=bearing, start_p=(robot_x, robot_y))
            robot_x, robot_y, heading = m_res.final_pose
            if m_res.halt_reason not in ("completed", "misaligned"):
                break
        if _dist_goal() <= cfg.goal_obstacle_tolerance_m:
            reached_target = True

    # 3. Final Target Alignment
    if reached_target and not driver.stop_flag:
        robot_x, robot_y, heading = _align_heading(driver, (robot_x, robot_y, heading), normalize_angle(target_yaw - heading))
        driver.live_pose = (robot_x, robot_y, heading)

    # 4. Save Pose
    save_status = tid if reached_target else ("stopped_obstacle" if obstacle_halted else "stopped")
    save_state(robot_x, robot_y, heading, save_status)

    print("-" * 64)
    if reached_target:
        print(f"[ARRIVED] Target '{tid}' reached at ({robot_x:+.2f} m, {robot_y:+.2f} m), {_dist_goal():.2f} m from approach point!")
    elif obstacle_halted:
        print(f"[STOPPED] Obstacle avoidance halted robot safely at ({robot_x:+.2f} m, {robot_y:+.2f} m).")
    else:
        print(f"[STOPPED] Robot stopped at ({robot_x:+.2f} m, {robot_y:+.2f} m). Target: '{tid}'.")
    print("=" * 64)

    return (robot_x, robot_y, heading)


def main():
    parser = argparse.ArgumentParser(description="Autonomous A* Navigation with D455 Gyro & Depth Obstacle Avoidance")
    parser.add_argument("target", nargs="?", default=None, help="Target landmark (e.g. 'desk 18', 'chair 1', 'origin')")
    parser.add_argument("--tour", action="store_true", help="Execute complete 10-station facility tour")
    parser.add_argument("--dwell", type=float, default=2.5, help="Observation dwell time at each tour station in seconds (default: 2.5s)")
    parser.add_argument("--cam-upright", action="store_true", help="Camera is mounted upright instead of inverted (default: inverted)")
    parser.add_argument("--flip-gyro", action="store_true", help="Invert hardware gyro yaw rate polarity")
    parser.add_argument("--invert-gyro", action="store_true", help="Invert hardware gyro yaw rate polarity")
    parser.add_argument("--invert-motors", action="store_true", help="Invert motor directions if hardware is wired in reverse polarity")
    parser.add_argument("--normal-motors", action="store_true", help="Legacy flag: ensure normal motor wiring")
    parser.add_argument("--dry-run", action="store_true", help="Simulate navigation without pulsing hardware motors")
    parser.add_argument("--monitor", action="store_true", help="Launch live real-time Camera vs Map telemetry monitor without driving")
    parser.add_argument("--speed", type=float, default=None, help="Override cruise speed PWM (e.g. 0.88)")
    parser.add_argument("--turn-speed", type=float, default=None, help="Override pivot turn speed PWM (e.g. 0.90)")
    parser.add_argument("--no-localize", action="store_true", help="Disable scan-to-map drift correction")
    parser.add_argument("--speed-mps", type=float, default=None, help="Measured real forward speed (m/s) at the cruise PWM. Use --calib-drive to find it.")
    parser.add_argument("--calib-drive", type=float, default=None, metavar="METRES", help="Calibration: drive straight this many (estimated) metres, then measure the real distance with a tape")
    parser.add_argument("--set-start", help="Manually set current robot pose to a known landmark")
    parser.add_argument("--from", dest="start_from", help="Plan navigation starting from a specific landmark")
    parser.add_argument("--from-origin", action="store_true", help="Start navigation from Origin (0, 0, 0)")
    parser.add_argument("--reset", action="store_true", help="Reset robot pose to Origin (0, 0, 0)")
    parser.add_argument("--status", action="store_true", help="Show current robot state")
    args = parser.parse_args()

    motors_inv = bool(args.invert_motors and not args.normal_motors)
    effective_flip_gyro = False if (args.flip_gyro or getattr(args, 'invert_gyro', False)) else True

    config = NavConfig(
        cam_inverted=not args.cam_upright,
        motors_inverted=motors_inv,
        flip_gyro=effective_flip_gyro,
    )

    global CRUISE_PWM, TURN_PWM, SPEED_MPS, TURN_RADPS
    if args.speed is not None:
        config.cruise_pwm = float(args.speed)
        config.speed_mps = speed_for_pwm(config.cruise_pwm)
        CRUISE_PWM = config.cruise_pwm
        SPEED_MPS = config.speed_mps
    if args.turn_speed is not None:
        config.turn_pwm = float(args.turn_speed)
        config.turn_radps = max(0.35, config.turn_pwm * 1.55)
        TURN_PWM = config.turn_pwm
        TURN_RADPS = config.turn_radps
    if args.speed_mps is not None:
        config.speed_mps = float(args.speed_mps)
        SPEED_MPS = config.speed_mps
    if args.no_localize:
        config.localize = False

    print("=" * 66)
    print(f"[CONFIGURATION RESOLVED] Cruise: {config.cruise_pwm*100:.0f}% ({config.speed_mps:.2f} m/s) | "
          f"Turn: {config.turn_pwm*100:.0f}% ({math.degrees(config.turn_radps):.1f}°/s) | "
          f"Cam: {'Inverted' if config.cam_inverted else 'Upright'} | Gyro Sign: {'Inverted' if config.flip_gyro else 'Normal'}")
    print("=" * 66)

    # Load semantic map entities
    sem_entities = {}
    sem_path = FUSED_DIR / "semantic_map.json"
    if sem_path.exists():
        try:
            with open(sem_path, "r", encoding="utf-8") as f:
                sem_entities = json.load(f).get("entities", {})
        except Exception:
            pass

    cur_x, cur_y, cur_yaw = load_state()

    if args.reset:
        cur_x, cur_y, cur_yaw = 0.0, 0.0, 0.0
        save_state(0.0, 0.0, 0.0, "origin")
        print("  [RESET] Robot state reset to Origin: (X = 0.00 m, Y = 0.00 m, Yaw = 0.0°)")
        if not args.target and not args.tour:
            return
    elif abs(cur_x) > 0.05 or abs(cur_y) > 0.05 or abs(cur_yaw) > 0.05:
        print(f"  [SAVED STATE LOADED] Resumed pose: (X = {cur_x:+.2f} m, Y = {cur_y:+.2f} m, Yaw = {math.degrees(cur_yaw):+.1f}°)")

    if args.from_origin:
        cur_x, cur_y, cur_yaw = 0.0, 0.0, 0.0
        print("  [FROM-ORIGIN] Starting from Origin: (X = 0.00 m, Y = 0.00 m, Yaw = 0.0°)")

    if args.set_start:
        tid, tinfo = resolve_target(args.set_start, sem_entities)
        if not tid:
            print(f"[ERROR] Landmark '{args.set_start}' not recognized.")
            return
        sx, sy, syaw = tinfo["approach"]
        cur_x, cur_y, cur_yaw = sx, sy, syaw
        save_state(sx, sy, syaw, tid)
        print(f"[CALIBRATED] Robot start pose set to '{tid}' ({tinfo['desc']})")
        if not args.target and not args.tour:
            return

    if args.start_from:
        tid, tinfo = resolve_target(args.start_from, sem_entities)
        if tid:
            cur_x, cur_y, cur_yaw = tinfo["approach"]
            print(f"[START OVERRIDE] Starting from '{tid}': (X={cur_x:.2f} m, Y={cur_y:.2f} m, Yaw={math.degrees(cur_yaw):.1f}°)")

    if args.calib_drive is not None:
        driver = MotorDriver(config=config)
        est = float(args.calib_drive)
        print("=" * 66)
        print(f"SPEED CALIBRATION: driving {est:.2f} m (estimated at {config.speed_mps:.3f} m/s).")
        print("  Clear at least that much floor in front of the robot. Starting in 3 s...")
        print("=" * 66)
        try:
            time.sleep(3.0)
            driver.live_pose = (0.0, 0.0, driver.current_yaw_rad)
            res = driver.drive_straight(est, bearing=driver.current_yaw_rad, start_p=(0.0, 0.0))
            print(f"  Result: {res.halt_reason} | {res.details}")
            print("  Now measure the REAL distance travelled (m) and run with:")
            print(f"    --speed-mps <{config.speed_mps:.3f} * real_distance / {est:.2f}>")
        except KeyboardInterrupt:
            print("\n[STOPPED] Calibration aborted.")
        finally:
            driver.cleanup()
        return

    if args.monitor:
        driver = MotorDriver(config=config)
        print("=" * 74)
        print("D455 HARDWARE GYRO & LIVE DEPTH TELEMETRY MONITOR")
        print(f"  Camera Mount: {'INVERTED (UPSIDE-DOWN)' if not args.cam_upright else 'UPRIGHT'}")
        print(f"  Gyro Sign Polarity: {driver.gyro_sign:+.1f}")
        print(f"  Hardware Gyro Status: {'ONLINE' if driver.has_hardware_gyro else 'OFFLINE (Depth Only - run with sudo -E)'}")
        print("  [TEST 1] Rotate car LEFT by hand -> Yaw MUST increase (+).")
        print("           If it decreases (-), re-run with --flip-gyro.")
        print("  [TEST 2] Place hand/box in front of camera (< 0.48m) -> Status must show [OBSTACLE].")
        print("  Press Ctrl+C to stop.")
        print("=" * 74)
        try:
            while True:
                yaw, depth = driver.read_sensors()
                depth_str = f"{depth:4.2f}m" if depth is not None else " N/A "
                status = "[OBSTACLE]" if (depth is not None and depth <= config.obstacle_halt_dist_m) else "[CLEAR]"
                sys.stdout.write(
                    f"\r[MONITOR] Yaw: {math.degrees(yaw):+6.1f}° | Rate: {math.degrees(driver.get_yaw_rate()):+5.1f}°/s | "
                    f"Forward Depth: {depth_str} | Status: {status}"
                )
                sys.stdout.flush()
                time.sleep(0.04)
        except KeyboardInterrupt:
            print("\n[STOPPED] Monitor closed.")
        finally:
            driver.cleanup()
        return

    if args.status or (not args.target and not args.tour):
        print("=" * 66)
        print("CURRENT ROBOT STATE:")
        print(f"  Pose: (X = {cur_x:.2f} m, Y = {cur_y:.2f} m, Yaw = {math.degrees(cur_yaw):.1f}°)")
        print(f"  State File: {STATE_FILE.resolve()}")
        print("\nAVAILABLE TARGETS:")
        for tid, info in TARGET_DATABASE.items():
            print(f"  • {tid:<12} (aliases: {', '.join(info['aliases'][:2])}) -> {info['desc']}")
        print("\nUsage example: python nav.py \"desk 18\"")
        print("Tour example:  python nav.py --tour")
        print("=" * 66)
        return

    driver = MotorDriver(config=config)
    planner = FastAStarPlanner(FUSED_DIR)

    try:
        if args.tour:
            print("=" * 70)
            print("STARTING 10-OBSERVATION SEQUENTIAL FACILITY TOUR")
            print(f"Stops: {len(TOUR_SEQUENCE)} stations | Dwell: {args.dwell:.1f}s/station | DryRun: {args.dry_run}")
            print("=" * 70)
            cur_pose = (cur_x, cur_y, cur_yaw)
            for idx, (tid, desc) in enumerate(TOUR_SEQUENCE, 1):
                tinfo = TARGET_DATABASE[tid]
                print(f"\n>>> [STEP {idx:02d}/10] Commencing navigation to '{tid}' ({desc})...")
                cur_pose = navigate_to_target(
                    tid, tinfo, cur_pose, sem_entities, driver,
                    planner=planner,
                    dry_run=args.dry_run,
                )
                if idx < len(TOUR_SEQUENCE):
                    next_tid, next_desc = TOUR_SEQUENCE[idx]
                    print(f"[OBSERVATION DWELL] Pausing {args.dwell:.1f}s at station {idx:02d} for observation...")
                    time.sleep(args.dwell)
                    print(f"[REROUTING TO NEXT STATION] Routing to station {idx+1:02d}/10: '{next_tid}' ({next_desc})...")

            print("\n" + "=" * 70)
            print(">>> 10-OBSERVATION FACILITY TOUR SUCCESSFULLY COMPLETED! <<<")
            print(f"Final Pose: (X = {cur_pose[0]:.2f} m, Y = {cur_pose[1]:.2f} m, Yaw = {math.degrees(cur_pose[2]):.1f}°)")
            print("=" * 70)
        else:
            tid, tinfo = resolve_target(args.target, sem_entities)
            if not tid:
                print(f"[ERROR] Target '{args.target}' not recognized.")
                print("Available targets: " + ", ".join(TARGET_DATABASE.keys()))
                sys.exit(1)

            navigate_to_target(
                tid, tinfo, (cur_x, cur_y, cur_yaw), sem_entities, driver,
                planner=planner,
                dry_run=args.dry_run,
            )
    except KeyboardInterrupt:
        print("\n[EMERGENCY STOP] Operator pressed Ctrl+C! Motors stopped.")
    finally:
        driver.cleanup()


if __name__ == "__main__":
    main()
