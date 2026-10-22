"""Differential-Drive / Skid-Steer Motion Controller for 4-Motor RC Car.

Translates planned waypoints into bounded linear and angular velocities (v, omega),
and converts them into differential wheel speeds (v_left, v_right) and PWM duty cycles.
Includes safety bounds, acceleration limiting, and final approach deceleration.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


@dataclass
class ControllerConfig:
    """Kinematic and dynamical parameters for 4-motor differential-drive RC car."""
    track_width_m: float = 0.20           # 20 cm lateral wheel baseline
    wheel_radius_m: float = 0.033         # 66 mm wheel diameter
    max_linear_speed_mps: float = 0.35    # Max 35 cm/s
    max_angular_speed_radps: float = 1.5  # Max 1.5 rad/s
    min_linear_speed_mps: float = 0.10    # Minimum speed above motor deadband
    lookahead_distance_m: float = 0.30    # Pure pursuit lookahead
    goal_tolerance_pos_m: float = 0.08    # 8 cm arrival tolerance
    goal_tolerance_yaw_rad: float = 0.12  # ~7 degrees arrival tolerance
    approach_slowdown_distance_m: float = 0.6  # Begin deceleration 60cm from target
    max_linear_accel_mps2: float = 0.40   # Slew rate limit for motor longevity


@dataclass
class MotionCommand:
    """Low-level motor velocity command."""
    linear_velocity_mps: float = 0.0
    angular_velocity_radps: float = 0.0
    v_left_mps: float = 0.0
    v_right_mps: float = 0.0
    pwm_left_pct: float = 0.0
    pwm_right_pct: float = 0.0
    is_stopped: bool = True
    phase: str = "IDLE"  # IDLE, TRACKING, APPROACHING, STOPPED, EMERGENCY_STOP

    def to_dict(self) -> Dict[str, Any]:
        return {
            "v_mps": round(self.linear_velocity_mps, 3),
            "omega_radps": round(self.angular_velocity_radps, 3),
            "v_left_mps": round(self.v_left_mps, 3),
            "v_right_mps": round(self.v_right_mps, 3),
            "pwm_left_pct": round(self.pwm_left_pct, 1),
            "pwm_right_pct": round(self.pwm_right_pct, 1),
            "is_stopped": self.is_stopped,
            "phase": self.phase,
        }


class DifferentialDriveController:
    """Pure pursuit controller and skid-steer velocity generator."""

    def __init__(self, config: Optional[ControllerConfig] = None):
        self.cfg = config or ControllerConfig()
        self.current_waypoint_idx = 0
        self.last_linear_v = 0.0
        self.last_update_time = time.time()

    def reset(self):
        self.current_waypoint_idx = 0
        self.last_linear_v = 0.0
        self.last_update_time = time.time()

    def compute_command(
        self,
        current_pose: Tuple[float, float, float],  # (x, y, yaw_rad)
        waypoints: List[Tuple[float, float, float]],
    ) -> MotionCommand:
        """Compute bounded velocity and differential wheel speeds for current pose."""
        if not waypoints:
            return MotionCommand(is_stopped=True, phase="STOPPED")

        cx, cy, cyaw = current_pose
        goal_x, goal_y, goal_yaw = waypoints[-1]
        dist_to_final = math.hypot(goal_x - cx, goal_y - cy)

        # 1. Check if already reached final goal
        if dist_to_final <= self.cfg.goal_tolerance_pos_m:
            yaw_err = self._normalize_angle(goal_yaw - cyaw)
            if abs(yaw_err) <= self.cfg.goal_tolerance_yaw_rad:
                self.last_linear_v = 0.0
                return MotionCommand(is_stopped=True, phase="STOPPED")
            else:
                # In-place final alignment rotation
                omega = np.clip(1.2 * yaw_err, -self.cfg.max_angular_speed_radps, self.cfg.max_angular_speed_radps)
                v_l, v_r, pwm_l, pwm_r = self._diff_drive_kinematics(0.0, omega)
                return MotionCommand(
                    linear_velocity_mps=0.0,
                    angular_velocity_radps=omega,
                    v_left_mps=v_l,
                    v_right_mps=v_r,
                    pwm_left_pct=pwm_l,
                    pwm_right_pct=pwm_r,
                    is_stopped=False,
                    phase="APPROACHING",
                )

        # 2. Advance active waypoint along path
        while self.current_waypoint_idx < len(waypoints) - 1:
            wx, wy, _ = waypoints[self.current_waypoint_idx]
            if math.hypot(wx - cx, wy - cy) < self.cfg.lookahead_distance_m:
                self.current_waypoint_idx += 1
            else:
                break

        # Lookahead target
        target_x, target_y, _ = waypoints[self.current_waypoint_idx]
        target_dx = target_x - cx
        target_dy = target_y - cy
        target_angle = math.atan2(target_dy, target_dx)
        heading_error = self._normalize_angle(target_angle - cyaw)

        # 3. Determine target linear speed with approach deceleration
        if dist_to_final < self.cfg.approach_slowdown_distance_m:
            ratio = max(0.0, dist_to_final / self.cfg.approach_slowdown_distance_m)
            v_target = self.cfg.min_linear_speed_mps + ratio * (self.cfg.max_linear_speed_mps - self.cfg.min_linear_speed_mps)
            phase = "APPROACHING"
        else:
            v_target = self.cfg.max_linear_speed_mps
            phase = "TRACKING"

        # Skid-steer behavior: Pivot in-place if heading error is large (> 35 deg)
        if abs(heading_error) > 0.60:
            v_target = 0.0
        else:
            # Scale forward speed by cosine of heading error when aligned
            v_target *= max(0.2, math.cos(heading_error))

        # Acceleration slew rate limit
        now = time.time()
        dt = max(0.01, min(0.2, now - self.last_update_time))
        self.last_update_time = now
        max_dv = self.cfg.max_linear_accel_mps2 * dt
        v_cmd = float(np.clip(v_target, self.last_linear_v - max_dv, self.last_linear_v + max_dv))
        if v_target > 0.0 and v_cmd > 0.0:
            v_cmd = max(self.cfg.min_linear_speed_mps, v_cmd)
        elif v_target == 0.0:
            v_cmd = 0.0
        self.last_linear_v = v_cmd

        # Pure-pursuit curvature steering: omega = v * (2 * sin(alpha) / L_d)
        L_d = max(0.10, self.cfg.lookahead_distance_m)
        if abs(v_cmd) > 0.01:
            curvature = (2.0 * math.sin(heading_error)) / L_d
            omega_cmd = v_cmd * curvature
        else:
            # In-place rotation when stationary
            omega_cmd = 1.8 * heading_error

        omega_cmd = float(np.clip(omega_cmd, -self.cfg.max_angular_speed_radps, self.cfg.max_angular_speed_radps))

        # 4. Map into differential wheel speeds
        v_l, v_r, pwm_l, pwm_r = self._diff_drive_kinematics(v_cmd, omega_cmd)

        return MotionCommand(
            linear_velocity_mps=v_cmd,
            angular_velocity_radps=omega_cmd,
            v_left_mps=v_l,
            v_right_mps=v_r,
            pwm_left_pct=pwm_l,
            pwm_right_pct=pwm_r,
            is_stopped=False,
            phase=phase,
        )

    def _diff_drive_kinematics(self, v: float, omega: float) -> Tuple[float, float, float, float]:
        """Convert (v, omega) to left/right wheel speeds and PWM duty percentages (0-100%)."""
        b = self.cfg.track_width_m
        v_l = v - 0.5 * omega * b
        v_r = v + 0.5 * omega * b

        # Map to PWM duty cycle (approximate linear mapping above deadband)
        # 0 m/s -> 0%, max_speed -> 100%
        max_s = self.cfg.max_linear_speed_mps
        pwm_l = np.clip((abs(v_l) / max_s) * 100.0, 0.0, 100.0) if abs(v_l) > 0.01 else 0.0
        pwm_r = np.clip((abs(v_r) / max_s) * 100.0, 0.0, 100.0) if abs(v_r) > 0.01 else 0.0

        return v_l, v_r, pwm_l, pwm_r

    @staticmethod
    def _normalize_angle(angle: float) -> float:
        """Wrap angle to [-pi, pi]."""
        while angle > math.pi:
            angle -= 2.0 * math.pi
        while angle < -math.pi:
            angle += 2.0 * math.pi
        return angle
