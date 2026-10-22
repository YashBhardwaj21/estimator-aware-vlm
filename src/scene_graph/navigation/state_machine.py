"""Deterministic Safety State Machine for Autonomous RC Car Navigation.

Enforces strict lifecycle state transitions and safety gates:
  DISABLED -> INITIALIZING -> LOCALIZING -> READY -> PLANNING -> EXECUTING -> APPROACHING -> STOPPED
  (Any critical anomaly transitions immediately to SAFE_STOP)
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


class NavigationState(Enum):
    DISABLED = "DISABLED"
    INITIALIZING = "INITIALIZING"
    LOCALIZING = "LOCALIZING"
    READY = "READY"
    PLANNING = "PLANNING"
    EXECUTING = "EXECUTING"
    APPROACHING = "APPROACHING"
    STOPPED = "STOPPED"
    SAFE_STOP = "SAFE_STOP"  # Latched fault state


@dataclass
class StateTransitionRecord:
    timestamp: float
    from_state: NavigationState
    to_state: NavigationState
    trigger: str
    details: str = ""


class NavigationStateMachine:
    """Manages system state, enforces safety checks, and latches on faults."""

    def __init__(self, max_localization_covariance: float = 0.05, max_sensor_age_s: float = 0.25):
        self.state = NavigationState.DISABLED
        self.max_cov = max_localization_covariance
        self.max_sensor_age_s = max_sensor_age_s
        self.fault_reason: Optional[str] = None
        self.is_motion_enabled = False
        self.history: List[StateTransitionRecord] = []

    def _transition(self, to_state: NavigationState, trigger: str, details: str = "") -> None:
        rec = StateTransitionRecord(
            timestamp=time.time(),
            from_state=self.state,
            to_state=to_state,
            trigger=trigger,
            details=details,
        )
        self.history.append(rec)
        logger.info("STATE TRANSITION: [%s] -> [%s] (Trigger: %s, Details: %s)",
                    self.state.value, to_state.value, trigger, details)
        self.state = to_state

    def trigger_safe_stop(self, reason: str) -> None:
        """Immediately latch into safe-stop state on any safety violation."""
        self.fault_reason = reason
        self.is_motion_enabled = False
        self._transition(NavigationState.SAFE_STOP, trigger="SAFETY_VIOLATION", details=reason)

    def initialize(self) -> bool:
        """Start system initialization."""
        if self.state in (NavigationState.DISABLED, NavigationState.STOPPED):
            self._transition(NavigationState.INITIALIZING, trigger="INIT_START")
            return True
        return False

    def on_sensors_ready(self) -> bool:
        """Sensors and camera stream ready; begin localization acquisition."""
        if self.state == NavigationState.INITIALIZING:
            self._transition(NavigationState.LOCALIZING, trigger="SENSORS_READY")
            return True
        return False

    def on_localization_update(
        self,
        pose_valid: bool,
        covariance: float,
        timestamp_s: float,
    ) -> bool:
        """Process localization health update."""
        now = time.time()
        # Check staleness
        if (now - timestamp_s) > self.max_sensor_age_s:
            if self.state in (NavigationState.EXECUTING, NavigationState.APPROACHING):
                self.trigger_safe_stop(f"Localization data is stale (age={now - timestamp_s:.3f}s > {self.max_sensor_age_s}s)")
            return False

        if not pose_valid or covariance > self.max_cov:
            if self.state in (NavigationState.EXECUTING, NavigationState.APPROACHING):
                self.trigger_safe_stop(f"Localization lost or covariance too high ({covariance:.3f} > {self.max_cov})")
            return False

        # If we were acquiring localization, transition to READY
        if self.state == NavigationState.LOCALIZING:
            self._transition(NavigationState.READY, trigger="LOCALIZATION_ESTABLISHED")
            return True

        return True

    def start_planning(self) -> bool:
        """Begin path planning towards target goal."""
        if self.state in (NavigationState.READY, NavigationState.STOPPED):
            self._transition(NavigationState.PLANNING, trigger="PLANNING_START")
            return True
        return False

    def on_plan_ready(self, path_found: bool) -> bool:
        """Planning completed."""
        if self.state == NavigationState.PLANNING:
            if path_found:
                self._transition(NavigationState.READY, trigger="PLAN_VALIDATED")
                return True
            else:
                self._transition(NavigationState.READY, trigger="PLAN_FAILED", details="No path found")
                return False
        return False

    def arm_and_execute(self, enable_motion: bool = False) -> bool:
        """Enable motion and start trajectory tracking.
        
        Requires explicit enable_motion=True flag. Without it, physical motion is blocked!
        """
        if self.state != NavigationState.READY:
            logger.warning("Cannot start execution: state is %s, expected READY", self.state.value)
            return False

        if not enable_motion:
            logger.warning("Physical motion NOT enabled (dry-run mode). Remaining in READY state.")
            return False

        self.is_motion_enabled = True
        self._transition(NavigationState.EXECUTING, trigger="MOTION_ARMED")
        return True

    def on_enter_approach(self, distance_to_goal_m: float) -> bool:
        """Transition to cautious final approach phase."""
        if self.state == NavigationState.EXECUTING:
            self._transition(NavigationState.APPROACHING, trigger="APPROACH_ZONE_ENTERED",
                             details=f"dist={distance_to_goal_m:.2f}m")
            return True
        return False

    def on_goal_reached(self) -> bool:
        """Target standoff clearance reached; halt motion."""
        if self.state in (NavigationState.APPROACHING, NavigationState.EXECUTING):
            self.is_motion_enabled = False
            self._transition(NavigationState.STOPPED, trigger="GOAL_ARRIVED")
            return True
        return False

    def recover(self) -> bool:
        """Clear latched safe stop after manual operator verification."""
        if self.state == NavigationState.SAFE_STOP:
            self.fault_reason = None
            self.is_motion_enabled = False
            self._transition(NavigationState.STOPPED, trigger="MANUAL_RECOVERY")
            return True
        return False

    def get_status(self) -> Dict[str, Any]:
        return {
            "current_state": self.state.value,
            "is_motion_enabled": self.is_motion_enabled,
            "fault_reason": self.fault_reason,
            "transition_count": len(self.history),
        }
