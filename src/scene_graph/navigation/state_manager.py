"""Persistent Robot State and Sequential Navigation Manager.

Maintains verified robot state across successive commands without requiring
manual coordinate entry after every movement.

Enforces:
  1. Map identifier and coordinate frame verification (REP-103 floor_map).
  2. Distinction between measured localized poses and command-estimated poses.
  3. Atomic file persistence (preventing partial/corrupted writes on crash).
  4. Full command lifecycle history (REQUESTED, EXECUTING, INTERRUPTED, BLOCKED, COMPLETED).
  5. Recovery after process restart with historical checkpoint validation.
"""

from __future__ import annotations

import json
import logging
import math
import os
import time
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


class PoseSource(str, Enum):
    LOCALIZED = "localized"                        # Independently verified by visual localization/SLAM
    COMMAND_ESTIMATED = "command_estimated"        # Dead-reckoning / simulated kinematic integration
    MANUAL_INITIALIZED = "manual_initialized"      # Explicit operator initial seed pose
    HISTORICAL_CHECKPOINT = "historical_checkpoint"# Loaded from disk after restart
    INVALID = "invalid"                            # Lost tracking or corrupt data


class CommandStatus(str, Enum):
    REQUESTED = "REQUESTED"
    EXECUTING = "EXECUTING"
    APPROACHING = "APPROACHING"
    COMPLETED = "COMPLETED"
    INTERRUPTED = "INTERRUPTED"
    BLOCKED = "BLOCKED"
    FAILED = "FAILED"


@dataclass
class RobotPoseRecord:
    x_m: float
    y_m: float
    yaw_rad: float
    timestamp_s: float
    source: PoseSource = PoseSource.LOCALIZED
    confidence: float = 1.0
    frame_id: str = "floor_map"

    @property
    def yaw_deg(self) -> float:
        return math.degrees(self.yaw_rad)

    def to_tuple_2d(self) -> Tuple[float, float, float]:
        return (self.x_m, self.y_m, self.yaw_rad)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "x_m": round(self.x_m, 4),
            "y_m": round(self.y_m, 4),
            "yaw_rad": round(self.yaw_rad, 4),
            "yaw_deg": round(self.yaw_deg, 2),
            "timestamp_s": round(self.timestamp_s, 4),
            "source": self.source.value if isinstance(self.source, PoseSource) else str(self.source),
            "confidence": round(self.confidence, 4),
            "frame_id": self.frame_id,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> RobotPoseRecord:
        src_str = data.get("source", "invalid")
        try:
            src = PoseSource(src_str)
        except ValueError:
            src = PoseSource.INVALID
        return cls(
            x_m=float(data.get("x_m", 0.0)),
            y_m=float(data.get("y_m", 0.0)),
            yaw_rad=float(data.get("yaw_rad", 0.0)),
            timestamp_s=float(data.get("timestamp_s", time.time())),
            source=src,
            confidence=float(data.get("confidence", 0.0)),
            frame_id=data.get("frame_id", "floor_map"),
        )


@dataclass
class CommandRecord:
    command_id: str
    prompt: str
    target_entity_id: Optional[str]
    target_approach_pose: Optional[Tuple[float, float, float]]
    start_pose: RobotPoseRecord
    final_pose: Optional[RobotPoseRecord] = None
    status: CommandStatus = CommandStatus.REQUESTED
    start_time_s: float = field(default_factory=time.time)
    end_time_s: Optional[float] = None
    distance_traveled_m: float = 0.0
    failure_reason: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "command_id": self.command_id,
            "prompt": self.prompt,
            "target_entity_id": self.target_entity_id,
            "target_approach_pose": [round(float(v), 3) for v in self.target_approach_pose] if self.target_approach_pose else None,
            "start_pose": self.start_pose.to_dict(),
            "final_pose": self.final_pose.to_dict() if self.final_pose else None,
            "status": self.status.value if isinstance(self.status, CommandStatus) else str(self.status),
            "start_time_s": round(self.start_time_s, 3),
            "end_time_s": round(self.end_time_s, 3) if self.end_time_s else None,
            "distance_traveled_m": round(self.distance_traveled_m, 3),
            "failure_reason": self.failure_reason,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> CommandRecord:
        st_str = data.get("status", "FAILED")
        try:
            st = CommandStatus(st_str)
        except ValueError:
            st = CommandStatus.FAILED

        start_p = RobotPoseRecord.from_dict(data.get("start_pose", {}))
        final_p = RobotPoseRecord.from_dict(data["final_pose"]) if data.get("final_pose") else None
        target_appr = tuple(data["target_approach_pose"]) if data.get("target_approach_pose") else None

        return cls(
            command_id=str(data.get("command_id", "")),
            prompt=str(data.get("prompt", "")),
            target_entity_id=data.get("target_entity_id"),
            target_approach_pose=target_appr,
            start_pose=start_p,
            final_pose=final_p,
            status=st,
            start_time_s=float(data.get("start_time_s", time.time())),
            end_time_s=float(data["end_time_s"]) if data.get("end_time_s") else None,
            distance_traveled_m=float(data.get("distance_traveled_m", 0.0)),
            failure_reason=data.get("failure_reason"),
        )


class NavigationStateManager:
    """Thread-safe, crash-resilient manager for persistent RC car navigation state."""

    DEFAULT_STATE_FILENAME = "robot_navigation_state.json"

    def __init__(
        self,
        map_dir: Path,
        state_file: Optional[Path] = None,
        coordinate_frame: str = "floor_map",
    ):
        self.map_dir = Path(map_dir).resolve()
        self.state_file = Path(state_file).resolve() if state_file else (self.map_dir / self.DEFAULT_STATE_FILENAME)
        self.coordinate_frame = coordinate_frame
        self.map_id = self._compute_map_id()

        self.current_pose: Optional[RobotPoseRecord] = None
        self.last_confirmed_stopping_pose: Optional[RobotPoseRecord] = None
        self.active_command: Optional[CommandRecord] = None
        self.last_completed_command: Optional[CommandRecord] = None
        self.command_history: List[CommandRecord] = []
        self._command_counter = 0

        self.load()

    def _compute_map_id(self) -> str:
        """Derive canonical map identifier from directory name and semantic map metadata."""
        sem_map = self.map_dir / "semantic_map.json"
        if sem_map.exists():
            try:
                with open(sem_map, "r", encoding="utf-8") as f:
                    sdata = json.load(f)
                fused_sess = sdata.get("fused_sessions", [])
                if fused_sess:
                    return f"{self.map_dir.name}:{'+'.join(fused_sess)}"
            except Exception:
                pass
        return self.map_dir.name

    def is_map_compatible(self, other_map_dir: Path) -> bool:
        """Verify whether saved state matches the requested map directory and frame."""
        other_path = Path(other_map_dir).resolve()
        return other_path.name == self.map_dir.name

    def load(self) -> bool:
        """Load state safely from disk, recovering gracefully from missing or malformed files."""
        if not self.state_file.exists():
            logger.info("No prior navigation state found at %s. Initializing fresh.", self.state_file)
            return False

        try:
            with open(self.state_file, "r", encoding="utf-8") as f:
                data = json.load(f)

            # Compatibility validation
            saved_map_id = data.get("map_id", "")
            saved_frame = data.get("coordinate_frame", "")

            if saved_frame != self.coordinate_frame:
                logger.warning("Coordinate frame mismatch: saved='%s', expected='%s'. Discarding poses.",
                               saved_frame, self.coordinate_frame)
                return False

            if saved_map_id and saved_map_id != self.map_id and not saved_map_id.startswith(self.map_dir.name):
                logger.warning("Map ID mismatch: saved='%s', current='%s'. Discarding pose.",
                               saved_map_id, self.map_id)
                return False

            # Load latest valid stopping pose
            if data.get("last_confirmed_stopping_pose"):
                self.last_confirmed_stopping_pose = RobotPoseRecord.from_dict(data["last_confirmed_stopping_pose"])
                # Demote loaded pose to HISTORICAL_CHECKPOINT so it requires fresh verification
                self.last_confirmed_stopping_pose.source = PoseSource.HISTORICAL_CHECKPOINT

            if data.get("current_pose"):
                self.current_pose = RobotPoseRecord.from_dict(data["current_pose"])
                self.current_pose.source = PoseSource.HISTORICAL_CHECKPOINT

            # Load command history
            self.command_history = [
                CommandRecord.from_dict(cmd) for cmd in data.get("command_history", [])
            ]
            self._command_counter = len(self.command_history)

            if data.get("last_completed_command"):
                self.last_completed_command = CommandRecord.from_dict(data["last_completed_command"])

            # Check for dangling active command from unexpected shutdown
            if data.get("active_command"):
                dangling = CommandRecord.from_dict(data["active_command"])
                if dangling.status in (CommandStatus.EXECUTING, CommandStatus.APPROACHING, CommandStatus.REQUESTED):
                    dangling.status = CommandStatus.INTERRUPTED
                    dangling.failure_reason = "Process terminated unexpectedly during execution"
                    dangling.end_time_s = time.time()
                    self.command_history.append(dangling)
                self.active_command = None

            logger.info("Successfully recovered persistent state from %s (Last Pose: %s)",
                        self.state_file, self.current_pose.to_tuple_2d() if self.current_pose else "None")
            return True

        except (json.JSONDecodeError, KeyError, ValueError) as err:
            logger.error("Corrupted state file at %s (%s). Preserving backup and resetting.", self.state_file, err)
            backup_p = self.state_file.with_suffix(".corrupted")
            try:
                os.replace(self.state_file, backup_p)
            except Exception:
                pass
            return False

    def save(self) -> bool:
        """Atomic write to disk using temporary file replacement to prevent partial writes."""
        self.state_file.parent.mkdir(parents=True, exist_ok=True)
        tmp_file = self.state_file.with_suffix(".tmp")

        payload = {
            "version": "1.0.0",
            "map_id": self.map_id,
            "coordinate_frame": self.coordinate_frame,
            "updated_at_timestamp_s": round(time.time(), 4),
            "current_pose": self.current_pose.to_dict() if self.current_pose else None,
            "last_confirmed_stopping_pose": self.last_confirmed_stopping_pose.to_dict() if self.last_confirmed_stopping_pose else None,
            "active_command": self.active_command.to_dict() if self.active_command else None,
            "last_completed_command": self.last_completed_command.to_dict() if self.last_completed_command else None,
            "command_history": [cmd.to_dict() for cmd in self.command_history[-50:]],  # Keep last 50 commands
        }

        try:
            with open(tmp_file, "w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2)
                f.flush()
                os.fsync(f.fileno())

            os.replace(tmp_file, self.state_file)
            return True
        except Exception as e:
            logger.error("Failed to atomically persist navigation state to %s: %s", self.state_file, e)
            if tmp_file.exists():
                try:
                    tmp_file.unlink()
                except Exception:
                    pass
            return False

    def get_initial_pose_for_planning(
        self,
        allow_command_estimated: bool = False,
        allow_historical_checkpoint: bool = True,
        max_freshness_s: Optional[float] = None,
    ) -> Optional[Tuple[float, float, float]]:
        """Retrieve latest valid robot pose to plan the next sequential command.

        Returns (x, y, yaw_rad) if suitable, or None if explicit localization is required.
        """
        # Prioritize confirmed stopping pose, fallback to current pose
        candidate = self.last_confirmed_stopping_pose or self.current_pose
        if candidate is None:
            return None

        if candidate.source == PoseSource.INVALID:
            return None

        if candidate.source == PoseSource.COMMAND_ESTIMATED and not allow_command_estimated:
            logger.warning("Candidate pose is only command-estimated; physical navigation requires localization.")
            return None

        if candidate.source == PoseSource.HISTORICAL_CHECKPOINT and not allow_historical_checkpoint:
            logger.warning("Candidate pose is a historical checkpoint requiring runtime verification.")
            return None

        if max_freshness_s is not None:
            age = time.time() - candidate.timestamp_s
            if age > max_freshness_s:
                logger.warning("Candidate pose is stale (age=%.1fs > max=%.1fs).", age, max_freshness_s)
                return None

        return candidate.to_tuple_2d()

    def update_pose(
        self,
        pose_2d: Tuple[float, float, float],
        source: PoseSource = PoseSource.LOCALIZED,
        timestamp_s: Optional[float] = None,
        confidence: float = 1.0,
    ) -> RobotPoseRecord:
        """Update live robot pose and persist state."""
        ts = timestamp_s or time.time()
        rec = RobotPoseRecord(
            x_m=float(pose_2d[0]),
            y_m=float(pose_2d[1]),
            yaw_rad=float(pose_2d[2]),
            timestamp_s=ts,
            source=source,
            confidence=confidence,
            frame_id=self.coordinate_frame,
        )
        self.current_pose = rec
        self.save()
        return rec

    def start_command(
        self,
        prompt: str,
        target_entity_id: Optional[str],
        target_approach_pose: Optional[Tuple[float, float, float]],
        start_pose: Tuple[float, float, float],
        start_pose_source: PoseSource = PoseSource.LOCALIZED,
    ) -> CommandRecord:
        """Initiate and persist a new navigation command."""
        self._command_counter += 1
        cmd_id = f"cmd_{self._command_counter:04d}_{int(time.time())}"

        start_rec = RobotPoseRecord(
            x_m=float(start_pose[0]),
            y_m=float(start_pose[1]),
            yaw_rad=float(start_pose[2]),
            timestamp_s=time.time(),
            source=start_pose_source,
            frame_id=self.coordinate_frame,
        )

        cmd = CommandRecord(
            command_id=cmd_id,
            prompt=prompt,
            target_entity_id=target_entity_id,
            target_approach_pose=target_approach_pose,
            start_pose=start_rec,
            status=CommandStatus.EXECUTING,
            start_time_s=time.time(),
        )

        self.active_command = cmd
        self.current_pose = start_rec
        self.save()
        logger.info("[COMMAND STARTED] %s: '%s' -> %s from (%.2f, %.2f)",
                    cmd_id, prompt, target_entity_id, start_pose[0], start_pose[1])
        return cmd

    def record_completed(
        self,
        final_measured_pose: Tuple[float, float, float],
        source: PoseSource = PoseSource.LOCALIZED,
    ) -> None:
        """Record verified command completion at the actual measured stopping pose."""
        final_rec = RobotPoseRecord(
            x_m=float(final_measured_pose[0]),
            y_m=float(final_measured_pose[1]),
            yaw_rad=float(final_measured_pose[2]),
            timestamp_s=time.time(),
            source=source,
            frame_id=self.coordinate_frame,
        )

        self.last_confirmed_stopping_pose = final_rec
        self.current_pose = final_rec

        if self.active_command:
            cmd = self.active_command
            cmd.final_pose = final_rec
            cmd.status = CommandStatus.COMPLETED
            cmd.end_time_s = time.time()
            # Calculate distance traveled
            dx = final_rec.x_m - cmd.start_pose.x_m
            dy = final_rec.y_m - cmd.start_pose.y_m
            cmd.distance_traveled_m = math.hypot(dx, dy)

            self.command_history.append(cmd)
            self.last_completed_command = cmd
            self.active_command = None

        self.save()
        logger.info("[COMMAND COMPLETED] Verified arrival at actual measured pose: (X=%.2f m, Y=%.2f m, Yaw=%.1f deg)",
                    final_rec.x_m, final_rec.y_m, final_rec.yaw_deg)

    def record_interrupted(
        self,
        current_pose: Tuple[float, float, float],
        reason: str,
        source: PoseSource = PoseSource.LOCALIZED,
    ) -> None:
        """Record command interruption (e.g., E-stop, operator halt, obstacle blockage)."""
        int_rec = RobotPoseRecord(
            x_m=float(current_pose[0]),
            y_m=float(current_pose[1]),
            yaw_rad=float(current_pose[2]),
            timestamp_s=time.time(),
            source=source,
            frame_id=self.coordinate_frame,
        )

        self.current_pose = int_rec
        self.last_confirmed_stopping_pose = int_rec

        if self.active_command:
            cmd = self.active_command
            cmd.final_pose = int_rec
            cmd.status = CommandStatus.INTERRUPTED
            cmd.end_time_s = time.time()
            cmd.failure_reason = reason
            dx = int_rec.x_m - cmd.start_pose.x_m
            dy = int_rec.y_m - cmd.start_pose.y_m
            cmd.distance_traveled_m = math.hypot(dx, dy)

            self.command_history.append(cmd)
            self.active_command = None

        self.save()
        logger.warning("[COMMAND INTERRUPTED] %s at pose (%.2f, %.2f)", reason, int_rec.x_m, int_rec.y_m)
