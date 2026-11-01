"""Explicit, robust timestamp model and clock mapping for sensor stream ingestion.

Enforces domain separation (HARDWARE_CLOCK, SYSTEM_TIME, GLOBAL_TIME, SIMULATED_TIME)
to prevent hazardous silent mixing of incompatible clock sources.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from enum import Enum


class TimestampDomain(str, Enum):
    """Authoritative clock domain designations for all sensor data."""
    HARDWARE_CLOCK = "hardware_clock"  # Device oscillator / ASIC clock (e.g. RealSense hardware timer)
    SYSTEM_TIME = "system_time"        # Host operating system time (e.g. Windows/Linux time.time())
    GLOBAL_TIME = "global_time"        # RealSense SDK hardware-synced global time protocol
    SIMULATED_TIME = "simulated_time"  # ROS /clock or dataset replay time
    UNKNOWN = "unknown"                # Unspecified / unverified clock domain

    @classmethod
    def from_string(cls, value: str | None) -> TimestampDomain:
        if not value:
            return cls.UNKNOWN
        clean = str(value).strip().lower().replace(" ", "_")
        for domain in cls:
            if domain.value == clean:
                return domain
        return cls.UNKNOWN


class IncompatibleTimestampDomainError(ValueError):
    """Raised when arithmetic or comparison is attempted between incompatible timestamp domains."""


@dataclass(frozen=True)
class Timestamp:
    """Immutable timestamp container preserving value, domain, and origin source."""
    value: float                     # Time in seconds (floating point, non-negative, finite)
    domain: TimestampDomain          # Hardware, system, global, or simulated domain
    source: str = "unknown"          # Identifier of originating component/driver (e.g. 'd455_color_hw')

    def __post_init__(self) -> None:
        if not isinstance(self.domain, TimestampDomain):
            object.__setattr__(self, "domain", TimestampDomain.from_string(str(self.domain)))
        val = float(self.value)
        if math.isnan(val) or not math.isfinite(val) or val < 0.0:
            raise ValueError(f"Timestamp value must be non-negative and finite, got {self.value}")
        object.__setattr__(self, "value", val)

    def is_comparable_with(self, other: Timestamp) -> bool:
        """Determines whether two timestamps share an identical or compatible clock domain."""
        if not isinstance(other, Timestamp):
            return False
        if self.domain == TimestampDomain.UNKNOWN or other.domain == TimestampDomain.UNKNOWN:
            return False
        if self.domain != other.domain:
            return False
        # Hardware and system clocks from distinct sources have arbitrary relative offsets
        if self.domain in (TimestampDomain.HARDWARE_CLOCK, TimestampDomain.SYSTEM_TIME):
            return self.source == other.source
        # Simulated and Global time are comparable if both share compatible or declared source
        if self.source != "unknown" and other.source != "unknown" and self.source != other.source:
            return False
        return True

    def delta_to(self, other: Timestamp) -> float:
        """Computes (other.value - self.value) in seconds, enforcing clock domain compatibility."""
        if not self.is_comparable_with(other):
            raise IncompatibleTimestampDomainError(
                f"Cannot compare timestamp in domain '{self.domain.value}' (source: {self.source}) "
                f"with timestamp in domain '{other.domain.value}' (source: {other.source})."
            )
        return other.value - self.value

    def __sub__(self, other: Timestamp) -> float:
        return -self.delta_to(other)

    def __lt__(self, other: Timestamp) -> bool:
        if not self.is_comparable_with(other):
            raise IncompatibleTimestampDomainError(
                f"Cannot order timestamp in domain '{self.domain.value}' and '{other.domain.value}'."
            )
        return self.value < other.value

    def __le__(self, other: Timestamp) -> bool:
        return self < other or self == other

    def __gt__(self, other: Timestamp) -> bool:
        return not (self <= other)

    def __ge__(self, other: Timestamp) -> bool:
        return not (self < other)

    @classmethod
    def now_system(cls, source: str = "host_os") -> Timestamp:
        """Constructs a timestamp from the current host system time."""
        return cls(value=time.time(), domain=TimestampDomain.SYSTEM_TIME, source=source)

    @classmethod
    def now_simulated(cls, seconds: float, source: str = "ros_clock") -> Timestamp:
        """Constructs a timestamp from a simulated clock value."""
        return cls(value=seconds, domain=TimestampDomain.SIMULATED_TIME, source=source)


class ClockMapping:
    """Manages online offset estimation and health tracking between two clock domains.
    
    Uses a minimum-delay sliding window filter (NTP-style) to estimate true clock offset
    without being distorted by network transit latency spikes, OS scheduling jitter, or queue delays.
    Guarantees monotonically non-decreasing mapped timestamps.
    """

    def __init__(
        self,
        smoothing_alpha: float = 0.05,
        max_drift_jump_sec: float = 0.5,
        max_stale_duration_sec: float = 2.0,
        window_size: int = 30,
    ):
        self.smoothing_alpha = float(smoothing_alpha)
        self.max_drift_jump_sec = float(max_drift_jump_sec)
        self.max_stale_duration_sec = float(max_stale_duration_sec)
        self.window_size = max(1, int(window_size))

        self._offset: float | None = None
        self._delay_window: list[float] = []
        self._last_update_monotonic: float = 0.0
        self._sample_count: int = 0
        self._is_healthy: bool = False
        self._last_raw_offset: float = 0.0
        self._drift_rate: float = 0.0
        self._last_mapped_time: float = -float("inf")

    @property
    def is_healthy(self) -> bool:
        if not self._is_healthy or self._offset is None:
            return False
        return (time.monotonic() - self._last_update_monotonic) <= self.max_stale_duration_sec

    @property
    def offset(self) -> float | None:
        return self._offset

    @property
    def sample_count(self) -> int:
        return self._sample_count

    @property
    def last_drift(self) -> float:
        return self._drift_rate

    def update(self, source_time: float, target_time: float) -> bool:
        """Updates the offset filter using arrival timestamp at the network/driver boundary.
        
        Args:
            source_time: Sensor hardware or capture timestamp.
            target_time: Exact arrival timestamp in target clock domain (before queues).
        
        Returns:
            True if sample was accepted and mapping is healthy, False if a discontinuity occurred.
        """
        now_mono = time.monotonic()
        raw_offset = target_time - source_time

        if self._offset is None:
            self._offset = raw_offset
            self._last_raw_offset = raw_offset
            self._last_update_monotonic = now_mono
            self._sample_count = 1
            self._is_healthy = True
            self._delay_window = [raw_offset]
            return True

        # Check for sudden clock jump (NTP synchronization, camera reset, or sleep/wake drift)
        step_delta = abs(raw_offset - self._offset)
        if step_delta > self.max_drift_jump_sec:
            # Discontinuity detected: reset filter to new offset but flag health warning
            self._offset = raw_offset
            self._last_raw_offset = raw_offset
            self._last_update_monotonic = now_mono
            self._sample_count = 1
            self._is_healthy = False
            self._delay_window = [raw_offset]
            return False

        # Maintain sliding window of recent transit offsets
        self._delay_window.append(raw_offset)
        if len(self._delay_window) > self.window_size:
            self._delay_window.pop(0)

        # Minimum transit delay in the window approximates the true clock offset with minimal network jitter
        min_delay_offset = min(self._delay_window)

        # Smooth tracking using minimum-delay estimate
        dt = max(1e-4, now_mono - self._last_update_monotonic)
        self._drift_rate = (raw_offset - self._last_raw_offset) / dt
        self._offset = (1.0 - self.smoothing_alpha) * self._offset + self.smoothing_alpha * min_delay_offset
        self._last_raw_offset = raw_offset
        self._last_update_monotonic = now_mono
        self._sample_count += 1
        self._is_healthy = True
        return True

    def map_timestamp(self, source_time: float) -> float:
        """Applies estimated offset to project source_time into target clock domain.
        
        Enforces monotonic non-decreasing guarantee: mapped output never jumps backward.
        """
        if self._offset is None:
            return source_time
        mapped = source_time + self._offset
        if mapped < self._last_mapped_time:
            mapped = self._last_mapped_time
        else:
            self._last_mapped_time = mapped
        return mapped

    def reset(self) -> None:
        """Resets the clock mapping upon reconnect, session restart, or disconnect."""
        self._offset = None
        self._delay_window.clear()
        self._last_update_monotonic = 0.0
        self._sample_count = 0
        self._is_healthy = False
        self._last_raw_offset = 0.0
        self._drift_rate = 0.0
        self._last_mapped_time = -float("inf")
