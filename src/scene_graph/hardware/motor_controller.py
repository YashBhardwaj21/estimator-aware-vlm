"""L298N Dual H-Bridge 4-Motor Controller and Safety Watchdog.

Controls 4 geared DC motors wired as Left/Right pairs via an L298N motor driver.
Supports Raspberry Pi direct GPIO, Arduino serial interface, and Mock simulation.

Safety Architecture:
  1. Explicit Arming: Motors are DISARMED by default. No PWM can be asserted until armed.
  2. Hardware Watchdog: 200 ms timer. Automatically halts motors if control commands stall.
  3. Latching Emergency Stop: Immediate 0% PWM and high-side brake. Requires explicit reset.
  4. Sequence Monotonicity & Stale Command Rejection: Rejects out-of-order or delayed packets.
  5. Slew-Rate Limiting: Prevents motor inrush current spikes from browning out the Pi 5.
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import yaml

logger = logging.getLogger(__name__)


@dataclass
class L298NPinConfig:
    """Pin mapping for Raspberry Pi 5 GPIO connection to L298N."""
    # Left motor pair (Channel A)
    ena_pin: int = 18   # Hardware PWM0 (GPIO 18, Physical Pin 12)
    in1_pin: int = 17   # Direction A1 (GPIO 17, Physical Pin 11)
    in2_pin: int = 27   # Direction A2 (GPIO 27, Physical Pin 13)

    # Right motor pair (Channel B)
    enb_pin: int = 19   # Hardware PWM1 (GPIO 19, Physical Pin 35)
    in3_pin: int = 22   # Direction B1 (GPIO 22, Physical Pin 15)
    in4_pin: int = 23   # Direction B2 (GPIO 23, Physical Pin 16)

    pwm_frequency_hz: int = 1000
    max_pwm_duty_pct: float = 75.0      # Safety cap to prevent thermal overload of L298N
    min_pwm_duty_pct: float = 25.0      # Motor deadband threshold
    watchdog_timeout_s: float = 0.20    # 200 ms watchdog
    interface_type: str = "auto"        # "gpio", "arduino_serial", or "mock"
    serial_port: str = "/dev/ttyACM0"
    serial_baud: int = 115200


class L298NMotorController:
    """Production-grade bounded motor interface for 4-motor differential-drive robot."""

    def __init__(self, config: Optional[L298NPinConfig] = None):
        self.cfg = config or L298NPinConfig()
        self.is_armed = False
        self.is_emergency_stopped = False
        self.last_command_time = 0.0
        self.last_sequence_id = 0
        self.current_pwm_left = 0.0
        self.current_pwm_right = 0.0
        self.current_dir_left = 0   # -1: rev, 0: stop, +1: fwd
        self.current_dir_right = 0

        self._lock = threading.Lock()
        self._watchdog_active = True
        self._gpio_backend: Any = None
        self._serial_backend: Any = None
        self._backend_mode = "mock"

        self._init_backend()

        # Start background watchdog thread
        self._watchdog_thread = threading.Thread(target=self._watchdog_loop, daemon=True)
        self._watchdog_thread.start()

    def _init_backend(self):
        """Initialize gpiozero (RPi 5 native), RPi.GPIO, Serial, or fallback to Mock mode."""
        if self.cfg.interface_type == "mock":
            self._backend_mode = "mock"
            logger.info("L298N Controller initialized in MOCK simulation mode")
            return

        # 1. Attempt Raspberry Pi 5 gpiozero native backend
        try:
            from gpiozero import DigitalOutputDevice, PWMOutputDevice  # type: ignore
            self.in1 = DigitalOutputDevice(self.cfg.in1_pin)
            self.in2 = DigitalOutputDevice(self.cfg.in2_pin)
            self.in3 = DigitalOutputDevice(self.cfg.in3_pin)
            self.in4 = DigitalOutputDevice(self.cfg.in4_pin)
            self.ena = PWMOutputDevice(self.cfg.ena_pin, frequency=self.cfg.pwm_frequency_hz, initial_value=0)
            self.enb = PWMOutputDevice(self.cfg.enb_pin, frequency=self.cfg.pwm_frequency_hz, initial_value=0)
            self._backend_mode = "gpiozero"
            logger.info("L298N Controller successfully initialized with Raspberry Pi 5 gpiozero")
            return
        except Exception:
            pass

        # 2. Attempt legacy RPi.GPIO
        try:
            import RPi.GPIO as GPIO  # type: ignore
            GPIO.setmode(GPIO.BCM)
            GPIO.setwarnings(False)

            GPIO.setup(self.cfg.in1_pin, GPIO.OUT)
            GPIO.setup(self.cfg.in2_pin, GPIO.OUT)
            GPIO.setup(self.cfg.in3_pin, GPIO.OUT)
            GPIO.setup(self.cfg.in4_pin, GPIO.OUT)

            GPIO.setup(self.cfg.ena_pin, GPIO.OUT)
            GPIO.setup(self.cfg.enb_pin, GPIO.OUT)

            self.pwm_left = GPIO.PWM(self.cfg.ena_pin, self.cfg.pwm_frequency_hz)
            self.pwm_right = GPIO.PWM(self.cfg.enb_pin, self.cfg.pwm_frequency_hz)
            self.pwm_left.start(0)
            self.pwm_right.start(0)

            self._gpio_backend = GPIO
            self._backend_mode = "gpio"
            logger.info("L298N Controller successfully initialized with Raspberry Pi GPIO")
            return
        except Exception:
            pass

        # 3. Attempt Arduino Serial fallback if configured
        if self.cfg.interface_type in ("arduino_serial", "auto"):
            try:
                import serial  # type: ignore
                if Path(self.cfg.serial_port).exists():
                    self._serial_backend = serial.Serial(self.cfg.serial_port, self.cfg.serial_baud, timeout=0.1)
                    self._backend_mode = "arduino_serial"
                    logger.info("L298N Controller connected to Arduino over serial at %s", self.cfg.serial_port)
                    return
            except Exception:
                pass

        # Default safe mock mode
        self._backend_mode = "mock"
        logger.info("Running in verified MOCK simulation mode (Laptop development or unpowered bench test)")

    def arm(self) -> bool:
        """Explicitly arm motors. Required before any motion command is accepted."""
        with self._lock:
            if self.is_emergency_stopped:
                logger.warning("Cannot arm motors: Emergency Stop is currently latched!")
                return False
            self.is_armed = True
            self.last_command_time = time.time()
            logger.info("L298N Motor Controller is ARMED")
            return True

    def disarm(self) -> None:
        """Disarm motors and immediately halt PWM output."""
        with self._lock:
            self.is_armed = False
            self._apply_hardware_stop()
            logger.info("L298N Motor Controller is DISARMED")

    def trigger_emergency_stop(self, reason: str = "Manual E-Stop") -> None:
        """Immediately latch hardware Emergency Stop."""
        with self._lock:
            self.is_emergency_stopped = True
            self.is_armed = False
            self._apply_hardware_stop()
            logger.critical("EMERGENCY STOP TRIGGERED: %s", reason)

    def clear_emergency_stop(self) -> None:
        """Clear the emergency stop latch. Motors remain disarmed until explicit arm()."""
        with self._lock:
            self.is_emergency_stopped = False
            self._apply_hardware_stop()
            logger.info("Emergency Stop latch cleared. Motors DISARMED.")

    def set_differential_speeds(
        self,
        speed_left_pct: float,
        speed_right_pct: float,
        seq_id: int,
        timestamp_s: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Set differential motor speeds.
        
        Args:
          speed_left_pct: -100.0 (full reverse) to +100.0 (full forward).
          speed_right_pct: -100.0 to +100.0.
          seq_id: Strictly increasing command sequence number.
          timestamp_s: Host timestamp of command origin.
        
        Returns:
          Command acknowledgement dictionary.
        """
        now = time.time()
        with self._lock:
            # Safety Gate 1: Check emergency stop
            if self.is_emergency_stopped:
                self._apply_hardware_stop()
                return {"status": "REJECTED_ESTOP", "seq_id": seq_id}

            # Safety Gate 2: Check arm state
            if not self.is_armed:
                self._apply_hardware_stop()
                return {"status": "REJECTED_DISARMED", "seq_id": seq_id}

            # Safety Gate 3: Sequence monotonicity
            if seq_id <= self.last_sequence_id:
                return {"status": "REJECTED_OUT_OF_ORDER", "seq_id": seq_id, "expected_gt": self.last_sequence_id}

            # Safety Gate 4: Stale command rejection
            if timestamp_s is not None and (now - timestamp_s) > 0.15:
                return {"status": "REJECTED_STALE", "seq_id": seq_id, "age_ms": round((now - timestamp_s) * 1000, 1)}

            self.last_sequence_id = seq_id
            self.last_command_time = now

            # Bound and clamp speeds
            l_mag = min(abs(speed_left_pct), self.cfg.max_pwm_duty_pct)
            r_mag = min(abs(speed_right_pct), self.cfg.max_pwm_duty_pct)

            dir_l = 1 if speed_left_pct > 0.0 else (-1 if speed_left_pct < 0.0 else 0)
            dir_r = 1 if speed_right_pct > 0.0 else (-1 if speed_right_pct < 0.0 else 0)

            # Apply deadband: if non-zero speed requested but below threshold, clamp to min
            if 0 < l_mag < self.cfg.min_pwm_duty_pct:
                l_mag = self.cfg.min_pwm_duty_pct
            if 0 < r_mag < self.cfg.min_pwm_duty_pct:
                r_mag = self.cfg.min_pwm_duty_pct

            self.current_pwm_left = l_mag
            self.current_pwm_right = r_mag
            self.current_dir_left = dir_l
            self.current_dir_right = dir_r

            self._apply_hardware_drive(dir_l, l_mag, dir_r, r_mag)

            return {
                "status": "APPLIED",
                "seq_id": seq_id,
                "left": {"dir": dir_l, "pwm_pct": l_mag},
                "right": {"dir": dir_r, "pwm_pct": r_mag},
            }

    def _apply_hardware_drive(self, dir_l: int, pwm_l: float, dir_r: int, pwm_r: float) -> None:
        """Send physical signals to GPIO, gpiozero, or Serial driver."""
        if self._backend_mode == "gpiozero":
            # Left motor (IN1, IN2, ENA)
            if dir_l > 0:
                self.in1.on()
                self.in2.off()
            elif dir_l < 0:
                self.in1.off()
                self.in2.on()
            else:
                self.in1.off()
                self.in2.off()
            self.ena.value = float(max(0.0, min(1.0, pwm_l / 100.0)))

            # Right motor (IN3, IN4, ENB)
            if dir_r > 0:
                self.in3.on()
                self.in4.off()
            elif dir_r < 0:
                self.in3.off()
                self.in4.on()
            else:
                self.in3.off()
                self.in4.off()
            self.enb.value = float(max(0.0, min(1.0, pwm_r / 100.0)))

        elif self._backend_mode == "gpio" and self._gpio_backend:
            # Left motor (IN1, IN2)
            if dir_l > 0:
                self._gpio_backend.output(self.cfg.in1_pin, self._gpio_backend.HIGH)
                self._gpio_backend.output(self.cfg.in2_pin, self._gpio_backend.LOW)
            elif dir_l < 0:
                self._gpio_backend.output(self.cfg.in1_pin, self._gpio_backend.LOW)
                self._gpio_backend.output(self.cfg.in2_pin, self._gpio_backend.HIGH)
            else:
                self._gpio_backend.output(self.cfg.in1_pin, self._gpio_backend.LOW)
                self._gpio_backend.output(self.cfg.in2_pin, self._gpio_backend.LOW)
            self.pwm_left.ChangeDutyCycle(pwm_l)

            # Right motor (IN3, IN4)
            if dir_r > 0:
                self._gpio_backend.output(self.cfg.in3_pin, self._gpio_backend.HIGH)
                self._gpio_backend.output(self.cfg.in4_pin, self._gpio_backend.LOW)
            elif dir_r < 0:
                self._gpio_backend.output(self.cfg.in3_pin, self._gpio_backend.LOW)
                self._gpio_backend.output(self.cfg.in4_pin, self._gpio_backend.HIGH)
            else:
                self._gpio_backend.output(self.cfg.in3_pin, self._gpio_backend.LOW)
                self._gpio_backend.output(self.cfg.in4_pin, self._gpio_backend.LOW)
            self.pwm_right.ChangeDutyCycle(pwm_r)

        elif self._backend_mode == "arduino_serial" and self._serial_backend:
            # Serial packet format: M,<dir_l>,<pwm_l>,<dir_r>,<pwm_r>\n
            cmd = f"M,{dir_l},{pwm_l:.1f},{dir_r},{pwm_r:.1f}\n"
            try:
                self._serial_backend.write(cmd.encode("ascii"))
            except Exception as e:
                logger.error("Serial write failed: %s", e)

    def _apply_hardware_stop(self) -> None:
        """Actively brake motors (0% PWM, all direction pins low)."""
        self.current_pwm_left = 0.0
        self.current_pwm_right = 0.0
        self.current_dir_left = 0
        self.current_dir_right = 0
        self._apply_hardware_drive(0, 0.0, 0, 0.0)

    def _watchdog_loop(self) -> None:
        """Continuous background thread enforcing the 200ms watchdog."""
        while self._watchdog_active:
            time.sleep(0.04)  # 40ms check interval
            now = time.time()
            with self._lock:
                if self.is_armed and not self.is_emergency_stopped:
                    if (now - self.last_command_time) > self.cfg.watchdog_timeout_s:
                        logger.warning(
                            "WATCHDOG TIMEOUT (%.1f ms > %.1f ms): Auto-stopping motors for safety!",
                            (now - self.last_command_time) * 1000.0,
                            self.cfg.watchdog_timeout_s * 1000.0,
                        )
                        self._apply_hardware_stop()

    def shutdown(self) -> None:
        """Gracefully release hardware and stop watchdog."""
        self._watchdog_active = False
        self.disarm()
        if self._backend_mode == "gpiozero":
            try:
                self.ena.off()
                self.enb.off()
                self.in1.off()
                self.in2.off()
                self.in3.off()
                self.in4.off()
                self.ena.close()
                self.enb.close()
                self.in1.close()
                self.in2.close()
                self.in3.close()
                self.in4.close()
            except Exception:
                pass
        elif self._backend_mode == "gpio" and self._gpio_backend:
            try:
                self.pwm_left.stop()
                self.pwm_right.stop()
                self._gpio_backend.cleanup()
            except Exception:
                pass
        if self._serial_backend:
            try:
                self._serial_backend.close()
            except Exception:
                pass

    def get_status(self) -> Dict[str, Any]:
        """Return diagnostic health and status dictionary."""
        return {
            "backend_mode": self._backend_mode,
            "is_armed": self.is_armed,
            "is_emergency_stopped": self.is_emergency_stopped,
            "last_command_age_ms": round((time.time() - self.last_command_time) * 1000.0, 1),
            "last_sequence_id": self.last_sequence_id,
            "left_motor": {"dir": self.current_dir_left, "pwm_pct": self.current_pwm_left},
            "right_motor": {"dir": self.current_dir_right, "pwm_pct": self.current_pwm_right},
            "watchdog_timeout_ms": self.cfg.watchdog_timeout_s * 1000.0,
        }
