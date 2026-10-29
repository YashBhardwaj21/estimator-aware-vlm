"""Configurable and isolated ORB-SLAM3 execution runtime with fail-closed semantics."""

from __future__ import annotations

import os
import platform
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

from estimator_aware_vlm.common.paths import get_repo_root


class SLAMError(RuntimeError):
    """Base exception for all SLAM runtime errors."""
    pass


class SLAMExecutionError(SLAMError):
    """Raised when ORB-SLAM3 process exits with non-zero status or fails to produce trajectory."""
    pass


class StaleTrajectoryError(SLAMError):
    """Raised when a pre-existing trajectory cannot be purged or is detected as stale."""
    pass


class EmptyTrajectoryError(SLAMError):
    """Raised when ORB-SLAM3 outputs an empty trajectory or 0 valid poses."""
    pass


def to_wsl_path(path: Union[str, Path]) -> str:
    """Convert Windows path to WSL /mnt/<drive>/... format if running on Windows."""
    s = str(path).replace("\\", "/")
    # Check if starts with Windows drive letter like C: or c:
    if len(s) >= 2 and s[1] == ":":
        drive = s[0].lower()
        rest = s[2:]
        if not rest.startswith("/"):
            rest = "/" + rest
        return f"/mnt/{drive}{rest}"
    return s


@dataclass
class SLAMRuntimeConfig:
    """Configuration for ORB-SLAM3 executable, vocabulary, sensor settings, and execution environment."""

    orb_binary: str = "/home/saturn/ORB_SLAM3/Examples/RGB-D/rgbd_tum"
    orb_vocab: str = "/home/saturn/ORB_SLAM3/Vocabulary/ORBvoc.txt"
    settings_path: Optional[str] = None
    wsl_distro: str = "Ubuntu-22.04"
    wsl_user: str = "saturn"
    timeout_sec: int = 900
    fail_closed: bool = True

    @classmethod
    def from_env_and_args(
        cls,
        orb_binary: Optional[str] = None,
        orb_vocab: Optional[str] = None,
        settings_path: Optional[Union[str, Path]] = None,
        wsl_distro: Optional[str] = None,
        wsl_user: Optional[str] = None,
        timeout_sec: int = 900,
        fail_closed: bool = True,
        sequence_name: Optional[str] = None,
    ) -> SLAMRuntimeConfig:
        """Resolve configuration prioritizing explicit args, then env vars, then defaults."""
        root_env = os.environ.get("ORB_SLAM3_ROOT")

        # 1. Resolve Binary
        if orb_binary is None:
            if "ORB_SLAM3_BIN" in os.environ:
                orb_binary = os.environ["ORB_SLAM3_BIN"]
            elif root_env:
                orb_binary = f"{root_env.rstrip('/')}/Examples/RGB-D/rgbd_tum"
            else:
                orb_binary = "/home/saturn/ORB_SLAM3/Examples/RGB-D/rgbd_tum"

        # 2. Resolve Vocabulary
        if orb_vocab is None:
            if "ORB_SLAM3_VOCAB" in os.environ:
                orb_vocab = os.environ["ORB_SLAM3_VOCAB"]
            elif root_env:
                orb_vocab = f"{root_env.rstrip('/')}/Vocabulary/ORBvoc.txt"
            else:
                orb_vocab = "/home/saturn/ORB_SLAM3/Vocabulary/ORBvoc.txt"

        # 3. Resolve WSL Distro and User
        if wsl_distro is None:
            wsl_distro = os.environ.get("WSL_DISTRO_NAME", "Ubuntu-22.04")
        if wsl_user is None:
            wsl_user = os.environ.get("WSL_USER", "saturn")

        # 4. Resolve Settings
        resolved_settings: Optional[str] = None
        if settings_path is not None:
            resolved_settings = str(settings_path)
        elif "ORB_SLAM3_SETTINGS" in os.environ:
            resolved_settings = os.environ["ORB_SLAM3_SETTINGS"]
        elif sequence_name:
            repo_root = get_repo_root()
            if "openloris" in sequence_name.lower():
                resolved_settings = str(repo_root / "configs" / "sensors" / "OpenLORIS_D400.yaml")
            else:
                resolved_settings = "/home/saturn/ORB_SLAM3/Examples/RGB-D/TUM3.yaml"

        return cls(
            orb_binary=orb_binary,
            orb_vocab=orb_vocab,
            settings_path=resolved_settings,
            wsl_distro=wsl_distro,
            wsl_user=wsl_user,
            timeout_sec=timeout_sec,
            fail_closed=fail_closed,
        )


class ORBSLAM3Runner:
    """Manages isolated and fail-closed execution of ORB-SLAM3."""

    def __init__(
        self,
        config: Optional[SLAMRuntimeConfig] = None,
        executor_fn: Optional[Callable[..., subprocess.CompletedProcess]] = None,
    ) -> None:
        self.config = config or SLAMRuntimeConfig.from_env_and_args()
        self._executor_fn = executor_fn or subprocess.run

    def purge_stale_trajectories(self, output_dir: Path) -> List[Path]:
        """Delete any existing trajectory files in output_dir before running."""
        purged = []
        for name in ["CameraTrajectory.txt", "KeyFrameTrajectory.txt"]:
            target = output_dir / name
            if target.exists():
                try:
                    target.unlink()
                    purged.append(target)
                except Exception as exc:
                    raise StaleTrajectoryError(
                        f"Failed to purge stale trajectory file {target}: {exc}"
                    ) from exc
        return purged

    def count_valid_trajectory_poses(self, traj_path: Path) -> int:
        """Count valid pose rows in a TUM/ORB-SLAM3 trajectory file."""
        if not traj_path.exists():
            return 0
        count = 0
        with open(traj_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                parts = line.split()
                if len(parts) >= 8:
                    count += 1
        return count

    def run(
        self,
        sequence_dir: Path,
        associations_file: Path,
        output_dir: Path,
        settings_path: Optional[Union[str, Path]] = None,
        b_rgbd: int = 0,
    ) -> Path:
        """Execute ORB-SLAM3 in an isolated working directory with fail-closed checks.

        Args:
            sequence_dir: Path to raw or relative sequence folder.
            associations_file: Path to generated association file (must exist).
            output_dir: Isolated output directory for logs and trajectory files.
            settings_path: Optional override for camera calibration settings YAML.
            b_rgbd: Optional mode flag for rgbd_tum (default 0).

        Returns:
            Path to validated non-empty CameraTrajectory.txt (or KeyFrameTrajectory.txt).

        Raises:
            SLAMExecutionError: On non-zero return code or missing trajectory.
            EmptyTrajectoryError: If produced trajectory has 0 valid poses.
        """
        output_dir = Path(output_dir).resolve()
        output_dir.mkdir(parents=True, exist_ok=True)

        if not associations_file.exists():
            raise FileNotFoundError(f"Associations file does not exist: {associations_file}")

        # 1. Purge stale trajectories
        self.purge_stale_trajectories(output_dir)

        # 2. Resolve settings path
        actual_settings = settings_path or self.config.settings_path
        if actual_settings is None:
            raise ValueError("Settings path must be specified either in SLAMRuntimeConfig or run().")

        wsl_settings = to_wsl_path(actual_settings)
        wsl_out_dir = to_wsl_path(output_dir)
        wsl_seq_dir = to_wsl_path(sequence_dir)
        wsl_assoc_file = to_wsl_path(associations_file)

        # 3. Construct command
        inner_cmd = (
            f"cd '{wsl_out_dir}' && "
            f"'{self.config.orb_binary}' '{self.config.orb_vocab}' '{wsl_settings}' "
            f"'{wsl_seq_dir}' '{wsl_assoc_file}' {b_rgbd}"
        )

        is_windows = platform.system() == "Windows"
        if is_windows:
            cmd = ["wsl", "-d", self.config.wsl_distro, "-u", self.config.wsl_user, "--", "bash", "-c", inner_cmd]
        else:
            cmd = ["bash", "-c", inner_cmd]

        log_file = output_dir / "orbslam3_execution.log"
        t0 = time.time()

        try:
            res = self._executor_fn(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                timeout=self.config.timeout_sec,
            )
        except subprocess.TimeoutExpired as exc:
            duration = time.time() - t0
            with open(log_file, "w", encoding="utf-8") as f:
                f.write(f"Execution timed out after {duration:.2f}s.\n")
                if exc.stdout:
                    f.write(str(exc.stdout))
            raise SLAMExecutionError(
                f"ORB-SLAM3 timed out after {self.config.timeout_sec} seconds."
            ) from exc

        duration = time.time() - t0

        # Save stdout/stderr log
        stdout_text = res.stdout if res.stdout is not None else ""
        with open(log_file, "w", encoding="utf-8") as f:
            f.write(stdout_text)

        # 4. Fail-closed on non-zero exit code
        if res.returncode != 0 and self.config.fail_closed:
            log_tail = stdout_text[-1500:] if stdout_text else "No output"
            raise SLAMExecutionError(
                f"ORB-SLAM3 failed with exit code {res.returncode} after {duration:.2f}s.\n"
                f"Log written to: {log_file}\n"
                f"Tail of stdout:\n{log_tail}"
            )

        # 5. Check trajectory generation
        traj_file = output_dir / "CameraTrajectory.txt"
        if not traj_file.exists():
            alt_traj = output_dir / "KeyFrameTrajectory.txt"
            if alt_traj.exists():
                traj_file = alt_traj
            else:
                log_tail = stdout_text[-1500:] if stdout_text else "No output"
                raise SLAMExecutionError(
                    f"ORB-SLAM3 finished with code {res.returncode} but produced no trajectory file.\n"
                    f"Log: {log_file}\nTail:\n{log_tail}"
                )

        # 6. Validate non-empty trajectory
        pose_count = self.count_valid_trajectory_poses(traj_file)
        if pose_count == 0:
            raise EmptyTrajectoryError(
                f"Produced trajectory {traj_file.name} is empty or contains 0 valid poses. "
                f"Tracking was lost immediately or failed to initialize."
            )

        return traj_file

