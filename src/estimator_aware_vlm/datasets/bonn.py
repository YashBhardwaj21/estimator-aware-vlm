"""Bonn RGB-D Dynamic Dataset Adapter.

Implements ingestion for the Bonn RGB-D Dynamic benchmark format
(Palazzolo et al., IEEE Transactions on Robotics 2019 / IROS 2018),
such as rgbd_bonn_crowd and rgbd_bonn_moving_obstructing_box.

Features:
- Association between RGB and depth streams from ASUS Xtion Pro Live.
- Strict TrajectoryRole.GROUND_TRUTH isolation for ground truth poses from OptiTrack.
- Correct depth scale factor (0.0002 m/unit = 5000 units/m).
- Immutability guarantee: strictly prevents writing into raw dataset folders.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterator, List, Optional, Tuple, Union
import numpy as np

from estimator_aware_vlm.datasets.base import (
    CameraIntrinsics,
    DatasetAdapter,
    DatasetConfig,
    FrameRecord,
    Trajectory,
    TrajectoryPose,
    TrajectoryRole,
    associate_streams,
    parse_tum_trajectory,
)


# Calibrated camera intrinsics for ASUS Xtion Pro Live on the Bonn RGB-D Dynamic Benchmark
BONN_XTION_INTRINSICS = CameraIntrinsics(
    fx=542.822841,
    fy=542.576870,
    cx=315.593520,
    cy=237.756098,
    width=640,
    height=480,
    distortion=(0.039903, -0.099343, -0.000730, -0.000144, 0.0),
    distortion_model="plumb_bob",
)


def parse_bonn_stream_file(file_path: Path) -> List[Tuple[float, str]]:
    """Parse a two-column timestamped file (timestamp filename)."""
    entries: List[Tuple[float, str]] = []
    if not file_path.exists():
        raise FileNotFoundError(f"Bonn stream file not found: {file_path}")

    with open(file_path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) >= 2:
                try:
                    ts = float(parts[0])
                    rel_path = parts[1]
                    entries.append((ts, rel_path))
                except ValueError:
                    continue

    entries.sort(key=lambda x: x[0])
    return entries


def parse_bonn_ground_truth(file_path: Path, name: str = "bonn_gt") -> Trajectory:
    """Parse Bonn groundtruth.txt file into a Trajectory with TrajectoryRole.GROUND_TRUTH."""
    if not file_path.exists():
        raise FileNotFoundError(f"Ground truth file not found: {file_path}")

    return parse_tum_trajectory(file_path, role=TrajectoryRole.GROUND_TRUTH, name=name)


class BonnDatasetAdapter(DatasetAdapter):
    """Adapter for the Bonn RGB-D Dynamic dataset."""

    def __init__(self, config: DatasetConfig):
        self.config = config
        self.sequence_dir = Path(config.sequence_dir)
        self.sequence_name = config.sequence_name

        self.rgb_file = config.rgb_file or (self.sequence_dir / "rgb.txt")
        self.depth_file = config.depth_file or (self.sequence_dir / "depth.txt")
        self.groundtruth_file = config.groundtruth_file or (self.sequence_dir / "groundtruth.txt")

        self.depth_scale = config.depth_scale
        self.intrinsics = config.intrinsics or BONN_XTION_INTRINSICS

        # Load raw streams
        self.rgb_entries = parse_bonn_stream_file(self.rgb_file)
        self.depth_entries = parse_bonn_stream_file(self.depth_file)

        # Load ground truth trajectory if present
        self.ground_truth_trajectory: Optional[Trajectory] = None
        if self.groundtruth_file.exists():
            self.ground_truth_trajectory = parse_bonn_ground_truth(
                self.groundtruth_file, name=f"{self.sequence_name}_ground_truth"
            )

        # 1-to-1 temporal stream association
        raw_pairs = associate_streams(
            self.rgb_entries,
            self.depth_entries,
            max_difference_sec=config.association_tolerance_sec,
            one_to_one=True,
        )

        # Confirm both files physically exist on disk
        self.associated_pairs = [
            pair for pair in raw_pairs
            if (self.sequence_dir / pair[1]).exists() and (self.sequence_dir / pair[3]).exists()
        ]

        # Materialize FrameRecord sequence
        self._frames: List[FrameRecord] = []
        for idx, (rgb_ts, rgb_rel, depth_ts, depth_rel) in enumerate(self.associated_pairs):
            rgb_path = self.sequence_dir / rgb_rel
            depth_path = self.sequence_dir / depth_rel

            gt_pose = None
            if self.ground_truth_trajectory is not None:
                gt_pose = self.ground_truth_trajectory.get_pose_at(
                    rgb_ts, max_dt_sec=config.association_tolerance_sec
                )

            self._frames.append(
                FrameRecord(
                    frame_index=idx,
                    timestamp=rgb_ts,
                    rgb_path=rgb_path,
                    depth_path=depth_path,
                    depth_scale=self.depth_scale,
                    intrinsics=self.intrinsics,
                    estimated_pose=None,
                    ground_truth_pose=gt_pose,
                )
            )

    def get_sequence_name(self) -> str:
        return self.sequence_name

    def get_camera_intrinsics(self) -> CameraIntrinsics:
        return self.intrinsics

    def get_frames(self) -> Iterator[FrameRecord]:
        return iter(self._frames)

    def get_frame(self, index: int) -> FrameRecord:
        if index < 0 or index >= len(self._frames):
            raise IndexError(f"Frame index {index} out of range (0..{len(self._frames)-1})")
        return self._frames[index]

    def get_ground_truth_trajectory(self) -> Trajectory:
        if self.ground_truth_trajectory is None:
            raise FileNotFoundError(f"Ground truth trajectory not available for {self.sequence_name}")
        return self.ground_truth_trajectory

    def __len__(self) -> int:
        return len(self._frames)

    def export_associations(self, output_path: Union[str, Path]) -> Path:
        """Export associations file for SLAM: rgb_ts rgb_file depth_ts depth_file.

        Strictly prohibits writing into the raw sequence data directory.
        """
        out = Path(output_path).resolve()
        seq_res = self.sequence_dir.resolve()
        try:
            if out.is_relative_to(seq_res):
                raise PermissionError(
                    f"Immutability violation: Cannot write generated associations into "
                    f"raw sequence directory '{self.sequence_dir}'. Export to outputs/ instead."
                )
        except AttributeError:
            if str(out).startswith(str(seq_res)):
                raise PermissionError(
                    "Immutability violation: Cannot write generated associations into raw sequence directory."
                )

        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w", encoding="utf-8") as f:
            for rgb_ts, rgb_rel, depth_ts, depth_rel in self.associated_pairs:
                f.write(f"{rgb_ts:.6f} {rgb_rel} {depth_ts:.6f} {depth_rel}\n")
        return out

