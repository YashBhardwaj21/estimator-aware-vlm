"""Canonical dataset and experiment registry for Estimator-Aware VLM.

Provides a unified single source of truth for all eight benchmark sequences,
eliminating duplicate sequence definitions, resolving sensor calibrations and
ORB-SLAM3 settings, and mapping baseline trajectories via run manifests.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple, Union

from estimator_aware_vlm.common.paths import get_repo_root
from estimator_aware_vlm.datasets.base import (
    CameraIntrinsics,
    DatasetAdapter,
    DatasetConfig,
    TrajectoryRole,
)


# ---------------------------------------------------------------------------
# Enums and Value Objects
# ---------------------------------------------------------------------------

class GroundTruthType(str, Enum):
    """Origin of ground-truth reference measurements."""
    VICON_MOCAP = "VICON_MOCAP"
    OPTITRACK_MOCAP = "OPTITRACK_MOCAP"
    LIDAR_ICP = "LIDAR_ICP"
    NONE = "NONE"


class BaselineStatus(str, Enum):
    """Availability status of the baseline SLAM trajectory."""
    AVAILABLE = "AVAILABLE"
    NOT_RUN = "NOT_RUN"


@dataclass(frozen=True)
class PerceptionConfig:
    """Sequence-specific perception and tracking parameters."""
    stride: int = 2
    max_depth_m: float = 5.5
    class_mapping: Optional[Dict[str, str]] = None


# Extended class mapping for corridor / service-robot environments (OpenLORIS)
CORRIDOR_CLASS_MAPPING: Dict[str, str] = {
    "person": "person",
    "chair": "chair",
    "tv": "monitor",
    "laptop": "laptop",
    "keyboard": "keyboard",
    "mouse": "mouse",
    "dining table": "desk",
    "desk": "desk",
    "cup": "cup",
    "bottle": "bottle",
    "backpack": "backpack",
    "handbag": "handbag",
    "suitcase": "suitcase",
    "cell phone": "cell_phone",
    "book": "book",
    "potted plant": "plant",
}


# ---------------------------------------------------------------------------
# Canonical Sequence Registry Entry
# ---------------------------------------------------------------------------

@dataclass
class SequenceRegistryEntry:
    """Canonical registry record for a single RGB-D benchmark sequence."""
    sequence_id: str                      # Canonical short identifier (e.g. "tum_xyz")
    sequence_name: str                    # Raw directory name (e.g. "rgbd_dataset_freiburg3_walking_xyz")
    dataset_type: str                     # Adapter type: "tum_rgbd", "openloris", "bonn", "realsense_d455"
    raw_dir: str                          # Repository-relative raw data folder
    dataset_yaml: str                     # Repository-relative dataset config YAML
    rgb_file: str                         # Repository-relative path to rgb index
    depth_file: str                       # Repository-relative path to depth index
    groundtruth_file: Optional[str]       # Repository-relative path to ground truth (None if absent)
    depth_scale: float                    # Metres per raw depth unit
    association_tolerance_sec: float      # Timestamp pairing threshold
    intrinsics: CameraIntrinsics          # Calibrated camera intrinsic parameters
    has_ground_truth: bool                # Whether reference trajectory exists
    ground_truth_type: GroundTruthType    # Physical source of ground truth
    orbslam3_settings_path: str           # Repository-relative ORB-SLAM3 sensor settings file
    perception_config: PerceptionConfig   # Stride, max depth, semantic class mapping
    baseline_status: BaselineStatus = BaselineStatus.NOT_RUN
    baseline_run_id: Optional[str] = None
    baseline_manifest_path: Optional[str] = None
    baseline_trajectory_path: Optional[str] = None
    baseline_trajectory_sha256: Optional[str] = None
    ground_truth_sha256: Optional[str] = None
    perception_run_id: Optional[str] = "run_20261101_234859"
    aliases: List[str] = field(default_factory=list)

    def resolve_raw_dir(self, repo_root: Optional[Path] = None) -> Path:
        root = repo_root or get_repo_root()
        return root / self.raw_dir

    def resolve_yaml_path(self, repo_root: Optional[Path] = None) -> Path:
        root = repo_root or get_repo_root()
        return root / self.dataset_yaml

    def resolve_settings_path(self, repo_root: Optional[Path] = None) -> Path:
        root = repo_root or get_repo_root()
        return root / self.orbslam3_settings_path

    def resolve_baseline_trajectory(self, repo_root: Optional[Path] = None) -> Optional[Path]:
        if not self.baseline_trajectory_path:
            return None
        root = repo_root or get_repo_root()
        p = root / self.baseline_trajectory_path
        return p if p.exists() else None

    def resolve_groundtruth_path(self, repo_root: Optional[Path] = None) -> Optional[Path]:
        if not self.groundtruth_file:
            return None
        root = repo_root or get_repo_root()
        p = root / self.groundtruth_file
        return p if p.exists() else None

    def to_dataset_config(self, repo_root: Optional[Path] = None) -> DatasetConfig:
        root = repo_root or get_repo_root()
        gt = (root / self.groundtruth_file) if self.groundtruth_file else None
        return DatasetConfig(
            dataset_type=self.dataset_type,
            sequence_name=self.sequence_name,
            sequence_dir=root / self.raw_dir,
            rgb_file=root / self.rgb_file,
            depth_file=root / self.depth_file,
            groundtruth_file=gt,
            depth_scale=self.depth_scale,
            association_tolerance_sec=self.association_tolerance_sec,
            intrinsics=self.intrinsics,
            sequence_id=self.sequence_id,
            settings_path=root / self.orbslam3_settings_path,
            has_ground_truth=self.has_ground_truth,
            ground_truth_type=self.ground_truth_type.value,
        )

    def get_adapter(self, repo_root: Optional[Path] = None) -> DatasetAdapter:
        from estimator_aware_vlm.datasets.factory import get_dataset_adapter
        cfg = self.to_dataset_config(repo_root=repo_root)
        return get_dataset_adapter(cfg)

    def to_sequence_spec(self, repo_root: Optional[Path] = None) -> Any:
        """Construct a SequenceSpec for unified_pipeline.py consumption."""
        from estimator_aware_vlm.perception.unified_pipeline import SequenceSpec

        traj_path = self.resolve_baseline_trajectory(repo_root=repo_root)
        return SequenceSpec(
            sequence_id=self.sequence_id,
            dataset_yaml=self.dataset_yaml,
            estimated_trajectory_path=str(traj_path.relative_to(repo_root or get_repo_root())).replace("\\", "/") if traj_path else None,
            stride=self.perception_config.stride,
            max_depth_m=self.perception_config.max_depth_m,
            class_mapping=self.perception_config.class_mapping,
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "sequence_id": self.sequence_id,
            "sequence_name": self.sequence_name,
            "dataset_type": self.dataset_type,
            "raw_dir": self.raw_dir,
            "dataset_yaml": self.dataset_yaml,
            "rgb_file": self.rgb_file,
            "depth_file": self.depth_file,
            "groundtruth_file": self.groundtruth_file,
            "depth_scale": self.depth_scale,
            "association_tolerance_sec": self.association_tolerance_sec,
            "intrinsics": {
                "fx": self.intrinsics.fx,
                "fy": self.intrinsics.fy,
                "cx": self.intrinsics.cx,
                "cy": self.intrinsics.cy,
                "width": self.intrinsics.width,
                "height": self.intrinsics.height,
                "distortion": list(self.intrinsics.distortion),
                "distortion_model": self.intrinsics.distortion_model,
            },
            "has_ground_truth": self.has_ground_truth,
            "ground_truth_type": self.ground_truth_type.value,
            "ground_truth_sha256": self.ground_truth_sha256,
            "orbslam3_settings_path": self.orbslam3_settings_path,
            "perception": {
                "stride": self.perception_config.stride,
                "max_depth_m": self.perception_config.max_depth_m,
                "has_custom_class_mapping": self.perception_config.class_mapping is not None,
            },
            "baseline": {
                "status": self.baseline_status.value,
                "run_id": self.baseline_run_id,
                "manifest_path": self.baseline_manifest_path,
                "trajectory_path": self.baseline_trajectory_path,
                "trajectory_sha256": self.baseline_trajectory_sha256,
            },
            "perception_run_id": self.perception_run_id,
            "aliases": self.aliases,
        }


# ---------------------------------------------------------------------------
# Calibrated Intrinsic Constants
# ---------------------------------------------------------------------------

_TUM_INTRINSICS = CameraIntrinsics(
    fx=535.4, fy=539.2, cx=320.1, cy=247.6, width=640, height=480,
    distortion=(0.0, 0.0, 0.0, 0.0, 0.0), distortion_model="plumb_bob",
)

_BONN_INTRINSICS = CameraIntrinsics(
    fx=542.822841, fy=542.576870, cx=315.593520, cy=237.756098, width=640, height=480,
    distortion=(0.039903, -0.099343, -0.000730, -0.000144, 0.0), distortion_model="plumb_bob",
)

_OPENLORIS_INTRINSICS = CameraIntrinsics(
    fx=611.45098876953125, fy=611.48571777343750, cx=433.20397949218750, cy=249.47302246093750,
    width=848, height=480, distortion=(0.0, 0.0, 0.0, 0.0, 0.0), distortion_model="plumb_bob",
)

_D455_INTRINSICS = CameraIntrinsics(
    fx=383.9205627441406, fy=383.4780578613281, cx=320.2670593261719, cy=244.56988525390625,
    width=640, height=480, distortion=(-0.05704856, 0.07136440, 2.77367e-05, 0.00079690, -0.02323733),
    distortion_model="distortion.inverse_brown_conrady",
)


# ---------------------------------------------------------------------------
# The Eight Canonical Dataset Entries
# ---------------------------------------------------------------------------

CANONICAL_ENTRIES: List[SequenceRegistryEntry] = [
    SequenceRegistryEntry(
        sequence_id="tum_xyz",
        sequence_name="rgbd_dataset_freiburg3_walking_xyz",
        dataset_type="tum_rgbd",
        raw_dir="data/raw/rgbd_dataset_freiburg3_walking_xyz",
        dataset_yaml="configs/datasets/tum_rgbd_fr3_walking_xyz.yaml",
        rgb_file="data/raw/rgbd_dataset_freiburg3_walking_xyz/rgb.txt",
        depth_file="data/raw/rgbd_dataset_freiburg3_walking_xyz/depth.txt",
        groundtruth_file="data/raw/rgbd_dataset_freiburg3_walking_xyz/groundtruth.txt",
        depth_scale=0.0002,
        association_tolerance_sec=0.02,
        intrinsics=_TUM_INTRINSICS,
        has_ground_truth=True,
        ground_truth_type=GroundTruthType.VICON_MOCAP,
        ground_truth_sha256="4117699f2b027a6e305f84d3b17f1928aa5361db300d5df10639824540d92b6d",
        orbslam3_settings_path="configs/sensors/TUM_Freiburg3.yaml",
        perception_config=PerceptionConfig(stride=2, max_depth_m=5.5),
        baseline_status=BaselineStatus.AVAILABLE,
        baseline_run_id="baseline_campaign_control",
        baseline_manifest_path="results/baselines/rgbd_dataset_freiburg3_walking_xyz/run_manifest.json",
        baseline_trajectory_path="results/baselines/rgbd_dataset_freiburg3_walking_xyz/CameraTrajectory.txt",
        baseline_trajectory_sha256="73df6701a924bff84be1191efb695421434bb511dc34e2824984b10027ba3dc2",
        perception_run_id="run_20261101_234859",
        aliases=["tum_fr3_walking_xyz", "fr3_walking_xyz", "freiburg3_walking_xyz"],
    ),
    SequenceRegistryEntry(
        sequence_id="tum_halfsphere",
        sequence_name="rgbd_dataset_freiburg3_walking_halfsphere",
        dataset_type="tum_rgbd",
        raw_dir="data/raw/rgbd_dataset_freiburg3_walking_halfsphere",
        dataset_yaml="configs/datasets/tum_rgbd_fr3_walking_halfsphere.yaml",
        rgb_file="data/raw/rgbd_dataset_freiburg3_walking_halfsphere/rgb.txt",
        depth_file="data/raw/rgbd_dataset_freiburg3_walking_halfsphere/depth.txt",
        groundtruth_file="data/raw/rgbd_dataset_freiburg3_walking_halfsphere/groundtruth.txt",
        depth_scale=0.0002,
        association_tolerance_sec=0.02,
        intrinsics=_TUM_INTRINSICS,
        has_ground_truth=True,
        ground_truth_type=GroundTruthType.VICON_MOCAP,
        ground_truth_sha256="209eea77bccf94bc1b0e98508d6f14a93e7fe71b2821ce1d8ee865e08e0fe100",
        orbslam3_settings_path="configs/sensors/TUM_Freiburg3.yaml",
        perception_config=PerceptionConfig(stride=2, max_depth_m=5.5),
        baseline_status=BaselineStatus.AVAILABLE,
        baseline_run_id="baseline_campaign_control",
        baseline_manifest_path="results/baselines/rgbd_dataset_freiburg3_walking_halfsphere/run_manifest.json",
        baseline_trajectory_path="results/baselines/rgbd_dataset_freiburg3_walking_halfsphere/CameraTrajectory.txt",
        baseline_trajectory_sha256="635cea05f6d933b65a0d9d64a2634bdbe207907f688be07c641245ccdc76c1a2",
        perception_run_id="run_20261101_234859",
        aliases=["tum_fr3_walking_halfsphere", "fr3_walking_halfsphere", "freiburg3_walking_halfsphere"],
    ),
    SequenceRegistryEntry(
        sequence_id="tum_static",
        sequence_name="rgbd_dataset_freiburg3_walking_static",
        dataset_type="tum_rgbd",
        raw_dir="data/raw/rgbd_dataset_freiburg3_walking_static",
        dataset_yaml="configs/datasets/tum_rgbd_fr3_walking_static.yaml",
        rgb_file="data/raw/rgbd_dataset_freiburg3_walking_static/rgb.txt",
        depth_file="data/raw/rgbd_dataset_freiburg3_walking_static/depth.txt",
        groundtruth_file="data/raw/rgbd_dataset_freiburg3_walking_static/groundtruth.txt",
        depth_scale=0.0002,
        association_tolerance_sec=0.02,
        intrinsics=_TUM_INTRINSICS,
        has_ground_truth=True,
        ground_truth_type=GroundTruthType.VICON_MOCAP,
        ground_truth_sha256="e18b9028ab473d1ed150a621c32b09a68c630290088f49abfac4f329b81fdc46",
        orbslam3_settings_path="configs/sensors/TUM_Freiburg3.yaml",
        perception_config=PerceptionConfig(stride=2, max_depth_m=5.5),
        baseline_status=BaselineStatus.AVAILABLE,
        baseline_run_id="baseline_campaign_control",
        baseline_manifest_path="results/baselines/rgbd_dataset_freiburg3_walking_static/run_manifest.json",
        baseline_trajectory_path="results/baselines/rgbd_dataset_freiburg3_walking_static/CameraTrajectory.txt",
        baseline_trajectory_sha256="da1861b40603b83989d580a98a7110a7e91138e8961846baf01c0e2075c2f3d1",
        perception_run_id="run_20261101_234859",
        aliases=["tum_fr3_walking_static", "fr3_walking_static", "freiburg3_walking_static"],
    ),
    SequenceRegistryEntry(
        sequence_id="openloris",
        sequence_name="OpenLORIS-Scene_corridor1-4",
        dataset_type="openloris",
        raw_dir="data/raw/OpenLORIS-Scene_corridor1-4",
        dataset_yaml="configs/datasets/openloris_corridor1_4.yaml",
        rgb_file="data/raw/OpenLORIS-Scene_corridor1-4/color.txt",
        depth_file="data/raw/OpenLORIS-Scene_corridor1-4/aligned_depth.txt",
        groundtruth_file="data/raw/OpenLORIS-Scene_corridor1-4/groundtruth.txt",
        depth_scale=0.001,
        association_tolerance_sec=0.02,
        intrinsics=_OPENLORIS_INTRINSICS,
        has_ground_truth=True,
        ground_truth_type=GroundTruthType.LIDAR_ICP,
        ground_truth_sha256="0c6125b1ae6b8753d5c0e05404c417af22d7e47b999d4b6178c9bb104291c683",
        orbslam3_settings_path="configs/sensors/OpenLORIS_D400.yaml",
        perception_config=PerceptionConfig(stride=3, max_depth_m=8.0, class_mapping=CORRIDOR_CLASS_MAPPING),
        baseline_status=BaselineStatus.AVAILABLE,
        baseline_run_id="baseline_campaign_control",
        baseline_manifest_path="results/baselines/OpenLORIS-Scene_corridor1-4/run_manifest.json",
        baseline_trajectory_path="results/baselines/OpenLORIS-Scene_corridor1-4/CameraTrajectory.txt",
        baseline_trajectory_sha256="4776ea3eba64bbbd50f07754e12817a462042ac12f9243bfb7186afb627fefd9",
        perception_run_id="run_20261101_234859",
        aliases=["openloris_corridor1-4", "openloris_corridor1_4", "corridor1-4"],
    ),
    SequenceRegistryEntry(
        sequence_id="bonn_crowd",
        sequence_name="rgbd_bonn_crowd",
        dataset_type="bonn",
        raw_dir="data/raw/rgbd_bonn_crowd",
        dataset_yaml="configs/datasets/bonn_crowd.yaml",
        rgb_file="data/raw/rgbd_bonn_crowd/rgb.txt",
        depth_file="data/raw/rgbd_bonn_crowd/depth.txt",
        groundtruth_file="data/raw/rgbd_bonn_crowd/groundtruth.txt",
        depth_scale=0.0002,
        association_tolerance_sec=0.02,
        intrinsics=_BONN_INTRINSICS,
        has_ground_truth=True,
        ground_truth_type=GroundTruthType.OPTITRACK_MOCAP,
        ground_truth_sha256="ec553838981a5148987b7eee71aeddc837d473b3923e0e22408a0f629cc5b844",
        orbslam3_settings_path="configs/sensors/Bonn_Xtion.yaml",
        perception_config=PerceptionConfig(stride=2, max_depth_m=5.5),
        baseline_status=BaselineStatus.AVAILABLE,
        baseline_run_id="baseline_campaign_control",
        baseline_manifest_path="results/baselines/rgbd_bonn_crowd/run_manifest.json",
        baseline_trajectory_path="results/baselines/rgbd_bonn_crowd/CameraTrajectory.txt",
        baseline_trajectory_sha256="f50c4e9b9816548b2610d270e3c162074995a61ceede593b06dcff6aee8700c3",
        perception_run_id="run_20261101_234859",
        aliases=["bonn_dynamic_crowd", "rgbd_bonn_crowd"],
    ),
    SequenceRegistryEntry(
        sequence_id="bonn_box",
        sequence_name="rgbd_bonn_moving_obstructing_box",
        dataset_type="bonn",
        raw_dir="data/raw/rgbd_bonn_moving_obstructing_box",
        dataset_yaml="configs/datasets/bonn_moving_obstructing_box.yaml",
        rgb_file="data/raw/rgbd_bonn_moving_obstructing_box/rgb.txt",
        depth_file="data/raw/rgbd_bonn_moving_obstructing_box/depth.txt",
        groundtruth_file="data/raw/rgbd_bonn_moving_obstructing_box/groundtruth.txt",
        depth_scale=0.0002,
        association_tolerance_sec=0.02,
        intrinsics=_BONN_INTRINSICS,
        has_ground_truth=True,
        ground_truth_type=GroundTruthType.OPTITRACK_MOCAP,
        ground_truth_sha256="84d3420a6bf5deee7a523c29f97142203d3458399f6c7d35517ab9d024449ba0",
        orbslam3_settings_path="configs/sensors/Bonn_Xtion.yaml",
        perception_config=PerceptionConfig(stride=2, max_depth_m=5.5),
        baseline_status=BaselineStatus.AVAILABLE,
        baseline_run_id="baseline_campaign_control",
        baseline_manifest_path="results/baselines/rgbd_bonn_moving_obstructing_box/run_manifest.json",
        baseline_trajectory_path="results/baselines/rgbd_bonn_moving_obstructing_box/CameraTrajectory.txt",
        baseline_trajectory_sha256="371ff635fbcd489e5eb36c45c21e27977290f603640ebfeb9926c33aa3448227",
        perception_run_id="run_20261101_234859",
        aliases=["bonn_moving_box", "rgbd_bonn_moving_obstructing_box"],
    ),
    SequenceRegistryEntry(
        sequence_id="d455_01",
        sequence_name="my_desk_recording_01",
        dataset_type="realsense_d455",
        raw_dir="data/raw/my_desk_recording_01",
        dataset_yaml="configs/datasets/realsense_d455_recording_01.yaml",
        rgb_file="data/raw/my_desk_recording_01/rgb.txt",
        depth_file="data/raw/my_desk_recording_01/depth.txt",
        groundtruth_file=None,
        depth_scale=0.001,
        association_tolerance_sec=0.03,
        intrinsics=_D455_INTRINSICS,
        has_ground_truth=False,
        ground_truth_type=GroundTruthType.NONE,
        ground_truth_sha256=None,
        orbslam3_settings_path="configs/sensors/realsense_d455.yaml",
        perception_config=PerceptionConfig(stride=5, max_depth_m=5.5),
        baseline_status=BaselineStatus.AVAILABLE,
        baseline_run_id="baseline_campaign_control",
        baseline_manifest_path="results/baselines/my_desk_recording_01/run_manifest.json",
        baseline_trajectory_path="results/baselines/my_desk_recording_01/CameraTrajectory.txt",
        baseline_trajectory_sha256="46fe8c3c64de7878a8d53bd35aae64b2b9881fe909048c3acd3a18eccf601851",
        perception_run_id="run_20261101_234859",
        aliases=["realsense_d455_01", "my_desk_recording_01", "desk_01"],
    ),
    SequenceRegistryEntry(
        sequence_id="d455_02",
        sequence_name="my_desk_recording_02",
        dataset_type="realsense_d455",
        raw_dir="data/raw/my_desk_recording_02",
        dataset_yaml="configs/datasets/realsense_d455_recording_02.yaml",
        rgb_file="data/raw/my_desk_recording_02/rgb.txt",
        depth_file="data/raw/my_desk_recording_02/depth.txt",
        groundtruth_file=None,
        depth_scale=0.001,
        association_tolerance_sec=0.03,
        intrinsics=_D455_INTRINSICS,
        has_ground_truth=False,
        ground_truth_type=GroundTruthType.NONE,
        ground_truth_sha256=None,
        orbslam3_settings_path="configs/sensors/realsense_d455.yaml",
        perception_config=PerceptionConfig(stride=5, max_depth_m=5.5),
        baseline_status=BaselineStatus.AVAILABLE,
        baseline_run_id="baseline_campaign_control",
        baseline_manifest_path="results/baselines/my_desk_recording_02/run_manifest.json",
        baseline_trajectory_path="results/baselines/my_desk_recording_02/CameraTrajectory.txt",
        baseline_trajectory_sha256="18cded24473120010e46368a6f69c6b998e240b13ee080c704fc2e8c81ff625c",
        perception_run_id="run_20261101_234859",
        aliases=["realsense_d455_02", "my_desk_recording_02", "desk_02"],
    ),
]


# ---------------------------------------------------------------------------
# Registry Access Functions
# ---------------------------------------------------------------------------

_REGISTRY_BY_ID: Dict[str, SequenceRegistryEntry] = {e.sequence_id: e for e in CANONICAL_ENTRIES}
_LOOKUP_MAP: Dict[str, SequenceRegistryEntry] = {}
for _e in CANONICAL_ENTRIES:
    _LOOKUP_MAP[_e.sequence_id.lower()] = _e
    _LOOKUP_MAP[_e.sequence_name.lower()] = _e
    for _alias in _e.aliases:
        _LOOKUP_MAP[_alias.lower()] = _e


def get_sequence(identifier: str) -> SequenceRegistryEntry:
    """Resolve a sequence by canonical ID, directory name, or alias (case-insensitive)."""
    key = identifier.strip().lower()
    if key in _LOOKUP_MAP:
        return _LOOKUP_MAP[key]
    raise KeyError(
        f"Unknown sequence '{identifier}'. Valid canonical IDs: {list_sequence_ids()}"
    )


def list_sequences() -> List[SequenceRegistryEntry]:
    """Return all eight canonical sequence registry entries in standard order."""
    return list(CANONICAL_ENTRIES)


def list_sequence_ids() -> List[str]:
    """Return canonical sequence IDs."""
    return [e.sequence_id for e in CANONICAL_ENTRIES]


def resolve_sequence(identifier: str) -> SequenceRegistryEntry:
    """Alias for get_sequence."""
    return get_sequence(identifier)


def resolve_settings_for_sequence(identifier: str, repo_root: Optional[Path] = None) -> Path:
    """Return the absolute path to the sensor settings file for the given sequence."""
    entry = get_sequence(identifier)
    return entry.resolve_settings_path(repo_root=repo_root)


def get_all_sequence_specs(repo_root: Optional[Path] = None) -> List[Any]:
    """Return SequenceSpec objects for all 8 sequences for unified perception pipeline."""
    return [e.to_sequence_spec(repo_root=repo_root) for e in CANONICAL_ENTRIES]


def validate_registry(repo_root: Optional[Path] = None) -> Dict[str, Any]:
    """Perform preflight validation across all 8 registry entries."""
    root = repo_root or get_repo_root()
    report: Dict[str, Any] = {"total_sequences": len(CANONICAL_ENTRIES), "entries": {}}

    for entry in CANONICAL_ENTRIES:
        raw_exists = entry.resolve_raw_dir(root).exists()
        yaml_exists = entry.resolve_yaml_path(root).exists()
        settings_exists = entry.resolve_settings_path(root).exists()
        gt_exists = entry.resolve_groundtruth_path(root).exists() if entry.has_ground_truth else True
        traj_exists = entry.resolve_baseline_trajectory(root) is not None if entry.baseline_status == BaselineStatus.AVAILABLE else True

        report["entries"][entry.sequence_id] = {
            "sequence_name": entry.sequence_name,
            "raw_dir_exists": raw_exists,
            "dataset_yaml_exists": yaml_exists,
            "settings_exists": settings_exists,
            "has_ground_truth": entry.has_ground_truth,
            "gt_file_exists": gt_exists,
            "baseline_status": entry.baseline_status.value,
            "baseline_traj_exists": traj_exists,
            "all_preflight_passed": all([raw_exists, yaml_exists, settings_exists, gt_exists, traj_exists]),
        }

    return report


def export_registry_to_json(dest: Path, repo_root: Optional[Path] = None) -> None:
    """Export canonical sequence registry to JSON file."""
    payload = {
        "schema_version": "1.0",
        "total_sequences": len(CANONICAL_ENTRIES),
        "sequences": [e.to_dict() for e in CANONICAL_ENTRIES],
    }
    dest.parent.mkdir(parents=True, exist_ok=True)
    with open(dest, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


