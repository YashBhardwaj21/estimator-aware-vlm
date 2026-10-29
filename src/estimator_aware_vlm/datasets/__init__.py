"""Dataset abstractions and protocols for Estimator-Aware VLM."""

from estimator_aware_vlm.datasets.base import (
    CameraIntrinsics,
    DatasetAdapter,
    DatasetConfig,
    FrameRecord,
    Trajectory,
    TrajectoryPose,
    TrajectoryRole,
    associate_streams,
    load_dataset_config_from_yaml,
    parse_tum_trajectory,
    quaternion_slerp,
)
from estimator_aware_vlm.datasets.factory import get_dataset_adapter
from estimator_aware_vlm.datasets.openloris import (
    OPENLORIS_D400_INTRINSICS,
    OpenLORISAdapter,
    parse_openloris_association_file,
    parse_openloris_ground_truth,
)
from estimator_aware_vlm.datasets.tum_rgbd import (
    TUMRGBDAdapter,
    TUM_FREIBURG3_INTRINSICS,
    parse_tum_association_file,
    parse_tum_ground_truth,
)

__all__ = [
    "CameraIntrinsics",
    "DatasetAdapter",
    "DatasetConfig",
    "FrameRecord",
    "Trajectory",
    "TrajectoryPose",
    "TrajectoryRole",
    "associate_streams",
    "get_dataset_adapter",
    "load_dataset_config_from_yaml",
    "parse_tum_trajectory",
    "quaternion_slerp",
    "OPENLORIS_D400_INTRINSICS",
    "OpenLORISAdapter",
    "parse_openloris_association_file",
    "parse_openloris_ground_truth",
    "TUMRGBDAdapter",
    "TUM_FREIBURG3_INTRINSICS",
    "parse_tum_association_file",
    "parse_tum_ground_truth",
]
