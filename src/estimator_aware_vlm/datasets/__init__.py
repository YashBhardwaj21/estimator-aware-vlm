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
from estimator_aware_vlm.datasets.common_record import CommonObservationRecord
from estimator_aware_vlm.datasets.factory import get_dataset_adapter
from estimator_aware_vlm.datasets.bonn import (
    BONN_XTION_INTRINSICS,
    BonnDatasetAdapter,
    parse_bonn_stream_file,
    parse_bonn_ground_truth,
)
from estimator_aware_vlm.datasets.openloris import (
    OPENLORIS_D400_INTRINSICS,
    OpenLORISAdapter,
    parse_openloris_association_file,
    parse_openloris_ground_truth,
)
from estimator_aware_vlm.datasets.realsense_d455 import (
    REALSENSE_D455_INTRINSICS,
    RealSenseD455Adapter,
    parse_d455_association_file,
)
from estimator_aware_vlm.datasets.tum_rgbd import (
    TUMRGBDAdapter,
    TUM_FREIBURG3_INTRINSICS,
    parse_tum_association_file,
    parse_tum_ground_truth,
)

from estimator_aware_vlm.datasets.registry import (
    BaselineStatus,
    GroundTruthType,
    PerceptionConfig,
    SequenceRegistryEntry,
    get_sequence,
    list_sequences,
    list_sequence_ids,
    resolve_sequence,
    resolve_settings_for_sequence,
    validate_registry,
)

__all__ = [
    "BaselineStatus",
    "GroundTruthType",
    "PerceptionConfig",
    "SequenceRegistryEntry",
    "get_sequence",
    "list_sequences",
    "list_sequence_ids",
    "resolve_sequence",
    "resolve_settings_for_sequence",
    "validate_registry",
    "CameraIntrinsics",
    "CommonObservationRecord",
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
    "REALSENSE_D455_INTRINSICS",
    "RealSenseD455Adapter",
    "parse_d455_association_file",
    "TUMRGBDAdapter",
    "TUM_FREIBURG3_INTRINSICS",
    "parse_tum_association_file",
    "parse_tum_ground_truth",
    "BONN_XTION_INTRINSICS",
    "BonnDatasetAdapter",
    "parse_bonn_stream_file",
    "parse_bonn_ground_truth",
]
