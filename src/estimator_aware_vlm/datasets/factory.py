"""Dataset Factory for Estimator-Aware VLM.

Provides unified dispatch for instantiating dataset adapters based on configuration.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Union

from estimator_aware_vlm.datasets.base import (
    DatasetAdapter,
    DatasetConfig,
    load_dataset_config_from_yaml,
)
from estimator_aware_vlm.datasets.openloris import OpenLORISAdapter
from estimator_aware_vlm.datasets.realsense_d455 import RealSenseD455Adapter
from estimator_aware_vlm.datasets.tum_rgbd import TUMRGBDAdapter


def get_dataset_adapter(
    config_or_path: Optional[Union[DatasetConfig, str, Path]] = None,
    **kwargs,
) -> DatasetAdapter:
    """Instantiate the appropriate DatasetAdapter for a dataset configuration or YAML path.
    
    Supports:
        - "tum_rgbd" / "tum": TUMRGBDAdapter
        - "openloris" / "openloris_scene": OpenLORISAdapter
        - "realsense_d455" / "d455" / "d455_rgbd": RealSenseD455Adapter
    Also accepts keyword arguments (e.g. dataset_type, dataset_path, sequence_dir) for flexibility.
    """
    if config_or_path is not None:
        if isinstance(config_or_path, (str, Path)):
            config = load_dataset_config_from_yaml(config_or_path)
        elif isinstance(config_or_path, DatasetConfig):
            config = config_or_path
        else:
            raise TypeError(f"Expected DatasetConfig or path, got {type(config_or_path).__name__}")
    elif kwargs:
        # Build DatasetConfig from kwargs
        dtype = kwargs.get("dataset_type", "tum_rgbd")
        seq_path = kwargs.get("dataset_path") or kwargs.get("sequence_dir")
        if seq_path is None:
            raise ValueError("Must provide either config_or_path or dataset_path/sequence_dir.")
        seq_dir = Path(seq_path)
        config = DatasetConfig(
            dataset_type=dtype,
            sequence_name=kwargs.get("sequence_name", seq_dir.name),
            sequence_dir=seq_dir,
            rgb_file=kwargs.get("rgb_file"),
            depth_file=kwargs.get("depth_file"),
            groundtruth_file=kwargs.get("groundtruth_file"),
            depth_scale=kwargs.get("depth_scale", 0.0002 if "tum" in dtype else 0.001),
            intrinsics=kwargs.get("intrinsics") or kwargs.get("camera_config"),
        )
    else:
        raise ValueError("Must provide config_or_path or keyword configuration parameters.")

    dtype = config.dataset_type.strip().lower()
    if dtype in ("tum_rgbd", "tum"):
        return TUMRGBDAdapter(config)
    elif dtype in ("openloris", "openloris_scene"):
        return OpenLORISAdapter(config)
    elif dtype in ("realsense_d455", "d455", "d455_rgbd"):
        return RealSenseD455Adapter(config)
    else:
        raise ValueError(
            f"Unsupported dataset_type '{config.dataset_type}'. Supported types: 'tum_rgbd', 'openloris', 'realsense_d455'"
        )
