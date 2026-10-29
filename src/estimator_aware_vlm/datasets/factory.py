"""Dataset Factory for Estimator-Aware VLM.

Provides unified dispatch for instantiating dataset adapters based on configuration.
"""

from __future__ import annotations

from pathlib import Path
from typing import Union

from estimator_aware_vlm.datasets.base import (
    DatasetAdapter,
    DatasetConfig,
    load_dataset_config_from_yaml,
)
from estimator_aware_vlm.datasets.openloris import OpenLORISAdapter
from estimator_aware_vlm.datasets.tum_rgbd import TUMRGBDAdapter


def get_dataset_adapter(config_or_path: Union[DatasetConfig, str, Path]) -> DatasetAdapter:
    """Instantiate the appropriate DatasetAdapter for a dataset configuration or YAML path.
    
    Supports:
        - "tum_rgbd" / "tum": TUMRGBDAdapter
        - "openloris" / "openloris_scene": OpenLORISAdapter
    """
    if isinstance(config_or_path, (str, Path)):
        config = load_dataset_config_from_yaml(config_or_path)
    elif isinstance(config_or_path, DatasetConfig):
        config = config_or_path
    else:
        raise TypeError(f"Expected DatasetConfig or path, got {type(config_or_path).__name__}")

    dtype = config.dataset_type.strip().lower()
    if dtype in ("tum_rgbd", "tum"):
        return TUMRGBDAdapter(config)
    elif dtype in ("openloris", "openloris_scene"):
        return OpenLORISAdapter(config)
    else:
        raise ValueError(
            f"Unsupported dataset_type '{config.dataset_type}'. Supported types: 'tum_rgbd', 'openloris'"
        )
