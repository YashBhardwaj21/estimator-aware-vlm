"""Estimator-Aware Vision-Language Model Subsystem.

Provides selective multimodal reasoning for physical object affordances
and landmark lifecycle management in dynamic indoor environments.
"""

__version__ = "1.0.0"

from .common.paths import (
    get_repo_root,
    get_benchmarks_dir,
    get_outputs_dir,
    get_data_dir,
    get_configs_dir,
    resolve_path,
)

__all__ = [
    "__version__",
    "get_repo_root",
    "get_benchmarks_dir",
    "get_outputs_dir",
    "get_data_dir",
    "get_configs_dir",
    "resolve_path",
]
