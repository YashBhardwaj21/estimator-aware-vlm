"""Centralized repository root and path resolution helper.

Guarantees working-directory independence and eliminates fragile relative parent lookups.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Union


def get_repo_root() -> Path:
    """Resolve repository root directory reliably.
    
    Checks in order:
    1. REPO_ROOT environment variable if set and valid.
    2. Ascending parent directories of this file containing 'pyproject.toml' or '.git'.
    3. Structural fallback based on known source tree location (src/estimator_aware_vlm/common/paths.py -> parents[3]).
    """
    env_root = os.environ.get("REPO_ROOT")
    if env_root:
        p = Path(env_root).resolve()
        if p.exists() and p.is_dir():
            return p

    current = Path(__file__).resolve()
    for parent in [current] + list(current.parents):
        if (parent / "pyproject.toml").exists() or (parent / ".git").exists():
            return parent

    # Fallback: paths.py is at src/estimator_aware_vlm/common/paths.py
    # parents: [0]=common, [1]=estimator_aware_vlm, [2]=src, [3]=repo_root
    return current.parents[3]


def resolve_path(relative_or_absolute: Union[str, Path]) -> Path:
    """Resolve a path relative to repo root if not already absolute."""
    p = Path(relative_or_absolute)
    if p.is_absolute():
        return p
    return (get_repo_root() / p).resolve()


def get_benchmarks_dir() -> Path:
    """Return canonical benchmarks directory."""
    return get_repo_root() / "benchmarks"


def get_outputs_dir() -> Path:
    """Return canonical outputs directory."""
    return get_repo_root() / "outputs"


def get_data_dir() -> Path:
    """Return canonical data directory."""
    return get_repo_root() / "data"


def get_configs_dir() -> Path:
    """Return canonical configs directory."""
    return get_repo_root() / "configs"


def get_raw_data_dir() -> Path:
    """Return canonical raw recording data directory (data/raw)."""
    return get_repo_root() / "data" / "raw"


def get_processed_data_dir() -> Path:
    """Return canonical processed RGB-D datasets directory (data/processed)."""
    return get_repo_root() / "data" / "processed"


def get_mapping_outputs_dir() -> Path:
    """Return canonical mapping outputs directory (outputs/mapping)."""
    return get_repo_root() / "outputs" / "mapping"
