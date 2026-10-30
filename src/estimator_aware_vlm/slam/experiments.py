"""Experiment directory management and cryptographic run manifest generation for SLAM."""

from __future__ import annotations

import datetime
import hashlib
import json
import os
import platform
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from estimator_aware_vlm.common.paths import get_repo_root


def get_git_commit_sha() -> str:
    """Retrieve current git commit hash, or return fallback string."""
    try:
        res = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            check=True,
            cwd=str(get_repo_root()),
        )
        return res.stdout.strip()
    except Exception:
        return "UNKNOWN_COMMIT"


def compute_file_sha256(path: Path) -> str:
    """Compute hex SHA256 digest of a file."""
    if not path.exists():
        return "FILE_NOT_FOUND"
    hasher = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()


def create_experiment_dir(
    experiment_id: Optional[str] = None,
    prefix: str = "slam_exp",
    base_dir: Optional[Union[str, Path]] = None,
) -> Path:
    """Create and return an isolated experiment output directory.

    Args:
        experiment_id: Explicit unique ID for experiment. If None, generated with timestamp.
        prefix: Prefix for auto-generated experiment folder names.
        base_dir: Base directory for experiments (defaults to <repo_root>/outputs/experiments).

    Returns:
        Absolute Path to newly created isolated experiment directory.
    """
    repo_root = get_repo_root()
    if base_dir is None:
        experiments_base = repo_root / "outputs" / "experiments"
    else:
        experiments_base = Path(base_dir).resolve()

    experiments_base.mkdir(parents=True, exist_ok=True)

    if experiment_id is None:
        timestamp_str = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        experiment_id = f"{prefix}_{timestamp_str}"

    exp_dir = (experiments_base / experiment_id).resolve()
    exp_dir.mkdir(parents=True, exist_ok=True)
    return exp_dir


def export_run_manifest(
    experiment_dir: Path,
    run_id: str,
    pipeline_module: Optional[str] = None,
    config: Optional[Dict[str, Any]] = None,
    input_files: Optional[Dict[str, Path]] = None,
    output_artifacts: Optional[Dict[str, Path]] = None,
    metrics: Optional[Dict[str, Any]] = None,
    git_commit: Optional[str] = None,
    **kwargs: Any,
) -> Path:
    """Export a cryptographic run manifest recording provenance and artifact hashes.

    Args:
        experiment_dir: Path to experiment folder.
        run_id: Identifier of the run.
        pipeline_module: Name of pipeline execution module (e.g. 'Baseline SLAM Subsystem').
        config: Dictionary of runtime parameters and flags.
        input_files: Dictionary mapping labels to input file paths.
        output_artifacts: Dictionary mapping labels to output artifact paths.
        metrics: Optional summary metrics (e.g. ATE, RPE).
        git_commit: Optional git commit SHA (resolved automatically if omitted).

    Returns:
        Path to generated run_manifest.json.
    """
    repo_root = get_repo_root()
    commit_sha = git_commit or get_git_commit_sha()
    module_name = pipeline_module or "Baseline SLAM Subsystem"

    # Index inputs
    indexed_inputs = {}
    if input_files:
        for label, p in input_files.items():
            path_obj = Path(p).resolve()
            try:
                rel_path = str(path_obj.relative_to(repo_root)).replace("\\", "/")
            except ValueError:
                rel_path = str(path_obj).replace("\\", "/")
            indexed_inputs[label] = {
                "path": rel_path,
                "sha256": compute_file_sha256(path_obj) if path_obj.is_file() else "DIRECTORY",
            }

    # Index outputs
    indexed_outputs = {}
    if output_artifacts:
        for label, p in output_artifacts.items():
            path_obj = Path(p).resolve()
            try:
                rel_path = str(path_obj.relative_to(repo_root)).replace("\\", "/")
            except ValueError:
                rel_path = str(path_obj).replace("\\", "/")
            indexed_outputs[label] = {
                "path": rel_path,
                "sha256": compute_file_sha256(path_obj) if path_obj.is_file() else "DIRECTORY",
            }

    manifest_payload = {
        "run_id": run_id,
        "pipeline_module": module_name,
        "timestamp_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "git_commit_sha": commit_sha,
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "python_version": platform.python_version(),
        },
        "config": config,
        "input_files": indexed_inputs,
        "output_artifacts": indexed_outputs,
        "metrics": metrics or {},
    }

    manifest_path = experiment_dir / "run_manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest_payload, f, indent=2)

    return manifest_path

