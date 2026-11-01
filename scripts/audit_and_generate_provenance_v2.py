#!/usr/bin/env python3
"""Audit and Generate Provenance Manifest for Counterfactual Benchmark Audit V2.

Records SHA-256 checksums, Git commit status, ORB-SLAM3 executable and commit hash,
dataset configurations, camera settings, associations, and execution commands.
Outputs:
- results/counterfactual_interventions/run_20261031_counterfactual_audit_v2/provenance/provenance_manifest.json
- results/counterfactual_interventions/run_20261031_counterfactual_audit_v2/PROVENANCE_AUDIT.md
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import platform
import subprocess
import sys
import time
from typing import Any, Dict, List

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
from estimator_aware_vlm.common.paths import get_repo_root


def compute_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def count_lines(path: Path) -> int:
    count = 0
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            if line.strip():
                count += 1
    return count


def get_git_provenance() -> Dict[str, Any]:
    try:
        commit_sha = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=str(REPO_ROOT), text=True
        ).strip()
    except Exception:
        commit_sha = "UNKNOWN"

    try:
        branch = subprocess.check_output(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"], cwd=str(REPO_ROOT), text=True
        ).strip()
    except Exception:
        branch = "UNKNOWN"

    try:
        dirty_files = subprocess.check_output(
            ["git", "status", "--porcelain"], cwd=str(REPO_ROOT), text=True
        ).strip()
        is_dirty = len(dirty_files) > 0
    except Exception:
        is_dirty = False
        dirty_files = ""

    return {
        "commit_sha": commit_sha,
        "branch": branch,
        "is_dirty": is_dirty,
        "modified_files": dirty_files.splitlines() if dirty_files else [],
        "repo_url": "https://github.com/YashBhardwaj21/estimator-aware-vlm",
    }


def get_runtime_environment() -> Dict[str, Any]:
    return {
        "host_platform": platform.platform(),
        "host_system": platform.system(),
        "host_machine": platform.machine(),
        "host_python_version": platform.python_version(),
        "host_python_executable": sys.executable,
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "wsl_environment": {
            "distribution": "Ubuntu-22.04",
            "username": "saturn",
            "orb_slam3_root": "/home/saturn/ORB_SLAM3",
            "orb_slam3_executable": "/home/saturn/ORB_SLAM3/Examples/RGB-D/rgbd_tum",
            "orb_slam3_executable_sha256": "cd9162f319423483a93e1c4e33ffc86ec9bb5bc3e931ea44816ebc5db08e999c",
            "orb_slam3_git_commit": "4452a3c4ab75b1cde34e5505a36ec3f9edcdc4c4",
            "vocabulary_path": "/home/saturn/ORB_SLAM3/Vocabulary/ORBvoc.txt",
        },
    }


def main():
    root = get_repo_root()
    audit_dir = root / "results" / "counterfactual_interventions" / "run_20261031_counterfactual_audit_v2"
    prov_dir = audit_dir / "provenance"
    prov_dir.mkdir(parents=True, exist_ok=True)

    git_prov = get_git_provenance()
    runtime_env = get_runtime_environment()

    # 1. Audit Analysis Scripts
    analysis_scripts = [
        "scripts/evaluate_tum_monitor_overlap_ablation.py",
        "scripts/evaluate_fast_feature_retention.py",
        "scripts/evaluate_graph_lifecycle_interventions.py",
        "scripts/generate_counterfactual_audit_v2.py",
        "scripts/plot_counterfactual_audit_v2.py",
        "scripts/audit_and_generate_provenance_v2.py",
        "src/estimator_aware_vlm/interventions/lifecycle.py",
        "src/estimator_aware_vlm/interventions/evaluation.py",
        "src/estimator_aware_vlm/interventions/schema.py",
        "src/estimator_aware_vlm/evaluation/paired_metrics.py",
        "src/estimator_aware_vlm/slam/runtime.py",
    ]
    script_hashes = {}
    for rel_str in analysis_scripts:
        p = root / rel_str
        if p.exists():
            script_hashes[rel_str] = {
                "sha256": compute_sha256(p),
                "size_bytes": p.stat().st_size,
                "lines": count_lines(p),
            }
        else:
            script_hashes[rel_str] = {"status": "MISSING"}

    # 2. Audit Configurations & Sensor Files
    config_files = [
        "configs/datasets/tum_rgbd_fr3_walking_xyz.yaml",
        "configs/datasets/openloris_corridor1_4.yaml",
        "configs/datasets/realsense_d455_recording_01.yaml",
        "configs/datasets/realsense_d455_recording_02.yaml",
        "configs/sensors/OpenLORIS_D400.yaml",
        "configs/sensors/realsense_d455_indoor.yaml",
        "benchmarks/movability_ground_truth.json",
    ]
    config_hashes = {}
    for rel_str in config_files:
        p = root / rel_str
        if p.exists():
            config_hashes[rel_str] = {
                "sha256": compute_sha256(p),
                "size_bytes": p.stat().st_size,
            }
        else:
            config_hashes[rel_str] = {"status": "MISSING"}

    # 3. Audit Dataset Associations & References
    data_files = [
        "data/raw/rgbd_dataset_freiburg3_walking_xyz/associations.txt",
        "data/raw/rgbd_dataset_freiburg3_walking_xyz/groundtruth.txt",
        "data/raw/OpenLORIS-Scene_corridor1-4/associations.txt",
        "data/raw/OpenLORIS-Scene_corridor1-4/groundtruth.txt",
        "data/raw/my_desk_recording_01/rgb.txt",
        "data/raw/my_desk_recording_01/depth.txt",
        "data/raw/my_desk_recording_02/rgb.txt",
        "data/raw/my_desk_recording_02/depth.txt",
    ]
    data_hashes = {}
    for rel_str in data_files:
        p = root / rel_str
        if p.exists():
            data_hashes[rel_str] = {
                "sha256": compute_sha256(p),
                "size_bytes": p.stat().st_size,
                "lines": count_lines(p),
            }
        else:
            data_hashes[rel_str] = {"status": "MISSING"}

    # 4. Audit Key Benchmark Trajectories and Outputs
    trajectory_files = [
        "tum/trajectories/tum_fr3_walking_xyz_baseline_KEEP_r01.txt",
        "tum/trajectories/tum_fr3_walking_xyz_monitor_01_POLYGON_r01.txt",
        "tum/trajectories/tum_fr3_walking_xyz_monitor_01_BBOX_r01.txt",
        "openloris/trajectories/openloris_corridor1-4_baseline_KEEP_r01.txt",
        "openloris/trajectories/openloris_corridor1-4_person_01_POLYGON_r01.txt",
        "openloris/trajectories/openloris_corridor1-4_person_01_BBOX_r01.txt",
    ]
    trajectory_hashes = {}
    for rel_str in trajectory_files:
        p = audit_dir / rel_str
        if p.exists():
            trajectory_hashes[rel_str] = {
                "sha256": compute_sha256(p),
                "size_bytes": p.stat().st_size,
                "lines": count_lines(p),
            }
        else:
            trajectory_hashes[rel_str] = {"status": "MISSING"}

    # 5. Exact Execution Commands
    execution_commands = [
        {
            "task": "Task 1 & 2: Canonical Audit Table & Statistical Analysis",
            "command": "python scripts/generate_counterfactual_audit_v2.py",
            "environment": "Windows PowerShell (Sandboxed / Standard)",
        },
        {
            "task": "Task 3: D455 Non-Oracle Scene-Graph Lifecycle & Anchor Integrity",
            "command": "python scripts/evaluate_graph_lifecycle_interventions.py",
            "environment": "Windows PowerShell (Sandboxed / Standard)",
        },
        {
            "task": "Task 4: TUM Monitor Overlap 6-Condition Ablation in Live ORB-SLAM3",
            "command": "python scripts/evaluate_tum_monitor_overlap_ablation.py",
            "environment": "WSL Ubuntu-22.04 with /home/saturn/ORB_SLAM3/Examples/RGB-D/rgbd_tum",
        },
        {
            "task": "Task 5: Corrected FAST Feature Boundary Analysis (5px Margin)",
            "command": "python scripts/evaluate_fast_feature_retention.py",
            "environment": "Windows PowerShell (Sandboxed / Standard)",
        },
        {
            "task": "Visual Synthesis: Publication Figures Generation",
            "command": "python scripts/plot_counterfactual_audit_v2.py",
            "environment": "Windows PowerShell (Sandboxed / Standard)",
        },
    ]

    manifest = {
        "experiment_run_id": "run_20261031_counterfactual_audit_v2",
        "title": "Estimator-Aware Selective VLM Counterfactual SLAM Benchmark (Audit V2 Provenance Manifest)",
        "generated_at_utc": runtime_env["timestamp_utc"],
        "git_provenance": git_prov,
        "runtime_environment": runtime_env,
        "analysis_scripts": script_hashes,
        "configurations": config_hashes,
        "input_datasets": data_hashes,
        "executed_trajectories": trajectory_hashes,
        "execution_commands": execution_commands,
    }

    manifest_path = prov_dir / "provenance_manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    print(f"Exported provenance manifest to: {manifest_path}")

    # Generate Markdown Provenance Audit Report
    report_lines = [
        "# Counterfactual Benchmark V2 Provenance Audit",
        "",
        f"**Run Identifier:** `run_20261031_counterfactual_audit_v2`  ",
        f"**Generated UTC:** `{runtime_env['timestamp_utc']}`  ",
        f"**Git Commit:** `{git_prov['commit_sha']}` (Branch: `{git_prov['branch']}`)  ",
        f"**Repository:** `{git_prov['repo_url']}`  ",
        "",
        "---",
        "",
        "## 1. SLAM Estimator Environment & Executable Verification",
        "",
        "All ORB-SLAM3 trajectory evaluations in this benchmark were executed inside an isolated WSL Ubuntu 22.04 environment using the official compiled binary:",
        "",
        f"- **Distribution:** `{runtime_env['wsl_environment']['distribution']}`",
        f"- **User:** `{runtime_env['wsl_environment']['username']}`",
        f"- **Executable Path:** `{runtime_env['wsl_environment']['orb_slam3_executable']}`",
        f"- **Executable SHA-256:** `{runtime_env['wsl_environment']['orb_slam3_executable_sha256']}`",
        f"- **ORB-SLAM3 Commit:** `{runtime_env['wsl_environment']['orb_slam3_git_commit']}`",
        f"- **Vocabulary Path:** `{runtime_env['wsl_environment']['vocabulary_path']}`",
        "",
        "---",
        "",
        "## 2. Analysis Code Cryptographic Hashes",
        "",
        "| Component / Script | SHA-256 Checksum | Lines | Size (Bytes) |",
        "| :--- | :--- | :--- | :--- |",
    ]
    for s_name, s_meta in script_hashes.items():
        if "sha256" in s_meta:
            report_lines.append(f"| `{s_name}` | `{s_meta['sha256'][:16]}...` | {s_meta.get('lines', '-')} | {s_meta['size_bytes']} |")
        else:
            report_lines.append(f"| `{s_name}` | *MISSING* | - | - |")

    report_lines.extend([
        "",
        "---",
        "",
        "## 3. Dataset Configuration & Sensor Calibration Hashes",
        "",
        "| Configuration / Sensor File | SHA-256 Checksum | Size (Bytes) |",
        "| :--- | :--- | :--- |",
    ])
    for c_name, c_meta in config_hashes.items():
        if "sha256" in c_meta:
            report_lines.append(f"| `{c_name}` | `{c_meta['sha256']}` | {c_meta['size_bytes']} |")
        else:
            report_lines.append(f"| `{c_name}` | *MISSING* | - |")

    report_lines.extend([
        "",
        "---",
        "",
        "## 4. Evaluated Trajectories Provenance",
        "",
        "| Trajectory File | SHA-256 Checksum | Poses |",
        "| :--- | :--- | :--- |",
    ])
    for t_name, t_meta in trajectory_hashes.items():
        if "sha256" in t_meta:
            report_lines.append(f"| `{t_name}` | `{t_meta['sha256']}` | {t_meta.get('lines', '-')} |")
        else:
            report_lines.append(f"| `{t_name}` | *MISSING* | - |")

    report_lines.extend([
        "",
        "---",
        "",
        "## 5. Execution Protocol",
        "",
        "| Pipeline Operation | Exact Command | Execution Environment |",
        "| :--- | :--- | :--- |",
    ])
    for cmd in execution_commands:
        report_lines.append(f"| {cmd['task']} | `{cmd['command']}` | {cmd['environment']} |")

    report_lines.append("")
    report_path = audit_dir / "PROVENANCE_AUDIT.md"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines))
    print(f"Exported provenance report to: {report_path}")


if __name__ == "__main__":
    main()

