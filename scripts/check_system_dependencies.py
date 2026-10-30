#!/usr/bin/env python3
"""System Preflight and Dependency Diagnostics Tool.

Validates the local environment for Estimator-Aware VLM workflows:
- Core Python and scientific stack (numpy, scipy, torch, cv2, yaml)
- Hardware acceleration (CUDA)
- Optional hardware/recording drivers (pyrealsense2, rclpy / ROS 2)
- Optional SLAM binary paths (ORB_SLAM3_ROOT)
- Repository directory structure and benchmark readiness
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
import platform
import sys
from pathlib import Path

repo_root = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(repo_root / "scratch" / ".mplconfig"))
os.environ.setdefault("YOLO_VERBOSE", "False")
(repo_root / "scratch" / ".mplconfig").mkdir(parents=True, exist_ok=True)

if str(repo_root / "src") not in sys.path:
    sys.path.insert(0, str(repo_root / "src"))

from estimator_aware_vlm.common.paths import (
    get_repo_root,
    get_raw_data_dir,
    get_processed_data_dir,
    get_benchmarks_dir,
    get_outputs_dir,
)


def check_python_version() -> dict:
    vi = sys.version_info
    ver_str = f"{vi.major}.{vi.minor}.{vi.micro}"
    passed = (vi.major == 3 and vi.minor >= 10)
    return {
        "name": "python",
        "version": ver_str,
        "required": ">=3.10",
        "status": "PASS" if passed else "FAIL",
    }


def check_package(pkg_name: str, import_name: str | None = None, required: bool = True) -> dict:
    name = import_name or pkg_name
    try:
        mod = importlib.import_module(name)
        ver = getattr(mod, "__version__", "installed")
        return {
            "name": pkg_name,
            "version": str(ver),
            "required": required,
            "status": "PASS",
        }
    except ImportError as e:
        return {
            "name": pkg_name,
            "version": None,
            "required": required,
            "status": "FAIL" if required else "OPTIONAL_MISSING",
            "error": str(e),
        }


def check_cuda() -> dict:
    try:
        import torch
        available = torch.cuda.is_available()
        device_count = torch.cuda.device_count() if available else 0
        device_name = torch.cuda.get_device_name(0) if available else None
        return {
            "cuda_available": available,
            "device_count": device_count,
            "device_name": device_name,
            "status": "PASS" if available else "CPU_ONLY",
        }
    except ImportError:
        return {
            "cuda_available": False,
            "device_count": 0,
            "device_name": None,
            "status": "NO_TORCH",
        }


def check_orbslam3() -> dict:
    root = os.getenv("ORB_SLAM3_ROOT", "")
    if not root:
        for candidate in ["/opt/ORB_SLAM3", "/usr/local/ORB_SLAM3"]:
            if Path(candidate).exists():
                root = candidate
                break

    if root and Path(root).exists():
        exec_path = Path(root) / "Examples" / "RGB-D" / "rgbd_tum"
        vocab_path = Path(root) / "Vocabulary" / "ORBvoc.txt"
        return {
            "root": root,
            "executable_found": exec_path.exists(),
            "vocabulary_found": vocab_path.exists(),
            "status": "PASS" if (exec_path.exists() and vocab_path.exists()) else "PARTIAL",
        }
    return {
        "root": None,
        "executable_found": False,
        "vocabulary_found": False,
        "status": "OPTIONAL_MISSING",
    }


def check_directories() -> dict:
    root = get_repo_root()
    dirs = {
        "raw_data": get_raw_data_dir(),
        "processed_data": get_processed_data_dir(),
        "benchmarks": get_benchmarks_dir(),
        "outputs": get_outputs_dir(),
    }
    status = {}
    for k, d in dirs.items():
        exists = d.exists()
        status[k] = {
            "path": str(d.relative_to(root)),
            "exists": exists,
        }
    return status


def run_preflight(profile: str = "core") -> dict:
    py_check = check_python_version()

    require_core = True
    require_mapping = (profile in ("mapping", "full"))
    require_recording = (profile in ("recording", "full"))

    core_packages = [
        ("numpy", None, require_core),
        ("scipy", None, require_core),
        ("pillow", "PIL", require_core),
        ("pydantic", None, require_core),
        ("pyyaml", "yaml", require_core),
        ("pandas", None, require_core),
        ("scikit-learn", "sklearn", require_core),
        ("matplotlib", None, require_core),
        ("requests", None, require_core),
    ]

    mapping_packages = [
        ("opencv-python", "cv2", require_mapping),
        ("torch", None, require_mapping),
        ("torchvision", None, require_mapping),
        ("open3d", None, require_mapping),
        ("ultralytics", None, require_mapping),
    ]

    recording_packages = [
        ("rclpy", "rclpy", require_recording),
        ("sensor_msgs", "sensor_msgs", require_recording),
        ("pyrealsense2", "pyrealsense2", require_recording),
    ]

    core_results = [check_package(p, i, req) for p, i, req in core_packages]
    mapping_results = [check_package(p, i, req) for p, i, req in mapping_packages]
    rec_results = [check_package(p, i, req) for p, i, req in recording_packages]
    cuda_res = check_cuda()
    orbslam3_res = check_orbslam3()
    dir_res = check_directories()

    core_passed = (py_check["status"] == "PASS") and all(r["status"] == "PASS" for r in core_results)
    mapping_passed = all(r["status"] == "PASS" for r in mapping_results) if require_mapping else True
    recording_passed = all(r["status"] == "PASS" for r in rec_results) if require_recording else True

    overall_passed = core_passed and mapping_passed and recording_passed

    return {
        "system": {
            "platform": platform.platform(),
            "python_implementation": platform.python_implementation(),
            "architecture": platform.machine(),
            "profile": profile,
        },
        "python": py_check,
        "core_dependencies": core_results,
        "mapping_dependencies": mapping_results,
        "recording_dependencies": rec_results,
        "acceleration": cuda_res,
        "slam": orbslam3_res,
        "directories": dir_res,
        "overall_status": "READY" if overall_passed else "INCOMPLETE",
    }


def main():
    parser = argparse.ArgumentParser(description="System and Dependency Preflight Checker")
    parser.add_argument(
        "--profile",
        choices=["core", "mapping", "recording", "full"],
        default="core",
        help="Target dependency profile to validate (default: core)",
    )
    parser.add_argument(
        "--strict-mapping",
        action="store_true",
        help="Enforce all mapping and 3D point cloud dependencies",
    )
    parser.add_argument("--json", action="store_true", help="Output machine-readable JSON")
    args = parser.parse_args()

    profile = "mapping" if args.strict_mapping else args.profile
    report = run_preflight(profile=profile)

    if args.json:
        print(json.dumps(report, indent=2))
        sys.exit(0 if report["overall_status"] == "READY" else 1)

    print("=" * 68)
    print("ESTIMATOR-AWARE VLM — SYSTEM PREFLIGHT CHECK")
    print("=" * 68)
    print(f"Platform:      {report['system']['platform']} ({report['system']['architecture']})")
    print(f"Python:        {report['python']['version']} [{report['python']['status']}]")
    print(f"Target Profile: [{report['system']['profile'].upper()}]")
    print(f"Hardware Acc:  {report['acceleration']['status']} "
          + (f"({report['acceleration']['device_name']})" if report['acceleration']['device_name'] else ""))
    print("\n--- Core Packages (Required for Evaluation & Routing) ---")
    for r in report["core_dependencies"]:
        print(f"  {r['name']:<18}: {r['version'] or 'MISSING':<15} [{r['status']}]")

    print("\n--- Mapping & 3D Packages (Optional / Semantic Point Clouds) ---")
    for r in report["mapping_dependencies"]:
        print(f"  {r['name']:<18}: {r['version'] or 'NOT INSTALLED':<15} [{r['status']}]")

    print("\n--- Recording Packages (ROS 2 / Hardware Drivers) ---")
    for r in report["recording_dependencies"]:
        print(f"  {r['name']:<18}: {r['version'] or 'NOT FOUND':<15} [{r['status']}]")

    print("\n--- Optional SLAM Binary ---")
    print(f"  ORB-SLAM3:       [{report['slam']['status']}] (root: {report['slam']['root']})")

    print("\n--- Canonical Directories ---")
    for name, info in report["directories"].items():
        print(f"  {name:<18}: {info['path']:<25} [EXISTS: {info['exists']}]")

    print("=" * 68)
    print(f"OVERALL STATUS: {report['overall_status']}")
    print("=" * 68)

    sys.exit(0 if report["overall_status"] == "READY" else 1)


if __name__ == "__main__":
    main()
