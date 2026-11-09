"""Dataset integrity, sensory synchronization, and preflight contract verification.

Performs exhaustive verification of association files, image headers, temporal synchronization,
camera calibrations, depth encodings, and ground-truth trajectory contracts across all benchmark sequences.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

repo_root_candidate = Path(__file__).resolve().parent.parent
src_dir = repo_root_candidate / "src"
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

from estimator_aware_vlm.common.paths import get_repo_root
from estimator_aware_vlm.datasets import (
    CameraIntrinsics,
    DatasetAdapter,
    FrameRecord,
    TrajectoryRole,
    get_dataset_adapter,
    load_dataset_config_from_yaml,
)
from estimator_aware_vlm.datasets.registry import (
    GroundTruthType,
    SequenceRegistryEntry,
    list_sequences,
)

PNG_MAGIC = b"\x89PNG\r\n\x1a\n"
JPEG_MAGIC = b"\xff\xd8\xff"


def verify_image_file_header(file_path: Path) -> Tuple[bool, str]:
    """Verify image file exists, is non-empty, and has valid PNG/JPEG magic bytes."""
    try:
        with open(file_path, "rb") as f:
            header = f.read(8)
        if len(header) == 0:
            return False, "ZERO_BYTE_FILE"
        if header.startswith(PNG_MAGIC) or header.startswith(JPEG_MAGIC[:2]):
            return True, "VALID_HEADER"
        return False, f"INVALID_HEADER_MAGIC_{header[:4].hex()}"
    except FileNotFoundError:
        return False, "FILE_NOT_FOUND"
    except Exception as exc:
        return False, f"IO_ERROR_{exc}"


def audit_raw_stream_file(
    file_path: Path,
) -> Tuple[List[Tuple[float, str]], int, int, List[str]]:
    """Parse raw stream index file, verifying formatting and detecting duplicate timestamps or paths."""
    entries: List[Tuple[float, str]] = []
    issues: List[str] = []
    if not file_path.exists():
        return [], 0, 0, [f"Missing raw stream index: {file_path}"]

    seen_ts: Dict[float, int] = {}
    seen_paths: Dict[str, int] = {}
    dup_ts_count = 0
    dup_path_count = 0

    with open(file_path, "r", encoding="utf-8", errors="replace") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 2:
                issues.append(f"{file_path.name}:{line_no}: Malformed line with < 2 tokens")
                continue
            try:
                ts = float(parts[0])
                rel_path = parts[1]
                entries.append((ts, rel_path))
                if ts in seen_ts:
                    dup_ts_count += 1
                seen_ts[ts] = seen_ts.get(ts, 0) + 1
                if rel_path in seen_paths:
                    dup_path_count += 1
                seen_paths[rel_path] = seen_paths.get(rel_path, 0) + 1
            except ValueError:
                issues.append(f"{file_path.name}:{line_no}: Invalid timestamp '{parts[0]}'")

    return entries, dup_ts_count, dup_path_count, issues


def verify_sequence_comprehensive(
    entry: SequenceRegistryEntry,
    repo_root: Path,
    sample_size: int = 50,
) -> Dict[str, Any]:
    """Perform exhaustive preflight validation across the 12 integrity criteria."""
    issues: List[str] = []
    seq_dir = entry.resolve_raw_dir(repo_root)

    # 1. Immutability check
    stray_files = [
        "associations.txt",
        "associations_pass_b.txt",
        "associations_pass_b.manifest.json",
    ]
    for sf in stray_files:
        if (seq_dir / sf).exists():
            issues.append(f"Immutability violation: stray file '{sf}' found in raw data directory '{seq_dir}'")
    if (seq_dir / "masked_pass_b").exists():
        issues.append(f"Immutability violation: stray folder 'masked_pass_b' found in raw data directory '{seq_dir}'")

    # 2. Raw stream index audit
    rgb_index_path = repo_root / entry.rgb_file
    depth_index_path = repo_root / entry.depth_file
    raw_rgb_entries, rgb_dup_ts, rgb_dup_paths, rgb_issues = audit_raw_stream_file(rgb_index_path)
    raw_depth_entries, depth_dup_ts, depth_dup_paths, depth_issues = audit_raw_stream_file(depth_index_path)
    issues.extend(rgb_issues)
    issues.extend(depth_issues)

    if rgb_dup_ts > 0:
        issues.append(f"Raw RGB stream contains {rgb_dup_ts} duplicate timestamps")
    if depth_dup_ts > 0:
        issues.append(f"Raw Depth stream contains {depth_dup_ts} duplicate timestamps")
    if rgb_dup_paths > 0:
        issues.append(f"Raw RGB stream contains {rgb_dup_paths} duplicate image paths")
    if depth_dup_paths > 0:
        issues.append(f"Raw Depth stream contains {depth_dup_paths} duplicate depth paths")

    # Audit raw file existence independently before adapter association filtering
    raw_missing_rgb = sum(1 for _, r_rel in raw_rgb_entries if not (seq_dir / r_rel).exists())
    raw_missing_depth = sum(1 for _, d_rel in raw_depth_entries if not (seq_dir / d_rel).exists())

    doc_excl = getattr(entry, "documented_exclusions", {}) or {}
    expected_missing_rgb = doc_excl.get("expected_raw_missing_rgb", 0)
    expected_missing_depth = doc_excl.get("expected_raw_missing_depth", 0)

    if raw_missing_rgb > expected_missing_rgb:
        issues.append(
            f"Raw RGB index references {raw_missing_rgb} files missing on disk in {seq_dir} (expected {expected_missing_rgb})"
        )
    if raw_missing_depth > expected_missing_depth:
        issues.append(
            f"Raw Depth index references {raw_missing_depth} files missing on disk in {seq_dir} (expected {expected_missing_depth})"
        )

    # 3. Instantiate Adapter
    try:
        adapter = entry.get_adapter(repo_root=repo_root)
    except Exception as exc:
        return {
            "sequence_id": entry.sequence_id,
            "sequence_name": entry.sequence_name,
            "status": "FAIL",
            "critical_issues": [f"Failed to instantiate adapter: {exc}"],
        }

    num_frames = len(adapter)
    if num_frames == 0:
        issues.append(f"Adapter produced 0 synchronized frames for {entry.sequence_name}")
        return {
            "sequence_id": entry.sequence_id,
            "sequence_name": entry.sequence_name,
            "status": "FAIL",
            "critical_issues": issues,
        }

    # 4. Intrinsics and distortion check
    intr = adapter.get_camera_intrinsics()
    if intr.fx <= 0 or intr.fy <= 0:
        issues.append(f"Invalid focal length: fx={intr.fx}, fy={intr.fy}")
    if intr.width <= 0 or intr.height <= 0:
        issues.append(f"Invalid image resolution: {intr.width}x{intr.height}")
    if intr.cx <= 0 or intr.cx >= intr.width:
        issues.append(f"Principal point cx={intr.cx} outside frame width [0, {intr.width}]")
    if intr.cy <= 0 or intr.cy >= intr.height:
        issues.append(f"Principal point cy={intr.cy} outside frame height [0, {intr.height}]")

    # 5. 100% Header and existence audit for all frames
    corrupt_rgb_count = 0
    corrupt_depth_count = 0
    missing_rgb_count = 0
    missing_depth_count = 0

    for idx, frame in enumerate(adapter.get_frames()):
        ok_rgb, err_rgb = verify_image_file_header(frame.rgb_path)
        if not ok_rgb:
            if err_rgb == "FILE_NOT_FOUND":
                missing_rgb_count += 1
            else:
                corrupt_rgb_count += 1
            if missing_rgb_count + corrupt_rgb_count <= 3:
                issues.append(f"Frame {idx} RGB defect: {err_rgb} at {frame.rgb_path}")

        ok_depth, err_depth = verify_image_file_header(frame.depth_path)
        if not ok_depth:
            if err_depth == "FILE_NOT_FOUND":
                missing_depth_count += 1
            else:
                corrupt_depth_count += 1
            if missing_depth_count + corrupt_depth_count <= 3:
                issues.append(f"Frame {idx} Depth defect: {err_depth} at {frame.depth_path}")

        if frame.get_safe_estimated_pose() is not None:
            issues.append(f"Security violation: Frame {idx} returned estimated pose prior to SLAM tracking")

    # 6. Synchronization statistics
    associated_pairs = adapter.associated_pairs
    sync_diffs_sec = np.array([abs(p[0] - p[2]) for p in associated_pairs], dtype=np.float64)
    sync_diffs_ms = sync_diffs_sec * 1000.0

    max_dt_sec = float(np.max(sync_diffs_sec)) if len(sync_diffs_sec) > 0 else 0.0
    mean_dt_ms = float(np.mean(sync_diffs_ms)) if len(sync_diffs_ms) > 0 else 0.0
    std_dt_ms = float(np.std(sync_diffs_ms)) if len(sync_diffs_ms) > 0 else 0.0
    median_dt_ms = float(np.median(sync_diffs_ms)) if len(sync_diffs_ms) > 0 else 0.0
    p95_dt_ms = float(np.percentile(sync_diffs_ms, 95)) if len(sync_diffs_ms) > 0 else 0.0
    p99_dt_ms = float(np.percentile(sync_diffs_ms, 99)) if len(sync_diffs_ms) > 0 else 0.0

    eff_tol = entry.association_tolerance_sec + 1e-6
    if max_dt_sec > eff_tol:
        issues.append(
            f"Max synchronization difference {max_dt_sec*1000:.2f}ms exceeds tolerance {entry.association_tolerance_sec*1000:.2f}ms"
        )

    # 7. Dense sample image decoding and depth verification
    step = max(1, num_frames // sample_size)
    sample_indices = list(range(0, num_frames, step))
    if (num_frames - 1) not in sample_indices:
        sample_indices.append(num_frames - 1)

    decoded_rgb_samples = 0
    decoded_depth_samples = 0
    invalid_depth_pixel_fractions = []
    metric_min_depths = []
    metric_max_depths = []

    for s_idx in sample_indices:
        frame_rec = adapter.get_frame(s_idx)
        rgb_mat = cv2.imread(str(frame_rec.rgb_path))
        if rgb_mat is None:
            issues.append(f"Frame {s_idx} failed RGB decoding at {frame_rec.rgb_path}")
        else:
            decoded_rgb_samples += 1
            if (rgb_mat.shape[0], rgb_mat.shape[1]) != (intr.height, intr.width):
                issues.append(f"RGB dimension mismatch: expected {intr.width}x{intr.height}, got {rgb_mat.shape[1]}x{rgb_mat.shape[0]}")
            if len(rgb_mat.shape) != 3 or rgb_mat.shape[2] != 3:
                issues.append(f"RGB channel mismatch: expected 3 channels, got {rgb_mat.shape}")

        depth_mat = cv2.imread(str(frame_rec.depth_path), cv2.IMREAD_UNCHANGED)
        if depth_mat is None:
            issues.append(f"Frame {s_idx} failed Depth decoding at {frame_rec.depth_path}")
        else:
            decoded_depth_samples += 1
            if (depth_mat.shape[0], depth_mat.shape[1]) != (intr.height, intr.width):
                issues.append(f"Depth dimension mismatch: expected {intr.width}x{intr.height}, got {depth_mat.shape[1]}x{depth_mat.shape[0]}")
            if depth_mat.dtype != np.uint16:
                issues.append(f"Depth bit-depth violation: expected uint16, got {depth_mat.dtype}")

            val_mask = (depth_mat > 0) & (depth_mat < 65535)
            invalid_frac = 1.0 - (float(np.sum(val_mask)) / float(depth_mat.size))
            invalid_depth_pixel_fractions.append(invalid_frac)

            if np.any(val_mask):
                depth_meters = depth_mat[val_mask].astype(np.float32) * frame_rec.depth_scale
                metric_min_depths.append(float(np.min(depth_meters)))
                metric_max_depths.append(float(np.max(depth_meters)))

    mean_invalid_depth_pct = float(np.mean(invalid_depth_pixel_fractions) * 100.0) if invalid_depth_pixel_fractions else 0.0
    global_min_depth_m = float(np.min(metric_min_depths)) if metric_min_depths else 0.0
    global_max_depth_m = float(np.max(metric_max_depths)) if metric_max_depths else 0.0

    if global_max_depth_m > 70.0:
        issues.append(f"Unrealistic indoor depth maximum: {global_max_depth_m:.2f}m (> 70m)")

    # 8. Frame duration, FPS, and jitter
    f0 = adapter.get_frame(0)
    f_last = adapter.get_frame(num_frames - 1)
    duration_sec = f_last.timestamp - f0.timestamp
    eff_fps = (num_frames - 1) / duration_sec if duration_sec > 0 else 0.0

    frame_timestamps = np.array([f.timestamp for f in adapter.get_frames()], dtype=np.float64)
    frame_intervals = np.diff(frame_timestamps)
    mean_frame_interval_ms = float(np.mean(frame_intervals) * 1000.0) if len(frame_intervals) > 0 else 0.0
    std_frame_interval_ms = float(np.std(frame_intervals) * 1000.0) if len(frame_intervals) > 0 else 0.0

    # 9. Ground truth trajectory and temporal overlap
    gt_poses_count = 0
    gt_span_str = "N/A"
    gt_matched_count = 0
    gt_overlap_pct = 0.0
    gt_evaluation_status = "NOT_EVALUATED"

    if adapter.has_ground_truth():
        gt_traj = adapter.get_ground_truth_trajectory()
        gt_poses_count = len(gt_traj)
        gt_evaluation_status = "EVALUATED"

        if gt_traj.role != TrajectoryRole.GROUND_TRUTH:
            issues.append(f"Ground truth trajectory role violation: {gt_traj.role.value}")

        gt_ts = gt_traj.timestamps
        if np.any(np.diff(gt_ts) <= 0):
            issues.append("Ground truth trajectory contains non-strictly monotonic timestamps")

        gt_pos = gt_traj.positions
        if np.any(~np.isfinite(gt_pos)):
            issues.append("Ground truth trajectory contains NaN or Inf coordinates")

        for p in gt_traj.poses:
            norm_q = float(np.linalg.norm(p.orientation))
            if abs(norm_q - 1.0) > 1e-3:
                issues.append(f"Ground truth non-unit quaternion: ||q|| = {norm_q}")
                break

        gt_span_str = f"{gt_ts[0]:.3f} - {gt_ts[-1]:.3f}"
        gt_matched_count = sum(1 for f in adapter.get_frames() if f.get_evaluation_ground_truth_pose() is not None)
        gt_overlap_pct = (gt_matched_count / num_frames) * 100.0

        if gt_overlap_pct < 95.0:
            issues.append(f"Ground truth temporal coverage {gt_overlap_pct:.1f}% is below 95.0% threshold")
    else:
        gt_evaluation_status = "EXCLUDED_BY_DESIGN (Intentionally ground-truth-free sensor capture)"

    status = "PASS" if len(issues) == 0 else "FAIL"

    return {
        "sequence_id": entry.sequence_id,
        "sequence_name": entry.sequence_name,
        "dataset_type": entry.dataset_type,
        "status": status,
        "num_frames": num_frames,
        "duration_sec": round(duration_sec, 3),
        "effective_fps": round(eff_fps, 2),
        "frame_interval_ms": {
            "mean": round(mean_frame_interval_ms, 2),
            "std_jitter": round(std_frame_interval_ms, 2),
        },
        "resolution": f"{intr.width}x{intr.height}",
        "depth_scale_m_per_unit": entry.depth_scale,
        "depth_range_m": {
            "min": round(global_min_depth_m, 3),
            "max": round(global_max_depth_m, 3),
            "invalid_pixel_pct": round(mean_invalid_depth_pct, 2),
        },
        "synchronization_ms": {
            "tolerance_limit": round(entry.association_tolerance_sec * 1000.0, 2),
            "max": round(float(np.max(sync_diffs_ms)), 2) if len(sync_diffs_ms) > 0 else 0.0,
            "mean": round(mean_dt_ms, 3),
            "std": round(std_dt_ms, 3),
            "median": round(median_dt_ms, 3),
            "p95": round(p95_dt_ms, 3),
            "p99": round(p99_dt_ms, 3),
        },
        "raw_stream_integrity": {
            "raw_rgb_lines": len(raw_rgb_entries),
            "raw_depth_lines": len(raw_depth_entries),
            "rgb_duplicate_timestamps": rgb_dup_ts,
            "depth_duplicate_timestamps": depth_dup_ts,
            "rgb_duplicate_paths": rgb_dup_paths,
            "depth_duplicate_paths": depth_dup_paths,
        },
        "image_file_verification": {
            "total_rgb_files_checked": num_frames,
            "total_depth_files_checked": num_frames,
            "missing_rgb": missing_rgb_count,
            "missing_depth": missing_depth_count,
            "corrupt_rgb": corrupt_rgb_count,
            "corrupt_depth": corrupt_depth_count,
            "decoded_samples_checked": decoded_rgb_samples,
        },
        "camera_calibration": {
            "fx": round(intr.fx, 4),
            "fy": round(intr.fy, 4),
            "cx": round(intr.cx, 4),
            "cy": round(intr.cy, 4),
            "distortion_model": intr.distortion_model,
            "distortion_coefficients": list(intr.distortion),
            "settings_path": entry.orbslam3_settings_path,
        },
        "ground_truth_contract": {
            "declared_has_ground_truth": entry.has_ground_truth,
            "ground_truth_type": entry.ground_truth_type.value,
            "ground_truth_poses": gt_poses_count,
            "ground_truth_span": gt_span_str,
            "matched_pairs": f"{gt_matched_count}/{num_frames}",
            "coverage_pct": round(gt_overlap_pct, 2),
            "evaluation_status": gt_evaluation_status,
        },
        "critical_issues": issues,
    }


def main() -> int:
    """Execute complete dataset integrity, preflight, and synchronization validation."""
    import argparse
    parser = argparse.ArgumentParser(description="Dataset integrity and preflight audit")
    parser.add_argument(
        "--ci",
        action="store_true",
        help="Run in CI mode (validating registry, baseline trajectories, and committed preflight deliverables)",
    )
    args, _ = parser.parse_known_args()

    repo_root = get_repo_root()
    entries = list_sequences()

    has_raw_data = any((repo_root / entry.raw_dir).exists() for entry in entries)

    if args.ci or not has_raw_data:
        print("=" * 80, flush=True)
        print("ESTIMATOR-AWARE VLM: CANONICAL REGISTRY & PREFLIGHT VALIDATION (CI MODE)", flush=True)
        print(f"Audit Root: {repo_root}", flush=True)
        print(f"Timestamp:  {datetime.now(timezone.utc).isoformat()}", flush=True)
        print("Notice: Raw sensor recording directories not present on disk (or --ci requested).", flush=True)
        print("Validating canonical registry, sensor configurations, and baseline digests...", flush=True)
        print("=" * 80, flush=True)

        from estimator_aware_vlm.datasets.registry import validate_registry
        registry_report = validate_registry(repo_root)
        ci_issues = 0

        for entry in entries:
            reg_entry = registry_report.get("entries", {}).get(entry.sequence_id, {})
            if not reg_entry.get("settings_exists", False):
                print(f"  [CRITICAL] Missing sensor settings: {entry.orbslam3_settings_path}", flush=True)
                ci_issues += 1
            if not reg_entry.get("dataset_yaml_exists", False):
                print(f"  [CRITICAL] Missing dataset config: {entry.dataset_yaml}", flush=True)
                ci_issues += 1
            if not reg_entry.get("baseline_traj_exists", False):
                print(f"  [CRITICAL] Missing baseline trajectory: {entry.baseline_trajectory_path}", flush=True)
                ci_issues += 1
            elif not reg_entry.get("baseline_traj_hash_valid", False):
                print(f"  [CRITICAL] Baseline trajectory SHA-256 mismatch: {entry.baseline_trajectory_path}", flush=True)
                ci_issues += 1
            if not reg_entry.get("baseline_manifest_exists", False):
                print(f"  [CRITICAL] Missing baseline manifest: {entry.baseline_manifest_path}", flush=True)
                ci_issues += 1

        preflight_dir = repo_root / "results" / "dataset_preflight"
        for req_deliverable in [
            "dataset_preflight.json",
            "dataset_inventory.json",
            "per_sequence_sync_stats.json",
            "calibration_validation.json",
            "missing_corrupt_frames.json",
            "canonical_registry.json",
            "preflight_summary.md",
        ]:
            deliv_p = preflight_dir / req_deliverable
            if not deliv_p.exists() or deliv_p.stat().st_size == 0:
                print(f"  [CRITICAL] Missing committed preflight deliverable: {deliv_p}", flush=True)
                ci_issues += 1

        if ci_issues == 0:
            print("  -> All 8 sequences validated: registry, sensor configs, and baseline digests verified.", flush=True)
            print("  -> All committed preflight deliverables verified.", flush=True)
            print("\n" + "=" * 80, flush=True)
            print("CI PREFLIGHT VALIDATION: PASS", flush=True)
            print("=" * 80, flush=True)
            return 0
        else:
            print(f"\nCI PREFLIGHT VALIDATION FAILED: {ci_issues} critical issue(s) detected.", flush=True)
            return 1

    print("=" * 80, flush=True)
    print("ESTIMATOR-AWARE VLM: COMPLETE DATASET INTEGRITY & SYNCHRONIZATION AUDIT", flush=True)
    print(f"Audit Root: {repo_root}", flush=True)
    print(f"Timestamp:  {datetime.now(timezone.utc).isoformat()}", flush=True)
    print("=" * 80, flush=True)

    results: List[Dict[str, Any]] = []
    total_issues = 0

    for entry in entries:
        t0 = time.perf_counter()
        print(f"\nAuditing [{entry.sequence_id}] ({entry.sequence_name})...", flush=True)
        meta = verify_sequence_comprehensive(entry, repo_root=repo_root)
        dt = time.perf_counter() - t0
        results.append(meta)
        total_issues += len(meta["critical_issues"])

        issue_str = f"{len(meta['critical_issues'])} issue(s)" if meta["critical_issues"] else "0 defects"
        print(f"  -> {meta['status']} in {dt:.2f}s [{issue_str}, {meta.get('num_frames', 0)} frames]", flush=True)
        if meta["critical_issues"]:
            for iss in meta["critical_issues"]:
                print(f"     [CRITICAL] {iss}", flush=True)

    # Canonical output directory in results (tracked in version control)
    out_dir = repo_root / "results" / "dataset_preflight"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Legacy output mirror for backwards compatibility
    legacy_dir = repo_root / "outputs" / "reports" / "dataset_preflight"
    legacy_dir.mkdir(parents=True, exist_ok=True)

    # 1. Export dataset_inventory.json
    inventory_file = out_dir / "dataset_inventory.json"
    inventory_summary = [
        {
            "sequence_id": r["sequence_id"],
            "sequence_name": r["sequence_name"],
            "dataset_type": r["dataset_type"],
            "status": r["status"],
            "num_frames": r["num_frames"],
            "resolution": r["resolution"],
            "depth_scale_m": r["depth_scale_m_per_unit"],
            "fps": r["effective_fps"],
            "gt_poses": r["ground_truth_contract"]["ground_truth_poses"],
            "gt_coverage_pct": r["ground_truth_contract"]["coverage_pct"],
            "gt_status": r["ground_truth_contract"]["evaluation_status"],
        }
        for r in results
    ]
    with open(inventory_file, "w", encoding="utf-8") as f:
        json.dump(inventory_summary, f, indent=2)

    # 2. Export per_sequence_sync_stats.json
    sync_file = out_dir / "per_sequence_sync_stats.json"
    sync_stats = {r["sequence_id"]: r["synchronization_ms"] for r in results if "synchronization_ms" in r}
    with open(sync_file, "w", encoding="utf-8") as f:
        json.dump(sync_stats, f, indent=2)

    # 3. Export calibration_validation.json
    calib_file = out_dir / "calibration_validation.json"
    calib_data = {r["sequence_id"]: r["camera_calibration"] for r in results if "camera_calibration" in r}
    with open(calib_file, "w", encoding="utf-8") as f:
        json.dump(calib_data, f, indent=2)

    # 4. Export missing_corrupt_frames.json
    defects_file = out_dir / "missing_corrupt_frames.json"
    defect_data = {
        r["sequence_id"]: {
            "image_verification": r["image_file_verification"],
            "issues": r["critical_issues"],
        }
        for r in results
        if "image_file_verification" in r
    }
    with open(defects_file, "w", encoding="utf-8") as f:
        json.dump(defect_data, f, indent=2)

    # 5. Validate canonical registry entries and integrate findings into audit verdict
    from estimator_aware_vlm.datasets.registry import validate_registry
    registry_report = validate_registry(repo_root)
    with open(out_dir / "canonical_registry.json", "w", encoding="utf-8") as f:
        json.dump(registry_report, f, indent=2)

    registry_issues = 0
    if not registry_report.get("all_passed", True):
        for seq_id, entry_info in registry_report.get("entries", {}).items():
            if not entry_info.get("all_preflight_passed", True):
                registry_issues += 1
                print(f"  [Registry Validation Failure] {seq_id}: {entry_info}")
    total_issues += registry_issues

    # 6. Export dataset_preflight.json (Complete Manifest)
    git_commit = "unknown"
    try:
        git_commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=repo_root, text=True
        ).strip()
    except Exception:
        pass

    preflight_manifest = {
        "report_type": "Canonical Dataset Preflight and Synchronization Audit",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": git_commit,
        "total_sequences": len(results),
        "total_critical_issues": total_issues,
        "overall_status": "PASS" if total_issues == 0 else "FAIL",
        "registry_validation_passed": registry_report.get("all_passed", True),
        "sequences": results,
    }
    preflight_file = out_dir / "dataset_preflight.json"
    with open(preflight_file, "w", encoding="utf-8") as f:
        json.dump(preflight_manifest, f, indent=2)

    # Legacy compatibility mirrors
    legacy_file = repo_root / "outputs" / "reports" / "dataset_inventory.json"
    legacy_file.parent.mkdir(parents=True, exist_ok=True)
    with open(legacy_file, "w", encoding="utf-8") as f:
        json.dump(preflight_manifest, f, indent=2)

    for item in out_dir.glob("*"):
        if item.is_file():
            shutil.copy2(item, legacy_dir / item.name)

    # 7. Export preflight_summary.md
    summary_md_file = out_dir / "preflight_summary.md"
    with open(summary_md_file, "w", encoding="utf-8") as f:
        f.write("# Dataset Preflight and Synchronization Summary\n\n")
        f.write(f"- **Audit Status**: `{'PASS' if total_issues == 0 else 'FAIL'}`\n")
        f.write(f"- **Total Sequences**: {len(results)}\n")
        f.write(f"- **Total Critical Issues**: {total_issues}\n")
        f.write(f"- **Registry Validation**: `{'PASS' if registry_report.get('all_passed', True) else 'FAIL'}`\n")
        f.write(f"- **Timestamp**: `{datetime.now(timezone.utc).isoformat()}`\n\n")
        f.write("## Sequence Status Inventory\n\n")
        f.write("| Sequence ID | Adapter | Frames | FPS | Max Sync $\\Delta t$ | Invalid Depth Pixel % | GT Overlap % | Verdict |\n")
        f.write("| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |\n")
        for r in results:
            sync_max = f"{r['synchronization_ms']['max']:.2f} ms" if "synchronization_ms" in r else "N/A"
            depth_pct = f"{r['depth_range_m']['invalid_pixel_pct']:.1f}%" if "depth_range_m" in r else "N/A"
            gt_cov = f"{r['ground_truth_contract']['coverage_pct']:.1f}%" if r["ground_truth_contract"]["declared_has_ground_truth"] else "N/A (No MoCap)"
            f.write(f"| `{r['sequence_id']}` | {r['dataset_type']} | {r['num_frames']} | {r['effective_fps']} | {sync_max} | {depth_pct} | {gt_cov} | **{r['status']}** |\n")

    if (legacy_dir / "preflight_summary.md").exists():
        shutil.copy2(summary_md_file, legacy_dir / "preflight_summary.md")

    print("\n" + "=" * 80)
    print("PREFLIGHT AUDIT SUMMARY:")
    print(f"Total Sequences Evaluated: {len(results)}")
    print(f"Total Defects Found:       {total_issues}")
    print(f"Audit Status:              {'ALL PASSED' if total_issues == 0 else 'FAILED'}")
    print(f"Deliverables Materialized: {out_dir}")
    print("=" * 80)

    return 0 if total_issues == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
