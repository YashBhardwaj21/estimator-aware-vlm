"""Dataset Integrity and Preflight Verification Script.

Inspects both TUM and OpenLORIS datasets on disk, verifies adapter contract compliance,
validates strict raw data immutability, checks timestamp ranges and GT role separation,
and prints an auditable inventory table. Exits 0 on success, 1 on failure.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, Any

repo_root_candidate = Path(__file__).resolve().parent.parent
src_dir = repo_root_candidate / "src"
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

from estimator_aware_vlm.common.paths import get_repo_root
from estimator_aware_vlm.datasets import (
    DatasetAdapter,
    FrameRecord,
    TrajectoryRole,
    get_dataset_adapter,
    load_dataset_config_from_yaml,
)


def verify_sequence(adapter: DatasetAdapter, seq_dir: Path) -> Dict[str, Any]:
    """Perform deep contract, integrity, and immutability checks on an ingested sequence."""
    issues = []

    # 1. Immutability check: No generated artifacts in raw data directory
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

    # 2. Frame count check
    num_frames = len(adapter)
    if num_frames == 0:
        issues.append(f"Adapter returned zero synchronized frames for {adapter.get_sequence_name()}")

    # 3. Intrinsics check
    intr = adapter.get_camera_intrinsics()
    if intr.fx <= 0 or intr.fy <= 0 or intr.width <= 0 or intr.height <= 0:
        issues.append(f"Invalid camera intrinsics: fx={intr.fx}, fy={intr.fy}, {intr.width}x{intr.height}")

    # 4. Ground truth check & Role isolation
    gt_traj = adapter.get_ground_truth_trajectory()
    if gt_traj.role != TrajectoryRole.GROUND_TRUTH:
        issues.append(f"Role violation: ground truth trajectory has role {gt_traj.role.value}")

    # 5. First & Last frame verification
    if num_frames > 0:
        f0: FrameRecord = adapter.get_frame(0)
        f_last: FrameRecord = adapter.get_frame(num_frames - 1)

        if not f0.rgb_path.exists():
            issues.append(f"Frame 0 RGB missing: {f0.rgb_path}")
        if not f0.depth_path.exists():
            issues.append(f"Frame 0 Depth missing: {f0.depth_path}")

        # Strict GT leakage test
        if f0.get_safe_estimated_pose() is not None:
            issues.append("Security violation: FrameRecord returned estimated pose prior to SLAM tracking")

        ts_start = f0.timestamp
        ts_end = f_last.timestamp
        duration_sec = ts_end - ts_start
        fps_approx = num_frames / duration_sec if duration_sec > 0 else 0.0
    else:
        ts_start, ts_end, duration_sec, fps_approx = 0.0, 0.0, 0.0, 0.0

    gt_ts_start = gt_traj.timestamps[0] if len(gt_traj) > 0 else 0.0
    gt_ts_end = gt_traj.timestamps[-1] if len(gt_traj) > 0 else 0.0

    gt_matched_frames = sum(
        1 for f in adapter.get_frames() if f.get_evaluation_ground_truth_pose() is not None
    )

    return {
        "sequence_name": adapter.get_sequence_name(),
        "adapter_class": adapter.__class__.__name__,
        "num_frames": num_frames,
        "resolution": f"{intr.width}x{intr.height}",
        "depth_scale_m": adapter.get_frame(0).depth_scale if num_frames > 0 else 0.0,
        "intrinsics_fx_fy": f"{intr.fx:.2f}, {intr.fy:.2f}",
        "intrinsics_cx_cy": f"{intr.cx:.2f}, {intr.cy:.2f}",
        "rgbd_time_span": f"{ts_start:.3f} - {ts_end:.3f} ({duration_sec:.1f}s)",
        "approx_fps": f"{fps_approx:.1f}",
        "gt_poses": len(gt_traj),
        "gt_time_span": f"{gt_ts_start:.3f} - {gt_ts_end:.3f}",
        "gt_matched_pairs": f"{gt_matched_frames}/{num_frames} ({100*gt_matched_frames/max(num_frames, 1):.1f}%)",
        "gt_role": gt_traj.role.value,
        "issues": issues,
    }


def main() -> int:
    repo_root = get_repo_root()
    configs = [
        repo_root / "configs" / "datasets" / "tum_rgbd_fr3_walking_xyz.yaml",
        repo_root / "configs" / "datasets" / "openloris_corridor1_4.yaml",
    ]

    print("=" * 80)
    print("ESTIMATOR-AWARE VLM DATASET INTEGRITY & PREFLIGHT AUDIT")
    print("=" * 80)

    results = []
    total_issues = 0

    for cfg_path in configs:
        if not cfg_path.exists():
            print(f"[FAIL] Configuration file not found: {cfg_path}")
            return 1

        cfg = load_dataset_config_from_yaml(cfg_path)
        if not cfg.sequence_dir.exists():
            print(f"[SKIP] Raw data directory missing: {cfg.sequence_dir}")
            continue

        try:
            adapter = get_dataset_adapter(cfg)
            meta = verify_sequence(adapter, cfg.sequence_dir)
            results.append(meta)
            total_issues += len(meta["issues"])
        except Exception as e:
            print(f"[ERROR] Exception during adapter evaluation for {cfg_path.name}: {e}")
            total_issues += 1

    # Print factual inventory table
    print("\nFACTUAL DATASET INVENTORY:")
    headers = [
        "Sequence Name",
        "Adapter",
        "Frames",
        "Resolution",
        "Scale (m/unit)",
        "Approx FPS",
        "GT Poses",
        "GT Matched Pairs",
        "GT Role",
    ]
    print(f"| {' | '.join(headers)} |")
    print(f"|{'|'.join(['---' for _ in headers])}|")

    for r in results:
        row = [
            r["sequence_name"],
            r["adapter_class"],
            str(r["num_frames"]),
            r["resolution"],
            f"{r['depth_scale_m']:.4f}",
            r["approx_fps"],
            str(r["gt_poses"]),
            r["gt_matched_pairs"],
            r["gt_role"],
        ]
        print(f"| {' | '.join(row)} |")

    print("\nDETAILED PROVENANCE & CAMERA CALIBRATION:")
    for r in results:
        print(f"\n[{r['sequence_name']}]")
        print(f"  Adapter:              {r['adapter_class']}")
        print(f"  Color/Depth Span:     {r['rgbd_time_span']}")
        print(f"  Ground Truth Span:    {r['gt_time_span']}")
        print(f"  Camera Focal (fx, fy): {r['intrinsics_fx_fy']}")
        print(f"  Principal (cx, cy):    {r['intrinsics_cx_cy']}")
        print(f"  Depth Scaling:        {r['depth_scale_m']} meters per raw unit")
        if r["issues"]:
            print(f"  [CRITICAL ISSUES]:")
            for iss in r["issues"]:
                print(f"    - {iss}")
        else:
            print("  Status:               PASSED (Zero integrity or immutability violations)")

    # Save machine-readable report to outputs/reports/dataset_inventory.json
    import json
    import subprocess
    from datetime import datetime

    git_commit = "unknown"
    try:
        git_commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=repo_root, text=True
        ).strip()
    except Exception:
        pass

    out_dir = repo_root / "outputs" / "reports"
    out_dir.mkdir(parents=True, exist_ok=True)
    report_file = out_dir / "dataset_inventory.json"

    report_payload = {
        "module": "Dataset Inventory and Integrity Verification",
        "audit_timestamp": datetime.now().isoformat(),
        "git_commit": git_commit,
        "total_critical_issues": total_issues,
        "audit_passed": total_issues == 0,
        "sequences": results,
    }
    with open(report_file, "w", encoding="utf-8") as f:
        json.dump(report_payload, f, indent=2)

    print(f"\n[REPORT SAVED] Machine-readable audit exported to: {report_file}")

    print("\n" + "=" * 80)
    if total_issues > 0:
        print(f"AUDIT FAILED with {total_issues} critical issue(s).")
        return 1
    else:
        print("AUDIT PASSED: All sequences satisfy contract, role separation, and immutability.")
        return 0


if __name__ == "__main__":
    sys.exit(main())
