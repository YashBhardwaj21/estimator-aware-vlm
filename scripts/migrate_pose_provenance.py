#!/usr/bin/env python3
"""Deterministic Versioned Migration Script: Pose Provenance.

Migrates 'pose_provenance' field in object observation datasets from:
  "MEASURED_TRAJECTORY" -> "ESTIMATED_SLAM_TRAJECTORY"

Rationale:
Independent inspection of raw data revealed that groundtruth.txt in both sessions
consists of synthetic identity placeholders (# Camera trajectory (SYNTHETIC IDENTITY
PLACEHOLDER - NO GROUND TRUTH AVAILABLE)), not optical motion capture.
Consequently, camera poses originate from visual-inertial SLAM trajectory estimates
(ORB-SLAM3/RTAB-Map), not an external ground-truth measurement system.
ATE/RPE evaluation claims are suppressed.
"""

import argparse
import hashlib
import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("migrate_pose_provenance")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def migrate_file(file_path: Path) -> Tuple[int, str, str]:
    """Migrate pose_provenance in a jsonl file and return (migrated_count, pre_sha, post_sha)."""
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    pre_sha = sha256_file(file_path)

    migrated_lines: List[str] = []
    count = 0
    with open(file_path, "r", encoding="utf-8") as f:
        for line_idx, line in enumerate(f, 1):
            if not line.strip():
                continue
            rec = json.loads(line)
            if rec.get("pose_provenance") == "MEASURED_TRAJECTORY":
                rec["pose_provenance"] = "ESTIMATED_SLAM_TRAJECTORY"
                count += 1
            migrated_lines.append(json.dumps(rec))

    with open(file_path, "w", encoding="utf-8") as f:
        for mline in migrated_lines:
            f.write(mline + "\n")

    post_sha = sha256_file(file_path)
    return count, pre_sha, post_sha


def main() -> None:
    parser = argparse.ArgumentParser(description="Migrate pose_provenance to ESTIMATED_SLAM_TRAJECTORY.")
    parser.add_argument(
        "--target",
        type=Path,
        default=None,
        help="Optional single target jsonl file to migrate. If omitted, migrates standard files.",
    )
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parent.parent.parent

    if args.target:
        targets = [args.target]
    else:
        targets = [
            repo_root / "vlm" / "data" / "observations" / "object_observations.jsonl",
            repo_root / "vlm" / "tests" / "fixtures" / "real_pipeline_fixture" / "fixture_observations.jsonl",
        ]

    for target in targets:
        logger.info(f"Processing: {target}")
        migrated, pre_sha, post_sha = migrate_file(target)
        logger.info(f"  Migrated records: {migrated}")
        logger.info(f"  Pre-SHA256:       {pre_sha}")
        logger.info(f"  Post-SHA256:      {post_sha}")


if __name__ == "__main__":
    main()
