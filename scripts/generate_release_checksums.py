"""Canonical Release Checksum Generator for Estimator-Aware VLM.

Computes SHA-256 digests and byte sizes for all canonical release assets,
ensuring immutable cryptographic provenance.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Dict, Any


def compute_sha256(file_path: Path) -> str:
    h = hashlib.sha256()
    with open(file_path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def generate_checksum_manifest(
    target_dir: Path,
    output_manifest_path: Path | None = None,
) -> Dict[str, Dict[str, Any]]:
    target_dir = target_dir.resolve()
    if output_manifest_path is None:
        output_manifest_path = target_dir / "release_checksums.json"
    else:
        output_manifest_path = output_manifest_path.resolve()

    manifest: Dict[str, Dict[str, Any]] = {}

    for file_path in sorted(target_dir.iterdir()):
        if not file_path.is_file():
            continue
        # Exclude release_checksums.json itself from hashing to avoid circularity
        if file_path.name == output_manifest_path.name:
            continue

        sha256 = compute_sha256(file_path)
        size_bytes = file_path.stat().st_size
        manifest[file_path.name] = {
            "sha256": sha256,
            "size_bytes": size_bytes,
        }

    output_manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate cryptographic release checksum manifest.")
    parser.add_argument(
        "--target-dir",
        type=str,
        default="benchmarks/release_results/v2_verified",
        help="Directory containing release artifacts.",
    )
    parser.add_argument(
        "--output-file",
        type=str,
        default=None,
        help="Path to output release_checksums.json.",
    )
    args = parser.parse_args()

    target_dir = Path(args.target_dir)
    out_file = Path(args.output_file) if args.output_file else None

    manifest = generate_checksum_manifest(target_dir, out_file)
    print(f"Generated checksums for {len(manifest)} release artifacts in {target_dir}:")
    for name, info in manifest.items():
        print(f"  - {name}: {info['sha256']} ({info['size_bytes']} bytes)")


if __name__ == "__main__":
    main()

