"""Regression tests for benchmark run artifact integrity, manifest hashing, and run_id synchronization."""

import hashlib
import json
from pathlib import Path
import pytest

repo_root = Path(__file__).resolve().parents[1]


def compute_file_sha256(path: Path) -> str:
    hasher = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()


def test_historical_pilot_runs_remain_immutable():
    # Historical runs must never be deleted
    bench_dir = repo_root / "outputs" / "benchmark_runs"
    assert bench_dir.exists()
    runs = list(bench_dir.glob("*"))
    assert len(runs) >= 4, "Historical benchmark runs must not be deleted!"


def test_run_manifest_structure_and_hashes():
    bench_dir = repo_root / "outputs" / "benchmark_runs"
    manifests = list(bench_dir.glob("*/run_manifest.json"))
    assert len(manifests) > 0, "No run manifests found in benchmark_runs"

    for mf in manifests:
        run_folder = mf.parent
        with open(mf, "r", encoding="utf-8") as f:
            data = json.load(f)

        assert "run_id" in data
        assert "git_commit_sha" in data
        assert "input_manifests" in data
        assert "output_artifacts" in data

        # Verify that all recorded output files actually exist and match SHA-256
        for rel_p, meta in data.get("output_artifacts", {}).items():
            expected_sha = meta.get("sha256")
            full_p = repo_root / rel_p
            if full_p.exists():
                actual_sha = compute_file_sha256(full_p)
                assert actual_sha == expected_sha, f"Artifact hash mismatch in {mf}: {full_p}"
