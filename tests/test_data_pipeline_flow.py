"""Unit and smoke tests for recording, synchronization, and data pipeline flow."""

import json
from pathlib import Path
import pytest

from estimator_aware_vlm.common.paths import (
    get_repo_root,
    get_raw_data_dir,
    get_processed_data_dir,
    get_mapping_outputs_dir,
    resolve_path,
)
from scene_graph.data.synchronization import associate
from scripts.check_system_dependencies import run_preflight


def test_canonical_paths_structure():
    """Verify canonical directory structure helpers return valid paths."""
    root = get_repo_root()
    assert root.exists()
    assert (root / "pyproject.toml").exists()

    raw_dir = get_raw_data_dir()
    assert raw_dir == root / "data" / "raw"

    proc_dir = get_processed_data_dir()
    assert proc_dir == root / "data" / "processed"

    map_out = get_mapping_outputs_dir()
    assert map_out == root / "outputs" / "mapping"

    # Relative path resolution
    resolved = resolve_path("configs/default.yaml")
    assert resolved == (root / "configs" / "default.yaml").resolve()


def test_monotonic_rgbd_association():
    """Verify monotonic timestamp association correctly pairs timestamps within delta threshold."""
    # 30 fps stream: ~33.3ms intervals
    rgb_ts = [100.000, 100.033, 100.067, 100.100, 100.133]
    # Depth stream slightly shifted by 2ms
    depth_ts = [100.002, 100.035, 100.069, 100.101, 100.136]

    matches = associate(rgb_ts, depth_ts, max_dt=0.015)
    assert len(matches) == 5
    for i_rgb, i_depth in matches:
        assert i_rgb == i_depth
        assert abs(rgb_ts[i_rgb] - depth_ts[i_depth]) < 0.015


def test_monotonic_association_skips_dropped_frames():
    """Verify dropped frames in one stream do not corrupt subsequent monotonic matching."""
    rgb_ts = [10.0, 10.1, 10.2, 10.3, 10.4]
    # Depth dropped frame at 10.2
    depth_ts = [10.0, 10.1, 10.3, 10.4]

    matches = associate(rgb_ts, depth_ts, max_dt=0.03)
    assert len(matches) == 4
    # Check that rgb[2] (10.2) was not paired
    matched_rgb_indices = [m[0] for m in matches]
    assert 2 not in matched_rgb_indices


def test_dataset_association_format_fixture(tmp_path: Path):
    """Test generating and parsing standard associations.txt format."""
    seq_dir = tmp_path / "mock_sequence"
    seq_dir.mkdir()

    rgb_dir = seq_dir / "rgb"
    depth_dir = seq_dir / "depth"
    rgb_dir.mkdir()
    depth_dir.mkdir()

    # Create dummy frame files
    (rgb_dir / "100.000000.png").write_bytes(b"RGB")
    (depth_dir / "100.002000.png").write_bytes(b"DEPTH")

    # Write associations file
    assoc_file = seq_dir / "associations.txt"
    assoc_content = (
        "# timestamp_rgb rgb_path timestamp_depth depth_path\n"
        "100.000000 rgb/100.000000.png 100.002000 depth/100.002000.png\n"
    )
    assoc_file.write_text(assoc_content, encoding="utf-8")

    # Parse associations
    lines = [
        l.strip().split()
        for l in assoc_file.read_text(encoding="utf-8").splitlines()
        if l.strip() and not l.startswith("#")
    ]
    assert len(lines) == 1
    t_r, p_r, t_d, p_d = lines[0]
    assert float(t_r) == 100.0
    assert float(t_d) == 100.002
    assert (seq_dir / p_r).exists()
    assert (seq_dir / p_d).exists()


def test_preflight_tool_execution():
    """Verify check_system_dependencies returns valid structured diagnostic data."""
    report = run_preflight()
    assert "system" in report
    assert "python" in report
    assert report["python"]["status"] in ("PASS", "FAIL")
    assert "core_dependencies" in report
    assert "directories" in report
    assert "overall_status" in report
