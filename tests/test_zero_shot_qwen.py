"""Tests for Zero-Shot Qwen Live Inference Script."""

from __future__ import annotations

import json
import os
from pathlib import Path
import pytest

from scripts.evaluate_zero_shot_qwen import (
    build_zero_shot_prompt,
    run_zero_shot_evaluation,
)
from estimator_aware_vlm.selective_router import (
    check_vlm_server_health,
    DEFAULT_SERVER_URL,
)


def test_build_zero_shot_prompt():
    """Verify zero-shot prompt contains all three classes and JSON formatting requirements."""
    prompt = build_zero_shot_prompt("inst_chair_001", "chair")
    assert "inst_chair_001" in prompt
    assert "chair" in prompt
    assert "IMMOVABLE" in prompt
    assert "MOVABLE" in prompt
    assert "MOBILE" in prompt
    assert "STRICT JSON ONLY" in prompt


def test_zero_shot_live_optional_single_instance(tmp_path):
    """If live server is online, run a single-instance live zero-shot evaluation and verify output metrics."""
    server_url = os.environ.get("VLM_SERVER_URL", DEFAULT_SERVER_URL)
    health = check_vlm_server_health(server_url, timeout=1.0)
    if health.get("status") != "ONLINE":
        pytest.skip(f"Live VLM server offline at {server_url}")

    out_file = tmp_path / "zero_shot_test_out.json"
    summary = run_zero_shot_evaluation(
        server_url=server_url,
        max_instances=1,
        out_path=out_file,
    )

    assert summary["live_execution"] is True
    assert summary["total_instances_evaluated"] == 1
    assert "latency_stats_ms" in summary
    assert summary["latency_stats_ms"]["mean"] > 0.0
    rec = summary["records"][0]
    assert rec["prediction"] in ["IMMOVABLE", "MOVABLE", "MOBILE", "UNKNOWN"]
    assert rec["latency_ms"] > 0.0
    assert out_file.exists()
