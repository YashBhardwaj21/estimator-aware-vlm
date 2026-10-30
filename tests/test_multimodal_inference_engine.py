"""Unit and Integration Tests for Unified Multimodal Inference Engine and Visual Classification.

Verifies:
1. Canonical prompt generation for zero-shot and class-conditioned queries.
2. Multimodal prediction output parsing, schema validation, and defensive fallback on malformed responses.
3. Wire query execution and telemetry recording via mocked transport.
4. Offline server error handling, fail-closed contracts, and graceful skipping when offline.
5. Multimodal visual classification evaluation metrics (balanced accuracy, macro-F1, confusion matrix).
6. Cache separation: strict isolation between cached historical records and live inference.
7. Optional live server end-to-end integration (skipped when live server is offline).
"""

from __future__ import annotations

import io
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional
import urllib.error
import urllib.request
from unittest.mock import MagicMock, patch
import pytest
from PIL import Image

from estimator_aware_vlm.common.paths import get_repo_root
from estimator_aware_vlm.multimodal_inference_engine import (
    MultimodalInferenceEngine,
    encode_image_base64,
    parse_multimodal_prediction,
)
from estimator_aware_vlm.selective_router import (
    SelectiveRouter,
    check_vlm_server_health,
)
from scripts.evaluate_visual_classification import (
    build_canonical_prompt,
    compute_classification_metrics,
    evaluate_visual_classification,
)
from scripts.evaluate_zero_shot_qwen import build_zero_shot_prompt

repo_root = get_repo_root()


# ---------------------------------------------------------------------------
# 1. Prompt Construction Tests
# ---------------------------------------------------------------------------

def test_build_canonical_prompt():
    """Verify visual classification prompt template contains required taxonomy and no GT leakage."""
    prompt = build_canonical_prompt("inst_chair_001", "chair")
    assert "inst_chair_001" in prompt
    assert "chair" in prompt
    assert "IMMOVABLE" in prompt
    assert "MOVABLE" in prompt
    assert "MOBILE" in prompt
    assert "contextual_movability_gt" not in prompt
    assert "current_motion_gt" not in prompt


def test_build_zero_shot_prompt():
    """Verify zero-shot prompt template contains required instructions and definitions."""
    prompt = build_zero_shot_prompt("inst_chair_001", "chair")
    assert "inst_chair_001" in prompt
    assert "chair" in prompt
    assert "IMMOVABLE" in prompt
    assert "MOVABLE" in prompt
    assert "MOBILE" in prompt
    assert "carried" in prompt.lower()


# ---------------------------------------------------------------------------
# 2. Output Parsing & Schema Validation Tests
# ---------------------------------------------------------------------------

def test_parse_multimodal_prediction_valid():
    """Verify clean JSON parsing and schema validation."""
    valid_text = json.dumps({
        "object_id": "inst_desk_001",
        "object_class": "desk",
        "movability": "MOVABLE",
        "confidence": 0.92,
        "physical_support": "freestanding_on_surface",
        "short_explanation": "Movable office desk.",
        "abstain": False,
    })
    parsed, parse_valid, schema_valid = parse_multimodal_prediction(valid_text, "inst_desk_001", "desk")
    assert parse_valid is True
    assert schema_valid is True
    assert parsed["movability"] == "MOVABLE"
    assert parsed["confidence"] == 0.92
    assert parsed["abstain"] is False


def test_parse_multimodal_prediction_malformed():
    """Verify malformed non-JSON text produces safe fallback without crashing."""
    bad_text = "I cannot determine movability because { unclosed json"
    parsed, parse_valid, schema_valid = parse_multimodal_prediction(bad_text, "inst_desk_001", "desk")
    assert parse_valid is False
    assert schema_valid is False
    assert parsed["movability"] == "UNKNOWN"
    assert parsed["abstain"] is True
    assert "json_parse_failed" in parsed["validation_error"]


# ---------------------------------------------------------------------------
# 3. Mock Wire Execution & Telemetry Tests
# ---------------------------------------------------------------------------

def test_vlm_server_health_check_offline():
    """Verify health probe returns OFFLINE when server cannot be reached."""
    res = check_vlm_server_health("http://127.0.0.1:59999/v1/chat/completions", timeout=0.1)
    assert res["status"] == "OFFLINE"


def test_vlm_server_health_check_online_mock():
    """Verify health probe returns ONLINE when endpoint responds 200 OK."""
    mock_response = MagicMock()
    mock_response.getcode.return_value = 200
    mock_response.read.return_value = b'{"status": "ok"}'
    mock_response.__enter__.return_value = mock_response

    with patch("urllib.request.urlopen", return_value=mock_response):
        res = check_vlm_server_health("http://mock-server:8089/v1/chat/completions", timeout=1.0)
        assert res["status"] == "ONLINE"


def test_qwen_wire_execution_mock(tmp_path: Path):
    """Verify Qwen wire inference format and telemetry recording with mocked transport."""
    dummy_crop = tmp_path / "crop.jpg"
    dummy_scene = tmp_path / "scene.jpg"
    im = Image.new("RGB", (64, 64), color="blue")
    im.save(dummy_crop)
    im.save(dummy_scene)

    client = MultimodalInferenceEngine(
        model_key="qwen2_5_vl_3b",
        server_url="http://mock-server:8089/v1/chat/completions",
    )

    mock_payload = {
        "object_id": "inst_chair_002",
        "object_class": "chair",
        "movability": "MOVABLE",
        "confidence": 0.95,
        "physical_support": "freestanding_on_surface",
        "transport_context": "stationary_environment",
        "short_explanation": "Office chair on casters.",
        "abstain": False,
    }
    server_resp = {
        "choices": [{"message": {"role": "assistant", "content": json.dumps(mock_payload)}}],
        "usage": {"prompt_tokens": 120, "completion_tokens": 45},
    }

    mock_resp = MagicMock()
    mock_resp.getcode.return_value = 200
    mock_resp.read.return_value = json.dumps(server_resp).encode("utf-8")
    mock_resp.__enter__.return_value = mock_resp

    with patch("urllib.request.urlopen", return_value=mock_resp):
        record = client.execute_live_query(
            physical_instance_id="inst_chair_002",
            semantic_class="chair",
            prompt="Classify this object",
            crop_image_path=dummy_crop,
            scene_image_path=dummy_scene,
            view_mode="CROP_PLUS_CONTEXT",
            image_hashes=["hash1", "hash2"],
        )

    assert record.request_attempted is True
    assert record.http_success is True
    assert record.response_received is True
    assert record.parse_valid is True
    assert record.schema_valid is True
    assert record.prediction_accepted is True
    assert record.parsed_prediction == "MOVABLE"
    assert record.confidence == 0.95
    assert record.token_usage == {"prompt_tokens": 120, "completion_tokens": 45}


def test_offline_server_error_handling(tmp_path: Path):
    """Verify that an unreachable server raises ConnectionError when executing live queries."""
    client = MultimodalInferenceEngine(
        model_key="qwen2_5_vl_3b",
        server_url="http://127.0.0.1:59999/v1/chat/completions",
        timeout=0.1,
    )
    dummy_crop = tmp_path / "crop.jpg"
    Image.new("RGB", (64, 64), color="red").save(dummy_crop)

    record = client.execute_live_query(
        physical_instance_id="inst_001",
        semantic_class="chair",
        prompt="Test",
        crop_image_path=dummy_crop,
    )
    assert record.http_success is False
    assert record.response_received is False
    assert record.parsed_prediction == "UNKNOWN"
    assert record.prediction_accepted is False
    assert "http_transport_error" in (record.validation_error or "")


# ---------------------------------------------------------------------------
# 4. Classification Metrics Computation Tests
# ---------------------------------------------------------------------------

def test_compute_classification_metrics():
    """Verify accuracy, balanced accuracy, macro-F1, and confusion matrix calculation."""
    gt_map = {
        "inst_1": {"contextual_movability_gt": "IMMOVABLE"},
        "inst_2": {"contextual_movability_gt": "MOVABLE"},
        "inst_3": {"contextual_movability_gt": "MOBILE"},
    }

    # Case A: Perfect predictions
    preds_perfect = [
        {"physical_instance_id": "inst_1", "parsed_prediction": "IMMOVABLE", "response_received": True, "schema_valid": True, "prediction_accepted": True, "client_latency_ms": 100.0},
        {"physical_instance_id": "inst_2", "parsed_prediction": "MOVABLE", "response_received": True, "schema_valid": True, "prediction_accepted": True, "client_latency_ms": 120.0},
        {"physical_instance_id": "inst_3", "parsed_prediction": "MOBILE", "response_received": True, "schema_valid": True, "prediction_accepted": True, "client_latency_ms": 110.0},
    ]

    m_perf = compute_classification_metrics(preds_perfect, gt_map)
    assert m_perf["overall_accuracy"] == 1.0
    assert m_perf["balanced_accuracy"] == 1.0
    assert m_perf["macro_f1"] == 1.0
    assert m_perf["wire_quality"]["coverage_pct"] == 100.0
    assert m_perf["wire_quality"]["selective_accuracy_on_accepted"] == 1.0
    assert m_perf["client_latency_ms"]["mean"] == 110.0

    # Case B: Mixed with UNKNOWN / abstain
    preds_mixed = [
        {"physical_instance_id": "inst_1", "parsed_prediction": "IMMOVABLE", "response_received": True, "schema_valid": True, "prediction_accepted": True, "client_latency_ms": 100.0},
        {"physical_instance_id": "inst_2", "parsed_prediction": "UNKNOWN", "response_received": True, "schema_valid": True, "prediction_accepted": False, "client_latency_ms": 120.0},
        {"physical_instance_id": "inst_3", "parsed_prediction": "MOVABLE", "response_received": True, "schema_valid": True, "prediction_accepted": True, "client_latency_ms": 110.0},
    ]
    m_mixed = compute_classification_metrics(preds_mixed, gt_map)
    assert m_mixed["overall_accuracy"] == pytest.approx(1 / 3, 0.01)
    assert m_mixed["confusion_matrix"]["MOVABLE"]["UNKNOWN"] == 1
    assert m_mixed["confusion_matrix"]["MOBILE"]["MOVABLE"] == 1
    assert m_mixed["wire_quality"]["accepted_predictions"] == 2


def test_skip_offline_models(tmp_path: Path):
    """Verify skip_offline_models gracefully skips offline models without raising ConnectionError."""
    res = evaluate_visual_classification(
        model_keys=["qwen2_5_vl_3b"],
        view_modes=["CROP_ONLY"],
        run_dir=tmp_path / "offline_run",
        max_instances=2,
        skip_offline_models=True,
    )
    assert "qwen2_5_vl_3b" in res["evaluations"]
    assert res["evaluations"]["qwen2_5_vl_3b"]["status"] == "SKIPPED_OFFLINE"


# ---------------------------------------------------------------------------
# 5. Cache Separation & Live Integration Tests
# ---------------------------------------------------------------------------

def test_cached_historical_vs_live_mode_separation(tmp_path: Path):
    """Verify that cached historical mode never calls the wire and live mode never reads the cache."""
    sample_crop_dir = tmp_path / "sample_crops"
    sample_crop_dir.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (64, 64), color="red").save(sample_crop_dir / "crop_view_0.jpg")
    Image.new("RGB", (64, 64), color="blue").save(sample_crop_dir / "scene_view_0.jpg")

    cache_file = tmp_path / "test_cache.json"
    temp_router = SelectiveRouter(cache_path=cache_file, mode="CACHED_HISTORICAL", configured_views=1)
    _, img_tel = temp_router.validate_image_inputs(sample_crop_dir)
    ckey = temp_router.compute_cache_key("inst_chair_002", "chair", img_tel)

    cache_data = {
        ckey: {
            "movability": "MOVABLE",
            "confidence": 0.88,
            "physical_support": "freestanding_on_surface",
            "transport_context": "stationary_environment",
            "abstain": False,
            "latency_s": 1.5,
        }
    }
    with open(cache_file, "w", encoding="utf-8") as f:
        json.dump(cache_data, f)

    # 1. Historical Cached Mode: Must hit cache and never attempt socket connection
    router_cached = SelectiveRouter(
        cache_path=cache_file,
        server_url="http://127.0.0.1:59999",
        mode="CACHED_HISTORICAL",
        configured_views=1,
    )

    ev = {
        "entity_id": "inst_chair_002",
        "semantic_class": "chair",
        "has_valid_pose": True,
        "depth_valid": True,
        "observed_motion": "DISPLACED",
        "world_displacement_m": 0.563,
        "displacement_uncertainty_m": 0.04,
        "has_images": True,
        "views_count": 1,
        "crops_dir": str(sample_crop_dir),
        "s1_obs_count": 15,
        "s2_obs_count": 12,
        "geometric_consistency_score": 0.95,
    }

    res_cached = router_cached.route_entity(ev)
    assert res_cached["execution_mode"] == "CACHED_HISTORICAL"
    assert res_cached["vlm_cache_hit"] is True
    assert res_cached["vlm_inference_executed"] is False
    assert res_cached["prediction"] == "MOVABLE"

    # 2. Live Inference Mode: Must bypass cache file
    router_live = SelectiveRouter(
        cache_path=cache_file,
        server_url="http://127.0.0.1:59999",
        mode="LIVE_INFERENCE",
        configured_views=1,
    )
    res_live = router_live.route_entity(ev)
    assert res_live["execution_mode"] == "LIVE_INFERENCE"
    assert res_live["vlm_cache_hit"] is False


def test_live_qwen_smoke_test_optional(tmp_path: Path):
    """Optional smoke test against live Qwen server; gracefully skips if server is offline or unreachable."""
    client = MultimodalInferenceEngine(model_key="qwen2_5_vl_3b")
    try:
        health = client.check_health()
    except OSError:
        pytest.skip(f"Live Qwen server at {client.server_url} is unreachable.")

    if health.get("status") != "ONLINE":
        pytest.skip(f"Live Qwen server offline at {client.server_url}")

    sample_crop = repo_root / "benchmarks" / "assets" / "multiview_samples" / "inst_chair_002" / "crop_view_0.jpg"
    if not sample_crop.exists():
        pytest.skip("Benchmark crop asset not found on disk")

    rec = client.execute_live_query(
        physical_instance_id="inst_chair_002",
        semantic_class="chair",
        prompt=build_canonical_prompt("inst_chair_002", "chair"),
        crop_image_path=sample_crop,
        view_mode="CROP_ONLY",
    )
    assert rec.http_success is True
    assert rec.response_received is True
    assert rec.parsed_prediction in ["IMMOVABLE", "MOVABLE", "MOBILE", "UNKNOWN"]
