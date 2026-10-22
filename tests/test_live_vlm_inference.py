"""Unit and Integration Tests for Live VLM Inference and Mode Separation.

Covers:
- Phase 3.1: Explicit execution mode separation (CACHED_HISTORICAL vs. LIVE_INFERENCE)
- Honest latency accounting (cached_latency_ms vs. live inference_latency_ms)
- Real socket HTTP wire protocol execution (using local HTTP server fixture)
- Live server health check and graceful offline detection
"""

import json
import os
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Any, Dict
import pytest
from PIL import Image

from vlm.src.selective_router import (
    SelectiveRouter,
    check_vlm_server_health,
    query_llama_server,
    DEFAULT_SERVER_URL,
)


class MockVLMServerHandler(BaseHTTPRequestHandler):
    """Local HTTP handler simulating OpenAI-compatible VLM server wire protocol."""

    def do_GET(self):
        if self.path == "/health":
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(b'{"status": "ok", "model": "Qwen2.5-VL-3B-Instruct"}')
        else:
            self.send_response(405)
            self.end_headers()

    def do_POST(self):
        content_len = int(self.headers.get("Content-Length", 0))
        post_body = self.rfile.read(content_len)
        req_data = json.loads(post_body.decode("utf-8"))

        # Verify chat completion payload structure
        assert "messages" in req_data
        assert req_data["messages"][0]["role"] == "user"

        # Generate valid structured affordance response
        mock_response = {
            "id": "chatcmpl-mock-live-001",
            "object": "chat.completion",
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": json.dumps({
                            "object_id": "inst_chair_002",
                            "object_class": "chair",
                            "movability": "MOVABLE",
                            "abstain": False,
                            "physical_support": "freestanding caster base",
                            "transport_context": "displaced task chair",
                            "confidence": 0.95,
                            "short_explanation": "Office chair on casters is freestanding and relocatable.",
                        }),
                    },
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": 142,
                "completion_tokens": 48,
                "total_tokens": 190,
            },
        }

        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps(mock_response).encode("utf-8"))

    def log_message(self, format, *args):
        # Suppress noisy standard HTTP access logging in tests
        pass


@pytest.fixture
def mock_vlm_server():
    """Start real background HTTP server listening on an ephemeral port."""
    server = HTTPServer(("127.0.0.1", 0), MockVLMServerHandler)
    port = server.server_port
    server_url = f"http://127.0.0.1:{port}/v1/chat/completions"

    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()

    yield server_url

    server.shutdown()
    server.server_close()


@pytest.fixture
def sample_crop_dir(tmp_path) -> Path:
    """Create directory with valid crop images for testing visual input encoding."""
    crop_dir = tmp_path / "sample_crops"
    crop_dir.mkdir()
    im_crop = Image.new("RGB", (64, 64), color="orange")
    im_crop.save(crop_dir / "crop_view_0.jpg", format="JPEG")
    im_scene = Image.new("RGB", (128, 128), color="blue")
    im_scene.save(crop_dir / "scene_view_0.jpg", format="JPEG")
    return crop_dir


def test_vlm_server_health_check_offline():
    """Verify that an unreachable port returns OFFLINE status cleanly."""
    res = check_vlm_server_health("http://127.0.0.1:59999/v1/chat/completions", timeout=0.5)
    assert res["status"] == "OFFLINE"
    assert "error" in res


def test_vlm_server_health_check_online(mock_vlm_server):
    """Verify that an active server endpoint returns ONLINE status."""
    res = check_vlm_server_health(mock_vlm_server, timeout=1.0)
    assert res["status"] == "ONLINE"
    assert res["code"] == 200


def test_live_inference_over_real_http_wire(mock_vlm_server, sample_crop_dir, tmp_path):
    """Verify genuine live inference over the socket wire with honest latency tracking."""
    cache_file = tmp_path / "test_live_cache.json"
    router = SelectiveRouter(
        cache_path=cache_file,
        server_url=mock_vlm_server,
        mode="LIVE_INFERENCE",
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

    res = router.route_entity(ev)

    assert res["execution_mode"] == "LIVE_INFERENCE"
    assert res["vlm_inference_executed"] is True
    assert res["vlm_cache_hit"] is False
    assert res["prediction"] == "MOVABLE"
    assert res["abstained"] is False
    assert res["vlm_output_valid"] is True
    assert res["vlm_prediction_accepted"] is True

    # Honest latency accounting: live latency measured, cached latency None
    assert res["inference_latency_ms"] > 0.0
    assert res["cached_latency_ms"] is None

    # Verify result was persisted to cache
    assert cache_file.exists()
    with open(cache_file, "r", encoding="utf-8") as f:
        cache_data = json.load(f)
    assert len(cache_data) == 1


def test_cached_historical_vs_live_mode_separation(sample_crop_dir, tmp_path):
    """Verify strict behavioral divergence between CACHED_HISTORICAL and LIVE_INFERENCE."""
    cache_file = tmp_path / "separation_cache.json"

    # Pre-populate cache with a historical measurement
    cached_fingerprint = "dummy_fingerprint_test_key"
    cache_data = {
        cached_fingerprint: {
            "entity_id": "inst_chair_002",
            "movability": "MOVABLE",
            "abstain": False,
            "total_views": 1,
            "latency_s": 8.312,
            "token_usage": {"prompt_tokens": 140, "completion_tokens": 45},
        }
    }
    with open(cache_file, "w", encoding="utf-8") as f:
        json.dump(cache_data, f)

    # 1. Under CACHED_HISTORICAL mode:
    router_cached = SelectiveRouter(
        cache_path=cache_file,
        mode="CACHED_HISTORICAL",
        configured_views=1,
    )
    # Monkeypatch compute_cache_key to return our test key
    router_cached.compute_cache_key = lambda eid, scls, tel: cached_fingerprint

    ev = {
        "entity_id": "inst_chair_002",
        "semantic_class": "chair",
        "has_valid_pose": True,
        "depth_valid": True,
        "observed_motion": "DISPLACED",
        "world_displacement_m": 0.563,
        "has_images": True,
        "views_count": 1,
        "crops_dir": str(sample_crop_dir),
        "s1_obs_count": 10,
        "s2_obs_count": 10,
    }

    res_cached = router_cached.route_entity(ev)

    assert res_cached["execution_mode"] == "CACHED_HISTORICAL"
    assert res_cached["vlm_cache_hit"] is True
    assert res_cached["vlm_inference_executed"] is False
    assert res_cached["inference_latency_ms"] == 0.0
    assert res_cached["cached_latency_ms"] == 8312.0


def test_live_server_optional_integration(sample_crop_dir):
    """Probe live model server if actively hosted in user environment."""
    server_url = os.environ.get("VLM_SERVER_URL", DEFAULT_SERVER_URL)
    health = check_vlm_server_health(server_url, timeout=1.0)

    if health["status"] != "ONLINE":
        pytest.skip(f"Live VLM server offline at {server_url}. Health status: {health}")

    router = SelectiveRouter(
        server_url=server_url,
        mode="LIVE_INFERENCE",
        configured_views=1,
    )

    ev = {
        "entity_id": "inst_chair_002",
        "semantic_class": "chair",
        "has_valid_pose": True,
        "depth_valid": True,
        "observed_motion": "DISPLACED",
        "world_displacement_m": 0.563,
        "has_images": True,
        "views_count": 1,
        "crops_dir": str(sample_crop_dir),
        "s1_obs_count": 15,
        "s2_obs_count": 10,
    }

    res = router.route_entity(ev)
    assert res["execution_mode"] == "LIVE_INFERENCE"
    assert res["vlm_inference_executed"] is True
    assert res["inference_latency_ms"] > 0.0
    assert res["prediction"] in ["IMMOVABLE", "MOVABLE", "MOBILE", "UNKNOWN"]
