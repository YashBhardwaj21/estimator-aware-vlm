"""Unit and Mock Integration Tests for Unified Multimodal Inference Engine."""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
import pytest
from PIL import Image

from estimator_aware_vlm.multimodal_inference_engine import (
    MultimodalInferenceEngine,
    parse_multimodal_prediction,
    encode_image_base64,
)


class MockQwenServer(BaseHTTPRequestHandler):
    """Local server simulating OpenAI-compatible (Qwen) endpoint."""

    def do_GET(self):
        if self.path in ["/health"]:
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(b'{"status": "ok"}')
        else:
            self.send_response(404)
            self.end_headers()

    def do_POST(self):
        content_len = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(content_len)
        data = json.loads(body.decode("utf-8"))

        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()

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

        resp = {
            "choices": [{"message": {"role": "assistant", "content": json.dumps(mock_payload)}}],
            "usage": {"prompt_tokens": 120, "completion_tokens": 45},
        }
        self.wfile.write(json.dumps(resp).encode("utf-8"))


@pytest.fixture(scope="module")
def mock_server():
    server = HTTPServer(("127.0.0.1", 0), MockQwenServer)
    port = server.server_port
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{port}"
    server.shutdown()


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


def test_qwen_wire_execution_mock(mock_server, tmp_path):
    """Verify Qwen wire inference format and telemetry recording."""
    dummy_crop = tmp_path / "crop.jpg"
    dummy_scene = tmp_path / "scene.jpg"
    im = Image.new("RGB", (64, 64), color="blue")
    im.save(dummy_crop)
    im.save(dummy_scene)

    client = MultimodalInferenceEngine(
        model_key="qwen2_5_vl_3b",
        server_url=f"{mock_server}/v1/chat/completions",
    )

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
    assert record.cache_hit is False
    assert record.parsed_prediction == "MOVABLE"
    assert record.client_latency_ms > 0.0
    assert record.token_usage["prompt_tokens"] == 120


def test_offline_server_error_handling(tmp_path):
    """Verify offline server records failure cleanly without falling back to cache."""
    dummy_crop = tmp_path / "crop.jpg"
    im = Image.new("RGB", (32, 32), color="red")
    im.save(dummy_crop)

    client = MultimodalInferenceEngine(
        model_key="qwen2_5_vl_3b",
        server_url="http://127.0.0.1:59999/v1/chat/completions",
    )

    record = client.execute_live_query(
        physical_instance_id="inst_chair_002",
        semantic_class="chair",
        prompt="Classify this object",
        crop_image_path=dummy_crop,
    )

    assert record.request_attempted is True
    assert record.http_success is False
    assert record.response_received is False
    assert record.prediction_accepted is False
    assert record.cache_hit is False
    assert record.parsed_prediction == "UNKNOWN"
    assert "http_transport_error" in record.validation_error


def test_live_qwen_smoke_test_optional():
    """If live Qwen server is actively running, verify preflight smoke test passes."""
    engine = MultimodalInferenceEngine(model_key="qwen2_5_vl_3b")
    health = engine.check_health()
    if health.get("status") != "ONLINE":
        pytest.skip(f"Live Qwen server offline at {engine.server_url}")

    smoke_res = engine.run_preflight_smoke_test()
    assert smoke_res["status"] == "PASSED"
    assert smoke_res["latency_ms"] > 0.0

