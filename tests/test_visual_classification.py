"""Unit and Mock Integration Tests for Axis 1: Visual Movability Classification.

Verifies:
1. compute_classification_metrics correctly calculates balanced accuracy, macro-F1, per-class metrics, coverage, and confusion matrix.
2. evaluate_visual_classification runs end-to-end against mock servers without ground-truth leakage into prompts or engine.
3. Prediction outputs are saved as valid JSONL and summary JSON with proper denominators.
4. skip_offline_models flag behaves properly when a server is offline.
"""

from __future__ import annotations

import json
from pathlib import Path
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
import pytest

from vlm.scripts.evaluate_visual_classification import (
    compute_classification_metrics,
    evaluate_visual_classification,
    build_canonical_prompt,
)
from vlm.src.multimodal_inference_engine import MultimodalInferenceEngine

repo_root = Path(__file__).resolve().parent.parent.parent


def test_build_canonical_prompt():
    prompt = build_canonical_prompt("inst_chair_001", "chair")
    assert "inst_chair_001" in prompt
    assert "chair" in prompt
    assert "IMMOVABLE" in prompt
    assert "MOVABLE" in prompt
    assert "MOBILE" in prompt
    # Ground truth fields MUST NOT be in the prompt template
    assert "contextual_movability_gt" not in prompt
    assert "current_motion_gt" not in prompt


def test_compute_classification_metrics():
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


class MockClassificationServer(BaseHTTPRequestHandler):
    """Mock server serving Qwen mock responses."""

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

        prompt_text = ""
        msgs = data.get("messages", [])
        if msgs:
            content = msgs[0].get("content")
            if isinstance(content, list):
                for item in content:
                    if isinstance(item, dict) and item.get("type") == "text":
                        prompt_text = item.get("text", "")
            elif isinstance(content, str):
                prompt_text = content

        obj_id = "inst_door_001"
        obj_class = "door"
        if "Target Instance ID: " in prompt_text:
            obj_id = prompt_text.split("Target Instance ID: ")[1].split("\n")[0].strip()
        if "Expected Semantic Class: " in prompt_text:
            obj_class = prompt_text.split("Expected Semantic Class: ")[1].split("\n")[0].strip()
        elif "Target Semantic Class: " in prompt_text:
            obj_class = prompt_text.split("Target Semantic Class: ")[1].split("\n")[0].strip()
        elif "preflight_test_001" in prompt_text:
            obj_id = "preflight_test_001"
            obj_class = "chair"

        payload = {
            "object_id": obj_id,
            "object_class": obj_class,
            "movability": "IMMOVABLE",
            "physical_support": "mounted_to_structure",
            "transport_context": "stationary_environment",
            "confidence": 0.95,
            "abstain": False,
            "short_explanation": f"Evaluation for {obj_id} ({obj_class}).",
        }

        resp = {
            "choices": [{"message": {"role": "assistant", "content": json.dumps(payload)}}],
            "usage": {"prompt_tokens": 120, "completion_tokens": 45},
        }
        self.wfile.write(json.dumps(resp).encode("utf-8"))

    def log_message(self, format, *args):
        pass


@pytest.fixture(scope="module")
def mock_server():
    server = HTTPServer(("127.0.0.1", 0), MockClassificationServer)
    port = server.server_port
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{port}"
    server.shutdown()


def test_evaluate_visual_classification_mock(mock_server, tmp_path, monkeypatch):
    """Test full evaluation pipeline against mock server with custom endpoint."""
    monkeypatch.setenv("QWEN_SERVER_URL", f"{mock_server}/v1/chat/completions")

    results = evaluate_visual_classification(
        model_keys=["qwen2_5_vl_3b"],
        view_modes=["CROP_ONLY"],
        run_dir=tmp_path / "test_run",
        max_instances=2,
    )

    assert "evaluations" in results
    assert "qwen2_5_vl_3b" in results["evaluations"]
    qwen_eval = results["evaluations"]["qwen2_5_vl_3b"]["CROP_ONLY"]
    assert "metrics" in qwen_eval
    metrics = qwen_eval["metrics"]
    assert metrics["wire_quality"]["requests_attempted"] == 2
    assert metrics["wire_quality"]["successful_responses"] == 2
    assert metrics["wire_quality"]["valid_schema_count"] == 2

    # Check that output jsonl file was written
    pred_file = tmp_path / "test_run" / "visual_classification_predictions_qwen2_5_vl_3b_crop_only_class_conditioned.jsonl"
    assert pred_file.exists()
    lines = [json.loads(line) for line in pred_file.read_text(encoding="utf-8").strip().split("\n")]
    assert len(lines) == 2
    for l in lines:
        assert l["response_received"] is True
        assert l["schema_valid"] is True
        assert "input_image_hashes" in l
        assert l["cache_hit"] is False
        assert "server_endpoint" in l


def test_skip_offline_models(tmp_path, monkeypatch):
    """Test that skip_offline_models=True skips offline endpoints without crashing."""
    monkeypatch.setenv("QWEN_SERVER_URL", "http://127.0.0.1:59999/v1/chat/completions")
    results = evaluate_visual_classification(
        model_keys=["qwen2_5_vl_3b"],
        view_modes=["CROP_ONLY"],
        run_dir=tmp_path / "offline_run",
        max_instances=1,
        skip_offline_models=True,
    )
    assert "qwen2_5_vl_3b" in results["evaluations"]
    assert results["evaluations"]["qwen2_5_vl_3b"].get("status") == "SKIPPED_OFFLINE"
