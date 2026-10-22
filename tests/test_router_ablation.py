"""Unit tests for Stage 6 Router Ablation Matrix and Baseline Comparisons."""

import pytest
import sys
from pathlib import Path

# Add project root and vlm to sys.path
root_dir = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(root_dir))
sys.path.insert(0, str(root_dir / "vlm"))

from vlm.src.router_policy import evaluate_router_policy
from vlm.src.selective_router import SelectiveRouter
from vlm.scripts.evaluate_selective_router import compute_metrics


def test_router_reduces_vlm_query_budget():
    """Verify that selective router queries < 35% of objects, avoiding > 65% of VLM queries."""
    sample_entities = [
        {"entity_id": "door_001", "semantic_class": "door", "observed_motion": "STATIC", "has_images": True, "views_count": 1, "ground_truth": "IMMOVABLE"},
        {"entity_id": "fan_001", "semantic_class": "ceiling fan", "observed_motion": "STATIC", "has_images": True, "views_count": 1, "ground_truth": "IMMOVABLE"},
        {"entity_id": "desk_001", "semantic_class": "desk", "observed_motion": "STATIC", "has_images": True, "views_count": 3, "ground_truth": "MOVABLE"},
        {"entity_id": "desk_018", "semantic_class": "desk", "observed_motion": "STATIC", "has_images": True, "views_count": 1, "ground_truth": "MOVABLE"},
        {"entity_id": "laptop_001", "semantic_class": "laptop", "observed_motion": "STATIC", "has_images": True, "views_count": 2, "ground_truth": "MOVABLE"},
        {"entity_id": "backpack_001", "semantic_class": "backpack", "observed_motion": "STATIC", "has_images": True, "views_count": 2, "ground_truth": "MOVABLE"},
        {"entity_id": "monitor_001", "semantic_class": "computer monitor", "observed_motion": "STATIC", "has_images": True, "views_count": 2, "ground_truth": "MOVABLE"},
        {"entity_id": "person_001", "semantic_class": "person", "observed_motion": "DYNAMIC", "has_images": False, "views_count": 0, "ground_truth": "MOBILE"},
        {"entity_id": "robot_001", "semantic_class": "robot", "observed_motion": "DYNAMIC", "has_images": False, "views_count": 0, "ground_truth": "MOBILE"},
        # Only these 2 require escalation
        {"entity_id": "chair_002", "semantic_class": "chair", "observed_motion": "DISPLACED", "world_displacement_m": 0.56, "has_images": True, "views_count": 1, "ground_truth": "MOVABLE"},
        {"entity_id": "fan_002", "semantic_class": "pedestal fan", "observed_motion": "STATIC", "has_images": True, "views_count": 1, "ground_truth": "MOVABLE"},
    ]

    router = SelectiveRouter(enable_vlm=False)
    results = []
    for s in sample_entities:
        res = router.route_entity(s)
        res["ground_truth"] = s["ground_truth"]
        results.append(res)

    metrics = compute_metrics(results)
    eff = metrics["efficiency"]

    # Exactly 2 out of 11 should be requested by policy
    assert eff["vlm_queries_requested"] == 2
    # Since enable_vlm=False, executed inferences are strictly 0
    assert eff["vlm_queries_invoked"] == 0
    assert eff["vlm_queries_avoided"] == 11
    assert eff["vlm_query_rate_pct"] == 0.0
    # Average decision latency must be well under 1 ms
    assert eff["decision_latency_ms"]["mean"] < 1.0


def test_ablation_oracle_and_random_query_shapes():
    """Verify metrics calculation correctly separates quality, efficiency, and reliability."""
    mock_records = [
        {"entity_id": "e1", "decision": "RESOLVE_CHEAP", "prediction": "IMMOVABLE", "ground_truth": "IMMOVABLE", "abstained": False, "vlm_required": False, "views_used": 0, "evidence_valid": True, "decision_latency_ms": 0.04, "inference_latency_ms": 0.0},
        {"entity_id": "e2", "decision": "RESOLVE_CHEAP", "prediction": "MOVABLE", "ground_truth": "MOVABLE", "abstained": False, "vlm_required": False, "views_used": 0, "evidence_valid": True, "decision_latency_ms": 0.03, "inference_latency_ms": 0.0},
        {"entity_id": "e3", "decision": "ESCALATE", "prediction": "MOVABLE", "ground_truth": "MOVABLE", "abstained": False, "vlm_required": True, "vlm_inference_executed": True, "views_used": 2, "evidence_valid": True, "decision_latency_ms": 0.05, "inference_latency_ms": 12000.0},
        {"entity_id": "e4", "decision": "RESOLVE_CHEAP", "prediction": "MOBILE", "ground_truth": "MOBILE", "abstained": False, "vlm_required": False, "views_used": 0, "evidence_valid": True, "decision_latency_ms": 0.04, "inference_latency_ms": 0.0},
    ]

    metrics = compute_metrics(mock_records)
    assert metrics["quality"]["balanced_accuracy"] == 1.0
    assert metrics["quality"]["macro_f1"] == 1.0
    assert metrics["efficiency"]["vlm_queries_requested"] == 1
    assert metrics["efficiency"]["vlm_queries_invoked"] == 1
    assert metrics["efficiency"]["vlm_query_rate_pct"] == 25.0
    assert metrics["reliability"]["false_cheap_resolutions"] == 0
    assert metrics["reliability"]["invalid_evidence_escalations"] == 0
