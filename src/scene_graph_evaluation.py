"""Clean Prediction vs Evaluation Interface for Dynamic 3D Scene Graphs.

Decoupled Evaluation Contract:
- The prediction pipeline produces pure scene graph predictions with zero ground truth leakage.
- This module consumes the predicted scene graph and independently loads ground-truth evaluation artifacts:
  - controlled_scene_events.json (physical relocation benchmarks)
  - cross_session_identity_annotations.json (canonical track-to-instance mapping)
  - movability_ground_truth.json (dual-annotator adjudicated movability ontology)
- Evaluates landmark action reliability (retention of stationary entities vs invalidation of relocated entities),
  movability categorization, and cross-session identity resolution.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple


from vlm.src.cross_session_association import build_canonical_session_track_mapping


def evaluate_scene_graph(
    scene_graph: Dict[str, Any],
    controlled_events_path: Optional[Path] = None,
    identity_annotations_path: Optional[Path] = None,
    movability_gt_path: Optional[Path] = None,
    policy_name: str = "UNKNOWN",
) -> Dict[str, Any]:
    """Evaluate a purely predicted scene graph against independent ground-truth artifacts.

    Args:
        scene_graph: Prediction scene graph dictionary containing 'nodes'.
        controlled_events_path: Path to controlled_scene_events.json.
        identity_annotations_path: Path to cross_session_identity_annotations.json.
        movability_gt_path: Path to movability_ground_truth.json.
        policy_name: Name of routing policy evaluated (e.g. 'R0', 'R1', 'R2', 'R3', 'R4').

    Returns:
        Structured evaluation dictionary containing landmark reliability,
        relocation detection, and movability metrics.
    """
    nodes: Dict[str, Any] = scene_graph.get("nodes", {})
    total_nodes = len(nodes)

    # 1. Build canonical track-to-instance mapping for evaluation matching
    canonical_map: Dict[Tuple[str, str], str] = {}
    gt_instances: Set[str] = set()

    if identity_annotations_path is not None and Path(identity_annotations_path).exists():
        canonical_map, _ = build_canonical_session_track_mapping(identity_annotations_path)
        gt_instances = set(canonical_map.values())

    # 2. Match predicted nodes to GT instances via source track references
    node_to_gt: Dict[str, str] = {}
    gt_to_nodes: Dict[str, List[str]] = {}

    def _matches_session(s1: str, s2: str) -> bool:
        if s1 == s2:
            return True
        s1_clean = s1.replace("semantic_mapping_my_desk_", "").strip()
        s2_clean = s2.replace("semantic_mapping_my_desk_", "").strip()
        return s1_clean == s2_clean

    for node_id, node in nodes.items():
        matched_gt: Optional[str] = None
        source_tracks = node.get("source_track_ids", [])
        source_sessions = node.get("source_sessions", [])

        # Check tracklet matches
        for s in source_sessions:
            for trk in source_tracks:
                cand = canonical_map.get((s, trk))
                if cand:
                    matched_gt = cand
                    break
                for (map_s, map_trk), inst_id in canonical_map.items():
                    if map_trk == trk and _matches_session(map_s, s):
                        matched_gt = inst_id
                        break
                if matched_gt:
                    break
            if matched_gt:
                break

        # Fallback: check entity_id naming heuristics if not matched
        if matched_gt is None:
            clean_id = node_id.replace("pred_", "").replace("phys_", "").replace("inst_", "")
            for inst_id in gt_instances:
                if clean_id in inst_id or inst_id.replace("inst_", "") in clean_id:
                    matched_gt = inst_id
                    break

        if matched_gt:
            node_to_gt[node_id] = matched_gt
            gt_to_nodes.setdefault(matched_gt, []).append(node_id)

    # 3. Load Controlled Scene Events (Relocations)
    relocated_gt_ids: Set[str] = set()
    controlled_events_list: List[Dict[str, Any]] = []

    if controlled_events_path is not None and controlled_events_path.exists():
        with open(controlled_events_path, "r", encoding="utf-8") as f:
            c_data = json.load(f)
        controlled_events_list = c_data.get("events", [])
        for ev in controlled_events_list:
            if ev.get("slam_invalidated", False):
                tgt = ev.get("target_object_id")
                if tgt:
                    relocated_gt_ids.add(tgt)

    # 4. Evaluate Landmark Action Reliability
    # Stationary entities: GT instances not marked as displaced/relocated
    stationary_gt_ids = set(gt_instances) - relocated_gt_ids

    correctly_retained_stationary: List[str] = []
    falsely_invalidated_stationary: List[str] = []
    unobserved_stationary_gt: List[str] = []

    for gt_id in sorted(stationary_gt_ids):
        matched_nodes = gt_to_nodes.get(gt_id, [])
        if not matched_nodes:
            unobserved_stationary_gt.append(gt_id)
            continue
        # If any mapped node is invalidated, mark as false invalidation
        is_invalid = any(
            bool(nodes[n].get("slam_invalidated", False))
            or nodes[n].get("predicted_landmark_action") == "INVALIDATE"
            for n in matched_nodes
        )
        if is_invalid:
            falsely_invalidated_stationary.append(gt_id)
        else:
            correctly_retained_stationary.append(gt_id)

    correctly_invalidated_relocations: List[str] = []
    missed_relocations: List[str] = []
    unobserved_relocated_gt: List[str] = []

    for gt_id in sorted(relocated_gt_ids):
        matched_nodes = gt_to_nodes.get(gt_id, [])
        if not matched_nodes:
            unobserved_relocated_gt.append(gt_id)
            continue
        is_invalid = any(
            bool(nodes[n].get("slam_invalidated", False))
            or nodes[n].get("predicted_landmark_action") == "INVALIDATE"
            for n in matched_nodes
        )
        if is_invalid:
            correctly_invalidated_relocations.append(gt_id)
        else:
            missed_relocations.append(gt_id)

    n_eval_stationary = len(correctly_retained_stationary) + len(falsely_invalidated_stationary)
    n_eval_relocated = len(correctly_invalidated_relocations) + len(missed_relocations)

    retention_rate_pct = (
        round((len(correctly_retained_stationary) / n_eval_stationary) * 100.0, 2)
        if n_eval_stationary > 0
        else None
    )
    false_invalidation_rate_pct = (
        round((len(falsely_invalidated_stationary) / n_eval_stationary) * 100.0, 2)
        if n_eval_stationary > 0
        else None
    )
    relocation_invalidation_rate_pct = (
        round((len(correctly_invalidated_relocations) / n_eval_relocated) * 100.0, 2)
        if n_eval_relocated > 0
        else None
    )

    # 5. Evaluate Movability Predictions against movability_ground_truth.json
    movability_metrics: Dict[str, Any] = {}
    if movability_gt_path is not None and movability_gt_path.exists():
        with open(movability_gt_path, "r", encoding="utf-8") as f:
            mov_gt = json.load(f)
        mov_gt_instances: Dict[str, Any] = {}
        if "physical_objects" in mov_gt:
            for obj in mov_gt["physical_objects"]:
                inst_id = obj.get("object_instance_id") or obj.get("physical_instance_id")
                if inst_id:
                    mov_gt_instances[inst_id] = obj
        elif "instances" in mov_gt:
            mov_gt_instances = mov_gt["instances"]
        correct_mov = 0
        eval_mov_count = 0
        for node_id, node in nodes.items():
            gt_id = node_to_gt.get(node_id)
            if not gt_id or gt_id not in mov_gt_instances:
                continue
            true_mov = mov_gt_instances[gt_id].get("movability_gt")
            pred_mov = node.get("predicted_movability") or node.get("contextual_movability")
            if true_mov and pred_mov:
                eval_mov_count += 1
                if str(true_mov).upper() == str(pred_mov).upper():
                    correct_mov += 1

        movability_metrics = {
            "evaluable_nodes_count": eval_mov_count,
            "correct_movability_count": correct_mov,
            "movability_accuracy_pct": round((correct_mov / eval_mov_count) * 100.0, 2) if eval_mov_count > 0 else None,
        }

    # Summary
    return {
        "policy": policy_name,
        "evaluation_contract": "PURE_PREDICTION_EVALUATION_NO_LEAKAGE",
        "total_nodes_predicted": total_nodes,
        "nodes_matched_to_ground_truth": len(node_to_gt),
        "unmatched_nodes_count": total_nodes - len(node_to_gt),
        "landmark_reliability": {
            "evaluable_stationary_entities": n_eval_stationary,
            "correctly_retained_stationary_count": len(correctly_retained_stationary),
            "stationary_retention_rate_pct": retention_rate_pct,
            "falsely_invalidated_count": len(falsely_invalidated_stationary),
            "false_invalidation_rate_pct": false_invalidation_rate_pct,
            "falsely_invalidated_entity_ids": falsely_invalidated_stationary,
            "evaluable_relocated_entities": n_eval_relocated,
            "correctly_invalidated_relocations_count": len(correctly_invalidated_relocations),
            "relocation_invalidation_rate_pct": relocation_invalidation_rate_pct,
            "correctly_invalidated_entity_ids": correctly_invalidated_relocations,
            "missed_relocation_entity_ids": missed_relocations,
        },
        "movability_evaluation": movability_metrics,
    }
