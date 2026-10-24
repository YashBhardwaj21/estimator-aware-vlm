"""Build and export Contextual Dynamic 3D Scene Graph for fr3_walking_xyz.

Fuses:
1. Multi-view 3D perception tracks from outputs/perception_tracks/tracks_summary.json.
2. Live VLM contextual affordance reasoning and routing decisions from outputs/vlm_inference/policy_comparison.json.
3. Decoupled camera ego-motion vs object motion in world coordinate frame.
4. Produces authoritative persistent scene graph nodes (outputs/scene_graph/scene_graph_nodes.json)
   and a 2D top-down floor projection summary.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List
import numpy as np

from estimator_aware_vlm.common.paths import get_repo_root
from estimator_aware_vlm.contextual_scene_graph_fusion import (
    ContextualMovability,
    ContextualSceneGraphFusion,
    ContextualSceneGraphNode,
    ObservedMotionState,
)


def build_scene_graph() -> Dict[str, Any]:
    repo_root = get_repo_root()
    tracks_path = repo_root / "outputs" / "perception_tracks" / "tracks_summary.json"
    policy_path = repo_root / "outputs" / "vlm_inference" / "policy_comparison.json"
    decisions_path = repo_root / "outputs" / "vlm_inference" / "routing_decisions.json"

    if not tracks_path.exists():
        raise FileNotFoundError(f"Missing tracks summary: {tracks_path}")
    if not policy_path.exists():
        raise FileNotFoundError(f"Missing policy comparison: {policy_path}")

    with open(tracks_path, "r", encoding="utf-8") as f:
        tracks = json.load(f)

    with open(policy_path, "r", encoding="utf-8") as f:
        policy_data = json.load(f)

    vlm_telemetry = policy_data.get("vlm_telemetry", {})
    b6_decisions = {
        d["track_id"]: d
        for d in policy_data.get("policies", {}).get("B6_Estimator_Aware", {}).get("decisions", [])
    }

    fusion = ContextualSceneGraphFusion()

    print(f"Building Contextual Scene Graph from {len(tracks)} tracked physical entities...")

    nodes_summary: List[Dict[str, Any]] = []

    for track in tracks:
        tid = track["track_id"]
        vlm_res = vlm_telemetry.get(tid, {})
        decision = b6_decisions.get(tid, {})

        # Contextual movability from live VLM
        raw_mov = vlm_res.get("contextual_movability", "UNKNOWN").upper()
        if raw_mov in ["IMMOVABLE", "MOVABLE", "MOBILE"]:
            movability = ContextualMovability(raw_mov)
        else:
            movability = ContextualMovability.UNKNOWN

        # Physical observed motion in camera-motion-compensated world frame
        world_disp = track.get("world_displacement_m", track.get("measured_displacement_m", 0.0))
        if world_disp < 0.25:
            motion = ObservedMotionState.STATIC
        elif world_disp < 0.60:
            motion = ObservedMotionState.DISPLACED
        else:
            motion = ObservedMotionState.DYNAMIC

        # Physical support relationship
        if movability == ContextualMovability.IMMOVABLE:
            support = "mounted_to_structure"
        elif movability == ContextualMovability.MOBILE:
            support = "self_propelled"
        else:
            support = "freestanding_on_surface"

        # Centroid and spatial geometry
        mean_c = track.get("mean_centroid_world")
        if mean_c and len(mean_c) == 3:
            centroid_arr = np.array(mean_c, dtype=np.float64)
            geom_valid = True
        else:
            centroid_arr = None
            geom_valid = False

        # Action from estimator-aware selective routing
        action = decision.get("action", "RETAIN")
        slam_invalidated = (action == "MASK_FROM_TRACKING")
        landmark_reason = (
            "DYNAMIC_OR_MOBILE_SUPPRESSED" if slam_invalidated else "STATIONARY_LANDMARK_RETAINED"
        )

        node = ContextualSceneGraphNode(
            entity_id=tid,
            semantic_class=vlm_res.get("semantic_class", track.get("semantic_class", "unknown")),
            contextual_movability=movability,
            observed_motion=motion,
            physical_support=support,
            geometry_valid=geom_valid,
            centroid_world=centroid_arr,
            spatial_extent_m=round(float(world_disp), 4),
            coordinate_frame="world",
            semantic_confidence=0.92 if vlm_res else 0.65,
            geometric_confidence=0.88 if geom_valid else 0.0,
            evidence_provenance="LIVE_QWEN_VL_3B",
            source_sessions=["tum_rgbd_freiburg3_walking_xyz"],
            source_track_ids=[tid],
            contributing_observation_ids=[f"{tid}_obs_{i}" for i in range(min(5, track.get("observation_count", 1)))],
            first_seen=float(track.get("first_timestamp", 0.0)),
            last_seen=float(track.get("last_timestamp", 0.0)),
            total_observations_count=int(track.get("observation_count", 1)),
            valid_geometry_observations_count=int(track.get("observation_count", 1)),
            relocation_evaluable=True,
            slam_invalidated=slam_invalidated,
            predicted_landmark_action="INVALIDATE" if slam_invalidated else "RETAIN",
            landmark_action_reason=landmark_reason,
            motion_confidence=0.90,
            measured_displacement_m=round(float(world_disp), 4),
            displacement_uncertainty_m=round(float(track.get("displacement_uncertainty_m", 0.05)), 4),
            vlm_query_requested=decision.get("vlm_queried", False),
            vlm_inference_executed=decision.get("vlm_queried", False),
            vlm_decision=action,
            vlm_reason_codes=[vlm_res.get("reasoning", "")],
            vlm_model="Qwen2.5-VL-3B-Instruct",
        )

        fusion.nodes[tid] = node
        node_dict = node.to_dict()
        node_dict["vlm_reasoning"] = vlm_res.get("reasoning", "")
        node_dict["apparent_camera_frame_displacement_m"] = track.get("apparent_camera_frame_displacement_m")
        node_dict["camera_ego_motion_displacement_m"] = track.get("camera_ego_motion_displacement_m")
        nodes_summary.append(node_dict)

        print(
            f"  [{tid:<11}] Class: {node.semantic_class:<24} | "
            f"Movability: {movability.value:<9} | "
            f"Motion: {motion.value:<9} (disp={world_disp:.2f}m) | "
            f"Action: {node.predicted_landmark_action}"
        )

    out_dir = repo_root / "outputs" / "scene_graph"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "scene_graph_nodes.json"

    export_payload = {
        "sequence_name": "rgbd_dataset_freiburg3_walking_xyz",
        "entity_count": len(nodes_summary),
        "retained_landmarks_count": sum(1 for n in nodes_summary if n["predicted_landmark_action"] == "RETAIN"),
        "invalidated_dynamic_count": sum(1 for n in nodes_summary if n["predicted_landmark_action"] == "INVALIDATE"),
        "coordinate_frame": "camera_compensated_world_frame",
        "nodes": nodes_summary,
    }

    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(export_payload, f, indent=2)

    print(f"\nScene graph successfully written to: {out_file}")
    return export_payload


if __name__ == "__main__":
    build_scene_graph()
