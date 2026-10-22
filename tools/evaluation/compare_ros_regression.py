"""Compare non-ROS baseline (Test A) with ROS transport execution (Test B).

Acceptance criteria:
- frame_count difference == 0
- timestamp error < 1e-5s
- pose translation/rotation error < 1e-4
- zero track-ID mismatch
- identical object states
- identical relation predicates
"""

import argparse
import json
import math
from pathlib import Path
from typing import List, Dict, Any
import numpy as np

from scene_graph.config import load_config
from scene_graph.data.tum_source import TUMReplaySource
from scene_graph.pipeline.online_pipeline import OnlinePipeline


def run_baseline_test_a(config_path: str, max_frames: int = None) -> List[Dict[str, Any]]:
    """Run baseline Test A (TUMReplaySource -> OnlinePipeline) and record frame-by-frame state."""
    config = load_config(config_path)
    pipeline = OnlinePipeline(config)
    source = TUMReplaySource(config)

    records = []
    print(f"[Test A] Running baseline replay ({len(source)} frames)...")

    for i, packet in enumerate(source):
        if max_frames is not None and i >= max_frames:
            break

        graph = pipeline.update(packet)

        active_nodes = graph.get_active_nodes()
        all_edges = list(graph.edges.values())
        active_edges = [e for e in all_edges if e.is_active]

        node_map = {n.track.object_id: n.track.class_name for n in active_nodes}

        objects = [
            {
                "id": n.track.object_id,
                "class": n.track.class_name,
                "state": n.state.value,
                "confidence": float(n.track.detection_confidence),
                "centroid": [float(x) for x in n.track.centroid_world] if getattr(n.track, "centroid_world", None) is not None else None,
            }
            for n in active_nodes
        ]

        relations = [
            {
                "subject": e.subject_id,
                "subject_class": node_map.get(e.subject_id, "unknown"),
                "predicate": e.predicate,
                "object": e.object_id,
                "object_class": node_map.get(e.object_id, "unknown"),
                "state": e.state.value,
            }
            for e in sorted(active_edges, key=lambda x: (x.predicate, x.subject_id, x.object_id))
        ]

        pose_list = packet.world_T_camera.tolist() if packet.world_T_camera is not None else None

        records.append({
            "frame_index": packet.frame_index,
            "timestamp": float(packet.timestamp),
            "pose": pose_list,
            "objects": sorted(objects, key=lambda o: o["id"]),
            "relations": relations,
        })

    print(f"[Test A] Completed {len(records)} frames.")
    return records


def compare_records(
    baseline: List[Dict[str, Any]],
    ros_records: List[Dict[str, Any]],
    pose_atol: float = 1e-4,
    time_atol: float = 1e-5,
) -> bool:
    """Compare baseline records with ROS records against scientific acceptance criteria."""
    print("\n" + "=" * 60)
    print("SCIENTIFIC REGRESSION VERIFICATION: TEST A vs TEST B")
    print("=" * 60)

    if len(baseline) != len(ros_records):
        print(f"FAILED: Frame count mismatch! Baseline: {len(baseline)}, ROS: {len(ros_records)}")
        return False

    all_passed = True
    mismatched_frames = []

    for i, (rec_a, rec_b) in enumerate(zip(baseline, ros_records)):
        frame_idx = rec_a["frame_index"]

        # 1. Timestamp tolerance
        ts_diff = abs(rec_a["timestamp"] - rec_b["timestamp"])
        if ts_diff > time_atol:
            print(f"Frame {frame_idx}: Timestamp mismatch! A={rec_a['timestamp']:.6f}, B={rec_b['timestamp']:.6f}, diff={ts_diff:.2e}")
            all_passed = False
            mismatched_frames.append(frame_idx)
            continue

        # 2. Pose tolerance
        if rec_a["pose"] is not None and rec_b["pose"] is not None:
            T_a = np.array(rec_a["pose"])
            T_b = np.array(rec_b["pose"])
            pose_diff = np.max(np.abs(T_a - T_b))
            if pose_diff > pose_atol:
                print(f"Frame {frame_idx}: Pose matrix error {pose_diff:.2e} > tolerance {pose_atol}")
                all_passed = False
                mismatched_frames.append(frame_idx)
                continue

        # 3. Object Track IDs and States
        objs_a = {o["id"]: o for o in rec_a["objects"]}
        objs_b = {o["id"]: o for o in rec_b["objects"]}

        if set(objs_a.keys()) != set(objs_b.keys()):
            print(f"Frame {frame_idx}: Track ID mismatch! A={set(objs_a.keys())}, B={set(objs_b.keys())}")
            all_passed = False
            mismatched_frames.append(frame_idx)
            continue

        for obj_id, o_a in objs_a.items():
            o_b = objs_b[obj_id]
            if o_a["state"] != o_b["state"]:
                print(f"Frame {frame_idx}, Obj {obj_id}: State mismatch! A={o_a['state']}, B={o_b['state']}")
                all_passed = False
                mismatched_frames.append(frame_idx)
                break

        # 4. Relation Predicates
        rels_a = [(r["subject"], r["predicate"], r["object"]) for r in rec_a["relations"]]
        rels_b = [(r["subject"], r["predicate"], r["object"]) for r in rec_b["relations"]]

        if sorted(rels_a) != sorted(rels_b):
            print(f"Frame {frame_idx}: Relation mismatch! A={rels_a}, B={rels_b}")
            all_passed = False
            mismatched_frames.append(frame_idx)

    print("-" * 60)
    if all_passed:
        print(f"PASSED: All {len(baseline)} frames are numerically and semantically EQUIVALENT.")
        print(f"  - Frame count difference: 0")
        print(f"  - Max timestamp error: < {time_atol}s")
        print(f"  - Max pose error: < {pose_atol}")
        print(f"  - Track ID mismatches: 0")
        print(f"  - Object state mismatches: 0")
        print(f"  - Relation predicate mismatches: 0")
    else:
        print(f"FAILED: Found mismatches in {len(mismatched_frames)} frames.")
    print("=" * 60)
    return all_passed


def main():
    parser = argparse.ArgumentParser(description="Regression test comparing Test A and Test B.")
    parser.add_argument("--config", type=str, default="configs/tum_fr1_desk.yaml", help="Path to config YAML")
    parser.add_argument("--baseline_output", type=str, default="test_a_baseline.json", help="Path to save/load baseline JSON")
    parser.add_argument("--ros_output", type=str, default=None, help="Path to ROS JSON log file to compare against")
    parser.add_argument("--generate_baseline", action="store_true", help="Generate baseline records and exit")
    parser.add_argument("--max_frames", type=int, default=None, help="Stop after N frames")
    args = parser.parse_args()

    if args.generate_baseline or args.ros_output is None:
        records_a = run_baseline_test_a(args.config, max_frames=args.max_frames)
        with open(args.baseline_output, "w") as f:
            json.dump(records_a, f, indent=2)
        print(f"Saved baseline records to {args.baseline_output}")
        if args.ros_output is None:
            return

    with open(args.baseline_output, "r") as f:
        records_a = json.load(f)

    with open(args.ros_output, "r") as f:
        records_b = json.load(f)

    compare_records(records_a, records_b)


if __name__ == "__main__":
    main()
