#!/usr/bin/env python3
"""Full Pipeline Orchestrator: Dynamic 3D Scene Graph Experiment.

Connects and executes the complete experimental pipeline:
1. Input verification & SHA-256 provenance hashing
2. Authoritative SE(2) registration validation
3. pre-router evidence generation and key-aligned verification
4. Cross-session candidate matching and association baseline evaluation
5. Contextual scene graph fusion with relocation-aware pose decoupling and landmark invalidation
6. Multi-layer evaluation & run manifest generation
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple
import numpy as np

repo_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(repo_root))
sys.path.insert(0, str(repo_root / "vlm"))

if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

from vlm.src.evidence_adapter import load_registration_audit, RegistrationInfo
from vlm.src.evidence_pipeline import run_stage5_pipeline
from vlm.src.cross_session_association import (
    CrossSessionCandidateMatcher,
    compare_association_baselines,
    build_canonical_session_track_mapping,
)
from vlm.src.contextual_scene_graph_fusion import (
    ContextualSceneGraphFusion,
    ContextualMovability,
    ObservedMotionState,
)


def compute_file_sha256(path: Path) -> str:
    """Compute deterministic SHA-256 hash of a file."""
    hasher = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()


def get_git_commit_hash(repo_dir: Path) -> str:
    """Get current git commit hash if git repository is available."""
    try:
        res = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(repo_dir),
            capture_output=True,
            text=True,
            check=True,
        )
        return res.stdout.strip()
    except Exception:
        return "UNKNOWN_UNTRACKED_REVISION"


def run_full_experiment(
    s1_session: str = "semantic_mapping_my_desk_recording_01",
    s2_session: str = "semantic_mapping_my_desk_recording_02",
    obs_path: Optional[Path] = None,
    reg_path: Optional[Path] = None,
    annotations_path: Optional[Path] = None,
    out_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    """Execute complete reproducible experiment from observations to final evaluated scene graph."""
    start_time = time.time()
    print("=" * 80)
    print("STARTING FULL REPRODUCIBLE EXPERIMENT: DYNAMIC 3D SCENE GRAPH")
    print("=" * 80)

    # 1. Resolve & Validate Paths
    if obs_path is None:
        obs_path = repo_root / "vlm" / "outputs" / "object_observations.jsonl"
    if reg_path is None:
        reg_path = repo_root / "vlm" / "outputs" / "registration_audit.json"
    if annotations_path is None:
        annotations_path = repo_root / "vlm" / "benchmarks" / "cross_session_identity_annotations.json"
    if out_dir is None:
        out_dir = repo_root / "vlm" / "outputs" / "full_experiment_run"

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for required_file, name in [
        (obs_path, "Observations JSONL"),
        (reg_path, "Authoritative Registration Audit"),
        (annotations_path, "Canonical Cross-Session Identity Annotations"),
    ]:
        if not Path(required_file).exists():
            raise FileNotFoundError(f"Missing required artifact: {name} at {required_file}")

    # Compute provenance hashes
    obs_hash = compute_file_sha256(obs_path)
    reg_hash = compute_file_sha256(reg_path)
    ann_hash = compute_file_sha256(annotations_path)
    git_commit = get_git_commit_hash(repo_root)

    print(f"• Git Commit:                 {git_commit}")
    print(f"• Observations SHA-256:       {obs_hash[:16]}...")
    print(f"• Registration SHA-256:       {reg_hash[:16]}...")
    print(f"• Annotations SHA-256:        {ann_hash[:16]}...")

    # 2. Load & Validate Authoritative Registration
    reg_info = load_registration_audit(reg_path)
    if not reg_info.is_valid:
        raise ValueError(
            f"Authoritative registration failed validation: {reg_info.status} - {reg_info.diagnostics}"
        )
    print(
        f"• Verified Registration:      Rotation={reg_info.rotation_deg:.3f}°, "
        f"Translation=[{reg_info.translation_m[0]:.4f}, {reg_info.translation_m[1]:.4f}] m, "
        f"Inliers={reg_info.robust_inlier_count}/{reg_info.anchor_count}"
    )

    # 3. Evidence Generation & Record Alignment Check
    print("\n--- [Step 1: Evidence Pipeline Execution & Alignment Check] ---")
    pipeline_res = run_stage5_pipeline(obs_jsonl_path=obs_path)
    geo_path = repo_root / "vlm" / "outputs" / "geometric_evidence.jsonl"
    temp_path = repo_root / "vlm" / "outputs" / "temporal_evidence.jsonl"
    sem_path = repo_root / "vlm" / "outputs" / "semantic_evidence.jsonl"
    fused_path = repo_root / "vlm" / "outputs" / "fused_evidence.jsonl"

    observations: List[Dict[str, Any]] = []
    with open(obs_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                observations.append(json.loads(line))

    fused_records: List[Dict[str, Any]] = []
    with open(fused_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                fused_records.append(json.loads(line))

    if len(observations) != len(fused_records):
        raise ValueError(
            f"Misaligned evidence pipeline: observations count ({len(observations)}) != "
            f"fused evidence count ({len(fused_records)})"
        )
    print(f"• Verified Evidence Records:  {len(fused_records):,} aligned observations")

    # 4. Cross-Session Association & Baseline Evaluation
    print("\n--- [Step 2: Cross-Session Association & Baseline Comparison] ---")
    canonical_map, gt_cross_pairs = build_canonical_session_track_mapping(annotations_path)
    s1_clusters = CrossSessionCandidateMatcher.extract_clusters_from_observations(
        observations, s1_session, min_obs=2
    )
    s2_clusters = CrossSessionCandidateMatcher.extract_clusters_from_observations(
        observations, s2_session, min_obs=2
    )
    print(f"• S1 Clusters ({s1_session}): {len(s1_clusters)}")
    print(f"• S2 Clusters ({s2_session}): {len(s2_clusters)}")

    baselines = compare_association_baselines(s1_clusters, s2_clusters, reg_info, annotations_path)

    reloc_matcher_eval = baselines["proposed_relocation_aware"]["evaluation"]
    reloc_matcher_res = baselines["proposed_relocation_aware"]["result"]

    print(f"• Proposed Relocation-Aware Matcher:")
    print(f"  - Matched Pairs:            {len(reloc_matcher_res.matched_pairs)}")
    print(f"  - Ambiguous Rejected:       {len(reloc_matcher_res.ambiguous_pairs)}")
    print(f"  - Candidate Precision:      {reloc_matcher_eval['precision_on_annotated_subset_pct']}%")
    print(f"  - Static Instance Recall:   {reloc_matcher_eval['recall_on_static_multi_session_pct']}%")
    print(f"  - Relocated Instance Recall:{reloc_matcher_eval['relocated_instance_recall_pct']}%")
    print(f"  - Instance-Level F1:        {reloc_matcher_eval['instance_level_f1_pct']}%")

    # 5. Contextual Scene Graph Fusion
    print("\n--- [Step 3: Contextual Scene Graph Fusion & Lifecycle Maintenance] ---")
    fusion = ContextualSceneGraphFusion()

    # Track contributing pairs
    matched_pairs = reloc_matcher_res.matched_pairs
    pair_map = {(m.s1_entity_id, m.s2_entity_id): m for m in matched_pairs}

    # Fuse observations into SceneGraph
    # We update nodes for S1 clusters first, then S2 clusters
    for t1, c1 in s1_clusters.items():
        entity_id = canonical_map.get((s1_session, t1), f"entity_s1_{t1}")
        obs_meta = {
            "timestamp": 100.0,
            "source_session": s1_session,
            "obs_id": f"obs_cluster_{t1}",
            "local_track_id": t1,
            "centroid": c1.centroid_floor.tolist() if len(c1.centroid_floor) == 3 else [c1.centroid_floor[0], c1.centroid_floor[1], 0.0],
            "spatial_extent_m": c1.spatial_extent_m,
            "depth_valid": True,
            "coordinate_frame": "floor_map",
        }
        routed = {
            "decision": "RESOLVE_CHEAP",
            "prediction": "MOVABLE" if c1.semantic_class in ["chair", "desk", "storage box", "laptop"] else ("IMMOVABLE" if c1.semantic_class in ["door", "wall", "floor"] else "MOBILE"),
            "observed_motion": "STATIC",
            "evidence_provenance": "MEASURED_CLUSTER",
            "relocation_evaluable": False,
        }
        fusion.update_or_create_node(
            entity_id=entity_id,
            semantic_class=c1.semantic_class,
            routed_result=routed,
            observation_meta=obs_meta,
        )

    # Now fuse S2 clusters using associations
    relocated_fused_count = 0
    stationary_fused_count = 0
    for t2, c2 in s2_clusters.items():
        # Check if matched with an S1 cluster
        matched_m = next((m for m in matched_pairs if m.s2_entity_id == t2), None)
        if matched_m is not None:
            t1 = matched_m.s1_entity_id
            entity_id = canonical_map.get((s1_session, t1), f"entity_s1_{t1}")
            is_reloc = (matched_m.status == "MATCHED_RELOCATED" or matched_m.distance_m >= 0.25)
            if is_reloc:
                relocated_fused_count += 1
            else:
                stationary_fused_count += 1

            # Transform S2 centroid into S1 reference coordinate frame
            c2_floor_2d = c2.centroid_floor[:2]
            c2_aligned_2d = reg_info.R_12 @ c2_floor_2d + reg_info.translation_m
            z_coord = float(c2.centroid_floor[2]) if len(c2.centroid_floor) >= 3 else 0.0
            aligned_centroid_3d = [float(c2_aligned_2d[0]), float(c2_aligned_2d[1]), z_coord]

            obs_meta = {
                "timestamp": 200.0,
                "source_session": s2_session,
                "obs_id": f"obs_cluster_{t2}",
                "local_track_id": t2,
                "centroid": aligned_centroid_3d,
                "spatial_extent_m": c2.spatial_extent_m,
                "depth_valid": True,
                "coordinate_frame": "reference_frame",
                "is_relocation_event": is_reloc,
            }
            routed = {
                "decision": "RESOLVE_CHEAP",
                "prediction": "MOVABLE",
                "observed_motion": "DISPLACED" if is_reloc else "STATIC",
                "world_displacement_m": matched_m.distance_m if is_reloc else 0.0,
                "displacement_uncertainty_m": c2.position_uncertainty_m,
                "evidence_provenance": "MEASURED_CLUSTER_ALIGNED",
                "relocation_evaluable": True,
            }
            fusion.update_or_create_node(
                entity_id=entity_id,
                semantic_class=c2.semantic_class,
                routed_result=routed,
                observation_meta=obs_meta,
            )
        else:
            # Unmatched S2 entity
            entity_id = canonical_map.get((s2_session, t2), f"entity_s2_{t2}")
            c2_floor_2d = c2.centroid_floor[:2]
            c2_aligned_2d = reg_info.R_12 @ c2_floor_2d + reg_info.translation_m
            z_coord = float(c2.centroid_floor[2]) if len(c2.centroid_floor) >= 3 else 0.0

            obs_meta = {
                "timestamp": 200.0,
                "source_session": s2_session,
                "obs_id": f"obs_cluster_{t2}",
                "local_track_id": t2,
                "centroid": [float(c2_aligned_2d[0]), float(c2_aligned_2d[1]), z_coord],
                "spatial_extent_m": c2.spatial_extent_m,
                "depth_valid": True,
                "coordinate_frame": "reference_frame",
            }
            routed = {
                "decision": "RESOLVE_CHEAP",
                "prediction": "MOVABLE",
                "observed_motion": "STATIC",
                "evidence_provenance": "MEASURED_CLUSTER_S2",
                "relocation_evaluable": False,
            }
            fusion.update_or_create_node(
                entity_id=entity_id,
                semantic_class=c2.semantic_class,
                routed_result=routed,
                observation_meta=obs_meta,
            )

    scene_graph_dict = fusion.to_dict()
    total_nodes = len(scene_graph_dict["nodes"])
    invalidated_landmarks = sum(1 for n in scene_graph_dict["nodes"].values() if n.get("slam_invalidated", False))
    nodes_with_history = sum(1 for n in scene_graph_dict["nodes"].values() if len(n.get("pose_history", [])) > 0)

    print(f"• Total Scene Graph Nodes:    {total_nodes}")
    print(f"• Stationary Merged Entities: {stationary_fused_count}")
    print(f"• Relocated Fused Entities:   {relocated_fused_count}")
    print(f"• Invalidated SLAM Landmarks: {invalidated_landmarks}")
    print(f"• Nodes Preserving History:   {nodes_with_history}")

    # 6. Save Artifacts & Manifest
    print("\n--- [Step 4: Writing Artifacts & Execution Manifest] ---")
    sg_path = out_dir / "scene_graph.json"
    with open(sg_path, "w", encoding="utf-8") as f:
        json.dump(scene_graph_dict, f, indent=2)

    baselines_summary_path = out_dir / "association_baselines_summary.json"
    baselines_serializable = {}
    for b_name, b_data in baselines.items():
        baselines_serializable[b_name] = {
            "matched_pairs_count": len(b_data["result"].matched_pairs),
            "ambiguous_pairs_count": len(b_data["result"].ambiguous_pairs),
            "evaluation": b_data["evaluation"],
        }
    with open(baselines_summary_path, "w", encoding="utf-8") as f:
        json.dump(baselines_serializable, f, indent=2)

    per_instance_table_path = out_dir / "per_instance_association_diagnostics.json"
    with open(per_instance_table_path, "w", encoding="utf-8") as f:
        json.dump(reloc_matcher_eval.get("per_instance_results", []), f, indent=2)

    elapsed_s = time.time() - start_time
    manifest = {
        "experiment_name": "dynamic_3d_scene_graph_full_pipeline",
        "timestamp": time.time(),
        "execution_duration_seconds": round(elapsed_s, 2),
        "git_commit": git_commit,
        "input_artifacts": {
            "observations_file": str(obs_path),
            "observations_sha256": obs_hash,
            "observations_count": len(observations),
            "registration_audit_file": str(reg_path),
            "registration_sha256": reg_hash,
            "identity_annotations_file": str(annotations_path),
            "identity_annotations_sha256": ann_hash,
        },
        "registration_diagnostics": {
            "rotation_deg": reg_info.rotation_deg,
            "translation_m": reg_info.translation_m.tolist(),
            "robust_inlier_count": reg_info.robust_inlier_count,
            "total_anchors": reg_info.anchor_count,
            "inlier_rms_residual_m": reg_info.inlier_rms_residual_m,
            "anchor_rms_residual_m": reg_info.anchor_rms_residual_m,
        },
        "association_summary": {
            "candidate_precision_pct": reloc_matcher_eval["precision_on_annotated_subset_pct"],
            "static_instance_recall_pct": reloc_matcher_eval["recall_on_static_multi_session_pct"],
            "relocated_instance_recall_pct": reloc_matcher_eval["relocated_instance_recall_pct"],
            "all_instance_recall_pct": reloc_matcher_eval["all_instance_recall_pct"],
            "instance_f1_pct": reloc_matcher_eval["instance_level_f1_pct"],
            "false_merge_rate_pct": reloc_matcher_eval["false_merge_rate_pct"],
            "false_split_rate_pct": reloc_matcher_eval["false_split_rate_pct"],
        },
        "scene_graph_fusion_summary": {
            "total_nodes": total_nodes,
            "stationary_fused_entities": stationary_fused_count,
            "relocated_fused_entities": relocated_fused_count,
            "invalidated_slam_landmarks": invalidated_landmarks,
            "nodes_preserving_pose_history": nodes_with_history,
        },
        "output_artifacts": {
            "scene_graph_json": str(sg_path),
            "baselines_summary_json": str(baselines_summary_path),
            "per_instance_table_json": str(per_instance_table_path),
        },
    }

    manifest_path = out_dir / "run_manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print(f"\nExperiment execution complete in {elapsed_s:.2f}s!")
    print(f"• Final Scene Graph:          {sg_path}")
    print(f"• Baseline Summary:           {baselines_summary_path}")
    print(f"• Per-Instance Diagnostics:   {per_instance_table_path}")
    print(f"• Run Manifest:               {manifest_path}")
    print("=" * 80)

    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run complete Dynamic 3D Scene Graph experiment.")
    parser.add_argument("--s1", default="semantic_mapping_my_desk_recording_01", help="Session 1 keyword")
    parser.add_argument("--s2", default="semantic_mapping_my_desk_recording_02", help="Session 2 keyword")
    parser.add_argument("--obs", default=None, help="Path to object_observations.jsonl")
    parser.add_argument("--reg", default=None, help="Path to registration_audit.json")
    parser.add_argument("--annotations", default=None, help="Path to cross_session_identity_annotations.json")
    parser.add_argument("--out-dir", default=None, help="Output directory")

    args = parser.parse_args()
    run_full_experiment(
        s1_session=args.s1,
        s2_session=args.s2,
        obs_path=Path(args.obs) if args.obs else None,
        reg_path=Path(args.reg) if args.reg else None,
        annotations_path=Path(args.annotations) if args.annotations else None,
        out_dir=Path(args.out_dir) if args.out_dir else None,
    )
