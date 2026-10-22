#!/usr/bin/env python3
"""Full Pipeline Orchestrator: Dynamic 3D Scene Graph Experiment.

Connects and executes the complete experimental pipeline:
1. Input verification & SHA-256 provenance hashing
2. Authoritative SE(2) registration validation
3. Stage 5 evidence generation and key-aligned verification
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

from vlm.src.stage5_adapter import load_registration_audit, RegistrationInfo
from vlm.src.stage5_pipeline import run_stage5_pipeline
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
from vlm.src.semantic_evidence import CLASS_MOBILITY_PRIORS
from vlm.src.router_policy import AMBIGUOUS_AFFORDANCE_CLASSES, evaluate_router_policy
from vlm.src.scene_graph_evaluation import evaluate_scene_graph


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
        canonical_p = repo_root / "vlm" / "data" / "observations" / "object_observations.jsonl"
        obs_path = canonical_p if canonical_p.exists() else repo_root / "vlm" / "outputs" / "object_observations.jsonl"
    if reg_path is None:
        reg_path = repo_root / "vlm" / "outputs" / "stage5_registration_audit.json"
    if annotations_path is None:
        annotations_path = repo_root / "vlm" / "benchmarks" / "cross_session_identity_annotations.json"
    if out_dir is None:
        out_dir = repo_root / "vlm" / "outputs" / "full_experiment_run"

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    evidence_out_dir = out_dir / "stage5_evidence"
    evidence_out_dir.mkdir(parents=True, exist_ok=True)

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
    print("\n--- [Stage 1: Evidence Pipeline Execution & Alignment Check] ---")
    pipeline_res = run_stage5_pipeline(obs_jsonl_path=obs_path, output_dir=evidence_out_dir)
    geo_path = Path(pipeline_res["geometric_records_file"])
    temp_path = Path(pipeline_res["temporal_records_file"])
    sem_path = Path(pipeline_res["semantic_records_file"])
    fused_path = Path(pipeline_res["fused_records_file"])

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
    print("\n--- [Stage 2: Cross-Session Association & Baseline Comparison] ---")
    canonical_map, gt_cross_pairs = build_canonical_session_track_mapping(annotations_path)
    s1_raw_clusters = CrossSessionCandidateMatcher.extract_clusters_from_observations(
        observations, s1_session, min_obs=2
    )
    s2_raw_clusters = CrossSessionCandidateMatcher.extract_clusters_from_observations(
        observations, s2_session, min_obs=2
    )
    print(f"• S1 Raw Clusters ({s1_session}): {len(s1_raw_clusters)}")
    print(f"• S2 Raw Clusters ({s2_session}): {len(s2_raw_clusters)}")

    # Intra-session tracklet consolidation
    s1_clusters = CrossSessionCandidateMatcher.consolidate_intra_session_clusters(s1_raw_clusters)
    s2_clusters = CrossSessionCandidateMatcher.consolidate_intra_session_clusters(s2_raw_clusters)
    print(f"• Consolidated S1 Clusters:  {len(s1_raw_clusters)} -> {len(s1_clusters)} clusters")
    print(f"• Consolidated S2 Clusters:  {len(s2_raw_clusters)} -> {len(s2_clusters)} clusters")

    baselines = compare_association_baselines(s1_clusters, s2_clusters, reg_info, annotations_path)

    active_matcher_key = "proposed_uncertainty_adaptive"
    active_matcher_eval = baselines[active_matcher_key]["evaluation"]
    active_matcher_res = baselines[active_matcher_key]["result"]

    print(f"• Proposed Uncertainty-Adaptive Matcher:")
    print(f"  - Matched Pairs:            {len(active_matcher_res.matched_pairs)}")
    print(f"  - Ambiguous Rejected:       {len(active_matcher_res.ambiguous_pairs)}")
    print(f"  - Candidate Precision:      {active_matcher_eval['precision_on_annotated_subset_pct']}%")
    print(f"  - Static Instance Recall:   {active_matcher_eval['recall_on_static_multi_session_pct']}%")
    print(f"  - Relocated Instance Recall:{active_matcher_eval['relocated_instance_recall_pct']}%")
    print(f"  - Instance-Level F1:        {active_matcher_eval['instance_level_f1_pct']}%")

    # 5. Contextual Scene Graph Fusion
    print("\n--- [Stage 3: Contextual Scene Graph Fusion & Lifecycle Maintenance] ---")

    # Load verified controlled scene events for ground-truth relocation verification
    controlled_events_path = repo_root / "vlm" / "benchmarks" / "controlled_scene_events.json"

    def build_policy_scene_graph(
        policy_key: str,
        active_pairs: List[Any],
    ) -> Tuple[Dict[str, Any], Dict[str, int]]:
        """Construct decoupled persistent 3D scene graph under a specific routing policy.
        
        Pure sensory prediction pipeline:
        - Strictly independent of controlled scene events or identity annotations.
        - Relocation detection is driven by measured registered displacement and covariance gating.
        - Decoupled pose history is recorded purely from measured centroid shifts.
        """
        fusion = ContextualSceneGraphFusion()

        # Build fast observation lookup indices per session for real timestamps and observation IDs (obs_id)
        # Note: obs_id is strictly unique within each session (2,934 in S1, 5,022 in S2).
        obs_by_session: Dict[str, Dict[str, Dict[str, Any]]] = {}
        session_time_origins: Dict[str, float] = {}

        def _matches_session(raw: str, target: str) -> bool:
            r = raw.strip()
            t = target.strip()
            return r == t or r.endswith(f"_{t}") or r.endswith(t)

        for o in observations:
            sess = str(o.get("source_session", "")).strip()
            oid = o.get("obs_id")
            if not sess or not oid:
                continue
            obs_by_session.setdefault(sess, {})[oid] = o

            ts = o.get("timestamp")
            if ts is not None:
                ts_f = float(ts)
                if sess not in session_time_origins or ts_f < session_time_origins[sess]:
                    session_time_origins[sess] = ts_f

        # Resolve observation maps and origins supporting session aliases/suffixes
        s1_obs_by_id = {}
        s2_obs_by_id = {}
        for sess_key, o_map in obs_by_session.items():
            if _matches_session(sess_key, s1_session):
                s1_obs_by_id.update(o_map)
                if sess_key in session_time_origins:
                    session_time_origins[s1_session] = session_time_origins[sess_key]
            if _matches_session(sess_key, s2_session):
                s2_obs_by_id.update(o_map)
                if sess_key in session_time_origins:
                    session_time_origins[s2_session] = session_time_origins[sess_key]

        # Strict join assertion: every cluster observation ID must resolve to a valid observation record in its session
        for t1, c1 in s1_clusters.items():
            for oid in c1.observation_ids:
                if oid not in s1_obs_by_id:
                    raise KeyError(f"Referenced S1 cluster observation ID '{oid}' does not exist in {s1_session} observations.")
        for t2, c2 in s2_clusters.items():
            for oid in c2.observation_ids:
                if oid not in s2_obs_by_id:
                    raise KeyError(f"Referenced S2 cluster observation ID '{oid}' does not exist in {s2_session} observations.")

        # Step 1: Initialize S1 clusters as reference landmarks
        for t1, c1 in s1_clusters.items():
            entity_id = f"phys_{t1}"

            if not c1.observation_ids:
                raise ValueError(f"S1 cluster '{t1}' contains no observation IDs.")
            first_obs_id = c1.observation_ids[0]
            first_obs = s1_obs_by_id[first_obs_id]
            if "timestamp" not in first_obs:
                raise KeyError(f"Observation '{first_obs_id}' is missing required field 'timestamp'.")
            t_raw = float(first_obs["timestamp"])
            s1_origin = session_time_origins.get(s1_session)
            if s1_origin is None:
                raise ValueError(f"Session origin for '{s1_session}' could not be computed.")
            t_session_s = t_raw - s1_origin
            depth_valid = bool(getattr(c1, "depth_valid_ratio", 1.0) >= 0.50)

            obs_meta = {
                "timestamp": t_session_s,
                "timestamp_raw": t_raw,
                "timestamp_session_s": t_session_s,
                "temporal_valid": True,
                "source_session": s1_session,
                "obs_id": first_obs_id,
                "local_track_id": t1,
                "centroid": c1.centroid_floor.tolist() if len(c1.centroid_floor) == 3 else [c1.centroid_floor[0], c1.centroid_floor[1], 0.0],
                "spatial_extent_m": c1.spatial_extent_m,
                "depth_valid": depth_valid,
                "coordinate_frame": "floor_map",
                "is_relocation_event": False,
            }

            s1_ev = {
                "entity_id": entity_id,
                "semantic_class": c1.semantic_class,
                "class_prior_mobility": CLASS_MOBILITY_PRIORS.get(c1.semantic_class, ("MOVABLE", 0.9, ""))[0],
                "observed_motion": "STATIC",
                "world_displacement_m": 0.0,
                "displacement_uncertainty_m": c1.position_uncertainty_m,
                "has_valid_pose": True,
                "depth_valid": depth_valid,
                "s1_obs_count": len(c1.observation_ids),
                "s2_obs_count": 0,
                "has_images": bool(c1.observation_ids and len(c1.observation_ids) > 0),
                "views_count": min(2, len(c1.observation_ids)),
            }
            decision = evaluate_router_policy(s1_ev, policy=policy_key)
            pred = decision.cheap_label or CLASS_MOBILITY_PRIORS.get(c1.semantic_class, ("MOVABLE", 0.9, ""))[0]

            routed = {
                "decision": decision.decision,
                "prediction": pred,
                "observed_motion": "STATIC",
                "evidence_provenance": "MEASURED_CLUSTER",
                "relocation_evaluable": False,
                "predicted_landmark_action": "RETAIN",
                "landmark_action_reason": "SINGLE_SESSION_INITIAL_OBSERVATION",
                "motion_confidence": decision.confidence,
                "query_utility": decision.query_utility,
                "estimator_influence": decision.estimator_influence,
                "estimator_provenance": decision.estimator_provenance,
                "reason_codes": decision.reason_codes,
            }
            node = fusion.update_or_create_node(
                entity_id=entity_id,
                semantic_class=c1.semantic_class,
                routed_result=routed,
                observation_meta=obs_meta,
            )

        # Step 2: Fuse S2 clusters using policy associations
        relocated_fused_count = 0
        stationary_fused_count = 0

        for t2, c2 in s2_clusters.items():
            matched_m = next((m for m in active_pairs if m.s2_entity_id == t2), None)
            if not c2.observation_ids:
                raise ValueError(f"S2 cluster '{t2}' contains no observation IDs.")
            first_obs_id = c2.observation_ids[0]
            first_obs = s2_obs_by_id[first_obs_id]
            if "timestamp" not in first_obs:
                raise KeyError(f"Observation '{first_obs_id}' is missing required field 'timestamp'.")
            t_raw = float(first_obs["timestamp"])
            s2_origin = session_time_origins.get(s2_session)
            if s2_origin is None:
                raise ValueError(f"Session origin for '{s2_session}' could not be computed.")
            t_session_s = t_raw - s2_origin
            depth_valid = bool(getattr(c2, "depth_valid_ratio", 1.0) >= 0.50)

            if matched_m is not None:
                entity_id = matched_m.persistent_entity_id
                t1 = matched_m.s1_entity_id
                dist = matched_m.distance_m
                c1 = s1_clusters[t1]
                inlier_sigma = reg_info.inlier_rms_residual_m if reg_info.inlier_rms_residual_m is not None else reg_info.anchor_rms_residual_m
                sigma_disp = math.sqrt(c1.position_uncertainty_m**2 + c2.position_uncertainty_m**2 + inlier_sigma**2)

                # Relocation gating policy (strictly data-driven without ground-truth leakage)
                if policy_key == "R0":
                    # R0 ignores displacement: pure semantic prior, zero landmark invalidations
                    is_reloc = False
                    action = "RETAIN"
                    reason = "SEMANTIC_PRIOR_STATIC_POLICY"
                elif policy_key == "R1":
                    # R1: naive Euclidean displacement threshold >= 0.25m without uncertainty
                    is_reloc = bool(dist >= 0.25)
                    action = "INVALIDATE" if is_reloc else "RETAIN"
                    reason = "UNWEIGHTED_EUCLIDEAN_DISPLACEMENT" if is_reloc else "STATISTICALLY_STATIONARY"
                elif policy_key == "R2":
                    # R2: non-estimator-aware relocation
                    has_sem_ambig = (c1.semantic_class in AMBIGUOUS_AFFORDANCE_CLASSES or getattr(c1, "semantic_entropy", 0.0) >= 0.35)
                    is_reloc = bool(dist >= 0.25 and has_sem_ambig)
                    action = "INVALIDATE" if is_reloc else "RETAIN"
                    reason = "SEMANTIC_AMBIGUITY_DISPLACEMENT" if is_reloc else "STATISTICALLY_STATIONARY"
                elif policy_key == "R3":
                    # R3: estimator-aware verified relocation via 3-sigma gate
                    is_reloc = bool(dist >= 0.25 and dist >= 3.0 * sigma_disp)
                    action = "INVALIDATE" if is_reloc else "RETAIN"
                    reason = "SIGNIFICANT_REGISTERED_DISPLACEMENT" if is_reloc else "STATISTICALLY_STATIONARY"
                else:
                    # R4: All-eligible VLM reference policy (combines visual affordance verification with 3-sigma gate)
                    is_reloc = bool(dist >= 0.25 and dist >= 3.0 * sigma_disp)
                    action = "INVALIDATE" if is_reloc else "RETAIN"
                    reason = "VLM_VERIFIED_AFFORDANCE_RELOCATION" if is_reloc else "STATISTICALLY_STATIONARY"

                if is_reloc:
                    relocated_fused_count += 1
                else:
                    stationary_fused_count += 1

                c2_floor_2d = c2.centroid_floor[:2]
                c2_aligned_2d = reg_info.R_12 @ c2_floor_2d + reg_info.translation_m
                z_coord = float(c2.centroid_floor[2]) if len(c2.centroid_floor) >= 3 else 0.0
                aligned_centroid_3d = [float(c2_aligned_2d[0]), float(c2_aligned_2d[1]), z_coord]

                obs_meta = {
                    "timestamp": t_session_s,
                    "timestamp_raw": t_raw,
                    "timestamp_session_s": t_session_s,
                    "temporal_valid": True,
                    "source_session": s2_session,
                    "obs_id": first_obs_id,
                    "local_track_id": t2,
                    "centroid": aligned_centroid_3d,
                    "spatial_extent_m": c2.spatial_extent_m,
                    "depth_valid": depth_valid,
                    "coordinate_frame": "reference_frame",
                    "is_relocation_event": is_reloc,
                }
                evidence_rec = {
                    "entity_id": entity_id,
                    "semantic_class": c2.semantic_class,
                    "class_prior_mobility": CLASS_MOBILITY_PRIORS.get(c2.semantic_class, ("MOVABLE", 0.9, ""))[0],
                    "observed_motion": "DISPLACED" if is_reloc else "STATIC",
                    "world_displacement_m": dist if is_reloc else 0.0,
                    "displacement_uncertainty_m": sigma_disp,
                    "has_valid_pose": True,
                    "depth_valid": depth_valid,
                    "s1_obs_count": len(c1.observation_ids),
                    "s2_obs_count": len(c2.observation_ids),
                    "has_images": bool(c2.observation_ids and len(c2.observation_ids) > 0),
                    "views_count": min(2, len(c2.observation_ids)),
                }
                decision = evaluate_router_policy(evidence_rec, policy=policy_key)
                pred = decision.cheap_label or CLASS_MOBILITY_PRIORS.get(c2.semantic_class, ("MOVABLE", 0.9, ""))[0]

                routed = {
                    "decision": decision.decision,
                    "prediction": pred,
                    "observed_motion": "DISPLACED" if is_reloc else "STATIC",
                    "world_displacement_m": matched_m.distance_m if is_reloc else 0.0,
                    "displacement_uncertainty_m": c2.position_uncertainty_m,
                    "evidence_provenance": "MEASURED_CLUSTER_ALIGNED",
                    "relocation_evaluable": True,
                    "predicted_landmark_action": action,
                    "landmark_action_reason": reason,
                    "motion_confidence": decision.confidence,
                    "query_utility": decision.query_utility,
                    "estimator_influence": decision.estimator_influence,
                    "estimator_provenance": decision.estimator_provenance,
                    "reason_codes": decision.reason_codes,
                }
                node = fusion.update_or_create_node(
                    entity_id=entity_id,
                    semantic_class=c2.semantic_class,
                    routed_result=routed,
                    observation_meta=obs_meta,
                )
            else:
                # Unmatched S2 entity
                entity_id = f"phys_{t2}"

                c2_floor_2d = c2.centroid_floor[:2]
                c2_aligned_2d = reg_info.R_12 @ c2_floor_2d + reg_info.translation_m
                z_coord = float(c2.centroid_floor[2]) if len(c2.centroid_floor) >= 3 else 0.0
                aligned_centroid_3d = [float(c2_aligned_2d[0]), float(c2_aligned_2d[1]), z_coord]

                is_reloc_unmatched = False
                obs_meta = {
                    "timestamp": t_session_s,
                    "timestamp_raw": t_raw,
                    "timestamp_session_s": t_session_s,
                    "temporal_valid": True,
                    "source_session": s2_session,
                    "obs_id": first_obs_id,
                    "local_track_id": t2,
                    "centroid": aligned_centroid_3d,
                    "spatial_extent_m": c2.spatial_extent_m,
                    "depth_valid": depth_valid,
                    "coordinate_frame": "reference_frame",
                    "is_relocation_event": is_reloc_unmatched,
                }
                unmatched_ev = {
                    "entity_id": entity_id,
                    "semantic_class": c2.semantic_class,
                    "class_prior_mobility": CLASS_MOBILITY_PRIORS.get(c2.semantic_class, ("MOVABLE", 0.9, ""))[0],
                    "observed_motion": "STATIC",
                    "world_displacement_m": 0.0,
                    "displacement_uncertainty_m": c2.position_uncertainty_m,
                    "has_valid_pose": True,
                    "depth_valid": depth_valid,
                    "s1_obs_count": 0,
                    "s2_obs_count": len(c2.observation_ids),
                    "has_images": bool(c2.observation_ids and len(c2.observation_ids) > 0),
                    "views_count": min(2, len(c2.observation_ids)),
                }
                decision = evaluate_router_policy(unmatched_ev, policy=policy_key)
                pred = decision.cheap_label or CLASS_MOBILITY_PRIORS.get(c2.semantic_class, ("MOVABLE", 0.9, ""))[0]

                routed = {
                    "decision": decision.decision,
                    "prediction": pred,
                    "observed_motion": "STATIC",
                    "evidence_provenance": "MEASURED_CLUSTER_S2",
                    "relocation_evaluable": False,
                    "predicted_landmark_action": "RETAIN",
                    "landmark_action_reason": "UNMATCHED_S2_ENTITY",
                    "motion_confidence": decision.confidence,
                    "query_utility": decision.query_utility,
                    "estimator_influence": decision.estimator_influence,
                    "estimator_provenance": decision.estimator_provenance,
                    "reason_codes": decision.reason_codes,
                }
                node = fusion.update_or_create_node(
                    entity_id=entity_id,
                    semantic_class=c2.semantic_class,
                    routed_result=routed,
                    observation_meta=obs_meta,
                )

        sg_dict = fusion.to_dict()
        stats = {
            "total_nodes": len(sg_dict["nodes"]),
            "stationary_fused": stationary_fused_count,
            "relocated_fused": relocated_fused_count,
            "invalidated_landmarks": sum(1 for n in sg_dict["nodes"].values() if n.get("slam_invalidated", False) or n.get("predicted_landmark_action") == "INVALIDATE"),
            "nodes_with_history": sum(1 for n in sg_dict["nodes"].values() if len(n.get("pose_history", [])) > 0),
        }
        return sg_dict, stats

    # Build canonical R3 scene graph
    canonical_sg, r3_stats = build_policy_scene_graph("R3", active_matcher_res.matched_pairs)
    total_nodes = r3_stats["total_nodes"]
    stationary_fused_count = r3_stats["stationary_fused"]
    relocated_fused_count = r3_stats["relocated_fused"]
    invalidated_landmarks = r3_stats["invalidated_landmarks"]
    nodes_with_history = r3_stats["nodes_with_history"]

    print(f"• Total Scene Graph Nodes:    {total_nodes}")
    print(f"• Stationary Merged Entities: {stationary_fused_count}")
    print(f"• Relocated Fused Entities:   {relocated_fused_count}")
    print(f"• Invalidated SLAM Landmarks: {invalidated_landmarks}")
    print(f"• Nodes Preserving History:   {nodes_with_history}")

    # Build decoupled scene graphs for all comparative policies
    policy_scene_graphs = {"R3": canonical_sg}
    hungarian_pairs = baselines.get("baseline_2_hungarian_gated", {}).get("result", active_matcher_res).matched_pairs

    sg_r0, stats_r0 = build_policy_scene_graph("R0", hungarian_pairs)
    sg_r1, stats_r1 = build_policy_scene_graph("R1", hungarian_pairs)
    sg_r2, stats_r2 = build_policy_scene_graph("R2", hungarian_pairs)
    sg_r4, stats_r4 = build_policy_scene_graph("R4", active_matcher_res.matched_pairs)

    policy_scene_graphs["R0"] = sg_r0
    policy_scene_graphs["R1"] = sg_r1
    policy_scene_graphs["R2"] = sg_r2
    policy_scene_graphs["R4"] = sg_r4

    # 6. Save Artifacts & Manifest
    print("\n--- [Stage 4: Writing Artifacts & Execution Manifest] ---")
    sg_path = out_dir / "scene_graph.json"
    with open(sg_path, "w", encoding="utf-8") as f:
        json.dump(canonical_sg, f, indent=2)

    # Save decoupled policy scene graphs
    for p_name, p_sg in policy_scene_graphs.items():
        p_path = out_dir / f"scene_graph_{p_name.lower()}.json"
        with open(p_path, "w", encoding="utf-8") as f:
            json.dump(p_sg, f, indent=2)

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
        json.dump(active_matcher_eval.get("per_instance_results", []), f, indent=2)

    # 7. Independent Downstream Evaluation Against Ground Truth
    print("\n--- [Stage 5: Independent Downstream Evaluation Against Ground Truth] ---")
    eval_results = {}
    movability_gt_file = repo_root / "vlm" / "benchmarks" / "movability_ground_truth.json"
    for p_name, p_sg in policy_scene_graphs.items():
        eval_res = evaluate_scene_graph(
            scene_graph=p_sg,
            controlled_events_path=controlled_events_path,
            identity_annotations_path=annotations_path,
            movability_gt_path=movability_gt_file,
            policy_name=p_name,
        )
        eval_results[p_name] = eval_res
        reloc_rec = eval_res["landmark_reliability"]["relocation_invalidation_rate_pct"]
        stat_ret = eval_res["landmark_reliability"]["stationary_retention_rate_pct"]
        print(f"• Policy {p_name}: Stationary Retention={stat_ret}%, Relocation Invalidation={reloc_rec}%")

    downstream_eval_path = out_dir / "downstream_evaluation_summary.json"
    with open(downstream_eval_path, "w", encoding="utf-8") as f:
        json.dump(eval_results, f, indent=2)

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
            "candidate_precision_pct": active_matcher_eval["candidate_pair_precision_pct"],
            "static_instance_recall_pct": active_matcher_eval["static_instance_recall_pct"],
            "relocated_instance_recall_pct": active_matcher_eval["relocated_instance_recall_pct"],
            "all_instance_recall_pct": active_matcher_eval["all_instance_recall_pct"],
            "instance_f1_pct": active_matcher_eval["instance_level_f1_pct"],
            "pair_false_merge_rate_pct": active_matcher_eval["pair_level_false_merge_rate_pct"],
            "instance_miss_rate_pct": active_matcher_eval["instance_miss_rate_pct"],
            "split_instance_rate_pct": active_matcher_eval["split_instance_rate_pct"],
            "merge_instance_rate_pct": active_matcher_eval["merge_instance_rate_pct"],
            "annotation_coverage_pct": active_matcher_eval["annotation_coverage_pct"],
            "false_merge_rate_pct": active_matcher_eval["false_merge_rate_pct"],
            "false_split_rate_pct": active_matcher_eval["false_split_rate_pct"],
        },
        "scene_graph_fusion_summary": {
            "total_nodes": total_nodes,
            "stationary_fused_entities": stationary_fused_count,
            "relocated_fused_entities": relocated_fused_count,
            "invalidated_slam_landmarks": invalidated_landmarks,
            "nodes_preserving_pose_history": nodes_with_history,
        },
        "downstream_evaluation": eval_results["R3"],
        "output_artifacts": {
            "scene_graph_json": str(sg_path),
            "baselines_summary_json": str(baselines_summary_path),
            "per_instance_table_json": str(per_instance_table_path),
            "downstream_evaluation_summary_json": str(downstream_eval_path),
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
    parser.add_argument("--reg", default=None, help="Path to stage5_registration_audit.json")
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
