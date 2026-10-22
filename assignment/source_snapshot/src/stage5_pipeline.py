"""Stage 5: Integrated Pipeline for Non-VLM Pre-Routing Evidence Generation.

Coordinates:
- Geometric evidence extraction (MapPoints, point_to_bbox consistency, depth uncertainty, inliers).
- Temporal evidence extraction (world-relative displacement, ego-motion compensation, persistence).
- Semantic evidence extraction (YOLOE detections, Shannon entropy, class priors).
- Exports timestamp-aligned records strictly into vlm/outputs/.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from vlm.src.geometric_evidence import GeometricEvidenceExtractor, GeometricEvidenceRecord
from vlm.src.temporal_evidence import TemporalEvidenceExtractor, TemporalEvidenceRecord
from vlm.src.semantic_evidence import SemanticEvidenceExtractor, SemanticEvidenceRecord


def run_stage5_pipeline(
    obs_jsonl_path: Optional[Path] = None,
    mappoints_ply_path: Optional[Path] = None,
    output_dir: Optional[Path] = None,
    limit: Optional[int] = None,
) -> Dict[str, Any]:
    """Execute complete Stage 5 evidence pipeline across all object observations.

    Args:
        obs_jsonl_path: Path to object_observations.jsonl.
        mappoints_ply_path: Path to MapPoints.ply.
        output_dir: Output directory (strictly under vlm/outputs/).
        limit: Optional maximum number of observations to process (for quick tests).

    Returns:
        Summary dictionary with record counts and file paths.
    """
    repo_root = Path(__file__).resolve().parent.parent.parent

    # Locate inputs
    if obs_jsonl_path is None:
        cand = [
            repo_root / "vlm" / "outputs" / "object_observations.jsonl",
            repo_root / "outputs" / "semantic_mapping_my_desk_recording_01" / "object_observations.jsonl",
        ]
        for c in cand:
            if c.exists():
                obs_jsonl_path = c
                break

    if mappoints_ply_path is None:
        cand_ply = [
            repo_root / "outputs" / "semantic_mapping_my_desk_recording_01" / "trajectory" / "MapPoints.ply",
            repo_root / "outputs" / "semantic_mapping_my_desk_recording_02" / "trajectory" / "MapPoints.ply",
        ]
        for c in cand_ply:
            if c.exists():
                mappoints_ply_path = c
                break

    if output_dir is None:
        output_dir = repo_root / "vlm" / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)

    geo_out_path = output_dir / "stage5_geometric_evidence.jsonl"
    temp_out_path = output_dir / "stage5_temporal_evidence.jsonl"
    sem_out_path = output_dir / "stage5_semantic_evidence.jsonl"
    fused_out_path = output_dir / "stage5_fused_evidence.jsonl"

    print(f"Initializing multi-session GeometricEvidenceExtractor (repo: {repo_root})")
    geo_extractor = GeometricEvidenceExtractor(mappoints_ply_path, repo_root=repo_root)
    temp_extractor = TemporalEvidenceExtractor()
    sem_extractor = SemanticEvidenceExtractor()

    print(f"Reading observations from: {obs_jsonl_path}")
    count = 0
    t_start = time.time()

    geo_records: List[Dict[str, Any]] = []
    temp_records: List[Dict[str, Any]] = []
    sem_records: List[Dict[str, Any]] = []
    fused_records: List[Dict[str, Any]] = []

    with open(obs_jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obs = json.loads(line)
            except json.JSONDecodeError:
                continue

            # 1. Geometric evidence
            geo_rec = geo_extractor.extract_observation_geometry(obs)
            # 2. Temporal evidence
            temp_rec = temp_extractor.extract_observation_temporal(obs)
            # 3. Semantic evidence
            sem_rec = sem_extractor.extract_observation_semantics(obs)

            g_dict = geo_rec.to_dict()
            t_dict = temp_rec.to_dict()
            s_dict = sem_rec.to_dict()

            # Dynamic evidence completeness check (never hardcoded True)
            depth_valid = bool(g_dict.get("depth_valid", False))
            pose_valid = bool(obs.get("has_valid_pose", False) and obs.get("camera_pose") is not None)
            cfloor_valid = bool(obs.get("centroid_floor") is not None and len(obs.get("centroid_floor")) >= 3)
            sem_valid = bool(s_dict.get("yoloe_class_name") != "unknown" and s_dict.get("class_prior_mobility") != "UNKNOWN")
            evidence_complete = bool(depth_valid and pose_valid and cfloor_valid and sem_valid)

            # Fused comprehensive evidence record for the router
            fused_dict = {
                "obs_id": obs.get("obs_id", ""),
                "frame_index": int(obs.get("frame_index", 0)),
                "timestamp": float(obs.get("timestamp", 0.0)),
                "entity_id": obs.get("global_entity_id", ""),
                "track_id": obs.get("local_track_id", ""),
                "source_session": obs.get("source_session", ""),
                "has_valid_evidence": geo_rec.has_valid_evidence and temp_rec.has_valid_evidence,
                # Geometry branch
                "geometry": {
                    "depth_valid": g_dict["depth_valid"],
                    "measured_depth_m": g_dict["measured_depth_m"],
                    "depth_uncertainty_sigma_m": g_dict["depth_uncertainty_sigma_m"],
                    "point_to_bbox_valid": g_dict["point_to_bbox_valid"],
                    "point_to_bbox_distance_px": g_dict["point_to_bbox_distance_px"],
                    "point_to_bbox_inlier_ratio": g_dict["point_to_bbox_inlier_ratio"],
                    "inlier_ratio": g_dict["inlier_ratio"],
                    "mean_residual_px": g_dict["mean_reprojection_residual_px"],
                    "landmark_support": g_dict["landmark_support_count"],
                    "geometric_consistency_score": g_dict["geometric_consistency_score"],
                    "provenance": g_dict["provenance"],
                },
                # Temporal branch
                "temporal": {
                    "current_motion": t_dict["current_motion"],
                    "world_displacement_m": t_dict["world_displacement_m"],
                    "world_displacement_sigma_m": t_dict["world_displacement_sigma_m"],
                    "camera_ego_displacement_m": t_dict["camera_ego_displacement_m"],
                    "ego_motion_compensation_ratio": t_dict["ego_motion_compensation_ratio"],
                    "visibility_duration_s": t_dict["visibility_duration_s"],
                    "observation_count": t_dict["observation_count"],
                    "occlusion_events_count": t_dict["occlusion_events_count"],
                    "is_reappearance_after_occlusion": t_dict["is_reappearance_after_occlusion"],
                    "temporal_persistence_score": t_dict["temporal_persistence_score"],
                    "missing_reason": t_dict.get("missing_reason"),
                },
                # Semantic branch
                "semantics": {
                    "yoloe_class": s_dict["yoloe_class_name"],
                    "detector_confidence": s_dict["detector_confidence"],
                    "class_prior_mobility": s_dict["class_prior_mobility"],
                    "class_prior_confidence": s_dict["class_prior_confidence"],
                    "semantic_entropy": s_dict["semantic_entropy"],
                    "semantic_purity": s_dict["semantic_purity"],
                    "cheap_prior_description": s_dict["cheap_prior_description"],
                },
                # Router readiness summary
                "router_summary": {
                    "evidence_complete": evidence_complete,
                    "requires_vlm_query": (
                        t_dict["current_motion"] in ["MOVING", "DISPLACED"]
                        or s_dict["semantic_entropy"] > 0.45
                        or g_dict["geometric_consistency_score"] < 0.50
                    ),
                    "cheap_verdict": (
                        "STATIONARY_" + s_dict["class_prior_mobility"]
                        if t_dict["current_motion"] == "STATIC"
                        else "UNSTABLE"
                    ),
                },
            }

            geo_records.append(g_dict)
            temp_records.append(t_dict)
            sem_records.append(s_dict)
            fused_records.append(fused_dict)

            count += 1
            if limit and count >= limit:
                break

    # Write separate timestamp-aligned JSONL files
    print(f"Writing {len(geo_records)} geometric records to {geo_out_path}")
    with open(geo_out_path, "w", encoding="utf-8") as f:
        for r in geo_records:
            f.write(json.dumps(r) + "\n")

    print(f"Writing {len(temp_records)} temporal records to {temp_out_path}")
    with open(temp_out_path, "w", encoding="utf-8") as f:
        for r in temp_records:
            f.write(json.dumps(r) + "\n")

    print(f"Writing {len(sem_records)} semantic records to {sem_out_path}")
    with open(sem_out_path, "w", encoding="utf-8") as f:
        for r in sem_records:
            f.write(json.dumps(r) + "\n")

    print(f"Writing {len(fused_records)} fused records to {fused_out_path}")
    with open(fused_out_path, "w", encoding="utf-8") as f:
        for r in fused_records:
            f.write(json.dumps(r) + "\n")

    elapsed = time.time() - t_start
    print(f"Stage 5 evidence pipeline completed: {count} records in {elapsed:.2f}s ({count/elapsed:.1f} rec/s)")

    return {
        "total_records": count,
        "elapsed_seconds": round(elapsed, 2),
        "geometric_records_file": str(geo_out_path),
        "temporal_records_file": str(temp_out_path),
        "semantic_records_file": str(sem_out_path),
        "fused_records_file": str(fused_out_path),
    }


if __name__ == "__main__":
    res = run_stage5_pipeline()
    print("Done:", res)
