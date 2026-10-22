"""Stage 5: Integrated Pipeline for Non-VLM Pre-Routing Evidence Generation.

Coordinates:
- Source observation input resolution and strict validation (zero silent loss).
- Geometric evidence extraction (MapPoints, point_to_bbox consistency, depth uncertainty).
- Temporal evidence extraction (world-relative displacement, ego-motion compensation, persistence).
- Semantic evidence extraction (YOLOE detections, Shannon entropy, class priors).
- Exports timestamp-aligned records strictly into vlm/outputs/ or designated run directory.
"""

from __future__ import annotations

import hashlib
import json
import time
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from vlm.src.stage5_validation import SourceRecordValidator
from vlm.src.geometric_evidence import GeometricEvidenceExtractor, GeometricEvidenceRecord, SessionGeometricConfig
from vlm.src.temporal_evidence import TemporalEvidenceExtractor, TemporalEvidenceRecord
from vlm.src.semantic_evidence import SemanticEvidenceExtractor, SemanticEvidenceRecord


def compute_file_sha256(path: Path) -> str:
    """Compute SHA-256 hash of a file."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def resolve_observations_input(
    obs_jsonl_path: Optional[Path],
    repo_root: Path,
) -> Tuple[Path, str, str]:
    """Resolve authoritative observations input file with explicit provenance policy.

    Returns:
        (resolved_path, sha256_hash, resolution_reason)
    """
    if obs_jsonl_path is not None:
        p = Path(obs_jsonl_path).resolve()
        if not p.exists():
            raise FileNotFoundError(f"Explicitly specified observations input file not found: {p}")
        sha = compute_file_sha256(p)
        return p, sha, "EXPLICIT_USER_OVERRIDE"

    # Canonical default path
    canonical_cand = repo_root / "vlm" / "data" / "observations" / "object_observations.jsonl"
    if canonical_cand.exists():
        sha = compute_file_sha256(canonical_cand)
        return canonical_cand, sha, "CANONICAL_DEFAULT"

    # Fallback search path candidates with explicit logging
    fallbacks = [
        (repo_root / "vlm" / "outputs" / "object_observations.jsonl", "STAGE3_OUTPUT_FALLBACK"),
        (repo_root / "vlm" / "data" / "semantic_mapping_my_desk_recording_01" / "object_observations.jsonl", "S1_DATA_FALLBACK"),
        (repo_root / "outputs" / "semantic_mapping_my_desk_recording_01" / "object_observations.jsonl", "LEGACY_OUTPUTS_FALLBACK"),
    ]
    for cand_path, reason in fallbacks:
        if cand_path.exists():
            sha = compute_file_sha256(cand_path)
            print(f"[Stage 5 Input Resolution] Using {reason}: {cand_path} (SHA-256: {sha[:16]}...)")
            return cand_path, sha, reason

    attempted = [str(canonical_cand)] + [str(p) for p, _ in fallbacks]
    raise FileNotFoundError(
        "No valid observations JSONL file found. Attempted the following paths in order:\n  - "
        + "\n  - ".join(attempted)
    )


def run_stage5_pipeline(
    obs_jsonl_path: Optional[Path] = None,
    mappoints_ply_path: Optional[Path] = None,
    output_dir: Optional[Path] = None,
    limit: Optional[int] = None,
    strict_validation: bool = True,
) -> Dict[str, Any]:
    """Execute complete Stage 5 evidence pipeline across all object observations.

    Args:
        obs_jsonl_path: Path to object_observations.jsonl. If None, resolves canonical input.
        mappoints_ply_path: Path to MapPoints.ply.
        output_dir: Output directory (strictly under vlm/outputs/).
        limit: Optional maximum number of observations to process (for quick tests).
        strict_validation: If True, fail on malformed JSON; otherwise quarantine.

    Returns:
        Structured run result dictionary with input SHA-256, counts, session configs, and file paths.
    """
    repo_root = Path(__file__).resolve().parent.parent.parent

    # 1. Authoritative Input Resolution
    chosen_obs_path, obs_sha256, resolution_reason = resolve_observations_input(obs_jsonl_path, repo_root)
    run_id = f"stage5_run_{int(time.time())}_{uuid.uuid4().hex[:8]}"

    if output_dir is None:
        output_dir = repo_root / "vlm" / "outputs"
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    geo_out_path = output_dir / "stage5_geometric_evidence.jsonl"
    temp_out_path = output_dir / "stage5_temporal_evidence.jsonl"
    sem_out_path = output_dir / "stage5_semantic_evidence.jsonl"
    fused_out_path = output_dir / "stage5_fused_evidence.jsonl"
    manifest_out_path = output_dir / "stage5_run_manifest.json"
    rejection_out_path = output_dir / "stage5_rejections.jsonl"

    print(f"Initializing multi-session GeometricEvidenceExtractor (repo: {repo_root})")
    geo_extractor = GeometricEvidenceExtractor(mappoints_ply_path, repo_root=repo_root)
    temp_extractor = TemporalEvidenceExtractor()
    sem_extractor = SemanticEvidenceExtractor()
    validator = SourceRecordValidator(strict_mode=strict_validation)

    print(f"Reading observations from: {chosen_obs_path} [{resolution_reason}] (SHA-256: {obs_sha256[:16]}...)")
    t_start = time.time()

    geo_records: List[Dict[str, Any]] = []
    temp_records: List[Dict[str, Any]] = []
    sem_records: List[Dict[str, Any]] = []
    fused_records: List[Dict[str, Any]] = []

    with open(chosen_obs_path, "r", encoding="utf-8") as f:
        for line_no, raw_line in enumerate(f, start=1):
            obs, mod_status, rejection = validator.validate_line(raw_line, line_no)
            if obs is None:
                continue

            # 1. Geometric evidence extraction
            geo_rec = geo_extractor.extract_observation_geometry(obs)
            # 2. Temporal evidence extraction
            temp_rec = temp_extractor.extract_observation_temporal(obs)
            # 3. Semantic evidence extraction
            sem_rec = sem_extractor.extract_observation_semantics(obs)

            g_dict = geo_rec.to_dict()
            t_dict = temp_rec.to_dict()
            s_dict = sem_rec.to_dict()

            # Modality-specific validity
            geometry_valid = bool(geo_rec.has_valid_evidence)
            temporal_valid = bool(temp_rec.has_valid_evidence)
            semantic_valid = bool(
                s_dict.get("yoloe_class_name") not in ["unknown", "UNKNOWN", ""]
                and s_dict.get("class_prior_mobility") != "UNKNOWN"
            )

            # Strict evidence completeness: all modalities valid and depth + pose + floor centroid verified
            evidence_complete = bool(
                geometry_valid
                and temporal_valid
                and semantic_valid
                and mod_status.has_valid_pose
                and mod_status.has_valid_floor_centroid
            )

            # Define top-level has_valid_evidence consistently: geometry and temporal evidence valid
            top_level_valid = bool(geometry_valid and temporal_valid)

            # Fused comprehensive evidence record
            fused_dict = {
                "obs_id": obs.get("obs_id", ""),
                "frame_index": int(obs.get("frame_index", 0)),
                "timestamp": float(obs.get("timestamp", 0.0)),
                "entity_id": obs.get("global_entity_id", ""),
                "track_id": obs.get("local_track_id", ""),
                "source_session": obs.get("source_session", ""),
                "has_valid_evidence": top_level_valid,
                "modality_validity": {
                    "geometry_valid": geometry_valid,
                    "temporal_valid": temporal_valid,
                    "semantic_valid": semantic_valid,
                    "evidence_complete": evidence_complete,
                },
                # Geometry branch
                "geometry": {
                    "depth_valid": g_dict["depth_valid"],
                    "measured_depth_m": g_dict["measured_depth_m"],
                    "depth_uncertainty_sigma_m": g_dict["depth_uncertainty_sigma_m"],
                    "point_to_bbox_valid": g_dict["point_to_bbox_valid"],
                    "point_to_bbox_distance_px": g_dict["point_to_bbox_distance_px"],
                    "point_to_bbox_inlier_ratio": g_dict["point_to_bbox_inlier_ratio"],
                    "inlier_ratio": g_dict["point_to_bbox_inlier_ratio"],
                    "mappoints_candidate_count": g_dict["mappoints_candidate_count"],
                    "mappoints_inlier_count": g_dict["mappoints_inlier_count"],
                    "observation_points_count": g_dict.get("observation_points_count", 0),
                    "landmark_support": g_dict["mappoints_candidate_count"],
                    "geometric_consistency_score": g_dict["geometric_consistency_score"],
                    "provenance": g_dict["provenance"],
                    "feature_reprojection_residual_valid": False,
                    "mean_residual_px": None,  # Strictly unaliased from bbox distance
                },
                # Temporal branch
                "temporal": {
                    "current_motion": t_dict["current_motion"],
                    "world_displacement_m": t_dict["world_displacement_m"],
                    "net_displacement_m": t_dict.get("net_displacement_m", t_dict["world_displacement_m"]),
                    "step_displacement_m": t_dict.get("step_displacement_m", 0.0),
                    "cumulative_path_length_m": t_dict.get("cumulative_path_length_m", 0.0),
                    "world_displacement_sigma_m": t_dict["world_displacement_sigma_m"],
                    "camera_ego_displacement_m": t_dict["camera_ego_displacement_m"],
                    "ego_motion_compensation_ratio": t_dict["ego_motion_compensation_ratio"],
                    "visibility_duration_s": t_dict["visibility_duration_s"],
                    "observation_count": t_dict["observation_count"],
                    "occlusion_events_count": t_dict["occlusion_events_count"],
                    "observation_gap_events_count": t_dict.get("observation_gap_events_count", t_dict["occlusion_events_count"]),
                    "is_reappearance_after_occlusion": t_dict["is_reappearance_after_occlusion"],
                    "is_reappearance_after_gap": t_dict.get("is_reappearance_after_gap", t_dict["is_reappearance_after_occlusion"]),
                    "temporal_persistence_score": t_dict["temporal_persistence_score"],
                    "missing_reason": t_dict.get("missing_reason"),
                },
                # Semantic branch
                "semantics": {
                    "yoloe_class": s_dict["yoloe_class_name"],
                    "detector_confidence": s_dict["detector_confidence"],
                    "bbox_area_px": s_dict.get("bbox_area_px"),
                    "class_prior_mobility": s_dict["class_prior_mobility"],
                    "class_prior_confidence": s_dict["class_prior_confidence"],
                    "semantic_entropy": s_dict["semantic_entropy"],
                    "semantic_purity": s_dict["semantic_purity"],
                    "is_unknown_class": s_dict.get("is_unknown_class", False),
                    "detection_valid": s_dict.get("detection_valid", True),
                    "class_known_to_prior": s_dict.get("class_known_to_prior", True),
                    "cheap_prior_description": s_dict["cheap_prior_description"],
                },
                # Router readiness summary
                "router_summary": {
                    "evidence_complete": evidence_complete,
                    "requires_vlm_query": bool(
                        not evidence_complete
                        or t_dict["current_motion"] in ["MOVING", "DISPLACED", "UNCERTAIN"]
                        or s_dict["semantic_entropy"] > 0.45
                        or g_dict["geometric_consistency_score"] < 0.50
                    ),
                    "cheap_verdict": (
                        "STATIONARY_" + s_dict["class_prior_mobility"]
                        if (t_dict["current_motion"] == "STATIC" and evidence_complete)
                        else "UNSTABLE"
                    ),
                },
            }

            geo_records.append(g_dict)
            temp_records.append(t_dict)
            sem_records.append(s_dict)
            fused_records.append(fused_dict)

            if limit and len(fused_records) >= limit:
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

    # Reconciliation check
    recon = validator.reconcile_summary()
    if not recon["reconciliation_passed"]:
        raise ValueError(f"Stage 5 reconciliation failed: {recon}")

    # Write rejections report if any
    if recon["rejections"]:
        with open(rejection_out_path, "w", encoding="utf-8") as f:
            for rej in recon["rejections"]:
                f.write(json.dumps(rej) + "\n")

    elapsed = time.time() - t_start

    session_cfgs = {
        sid: cfg.to_dict() for sid, cfg in geo_extractor.session_configs.items()
    }

    run_manifest = {
        "run_id": run_id,
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "input_obs_path": str(chosen_obs_path),
        "input_obs_sha256": obs_sha256,
        "input_resolution_reason": resolution_reason,
        "total_lines_read": recon["total_lines_read"],
        "total_non_empty_lines": recon["total_non_empty_lines"],
        "accepted_count": recon["total_accepted"],
        "rejected_count": recon["total_rejected"],
        "reconciliation_passed": recon["reconciliation_passed"],
        "unique_composite_ids_count": recon["unique_composite_ids_count"],
        "output_dir": str(output_dir),
        "geometric_records_file": str(geo_out_path),
        "temporal_records_file": str(temp_out_path),
        "semantic_records_file": str(sem_out_path),
        "fused_records_file": str(fused_out_path),
        "rejection_report_file": str(rejection_out_path) if recon["rejections"] else None,
        "session_configs": session_cfgs,
        "elapsed_seconds": round(elapsed, 2),
    }

    with open(manifest_out_path, "w", encoding="utf-8") as f:
        json.dump(run_manifest, f, indent=2)

    print(
        f"Stage 5 evidence pipeline completed: {recon['total_accepted']} accepted, "
        f"{recon['total_rejected']} rejected in {elapsed:.2f}s."
    )

    return run_manifest


if __name__ == "__main__":
    res = run_stage5_pipeline()
    print("Stage 5 Execution Succeeded:", res["run_id"])
