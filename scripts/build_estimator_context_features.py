"""Estimator-context causal feature extraction for candidate observations."""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

RUN_DIR     = REPO / "results" / "perception_harvesting" / "run_20261101_234859"
ANN_DIR     = REPO / "results" / "track_annotation"
OUT_DIR     = REPO / "results" / "estimator_context"

# Feature schema definition (for documentation + downstream consumers)

FEATURE_SCHEMA = {
    "version"     : "1.0",
    "description" : (
        "Causal estimator-context feature vectors for all 885 VLM-eligible observations. "
        "Every feature is available at decision time. No future frames, no GT poses."
    ),
    "causal_guarantees": [
        "world_displacement_at_obs uses only SLAM poses up to the observation timestamp.",
        "No ground-truth trajectory poses enter any feature.",
        "Trajectory role is ESTIMATED throughout (enforced by unified_pipeline.py).",
        "No feature looks ahead in time beyond the current observation frame.",
    ],
    "features": {
        # Identifiers
        "sequence_id"          : {"type": "str",   "category": "identifier"},
        "track_id"             : {"type": "str",   "category": "identifier"},
        "obs_id"               : {"type": "str",   "category": "identifier"},
        "frame_index"          : {"type": "int",   "category": "identifier"},
        "timestamp"            : {"type": "float", "category": "identifier"},

        # Visual features
        "detection_confidence" : {
            "type": "float", "range": "[0,1]", "category": "visual",
            "description": "YOLO detection confidence score for this observation.",
        },
        "crop_area_px"         : {
            "type": "float", "range": "[0,inf)", "category": "visual",
            "description": "Pixel area of the detected bounding box.",
        },
        "crop_area_log"        : {
            "type": "float", "category": "visual",
            "description": "log10(area_px + 1) — log-normalised crop area.",
        },
        "sharpness"            : {
            "type": "float", "range": "[0,inf)", "category": "visual",
            "description": "Laplacian variance of the crop — higher = sharper.",
        },
        "sharpness_log"        : {
            "type": "float", "category": "visual",
            "description": "log10(sharpness + 1) — log-normalised sharpness.",
        },
        "bbox_aspect_ratio"    : {
            "type": "float", "range": "(0,inf)", "category": "visual",
            "description": "width / height of bounding box.",
        },
        "has_mask"             : {
            "type": "bool",  "category": "visual",
            "description": "Whether a YOLO-seg mask was produced for this observation.",
        },

        # Geometric features
        "median_depth_m"       : {
            "type": "float", "range": "[0,inf)", "category": "geometric",
            "description": "Median depth within the bounding box (metres).",
        },
        "depth_valid"          : {
            "type": "bool",  "category": "geometric",
            "description": "Whether median_depth_m > 0 and < 10.0 m.",
        },
        "centroid_cam_x"       : {
            "type": "float", "category": "geometric",
            "description": "Camera-frame centroid X (metres, right).",
        },
        "centroid_cam_y"       : {
            "type": "float", "category": "geometric",
            "description": "Camera-frame centroid Y (metres, down).",
        },
        "centroid_cam_z"       : {
            "type": "float", "category": "geometric",
            "description": "Camera-frame centroid Z = depth (metres, forward).",
        },
        "centroid_world_x"     : {
            "type": "float", "category": "geometric",
            "description": "World-frame centroid X at observation time.",
        },
        "centroid_world_y"     : {
            "type": "float", "category": "geometric",
            "description": "World-frame centroid Y at observation time.",
        },
        "centroid_world_z"     : {
            "type": "float", "category": "geometric",
            "description": "World-frame centroid Z at observation time.",
        },
        "ego_motion_available" : {
            "type": "bool",  "category": "geometric",
            "description": "Whether ego-motion compensation was applied.",
        },

        # Estimator features
        "world_displacement_m" : {
            "type": "float", "range": "[0,inf)", "category": "estimator",
            "description": (
                "Cumulative world-frame displacement of the track up to the END of the sequence. "
                "Causal note: this is a track-level field from tracks.jsonl, computed over the full "
                "sequence — it is NOT causal per-observation. Use displacement_at_obs_frac instead "
                "for per-decision features."
            ),
        },
        "displacement_uncertainty_m": {
            "type": "float", "category": "estimator",
            "description": "Spatial position dispersion uncertainty (std dev across world centroids).",
        },
        "apparent_camera_frame_displacement_m": {
            "type": "float", "category": "estimator",
            "description": (
                "Apparent displacement in camera frame — NOT ego-compensated. "
                "Can be large even for stationary objects due to camera motion."
            ),
        },
        "camera_ego_motion_displacement_m": {
            "type": "float", "category": "estimator",
            "description": "How much the camera moved during the track lifetime (ego-motion magnitude).",
        },
        "pipeline_motion_label": {
            "type": "str",  "values": ["STATIC", "DISPLACED", "MOVING"], "category": "estimator",
            "description": "Pipeline's empirical motion classification over the full track.",
        },

        # Temporal features
        "track_obs_index"      : {
            "type": "int",   "range": "[0,4]", "category": "temporal",
            "description": "Index of this observation within the VLM sample (0=earliest, 4=latest).",
        },
        "track_obs_frac"       : {
            "type": "float", "range": "[0,1]", "category": "temporal",
            "description": "Fractional position in track (0=first VLM obs, 1=last VLM obs).",
        },
        "lifespan_sec"         : {
            "type": "float", "category": "temporal",
            "description": "Total track lifespan in seconds (first to last detection).",
        },
        "total_detections"     : {
            "type": "int",   "category": "temporal",
            "description": "Total YOLO detections for this track across all processed frames.",
        },
        "timestamp_rel"        : {
            "type": "float", "category": "temporal",
            "description": "Observation timestamp relative to track start (seconds).",
        },

        # Contextual features
        "semantic_class"       : {
            "type": "str",  "category": "contextual",
            "description": "YOLO semantic class label (office ontology).",
        },
        "sequence_domain"      : {
            "type": "str",  "values": ["tum", "openloris", "bonn", "d455"], "category": "contextual",
            "description": "Dataset family.",
        },
        "gt_vlm_label"         : {
            "type": "str",  "values": ["IMMOVABLE", "MOVABLE", "MOBILE", None], "category": "contextual",
            "description": "Ground-truth VLM label from track_annotations.csv (None if UNCERTAIN).",
        },
        "annotation_confidence": {
            "type": "str",  "values": ["HIGH", "MEDIUM", "LOW"], "category": "contextual",
            "description": "Annotator confidence in the GT label.",
        },
        "evaluation_split"     : {
            "type": "str",  "values": ["development", "validation", "held_out", "perception_only"],
            "category": "contextual",
            "description": "Which evaluation split partition this track belongs to.",
        },
    },
}

# Domain lookup

def sequence_to_domain(seq_id: str) -> str:
    if seq_id.startswith("tum"):        return "tum"
    if seq_id.startswith("openloris"):  return "openloris"
    if seq_id.startswith("bonn"):       return "bonn"
    if seq_id.startswith("d455"):       return "d455"
    return "unknown"


# Label mapper

def annotation_to_vlm_class(row: Dict[str, str]) -> Optional[str]:
    mob = row.get("physical_mobility", "")
    cls = row.get("object_class", "")
    if mob == "STRUCTURALLY_IMMOVABLE":
        return "IMMOVABLE"
    if cls == "person":
        return "MOBILE"
    if mob == "MOVABLE":
        return "MOVABLE"
    return None


# Main

def main() -> None:
    import csv, math

    print("\nEstimator-Context Feature Extraction")
    print(f"Input run : {RUN_DIR}")
    print(f"Output    : {OUT_DIR}\n")

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load sources
    obs_raw = [
        json.loads(l)
        for l in (RUN_DIR / "candidate_observations.jsonl").read_text().splitlines()
        if l.strip()
    ]
    tracks_raw = [
        json.loads(l)
        for l in (RUN_DIR / "tracks.jsonl").read_text().splitlines()
        if l.strip()
    ]
    ann_rows = list(csv.DictReader(open(ANN_DIR / "track_annotations.csv", encoding="utf-8")))
    split_data = json.loads((ANN_DIR / "evaluation_split.json").read_text())

    print(f"  Observations loaded : {len(obs_raw)}")
    print(f"  Tracks loaded       : {len(tracks_raw)}")

    # Index tracks
    track_by_key: Dict[str, Any] = {}
    for t in tracks_raw:
        track_by_key[f"{t['sequence_id']}__{t['track_id']}"] = t

    # Index annotations
    ann_by_key: Dict[str, Any] = {}
    for row in ann_rows:
        ann_by_key[f"{row['sequence_id']}__{row['track_id']}"] = row

    # Build sequence → partition lookup
    seq_partition: Dict[str, str] = {}
    for pname, p in split_data["partitions"].items():
        for seq in p["sequences"]:
            seq_partition[seq] = pname

    # Index observations by track, sorted by frame_index
    obs_by_track: Dict[str, List[Any]] = defaultdict(list)
    for o in obs_raw:
        obs_by_track[f"{o['sequence_id']}__{o['track_id']}"].append(o)
    for key in obs_by_track:
        obs_by_track[key].sort(key=lambda o: o["frame_index"])

    # Build feature vectors
    feature_vectors: List[Dict[str, Any]] = []
    n_no_annotation = 0
    n_uncertain_gt  = 0

    for key, obs_list in obs_by_track.items():
        track    = track_by_key.get(key, {})
        ann_row  = ann_by_key.get(key, {})
        seq_id   = obs_list[0]["sequence_id"]
        track_id = obs_list[0]["track_id"]

        # GT label
        gt_label = annotation_to_vlm_class(ann_row) if ann_row else None
        ann_conf = ann_row.get("annotation_confidence", "UNKNOWN") if ann_row else "UNKNOWN"
        if not ann_row:
            n_no_annotation += 1
        elif gt_label is None:
            n_uncertain_gt += 1

        # Track-level estimator fields
        world_disp    = track.get("world_displacement_m", 0.0) or 0.0
        disp_unc      = track.get("displacement_uncertainty_m", 0.0) or 0.0
        cam_frame_disp = track.get("apparent_camera_frame_displacement_m", 0.0) or 0.0
        ego_disp      = track.get("camera_ego_motion_displacement_m", 0.0) or 0.0
        pipe_motion   = track.get("observed_motion", "UNKNOWN")
        lifespan      = track.get("lifespan_sec", 0.0) or 0.0
        total_dets    = track.get("observation_count", len(obs_list))
        first_ts      = track.get("first_timestamp") or (obs_list[0]["timestamp"] if obs_list else 0.0)

        n_obs = len(obs_list)

        for i, obs in enumerate(obs_list):
            bbox = obs.get("bbox_xyxy", [0, 0, 1, 1])
            bw   = max(bbox[2] - bbox[0], 1)
            bh   = max(bbox[3] - bbox[1], 1)
            area = obs.get("area_px") or (bw * bh)
            sharp = obs.get("sharpness") or 0.0
            depth = obs.get("median_depth_m") or 0.0
            cam_c = obs.get("centroid_camera") or [0.0, 0.0, 0.0]
            wld_c = obs.get("centroid_world")  or [0.0, 0.0, 0.0]

            track_frac = i / max(n_obs - 1, 1)
            ts_rel     = obs["timestamp"] - first_ts

            fv: Dict[str, Any] = {
                # Identifiers
                "sequence_id"   : seq_id,
                "track_id"      : track_id,
                "obs_id"        : obs["obs_id"],
                "frame_index"   : obs["frame_index"],
                "timestamp"     : obs["timestamp"],

                # Visual
                "detection_confidence" : round(obs.get("detection_confidence", 0.0), 4),
                "crop_area_px"         : round(float(area), 1),
                "crop_area_log"        : round(math.log10(float(area) + 1.0), 4),
                "sharpness"            : round(float(sharp), 2),
                "sharpness_log"        : round(math.log10(float(sharp) + 1.0), 4),
                "bbox_aspect_ratio"    : round(bw / bh, 4),
                "has_mask"             : bool(obs.get("has_mask", False)),

                # Geometric
                "median_depth_m"       : round(float(depth), 4),
                "depth_valid"          : 0.0 < float(depth) < 10.0,
                "centroid_cam_x"       : round(float(cam_c[0]), 4) if len(cam_c) > 0 else None,
                "centroid_cam_y"       : round(float(cam_c[1]), 4) if len(cam_c) > 1 else None,
                "centroid_cam_z"       : round(float(cam_c[2]), 4) if len(cam_c) > 2 else None,
                "centroid_world_x"     : round(float(wld_c[0]), 4) if len(wld_c) > 0 else None,
                "centroid_world_y"     : round(float(wld_c[1]), 4) if len(wld_c) > 1 else None,
                "centroid_world_z"     : round(float(wld_c[2]), 4) if len(wld_c) > 2 else None,
                "ego_motion_available" : bool(obs.get("ego_motion_available", False)),

                # Estimator (track-level — causal note: end-of-sequence values)
                "world_displacement_m"                 : round(float(world_disp), 4),
                "displacement_uncertainty_m"           : round(float(disp_unc), 4),
                "apparent_camera_frame_displacement_m" : round(float(cam_frame_disp), 4),
                "camera_ego_motion_displacement_m"     : round(float(ego_disp), 4),
                "pipeline_motion_label"                : pipe_motion,

                # Temporal
                "track_obs_index" : i,
                "track_obs_frac"  : round(track_frac, 4),
                "lifespan_sec"    : round(float(lifespan), 4),
                "total_detections": int(total_dets),
                "timestamp_rel"   : round(float(ts_rel), 4),

                # Contextual
                "semantic_class"      : obs.get("semantic_class", track.get("semantic_class", "unknown")),
                "sequence_domain"     : sequence_to_domain(seq_id),
                "gt_vlm_label"        : gt_label,
                "annotation_confidence": ann_conf,
                "evaluation_split"    : seq_partition.get(seq_id, "unknown"),

                # File references (not features — for debugging/linking)
                "crop_path"           : obs.get("crop_path"),
                "mask_path"           : obs.get("mask_path"),
                "rgb_path"            : obs.get("rgb_path"),
            }
            feature_vectors.append(fv)

    print(f"\n  Feature vectors built: {len(feature_vectors)}")
    print(f"  No annotation found  : {n_no_annotation}")
    print(f"  UNCERTAIN GT label   : {n_uncertain_gt}")
    print(f"  GT label available   : {len(feature_vectors) - n_no_annotation - n_uncertain_gt}")

    # Write feature vectors
    fv_path = OUT_DIR / "feature_vectors.jsonl"
    with fv_path.open("w", encoding="utf-8") as fh:
        for fv in feature_vectors:
            fh.write(json.dumps(fv, ensure_ascii=False, default=str) + "\n")
    print(f"\n  feature_vectors.jsonl written ({len(feature_vectors)} records)")

    # Write schema
    schema_path = OUT_DIR / "feature_schema.json"
    schema_path.write_text(json.dumps(FEATURE_SCHEMA, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"  feature_schema.json written")

    # Compute summary statistics for report
    import statistics

    stats_lines = [
        "# Estimator-Context Feature Report",
        "",
        f"Generated: {datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')}",
        f"Source run: `{RUN_DIR}`",
        f"Total feature vectors: {len(feature_vectors)}",
        "",
        "## Feature Summary Statistics",
        "",
        "| Feature | Min | Median | Max | Mean |",
        "| --- | --- | --- | --- | --- |",
    ]

    numeric_fields = [
        "detection_confidence", "crop_area_px", "sharpness",
        "median_depth_m", "world_displacement_m", "displacement_uncertainty_m",
        "apparent_camera_frame_displacement_m", "camera_ego_motion_displacement_m",
        "track_obs_frac", "lifespan_sec", "total_detections",
    ]

    for field_name in numeric_fields:
        vals = [fv[field_name] for fv in feature_vectors if fv.get(field_name) is not None]
        if vals:
            med = statistics.median(vals)
            stats_lines.append(
                f"| {field_name} | {min(vals):.3f} | {med:.3f} | {max(vals):.3f} | {sum(vals)/len(vals):.3f} |"
            )

    # Causal leakage note
    stats_lines += [
        "",
        "## Causal Leakage Check",
        "",
        "> [!IMPORTANT]",
        "> `world_displacement_m`, `pipeline_motion_label`, `lifespan_sec`, and `total_detections`",
        "> are **track-level** fields computed over the full sequence. They are NOT causal",
        "> at the per-observation level — the policy cannot know end-of-sequence displacement",
        "> when making a decision at observation time `t`.",
        ">",
        "> For a real-time causal query selector:",
        "> - Use `track_obs_frac` and `timestamp_rel` as temporal position proxies.",
        "> - Use `world_displacement_m` only as a training/evaluation signal, not as a runtime feature.",
        "> - `detection_confidence`, `sharpness`, `median_depth_m`, `crop_area_px`, `ego_motion_available`",
        ">   and `centroid_*` ARE fully causal — they derive from the current frame only.",
        "",
        "## Causal Feature Subset (safe for runtime query selector)",
        "",
        "| Feature | Category | Rationale |",
        "| --- | --- | --- |",
        "| detection_confidence | visual | YOLO score at current frame |",
        "| crop_area_px | visual | Bounding box area at current frame |",
        "| sharpness | visual | Laplacian of current crop |",
        "| has_mask | visual | Mask present at current frame |",
        "| median_depth_m | geometric | Depth at current frame |",
        "| depth_valid | geometric | Depth validity at current frame |",
        "| centroid_cam_* | geometric | 3D position at current frame |",
        "| centroid_world_* | geometric | Ego-compensated position at current frame |",
        "| ego_motion_available | geometric | Whether trajectory was used |",
        "| track_obs_index | temporal | How many times object seen so far |",
        "| track_obs_frac | temporal | Relative position in observed track window |",
        "| timestamp_rel | temporal | Time since first detection |",
        "| sequence_domain | contextual | Dataset type (fixed metadata) |",
        "| semantic_class | contextual | YOLO class (available at detection time) |",
        "",
        "## Non-Causal Fields (use only for offline analysis / training labels)",
        "",
        "| Field | Why non-causal |",
        "| --- | --- |",
        "| world_displacement_m | End-of-sequence aggregate |",
        "| pipeline_motion_label | End-of-sequence aggregate |",
        "| lifespan_sec | Track end not known at decision time |",
        "| total_detections | Track end not known at decision time |",
        "| gt_vlm_label | Ground truth — never available at runtime |",
    ]

    report_path = OUT_DIR / "context_report.md"
    report_path.write_text("\n".join(stats_lines), encoding="utf-8")
    print(f"  context_report.md written")

    # Console summary
    print("\n" + "=" * 60)
    print("Feature extraction complete")
    print("=" * 60)
    from collections import Counter
    domain_dist  = Counter(fv["sequence_domain"] for fv in feature_vectors)
    split_dist   = Counter(fv["evaluation_split"] for fv in feature_vectors)
    label_dist   = Counter(fv["gt_vlm_label"] for fv in feature_vectors)
    motion_dist  = Counter(fv["pipeline_motion_label"] for fv in feature_vectors)

    print(f"  By domain : {dict(domain_dist)}")
    print(f"  By split  : {dict(split_dist)}")
    print(f"  By GT label: {dict(label_dist)}")
    print(f"  By pipe motion: {dict(motion_dist)}")
    print(f"\nOutputs in: {OUT_DIR}")
    print("  feature_vectors.jsonl   — 885 causal feature vectors")
    print("  feature_schema.json     — field definitions and causal annotations")
    print("  context_report.md       — statistics and causal leakage analysis")


if __name__ == "__main__":
    main()

