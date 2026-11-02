"""Rule-based track annotator for perception pipeline observations."""

from __future__ import annotations

import csv
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

RUN_DIR = REPO_ROOT / "results" / "perception_harvesting" / "run_20261101_234859"
OUT_DIR = REPO_ROOT / "results" / "track_annotation"

# Sequence context

# Sequences where the camera is actively moving (handheld / robot-mounted)
CAMERA_MOVING_SEQUENCES = {
    "tum_xyz", "tum_halfsphere", "tum_static",
    "openloris",
    "bonn_crowd", "bonn_box",
    "d455_01", "d455_02",
}

# Sequences with known walking people in scene
DYNAMIC_PERSON_SEQUENCES = {
    "tum_xyz",        # freiburg3/walking_xyz    — multiple walking people
    "tum_halfsphere", # freiburg3/walking_halfsphere — multiple walking people
    "tum_static",     # freiburg3/walking_static  — walking people, static bg
    "bonn_crowd",     # rgbd_bonn_crowd           — crowd walking
    "bonn_box",       # rgbd_bonn_moving_box      — person + moving box
    "openloris",      # corridor — occasional pedestrian
}

# D455 sequences are desk/office scenes — objects mostly static,
# but camera roams around the desk so ego-motion residuals can look like object motion.
DESK_SEQUENCES = {"d455_01", "d455_02"}

# bonn_box: a box is deliberately pushed/moved to obstruct the camera.
# YOLO may detect parts of the box as laptop/monitor due to rectangular shape.
BOX_OBSTRUCTION_SEQUENCE = "bonn_box"

# Physical mobility rules

# Physical mobility = is this object type CAPABLE of being moved?
# Independent of whether it actually moved during this recording.
PHYSICAL_MOBILITY_BY_CLASS = {
    "person":   "MOVABLE",               # self-mobile
    "chair":    "MOVABLE",               # can be pushed / relocated
    "monitor":  "MOVABLE",               # can be relocated (heavy but not fixed)
    "laptop":   "MOVABLE",               # portable by design
    "keyboard": "MOVABLE",               # portable
    "mouse":    "MOVABLE",               # portable
    "plant":    "MOVABLE",               # potted plant can be moved
    "desk":     "STRUCTURALLY_IMMOVABLE", # bolted / too heavy in office context
    "box":      "MOVABLE",
}

# Thresholds

# Ego-motion compensation residual error is ~0.1–0.2 m in TUM/Bonn sequences.
# Objects with world_displacement < LOW_THRESH are almost certainly stationary.
DISP_LOW  = 0.20   # below this  → STATIONARY
DISP_HIGH = 0.50   # above this  → strong evidence of real motion

# Detection confidence thresholds for visibility classification
CONF_CLEAR   = 0.65
CONF_PARTIAL = 0.50

# Pixel area thresholds
AREA_LARGE  = 10_000
AREA_MEDIUM = 3_000

# Short-track flag: tracks with < N total detections are less reliable
SHORT_TRACK_THRESHOLD = 10

# Core annotation logic

def annotate_track(
    track: Dict[str, Any],
    rep_obs: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Return annotation dict for one track. Rationale is included in 'rationale'."""

    seq   = track["sequence_id"]
    tid   = track["track_id"]
    cls   = track["semantic_class"]
    pipe_motion = track.get("observed_motion", "UNKNOWN")  # MOVING/DISPLACED/STATIC
    disp  = track.get("world_displacement_m")              # float or None
    declared_obs = track.get("observation_count", 0)       # VLM-sampled count (always 5)
    # Use the declared count from tracks.jsonl as the total detection count
    # (the track's observation_count in tracks.jsonl IS the total detections)
    total_detections = track.get("declared_count", declared_obs)

    rationale_parts: List[str] = []

    # 1. OBJECT CLASS
    # Start with YOLO label. Flag suspicious cases.
    object_class = cls

    # In bonn_box, YOLO sometimes detects the cardboard box as 'laptop' or 'monitor'
    # because of its rectangular shape. Flag for human review.
    if seq == BOX_OBSTRUCTION_SEQUENCE and cls in ("laptop", "monitor") and disp is not None and disp < 0.15:
        rationale_parts.append(
            f"[class] YOLO label '{cls}' in bonn_box sequence — may be cardboard box misclassified."
        )

    # 2. PHYSICAL MOBILITY
    physical_mobility = PHYSICAL_MOBILITY_BY_CLASS.get(cls, "UNCERTAIN")
    rationale_parts.append(f"[physical_mobility] class='{cls}' → {physical_mobility} (rule-based by class)")

    # 3. OBSERVED MOTION
    observed_motion: str
    motion_confidence: str  # internal, feeds into annotation_confidence

    if disp is None:
        observed_motion = "UNCERTAIN"
        motion_confidence = "LOW"
        rationale_parts.append("[observed_motion] world_displacement unavailable → UNCERTAIN")

    elif cls == "person":
        # Persons in dynamic sequences: trust world displacement + pipeline
        if seq in DYNAMIC_PERSON_SEQUENCES:
            if disp > DISP_LOW or pipe_motion in ("MOVING", "DISPLACED"):
                observed_motion = "MOVING"
                motion_confidence = "HIGH" if disp > DISP_HIGH else "MEDIUM"
            else:
                # Low disp + DYNAMIC sequence — person briefly visible, probably moving
                observed_motion = "MOVING"
                motion_confidence = "MEDIUM"
                rationale_parts.append(
                    "[observed_motion] low world_disp but dynamic sequence — short track, assume MOVING"
                )
        else:
            # d455 desk recordings — person walking through
            if disp > DISP_HIGH:
                observed_motion = "MOVING"
                motion_confidence = "HIGH"
            elif disp > DISP_LOW:
                observed_motion = "MOVING"
                motion_confidence = "MEDIUM"
            else:
                observed_motion = "STATIONARY"
                motion_confidence = "MEDIUM"

    elif cls in ("monitor", "keyboard", "mouse", "plant"):
        # Typically static office fixtures. High disp = ego-comp residual or ID error.
        if disp < DISP_LOW:
            observed_motion = "STATIONARY"
            motion_confidence = "HIGH"
        elif disp < DISP_HIGH:
            observed_motion = "STATIONARY"
            motion_confidence = "MEDIUM"
            rationale_parts.append(
                f"[observed_motion] disp={disp:.3f}m for static fixture — "
                "likely ego-comp residual; classified STATIONARY with MEDIUM confidence"
            )
        else:
            # disp >= 0.5m for a monitor/keyboard is suspicious
            observed_motion = "UNCERTAIN"
            motion_confidence = "LOW"
            rationale_parts.append(
                f"[observed_motion] disp={disp:.3f}m unexpectedly large for '{cls}' — "
                "possible ego-comp error or ID switch; classified UNCERTAIN"
            )

    elif cls == "desk":
        # Desks are structurally immovable
        observed_motion = "STATIONARY"
        motion_confidence = "HIGH" if disp is None or disp < DISP_LOW else "MEDIUM"
        if disp is not None and disp > DISP_LOW:
            rationale_parts.append(
                f"[observed_motion] desk shows disp={disp:.3f}m — "
                "ego-comp residual or poor detection; classified STATIONARY"
            )

    elif cls == "chair":
        # Chairs can be pushed. Trust pipeline + displacement.
        if pipe_motion == "STATIC" or disp < DISP_LOW:
            observed_motion = "STATIONARY"
            motion_confidence = "HIGH" if disp < DISP_LOW else "MEDIUM"
        elif disp > DISP_HIGH or pipe_motion == "MOVING":
            observed_motion = "MOVING"
            motion_confidence = "HIGH" if disp > DISP_HIGH else "MEDIUM"
        else:
            # DISPLACED (0.2–0.5m): chair may have been nudged or camera residual
            if pipe_motion == "DISPLACED":
                observed_motion = "MOVING"
                motion_confidence = "MEDIUM"
            else:
                observed_motion = "UNCERTAIN"
                motion_confidence = "MEDIUM"

    elif cls == "laptop":
        # Laptops are portable but usually left open on desk
        if seq == BOX_OBSTRUCTION_SEQUENCE:
            # In bonn_box: YOLO may see the obstructing box as a laptop
            if disp > DISP_HIGH:
                observed_motion = "MOVING"
                motion_confidence = "MEDIUM"
                rationale_parts.append(
                    "[observed_motion] laptop in bonn_box with high disp — "
                    "may be the obstructing box; classified MOVING"
                )
            else:
                observed_motion = "STATIONARY"
                motion_confidence = "MEDIUM"
        else:
            if disp < DISP_LOW and pipe_motion == "STATIC":
                observed_motion = "STATIONARY"
                motion_confidence = "HIGH"
            elif disp > DISP_HIGH:
                observed_motion = "UNCERTAIN"
                motion_confidence = "LOW"
                rationale_parts.append(
                    f"[observed_motion] laptop disp={disp:.3f}m unexpectedly large — "
                    "likely ego-comp residual or misclassification"
                )
            else:
                observed_motion = "STATIONARY"
                motion_confidence = "MEDIUM"

    else:
        # Fallback
        if pipe_motion == "STATIC" or (disp is not None and disp < DISP_LOW):
            observed_motion = "STATIONARY"
            motion_confidence = "MEDIUM"
        elif pipe_motion == "MOVING" and disp is not None and disp > DISP_HIGH:
            observed_motion = "MOVING"
            motion_confidence = "MEDIUM"
        else:
            observed_motion = "UNCERTAIN"
            motion_confidence = "LOW"

    rationale_parts.append(
        f"[observed_motion] pipe_motion={pipe_motion} world_disp={disp} → {observed_motion} ({motion_confidence})"
    )

    # 4. VISIBILITY
    if rep_obs:
        mean_conf = sum(r.get("detection_confidence", 0.5) for r in rep_obs) / len(rep_obs)
        mean_area = sum(r.get("area_px", 0) for r in rep_obs) / len(rep_obs)
    else:
        mean_conf = 0.5
        mean_area = 0.0

    if mean_conf >= CONF_CLEAR and mean_area >= AREA_MEDIUM:
        visibility = "CLEAR"
    elif mean_conf >= CONF_PARTIAL or mean_area >= AREA_MEDIUM:
        visibility = "PARTIALLY_OCCLUDED"
    else:
        visibility = "HEAVILY_OCCLUDED"

    rationale_parts.append(
        f"[visibility] mean_conf={mean_conf:.2f} mean_area={mean_area:.0f} → {visibility}"
    )

    # 5. TRACK IDENTITY VALIDITY
    track_identity_validity = "VALID"
    validity_flags: List[str] = []

    # Short tracks are less reliable
    if total_detections is not None and total_detections < SHORT_TRACK_THRESHOLD:
        track_identity_validity = "QUESTIONABLE"
        validity_flags.append(f"short track ({total_detections} total detections < {SHORT_TRACK_THRESHOLD})")

    # Static fixture showing implausibly large world displacement
    if cls in ("monitor", "keyboard", "desk", "mouse") and disp is not None and disp > 0.8:
        track_identity_validity = "QUESTIONABLE"
        validity_flags.append(
            f"static class '{cls}' shows world_disp={disp:.3f}m > 0.8m — "
            "possible ego-comp failure or identity switch"
        )

    # Laptop in bonn_box with low displacement but static: possible box misclassification
    if seq == BOX_OBSTRUCTION_SEQUENCE and cls == "laptop" and disp is not None and disp < 0.1:
        track_identity_validity = "QUESTIONABLE"
        validity_flags.append("laptop in bonn_box with near-zero disp — may be misclassified box fragment")

    if validity_flags:
        rationale_parts.append(f"[track_identity] QUESTIONABLE: {'; '.join(validity_flags)}")
    else:
        rationale_parts.append("[track_identity] VALID — no suspicious patterns detected")

    # 6. EVIDENCE SUFFICIENCY
    # Insufficient if: heavily occluded, very short track, or annotation is UNCERTAIN
    if (
        visibility == "HEAVILY_OCCLUDED"
        or (total_detections is not None and total_detections < SHORT_TRACK_THRESHOLD)
        or observed_motion == "UNCERTAIN"
    ):
        evidence_sufficiency = "INSUFFICIENT"
    else:
        evidence_sufficiency = "SUFFICIENT"

    rationale_parts.append(f"[evidence] → {evidence_sufficiency}")

    # 7. ANNOTATION CONFIDENCE
    # Combine motion confidence, identity validity, and evidence sufficiency
    penalty_count = 0
    if motion_confidence == "LOW":
        penalty_count += 2
    elif motion_confidence == "MEDIUM":
        penalty_count += 1

    if track_identity_validity == "QUESTIONABLE":
        penalty_count += 1
    if evidence_sufficiency == "INSUFFICIENT":
        penalty_count += 1
    if visibility == "HEAVILY_OCCLUDED":
        penalty_count += 1

    if penalty_count == 0:
        annotation_confidence = "HIGH"
    elif penalty_count <= 2:
        annotation_confidence = "MEDIUM"
    else:
        annotation_confidence = "LOW"

    # 8. ANNOTATOR NOTES
    notes: List[str] = []
    if disp is not None:
        notes.append(f"world_disp={disp:.3f}m")
    notes.append(f"pipe_motion={pipe_motion}")
    if total_detections is not None:
        notes.append(f"total_detections={total_detections}")
    if validity_flags:
        notes.extend(validity_flags)

    annotator_notes = "; ".join(notes)

    return {
        "object_class":             object_class,
        "physical_mobility":        physical_mobility,
        "observed_motion":          observed_motion,
        "visibility":               visibility,
        "track_identity_validity":  track_identity_validity,
        "evidence_sufficiency":     evidence_sufficiency,
        "annotation_confidence":    annotation_confidence,
        "annotator_notes":          annotator_notes,
        "_rationale":               " | ".join(rationale_parts),
    }


# Main

def main() -> None:
    print("\nRule-Based Track Annotator")
    print(f"Run dir : {RUN_DIR}")
    print(f"Out dir : {OUT_DIR}\n")

    if not RUN_DIR.exists():
        sys.exit(f"ERROR: run dir not found: {RUN_DIR}")
    if not OUT_DIR.exists():
        sys.exit(f"ERROR: out dir not found: {OUT_DIR}. Run build_track_annotation_dataset.py first.")

    # Load data
    tracks = [
        json.loads(l)
        for l in (RUN_DIR / "tracks.jsonl").read_text(encoding="utf-8").splitlines()
        if l.strip()
    ]
    reps = [
        json.loads(l)
        for l in (OUT_DIR / "representative_observations.jsonl").read_text(encoding="utf-8").splitlines()
        if l.strip()
    ]
    manifest = json.loads((OUT_DIR / "track_manifest.json").read_text(encoding="utf-8"))

    # Build lookup: seq__tid → track manifest entry (has declared_count)
    manifest_by_key: Dict[str, Any] = {}
    for t in manifest["tracks"]:
        key = f"{t['sequence_id']}__{t['track_id']}"
        manifest_by_key[key] = t

    # Index reps by seq__tid
    rep_by_track: Dict[str, List] = defaultdict(list)
    for r in reps:
        key = f"{r['sequence_id']}__{r['track_id']}"
        rep_by_track[key].append(r)

    # Index raw tracks by seq__tid
    raw_track_by_key: Dict[str, Any] = {}
    for t in tracks:
        key = f"{t['sequence_id']}__{t['track_id']}"
        raw_track_by_key[key] = t

    print(f"Tracks to annotate: {len(tracks)}")
    print("Applying rules ...\n")

    annotations: List[Dict[str, Any]] = []
    audit_rows: List[Dict[str, Any]] = []

    # Stats
    motion_dist: Dict[str, int] = defaultdict(int)
    mobility_dist: Dict[str, int] = defaultdict(int)
    validity_dist: Dict[str, int] = defaultdict(int)
    confidence_dist: Dict[str, int] = defaultdict(int)
    sufficiency_dist: Dict[str, int] = defaultdict(int)

    for t in tracks:
        seq = t["sequence_id"]
        tid = t["track_id"]
        key = f"{seq}__{tid}"

        # Enrich track with declared_count from manifest
        mentry = manifest_by_key.get(key, {})
        t_enriched = dict(t)
        t_enriched["declared_count"] = mentry.get("declared_count")

        rep_obs = rep_by_track.get(key, [])
        ann = annotate_track(t_enriched, rep_obs)

        rep_ids = "|".join(r["obs_id"] for r in rep_obs)

        row = {
            "sequence_id":                   seq,
            "track_id":                      tid,
            "object_class":                  ann["object_class"],
            "physical_mobility":             ann["physical_mobility"],
            "observed_motion":               ann["observed_motion"],
            "visibility":                    ann["visibility"],
            "track_identity_validity":       ann["track_identity_validity"],
            "evidence_sufficiency":          ann["evidence_sufficiency"],
            "annotation_confidence":         ann["annotation_confidence"],
            "annotator_notes":               ann["annotator_notes"],
            "representative_observation_ids": rep_ids,
            "_pipeline_semantic_class":      t.get("semantic_class"),
            "_pipeline_observed_motion":     t.get("observed_motion"),
            "_pipeline_world_displacement_m": t.get("world_displacement_m"),
            "_pipeline_observation_count":   mentry.get("observation_count"),
            "_pipeline_ego_motion_available": mentry.get("ego_motion_available"),
        }
        annotations.append(row)
        audit_rows.append({**row, "_rationale": ann["_rationale"]})

        motion_dist[ann["observed_motion"]] += 1
        mobility_dist[ann["physical_mobility"]] += 1
        validity_dist[ann["track_identity_validity"]] += 1
        confidence_dist[ann["annotation_confidence"]] += 1
        sufficiency_dist[ann["evidence_sufficiency"]] += 1

    # Write filled CSV
    csv_path = OUT_DIR / "track_annotations.csv"
    fieldnames = [
        "sequence_id", "track_id", "object_class",
        "physical_mobility", "observed_motion", "visibility",
        "track_identity_validity", "evidence_sufficiency", "annotation_confidence",
        "annotator_notes", "representative_observation_ids",
        "_pipeline_semantic_class", "_pipeline_observed_motion",
        "_pipeline_world_displacement_m", "_pipeline_observation_count",
        "_pipeline_ego_motion_available",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in annotations:
            writer.writerow({k: row.get(k, "") for k in fieldnames})

    print(f"track_annotations.csv written ({len(annotations)} rows)")

    # Write annotation audit report
    audit_lines = [
        "# Track Annotation Audit",
        "",
        f"Source run: `{RUN_DIR}`",
        f"Annotator: rule-based (scripts/auto_annotate_tracks.py)",
        f"Tracks: {len(annotations)}",
        "",
        "## Distribution Summary",
        "",
        "### Observed Motion",
        "",
        "| Label | Count |",
        "| --- | --- |",
    ]
    for label, count in sorted(motion_dist.items(), key=lambda x: -x[1]):
        audit_lines.append(f"| {label} | {count} |")

    audit_lines += [
        "",
        "### Physical Mobility",
        "",
        "| Label | Count |",
        "| --- | --- |",
    ]
    for label, count in sorted(mobility_dist.items(), key=lambda x: -x[1]):
        audit_lines.append(f"| {label} | {count} |")

    audit_lines += [
        "",
        "### Track Identity Validity",
        "",
        "| Label | Count |",
        "| --- | --- |",
    ]
    for label, count in sorted(validity_dist.items(), key=lambda x: -x[1]):
        audit_lines.append(f"| {label} | {count} |")

    audit_lines += [
        "",
        "### Annotation Confidence",
        "",
        "| Label | Count |",
        "| --- | --- |",
    ]
    for label, count in sorted(confidence_dist.items(), key=lambda x: -x[1]):
        audit_lines.append(f"| {label} | {count} |")

    audit_lines += [
        "",
        "### Evidence Sufficiency",
        "",
        "| Label | Count |",
        "| --- | --- |",
    ]
    for label, count in sorted(sufficiency_dist.items(), key=lambda x: -x[1]):
        audit_lines.append(f"| {label} | {count} |")

    audit_lines += [
        "",
        "## Per-Track Rationale",
        "",
    ]
    for row in audit_rows:
        audit_lines += [
            f"### {row['sequence_id']} / {row['track_id']}",
            "",
            f"| Field | Value |",
            f"| --- | --- |",
            f"| object_class | {row['object_class']} |",
            f"| physical_mobility | {row['physical_mobility']} |",
            f"| observed_motion | {row['observed_motion']} |",
            f"| visibility | {row['visibility']} |",
            f"| track_identity_validity | {row['track_identity_validity']} |",
            f"| evidence_sufficiency | {row['evidence_sufficiency']} |",
            f"| annotation_confidence | {row['annotation_confidence']} |",
            f"| _pipeline_observed_motion | {row['_pipeline_observed_motion']} |",
            f"| _pipeline_world_displacement_m | {row['_pipeline_world_displacement_m']} |",
            "",
            f"**Rationale:** {row['_rationale']}",
            "",
        ]

    audit_path = OUT_DIR / "annotation_audit.md"
    audit_path.write_text("\n".join(audit_lines), encoding="utf-8")
    print(f"annotation_audit.md written")

    # Console summary
    print(f"{'Observed motion':30s} {'Physical mobility':30s}")
    for (ml, mc), (pl, pc) in zip(
        sorted(motion_dist.items()), sorted(mobility_dist.items())
    ):
        print(f"  {ml:20s} {mc:5d}   {pl:20s} {pc:5d}")

    print(f"\nTrack identity: VALID={validity_dist.get('VALID',0)}  "
          f"QUESTIONABLE={validity_dist.get('QUESTIONABLE',0)}")
    print(f"Evidence: SUFFICIENT={sufficiency_dist.get('SUFFICIENT',0)}  "
          f"INSUFFICIENT={sufficiency_dist.get('INSUFFICIENT',0)}")
    print(f"Confidence: HIGH={confidence_dist.get('HIGH',0)}  "
          f"MEDIUM={confidence_dist.get('MEDIUM',0)}  "
          f"LOW={confidence_dist.get('LOW',0)}")

    print("\nOutputs:")
    print(f"  {csv_path}")
    print(f"  {audit_path}")
    print("\nReview annotation_audit.md before proceeding to VLM evaluation.")


if __name__ == "__main__":
    main()

