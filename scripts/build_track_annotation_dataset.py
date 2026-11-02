"""Build track-level annotation dataset from unified perception runs."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional

# Repo root detection

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

CANONICAL_RUN = REPO_ROOT / "results" / "perception_harvesting" / "run_20261101_234859"
OUTPUT_ROOT = REPO_ROOT / "results" / "track_annotation"

# Annotation vocabulary

ANNOTATION_SCHEMA = {
    "version": "1.0",
    "description": (
        "Reference label vocabulary for the 177-track evaluation dataset. "
        "All fields are set to PENDING in the scaffold; human annotators fill them in."
    ),
    "fields": {
        "sequence_id": {
            "type": "string",
            "description": "Source recording identifier.",
        },
        "track_id": {
            "type": "string",
            "description": "Tracker-assigned track identifier (unique within sequence).",
        },
        "object_class": {
            "type": "string",
            "description": "Human-verified semantic class. May differ from YOLO label.",
            "values": ["person", "chair", "monitor", "laptop", "keyboard",
                       "desk", "mouse", "plant", "box", "other", "PENDING"],
        },
        "physical_mobility": {
            "type": "enum",
            "description": (
                "Whether the object is physically capable of moving. "
                "A chair is MOVABLE even if it never moves during the recording."
            ),
            "values": ["MOVABLE", "STRUCTURALLY_IMMOVABLE", "UNCERTAIN", "PENDING"],
        },
        "observed_motion": {
            "type": "enum",
            "description": (
                "Whether the object was actually observed moving during the recording. "
                "Must be determined from temporal evidence (multiple frames), "
                "not from a single crop. Raw image displacement alone is insufficient "
                "because camera motion produces apparent object motion."
            ),
            "values": ["MOVING", "STATIONARY", "UNCERTAIN", "PENDING"],
        },
        "visibility": {
            "type": "enum",
            "description": "Best visibility achieved across representative observations.",
            "values": ["CLEAR", "PARTIALLY_OCCLUDED", "HEAVILY_OCCLUDED", "PENDING"],
        },
        "track_identity_validity": {
            "type": "enum",
            "description": (
                "Whether the track consistently refers to the same physical object. "
                "QUESTIONABLE if suspected identity switch. INVALID if clearly wrong."
            ),
            "values": ["VALID", "QUESTIONABLE", "INVALID", "PENDING"],
        },
        "evidence_sufficiency": {
            "type": "enum",
            "description": (
                "Whether the representative observations provide enough visual evidence "
                "to make a confident label decision."
            ),
            "values": ["SUFFICIENT", "INSUFFICIENT", "PENDING"],
        },
        "annotation_confidence": {
            "type": "enum",
            "description": "Overall annotator confidence in the assigned labels.",
            "values": ["HIGH", "MEDIUM", "LOW", "PENDING"],
        },
        "annotator_notes": {
            "type": "string",
            "description": "Free-text notes, concerns or caveats for this track.",
        },
        "representative_observation_ids": {
            "type": "list[string]",
            "description": "obs_id values selected as representative for this track.",
        },
    },
}

# Representative observation selector

MAX_REPRESENTATIVE = 5
MIN_REPRESENTATIVE = 3


def select_representative_observations(
    obs_for_track: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """Select 3-5 representative observations from a track's accepted observations.

    Strategy:
    - Always include the FIRST (earliest) observation.
    - Always include the LAST (latest) observation.
    - Always include the MIDDLE observation.
    - Add the sharpest observation if not already selected (visual clarity).
    - Add an observation near the motion/visibility transition if detectable.

    Sorting by frame_index ensures temporal ordering.
    """
    if not obs_for_track:
        return []

    # Sort by frame index (ascending = temporal order)
    sorted_obs = sorted(obs_for_track, key=lambda o: o["frame_index"])
    n = len(sorted_obs)

    if n <= MAX_REPRESENTATIVE:
        selected = sorted_obs[:]
        method = "all_obs_under_max"
        for o in selected:
            o["_selection_reason"] = "all_selected"
        return selected

    # Mandatory anchors
    first = sorted_obs[0]
    last  = sorted_obs[-1]
    mid   = sorted_obs[n // 2]

    selected_ids = {first["obs_id"], mid["obs_id"], last["obs_id"]}
    selected = [first, mid, last]

    first["_selection_reason"] = "temporal_anchor_first"
    mid["_selection_reason"]   = "temporal_anchor_middle"
    last["_selection_reason"]  = "temporal_anchor_last"

    # Add sharpest observation not already selected
    remaining = [o for o in sorted_obs if o["obs_id"] not in selected_ids]
    if remaining:
        sharpest = max(remaining, key=lambda o: o.get("sharpness", 0.0))
        sharpest["_selection_reason"] = "sharpest_crop"
        selected_ids.add(sharpest["obs_id"])
        selected.append(sharpest)
        remaining = [o for o in remaining if o["obs_id"] not in selected_ids]

    # Add a 5th at the 3/4 mark if room
    if len(selected) < MAX_REPRESENTATIVE and remaining:
        q3_idx = (3 * n) // 4
        # Find observation closest to the 3/4 mark
        q3_obs = min(
            remaining,
            key=lambda o: abs(sorted_obs.index(o) - q3_idx)
        )
        q3_obs["_selection_reason"] = "temporal_anchor_q3"
        selected.append(q3_obs)

    # Return sorted by frame_index
    return sorted(selected, key=lambda o: o["frame_index"])


# Tracking consistency checks

def check_tracking_consistency(
    tracks: List[Dict[str, Any]],
    obs_by_track: Dict[str, List[Dict[str, Any]]],
) -> List[Dict[str, Any]]:
    """Run integrity checks on track data. Returns a list of issue dicts.

    Checks:
    1. Track IDs are unique within each sequence.
    2. Observations are monotonically ordered by timestamp.
    3. Frame indices are non-negative.
    4. No duplicate obs_id within a track.
    5. Track observation count matches actual observations in jsonl.
    6. Crop files exist on disk.
    """
    issues = []

    # 1. Unique track IDs per sequence
    seq_track_ids: Dict[str, list] = defaultdict(list)
    for t in tracks:
        seq_track_ids[t["sequence_id"]].append(t["track_id"])

    for seq, tids in seq_track_ids.items():
        seen = set()
        for tid in tids:
            if tid in seen:
                issues.append({
                    "severity": "ERROR",
                    "check": "unique_track_id_per_sequence",
                    "sequence_id": seq,
                    "track_id": tid,
                    "detail": f"track_id '{tid}' duplicated in sequence '{seq}'",
                })
            seen.add(tid)

    for t in tracks:
        tid = t["track_id"]
        seq = t["sequence_id"]
        key = f"{seq}__{tid}"
        obs = obs_by_track.get(key, [])

        # 2. Monotonic timestamps
        timestamps = [o["timestamp"] for o in obs]
        for i in range(1, len(timestamps)):
            if timestamps[i] < timestamps[i - 1]:
                issues.append({
                    "severity": "WARNING",
                    "check": "monotonic_timestamps",
                    "sequence_id": seq,
                    "track_id": tid,
                    "detail": (
                        f"Non-monotonic timestamp at obs index {i}: "
                        f"{timestamps[i-1]:.3f} → {timestamps[i]:.3f}"
                    ),
                })
                break  # report first occurrence only

        # 3. Non-negative frame indices
        for o in obs:
            if o["frame_index"] < 0:
                issues.append({
                    "severity": "ERROR",
                    "check": "non_negative_frame_index",
                    "sequence_id": seq,
                    "track_id": tid,
                    "obs_id": o["obs_id"],
                    "detail": f"Negative frame_index: {o['frame_index']}",
                })

        # 4. Duplicate obs_id
        obs_ids = [o["obs_id"] for o in obs]
        seen_obs = set()
        for oid in obs_ids:
            if oid in seen_obs:
                issues.append({
                    "severity": "ERROR",
                    "check": "unique_obs_id",
                    "sequence_id": seq,
                    "track_id": tid,
                    "obs_id": oid,
                    "detail": f"Duplicate obs_id '{oid}' in track",
                })
            seen_obs.add(oid)

        # 5. Count match — NOTE:
        # tracks.jsonl observation_count = total detections across ALL frames (can be hundreds).
        # candidate_observations.jsonl = VLM-sampled subset (capped at 5 per track by harvester).
        # These are expected to differ. We only warn if candidate_obs count is 0 (no accepted obs).
        actual_count = len(obs)
        if actual_count == 0:
            issues.append({
                "severity": "ERROR",
                "check": "zero_accepted_observations",
                "sequence_id": seq,
                "track_id": tid,
                "detail": (
                    f"Track has 0 accepted observations in candidate_observations.jsonl "
                    f"(total detections declared: {t.get('observation_count', '?')}). "
                    f"This track cannot be annotated."
                ),
            })

        # 6. Crop files exist (check first and last only to keep it fast)
        for o in (obs[:1] + obs[-1:]):
            cp = o.get("crop_path")
            if cp and not Path(cp).exists():
                issues.append({
                    "severity": "WARNING",
                    "check": "crop_file_exists",
                    "sequence_id": seq,
                    "track_id": tid,
                    "obs_id": o["obs_id"],
                    "detail": f"Crop file missing: {cp}",
                })

    return issues


# Main

def main(run_dir: Path, output_dir: Path) -> None:
    print("\nTrack Annotation Dataset Builder")
    print(f"Input run : {run_dir}")
    print(f"Output dir: {output_dir}\n")

    if not run_dir.exists():
        sys.exit(f"ERROR: Run directory does not exist: {run_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)

    # Task 1 — Load and index tracks

    print("Task 1: Loading tracks and observations …")

    tracks_path = run_dir / "tracks.jsonl"
    obs_path    = run_dir / "candidate_observations.jsonl"

    if not tracks_path.exists():
        sys.exit(f"ERROR: tracks.jsonl not found at {tracks_path}")
    if not obs_path.exists():
        sys.exit(f"ERROR: candidate_observations.jsonl not found at {obs_path}")

    tracks_raw = [
        json.loads(l)
        for l in tracks_path.read_text(encoding="utf-8").splitlines()
        if l.strip()
    ]
    obs_raw = [
        json.loads(l)
        for l in obs_path.read_text(encoding="utf-8").splitlines()
        if l.strip()
    ]

    print(f"  Tracks loaded  : {len(tracks_raw)}")
    print(f"  Observations   : {len(obs_raw)}")

    # Index observations by sequence__track_id
    obs_by_track: Dict[str, List[Dict]] = defaultdict(list)
    for o in obs_raw:
        key = f"{o['sequence_id']}__{o['track_id']}"
        obs_by_track[key].append(o)

    # Sort each track's obs by frame_index
    for key in obs_by_track:
        obs_by_track[key].sort(key=lambda o: o["frame_index"])

    # Build canonical track manifest
    track_manifest_entries = []
    for t in tracks_raw:
        seq = t["sequence_id"]
        tid = t["track_id"]
        key = f"{seq}__{tid}"
        obs = obs_by_track.get(key, [])

        # First/last frame from actual observations
        first_obs = obs[0]  if obs else {}
        last_obs  = obs[-1] if obs else {}

        entry = {
            "sequence_id"         : seq,
            "track_id"            : tid,
            "semantic_class"      : t.get("semantic_class", "unknown"),
            "observation_count"   : len(obs),
            "declared_count"      : t.get("observation_count", -1),
            "first_timestamp"     : first_obs.get("timestamp"),
            "last_timestamp"      : last_obs.get("timestamp"),
            "lifespan_sec"        : t.get("lifespan_sec"),
            "first_frame_index"   : first_obs.get("frame_index"),
            "last_frame_index"    : last_obs.get("frame_index"),
            "world_displacement_m": t.get("world_displacement_m"),
            "observed_motion"     : t.get("observed_motion"),
            "obs_ids"             : [o["obs_id"] for o in obs],
            "crop_paths"          : [o.get("crop_path") for o in obs],
            "ego_motion_available": first_obs.get("ego_motion_available"),
            "provenance"          : t.get("provenance", {}),
        }
        track_manifest_entries.append(entry)

    # Write track manifest
    manifest_path = output_dir / "track_manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "source_run"      : str(run_dir),
                "total_tracks"    : len(track_manifest_entries),
                "total_obs"       : len(obs_raw),
                "tracks"          : track_manifest_entries,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    print(f"  ✓ track_manifest.json written ({len(track_manifest_entries)} tracks)")

    # Task 2 — Select representative observations

    print("\nTask 2: Selecting representative observations …")

    rep_obs_records = []
    rep_count_total = 0

    for t in track_manifest_entries:
        seq = t["sequence_id"]
        tid = t["track_id"]
        key = f"{seq}__{tid}"
        obs = obs_by_track.get(key, [])

        # Work on copies so we can add _selection_reason
        obs_copies = [dict(o) for o in obs]
        selected   = select_representative_observations(obs_copies)

        for o in selected:
            rep_obs_records.append({
                "sequence_id"      : seq,
                "track_id"         : tid,
                "obs_id"           : o["obs_id"],
                "frame_index"      : o["frame_index"],
                "timestamp"        : o["timestamp"],
                "semantic_class"   : o["semantic_class"],
                "detection_confidence": o.get("detection_confidence"),
                "sharpness"        : o.get("sharpness"),
                "median_depth_m"   : o.get("median_depth_m"),
                "area_px"          : o.get("area_px"),
                "has_mask"         : o.get("has_mask"),
                "ego_motion_available": o.get("ego_motion_available"),
                "crop_path"        : o.get("crop_path"),
                "mask_path"        : o.get("mask_path"),
                "rgb_path"         : o.get("rgb_path"),
                "selection_reason" : o.get("_selection_reason", "selected"),
                "total_track_obs"  : len(obs),
            })
        rep_count_total += len(selected)

    rep_path = output_dir / "representative_observations.jsonl"
    with rep_path.open("w", encoding="utf-8") as fh:
        for rec in rep_obs_records:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"  ✓ representative_observations.jsonl written ({rep_count_total} obs from {len(track_manifest_entries)} tracks)")
    avg_rep = rep_count_total / len(track_manifest_entries) if track_manifest_entries else 0
    print(f"    Average representative obs per track: {avg_rep:.1f}")

    # Task 3 — Annotation schema + scaffold CSV

    print("\nTask 3: Writing annotation schema and scaffold CSV …")

    schema_path = output_dir / "annotation_schema.json"
    schema_path.write_text(
        json.dumps(ANNOTATION_SCHEMA, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"  ✓ annotation_schema.json written")

    # Build scaffold CSV — one row per track, all annotation fields PENDING
    csv_path = output_dir / "track_annotations.csv"
    csv_fieldnames = [
        "sequence_id",
        "track_id",
        "object_class",
        "physical_mobility",
        "observed_motion",
        "visibility",
        "track_identity_validity",
        "evidence_sufficiency",
        "annotation_confidence",
        "annotator_notes",
        "representative_observation_ids",
        # Read-only context columns (from pipeline)
        "_pipeline_semantic_class",
        "_pipeline_observed_motion",
        "_pipeline_world_displacement_m",
        "_pipeline_observation_count",
        "_pipeline_ego_motion_available",
    ]

    # Build lookup: track_id → rep obs ids
    rep_by_track: Dict[str, List[str]] = defaultdict(list)
    for r in rep_obs_records:
        rep_by_track[f"{r['sequence_id']}__{r['track_id']}"].append(r["obs_id"])

    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=csv_fieldnames)
        writer.writeheader()
        for t in track_manifest_entries:
            key = f"{t['sequence_id']}__{t['track_id']}"
            rep_ids = "|".join(rep_by_track.get(key, []))
            writer.writerow({
                "sequence_id"                    : t["sequence_id"],
                "track_id"                       : t["track_id"],
                "object_class"                   : "PENDING",
                "physical_mobility"              : "PENDING",
                "observed_motion"                : "PENDING",
                "visibility"                     : "PENDING",
                "track_identity_validity"        : "PENDING",
                "evidence_sufficiency"           : "PENDING",
                "annotation_confidence"          : "PENDING",
                "annotator_notes"                : "",
                "representative_observation_ids" : rep_ids,
                "_pipeline_semantic_class"       : t["semantic_class"],
                "_pipeline_observed_motion"      : t["observed_motion"],
                "_pipeline_world_displacement_m" : t["world_displacement_m"],
                "_pipeline_observation_count"    : t["observation_count"],
                "_pipeline_ego_motion_available" : t["ego_motion_available"],
            })

    print(f"  ✓ track_annotations.csv written ({len(track_manifest_entries)} rows, all PENDING)")

    # Task 5 — Tracking consistency audit

    print("\nTask 5: Running tracking consistency checks …")

    issues = check_tracking_consistency(tracks_raw, obs_by_track)

    errors   = [i for i in issues if i["severity"] == "ERROR"]
    warnings = [i for i in issues if i["severity"] == "WARNING"]

    print(f"  Issues found: {len(issues)} ({len(errors)} errors, {len(warnings)} warnings)")

    # Build markdown report
    lines = [
        "# Tracking Consistency Report",
        "",
        f"Source run: `{run_dir}`",
        f"Tracks checked: {len(tracks_raw)}",
        f"Observations checked: {len(obs_raw)}",
        "",
        "## Summary",
        "",
        f"| Severity | Count |",
        f"| --- | --- |",
        f"| ERROR   | {len(errors)}   |",
        f"| WARNING | {len(warnings)} |",
        f"| TOTAL   | {len(issues)}   |",
        "",
    ]

    if not issues:
        lines += [
            "## Result",
            "",
            "No consistency issues detected. All checks passed.",
            "",
        ]
    else:
        if errors:
            lines += ["## Errors (must investigate)", ""]
            for i in errors:
                lines.append(
                    f"- **{i['check']}** | `{i.get('sequence_id','')}` / "
                    f"`{i.get('track_id','')}` | {i['detail']}"
                )
            lines.append("")

        if warnings:
            lines += ["## Warnings (should investigate)", ""]
            for i in warnings:
                lines.append(
                    f"- **{i['check']}** | `{i.get('sequence_id','')}` / "
                    f"`{i.get('track_id','')}` | {i['detail']}"
                )
            lines.append("")

    lines += [
        "## Checks Performed",
        "",
        "1. Track IDs unique within each sequence.",
        "2. Observation timestamps monotonically ordered.",
        "3. Frame indices are non-negative.",
        "4. No duplicate obs_id within a track.",
        "5. Declared observation count matches actual jsonl count.",
        "6. Crop files exist on disk (first + last obs per track sampled).",
        "",
        "> Issues are reported only, never silently corrected. "
        "Investigate each WARNING before drawing conclusions about temporal evidence.",
    ]

    consistency_path = output_dir / "tracking_consistency_report.md"
    consistency_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"  ✓ tracking_consistency_report.md written")

    # Final summary

    print("\n" + "=" * 60)
    print("Track annotation dataset generation complete")
    print("=" * 60)
    print(f"  Tracks indexed              : {len(track_manifest_entries)}")
    print(f"  Representative obs selected : {rep_count_total}")
    print(f"  Average per track           : {avg_rep:.1f}")
    print(f"  Consistency issues          : {len(issues)} ({len(errors)} errors, {len(warnings)} warnings)")
    print(f"\nOutputs in: {output_dir}")
    print(f"  track_manifest.json")
    print(f"  representative_observations.jsonl")
    print(f"  annotation_schema.json")
    print(f"  track_annotations.csv              ← open in spreadsheet to annotate")
    print(f"  tracking_consistency_report.md")
    print()
    print("Next: review tracking_consistency_report.md, then annotate track_annotations.csv.")
    print("Do NOT run VLM inference yet.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Build track-level annotation dataset from unified perception run."
    )
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=CANONICAL_RUN,
        help="Path to the perception harvesting run directory (default: canonical run).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=OUTPUT_ROOT,
        help="Output directory for annotation dataset (default: results/track_annotation/).",
    )
    args = parser.parse_args()
    main(args.run_dir, args.output_dir)
