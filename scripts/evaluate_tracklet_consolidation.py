#!/usr/bin/env python3
"""Evaluate Upstream Tracklet Consolidation and Its Downstream Effects on Cross-Session Association.

Measures:
1. Front-end track fragmentation reduction (72.78% raw tracks <= 4 frames).
2. Intra-session spatial-temporal cluster consolidation (< 0.35m threshold).
3. Cross-session physical instance association recall and false merge rate before vs after.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, List
import numpy as np

repo_root = Path(__file__).resolve().parents[1]
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from estimator_aware_vlm.cross_session_association import (
    CrossSessionCandidateMatcher,
    compare_association_baselines,
    build_canonical_session_track_mapping,
)
from estimator_aware_vlm.evidence_adapter import load_registration_audit


def run_tracklet_consolidation_audit(
    obs_path: Path | None = None,
    reg_path: Path | None = None,
    gt_path: Path | None = None,
    out_dir: Path | None = None,
    max_merge_distance_m: float = 0.35,
) -> Dict[str, Any]:
    repo_root = Path(__file__).resolve().parents[1]
    if obs_path is None:
        obs_path = repo_root / "outputs" / "object_observations.jsonl"
    if reg_path is None:
        reg_path = repo_root / "outputs" / "registration_audit.json"
    if gt_path is None:
        gt_path = repo_root / "benchmarks" / "cross_session_identity_annotations.json"
    if out_dir is None:
        out_dir = repo_root / "outputs"

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load observations
    observations: List[Dict[str, Any]] = []
    with open(obs_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                observations.append(json.loads(line))

    # 2. Compute front-end raw track length statistics
    s1_track_lengths: Dict[str, int] = {}
    s2_track_lengths: Dict[str, int] = {}
    for o in observations:
        sess = o.get("source_session", "")
        tid = o.get("local_track_id", "")
        if not tid:
            continue
        if "recording_01" in sess:
            s1_track_lengths[tid] = s1_track_lengths.get(tid, 0) + 1
        elif "recording_02" in sess:
            s2_track_lengths[tid] = s2_track_lengths.get(tid, 0) + 1

    all_tracks = {**s1_track_lengths, **{f"s2_{k}": v for k, v in s2_track_lengths.items()}}
    total_tracks_count = len(all_tracks)
    short_tracks_count = sum(1 for v in all_tracks.values() if v <= 4)
    short_track_ratio = (short_tracks_count / max(1, total_tracks_count)) * 100.0

    # 3. Extract raw clusters
    s1_raw = CrossSessionCandidateMatcher.extract_clusters_from_observations(
        observations, "recording_01", min_obs=2
    )
    s2_raw = CrossSessionCandidateMatcher.extract_clusters_from_observations(
        observations, "recording_02", min_obs=2
    )

    # 4. Consolidate clusters
    s1_cons = CrossSessionCandidateMatcher.consolidate_intra_session_clusters(
        s1_raw, max_merge_distance_m=max_merge_distance_m
    )
    s2_cons = CrossSessionCandidateMatcher.consolidate_intra_session_clusters(
        s2_raw, max_merge_distance_m=max_merge_distance_m
    )

    # 5. Load registration and run association before vs after
    reg = load_registration_audit(reg_path)
    baselines_raw = compare_association_baselines(s1_raw, s2_raw, reg, gt_path)
    baselines_cons = compare_association_baselines(s1_cons, s2_cons, reg, gt_path)

    audit_report = {
        "metadata": {
            "title": "Upstream Tracklet Consolidation Audit",
            "max_merge_distance_m": max_merge_distance_m,
            "observations_file": str(obs_path),
            "registration_file": str(reg_path),
        },
        "raw_tracking_fragmentation": {
            "total_unique_tracks": total_tracks_count,
            "s1_unique_tracks": len(s1_track_lengths),
            "s2_unique_tracks": len(s2_track_lengths),
            "short_tracks_le_4_frames": short_tracks_count,
            "short_track_ratio_pct": round(short_track_ratio, 2),
        },
        "cluster_consolidation": {
            "session_1": {
                "raw_clusters_count": len(s1_raw),
                "consolidated_clusters_count": len(s1_cons),
                "clusters_merged_count": len(s1_raw) - len(s1_cons),
                "reduction_ratio_pct": round((1.0 - len(s1_cons) / max(1, len(s1_raw))) * 100.0, 2),
            },
            "session_2": {
                "raw_clusters_count": len(s2_raw),
                "consolidated_clusters_count": len(s2_cons),
                "clusters_merged_count": len(s2_raw) - len(s2_cons),
                "reduction_ratio_pct": round((1.0 - len(s2_cons) / max(1, len(s2_raw))) * 100.0, 2),
            },
        },
        "association_comparison": {
            "before_consolidation": {
                name: {
                    "matched_pairs_count": len(data["result"].matched_pairs),
                    "annotated_eval_pairs": data["evaluation"]["annotated_eval_pairs_count"],
                    "true_positive_matches": data["evaluation"]["true_positive_matches"],
                    "false_merges": data["evaluation"]["false_merges"],
                    "candidate_precision_pct": data["evaluation"]["precision_on_annotated_subset_pct"],
                    "static_recall_pct": data["evaluation"]["recall_on_static_multi_session_pct"],
                    "recovered_instances_count": data["evaluation"]["recovered_all_instances_count"],
                    "total_instances_count": data["evaluation"]["total_multi_session_instances"],
                }
                for name, data in baselines_raw.items()
            },
            "after_consolidation": {
                name: {
                    "matched_pairs_count": len(data["result"].matched_pairs),
                    "annotated_eval_pairs": data["evaluation"]["annotated_eval_pairs_count"],
                    "true_positive_matches": data["evaluation"]["true_positive_matches"],
                    "false_merges": data["evaluation"]["false_merges"],
                    "candidate_precision_pct": data["evaluation"]["precision_on_annotated_subset_pct"],
                    "static_recall_pct": data["evaluation"]["recall_on_static_multi_session_pct"],
                    "recovered_instances_count": data["evaluation"]["recovered_all_instances_count"],
                    "total_instances_count": data["evaluation"]["total_multi_session_instances"],
                }
                for name, data in baselines_cons.items()
            },
        },
    }

    # Save JSON output
    out_json = out_dir / "tracklet_consolidation_audit.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(audit_report, f, indent=2)

    return audit_report


if __name__ == "__main__":
    rep = run_tracklet_consolidation_audit()
    print("=" * 80)
    print("UPSTREAM TRACKLET CONSOLIDATION AUDIT COMPLETE")
    print("=" * 80)
    print(f"• Total Raw Tracks:         {rep['raw_tracking_fragmentation']['total_unique_tracks']}")
    print(f"• Short Tracks (<= 4 frm):  {rep['raw_tracking_fragmentation']['short_tracks_le_4_frames']} ({rep['raw_tracking_fragmentation']['short_track_ratio_pct']}%)")
    print(f"• S1 Clusters:              {rep['cluster_consolidation']['session_1']['raw_clusters_count']} -> {rep['cluster_consolidation']['session_1']['consolidated_clusters_count']} (-{rep['cluster_consolidation']['session_1']['reduction_ratio_pct']}%)")
    print(f"• S2 Clusters:              {rep['cluster_consolidation']['session_2']['raw_clusters_count']} -> {rep['cluster_consolidation']['session_2']['consolidated_clusters_count']} (-{rep['cluster_consolidation']['session_2']['reduction_ratio_pct']}%)")
    print("=" * 80)
