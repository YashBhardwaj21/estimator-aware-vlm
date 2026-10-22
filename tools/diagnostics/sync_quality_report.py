#!/usr/bin/env python3
"""Generates a timestamp synchronization quality report for the TUM dataset (Stage 1 QA)."""

import argparse
from pathlib import Path
import numpy as np

from scene_graph.data.tum_loader import TUMLoader
from scene_graph.data.synchronization import associate


def print_association_stats(primary_ts, secondary_ts, matches, name):
    print(f"\n[{name} Synchronization]")
    if not matches:
        print("No matches found.")
        return
        
    p_len = len(primary_ts)
    s_len = len(secondary_ts)
    m_len = len(matches)
    
    print(f"Primary entries:   {p_len}")
    print(f"Secondary entries: {s_len}")
    print(f"Total Matches:     {m_len}")
    print(f"Primary coverage:    {(m_len/p_len)*100:.1f}%")
    print(f"Secondary util.:     {(m_len/s_len)*100:.1f}%")
    
    deltas = []
    matched_p = set()
    matched_s = set()
    
    for p_idx, s_idx in matches:
        deltas.append(abs(primary_ts[p_idx] - secondary_ts[s_idx]))
        matched_p.add(p_idx)
        matched_s.add(s_idx)
        
    deltas = np.array(deltas)
    print(f"Max delta-t:       {np.max(deltas):.6f} sec")
    print(f"Mean delta-t:      {np.mean(deltas):.6f} sec")
    print(f"Median delta-t:    {np.median(deltas):.6f} sec")
    print(f"95th percentile:   {np.percentile(deltas, 95):.6f} sec")
    
    unmatched_p = [i for i in range(p_len) if i not in matched_p]
    print(f"Unmatched primary count: {len(unmatched_p)}")
    if len(unmatched_p) > 0 and len(unmatched_p) <= 10:
        print(f"Unmatched primary indices: {unmatched_p}")
        print(f"Unmatched primary timestamps: {[primary_ts[i] for i in unmatched_p]}")


def report_for_window(rgb_entries, depth_entries, pose_entries, max_dt, start_idx=None, end_idx=None, title="FULL DATASET"):
    if start_idx is not None and end_idx is not None:
        rgb_entries = rgb_entries[start_idx:end_idx+1]
        
    rgb_ts = [e.timestamp for e in rgb_entries]
    depth_ts = [e.timestamp for e in depth_entries]
    pose_ts = [e.timestamp for e in pose_entries]
    
    print(f"\n[{title}]")
    print(f"RGB: {len(rgb_ts)}")
    print(f"Depth: {len(depth_ts)}")
    print(f"Pose: {len(pose_ts)}")
    
    rgb_depth_matches = associate(rgb_ts, depth_ts, max_dt=max_dt)
    print_association_stats(rgb_ts, depth_ts, rgb_depth_matches, "RGB <-> Depth")
    
    rgb_pose_matches = associate(rgb_ts, pose_ts, max_dt=max_dt)
    print_association_stats(rgb_ts, pose_ts, rgb_pose_matches, "RGB <-> Pose")
    
    # Calculate complete RGB-depth-pose
    depth_match_dict = dict(rgb_depth_matches)
    pose_match_dict = dict(rgb_pose_matches)
    
    complete = 0
    for i in range(len(rgb_ts)):
        if i in depth_match_dict and i in pose_match_dict:
            complete += 1
            
    print(f"\n[Complete Frame Packets]")
    print(f"RGB-depth matched: {len(rgb_depth_matches)}")
    print(f"RGB-pose matched: {len(rgb_pose_matches)}")
    print(f"Complete RGB-depth-pose: {complete}")


def main():
    parser = argparse.ArgumentParser(description="TUM Synchronization QA")
    parser.add_argument("--sequence_dir", type=str, default="rgbd_dataset_freiburg1_desk")
    parser.add_argument("--max_dt", type=float, default=0.02)
    args = parser.parse_args()
    
    seq_dir = Path(args.sequence_dir)
    if not seq_dir.is_dir():
        print(f"Dataset directory not found: {seq_dir}")
        return
        
    loader = TUMLoader(seq_dir)
    rgb = loader.load_rgb()
    depth = loader.load_depth()
    pose = loader.load_groundtruth()
    
    # Full dataset report
    report_for_window(rgb, depth, pose, args.max_dt, title="FULL DATASET")
    
    # Experiment window report
    # Window is 100 to 300 inclusive
    if len(rgb) > 300:
        report_for_window(rgb, depth, pose, args.max_dt, start_idx=100, end_idx=300, title="EXPERIMENT WINDOW [100, 300]")


if __name__ == "__main__":
    main()
