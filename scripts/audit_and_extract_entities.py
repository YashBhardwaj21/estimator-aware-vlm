"""Audit and cluster tracked objects across recordings to establish physical object ground truth."""

import json
from pathlib import Path
import numpy as np

def analyze_recording(seq_name: str, map_dir: Path):
    map_path = map_dir / "semantic_map.json"
    aliases_path = map_dir / "entity_aliases.json"
    
    with open(map_path) as f:
        data = json.load(f)
    
    entities_raw = data.get("entities", {})
    if isinstance(entities_raw, dict):
        entities = list(entities_raw.values())
    else:
        entities = entities_raw
    
    aliases = {}
    if aliases_path.exists():
        with open(aliases_path) as f:
            aliases = json.load(f)
            
    print(f"\n=======================================================")
    print(f"RECORDING: {seq_name}")
    print(f"Total confirmed entity tracks: {len(entities)}")
    print(f"=======================================================")
    
    parsed = []
    for e in entities:
        eid = e.get("entity_id", e.get("id"))
        cname = e.get("class_name", e.get("category"))
        pos = e.get("centroid_floor", e.get("center", e.get("centroid")))
        dims = e.get("dimensions", e.get("dimensions_m", e.get("extent")))
        obs_cnt = e.get("observation_count", e.get("num_observations", 0))
        mean_conf = e.get("mean_confidence", 0.0)
        f_start = e.get("first_frame_idx", 0)
        f_end = e.get("last_frame_idx", 0)
        t_span = e.get("last_seen_timestamp", 0.0) - e.get("first_seen_timestamp", 0.0)
        
        if pos is not None:
            pos = [round(float(x), 3) for x in pos]
        if dims is not None:
            dims = [round(float(x), 3) for x in dims]
            
        parsed.append({
            "id": eid,
            "class": cname,
            "centroid": pos,
            "dims": dims,
            "obs_cnt": obs_cnt,
            "mean_conf": round(mean_conf, 3),
            "frame_range": f"{f_start}->{f_end}",
            "time_span_s": round(t_span, 2),
            "raw": e
        })
        
    # Spatial clustering to find duplicate tracks of the same physical object
    # Distance threshold: 0.40m between centroids of same or compatible class
    clusters = []
    used = set()
    
    for i, e1 in enumerate(parsed):
        if i in used or e1["centroid"] is None:
            continue
        current_cluster = [e1]
        used.add(i)
        p1 = np.array(e1["centroid"][:2]) # XY floor distance
        
        for j, e2 in enumerate(parsed):
            if j in used or e2["centroid"] is None:
                continue
            p2 = np.array(e2["centroid"][:2])
            dist = np.linalg.norm(p1 - p2)
            
            # Same physical object if centroids are within 0.35m
            if dist < 0.35:
                current_cluster.append(e2)
                used.add(j)
                
        clusters.append(current_cluster)
        
    print(f"Resolved {len(parsed)} raw tracks into {len(clusters)} distinct spatial physical objects:")
    print("-----------------------------------------------------------------------------------------")
    
    for idx, cl in enumerate(clusters):
        track_ids = [c["id"] for c in cl]
        classes = list(set(c["class"] for c in cl))
        avg_pos = np.mean([c["centroid"] for c in cl], axis=0).round(2).tolist()
        total_obs = sum(c["obs_cnt"] for c in cl)
        mean_c = round(float(np.mean([c["mean_conf"] for c in cl])), 2)
        print(f"  Obj_{idx+1:02d}: classes={classes}, pos={avg_pos}, obs={total_obs}, conf={mean_c}, tracks={track_ids}")
        
    return clusters, parsed

if __name__ == "__main__":
    def resolve_mapping_path(name: str) -> Path:
        p_vlm = Path("vlm/data") / name
        if p_vlm.exists():
            return p_vlm
        return Path("outputs") / name

    r1 = resolve_mapping_path("semantic_mapping_desk_run_20261009_161552")
    r2 = resolve_mapping_path("semantic_mapping_d455_rgbd_20261010_123443")
    
    c1, p1 = analyze_recording("desk_run_20261009_161552", r1)
    c2, p2 = analyze_recording("d455_rgbd_20261010_123443", r2)
