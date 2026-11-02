# Estimator-Context Feature Report — Stage 4

Generated: 2026-11-01T19:22:39Z
Source run: `C:\Users\Yash Bhardwaj\Desktop\vlm\results\perception_harvesting\run_20261101_234859`
Total feature vectors: 885

## Feature Summary Statistics

| Feature | Min | Median | Max | Mean |
| --- | --- | --- | --- | --- |
| detection_confidence | 0.400 | 0.600 | 0.970 | 0.635 |
| crop_area_px | 910.000 | 9006.000 | 175050.000 | 20888.880 |
| sharpness | 26.780 | 981.570 | 9029.030 | 1497.825 |
| median_depth_m | 0.336 | 2.543 | 7.942 | 2.704 |
| world_displacement_m | 0.021 | 0.572 | 4.470 | 0.853 |
| displacement_uncertainty_m | 0.010 | 0.147 | 1.441 | 0.254 |
| apparent_camera_frame_displacement_m | 0.023 | 0.961 | 5.566 | 1.341 |
| camera_ego_motion_displacement_m | 0.039 | 0.499 | 4.940 | 0.816 |
| track_obs_frac | 0.000 | 0.500 | 1.000 | 0.500 |
| lifespan_sec | 0.268 | 6.442 | 193.184 | 19.356 |
| total_detections | 5.000 | 31.000 | 339.000 | 64.734 |

## Causal Leakage Check

> [!IMPORTANT]
> `world_displacement_m`, `pipeline_motion_label`, `lifespan_sec`, and `total_detections`
> are **track-level** fields computed over the full sequence. They are NOT causal
> at the per-observation level — the policy cannot know end-of-sequence displacement
> when making a decision at observation time `t`.
>
> For a real-time causal query selector:
> - Use `track_obs_frac` and `timestamp_rel` as temporal position proxies.
> - Use `world_displacement_m` only as a training/evaluation signal, not as a runtime feature.
> - `detection_confidence`, `sharpness`, `median_depth_m`, `crop_area_px`, `ego_motion_available`
>   and `centroid_*` ARE fully causal — they derive from the current frame only.

## Causal Feature Subset (safe for runtime query selector)

| Feature | Category | Rationale |
| --- | --- | --- |
| detection_confidence | visual | YOLO score at current frame |
| crop_area_px | visual | Bounding box area at current frame |
| sharpness | visual | Laplacian of current crop |
| has_mask | visual | Mask present at current frame |
| median_depth_m | geometric | Depth at current frame |
| depth_valid | geometric | Depth validity at current frame |
| centroid_cam_* | geometric | 3D position at current frame |
| centroid_world_* | geometric | Ego-compensated position at current frame |
| ego_motion_available | geometric | Whether trajectory was used |
| track_obs_index | temporal | How many times object seen so far |
| track_obs_frac | temporal | Relative position in observed track window |
| timestamp_rel | temporal | Time since first detection |
| sequence_domain | contextual | Dataset type (fixed metadata) |
| semantic_class | contextual | YOLO class (available at detection time) |

## Non-Causal Fields (use only for offline analysis / training labels)

| Field | Why non-causal |
| --- | --- |
| world_displacement_m | End-of-sequence aggregate |
| pipeline_motion_label | End-of-sequence aggregate |
| lifespan_sec | Track end not known at decision time |
| total_detections | Track end not known at decision time |
| gt_vlm_label | Ground truth — never available at runtime |