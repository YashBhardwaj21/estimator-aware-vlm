# Estimator-Aware VLM for Dynamic SLAM — Experiment Results

This directory contains permanent, reproducible stage-by-stage experimental results for the Estimator-Aware VLM research project.

## Directory Structure

```text
results/
├── README.md
├── perception_harvesting/          # Stage 1: YOLOv8-Seg tracking & crop harvesting
│   └── run_20261101_234859/        # Canonical run — 885 causal observations, 177 tracks
│       ├── candidate_observations.jsonl  # All 885 VLM-eligible observation records
│       ├── crops/                        # Per-observation image crops (PNG)
│       ├── masks/                        # Segmentation masks (PNG)
│       └── rgb_frames/                   # Full context RGB frames (PNG)
├── track_annotation/               # Stage 2: Track-level annotation & evaluation split
│   ├── track_annotations.csv             # 177 tracks × annotation columns
│   ├── evaluation_split.json             # 4-partition split (development / validation / held_out / perception_only)
│   ├── representative_observations.jsonl # 885 representative observations with metadata
│   ├── track_manifest.json               # Track inventory with per-sequence counts
│   ├── annotation_schema.json            # Schema definition for annotation fields
│   ├── ego_motion_provenance_report.md   # Ego-motion computation provenance
│   ├── annotation_dataset_report.md      # Summary statistics
│   ├── annotation_audit.md               # Quality audit
│   └── tracking_consistency_report.md    # Cross-sequence consistency check
├── vlm_evaluation/                 # Stage 3: Controlled VLM evaluation (4 prompt configs × 30 dev tracks)
│   └── run_20261101_193504/        # Canonical run — 120 inference records
│       ├── inference_records.jsonl        # Full VLM I/O (prompt, response, label, latency)
│       ├── evaluation_metrics.json        # Per-config metrics: V1=0.567, V2=0.379, V3=0.444, V4=0.544
│       └── evaluation_summary.md          # Human-readable evaluation report
├── estimator_context/              # Stage 4: Causal estimator-context feature vectors
│   ├── feature_vectors.jsonl             # 885 causal feature vectors (no future leakage)
│   ├── feature_schema.json               # Schema: 36 features, causal-guarantee flags
│   └── context_report.md                 # Feature distribution statistics
├── query_policy/                   # Stage 5: Query selection policy benchmark (P0–P7 × 7 budgets)
│   ├── benchmark_summary.json            # Per-policy × per-budget metrics
│   ├── query_decisions.jsonl             # Per-observation query decision records
│   └── policy_comparison_table.md        # Markdown comparison table
├── decision_fusion/                # Stage 6: Bayesian log-odds evidence fusion
│   ├── fused_interventions.jsonl         # 885 records — p_fused, MASK_DYNAMIC / DOWNWEIGHT / RETAIN
│   ├── fusion_metrics.json               # Track-level & obs-level precision/recall/F1
│   └── fusion_report.md                  # Full report: geometry breakdown, VLM coverage, anomaly explanations
├── native_estimator_integration/   # Stage 7/8: Native ORB-SLAM3 factorial experiment
│   └── run_20261101_native/        # Canonical run — I0–I3 factorial + live VLM + EDRR
│       ├── native_paired_factorial_matrix.csv
│       ├── live_vlm_query_records.jsonl
│       ├── telemetry_I0_baseline_control/
│       ├── telemetry_I1_image_mask_only/
│       ├── telemetry_I2_landmark_only_integration/
│       ├── telemetry_I3_dual_integration/
│       ├── NATIVE_SLAM_INTEGRATION_REPORT.md
│       └── EDRR_GROUPED_VALIDATION_REPORT.md
├── statistical_analysis/           # Stage 9: McNemar test & ablation study
│   ├── statistical_significance_report.json
│   └── ablation_summary.json
└── publication_campaign/           # Stage 10: Camera-ready manuscript
    └── CAMERA_READY_RESEARCH_MANUSCRIPT.md
```

## Governance & Integrity Protocols

1. **Permanence**: All final benchmark evaluations, trajectories, metrics, logs, and figures belong here in `results/`. Temporary or intermediate files belong in `scratch/` and must be purged after verified promotion. Obsolete runs are archived under `archive/vlm_publish/` (never deleted).
2. **Zero Fabrication**: When ground-truth reference data (e.g., optical MoCap) is unavailable for a dataset (e.g., RealSense D455 handheld capture), trajectory ATE and reference coverage are recorded explicitly as `null`/`None`. No synthetic identity placeholders or heuristic estimates may substitute for physical measurements.
3. **Traceability**: Every experiment run directory contains provenance records (run timestamps, input JSONL hashes, software environment) linking all results to exact reproducible conditions.
4. **Causal Integrity**: Feature vectors in `estimator_context/` and query decisions in `query_policy/` are produced under strict no-future-leakage constraints. The feature schema (`feature_schema.json`) documents per-field causal guarantees.
5. **Canonical Run IDs**: The authoritative perception run is `run_20261101_234859` (885 observations) and the authoritative VLM evaluation run is `run_20261101_193504` (120 records, 4 configs × 30 dev-split tracks). All downstream stages consume outputs from these canonical runs.

