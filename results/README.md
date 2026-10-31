# Estimator-Aware VLM Counterfactual SLAM Intervention Results

This directory contains permanent, reproducible experimental results for the Estimator-Aware VLM research project.

## Directory Structure

```text
results/
├── README.md
└── counterfactual_interventions/
    └── <experiment_run_id>/
        ├── experiment_manifest.json
        ├── dataset_inventory.json
        ├── intervention_manifest.json
        ├── counterfactual_results.csv
        ├── counterfactual_results.json
        ├── provenance/
        │   ├── input_hashes.json
        │   ├── source_commit.json
        │   └── runtime_environment.json
        ├── tum/
        │   ├── baseline_metrics.json
        │   ├── intervention_metrics.csv
        │   ├── track_level_results.json
        │   ├── fast_feature_retention.json
        │   ├── diagnostics/
        │   ├── trajectories/
        │   └── figures/
        ├── openloris/
        │   ├── baseline_metrics.json
        │   ├── intervention_metrics.csv
        │   ├── track_level_results.json
        │   ├── vlm_failure_analysis.json
        │   ├── trajectories/
        │   └── figures/
        ├── d455/
        │   ├── session_01_metrics.json
        │   ├── session_02_metrics.json
        │   ├── identity_association_metrics.json
        │   ├── relocation_metrics.json
        │   ├── lifecycle_metrics.json
        │   └── figures/
        ├── aggregate/
        │   ├── dataset_comparison.csv
        │   ├── dataset_comparison.json
        │   ├── statistical_analysis.json
        │   └── figures/
        ├── logs/
        └── REPORT.md
```

## Governance & Integrity Protocols

1. **Permanence**: All final benchmark evaluations, trajectories, metrics, logs, and figures belong here in `results/`. Temporary intermediate files belong in `scratch/` and must be purged after verified promotion.
2. **Zero Fabrication**: When ground-truth reference data (e.g., optical MoCap) is unavailable for a dataset (e.g., RealSense D455 handheld capture), trajectory ATE and reference coverage are recorded explicitly as `null`/`None`. No synthetic identity placeholders or heuristic estimates may substitute for physical measurements.
3. **Traceability**: Every experiment directory contains a cryptographically signed provenance record linking all results to exact input SHA-256 checksums, Git commit SHA, and software environment configuration.
4. **Historical Isolation**: Canonical benchmark releases (`benchmarks/release_results/v1_baseline/`, `v2_verified/`) and raw sensor datasets are strictly read-only and immutable.

