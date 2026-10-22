# VLM Assignment Bundle: Dynamic 3D Scene Graph

This directory (`vlm/assignment/`) is a clean, self-contained, verified bundle containing all source snapshots, benchmarks, ground truth annotations, raw observation data, derived metrics, research-quality figures, documentation, and review prompts for the **Dynamic 3D Scene Graph** assignment.

> [!IMPORTANT]
> **Audit-First, Non-Destructive Guarantee:**  
> All files in this bundle were copied and verified using SHA-256 cryptographic hashing. Not a single file in the original repository (`vlm/src`, `vlm/scripts`, `vlm/benchmarks`, `vlm/outputs`, `vlm/tests`) was moved, modified, renamed, or deleted.

---

## 1. Directory Structure

```text
vlm/assignment/
├── README.md                                # Root bundle overview and replication guide
├── manifest.json                            # Cryptographic SHA-256 provenance manifest (346 entries)
├── docs/                                    # Technical documentation and audit analyses
│   ├── PROJECT_OVERVIEW.md                  # Scope, architecture, and current vs incomplete states
│   ├── PIPELINE_AND_DATA_FLOW.md            # End-to-end data flow with functions and schemas
│   ├── METRIC_DICTIONARY.md                 # Mathematical definitions, denominators, and caveats
│   └── KNOWN_ISSUES.md                      # Confirmed and suspected algorithmic issues
├── prompts/
│   └── AI_REVIEW_PROMPT.md                  # Strict, non-guessing diagnostic review prompt
├── source_snapshot/                         # Frozen snapshot of experimental source code
│   ├── src/                                 # Core modules (fusion, registration, router, evidence)
│   ├── scripts/                             # Evaluation and experiment runners
│   └── tests/                               # 99 passing unit & integration tests + fixtures
├── benchmarks/                              # Benchmark definitions and ground truth
│   ├── ground_truth/                        # Canonical cross-session identity & movability GT
│   ├── registration/                        # Robust SE(2) anchor correspondence & residual audit
│   └── protocols/                           # Benchmark reports and evaluation protocols
├── data/                                    # Input sensory data and visual crops
│   ├── observations/                        # Raw observations JSONL (7,956 records)
│   └── multiview_samples/                   # Multi-view RGB crops across 39 physical instances
├── metrics/                                 # Raw metrics and derived evaluation state
│   ├── raw/                                 # Evidence metrics, VLM cache, ablation outputs
│   ├── derived/                             # Final scene graph (195 nodes), baseline comparisons
│   └── summary.json                         # Reconciled multi-stage summary with true denominators
├── figures/                                 # Research-grade figures (SVG + PNG) and plotting script
│   ├── association/                         # Baseline performance & error breakdown
│   ├── registration/                        # Anchor residuals, inlier thresholds, 5-fold CV
│   ├── displacement/                        # Controlled relocation error & uncertainty
│   ├── semantic_evidence/                   # Coverage, Macro-F1, temporal track fragmentation
│   ├── router/                              # Cost vs accuracy ablation trade-offs
│   └── generate_figures.py                  # Standalone figure generation script
└── reports/                                 # Formal evaluation reports
    ├── source_reports/                      # Historical Stage 5, Stage 6, Affordance, and Baseline reports
    └── audit/                               # Semantic association audit
```

---

## 2. Source-to-Copy Mapping & Provenance Manifest

Every file copied into this bundle is catalogued in [`manifest.json`](manifest.json) with:
- Original repository relative path
- Assignment-relative destination path
- File size in bytes
- Cryptographic SHA-256 digest
- Verification status (`VERIFIED` or `VERIFIED_WITH_WARNINGS`)
- Categorization and functional purpose

### Large Artifact Handling & Omissions
To ensure a compact, focused assignment bundle without losing reproducibility:
1. **Included:** `data/observations/object_observations.jsonl` (6.48 MB, 7,956 valid records) is fully included to ensure zero external data dependencies.
2. **Included:** `data/multiview_samples/` (273 multi-view crops across 39 physical object instances) is included for complete visual affordance replication.
3. **Selectively Omitted:**
   - `vlm/outputs/crops/` (170 loose track crops, ~8.6 MB): Omitted as redundant with structured `multiview_samples/`.
   - `vlm/outputs/stage5_*.jsonl` (~26 MB total intermediate evidence): Omitted because they are deterministically regenerable in 8 seconds from `object_observations.jsonl` via `source_snapshot/src/stage5_pipeline.py`.
   - All omissions are formally recorded in `manifest.json` under `excluded_artifacts`.

---

## 3. Visualization Index

Publication-quality figures are generated in both **SVG** and **PNG** formats:

| Category | Figure Title | Formats | Plotted Ground-Truth & Metric Sources |
| :--- | :--- | :--- | :--- |
| **Association** | [Association Baselines Performance](figures/association/association_baselines_performance.png) | [SVG](figures/association/association_baselines_performance.svg), [PNG](figures/association/association_baselines_performance.png) | Precision, instance recall (23.33%), relocation recall (25%), instance F1 across 4 baselines |
| **Association** | [Association Error Breakdown](figures/association/association_errors_breakdown.png) | [SVG](figures/association/association_errors_breakdown.svg), [PNG](figures/association/association_errors_breakdown.png) | False merge rate (41.7%), false split rate (76.7%), per-instance outcome breakdown (N=30) |
| **Registration** | [Anchor Residuals & Inliers](figures/registration/anchor_residuals_and_inliers.png) | [SVG](figures/registration/anchor_residuals_and_inliers.svg), [PNG](figures/registration/anchor_residuals_and_inliers.png) | 10 SE(2) anchors, 0.20m inlier gate, inlier RMS (0.1198m) vs all-anchor RMS (0.3087m), 5-fold CV |
| **Displacement** | [Controlled Relocation Errors](figures/displacement/controlled_relocation_errors.png) | [SVG](figures/displacement/controlled_relocation_errors.svg), [PNG](figures/displacement/controlled_relocation_errors.png) | Tape GT vs estimate, absolute error, relative error (30.42% mean), empirical uncertainty |
| **Semantics** | [Semantic Evidence & Coverage](figures/semantic_evidence/semantic_macro_f1_and_coverage.png) | [SVG](figures/semantic_evidence/semantic_macro_f1_and_coverage.svg), [PNG](figures/semantic_evidence/semantic_macro_f1_and_coverage.png) | Evaluation coverage (55.57% evaluated vs 44.43% unmapped), matched Macro-F1 (99.76%), per-class support |
| **Temporal** | [Temporal Track Lifespans](figures/semantic_evidence/temporal_track_lifespans.png) | [SVG](figures/semantic_evidence/temporal_track_lifespans.svg), [PNG](figures/semantic_evidence/temporal_track_lifespans.png) | Track fragmentation rate (72.78% <=4 frames), multi-track rate (79.2%), occlusion recovery (100%) |
| **Router** | [Router Efficiency vs Accuracy](figures/router/router_efficiency_vs_accuracy.png) | [SVG](figures/router/router_efficiency_vs_accuracy.svg), [PNG](figures/router/router_efficiency_vs_accuracy.png) | Selective VLM (88.89% acc, 26.67% query rate) vs M0 Prior vs M3 All-VLM vs Oracle |

### How to Regenerate Figures
To regenerate all figures from the raw verified metrics:
```bash
python vlm/assignment/figures/generate_figures.py
```

---

## 4. How to Run the Tests

To run the complete test suite:
```bash
pytest vlm/tests/ -v
```
All 99 unit and integration tests pass in ~7 seconds, validating registration stability, multi-view crop schemas, evidence adapter normalization, and full pipeline execution contracts. The test fixtures are also snapshotted in `source_snapshot/tests/fixtures/real_pipeline_fixture/`.
