# Implementation

## 1. Source Code Architecture (`vlm/src/`)

The core algorithms are implemented as modular Python modules in `vlm/src/`:

| File | Primary Classes / Functions | Responsibility |
|---|---|---|
| `geometric_evidence.py` | `GeometricEvidenceExtractor` | Processes 3D point clouds from depth images and 2D bounding boxes. Computes 3D bounding boxes, inlier ratios, and depth variance. |
| `temporal_evidence.py` | `TemporalEvidenceExtractor` | Tracks tracklet lifespan, observation frequency, frame gap intervals, and persistence scores. |
| `semantic_evidence.py` | `SemanticEvidenceExtractor` | Accumulates class prediction votes across frames, computes Shannon entropy, and provides class mobility priors. |
| `stage5_adapter.py` | `Stage5Adapter`, `load_evidence()` | Bundles geometric, temporal, and semantic records into unified observation contexts for routing. |
| `router_policy.py` | `RouterPolicy`, `EstimatorAwareRouter` | Implements policies B0 through B7, calculates estimator influence $I_{\text{estimator}}$, and evaluates query utility $U(q)$. |
| `selective_router.py` | `SelectiveRouter`, `build_affordance_prompt()` | High-level router orchestrator; formats prompts using the neutral ontology schema contract and communicates with local VLM endpoints. |
| `cross_session_association.py` | `CrossSessionAssociator`, `compute_se2_alignment()` | Intra-session tracklet consolidation, consensus RANSAC SE(2) registration between sessions, and Hungarian bipartite matching. |
| `contextual_scene_graph_fusion.py` | `ContextualSceneGraphFusion` | Constructs and maintains the 3D scene graph, manages landmark lifecycle actions, and decouples historical poses for moved objects. |
| `statistical_analysis.py` | `run_statistical_analysis()`, `wilson_ci()` | Computes Wilson score confidence intervals, McNemar exact tests, paired bootstrap differences, and Holm-Bonferroni corrections. |

---

## 2. Execution Scripts (`vlm/scripts/`)

Command-line entry points for executing evaluations, running ablations, and auditing outputs:

| Script | Purpose | Key Arguments |
|---|---|---|
| `run_paper_benchmark.py` | Unified benchmark entry point for running live visual evaluation, policy replay, and statistical reporting. | `--mode {visual-live, policy-replay, full}`<br>`--server-url <url>`<br>`--output-dir <path>` |
| `evaluate_visual_classification.py` | Evaluates visual affordance prediction under crop-only and crop-plus-context views. | `--inputs-manifest <path>`<br>`--ground-truth-manifest <path>`<br>`--view-mode {CROP_ONLY, CROP_PLUS_CONTEXT}` |
| `evaluate_router_policies.py` | Evaluates all router policies (B0–B7) against the benchmark dataset and computes accuracy, query rate, and cost. | `--inputs-manifest <path>`<br>`--ground-truth-manifest <path>`<br>`--output-dir <path>` |
| `evaluate_downstream_lifecycle.py` | Evaluates downstream SLAM landmark actions: retention recall, false invalidation, corruption risk, and agent rejection. | `--router-summary <path>`<br>`--output-dir <path>` |
| `audit_benchmark_population.py` | Scans all physical instances, observations, and image crops, reporting physical census and exclusion reasons. | `--output-dir <path>` |
| `validate_live_servers.py` | Smoke-tests the local VLM server, verifying response latency, HTTP status, and JSON schema compliance. | `--server-url <url>` |
| `validate_final_reports.py` | Audits the final deliverable package for metric agreement, proper denominators, and file consistency. | `--report-dir <path>` |

---

## 3. Dependencies and Environment

- **Python Version**: Python 3.10 or higher.
- **Key Python Packages** (`vlm/requirements.txt`):
  - `numpy`: Numerical calculations and matrix operations.
  - `scipy`: Statistical hypothesis testing and Hungarian bipartite assignment (`scipy.optimize.linear_sum_assignment`).
  - `scikit-learn`: Metric computations (confusion matrices, balanced accuracy, Macro-F1).
  - `pillow` (`PIL`): Image loading, cropping, and encoding for multimodal prompts.
  - `requests`: HTTP communication with local OpenAI-compatible VLM server endpoints.
  - `pytest`: Automated test execution.

---

## 4. Input and Output File Traceability

The following table maps each script to its required inputs and generated outputs:

```
[vlm/benchmarks/paper_benchmark_inputs_expanded.json]
[vlm/data/multiview_samples/]
       │
       ▼
evaluate_visual_classification.py
       │
       ▼
[visual_classification_predictions_*.jsonl]
[visual_classification_summary.json]
       │
       ├─────────────────────────────────────────────┐
       ▼                                             ▼
[vlm/benchmarks/router_benchmark_inputs_expanded.json]
       │
       ▼
evaluate_router_policies.py
       │
       ▼
[router_decisions_b*.jsonl]
[router_policies_summary.json]
       │
       ▼
evaluate_downstream_lifecycle.py
       │
       ▼
[downstream_lifecycle_summary.json]
[scene_graph.json]
       │
       ▼
statistical_analysis.py
       │
       ▼
[claim_to_evidence_matrix.md]
[paper_benchmark_report.md]
[statistical_metrics_summary.json]
```
