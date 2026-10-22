# Usage Guide

## 1. Environment Setup

### 1.1 Install Python Dependencies
From the repository root:
```bash
pip install -r vlm/requirements.txt
```

### 1.2 Verify Test Suite
Run the automated pytest suite:
```bash
pytest vlm/tests/ -q
```
All 232 tests should pass without errors.

---

## 2. Serving the Local Vision-Language Model

The subsystem uses **Qwen2.5-VL-3B-Instruct** served via `llama.cpp` (`llama-server`).

### 2.1 Launch llama-server
```bash
llama-server \
  -m /path/to/Qwen2.5-VL-3B-Instruct-Q4_K_M.gguf \
  --mmproj /path/to/mmproj-Qwen2.5-VL-3B-Instruct-Q8_0.gguf \
  --host 127.0.0.1 \
  --port 8089 \
  -c 4096 \
  -t 4
```

### 2.2 Verify Server Health
Check the HTTP health status:
```bash
curl http://127.0.0.1:8089/health
```

Run the preflight validation script:
```bash
python vlm/scripts/validate_live_servers.py --server-url http://127.0.0.1:8089
```

---

## 3. Running Benchmark Workflows

### 3.1 Policy Replay Benchmark (Deterministic Offline)
Evaluates all routing policies (B0–B7) against the 53-entity benchmark dataset using pre-computed visual model predictions:
```bash
python vlm/scripts/run_paper_benchmark.py --mode policy-replay
```
Outputs are written to `vlm/outputs/benchmark_runs/paper_benchmark_<timestamp>/`:
- `router_decisions_b0_static_majority.jsonl`
- `router_decisions_b1_semantic_prior.jsonl`
- `router_decisions_b2_geometry_motion.jsonl`
- `router_decisions_b3_zero_shot_crop_only.jsonl`
- `router_decisions_b4_zero_shot_crop_plus_context.jsonl`
- `router_decisions_b5_always_vlm.jsonl`
- `router_decisions_b6_estimator_aware_router.jsonl`
- `router_decisions_b7_oracle.jsonl`
- `router_policies_summary.json`
- `downstream_lifecycle_summary.json`
- `claim_to_evidence_matrix.md`
- `paper_benchmark_report.md`
- `statistical_metrics_summary.json`

### 3.2 Live Visual Benchmark
Queries the live VLM server on the 42 physical visual benchmark instances under both crop-only and crop-plus-context views:
```bash
python vlm/scripts/run_paper_benchmark.py \
  --mode visual-live \
  --server-url http://127.0.0.1:8089 \
  --inputs-manifest vlm/benchmarks/paper_benchmark_inputs_expanded.json \
  --ground-truth-manifest vlm/benchmarks/paper_benchmark_ground_truth_expanded.json
```

Outputs are saved in the generated benchmark run directory:
- `visual_classification_predictions_qwen2_5_vl_3b_crop_only.jsonl`
- `visual_classification_predictions_qwen2_5_vl_3b_crop_plus_context.jsonl`
- `visual_classification_summary.json`

---

## 4. Running Component Evaluations Independently

### 4.1 Visual Classification Evaluation Only
```bash
python vlm/scripts/evaluate_visual_classification.py \
  --server-url http://127.0.0.1:8089 \
  --inputs-manifest vlm/benchmarks/paper_benchmark_inputs_expanded.json \
  --ground-truth-manifest vlm/benchmarks/paper_benchmark_ground_truth_expanded.json \
  --output-dir vlm/outputs/visual_eval/
```

### 4.2 Router Policy Evaluation Only
```bash
python vlm/scripts/evaluate_router_policies.py \
  --inputs-manifest vlm/benchmarks/router_benchmark_inputs_expanded.json \
  --ground-truth-manifest vlm/benchmarks/router_benchmark_ground_truth_expanded.json \
  --output-dir vlm/outputs/router_eval/
```

### 4.3 Downstream Lifecycle Evaluation Only
```bash
python vlm/scripts/evaluate_downstream_lifecycle.py \
  --router-summary vlm/outputs/router_eval/router_policies_summary.json \
  --output-dir vlm/outputs/lifecycle_eval/
```

### 4.4 Population Audit
Scans physical instances, observation records, and multi-view crop assets to report a census and exclusion ledger:
```bash
python vlm/scripts/audit_benchmark_population.py \
  --output-dir vlm/outputs/benchmark_audit/
```

### 4.5 Package Validation
Audits deliverable packages for mathematical consistency across tables, summaries, and per-instance JSONL records:
```bash
python vlm/scripts/validate_final_reports.py \
  --report-dir vlm/outputs/final_report/
```

---

## 5. Output Locations

| Directory | Contents |
|---|---|
| `vlm/outputs/benchmark_runs/<run_id>/` | Individual benchmark execution records, per-instance JSONL decisions, latency logs, and intermediate summaries. |
| `vlm/outputs/final_report/` | Final deliverables: `FINAL_RESULTS.md`, `model_comparison.md`, `model_comparison.csv`, `router_policy_comparison.csv`, and SVG/PNG publication figures. |
| `vlm/outputs/benchmark_audit/` | Population audit reports: `population_audit.csv`, `excluded_instances.csv`, and `population_audit_report.md`. |
