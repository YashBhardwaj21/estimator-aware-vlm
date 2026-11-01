# AUDIT REPORT: VLM PROVENANCE, PROMPT LEAKAGE, AND RESPONSE-BANK REPLAY MECHANICS

**Audit Document ID**: AUD-REP-03-VLM-PROVENANCE  
**Campaign Audited**: `results/publication_campaign/run_20261101_execution`  
**Classification**: `VERIFIED PHYSICAL INFERENCE WITH METHODOLOGICAL LEAKAGE`  
**Auditor**: Independent Research Quality & Experimental Integrity Auditor  
**Date**: 2026-11-01  

---

## 1. Executive Summary

This audit evaluates the Vision-Language Model (VLM) execution in the experimental campaign. Two primary aspects were audited:
1. **Physical Authenticity**: Whether the reported 36 Qwen inferences were physically executed against the local llama.cpp server on port 8089.
2. **Methodological Validity**: Whether prompt structures maintained independent semantic reasoning or inadvertently leaked estimator state, and whether routing policies operated as live closed-loop decision agents or offline response replays.

The audit confirms that **36 live HTTP inferences were physically executed** against `Qwen2.5-VL-3B-Instruct-Q4_K_M.gguf`. However, two significant methodological vulnerabilities were uncovered:
- **Displacement Leakage**: The prompt supplied the VLM with quantitative 3D displacement measurements (`displacement: X.XXX m`), allowing the VLM to classify object dynamics from text cues rather than visual reasoning.
- **Offline Response-Bank Replay**: The 36 inferences were executed in a single pre-run pass and cached. Policies R3–R8 did not invoke the model interactively; they replayed cached answers from memory.
- **Token Accounting**: Token reduction was reported as 75.0%, but the actual token count dropped from 9,522 to 3,600, which represents **62.19% reduction**, not 75.0%.

---

## 2. Verification of Live Inference Execution

### 2.1 Server and Hardware Execution
- **Server Instance**: `llama-server` running under WSL2 Ubuntu 22.04 on port 8089.
- **Model Checkpoint**: `Qwen2.5-VL-3B-Instruct-Q4_K_M.gguf` with multimodal projector `mmproj-Qwen2.5-VL-3B-Instruct-Q8_0.gguf`.
- **Artifact Verified**: `results/publication_campaign/run_20261101_execution/vlm_inference/live_qwen_inferences.json` (SHA-256: `893f18bebb56a595ec0a892e624c9d5d852cb78c668ff4ba128a39e83cb2b115`, 36 records).
- **Execution Timestamps**: Generated sequentially between 16:22:15 and 16:25:30 on 2026-11-01. Average inference latency was **1.84 seconds per candidate**.

All 36 JSON entries contain valid HTTP 200 payload responses, token counts (mean: 264.5 tokens per query), and structured JSON outputs (`is_dynamic`, `confidence`, `reasoning`).

---

## 3. Forensic Analysis of Prompt Engineering: Displacement Leakage

### 3.1 Prompt Template Code Audit
In `scripts/run_master_publication_campaign.py`, lines 424–445 construct the prompt sent to Qwen:

```python
prompt_text = (
    "You are an expert SLAM assistant analyzing an object candidate in an RGB-D camera feed.\n"
    f"Candidate Label: {label}\n"
    f"Bounding Box: {bbox}\n"
    f"Tracked Keypoints: {n_kpts}\n"
    f"Optimizer Inliers: {n_inliers}\n"
    f"Mean Reprojection Error: {mean_rep:.2f} px\n"
    f"Measured 3D Landmark Displacement: {disp:.3f} m (uncertainty: {unc:.3f} m)\n"
    "Determine whether this object candidate is currently dynamic (in motion) "
    "or static, and whether its features should be excluded from SLAM optimization.\n"
    "Respond ONLY with valid JSON: {\"is_dynamic\": bool, \"confidence\": float, \"reasoning\": str}"
)
```

### 3.2 Methodological Impact: Feature Leakage
By providing `Measured 3D Landmark Displacement: 1.240 m (uncertainty: 0.180 m)` directly in the text prompt:
1. The model does not need to analyze visual texture, deformation, or semantic context in the image.
2. The language model can simply check if `disp > 0.10 m` to classify `is_dynamic: true`.
3. Inspection of Qwen's output strings confirms this leakage: multiple outputs explicitly state *"The candidate shows significant measured displacement of 1.24 m, indicating active motion."*

**Verdict**: The VLM was acting as a text-parsing proxy for the geometric telemetry rather than an independent visual semantic classifier. Future work must strip metric displacement from the prompt and provide only the cropped image, bounding box, and semantic class prompt.

---

## 4. Response-Bank Replay Mechanics

In lines 490–570 of the master campaign script:
- `vlm_dict` was constructed once from the 36 inference responses.
- Policies R0 through R8 were then evaluated by filtering and indexing `vlm_dict`.
- **Impact**: No policy incurred runtime latency or network jitter during decision making. While this is standard for benchmark reproducibility (evaluating routing policies on a fixed evaluation bank), it must be clearly disclosed as **Offline Response-Bank Replay**, not closed-loop online inference.

---

## 5. Token Arithmetic Reconciliation

| Metric | Master Report Claim | Audited Reality | Discrepancy Cause |
| :--- | :--- | :--- | :--- |
| **Full Budget Tokens (R0)** | 9,522 | 9,522 | Verified exact match (36 queries) |
| **Selected Queries (R5–R7)** | 9 (25.0%) | 14 (38.89%) | Per-sequence ceiling applied instead of global 25% |
| **Selective Tokens Consumed** | ~2,380 | 3,600 | 14 queries executed $\times \approx 257$ tokens |
| **Reported Token Reduction** | **75.0%** | **62.19%** | $(9522 - 3600) / 9522 = 62.19\%$ |

If evaluated under the declared **global 25% budget ($K=9$)**, token consumption is 2,382 tokens, achieving exactly **74.98% $\approx$ 75.0% token reduction**.

---

## 6. Recommendations for Publication

1. **Disclose Response-Bank Evaluation**: Update the experimental section to state: *"All 36 candidate queries were pre-inferred using Qwen2.5-VL-3B-Instruct to create a fixed evaluation bank, ensuring identical semantic responses across routing policy comparisons."*
2. **Correct Token Savings to 62.2% or 75.0%**: Explicitly state both:
   - Under the per-sequence ceiling allocation (14/36 = 38.89% queries): token savings = **62.2%**.
   - Under the strict global budget ($K = 9$, 25.0% queries): token savings = **75.0%**.
3. **Note Prompt Leakage as a Limitation**: Document in Section V (Limitations) that displacement telemetry was provided in the prompt text, and outline ablation experiments without geometric telemetry in future revisions.

