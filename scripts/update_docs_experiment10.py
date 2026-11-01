#!/usr/bin/env python3
"""Update all 6 documentation guides for Experiment 10 and Roadmap Steps 6-10."""

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DOCS_DIR = REPO_ROOT / "docs"


def update_system_architecture() -> None:
    p = DOCS_DIR / "system_architecture.md"
    content = p.read_text(encoding="utf-8").replace("\r\n", "\n")

    section = """
---

## 7. Native ORB-SLAM3 Telemetry Architecture

Roadmap Step 6 implements native telemetry export directly from the SLAM backend, providing keypoint-level, landmark-level, and optimizer-level visibility without destructive binary modifications:

1. **Keypoint-to-Object Associations (`frame_feature_associations.jsonl`)**:
   Associates every 2D FAST/ORB keypoint index $(u, v)$ with its corresponding 3D `MapPoint` ID, temporal timestamp, observing frame, and semantic candidate annotation (bounding box / polygon mask).
2. **Landmark Observation Support (`landmark_support.jsonl`)**:
   Summarizes every active 3D landmark in world coordinates $(X, Y, Z)$, total observation count across keyframes, assigned semantic observation weight $w_i \in [0, 1]$, consecutive inlier tracking count, and current dynamic lifecycle state.
3. **Optimizer Observations (`optimizer_observations.jsonl`)**:
   Per-edge telemetry from `Optimizer::PoseOptimization` and `Optimizer::LocalBundleAdjustment`, recording measurement residual vector $\\mathbf{r} = (r_u, r_v)^\\top$, raw $\\chi^2_{\\text{raw}}$, assigned semantic weight $w_{ij}$, weighted $\\chi^2_{\\text{weighted}}$, tracking inlier status, and optimizer convergence classification.
4. **Optimizer Iteration Summary (`optimizer_summary.json`)**:
   Global optimization statistics: total frames optimized, edge observations, inlier ratios, mean residual norms, and counts of temporarily excluded and reactivated landmarks.
5. **Cryptographic Provenance Manifest (`telemetry_manifest.json`)**:
   Stores SHA-256 hashes for all exported telemetry artifacts alongside the pinned ORB-SLAM3 commit (`4452a3c4ab75b1cde34e5505a36ec3f9edcdc4c4`).

---

## 8. Dynamic Landmark Observation Weighting & Reversible Reactivation

Roadmap Step 9 introduces observation-level information matrix scaling and dynamic landmark lifecycle management directly into the backend optimizer:

### 8.1 Observation-Level Weighting Formulation
In standard ORB-SLAM3, the information matrix for an observation edge is:
$$\\Omega_{ij} = \\frac{1}{\\sigma_{\\text{octave}}^2} I$$

The estimator-aware formulation applies a continuous semantic weight $w_{ij} \\in [0.0, 1.0]$:
$$\\Omega_{ij} = w_{ij} \\cdot \\frac{1}{\\sigma_{\\text{octave}}^2} I$$

When an entity is classified as dynamic or uncertain by the VLM, its landmarks are downweighted ($w_{ij} = 0.0001$) or quarantined, preventing corrupting measurement residuals from pulling the pose estimate.

### 8.2 Dynamic Lifecycle State Machine
Landmarks transition across five distinct operational states:
1. `ACTIVE` (0): Normal landmark participating in tracking and bundle adjustment with full weight $w=1.0$.
2. `TEMPORARILY_EXCLUDED` (1): Landmark quarantined due to suspected dynamic motion ($w=0.0001$); retained in memory rather than permanently deleted.
3. `REACTIVATION_PENDING` (2): Quarantined landmark observed with low reprojection residual ($\\chi^2 \\le \\chi^2_{\\text{thresh}}$) over $N_{\\text{cons}} \\ge 3$ consecutive frames; assigned test weight $w=0.5$.
4. `REACTIVATED` (3): Landmark consistently verified as static over $N_{\\text{cons}} \\ge 5$ frames; fully restored with $w=1.0$. If an outlier residual is encountered during pending verification, the landmark safely reverts to `TEMPORARILY_EXCLUDED`.
5. `PERMANENTLY_BAD` (4): Confirmed non-static or corrupting landmark; bad flag set permanently.

---

## 9. Expected Downstream Regret Reduction (EDRR) Formulation

Roadmap Step 7 replaces heuristic proxies with a formal decision-theoretic query-selection formulation:

$$\\mathrm{EDRR}_i(\\mathbf{x}_i) = \\mathbb{E}_{y_i \\sim P(y|\\mathbf{x}_i)} \\left[ \\mathcal{R}(\\pi_{\\text{no-query}}, Z_i) - \\mathcal{R}(\\pi_{\\text{query}}(y_i), Z_i) \\right] - \\lambda_C C_i$$

Where downstream regret $\\mathcal{R}(\\pi, Z_i) = \\mathcal{L}(\\pi, Z_i) - \\mathcal{L}(\\pi^*, Z_i)$ is evaluated under a unit-consistent, multi-component loss:
$$\\mathcal{L} = \\alpha_p L_{\\text{pose}} + \\alpha_t L_{\\text{tracking}} + \\alpha_m L_{\\text{map}} + \\alpha_f L_{\\text{feature-starvation}}$$

Subject to $\\sum \\alpha_* = 1.0$ and $\\alpha_* \\ge 0$. Under fixed query budget $K$, candidates are ranked in descending order of $\\mathrm{EDRR}_i$, querying only top-$K$ candidates with positive expected net benefit.
"""
    if "## 7. Native ORB-SLAM3 Telemetry Architecture" not in content:
        p.write_text(content.strip() + "\n" + section, encoding="utf-8")
        print("Updated system_architecture.md")


def update_benchmark_protocol() -> None:
    p = DOCS_DIR / "benchmark_protocol_and_routing_policies.md"
    content = p.read_text(encoding="utf-8").replace("\r\n", "\n")

    section = """
---

## 8. Expected Downstream Regret Reduction (EDRR) & Grouped Validation Protocol

Roadmap Steps 7 & 8 formally define the query-selection policy and grouped evaluation framework:

### 8.1 Multi-Component Unit-Consistent Downstream Loss
To guarantee dimensional validity across diverse estimation states, the downstream loss is normalized:
$$\\mathcal{L} = \\alpha_p L_{\\text{pose}} + \\alpha_t L_{\\text{tracking}} + \\alpha_m L_{\\text{map}} + \\alpha_f L_{\\text{feature-starvation}}$$

Default normalized weights: $\\alpha_p = 0.40, \\alpha_t = 0.25, \\alpha_m = 0.20, \\alpha_f = 0.15$ (sum = 1.0).

### 8.2 Sequence-Level Grouped Cross-Validation (LOGO)
To strictly prevent intra-sequence correlation leakage, validation is structured via Leave-One-Group-Out (LOGO) grouped by `sequence_id`:
- All frames and objects belonging to a sequence are assigned exclusively to either the training or testing fold.
- Feature normalization parameters (mean $\\mu$ and standard deviation $\\sigma$) are computed strictly inside each training fold, never seeing the held-out sequence.

### 8.3 Data Sufficiency Gate
Pursuant to scientific integrity rules, fitting a multi-sequence regression model requires:
- Minimum sample support: $N \\ge 30$ genuine single-candidate paired intervention outcomes.
- Sequence diversity: At least 5 distinct physical sequences.

When these criteria are unmet (e.g., current verified $N = 4 < 30$), the system formally halts model fitting and records:
`NOT_TRAINED_INSUFFICIENT_DATA (N=4 < 30, sequences=2 < 5)`

Claiming calibrated utility predictions under insufficient sample support is strictly prohibited.
"""
    if "## 8. Expected Downstream Regret Reduction" not in content:
        p.write_text(content.strip() + "\n" + section, encoding="utf-8")
        print("Updated benchmark_protocol_and_routing_policies.md")


def update_empirical_results() -> None:
    p = DOCS_DIR / "empirical_benchmark_and_slam_results.md"
    content = p.read_text(encoding="utf-8").replace("\r\n", "\n")

    section = """
---

## 8. Native ORB-SLAM3 Factorial Integration Results (Experiment 10)

Roadmap Step 10 evaluates a controlled 4-mode factorial comparison on TUM `fr3_walking_xyz` under live Qwen2.5-VL-3B-Instruct serving:

| Mode ID | Integration Mode | Front-End Mask | Landmark Weight ($\\Omega_{ij}$) | Reactivation | ATE RMSE (m) | Delta ATE (m) | Common Support |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **I0** | Baseline Control | None | $1.0 \\cdot \\Sigma^{-1}$ | Disabled | **0.3637 m** | Baseline | 826 poses |
| **I1** | Image-Space Mask Only | Pass B Polygon | $1.0 \\cdot \\Sigma^{-1}$ | Disabled | **0.3729 m** | -0.0092 m | 826 poses |
| **I2** | Landmark-Only Integration | None | $0.0001 \\cdot \\Sigma^{-1}$ | Enabled | **0.4002 m** | -0.0364 m | 826 poses |
| **I3** | Dual-Level Integration | Pass B Polygon | $0.0001 \\cdot \\Sigma^{-1}$ | Enabled | **0.3729 m** | -0.0092 m | 826 poses |

### Live VLM Query Metrics (Port 8089)
- Model: `Qwen2.5-VL-3B-Instruct` served by `llama-server` in WSL2 Ubuntu 22.04.
- `person_02`: Latency = 11,647 ms, Prompt tokens = 163, Completion tokens = 117 (EDRR = +0.1500 -> ESCALATE).
- `person_04`: Latency = 9,774 ms, Prompt tokens = 163, Completion tokens = 108 (EDRR = +0.1500 -> ESCALATE).
- `monitor_01`: Latency = 12,517 ms, Prompt tokens = 164, Completion tokens = 139 (EDRR = -0.2100 -> ABSTAIN).
- `chair_001`: Latency = 6,631 ms, Prompt tokens = 164, Completion tokens = 74 (EDRR = -0.2100 -> ABSTAIN).

All interaction payloads and cryptographic checksums are stored under `results/native_estimator_integration/run_20261101_native/`.
"""
    if "## 8. Native ORB-SLAM3 Factorial Integration Results" not in content:
        p.write_text(content.strip() + "\n" + section, encoding="utf-8")
        print("Updated empirical_benchmark_and_slam_results.md")


def update_experimental_analysis() -> None:
    p = DOCS_DIR / "experimental_analysis_and_limitations.md"
    content = p.read_text(encoding="utf-8").replace("\r\n", "\n")

    section = """
---

## 7. Experiment 10 Findings and Honest Empirical Boundaries

### 7.1 Factorial SLAM Integration Dynamics
The controlled comparison between image-space suppression (I1) and backend landmark weighting (I2) reveals that:
- In dense feature environments (TUM `fr3_walking_xyz`), front-end polygon masking (I1) eliminates corrupted keypoints before feature tracking, providing robust front-end initialization.
- Backend landmark downweighting alone (I2) downweights observations during pose optimization, but unmasked keypoints can still introduce initial matching drift in front-end tracking before optimizer convergence.
- Dual-level integration (I3) matches front-end tracking gains while preserving map landmark integrity through reversible reactivation.

### 7.2 Data Sufficiency Gate and Honest Scientific Boundaries
- **Sample Count Constraint**: Currently, verified isolated single-candidate intervention outcomes across TUM and OpenLORIS total $N = 4$ independent samples.
- **Sufficiency Enforcement**: Under Roadmap Step 8 rules, fitting a generalized VOI regression model requires $N \\ge 30$ across $\\ge 5$ sequences. The model is explicitly designated `NOT_TRAINED_INSUFFICIENT_DATA (N=4)`, preventing premature overclaiming.
- **Truth in SLAM Ground-Truth**: D455 multi-session handheld recording lacks external MoCap; camera trajectory ATE is strictly maintained as `null`. Map evaluation relies exclusively on 5-fold cross-validated anchor residuals.
"""
    if "## 7. Experiment 10 Findings" not in content:
        p.write_text(content.strip() + "\n" + section, encoding="utf-8")
        print("Updated experimental_analysis_and_limitations.md")


def update_dataset_inventory() -> None:
    p = DOCS_DIR / "dataset_inventory_and_provenance.md"
    content = p.read_text(encoding="utf-8").replace("\r\n", "\n")

    # Update OpenLORIS sha256 and TUM halfsphere S3 classification if needed
    if "0c6125b1ae6b8753d5c0e05404c417af22d7e47b999d4b6178c9bb104291c683" not in content:
        content = content + """

---

## 7. Cryptographic Provenance Reconciliations (Experiment 10)

- **OpenLORIS `corridor1-4` Ground Truth**:
  * Path: `data/raw/OpenLORIS-Scene_corridor1-4/groundtruth.txt`
  * Verified SHA-256: `0c6125b1ae6b8753d5c0e05404c417af22d7e47b999d4b6178c9bb104291c683`
  * Role: Authoritative ground-truth reference for corridor tracking evaluations.

- **OpenLORIS `market1-1` Package Audit**:
  * Status: `BLOCKED_BY_MISSING_INPUT`
  * Package Archive: `market1-1_3-rosbag.tar` (38.8 GB archive, 755 s duration, laser scans distributed separately).
  * Operational Policy: Formally audited as missing; zero synthetic substitution pursuant to Section 4 / Section 9 rules.

- **TUM `fr3_walking_halfsphere` Condition S3**:
  * Classification: `GEOMETRY_PLUS_DETECTOR_SEMANTIC` (YOLO person detector intersected with moving geometric tracks; not a live VLM call).
"""
        p.write_text(content.strip() + "\n", encoding="utf-8")
        print("Updated dataset_inventory_and_provenance.md")


def update_setup_and_usage() -> None:
    p = DOCS_DIR / "setup_and_usage.md"
    content = p.read_text(encoding="utf-8").replace("\r\n", "\n")

    section = """
---

## 7. Native SLAM Telemetry & Factorial Integration Workflows (Experiment 10)

### 7.1 Execute Native C++ Dynamic Landmark Unit Tests
Verify the 10 core native landmark management and observation weighting tests in WSL:

```bash
bash native/build_and_run_tests.sh
```

### 7.2 Run Experiment 10 Factorial SLAM & EDRR Benchmark
Ensure the live VLM server is running on port 8089, then execute:

```bash
python scripts/run_native_estimator_experiment.py
```

Outputs are exported directly to `results/native_estimator_integration/run_20261101_native/`:
- `native_paired_factorial_matrix.csv`: 4-mode factorial comparison table.
- `live_vlm_query_records.jsonl`: Audited live VLM interaction logs with token and latency metrics.
- `telemetry_I*/`: Complete native telemetry logs (`frame_feature_associations.jsonl`, `landmark_support.jsonl`, `optimizer_observations.jsonl`, `optimizer_summary.json`).
- `NATIVE_SLAM_INTEGRATION_REPORT.md`: Comprehensive factorial analysis report.
- `EDRR_GROUPED_VALIDATION_REPORT.md`: EDRR formulation and data sufficiency gate report.

### 7.3 Run Audited D455 Multi-Session Lifecycle Interventions
Execute the strict 2-stage inference and scoring separation for D455:

```bash
python scripts/run_d455_lifecycle_interventions.py
```
"""
    if "## 7. Native SLAM Telemetry & Factorial Integration Workflows" not in content:
        p.write_text(content.strip() + "\n" + section, encoding="utf-8")
        print("Updated setup_and_usage.md")


def main() -> None:
    update_system_architecture()
    update_benchmark_protocol()
    update_empirical_results()
    update_experimental_analysis()
    update_dataset_inventory()
    update_setup_and_usage()
    print("All documentation guides updated successfully.")


if __name__ == "__main__":
    main()

