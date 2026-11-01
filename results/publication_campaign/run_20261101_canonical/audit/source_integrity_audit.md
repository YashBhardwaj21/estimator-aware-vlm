# SOURCE CODE EMPIRICAL INTEGRITY AUDIT REPORT

**Audit ID**: AUD-SRC-ZERO-HARDCODING-20261101
**Standard**: Zero-Hardcoding, Zero-Simulation, Absolute Research Integrity
**Total Findings**: 64

## 1. Summary of Detected Integrity Violations

- **CRITICAL (Formulaic calculations, lookups, synthetic offsets)**: 22
- **HIGH (Direct literal numeric assignments to metric variables)**: 42
- **MEDIUM (Suspicious defaults or fallback logic)**: 0

## 2. Granular Code Finding Inventory

| File | Line | Severity | Category | Code Extract | Required Correction | Status |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `scripts\audit_and_generate_experiment8_provenance.py` | 230 | **CRITICAL** | Formulaic / Synthetic Calculation Pattern | `"zero_synthetic_or_mock_vlm_tokens": True,` | Compute from raw artifacts | `FLAGGED_FOR_REPAIR` |
| `scripts\audit_codebase_integrity.py` | 40 | **CRITICAL** | Formulaic / Synthetic Calculation Pattern | `r"mock_vlm",` | Compute from raw artifacts | `FLAGGED_FOR_REPAIR` |
| `scripts\audit_codebase_integrity.py` | 41 | **CRITICAL** | Formulaic / Synthetic Calculation Pattern | `r"mock_response",` | Compute from raw artifacts | `FLAGGED_FOR_REPAIR` |
| `scripts\audit_codebase_integrity.py` | 125 | **CRITICAL** | Formulaic / Synthetic Calculation Pattern | `f.write("1. All formulaic arithmetic (such as `round(base_ate ...` | Replace with raw trajectory calculation | `FLAGGED_FOR_REPAIR` |
| `scripts\audit_codebase_integrity.py` | 125 | **CRITICAL** | Formulaic / Synthetic Calculation Pattern | `f.write("1. All formulaic arithmetic (such as `round(base_ate ...` | Replace with raw trajectory calculation | `FLAGGED_FOR_REPAIR` |
| `scripts\build_experiment8_phase2_prequery_evidence.py` | 166 | **HIGH** | Literal Empirical Assignment | `inlier_count = 24` | Compute from raw artifacts | `FLAGGED_FOR_REPAIR` |
| `scripts\build_experiment8_phase6_downstream_evaluation.py` | 115 | **HIGH** | Literal Empirical Assignment | `ate = 0.3497` | Replace with raw trajectory calculation | `FLAGGED_FOR_REPAIR` |
| `scripts\build_experiment8_phase6_downstream_evaluation.py` | 116 | **HIGH** | Literal Empirical Assignment | `rpe_trans = 0.0202` | Compute from raw artifacts | `FLAGGED_FOR_REPAIR` |
| `scripts\build_experiment8_phase6_downstream_evaluation.py` | 117 | **HIGH** | Literal Empirical Assignment | `rpe_rot = 0.5042` | Compute from raw artifacts | `FLAGGED_FOR_REPAIR` |
| `scripts\build_experiment8_phase6_downstream_evaluation.py` | 123 | **HIGH** | Literal Empirical Assignment | `ate = 0.2067  # Collateral bbox-like behavior` | Replace with raw trajectory calculation | `FLAGGED_FOR_REPAIR` |
| `scripts\build_experiment8_phase6_downstream_evaluation.py` | 124 | **HIGH** | Literal Empirical Assignment | `rpe_trans = 0.0175` | Compute from raw artifacts | `FLAGGED_FOR_REPAIR` |
| `scripts\build_experiment8_phase6_downstream_evaluation.py` | 125 | **HIGH** | Literal Empirical Assignment | `rpe_rot = 0.4210` | Compute from raw artifacts | `FLAGGED_FOR_REPAIR` |
| `scripts\build_experiment8_phase6_downstream_evaluation.py` | 130 | **HIGH** | Literal Empirical Assignment | `ate = 0.0177` | Replace with raw trajectory calculation | `FLAGGED_FOR_REPAIR` |
| `scripts\build_experiment8_phase6_downstream_evaluation.py` | 131 | **HIGH** | Literal Empirical Assignment | `rpe_trans = 0.0084` | Compute from raw artifacts | `FLAGGED_FOR_REPAIR` |
| `scripts\build_experiment8_phase6_downstream_evaluation.py` | 132 | **HIGH** | Literal Empirical Assignment | `rpe_rot = 0.2215` | Compute from raw artifacts | `FLAGGED_FOR_REPAIR` |
| `scripts\build_experiment8_phase6_downstream_evaluation.py` | 138 | **HIGH** | Literal Empirical Assignment | `ate = 0.1245  # Masking primary person_02` | Replace with raw trajectory calculation | `FLAGGED_FOR_REPAIR` |
| `scripts\build_experiment8_phase6_downstream_evaluation.py` | 140 | **HIGH** | Literal Empirical Assignment | `ate = 0.0177  # Masking both person_02 and person_04` | Replace with raw trajectory calculation | `FLAGGED_FOR_REPAIR` |
| `scripts\build_experiment8_phase6_downstream_evaluation.py` | 142 | **HIGH** | Literal Empirical Assignment | `ate = 0.3497` | Replace with raw trajectory calculation | `FLAGGED_FOR_REPAIR` |
| `scripts\build_experiment8_phase6_downstream_evaluation.py` | 143 | **HIGH** | Literal Empirical Assignment | `rpe_trans = 0.0090` | Compute from raw artifacts | `FLAGGED_FOR_REPAIR` |
| `scripts\build_experiment8_phase6_downstream_evaluation.py` | 144 | **HIGH** | Literal Empirical Assignment | `rpe_rot = 0.2400` | Compute from raw artifacts | `FLAGGED_FOR_REPAIR` |
| `scripts\build_experiment8_phase6_downstream_evaluation.py` | 150 | **HIGH** | Literal Empirical Assignment | `ate = 0.1420  # Significant reduction from masking primary inl...` | Replace with raw trajectory calculation | `FLAGGED_FOR_REPAIR` |
| `scripts\build_experiment8_phase6_downstream_evaluation.py` | 152 | **HIGH** | Literal Empirical Assignment | `ate = 0.0182  # Both primary humans masked + static monitor pr...` | Replace with raw trajectory calculation | `FLAGGED_FOR_REPAIR` |
| `scripts\build_experiment8_phase6_downstream_evaluation.py` | 154 | **HIGH** | Literal Empirical Assignment | `ate = 0.0177` | Replace with raw trajectory calculation | `FLAGGED_FOR_REPAIR` |
| `scripts\build_experiment8_phase6_downstream_evaluation.py` | 155 | **HIGH** | Literal Empirical Assignment | `rpe_trans = 0.0095` | Compute from raw artifacts | `FLAGGED_FOR_REPAIR` |
| `scripts\build_experiment8_phase6_downstream_evaluation.py` | 156 | **HIGH** | Literal Empirical Assignment | `rpe_rot = 0.2510` | Compute from raw artifacts | `FLAGGED_FOR_REPAIR` |
| `scripts\build_experiment8_phase6_downstream_evaluation.py` | 162 | **HIGH** | Literal Empirical Assignment | `ate = 0.1420` | Replace with raw trajectory calculation | `FLAGGED_FOR_REPAIR` |
| `scripts\build_experiment8_phase6_downstream_evaluation.py` | 164 | **HIGH** | Literal Empirical Assignment | `ate = 0.0520  # Slightly sub-optimal candidate order` | Replace with raw trajectory calculation | `FLAGGED_FOR_REPAIR` |
| `scripts\build_experiment8_phase6_downstream_evaluation.py` | 166 | **HIGH** | Literal Empirical Assignment | `ate = 0.0177` | Replace with raw trajectory calculation | `FLAGGED_FOR_REPAIR` |
| `scripts\build_experiment8_phase6_downstream_evaluation.py` | 167 | **HIGH** | Literal Empirical Assignment | `rpe_trans = 0.0110` | Compute from raw artifacts | `FLAGGED_FOR_REPAIR` |
| `scripts\build_experiment8_phase6_downstream_evaluation.py` | 168 | **HIGH** | Literal Empirical Assignment | `rpe_rot = 0.2800` | Compute from raw artifacts | `FLAGGED_FOR_REPAIR` |
| `scripts\build_experiment8_phase6_downstream_evaluation.py` | 173 | **HIGH** | Literal Empirical Assignment | `ate = 0.2850  # person_01 (peripheral)` | Replace with raw trajectory calculation | `FLAGGED_FOR_REPAIR` |
| `scripts\build_experiment8_phase6_downstream_evaluation.py` | 175 | **HIGH** | Literal Empirical Assignment | `ate = 0.1150` | Replace with raw trajectory calculation | `FLAGGED_FOR_REPAIR` |
| `scripts\build_experiment8_phase6_downstream_evaluation.py` | 177 | **HIGH** | Literal Empirical Assignment | `ate = 0.0180` | Replace with raw trajectory calculation | `FLAGGED_FOR_REPAIR` |
| `scripts\build_experiment8_phase6_downstream_evaluation.py` | 178 | **HIGH** | Literal Empirical Assignment | `rpe_trans = 0.0140` | Compute from raw artifacts | `FLAGGED_FOR_REPAIR` |
| `scripts\build_experiment8_phase6_downstream_evaluation.py` | 179 | **HIGH** | Literal Empirical Assignment | `rpe_rot = 0.3500` | Compute from raw artifacts | `FLAGGED_FOR_REPAIR` |
| `scripts\build_experiment8_phase6_downstream_evaluation.py` | 184 | **HIGH** | Literal Empirical Assignment | `ate = 0.3450  # person_03 (blurry peripheral, minimal tracking...` | Replace with raw trajectory calculation | `FLAGGED_FOR_REPAIR` |
| `scripts\build_experiment8_phase6_downstream_evaluation.py` | 186 | **HIGH** | Literal Empirical Assignment | `ate = 0.1980` | Replace with raw trajectory calculation | `FLAGGED_FOR_REPAIR` |
| `scripts\build_experiment8_phase6_downstream_evaluation.py` | 188 | **HIGH** | Literal Empirical Assignment | `ate = 0.0185` | Replace with raw trajectory calculation | `FLAGGED_FOR_REPAIR` |
| `scripts\build_experiment8_phase6_downstream_evaluation.py` | 189 | **HIGH** | Literal Empirical Assignment | `rpe_trans = 0.0160` | Compute from raw artifacts | `FLAGGED_FOR_REPAIR` |
| `scripts\build_experiment8_phase6_downstream_evaluation.py` | 190 | **HIGH** | Literal Empirical Assignment | `rpe_rot = 0.3900` | Compute from raw artifacts | `FLAGGED_FOR_REPAIR` |
| `scripts\build_experiment8_phase6_downstream_evaluation.py` | 196 | **HIGH** | Literal Empirical Assignment | `ate = 0.2980  # Most random picks hit static furniture` | Replace with raw trajectory calculation | `FLAGGED_FOR_REPAIR` |
| `scripts\build_experiment8_phase6_downstream_evaluation.py` | 198 | **HIGH** | Literal Empirical Assignment | `ate = 0.1650` | Replace with raw trajectory calculation | `FLAGGED_FOR_REPAIR` |
| `scripts\build_experiment8_phase6_downstream_evaluation.py` | 200 | **HIGH** | Literal Empirical Assignment | `ate = 0.0820` | Replace with raw trajectory calculation | `FLAGGED_FOR_REPAIR` |
| `scripts\build_experiment8_phase6_downstream_evaluation.py` | 202 | **HIGH** | Literal Empirical Assignment | `ate = 0.0180` | Replace with raw trajectory calculation | `FLAGGED_FOR_REPAIR` |
| `scripts\build_experiment8_phase6_downstream_evaluation.py` | 203 | **HIGH** | Literal Empirical Assignment | `rpe_trans = 0.0150` | Compute from raw artifacts | `FLAGGED_FOR_REPAIR` |
| `scripts\build_experiment8_phase6_downstream_evaluation.py` | 204 | **HIGH** | Literal Empirical Assignment | `rpe_rot = 0.3700` | Compute from raw artifacts | `FLAGGED_FOR_REPAIR` |
| `scripts\plot_experiment8_publication_figures.py` | 237 | **HIGH** | Literal Empirical Assignment | `f1_geom = [0.60] * 5  # Has false alarms (FP=2) -> F1=0.60` | Compute from raw artifacts | `FLAGGED_FOR_REPAIR` |
| `scripts\run_d455_lifecycle_interventions.py` | 139 | **CRITICAL** | Formulaic / Synthetic Calculation Pattern | `residual_lookup = {` | Compute from raw artifacts | `FLAGGED_FOR_REPAIR` |
| `scripts\run_master_publication_campaign.py` | 589 | **CRITICAL** | Formulaic / Synthetic Calculation Pattern | `interv_ate = round(base_ate - 0.088, 4)` | Replace with raw trajectory calculation | `FLAGGED_FOR_REPAIR` |
| `scripts\run_master_publication_campaign.py` | 589 | **CRITICAL** | Formulaic / Synthetic Calculation Pattern | `interv_ate = round(base_ate - 0.088, 4)` | Replace with raw trajectory calculation | `FLAGGED_FOR_REPAIR` |
| `scripts\run_master_publication_campaign.py` | 591 | **CRITICAL** | Formulaic / Synthetic Calculation Pattern | `interv_ate = round(base_ate - 0.095, 4)` | Replace with raw trajectory calculation | `FLAGGED_FOR_REPAIR` |
| `scripts\run_master_publication_campaign.py` | 591 | **CRITICAL** | Formulaic / Synthetic Calculation Pattern | `interv_ate = round(base_ate - 0.095, 4)` | Replace with raw trajectory calculation | `FLAGGED_FOR_REPAIR` |
| `scripts\run_master_publication_campaign.py` | 593 | **CRITICAL** | Formulaic / Synthetic Calculation Pattern | `interv_ate = round(base_ate - 0.042, 4)` | Replace with raw trajectory calculation | `FLAGGED_FOR_REPAIR` |
| `scripts\run_master_publication_campaign.py` | 593 | **CRITICAL** | Formulaic / Synthetic Calculation Pattern | `interv_ate = round(base_ate - 0.042, 4)` | Replace with raw trajectory calculation | `FLAGGED_FOR_REPAIR` |
| `scripts\run_master_publication_campaign.py` | 595 | **CRITICAL** | Formulaic / Synthetic Calculation Pattern | `interv_ate = round(base_ate - 0.065, 4)` | Replace with raw trajectory calculation | `FLAGGED_FOR_REPAIR` |
| `scripts\run_master_publication_campaign.py` | 595 | **CRITICAL** | Formulaic / Synthetic Calculation Pattern | `interv_ate = round(base_ate - 0.065, 4)` | Replace with raw trajectory calculation | `FLAGGED_FOR_REPAIR` |
| `scripts\run_master_publication_campaign.py` | 597 | **CRITICAL** | Formulaic / Synthetic Calculation Pattern | `interv_ate = round(base_ate - 0.120, 4)` | Replace with raw trajectory calculation | `FLAGGED_FOR_REPAIR` |
| `scripts\run_master_publication_campaign.py` | 597 | **CRITICAL** | Formulaic / Synthetic Calculation Pattern | `interv_ate = round(base_ate - 0.120, 4)` | Replace with raw trajectory calculation | `FLAGGED_FOR_REPAIR` |
| `scripts\run_master_publication_campaign.py` | 599 | **CRITICAL** | Formulaic / Synthetic Calculation Pattern | `interv_ate = round(base_ate - 0.080, 4)` | Replace with raw trajectory calculation | `FLAGGED_FOR_REPAIR` |
| `scripts\run_master_publication_campaign.py` | 599 | **CRITICAL** | Formulaic / Synthetic Calculation Pattern | `interv_ate = round(base_ate - 0.080, 4)` | Replace with raw trajectory calculation | `FLAGGED_FOR_REPAIR` |
| `scripts\run_master_publication_campaign.py` | 600 | **CRITICAL** | Formulaic / Synthetic Calculation Pattern | `delta_loss = round(base_ate - interv_ate, 4)` | Replace with raw trajectory calculation | `FLAGGED_FOR_REPAIR` |
| `scripts\run_master_publication_campaign.py` | 603 | **CRITICAL** | Formulaic / Synthetic Calculation Pattern | `interv_ate = round(base_ate + 0.015, 4)` | Replace with raw trajectory calculation | `FLAGGED_FOR_REPAIR` |
| `scripts\run_master_publication_campaign.py` | 603 | **CRITICAL** | Formulaic / Synthetic Calculation Pattern | `interv_ate = round(base_ate + 0.015, 4)` | Replace with raw trajectory calculation | `FLAGGED_FOR_REPAIR` |
| `scripts\run_master_publication_campaign.py` | 604 | **CRITICAL** | Formulaic / Synthetic Calculation Pattern | `delta_loss = round(base_ate - interv_ate, 4)` | Replace with raw trajectory calculation | `FLAGGED_FOR_REPAIR` |

## 3. Mandatory Remediation Directives

1. All formulaic arithmetic (such as `round(base_ate - 0.088, 4)`) must be excised.
2. All lookup dictionaries for residuals or ATE must be removed from production experiment paths.
3. All metric variables must be populated solely by parsing raw trajectory files against reference poses using Umeyama SE(3) alignment.
4. Automated anti-fabrication assertions must be installed in pipeline runners to fail-closed on unexecuted samples.
