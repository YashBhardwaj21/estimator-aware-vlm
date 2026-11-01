# Expected Downstream Regret Reduction (EDRR) & Grouped Validation Report

## 1. Decision-Theoretic Formulation

$$\mathrm{EDRR}_i(\mathbf{x}_i) = \mathbb{E}_{y_i} [\mathcal{R}(\pi_{\mathrm{no-query}}, Z_i) - \mathcal{R}(\pi_{\mathrm{query}}(y_i), Z_i)] - \lambda_C C_i$$

Where the downstream multi-component loss is normalized across pose, tracking, map integrity, and feature starvation:
$$\mathcal{L} = \alpha_p L_{\mathrm{pose}} + \alpha_t L_{\mathrm{tracking}} + \alpha_m L_{\mathrm{map}} + \alpha_f L_{\mathrm{feature-starvation}}$$

## 2. Data Sufficiency Gate Audit

- **Current Audited Single-Candidate Outcomes**: N = 4 records across 2 sequences.
- **Sufficiency Threshold**: N >= 30 records across >= 5 distinct sequences.
- **Gate Status**: `NOT_TRAINED_INSUFFICIENT_DATA (N=4 < 30, sequences=2 < 5)`
- **Enforcement**: Model fitting is formally refused pursuant to Roadmap Step 8 rules. Claiming a calibrated multi-sequence VOI model on insufficient empirical support is strictly prohibited.

## 3. Sequence Provenance & Audit Records

| Intervention ID | Sequence ID | Target ID | Class | Action | Baseline ATE (m) | Intervention ATE (m) | Delta ATE (m) |
| :--- | :--- | :--- | :--- | :--- | :---: | :---: | :---: |
| `tum_fr3_walking_xyz_person_02_POLYGON` | `tum_fr3_walking_xyz` | `person_02` | `person` | `IMAGE_POLYGON_MASK` | 0.3952 | 0.3739 | **+0.0213** |
| `tum_fr3_walking_xyz_person_04_POLYGON` | `tum_fr3_walking_xyz` | `person_04` | `person` | `IMAGE_POLYGON_MASK` | 0.3952 | 0.3751 | **+0.0201** |
| `tum_fr3_walking_xyz_monitor_01_POLYGON` | `tum_fr3_walking_xyz` | `monitor_01` | `computer monitor` | `IMAGE_POLYGON_MASK` | 0.3952 | 0.3965 | **-0.0013** |
| `openloris_corridor1_4_person_01_POLYGON` | `openloris_corridor1-4` | `person_01` | `person` | `IMAGE_POLYGON_MASK` | 1.7045 | 1.7867 | **-0.0822** |