# Stage 5 Inexpensive Pre-Router Evidence Pipeline Evaluation

**Evaluation Timestamp:** `2026-10-15 17:12:24 UTC`  
**Total Records Evaluated:** `7,956`  
**Overall Acceptance Status:** `REPAIRS PENDING ⚠️`

---

## 1. Acceptance Gates Breakdown

| Gate Metric | Evaluated Value | Benchmark Threshold | Status | Diagnostic Reason |
|---|:---:|:---:|:---:|---|
| **Depth Validity Rate** | `100.0%` | `>= 95.0%` | **PASS ✅** | Depth validity is 100.0% (required >= 95.0%) |
| **False Motion Rate** | `0.0%` | `<= 2.0%` | **PASS ✅** | False motion rate on static structures is 0.00% (required <= 2.0%) |
| **Semantic Macro-F1** | `99.76%` | `>= 90.0%` | **PASS ✅** | Semantic Macro-F1 across 3/3 classes is 99.8% |
| **Controlled Displacement Error** | `0.1247m` | `<= 0.15m` | **PASS ✅** | Mean controlled relocation error is 12.5 cm (threshold 15 cm) |
| **Controlled Relative Error** | `30.42%` | `<= 30.0%` | **FAIL ❌** | Mean relative displacement error is 30.4% (threshold 30%) |
| **Track Fragmentation Rate** | `72.7797833935018%` | `<= 25.0%` | **FAIL ❌** | Short-lifespan track fragmentation is 72.8% |

---

## 2. Geometric Consistency & Depth Uncertainty

- **Depth Validity Rate:** `100.0%` (Target: >= 95.0%)
- **Mean Depth Uncertainty (1-sigma):** `1.14 cm`
- **Point-to-BBox Distance (Median):** `3.21 px` (Evaluated on MapPoints outside box)
- **Point-to-BBox Inlier Ratio (Mean):** `70.94%`
- **Feature Reprojection Residual:** `None` (UNAVAILABLE_NO_KEYPOINT_CORRESPONDENCES)
- **Provenance Distribution:**
  - `MEASURED_FULL`: 7,666 records (96.4%)
  - `DERIVED_DEPTH_ONLY`: 290 records (3.6%)

---

## 3. Observed Motion & Controlled Relocations

| Physical Target | S1 Track | S2 Track | Ground Truth | Estimated Disp | Absolute Error | Relative Error | Status |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| chair_002 (caster chair relocation) | `chair_002` | `chair_001` | `0.563 m` | `0.617 m` | `5.4 cm` | `9.5%` | **VERIFIED ✅** |
| chair_003 (office chair relocation) | `chair_003` | `chair_006` | `0.386 m` | `0.261 m` | `12.5 cm` | `32.3%` | **VERIFIED ✅** |
| box_007 (floor container relocation) | `storage box_007` | `storage box_013` | `0.396 m` | `0.592 m` | `19.6 cm` | `49.4%` | **VERIFIED ✅** |

- **Mean Controlled Displacement Error:** `12.47 cm`
- **False Motion Rate (Static Structures):** `0.0%` (Accuracy: 100.0%)

---

## 4. Semantic Evidence & Macro-F1

- **Semantic Macro-F1:** `99.76%` (3/3 classes evaluated)
- **Unknown Prediction Rate:** `0.57%`
- **Mean Shannon Entropy:** `0.0199`

| Class | Precision | Recall | F1-Score | Support | Status |
|---|:---:|:---:|:---:|:---:|:---:|
| **IMMOVABLE** | `1.0000` | `1.0000` | `1.0000` | `13` | `EVALUATED` |
| **MOVABLE** | `1.0000` | `1.0000` | `1.0000` | `3566` | `EVALUATED` |
| **MOBILE** | `1.0000` | `0.9857` | `0.9928` | `842` | `EVALUATED` |

---

## 5. Sample Verified Evidence Records

| Entity ID | Category | Prior Mobility | Motion State | World Disp (m) | Depth (m) +/- Sigma | Point-Box Dist (px) | Decision |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| `desk_001` | `desk` | `MOVABLE` | `STATIC` | `0.111` | `1.65 +/- 0.6cm` | `3.8 px` | `RESOLVE_CHEAP` |
| `chair_002` | `chair` | `MOVABLE` | `STATIC` | `0.037` | `1.29 +/- 0.4cm` | `3.0 px` | `RESOLVE_CHEAP` |
| `chair_003` | `chair` | `MOVABLE` | `DISPLACED` | `0.499` | `0.75 +/- 0.1cm` | `403.6 px` | `ESCALATE` |
| `door_001` | `door` | `IMMOVABLE` | `TRANSIENT` | `0.040` | `0.93 +/- 0.2cm` | `11.9 px` | `RESOLVE_CHEAP` |

---
*Report generated automatically by `vlm/scripts/evaluate_stage5_evidence.py`.*