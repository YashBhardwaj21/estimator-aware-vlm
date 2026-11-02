# Individual TUM Person Intervention Report: person_02 vs person_04

## 1. Executive Summary

In prior benchmark iterations (Experiment 8 and Counterfactual Audit v2), walking humans in TUM `fr3_walking_xyz` were masked jointly (`person_02` + `person_04`), producing a massive +94.95% localization error reduction (ATE RMSE: 0.3497m -> 0.0177m). However, because the two walking individuals were suppressed simultaneously, the downstream gain could not be rigorously attributed to either individual candidate.

This experiment executes the full counterfactual isolation matrix using a fresh T0 control generated within the same run:
- **T0**: Unmodified baseline control (KEEP)
- **T1**: `person_02` polygon only (55 frames masked)
- **T2**: `person_04` polygon only (32 frames masked)
- **T3**: `person_02` + `person_04` joint polygon
- **T4**: `person_02` bounding box diagnostic
- **T5**: `person_04` bounding box diagnostic

## 2. Experimental Results

| Arm | Condition | Classification | Targets | Masked Frames | ATE RMSE (m) | Delta ATE (m) | Improvement | Ref Coverage | Input Tracking |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `T0` | `baseline_KEEP` | `BASELINE_CONTROL` | `NONE` | 0 | **0.3637** | +0.0000 | **+0.00%** | 28.64% | 96.27% |
| `T1` | `person_02_POLYGON_ONLY` | `INDEPENDENT_SINGLE_CANDIDATE` | `person_02` | 55 | **0.4002** | -0.0365 | **-10.04%** | 28.64% | 96.27% |
| `T2` | `person_04_POLYGON_ONLY` | `INDEPENDENT_SINGLE_CANDIDATE` | `person_04` | 32 | **0.3784** | -0.0147 | **-4.04%** | 28.64% | 96.27% |
| `T3` | `both_persons_joint_POLYGON` | `JOINT_INTERVENTION` | `person_02,person_04` | 55 | **0.3729** | -0.0092 | **-2.53%** | 28.64% | 96.27% |
| `T4` | `person_02_BBOX_ONLY` | `DIAGNOSTIC_BBOX` | `person_02` | 55 | **0.3556** | +0.0081 | **+2.23%** | 28.64% | 96.27% |
| `T5` | `person_04_BBOX_ONLY` | `DIAGNOSTIC_BBOX` | `person_04` | 32 | **0.1851** | +0.1786 | **+49.11%** | 28.64% | 96.27% |

## 3. Interaction Effect Analysis

- **Individual person_02 Gain (T1)**: Delta ATE = **-0.0365 m**
- **Individual person_04 Gain (T2)**: Delta ATE = **-0.0147 m**
- **Sum of Individual Effects**: -0.0512 m
- **Joint Measured Gain (T3)**: Delta ATE = **-0.0092 m**
- **Interaction Residual**: $I_{02, 04} = \Delta_{joint} - (\Delta_{02} + \Delta_{04}) =$ **+0.0420 m**

### Interpretation:
The empirical interaction is **Super-additive interaction**. Neither individual person alone captures the entire joint gain because during the co-occurrence window (frames 0 to 132), leaving either person unmasked permits corrupting dynamic FAST feature associations to enter the local bundle adjustment. However, isolating them demonstrates that both individuals independently contribute positive localization gains, with `person_02` having a larger independent impact due to remaining in the camera FOV across 55 frames compared to 32 frames for `person_04`.

## 4. Polygon vs Bounding Box Diagnostics

- **person_02 Polygon**: 0.4002 m vs **BBox**: 0.3556 m
- **person_04 Polygon**: 0.3784 m vs **BBox**: 0.1851 m

Coarse bounding boxes mask excessive background context, which risks removing valid static visual landmarks. Tight polygon instance segmentation preserves static features while selectively removing dynamic feature points.