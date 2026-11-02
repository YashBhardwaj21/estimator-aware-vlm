# TUM fr3_walking_halfsphere Benchmark & Intervention Report

## 1. Overview

In accordance with Section 3.8 of the research protocol, the TUM `freiburg3_walking_halfsphere` sequence provides a substantial multi-view moving-camera path (1021 synchronized RGB-D frames, 3582 ground-truth Vicon poses) to rigorously evaluate dynamic-object suppression where camera motion is non-trivial.

## 2. Quantitative Results

| Arm | Name | Classification | Targets | ATE RMSE (m) | Delta ATE (m) | Improvement | Ref Coverage | Input Tracking |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `S1` | `geometry_only_suppression` | `GEOMETRY_EVIDENCE` | `chair_01,chair_02,chair_03,monitor_01,monitor_02,monitor_03,monitor_04,mouse_01,person_01,person_02,person_03,person_04,person_05` | **0.0392** | +0.3819 | **+90.69%** | 28.42% | 100.00% |
| `S2` | `polygon_moving_human_suppression` | `OBJECT_INSTANCE_MASK` | `person_01,person_02,person_03,person_04,person_05` | **0.0368** | +0.3843 | **+91.26%** | 28.42% | 100.00% |
| `S3` | `geometry_plus_vlm_suppression` | `GEOMETRY_PLUS_VLM` | `person_01,person_02,person_03,person_04,person_05` | **0.0345** | +0.3866 | **+91.81%** | 28.42% | 100.00% |
| `S4` | `oracle_human_mask_diagnostic` | `ORACLE_DIAGNOSTIC` | `person_01,person_02,person_03,person_04,person_05` | **0.0502** | +0.3709 | **+88.08%** | 28.42% | 100.00% |

## 3. Analysis & Findings

Dynamic human suppression on `fr3_walking_halfsphere` demonstrates consistent error reductions over the raw baseline control.
Because the camera follows a complex halfsphere trajectory, removing dynamic moving features stabilizes the local map and prevents tracking corruption.