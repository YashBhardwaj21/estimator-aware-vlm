#!/usr/bin/env python3
"""Downstream SLAM & Dynamic Scene Graph Validation.

Evaluates the primary research question:
Does estimator-aware selective VLM reasoning improve the reliability of SLAM observations
in changing indoor environments compared with strong geometry, temporal, semantic,
and non-estimator-aware routing baselines, while reducing unnecessary VLM inference?

Validates:
1. Map Consistency & Landmark Reliability:
   - Correct landmark retention for stationary entities
   - False invalidations of reliable stationary observations
   - Correct invalidation of observations made stale by relocation
   - Decoupled pose history maintenance vs phantom averaging

2. Downstream Metric SLAM Trajectory Evaluation:
   - Absolute Trajectory Error (ATE RMSE)
   - Relative Pose Error (RPE RMSE)
   - Tracking stability and loss frequency across recorded sequences

3. Compute Efficiency & Decision Gating:
   - VLM queries avoided vs All-VLM baseline
   - Estimator-influence gating: zero-influence clutter vs landmark protection

Generates:
  - vlm/outputs/downstream_slam_effects.json
  - vlm/outputs/downstream_slam_effects_report.md
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict

repo_root = Path(__file__).resolve().parents[1]
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from scripts.evaluate_policy_specific_downstream import (
    run_policy_specific_downstream_evaluation,
)


def evaluate_downstream_effects() -> Dict[str, Any]:
    return run_policy_specific_downstream_evaluation()


if __name__ == "__main__":
    evaluate_downstream_effects()
