"""Intervention Manifest Management.

Maintains cryptographic provenance and status tracking for counterfactual experiments:
- Tracks all planned, running, succeeded, failed, and not-implemented interventions.
- Records input hashes, start/end timestamps, and execution summaries.
- Supports atomic JSON file persistence.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from estimator_aware_vlm.interventions.schema import (
    InterventionRunResult,
    InterventionSpec,
    InterventionStatus,
)


@dataclass
class InterventionManifest:
    """Persistent ledger of counterfactual intervention executions."""

    experiment_run_id: str
    manifest_version: str = "1.0.0"
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    total_planned: int = 0
    total_executed: int = 0
    total_success: int = 0
    total_failed: int = 0
    total_not_implemented: int = 0
    specs: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    results: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    @classmethod
    def create(cls, experiment_run_id: str, specs: List[InterventionSpec]) -> InterventionManifest:
        """Create a new manifest initialized with planned specs."""
        now = time.time()
        spec_dict = {s.intervention_id: s.to_dict() for s in specs}
        return cls(
            experiment_run_id=experiment_run_id,
            created_at=now,
            updated_at=now,
            total_planned=len(specs),
            total_executed=0,
            total_success=0,
            total_failed=0,
            total_not_implemented=0,
            specs=spec_dict,
            results={},
        )

    def record_result(self, result: InterventionRunResult) -> None:
        """Record the outcome of an intervention run."""
        self.results[result.intervention_id] = result.to_dict()
        self.updated_at = time.time()
        self.total_executed = len(self.results)
        self.total_success = sum(
            1 for r in self.results.values() if r.get("status") == InterventionStatus.SUCCESS.value
        )
        self.total_failed = sum(
            1 for r in self.results.values() if r.get("status") == InterventionStatus.FAILED.value
        )
        self.total_not_implemented = sum(
            1 for r in self.results.values() if r.get("status") == InterventionStatus.NOT_IMPLEMENTED.value
        )

    def save(self, filepath: Path) -> None:
        """Atomically save manifest to disk."""
        filepath.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = filepath.with_suffix(".tmp")
        data = asdict(self)
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        if filepath.exists():
            filepath.unlink()
        tmp_path.rename(filepath)

    @classmethod
    def load(cls, filepath: Path) -> InterventionManifest:
        """Load manifest from disk."""
        if not filepath.exists():
            raise FileNotFoundError(f"Manifest not found: {filepath}")
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls(
            experiment_run_id=data["experiment_run_id"],
            manifest_version=data.get("manifest_version", "1.0.0"),
            created_at=data.get("created_at", time.time()),
            updated_at=data.get("updated_at", time.time()),
            total_planned=data.get("total_planned", 0),
            total_executed=data.get("total_executed", 0),
            total_success=data.get("total_success", 0),
            total_failed=data.get("total_failed", 0),
            total_not_implemented=data.get("total_not_implemented", 0),
            specs=data.get("specs", {}),
            results=data.get("results", {}),
        )

