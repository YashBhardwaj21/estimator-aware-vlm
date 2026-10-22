"""Pytest configuration and Windows-safe fixtures for vlm tests."""

import os
from pathlib import Path
import shutil
import uuid
import pytest

_TEST_TMP_BASE = Path(__file__).resolve().parent / "_test_run_tmp"


@pytest.fixture
def tmp_path():
    """Workspace-contained, Windows-safe tmp_path fixture that avoids 0o700 ACL issues."""
    _TEST_TMP_BASE.mkdir(parents=True, exist_ok=True)
    temp_dir = _TEST_TMP_BASE / f"run_{uuid.uuid4().hex[:12]}"
    temp_dir.mkdir(parents=True, exist_ok=True)
    try:
        yield temp_dir
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)
