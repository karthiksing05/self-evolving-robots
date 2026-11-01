"""Shared test fixtures."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

os.environ.setdefault("MUJOCO_GL", "egl" if os.uname().sysname == "Linux" else "glfw")

ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / "tests" / "fixtures"


@pytest.fixture
def repo_root() -> Path:
    return ROOT


@pytest.fixture
def fixtures_dir() -> Path:
    return FIXTURES
