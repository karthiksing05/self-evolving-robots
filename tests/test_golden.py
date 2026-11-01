"""Golden verdicts (spec 12.2): stock arm + seed library on the reference panel."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
GOLDEN = ROOT / "tests" / "golden" / "reference_verdicts.json"


@pytest.mark.slow
def test_reference_panel_matches_golden() -> None:
    spec = importlib.util.spec_from_file_location("record_golden", ROOT / "scripts" / "record_golden.py")
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    got = mod.record()
    want = json.loads(GOLDEN.read_text())["verdicts"]
    assert {k: v["verdict"] for k, v in got.items()} == {k: v["verdict"] for k, v in want.items()}, (
        "golden verdicts moved: run scripts/record_golden.py, review the diff, and record a decision "
        "(docs/iteration.md 4.1-4.4)")
