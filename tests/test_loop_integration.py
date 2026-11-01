"""Loop integration (spec 12.2): 3 iterations, 3 panels, MockLLM; logs, reward audit, determinism."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from sea.contracts import RunSpec
from sea.loop import evolve
from sea.verify.reward import main as reward_main

ROOT = Path(__file__).resolve().parents[1]


def spec(**kw) -> RunSpec:
    base = yaml.safe_load((ROOT / "configs" / "exp" / "smoke.yaml").read_text())
    base.update(kw)
    return RunSpec.model_validate(base)


def _files(run: Path) -> dict[str, bytes]:
    skip = {"timing.json", "llm_calls.jsonl"}
    return {str(p.relative_to(run)): p.read_bytes() for p in sorted(run.rglob("*"))
            if p.is_file() and p.name not in skip and p.suffix in (".json", ".jsonl", ".yaml")}


@pytest.mark.slow
def test_three_iterations_mock_llm_deterministic(tmp_path: Path) -> None:
    fixtures = tmp_path / "fixtures"
    # Record System 2 answers once (scripted System 2), then replay them twice through MockLLM.
    evolve(spec(llm="scripted", llm_fixtures=str(fixtures)), tmp_path / "rec" / "run", log=lambda s: None)
    assert any(fixtures.glob("*.json")), "the scripted run should have escalated at least once"
    outs = []
    for k in ("a", "b"):
        run = tmp_path / k / "run"
        outs.append(evolve(spec(llm="mock", llm_fixtures=str(fixtures)), run, log=lambda s: None))
        for it in sorted(run.glob("iter_*")):
            for name in ("results.jsonl", "report.json", "cost.json", "proposal.json", "decision.json"):
                assert (it / name).exists(), f"{it.name}/{name}"
            assert any((it / "traces").glob("*.npz"))
        assert (run / "memory" / "sidecar.json").exists()
        rec = reward_main(run)
        assert [r["reward"] for r in rec["iterations"]] == [
            json.loads(p.read_text())["reward"] for p in sorted(run.glob("iter_*/report.json"))]
    a, b = (_files(tmp_path / k / "run") for k in ("a", "b"))
    assert a.keys() == b.keys()
    diff = [name for name in a if a[name] != b[name]]
    assert diff == [], f"non-deterministic files: {diff}"
    rec_files = _files(tmp_path / "rec" / "run")
    assert rec_files["iter_000/report.json"] == a["iter_000/report.json"]
    assert outs[0] == outs[1]
