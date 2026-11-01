"""CLI, report and harness fingerprint on one tiny offline run."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
import pytest
from typer.testing import CliRunner

from sea import harness
from sea.cli import app
from sea.loop import evolve
from sea.report import bootstrap_ci, write_report

from .test_reward_audit import tiny_spec


@pytest.fixture(scope="module")
def run_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    run = tmp_path_factory.mktemp("cli") / "tiny"
    evolve(tiny_spec(), run, log=lambda s: None)
    return run


def test_panels_command(tmp_path: Path) -> None:
    out = tmp_path / "panels.jsonl"
    res = CliRunner().invoke(app, ["panels", "--split", "train", "--n", "9", "--seed", "1", "--out", str(out)])
    assert res.exit_code == 0, res.output
    data = json.loads(res.output)
    assert data["panels"] == 9 and sum(data["controls_by_class"].values()) >= 9 * 4
    assert len(out.read_text().splitlines()) == 9
    res = CliRunner().invoke(app, ["panels", "--split", "reference"])
    assert json.loads(res.output)["panels"] == 1


def test_reward_command_matches_reports(run_dir: Path) -> None:
    res = CliRunner().invoke(app, ["reward", str(run_dir)])
    assert res.exit_code == 0, res.output
    rec = json.loads(res.output)
    reports = [json.loads(p.read_text()) for p in sorted(run_dir.glob("iter_*/report.json"))]
    assert [r["reward"] for r in rec["iterations"]] == [r["reward"] for r in reports]


def test_eval_command_writes_split_dir(run_dir: Path) -> None:
    res = CliRunner().invoke(app, ["eval", str(run_dir), "--split", "val"])
    assert res.exit_code == 0, res.output
    assert json.loads(res.output)["split"] == "val"
    assert (run_dir / "eval_val" / "results.jsonl").exists() and (run_dir / "eval_val" / "summary.json").exists()


def test_report_single_and_grouped(run_dir: Path, tmp_path: Path) -> None:
    res = CliRunner().invoke(app, ["report", str(run_dir)])
    assert res.exit_code == 0, res.output
    md = Path(res.output.strip())
    assert md.exists() and "| s2 |" in md.read_text()
    assert (run_dir / "final" / "plots" / "coverage_vs_iteration.png").exists()
    group = tmp_path / "group"
    for k in ("a", "b"):
        shutil.copytree(run_dir, group / k)
    out = write_report(group)
    assert "| s2 | 2 |" in out.read_text()
    # a run judged by a different verifier version must not be pooled silently
    hj = json.loads((group / "b" / "harness.json").read_text())
    hj["judge_key"] = "different"
    (group / "b" / "harness.json").write_text(json.dumps(hj))
    with pytest.raises(ValueError, match="different harness versions"):
        write_report(group)
    assert write_report(group, allow_mixed=True).exists()


def test_bootstrap_ci_edges() -> None:
    assert np.isnan(bootstrap_ci(np.array([]))[0])
    assert bootstrap_ci(np.array([0.5])) == (0.5, 0.5, 0.5)
    m, lo, hi = bootstrap_ci(np.array([0.0, 1.0, 0.5, 0.25]))
    assert lo <= m <= hi


def test_every_module_is_fingerprinted() -> None:
    covered = {f for _, files in harness.COMPONENTS.values() for f in files}
    modules = {str(p.relative_to(harness.ROOT)) for p in harness.SRC.rglob("*.py") if p.name != "__init__.py"}
    assert modules - covered - set(harness.TOOLING) == set()  # a new module needs a component and a tier
    assert all((harness.ROOT / f).exists() for f in covered)


def test_fingerprint_tiers_and_diff() -> None:
    fp = harness.fingerprint()
    assert {v["tier"] for v in fp.values()} == {"judge", "world", "skill", "policy"}
    key = harness.judge_key(fp)
    changed = {**fp, "physics": {"tier": "world", "hash": "x"}}
    assert harness.judge_key(changed) == key  # world-tier edits keep runs comparable
    assert harness.diff(fp, changed) == {"physics": "world"}
    judged = {**fp, "verifier": {"tier": "judge", "hash": "x"}}
    assert harness.judge_key(judged) != key
