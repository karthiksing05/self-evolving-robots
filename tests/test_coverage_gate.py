"""The per-package coverage gate script (spec 12.1)."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("check_package_coverage", ROOT / "scripts" / "check_package_coverage.py")
gate = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(gate)


def _file(cov: int, n: int, bcov: int = 0, bn: int = 0) -> dict:
    return {"summary": {"covered_lines": cov, "num_statements": n, "covered_branches": bcov, "num_branches": bn}}


def _report(pct: dict[str, tuple[int, int]], extra: dict | None = None) -> dict:
    files = {}
    for pkg, (c, n) in pct.items():
        name = "src/sea/contracts.py" if pkg == "contracts" else f"src/sea/{pkg}/mod.py"
        files[name] = _file(c, n)
    files.update(extra or {})
    cov = sum(f["summary"]["covered_lines"] for f in files.values())
    num = sum(f["summary"]["num_statements"] for f in files.values())
    return {"files": files, "totals": {"covered_lines": cov, "num_statements": num, "covered_branches": 0,
                                       "num_branches": 0}}


def test_package_mapping():
    assert gate.package_of("src/sea/verify/verdicts.py") == "verify"
    assert gate.package_of("/abs/.venv/x/sea/contracts.py") == "contracts"
    assert gate.package_of("src/sea/loop.py") == "loop"
    assert gate.package_of("tests/test_x.py") is None


def test_passing_and_failing_reports(tmp_path, capsys):
    ok = _report({k: (96, 100) for k in gate.GATES})
    assert gate.check(ok) == []
    bad = _report({**{k: (96, 100) for k in gate.GATES}, "verify": (90, 100)},
                  extra={"src/sea/loop.py": _file(0, 300)})
    fails = gate.check(bad)
    assert any("sea.verify" in f for f in fails) and any("overall" in f for f in fails)
    missing = _report({k: (96, 100) for k in gate.GATES if k != "sim"})
    assert gate.check(missing) == ["sea.sim: no files measured (gate 80%)"]
    p = tmp_path / "coverage.json"
    p.write_text(json.dumps(ok))
    assert gate.main([str(p)]) == 0
    p.write_text(json.dumps(bad))
    assert gate.main([str(p)]) == 1
    assert "FAIL" in capsys.readouterr().out


def test_branches_count():
    pct, overall = gate.summarize({"files": {"src/sea/sim/a.py": _file(8, 10, 1, 10)},
                                   "totals": {"covered_lines": 8, "num_statements": 10, "covered_branches": 1,
                                              "num_branches": 10}})
    assert pct["sim"] == overall == 45.0
