"""Per-package coverage gates (spec 12.1).

Reads ``coverage.json`` (``pytest --cov=sea --cov-branch --cov-report=json``) and fails when a
package is below its minimum.  Coverage is coverage.py's combined line + branch measure:
(covered lines + covered branches) / (statements + branches), summed over the package's files.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

GATES: dict[str, float] = {
    "verify": 95.0, "spec": 95.0, "compose": 95.0,
    "contracts": 90.0, "memory": 90.0, "propose": 90.0,
    "control": 80.0, "design": 80.0, "sim": 80.0,
}
OVERALL = 85.0


def package_of(path: str) -> str | None:
    parts = Path(path).as_posix().split("/")
    if "sea" not in parts:
        return None
    rest = parts[len(parts) - 1 - parts[::-1].index("sea") + 1:]
    if not rest:
        return None
    if len(rest) == 1:
        return rest[0].removesuffix(".py")  # sea/contracts.py -> "contracts"; loop.py, cli.py -> overall only
    return rest[0]


def summarize(report: dict) -> tuple[dict[str, float], float]:
    acc: dict[str, list[int]] = {}
    for path, data in report["files"].items():
        pkg = package_of(path)
        if pkg is None:
            continue
        s = data["summary"]
        a = acc.setdefault(pkg, [0, 0])
        a[0] += s["covered_lines"] + s.get("covered_branches", 0)
        a[1] += s["num_statements"] + s.get("num_branches", 0)
    pct = {k: (100.0 * c / n if n else 100.0) for k, (c, n) in acc.items()}
    t = report["totals"]
    total_n = t["num_statements"] + t.get("num_branches", 0)
    overall = 100.0 * (t["covered_lines"] + t.get("covered_branches", 0)) / total_n if total_n else 100.0
    return pct, overall


def check(report: dict) -> list[str]:
    pct, overall = summarize(report)
    failures = []
    for pkg, gate in GATES.items():
        got = pct.get(pkg)
        if got is None:
            failures.append(f"sea.{pkg}: no files measured (gate {gate:.0f}%)")
        elif got + 1e-9 < gate:
            failures.append(f"sea.{pkg}: {got:.1f}% < {gate:.0f}%")
    if overall + 1e-9 < OVERALL:
        failures.append(f"overall: {overall:.1f}% < {OVERALL:.0f}%")
    return failures


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    path = Path(argv[0] if argv else "coverage.json")
    report = json.loads(path.read_text())
    pct, overall = summarize(report)
    for pkg in sorted(set(GATES) | set(pct)):
        gate = GATES.get(pkg)
        got = pct.get(pkg)
        print(f"sea.{pkg:<10s} {'-' if got is None else f'{got:5.1f}%'}  gate {'-' if gate is None else f'{gate:.0f}%'}")
    print(f"overall       {overall:5.1f}%  gate {OVERALL:.0f}%")
    failures = check(report)
    for f in failures:
        print("FAIL", f)
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
