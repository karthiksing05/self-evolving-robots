"""The outer loop's input: every idea System 2 recorded, across runs, by world (docs/evolve.md, D-047).

    uv run python scripts/ideas.py runs/evolve/gripper_s2_s0 runs/evolve/gripper_s2v2_s0 ...
    uv run python scripts/ideas.py runs/evolve/*/ --world gripper --out results/evolve/ideas_gripper.md

An idea is a part the catalogue lacks.  It is never trialled; it waits for someone to build it into the
world's catalogue.  This lists each idea with:
- the run and iteration it came from;
- the slot it would fill;
- its costing against the bench, when the run recorded one (D-047);
- the failure mechanisms still open when it was proposed.

Ideas are grouped by the words their names share, so the ones that recur across runs stand out.
Choosing which to build is a decision for whoever owns the world's catalogue.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

STOP = {"a", "an", "the", "of", "for", "with", "and", "on", "to", "in", "jaw", "pad", "moving", "fixed"}


def _words(name: str) -> frozenset[str]:
    return frozenset(w for w in re.findall(r"[a-z0-9]+", name.lower()) if w not in STOP and len(w) > 2)


def _needs(idea: dict) -> str:
    needs = idea.get("needs", [])
    return ", ".join(n if isinstance(n, str) else f"{n['item']} x{n.get('qty', 1):g}" for n in needs) or "none listed"


def collect(runs: list[Path], world: str | None) -> list[dict]:
    out = []
    for run in runs:
        if not (run / "iterations.jsonl").exists():
            continue
        recs = [json.loads(x) for x in (run / "iterations.jsonl").read_text().splitlines()]
        cfg = json.loads((run / "config.json").read_text()) if (run / "config.json").exists() else {}
        if world and cfg.get("world") != world:
            continue
        for r in recs:
            open_mech = Counter(o["mechanism"] for o in r.get("failing", {}).values())
            for t in r.get("trials", []):
                idea = (t.get("proposal") or {}).get("idea")
                if not idea or t.get("move") != "idea" and (t.get("proposal") or {}).get("move") != "idea":
                    continue
                out.append({"world": cfg.get("world", "?"), "run": run.name, "iteration": r["iteration"] + 1,
                            "name": idea["name"], "slot": idea.get("slot", ""), "description": idea.get("description", ""),
                            "needs": _needs(idea), "test": idea.get("test", ""), "result": t.get("result", ""),
                            "open": dict(open_mech.most_common(4)), "mechanism": t["proposal"].get("mechanism", "")})
    return out


def group(ideas: list[dict]) -> list[list[dict]]:
    """Ideas whose names share at least two content words (or one, for one-word names) form a family."""
    fams: list[tuple[set[str], list[dict]]] = []
    for i in ideas:
        w = set(_words(i["name"]))
        for keys, members in fams:
            if len(w & keys) >= min(2, len(w), len(keys)):
                keys |= w
                members.append(i)
                break
        else:
            fams.append((w, [i]))
    return [m for _, m in sorted(fams, key=lambda f: (-len({x["run"] for x in f[1]}), -len(f[1])))]


def render(fams: list[list[dict]]) -> str:
    out = ["# Ideas System 2 recorded (the outer loop's input)", ""]
    for fam in fams:
        runs = sorted({i["run"] for i in fam})
        out.append(f"## {fam[0]['name']} ({len(fam)} times, {len(runs)} runs)")
        for i in fam:
            out.append(f"- **{i['name']}** ({i['run']}, iteration {i['iteration']}"
                       + (f", slot `{i['slot']}`" if i["slot"] else "") + f"): {i['description'][:400]}")
            out.append(f"  - addresses: {i['mechanism'][:160]}; open then: "
                       + ", ".join(f"{k} x{v}" for k, v in i["open"].items()))
            out.append(f"  - needs: {i['needs']}")
            cost = i["result"].split("it would need ", 1)
            if len(cost) == 2:
                out.append(f"  - costed: {cost[1]}")
        out.append("")
    return "\n".join(out)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+", type=Path)
    ap.add_argument("--world", default=None)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args(argv)
    ideas = collect(args.runs, args.world)
    text = render(group(ideas))
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text + "\n")
    print(text)
    counts = defaultdict(int)
    for i in ideas:
        counts[i["world"]] += 1
    print(f"\n{len(ideas)} ideas: {dict(counts)}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
