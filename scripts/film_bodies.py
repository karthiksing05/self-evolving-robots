"""Films of each change a run kept: the body before it failing and the body after it holding, on the task
groups the change gained (docs/evolve.md, D-050).

    uv run python scripts/film_bodies.py runs/evolve/gripper_s2v3_s0 [more runs ...] --out results/evolve/films

For each kept change, up to ``--per-change`` gained groups are filmed, distinct failure mechanisms first.
Each film replays one train task twice with the controls the run recorded (``sea.worlds.film``): the
previous body with System 1's best failing controls, then the new body with the controls that passed.
Nothing is searched again.  The two play side by side (in turn, when they have sound) and end on their
verdicts.  Writes ``<out>/<run>/<k>_<body id>_<group>.mp4`` with a JPEG poster, and ``films.json``: per
film, the change, the task, and each side's verdict on the replay beside the run's own.

A run recorded when any harness component its world's replay reads differed from today's (``film.READS``),
or with no harness.json to tell, is skipped: the replay would show something the run never saw.  Replays
run under the run's seed, and a film whose replay disagrees with the run's record is not written; nor is one
whose drawn robot goes more than ``film.CLIP_MM`` into the object or the ground (``Clip.clipping_mm``).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from render_bodies import _kept, frames


def _human(s: str) -> str:
    return s.replace("_", " ")


def _num(v) -> str:
    return f"{v:g}" if isinstance(v, (int, float)) else str(v)


def _called(body) -> str:
    """A body by what it is: the mechanism it was built as, or its id where it was changed by parameters and swaps."""
    for part in body.parts.values():
        if part.mechanism:
            return _human(json.loads(part.mechanism)["name"])
    return body.id


def changes(prev, new) -> tuple[str, frozenset[str]]:
    """What a kept change did to the body, in a line ("What changed: pads none -> pads (pad angle deg 7, ...)"),
    and the slots it touched."""
    said, slots = [], set()
    for slot in sorted(set(prev.parts) | set(new.parts)):
        a, b = prev.parts.get(slot), new.parts.get(slot)
        if a == b:
            continue
        slots.add(slot)
        if a is not None and b is not None and a.mechanism != b.mechanism and not (set(b.params.items()) - set(a.params.items())):
            said.append(f"{_human(json.loads(a.mechanism)['name']) if a.mechanism else _human(a.type)} → {_human(json.loads(b.mechanism)['name'])}")
        elif a is None or b is None or a.type != b.type:
            shown = ", ".join(f"{_human(k)} {_num(v)}" for k, v in list((b.params if b else {}).items())[:3])
            said.append(f"{_human(slot)} {_human(a.type) if a else 'none'} → {_human(b.type) if b else 'none'}"
                        + (f" ({shown})" if shown else ""))
        else:
            moved = [f"{_human(k)} {_num(a.params.get(k, '-'))} → {_num(v)}" for k, v in b.params.items()
                     if a.params.get(k) != v]
            said.append(f"{_human(slot)}: " + ", ".join(moved[:3]))
    return "What changed: " + "; ".join(said), frozenset(slots)


def main(argv: list[str] | None = None) -> int:
    from sea.evolve.core import Body
    from sea.evolve.loop import make_world
    from sea.harness import fingerprint
    from sea.worlds.film import CLIP_MM, FILM, READS, encode, grip_headline, pair

    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--per-change", type=int, default=3)
    args = ap.parse_args(argv)
    now = fingerprint()
    for run in args.runs:
        if not (run / "summary.json").exists():
            continue
        if not (run / "harness.json").exists():
            print(f"{run.name}: no harness.json, so what it was judged by is unknown; not filmed", file=sys.stderr)
            continue
        world_name, fs, _ = frames(run)
        if world_name not in FILM:
            print(f"{run.name}: the {world_name} world has no films", file=sys.stderr)
            continue
        then = json.loads((run / "harness.json").read_text()).get("components", {})
        moved = [c for c in READS[world_name] if then.get(c, {}).get("hash") != now[c]["hash"]]
        if moved:
            print(f"{run.name}: recorded before {', '.join(moved)} changed; not filmed", file=sys.stderr)
            continue
        tasks = {t.id: t for t in make_world(world_name).tasks()}
        seed = int(json.loads((run / "summary.json").read_text()).get("seed", 0))
        recs = {r["iteration"]: r for r in map(json.loads, (run / "iterations.jsonl").read_text().splitlines())}
        out, d = [], args.out / run.name
        for k, f in enumerate(fs):
            rec = recs.get(f["iteration"] - 1)
            trial = _kept(rec, f["id"]) if rec else None
            if trial is None:
                continue
            chosen, seen = [], set()
            fails = [(tid, o) for tid, o in rec["failing"].items() if tid in tasks and tasks[tid].split == "train"]
            for distinct in (True, False):  # one film per failure mechanism first, then the rest
                for g in trial.get("gained", []):
                    hit = next(((tid, o) for tid, o in fails if tasks[tid].group == g), None)
                    if hit and g not in {c[0] for c in chosen} and (not distinct or hit[1]["mechanism"] not in seen):
                        chosen.append((g, *hit))
                        seen.add(hit[1]["mechanism"])
            for g, tid, before in chosen[: args.per_change]:
                after = trial.get("outcomes", {}).get(tid, {})
                prev_body, new_body = Body.model_validate(fs[k - 1]["body"]), Body.model_validate(f["body"])
                change, slots = changes(prev_body, new_body)
                kw = {"changed": slots} if world_name.startswith("gripper") else {}
                b = FILM[world_name](prev_body, tasks[tid], before.get("controls", {}), seed, **kw)
                a = FILM[world_name](new_body, tasks[tid], after.get("controls", {}), seed, **kw)
                if b.ok or a.ok != bool(after.get("success")):
                    print(f"{run.name} {f['id']} {tid}: the replay ({b.verdict} / {a.verdict}) disagrees with the "
                          f"record; not filmed", file=sys.stderr)
                    continue
                deep = max(b.clipping_mm, a.clipping_mm)
                if deep > CLIP_MM:  # a film must not show parts inside what they touch
                    print(f"{run.name} {f['id']} {tid}: a drawn part goes {deep:.1f} mm into what it touches "
                          f"(limit {CLIP_MM} mm); not filmed", file=sys.stderr)
                    continue
                if not b.ok and before.get("mechanism") and world_name.startswith("gripper"):
                    b.verdict = f"{_human(before['mechanism'])} ({b.verdict})"
                    b.headline = grip_headline(tasks[tid], before["mechanism"])
                title = tasks[tid].facts if len(tasks[tid].facts) <= 110 else tasks[tid].facts[:107] + "…"
                film, audio = pair(b, a, title, (f"before · {_called(prev_body)}", f"after · {_called(new_body)} (this change)"),
                                   change=change)
                name = f"{k:02d}_{f['id']}_{g}.mp4"
                encode(film, d / name, audio)
                out.append({"frame": k, "id": f["id"], "group": g, "task": tid, "file": name,
                            "poster": name.replace(".mp4", ".jpg"), "seconds": round(len(film) / 25, 1),
                            "sound": audio is not None, "clipping_mm": round(deep, 2),
                            "change": change,
                            "before": {"ok": b.ok, "verdict": b.verdict, "headline": b.headline,
                                       "recorded": before.get("mechanism", "")},
                            "after": {"ok": a.ok, "verdict": a.verdict, "headline": a.headline,
                                      "recorded": after.get("success")}})
                print(f"{run.name} {name}: before {'pass' if b.ok else 'fail'}, after {'pass' if a.ok else 'fail'}")
        if out:
            d.mkdir(parents=True, exist_ok=True)
            (d / "films.json").write_text(json.dumps({"run": run.name, "world": world_name, "films": out}, indent=1,
                                                     ensure_ascii=False) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
