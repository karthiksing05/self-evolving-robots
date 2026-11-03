"""What breaks physical law in the bodies the runs kept, measured before any law decides anything (D-051).

    uv run python scripts/audit_mechanism.py runs/evolve/gripper_s2v3_s0 [more runs ...] --out results/mechanism_audit

Per kept body:
- **Gripper:** the SO-101's own STL parts and the pads, posed through the jaw's range.
  - Where parts on different bodies overlap: how many surface crossings, over what region, and how far
    from the jaw joint's axis.
  - Whether material carries each moving body: its clearance to its parent through the range. A body
    with no geometry at all is reported as such.
  - Contact penetration in the replays of the films (D-050): the deepest each step by kind of pair.
- **Mobile and talking robot:** these simulators carry no geometry, so the facts come from their own
  models.
  - Mobile: pack voltage, which the model never uses; stall current against the driver channel's
    rating; and rocker wheels that overlap.
  - Talking robot: articulator places past the end of the tract, and tongue pad and tip ranges that overlap.

Writes ``<out>/<run>.json`` and ``<out>/summary.md``.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from render_bodies import frames

from sea.mechanism.gripper import SAMPLES
from sea.mechanism.gripper import geometry as gripper_geometry

# Datasheet ratings per H-bridge channel (A, continuous): Toshiba TB6612FNG 1.2 A (3.2 A peak); ST L298 2 A.
CHANNEL_A = {"driver_tb6612": 1.2, "driver_l298n": 2.0}
CELL_V = {"nimh_pack": 1.2, "liion_pack": 3.7}  # nominal volts per cell: NiMH AA, 18650 Li-ion


# ---------------------------------------------------------------- gripper replays
def _labeller(model):
    arm = model.body("base").id  # the robot is every body on the arm, its jaw mechanism's included

    def label(b: int) -> str:
        name = model.body(b).name
        return "robot" if model.body_rootid[b] == arm else ("object" if name.startswith(("ctl_", "obj_")) else "table")
    return label


def gripper_contacts(run: Path, films: Path, world) -> list[dict]:
    from sea.evolve.core import Body
    from sea.mechanism.audit import Penetration
    from sea.worlds import film as F

    path = films / run.name / "films.json"
    if not path.exists():
        return []
    seed = int(json.loads((run / "summary.json").read_text()).get("seed", 0))
    tasks = {t.id: t for t in world.tasks()}
    recs = {r["iteration"]: r for r in map(json.loads, (run / "iterations.jsonl").read_text().splitlines())}
    _, fs, _ = frames(run)
    by_id = {f["id"]: (k, f) for k, f in enumerate(fs)}
    out = []
    for fm in json.loads(path.read_text())["films"]:
        k, f = by_id[fm["id"]]
        rec = recs[f["iteration"] - 1]
        trial = next(t for t in rec["trials"] if f"b{rec['iteration']:02d}{t['k']}" == f["id"])
        sides = {"before": (fs[k - 1]["body"], rec["failing"][fm["task"]].get("controls", {})),
                 "after": (f["body"], trial["outcomes"][fm["task"]].get("controls", {}))}
        row = {"film": fm["file"], "task": fm["task"]}
        for side, (b, ctl) in sides.items():
            clip = F.gripper(Body.model_validate(b), tasks[fm["task"]], ctl, seed, probe=lambda m: Penetration(m, _labeller(m)),
                             render=False)
            row[side] = {"ok": clip.ok, "body": b.get("id"), "penetration": clip.probe.summary()}
        out.append(row)
    return out


# ---------------------------------------------------------------- mobile and talking-robot facts
def mobile_facts(body) -> dict:
    from sea.worlds import mobile as M

    bt, dr, lo = body.parts["battery"], body.parts["drive"], body.parts["locomotion"]
    cells, n = M._cells(bt), M._motors(body)
    driver = M.DRIVER[dr.type][0]
    stall_a = M.MOTOR[dr.type][4]
    out = {"pack": f"{cells} x {bt.type}", "pack_v_if_series": round(cells * CELL_V[bt.type], 1),
           "voltage_in_model": False, "motors": n, "stall_a_each": stall_a, "driver": driver,
           "channel_a": CHANNEL_A[driver], "stall_over_channel": stall_a > CHANNEL_A[driver]}
    if lo.type == "rocker_bogie6":
        r, bogie = lo.params["radius"], lo.params["bogie"]
        out["bogie_wheels_overlap_cm"] = round(max(0.0, 2 * r - bogie), 2)
    return out


def phonology_facts(body) -> dict:
    from sea.worlds import phonology as P

    r = P.rig(body)
    out: dict = {"tract_cm": r.length, "controls": {k: [round(a, 2), round(b, 2)] for k, (a, b) in r.controls.items()}}
    out["past_the_tract_cm"] = {k: round(b - r.length, 2) for k, (_, b) in r.controls.items()
                                if k in ("tongue_place", "tip_place") and b > r.length}
    if "tip_place" in r.controls and "tongue_place" in r.controls:
        a = max(r.controls["tip_place"][0], r.controls["tongue_place"][0])
        b = min(r.controls["tip_place"][1], r.controls["tongue_place"][1])
        out["tip_and_pad_share_cm"] = [round(a, 2), round(b, 2)] if b > a else None
    return out


# ---------------------------------------------------------------- report
CAP = 4000  # surface crossings listed per pair (sea.mechanism.geometry.crossings)


def _n(k: int) -> str:
    return f"{k}+" if k >= CAP else str(k)


def _md(rows: list[dict]) -> str:
    out = ["# Mechanism audit (D-051)", "", "What breaks physical law in the bodies the runs kept, before any law decides "
           "anything. Generated by `scripts/audit_mechanism.py`.", ""]
    for r in rows:
        out.append(f"## {r['run']} ({r['world']})")
        for b in r["bodies"]:
            out.append(f"- **{b['id']}**")
            g = b.get("geometry")
            if g:
                for name, c in g["carried"].items():
                    out.append(f"  - {name} on a {c['joint']} from {c['parent']}: " + (
                        "no material at all" if not c["material"] else f"worst clearance to its parent {c['worst_gap_mm']} mm"))
                for pair, c in g["clashes"].items():
                    out.append(f"  - overlap {pair}: at {len(c['at_q'])}/{SAMPLES} jaw positions, up to {_n(max(c['crossings']))} "
                               f"crossings over {c['region_mm']} mm, {c['from_axis_mm'][0]}–{c['from_axis_mm'][1]} mm from the jaw axis")
            for k in ("facts",):
                if b.get(k):
                    out.append(f"  - {json.dumps(b[k], ensure_ascii=False)}")
        for c in r.get("contacts", []):
            line = []
            for side in ("before", "after"):
                pen = c[side]["penetration"]
                line.append(f"{side} ({c[side]['body']}, {'held' if c[side]['ok'] else 'failed'}): "
                            + ", ".join(f"{k} peak {v['peak_mm']} / p95 {v['p95_mm']} mm" for k, v in pen.items()))
            out.append(f"- replay {c['task']}: " + "; ".join(line))
        out.append("")
    return "\n".join(out) + "\n"


def main(argv: list[str] | None = None) -> int:
    from sea.evolve.core import Body
    from sea.evolve.loop import make_world

    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--films", type=Path, default=Path("results/evolve/films"))
    args = ap.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)
    rows, seen = [], {}
    for run in args.runs:
        if not (run / "summary.json").exists():
            continue
        world_name, fs, _ = frames(run)
        world = make_world(world_name)
        bodies = []
        for f in fs:
            body = Body.model_validate(f["body"])
            row: dict = {"id": f["id"], "key": body.key()}
            if body.key() in seen:
                row |= {k: v for k, v in seen[body.key()].items() if k not in ("id",)}
            elif world_name.startswith("gripper"):
                row["geometry"] = gripper_geometry(body)
            elif world_name == "mobile":
                row["facts"] = mobile_facts(body)
            elif world_name == "phonology":
                row["facts"] = phonology_facts(body)
            seen.setdefault(body.key(), row)
            bodies.append(row)
        rec = {"run": run.name, "world": world_name, "bodies": bodies}
        if world_name.startswith("gripper"):
            rec["contacts"] = gripper_contacts(run, args.films, world)
        (args.out / f"{run.name}.json").write_text(json.dumps(rec, indent=1, ensure_ascii=False) + "\n")
        rows.append(rec)
        print(f"{run.name}: {len(bodies)} bodies, {len(rec.get('contacts', []))} replays", file=sys.stderr)
    (args.out / "summary.md").write_text(_md(rows))
    return 0


if __name__ == "__main__":
    sys.exit(main())
