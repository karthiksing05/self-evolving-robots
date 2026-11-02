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

import numpy as np
from render_bodies import frames

HAND = ("gripper", "moving_jaw_so101_v1", "jaw_coupler", "jaw_follower", "jaw_carriage")
OBSTACLES = ("wrist",)  # what the jaw mechanism must clear beyond the gripper (a pinion can reach it, D-058)
ROBOT = {"base", "shoulder", "upper_arm", "lower_arm", "wrist", *HAND}
SAMPLES = 9  # jaw positions, ends included
ROLLS = 13  # wrist roll positions, ends included
# Datasheet ratings per H-bridge channel (A, continuous): Toshiba TB6612FNG 1.2 A (3.2 A peak); ST L298 2 A.
CHANNEL_A = {"driver_tb6612": 1.2, "driver_l298n": 2.0}
CELL_V = {"nimh_pack": 1.2, "liion_pack": 3.7}  # nominal volts per cell: NiMH AA, 18650 Li-ion


# ---------------------------------------------------------------- gripper geometry
def _hand(model):
    import mujoco
    import trimesh

    from sea.mechanism.geometry import Solid

    out = []
    for i in range(model.ngeom):
        body = model.body(model.geom_bodyid[i]).name
        if model.geom_group[i] != 2 or body not in HAND + OBSTACLES:
            continue
        if model.geom_type[i] == mujoco.mjtGeom.mjGEOM_MESH:
            k = model.geom_dataid[i]
            v0, nv, f0, nf = model.mesh_vertadr[k], model.mesh_vertnum[k], model.mesh_faceadr[k], model.mesh_facenum[k]
            mesh = trimesh.Trimesh(model.mesh_vert[v0:v0 + nv].copy(), model.mesh_face[f0:f0 + nf].copy(), process=False)
            name = model.mesh(k).name
        elif model.geom_type[i] == mujoco.mjtGeom.mjGEOM_BOX:
            mesh = trimesh.creation.box(extents=2 * model.geom_size[i])
            name = model.geom(i).name
        elif model.geom_type[i] == mujoco.mjtGeom.mjGEOM_CYLINDER:
            mesh = trimesh.creation.cylinder(radius=model.geom_size[i][0], height=2 * model.geom_size[i][1], sections=48)
            name = model.geom(i).name
        else:
            continue
        rot = np.zeros(9)
        mujoco.mju_quat2Mat(rot, model.geom_quat[i])
        pose = np.eye(4)
        pose[:3, :3], pose[:3, 3] = rot.reshape(3, 3), model.geom_pos[i]
        mesh.apply_transform(pose)
        mesh.apply_scale(1000.0)
        out.append(Solid(name, body, mesh))
    return out


def gripper_geometry(body) -> dict:
    import mujoco

    from sea.contracts import ARM_JOINTS
    from sea.control.executor import HOME_Q
    from sea.mechanism.geometry import clearance, extent, from_axis, overlap
    from sea.sim.build_model import build_model
    from sea.worlds.gripper import to_design

    sim = build_model(to_design(body))
    m, d = sim.model, sim.data
    solids = _hand(m)
    jid = m.joint("gripper").id
    lo, hi = m.jnt_range[jid]
    bodies = {s.body for s in solids}
    clashes: dict[str, dict] = {}
    names = {m.body(i).name for i in range(m.nbody)}
    support: dict[str, list[float]] = {b: [] for b in HAND[1:] if b in names}

    def pose(b: str) -> np.ndarray:  # in the gripper's frame, mm
        rot, at = d.xmat[m.body("gripper").id].reshape(3, 3), d.xpos[m.body("gripper").id]
        t = np.eye(4)
        t[:3, :3], t[:3, 3] = rot.T @ d.xmat[m.body(b).id].reshape(3, 3), 1000 * rot.T @ (d.xpos[m.body(b).id] - at)
        return t

    roll = ARM_JOINTS.index("wrist_roll")
    wrists = []  # the wrist seen from the gripper over the wrist's roll range: a jaw part must clear it at every roll
    for r in np.linspace(*m.jnt_range[m.joint("wrist_roll").id], ROLLS):
        qa = HOME_Q.copy()
        qa[roll] = r
        sim.set_arm_q(qa)
        wrists.append((round(float(r), 2), pose("wrist")))
    for q in np.linspace(lo, hi, SAMPLES):
        qa = HOME_Q.copy()
        qa[-1] = q
        sim.set_arm_q(qa)  # the jaw mechanism's linked joints follow (Sim.linked)
        poses = {b: pose(b) for b in (bodies | set(support)) - set(OBSTACLES)}
        rot = d.xmat[m.body("gripper").id].reshape(3, 3)
        axis_o = rot.T @ (1000 * (d.xanchor[jid] - d.xpos[m.body("gripper").id]))
        axis_u = rot.T @ d.xaxis[jid]
        for i, a in enumerate(solids):
            for b in solids[i + 1:]:
                if a.body == b.body or {a.body, b.body} == {"wrist", "gripper"}:  # the wrist carries the gripper
                    continue
                if a.body in OBSTACLES or b.body in OBSTACLES:
                    tries = [(rl, {**poses, "wrist": w}) for rl, w in wrists]
                else:
                    tries = [(None, poses)]
                for rl, ps in tries:
                    pts = overlap(a, ps[a.body], b, ps[b.body])
                    if not len(pts):
                        continue
                    r = from_axis(pts, axis_o, axis_u)
                    c = clashes.setdefault(f"{a.name} / {b.name}", {"bodies": [a.body, b.body], "at_q": [], "crossings": [],
                                                                    "region_mm": [0.0, 0.0, 0.0], "from_axis_mm": [1e9, 0.0]})
                    if rl is not None:
                        c.setdefault("at_roll", []).append(rl)
                    c["at_q"].append(round(float(q), 3))
                    c["crossings"].append(int(len(pts)))
                    c["region_mm"] = [max(x, y) for x, y in zip(c["region_mm"], extent(pts), strict=True)]
                    c["from_axis_mm"] = [round(min(c["from_axis_mm"][0], float(r.min())), 1),
                                         round(max(c["from_axis_mm"][1], float(r.max())), 1)]
        for child in support:
            parent = m.body(m.body_parentid[m.body(child).id]).name
            mine = [s for s in solids if s.body == child]
            theirs = [s for s in solids if s.body == parent]
            gap = min((clearance(a, poses[child], b, poses[parent]) for a in mine for b in theirs), default=float("nan"))
            support[child].append(round(float(gap), 1))
    carried = {}
    for child, gaps in support.items():
        parent = m.body(m.body_parentid[m.body(child).id]).name
        no_material = not any(s.body == child for s in solids)
        own = m.jnt_type[m.body_jntadr[m.body(child).id]]
        carried[child] = {"parent": parent, "joint": "slide" if own == mujoco.mjtJoint.mjJNT_SLIDE else "hinge",
                          "material": not no_material,
                          "worst_gap_mm": None if no_material else max(gaps), "gap_by_q_mm": None if no_material else gaps}
    mass = {b: round(1000 * float(m.body_mass[m.body(b).id]), 1) for b in [*support, "gripper"]}
    return {"jaw_range_servo_rad": [round(float(lo), 3), round(float(hi), 3)], "clashes": clashes, "carried": carried,
            "sim_mass_g": mass}


# ---------------------------------------------------------------- gripper replays
def _labeller(model):
    def label(b: int) -> str:
        name = model.body(b).name
        return "robot" if name in ROBOT else ("object" if name.startswith(("ctl_", "obj_")) else "table")
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
