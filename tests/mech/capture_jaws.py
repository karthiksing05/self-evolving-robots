"""Record what the hand-written jaw builders compile (D-078): the golden file the mechanism graph must
reproduce.  Run once, before the builders were replaced:  python tests/mech/capture_jaws.py"""

from __future__ import annotations

import json
from pathlib import Path

import mujoco
import numpy as np

from sea.evolve.core import Part
from sea.sim.build_model import build_model
from sea.worlds.gripper import GripperWorld, to_design

GRID = {"revolute": [{}],
        "parallelogram": [{"link_mm": v} for v in (50.0, 70.0, 89.0, 90.0)],
        "prismatic": [{"pinion_teeth": v} for v in (17.0, 30.0, 40.0, 60.0)],
        "long_slide": [{"sector_radius_mm": v} for v in (30.0, 40.0, 46.0, 50.0)]}
OUT = Path(__file__).with_name("golden_jaws.json")


def snapshot(m: mujoco.MjModel, linked) -> dict:
    name = lambda t, i: mujoco.mj_id2name(m, t, i) or f"#{i}"  # noqa: E731
    O = mujoco.mjtObj  # noqa: E741
    return {
        "bodies": {name(O.mjOBJ_BODY, b): {"parent": name(O.mjOBJ_BODY, int(m.body_parentid[b])),
                                           "pos": m.body_pos[b].tolist(), "quat": m.body_quat[b].tolist(),
                                           "mass": float(m.body_mass[b]), "ipos": m.body_ipos[b].tolist(),
                                           "inertia": m.body_inertia[b].tolist()} for b in range(m.nbody)},
        "joints": {name(O.mjOBJ_JOINT, j): {"body": name(O.mjOBJ_BODY, int(m.jnt_bodyid[j])), "type": int(m.jnt_type[j]),
                                            "axis": m.jnt_axis[j].tolist(), "pos": m.jnt_pos[j].tolist(),
                                            "range": m.jnt_range[j].tolist(), "armature": float(m.dof_armature[m.jnt_dofadr[j]]),
                                            "damping": float(m.dof_damping[m.jnt_dofadr[j]])} for j in range(m.njnt)},
        "geoms": {name(O.mjOBJ_GEOM, g): {"body": name(O.mjOBJ_BODY, int(m.geom_bodyid[g])), "type": int(m.geom_type[g]),
                                          "size": m.geom_size[g].tolist(), "pos": m.geom_pos[g].tolist(),
                                          "quat": m.geom_quat[g].tolist(), "group": int(m.geom_group[g]),
                                          "contype": int(m.geom_contype[g]), "conaffinity": int(m.geom_conaffinity[g]),
                                          "rgba": m.geom_rgba[g].tolist(),
                                          "mesh_verts": (int(m.mesh_vertnum[m.geom_dataid[g]]) if m.geom_type[g] == 7 else 0),
                                          "mesh_sum": (float(np.abs(m.mesh_vert[m.mesh_vertadr[m.geom_dataid[g]]:m.mesh_vertadr[m.geom_dataid[g]] + m.mesh_vertnum[m.geom_dataid[g]]]).sum()) if m.geom_type[g] == 7 else 0.0)}
                  for g in range(m.ngeom)},
        "equality": [{"type": int(m.eq_type[e]), "obj1": int(m.eq_obj1id[e]), "obj2": int(m.eq_obj2id[e]),
                      "data": m.eq_data[e].tolist(), "solref": m.eq_solref[e].tolist(), "solimp": m.eq_solimp[e].tolist()}
                     for e in range(m.neq)],
        "excludes": sorted([sorted([name(O.mjOBJ_BODY, int(p >> 16)), name(O.mjOBJ_BODY, int(p & 0xFFFF))])
                            for p in m.exclude_signature.tolist()]),
        "numeric": {name(O.mjOBJ_NUMERIC, i): m.numeric_data[m.numeric_adr[i]:m.numeric_adr[i] + m.numeric_size[i]].tolist()
                    for i in range(m.nnumeric)},
        "linked": [[mujoco.mj_id2name(m, O.mjOBJ_JOINT, int(np.nonzero(m.jnt_qposadr == a)[0][0])), r] for a, r in linked],
    }


def capture() -> dict:
    w = GripperWorld()
    out = {}
    for jaw, grid in GRID.items():
        for ps in grid:
            b = w.stock()
            b = b.model_copy(update={"parts": {**b.parts, "jaw": Part(type=jaw, params=ps)}})
            sim = build_model(to_design(b))
            out[f"{jaw} {json.dumps(ps, sort_keys=True)}"] = snapshot(sim.model, sim.linked)
    return out


if __name__ == "__main__":
    OUT.write_text(json.dumps(capture(), indent=0))
    print(OUT, OUT.stat().st_size)
