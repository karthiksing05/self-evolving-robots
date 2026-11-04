"""The graph language can say what the ME 2110 catalogue said (D-094): same model, and the same match replayed."""

from __future__ import annotations

import mujoco
import numpy as np
import pytest

from sea.evolve.core import Edit
from sea.sim import me2110_model as MM
from sea.sim.kit_match import KitMatch, compile_robot
from sea.worlds.me2110 import _robot
from tests.framework.me2110_graphs import graph_of, timeline
from tests.test_world_me2110 import _body

V = MM.Variant(barrier=7.0, rpm=3.0, mu=0.5, phase=0.4, others=(6.0, 8.0, 8.75))
ROBOTS = {
    "stock": _body(),
    "arms": _body(Edit(slot="mermaid", type="trap_arms", params={"arm_len": 10.0, "arm_z": 5.0})),
    "cannon": _body(Edit(slot="cannon", type="gate_chute", params={"exit_z": 9.5, "reach": 9.8}),
                    Edit(slot="frame", params={"width": 8.0, "deck_z": 8.0})),
    "mast": _body(Edit(slot="flag", type="band_mast", params={"stages": 4, "stage_len": 9.0, "lean_deg": 40.0, "bands": 2})),
}


def _arrays(m: mujoco.MjModel, root: int) -> dict[str, np.ndarray]:
    bodies = [i for i in range(m.nbody) if m.body_rootid[i] == root]
    geoms = [g for i in bodies for g in range(m.body_geomadr[i], m.body_geomadr[i] + m.body_geomnum[i])
             if m.geom(g).name != "kit_parts"]
    joints = [j for i in bodies for j in range(m.body_jntadr[i], m.body_jntadr[i] + m.body_jntnum[i])]
    return {"parent": np.array([bodies.index(m.body_parentid[i]) if i != root else -1 for i in bodies], float),
            "pos": m.body_pos[bodies[1:]], "quat": m.body_quat[bodies[1:]], "mass": m.body_mass[bodies],
            "inertia": m.body_inertia[bodies], "geom_pos": m.geom_pos[geoms], "geom_quat": m.geom_quat[geoms],
            "geom_size": m.geom_size[geoms], "geom_type": m.geom_type[geoms].astype(float),
            "mu": m.geom_friction[geoms][:, 0], "jnt_type": m.jnt_type[joints].astype(float),
            "jnt_axis": m.jnt_axis[joints], "jnt_pos": m.jnt_pos[joints], "jnt_range": m.jnt_range[joints],
            "stiffness": m.jnt_stiffness[joints]}


@pytest.mark.parametrize("which", sorted(ROBOTS))
def test_the_graph_compiles_to_the_same_bodies_joints_solids_and_masses(which):
    rb = _robot(ROBOTS[which])
    old = mujoco.MjModel.from_xml_string(MM.scene(rb, V, 0.0)[0])
    new = compile_robot(graph_of(rb, V), V, add_kit_mass=False, x0=-(rb.L / 2 + 0.25)).model
    a, b = _arrays(old, old.body("robot").id), _arrays(new, new.body("robot").id)
    for k in a:
        assert a[k].shape == b[k].shape, k
        assert np.allclose(a[k], b[k], atol=1e-6, rtol=1e-5), k
    assert old.nq == new.nq
    assert np.allclose(old.body_pos[old.body("robot").id], new.body_pos[new.body("robot").id], atol=1e-6)


def _replay(body, ctl, horizon=40.0):
    rb = _robot(body)
    r = MM.Match(rb, V, ctl, seed=0, horizon=horizon).run()
    g = graph_of(rb, V)
    prog = {**timeline(rb, g, r.events, ctl), "y0": ctl.get("y0", 0.0)}
    return r, KitMatch(g, V, prog, seed=0, horizon=horizon, add_kit_mass=False, x0=-(rb.L / 2 + 0.25), jitter=False).run()


@pytest.mark.parametrize("which,ctl", [("stock", {"park_x": 14.5, "duty": 0.6}),
                                       ("arms", {"park_x": 10.0, "duty": 0.6, "t_mermaid": 4.0})])
def test_a_match_with_no_loose_items_replays_as_a_program_within_a_point(which, ctl):
    old, new = _replay(ROBOTS[which], ctl)
    assert all(abs(old.points[k] - new.points[k]) <= 1.0 for k in old.points), (old.points, new.points)
    assert not old.laws and not new.laws


def test_a_chaotic_match_replays_within_one_ball_on_average():
    """Balls on a chute are chaotic in the old model too: 18 to 36 points for a 0.1 in change (D-094), so one match
    says nothing and the means over nearby starts are what must agree."""
    runs = [_replay(ROBOTS["cannon"], {"y0": y, "park_x": x, "duty": d, "gate_delay": 0.0, "gate_open": 0.8}, horizon=26.0)
            for y, x, d in ((-4.0, 11.5, 0.5), (-3.9, 11.5, 0.5), (-4.1, 11.5, 0.5), (-4.0, 11.5, 0.49), (-4.0, 11.5, 0.51),
                            (-4.0, 11.4, 0.5))]
    old, new = (float(np.mean([r[k].points["cannon"] for r in runs])) for k in (0, 1))
    assert abs(old - new) <= 9.0, (old, new)
