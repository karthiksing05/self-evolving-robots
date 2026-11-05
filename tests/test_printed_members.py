"""Printed members that carry a pin's force are judged at that force, and a roller's arm bends (D-103)."""

from __future__ import annotations

import json
import math
from pathlib import Path

import mujoco
import numpy as np
import pytest

from sea.compose.library import Library
from sea.contracts import PanelSpec
from sea.design.build import build
from sea.evolve.core import Body
from sea.evolve.loop import make_world
from sea.mech.build import printed_members
from sea.mech.graph import Graph
from sea.sim import jaws as J
from sea.sim.build_model import build_model
from sea.sim.env import PanelRunner, SuiteConfig, candidate_programs
from sea.worlds import gripper as G

DATA = Path(__file__).parent / "mech" / "data"


def member(kind="strut", plates=2, root=0.0, pin=(0.09, 0.0, 0.0)):
    w, t = 0.009, 0.004
    return J.Member("m", kind, 0, "b", pin, plates, root, w * t, t * w**3 / 12, w / 2, min(t * w**3, w * t**3) / 12)


def far_pivot() -> dict:
    return json.loads((DATA / "far_pivot_L90_b40.json").read_text())


def built(body: dict):
    return build_model(build(G.to_design(Body.model_validate(body)), G._limits()).design)


def gripped(body: dict, close_s: float = 5.0, watch=None):
    """``body`` holding the dice, closed in ``close_s`` with the law read and not enforced: its sim, once it has lifted."""
    task = next(t for t in make_world("gripper").tasks() if t.id == "reg_dice")
    b = build(G.to_design(Body.model_validate(body)), G._limits())
    panel = PanelSpec.model_validate(task.data["panel"])
    cfg = SuiteConfig(design=b.design, tools=b.tools, library=Library.seed(), limits=G._limits(), seed=0)
    runner = PanelRunner(panel, cfg, 0)
    control = panel.controls[0]
    sim = runner.sim
    sim.members.broken = lambda: ""
    if watch is not None:
        step = sim.step

        def stepped(*a, **k):
            step(*a, **k)
            watch(sim)

        sim.step = stepped
    runner.attempt(control, dict(candidate_programs(control, panel, cfg))["pick_up"], {"grasp": {"close_s": close_s}})
    return sim


def test_a_strut_shares_its_load_and_buckles_only_in_compression():
    m = member()
    axis = np.array([1.0, 0.0, 0.0])
    stress, load, at = m.loads(np.array([-100.0, 0.0, 0.0]), axis)
    assert stress == pytest.approx(50.0 / m.area)
    assert load == pytest.approx(50.0)
    assert at == pytest.approx(math.pi**2 * J.PLA_E_PA * m.i_min / 0.09**2)
    assert m.loads(np.array([100.0, 0.0, 0.0]), axis)[1] == 0.0


def test_a_crank_bends_at_its_hub_by_the_force_across_it():
    m = member("crank", plates=1, root=0.006, pin=(0.02, 0.0, 0.0))
    axis = np.array([1.0, 0.0, 0.0])
    stress, load, at = m.loads(np.array([40.0, 80.0, 0.0]), axis)
    assert stress == pytest.approx(80.0 * 0.014 * m.c / m.i_bend)
    assert load == 0.0 and at == math.inf


def test_the_far_pivot_members_are_found_as_they_are_by_hand():
    design = build(G.to_design(Body.model_validate(far_pivot())), G._limits()).design
    graph = Graph.model_validate_json(design.fingers.mechanism)
    found = {m.name: m for m in printed_members(graph, build_model(design).model)}
    assert found["jaw_horn_arm"].kind == "crank" and found["jaw_horn_arm"].root == pytest.approx(0.008)
    assert found["jaw_rocker_arm"].kind == "crank" and found["jaw_rocker_arm"].root == pytest.approx(0.006)
    assert found["jaw_drive_rod_web"].kind == "strut"
    assert found["jaw_follower_horn"].plates == found["jaw_follower_rear"].plates == 2
    assert found["jaw_follower_horn"].length == pytest.approx(0.09, abs=1e-3)


def test_a_load_counts_only_once_it_is_held():
    model = mujoco.MjModel.from_xml_string(
        "<mujoco><worldbody><body name='b'><freejoint/><geom size='0.01'/></body>"
        "<body name='c'><freejoint/><geom size='0.01' pos='0.1 0 0'/></body></worldbody>"
        "<equality><connect body1='b' body2='c' anchor='0.05 0 0'/></equality></mujoco>")
    p = J.PrintedMembers(model, [member(plates=1)])
    assert p.broken() == ""
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    assert data.nefc == 3
    # A spike shorter than the hold leaves no reading: a load is the least over the last HOLD_STEPS steps.
    for f in [0.0] * (J.HOLD_STEPS - 1) + [1e4]:
        data.efc_force[:3] = (f, 0.0, 0.0)
        p.read(data)
    assert p.stress[0] < J.PLA_ALLOW_PA / 1e6 and p.broken() == ""
    for _ in range(J.HOLD_STEPS):
        data.efc_force[:3] = (1e4, 0.0, 0.0)
        p.read(data)
    assert p.broken().startswith("unphysical: the printed m carried")
    p.reset()
    assert p.broken() == ""


def test_a_backed_rack_bends_as_the_roller_arm_does():
    body = json.loads((DATA / "long_slide_r52_farplane_roller.json").read_text())
    sim = built(body)
    data = sim.model.numeric("jaw_rack").data
    assert len(data) == 10
    back_l, back_ei, back_c = data[-3:]
    assert back_l == pytest.approx(0.0262, abs=1e-4) and back_ei > 0 and back_c > 0


def test_the_steady_grip_loads_the_rod_as_statics_says():
    """Held still, the servo's stall torque over the rod's lever about the hinge's axis is the rod's force: the reading is true."""
    torques = []

    def watch(sim):
        k = next(i for i, m in enumerate(sim.members.members) if m.name == "jaw_drive_rod_web")
        rod, d = sim.members.members[k], sim.data
        rows = (d.efc_type[: d.nefc] == mujoco.mjtConstraint.mjCNSTR_EQUALITY) & (d.efc_id[: d.nefc] == sim.members.eq[k])
        v = sim.model.joint("gripper").dofadr[0]
        if rows.sum() == 3 and abs(d.qvel[v]) < 1e-3 and abs(d.qfrc_actuator[v]) > 0.98 * G.GRIP_TORQUE_NM:
            body = sim.model.body(rod.body).id
            along = d.xmat[body].reshape(3, 3) @ (np.array(rod.pin) / rod.length)
            j = sim.model.joint("gripper").id
            lever = abs(float(np.cross(d.xpos[body] - d.xanchor[j], along) @ d.xaxis[j]))
            torques.append(float(np.linalg.norm(d.efc_force[: d.nefc][rows])) * lever)

    sim = gripped(far_pivot(), watch=watch)
    assert len(torques) > 100
    assert float(np.median(torques)) == pytest.approx(G.GRIP_TORQUE_NM, rel=0.1)
    # On closing the rod pushes, so it can buckle.
    assert sim.members.buckled[2] == "jaw_drive_rod_web" and sim.members.buckled[0] > 50


def test_closing_slower_takes_the_impact_off_the_arms():
    """The jaw's arrival at the servo's full speed is a real load (D-103): slowing the close is how it is avoided."""
    assert gripped(far_pivot(), 0.2).members.stress[0] > 1.5 * gripped(far_pivot(), 5.0).members.stress[0]


def test_a_rod_of_half_the_depth_buckles():
    body = far_pivot()
    graph = json.loads(body["parts"]["jaw"]["mechanism"])
    web = next(p for b in graph["bodies"] for p in b["parts"] if p.get("name") == "jaw_drive_rod_web")
    w = min(web["half"])
    web["half"] = [w / 2 if h == w else h for h in web["half"]]
    body["parts"]["jaw"]["mechanism"] = json.dumps(graph)
    load, at, name = gripped(body).members.buckled
    assert name == "jaw_drive_rod_web" and load > at / J.PRINT_FACTOR


def test_a_strut_may_carry_its_euler_load_over_the_print_factor():
    """A printed strut's pins are not concentric nor its section true, so it is held short of the ideal column (D-104)."""
    p = J.PrintedMembers.__new__(J.PrintedMembers)
    p.members = []
    p.reset()
    p.buckled = (0.9 * 150.0, 150.0, "rod")
    assert "may carry 100" in p.broken()
    p.buckled = (0.6 * 150.0, 150.0, "rod")
    assert p.broken() == ""
