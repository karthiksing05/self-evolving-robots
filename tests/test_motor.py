"""Motor primitives on the real sim (spec 8.1; test list 12.2)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import yaml

from sea.compose import dsl
from sea.compose.library import Library
from sea.contracts import Design, HardLimits, Program, StepTrace
from sea.control import executor as ex
from sea.control import motor
from sea.sim.build_model import build_model
from sea.sim.controls import Faults
from sea.sim.panels import reference_panel

ROOT = Path(__file__).resolve().parents[1]
LIMITS = HardLimits.model_validate(yaml.safe_load((ROOT / "configs" / "limits.yaml").read_text()))
LIB = Library.seed()
PANEL = reference_panel()
CID = {c.cls: c.id for c in PANEL.controls}


def _cfg(cap: float, cls: str = "push_button") -> ex.EpisodeConfig:
    return ex.EpisodeConfig(limits=LIMITS, force_cap=cap, class_cap=PANEL.rules.force_cap_by_class[cls],
                            banned=frozenset({"pinch"}), library=LIB.prims)


class Recorder:
    """Per-tick commanded and measured tool-tip positions."""

    def __init__(self, monkeypatch):
        self.cmd: list[np.ndarray] = []
        self.act: list[np.ndarray] = []
        self.force: list[float] = []
        orig = ex.Episode.tick

        def tick(ep, q):
            self.cmd.append(ep.tip_cmd.copy())
            orig(ep, q)
            self.act.append(ep.tip_actual()[:3, 3].copy())
            self.force.append(ep.last_force)

        monkeypatch.setattr(ex.Episode, "tick", tick)

    def speeds(self, xs: list[np.ndarray]) -> np.ndarray:
        a = np.asarray(xs)
        return np.linalg.norm(np.diff(a, axis=0), axis=1) / ex.TICK_DT


def _run(sim, text_or_name: str, cid: str, cfg: ex.EpisodeConfig) -> StepTrace:
    ex.reset_episode(sim)
    d = LIB.get(text_or_name)
    root = d.body.root if d is not None else dsl.parse(text_or_name)
    prog = Program(id="t", target_cls=sim.controls[cid].spec.cls, root=root, source="s2")
    return ex.run_program(sim, prog, cid, cfg, panel_id=PANEL.id)


@pytest.fixture(scope="module")
def sim():
    return build_model(Design(id="stock"), PANEL)


@pytest.mark.parametrize("scale, cap", [(5.0, 2.0), (10.0, 2.0), (10.0, 4.0), (20.0, 4.0)])
def test_guarded_contact_stops_within_half_a_newton_of_the_cap(scale, cap, monkeypatch):
    """A button too stiff for the cap (the planted 'stiff button' fault and stiffer, up to ~12 kN/m)."""
    cid = CID["push_button"]
    stiff = build_model(Design(id="stock"), PANEL, faults={cid: Faults(stiffness_scale=scale)})
    rec = Recorder(monkeypatch)
    tr = _run(stiff, "press_button", cid, _cfg(cap))
    assert tr.success_time_s is None and not tr.program_ok
    pushing = [f for f, p in zip(rec.force, tr.push_phase, strict=True) if p]
    assert pushing, "no push phase recorded"
    assert max(pushing) <= cap + 0.5
    assert max(pushing) >= cap - 0.5  # it did lean on the control up to the cap
    assert tr.failed_primitive == "push"


@pytest.mark.parametrize("name, cls", [("press_button", "push_button"), ("slide_slider", "slider"),
                                       ("flip_toggle", "toggle")])
def test_tip_speed_limits(sim, name, cls, monkeypatch):
    rec = Recorder(monkeypatch)
    _run(sim, name, CID[cls], _cfg(9.0, cls))
    vmax = LIMITS.max_tip_speed_mps
    assert rec.speeds(rec.cmd).max() <= vmax * (1 + 1e-6)
    assert rec.speeds(rec.act).max() < 1.2 * vmax


def test_approach_offset_is_30_mm(sim):
    cid = CID["push_button"]
    ex.reset_episode(sim)
    tr = StepTrace(panel_id=PANEL.id, control_id=cid, program_id="a", attempt=0)
    ep = ex.Episode(sim, cid, _cfg(9.0), tr)
    ok, reason = ex.drive(ep, motor.approach(ep, cid))
    assert ok and reason == ""
    p, d, _ = motor.engage_pose(ep, cid)
    gap = p - ep.tip_cmd
    assert float(gap @ d) == pytest.approx(motor.APPROACH_OFFSET, abs=1e-9)
    assert np.linalg.norm(gap - d * (gap @ d)) < 1e-9
    assert motor.APPROACH_OFFSET == 0.03
    for _ in ep.hold(0.4):
        ep.tick(ep.q_cmd)
    assert np.linalg.norm(ep.tip_actual()[:3, 3] - ep.tip_cmd) < 2e-3


def test_half_speed_near_contact(sim, monkeypatch):
    cid = CID["push_button"]
    rec = Recorder(monkeypatch)
    ex.reset_episode(sim)
    tr = StepTrace(panel_id=PANEL.id, control_id=cid, program_id="a", attempt=0)
    ep = ex.Episode(sim, cid, _cfg(9.0), tr)
    p, d, _ = motor.engage_pose(ep, cid)
    ok, _ = ex.drive(ep, exec_seq(ep, cid))
    cmd = np.asarray(rec.cmd)
    v = rec.speeds(rec.cmd)
    to_go = (p - cmd[1:]) @ d - motor.STOCK_TIP_EXT  # tool end to the control surface
    near = to_go < motor.SLOW_ZONE - 1e-6
    assert near.any() and (~near).any()
    assert v[near].max() <= 0.5 * LIMITS.max_tip_speed_mps + 1e-9
    assert v[~near].max() > 0.5 * LIMITS.max_tip_speed_mps  # it is fast elsewhere


def exec_seq(ep, cid):
    yield from motor.approach(ep, cid)
    return (yield from motor.contact(ep, cid))


def test_workspace_abort_and_unknown_primitive(sim):
    cid = CID["push_button"]
    tr = _run(sim, "retreat($target, 0.3)", cid, _cfg(9.0))
    assert tr.note == "workspace" and tr.failed_primitive == "retreat"
    assert _run(sim, "fly($target)", cid, _cfg(9.0)).program_ok is False


# ---------------------------------------------------------------- tool primitives on the reference panel
@pytest.fixture(scope="module")
def rack():
    from sea.design.build import build
    from sea.design.space import make_tool_spec

    tools = [make_tool_spec("hook", depth=14, throat=6, length=45),
             make_tool_spec("knob_cup", inner_diameter=14, depth=10, ribs=4),
             make_tool_spec("pad", w=16, h=16, thickness=5, friction=0.9),
             make_tool_spec("slot_inserter", blade_thickness=1.4, width=24, length=40)]
    built = build(Design(id="rack", tool_rack=tools), LIMITS)
    tab = PANEL.model_copy(update={"controls": [c.model_copy(update={"handle": "tab"}) if c.cls == "door" else c
                                                for c in PANEL.controls]})
    return built, tab


def _rack_run(rack, cls: str, text: str, slot: int | None, cap: float | None = None) -> tuple[StepTrace, object]:
    built, panel = rack
    sim = build_model(built.design, panel, built.tools)
    cid = next(c.id for c in panel.controls if c.cls == cls)
    d = LIB.get(text)
    root = d.body.root if d is not None else dsl.parse(text)
    if slot is not None:
        root = dsl.with_tool(slot, root)
    ex.reset_episode(sim)
    cap = cap if cap is not None else 0.9 * panel.rules.force_cap_by_class[cls]
    tr = ex.run_program(sim, Program(id="t", target_cls=cls, root=root, source="s2"), cid, _cfg(cap, cls),
                        panel_id=panel.id)
    return tr, sim.controls[cid]


def test_pad_pushes_the_lever(rack):
    tr, h = _rack_run(rack, "lever", "pull_lever", 2)
    assert tr.success_time_s is not None and "pad" in tr.contact_types and tr.tool_swaps == 1


def test_hook_opens_a_tab_door_part_way(rack):
    tr, h = _rack_run(rack, "door", "open_door_pull", 0)
    assert tr.contact_types == ["hook"] and max(tr.control_q) > 0.5 and not h.dyn.latched


def test_hook_unlatches_the_tab_drawer(rack):
    tr, h = _rack_run(rack, "drawer", "open_door_pull", 0)
    assert "hook" in tr.contact_types and max(tr.control_q) > 0.02


def test_stock_pinches_a_tab(rack):
    tr, _ = _rack_run(rack, "door", "open_door_pull", None)
    assert "pinch" in tr.contact_types


def test_cup_and_pinch_turn_attempts(rack):
    tr, _ = _rack_run(rack, "knob", "turn_knob", 1)
    assert "cup" in tr.contact_types and len(tr.t) > 100
    tr2, _ = _rack_run(rack, "knob", "turn_knob", None)
    assert set(tr2.contact_types) & {"pinch", "edge"}


def test_blade_inserts_into_the_slot(rack):
    tr, _ = _rack_run(rack, "slot", "Seq(approach($target), contact($target), insert($target), release(), retreat())", 3)
    assert "edge" in tr.contact_types and max(tr.control_q) > 0.0
