"""System 1: fixed retry rule and CMA-ES local search (spec 8.1, 3.3)."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import yaml

from sea.compose import dsl
from sea.compose.library import Library
from sea.contracts import ControlResult, Design, HardLimits, Program, StepTrace
from sea.control import local_search as LS
from sea.control.executor import EpisodeConfig

ROOT = Path(__file__).resolve().parents[1]
LIMITS = HardLimits.model_validate(yaml.safe_load((ROOT / "configs" / "limits.yaml").read_text()))
LIB = Library.seed()
PRESS = LIB.get("press_button").body
TRAVEL = 0.002


def _cfg(force_cap: float = 9.0, **kw) -> EpisodeConfig:
    return EpisodeConfig(limits=LIMITS, force_cap=force_cap, class_cap=10.0, banned=frozenset({"pinch"}),
                         library=LIB.prims, **kw)


class FakeSim:
    """Just enough of ``Sim`` for System 1: one push button, centre at the origin."""

    def __init__(self, contact=(0.003, 0.0, 0.004)):
        t = np.eye(4)
        self.controls = {"c0": SimpleNamespace(
            spec=SimpleNamespace(travel=TRAVEL),
            geo=SimpleNamespace(lever_arm=0.0, rotary=False, contact_body=np.array(contact)),
            T_world_ctl=t)}

    def control_point_world(self, cid, p):
        return self.controls[cid].T_world_ctl[:3, :3] @ p + self.controls[cid].T_world_ctl[:3, 3]


def _judge(tr: StepTrace) -> ControlResult:
    verdict = tr.note or ("pass" if tr.success_time_s is not None else "stuck")
    return ControlResult(panel_id="p", control_id=tr.control_id, cls="push_button", verdict=verdict,
                         peak_force=1.0, travel_reached=0.0, detents=0, time_s=1.0)


def _install(monkeypatch, outcome):
    """outcome(cfg, attempt) -> (progress fraction, note); a trace reaches pass when progress >= 1."""
    calls = {"reset": 0, "runs": []}

    def fake_run(sim, program, cid, cfg, *, panel_id="", attempt=0, finish=True):
        frac, note = outcome(cfg, attempt)
        calls["runs"].append(cfg)
        tr = StepTrace(panel_id=panel_id, control_id=cid, program_id=program.id, attempt=attempt,
                       control_q=[0.0, frac * TRAVEL], q_rest=0.0, failed_primitive="push", note=note)
        if frac >= 1.0:
            tr.success_time_s = 1.0
        return tr

    def fake_reset(sim):
        calls["reset"] += 1

    monkeypatch.setattr(LS, "run_program", fake_run)
    monkeypatch.setattr(LS, "reset_episode", fake_reset)
    return calls


def _go(sim=None, cfg=None, m=8, delta=0.3, seed=0, reparam=None, program=PRESS):
    return LS.run_system1(sim or FakeSim(), program, "c0", cfg or _cfg(), _judge, m=m, max_param_delta=delta,
                          seed=seed, panel_id="p", reparam=reparam)


def test_first_attempt_pass(monkeypatch):
    calls = _install(monkeypatch, lambda cfg, a: (1.2, ""))
    out = _go()
    assert out.result.verdict == "pass" and out.trials == 0 and out.escalate is None
    assert len(out.history) == 1 and calls["reset"] == 1 and out.result.s1_trials == 0


@pytest.mark.parametrize("verdict", ["unreachable", "rule_violation", "no_applicable_primitive"])
def test_structural_verdicts_return_at_once(monkeypatch, verdict):
    calls = _install(monkeypatch, lambda cfg, a: (0.1, verdict))
    out = _go()
    assert out.result.verdict == verdict and out.escalate == "structural" and len(calls["runs"]) == 1


def test_fixed_retry_offsets_one_mm_toward_the_centre(monkeypatch):
    calls = _install(monkeypatch, lambda cfg, a: (1.0 if a == 1 else 0.3, ""))
    out = _go()
    assert out.result.verdict == "pass" and out.trials == 0 and len(out.history) == 2
    off = calls["runs"][1].perception_offset
    np.testing.assert_allclose(off, [-0.001, 0.0, 0.0], atol=1e-12)  # contact at +x 3 mm, centre at 0
    assert "retry" in out.history[1][0]


def test_retry_structural_escalates(monkeypatch):
    _install(monkeypatch, lambda cfg, a: (0.3, "" if a == 0 else "rule_violation"))
    assert _go().escalate == "structural"


def test_local_search_finds_a_pass_and_reports_its_parameters(monkeypatch):
    def outcome(cfg, a):
        f = cfg.param_overrides.get("push", {}).get("force")
        return (1.0 if f is not None and f > 9.0 * 0.85 else 0.4 if f is None else f / 9.0 * 0.8), ""

    calls = _install(monkeypatch, outcome)
    out = _go(m=8, delta=0.3, seed=3)
    assert out.result.verdict == "pass" and out.escalate is None and out.searched == "push"
    assert 1 <= out.trials <= 8 and out.result.s1_trials == out.trials and out.trace.s1_trials == out.trials
    assert out.params["push"]["force"] > 9.0 * 0.85
    assert calls["reset"] == len(calls["runs"]) == 2 + out.trials
    for cfg in calls["runs"][2:]:  # every trial respects the cap and the declared ranges
        over = cfg.param_overrides["push"]
        assert set(over) == {"depth", "force", "speed", "offset_u", "offset_v"}
        assert 0.5 <= over["force"] <= 9.0 and over["speed"] <= LIMITS.max_tip_speed_mps


def test_local_search_exhausts_m_trials_and_keeps_the_best(monkeypatch):
    def outcome(cfg, a):
        over = cfg.param_overrides.get("push", {})
        return 0.2 + 0.5 * abs(over.get("offset_u", 0.0)) / 0.01, ""

    _install(monkeypatch, outcome)
    out = _go(m=5, seed=1)
    assert out.result.verdict == "stuck" and out.trials == 5 and out.escalate == "s1_exhausted"
    assert len(out.history) == 2 + 5
    best = max(abs(h[0]["push"]["offset_u"]) for h in out.history[2:])
    assert abs(out.params["push"]["offset_u"]) == pytest.approx(best)


def test_same_seed_same_trials(monkeypatch):
    _install(monkeypatch, lambda cfg, a: (0.3, ""))
    a = _go(m=4, seed=7).history
    b = _go(m=4, seed=7).history
    c = _go(m=4, seed=8).history
    assert a == b and a != c


def test_no_free_parameters_or_budget(monkeypatch):
    _install(monkeypatch, lambda cfg, a: (0.3, ""))
    fixed = Program(id="f", target_cls="push_button", source="s2", root=dsl.parse(
        "Seq(approach($target, 0.03, 0.05), contact($target, 0.5, 0.01), "
        "push($target, 0.002, 5.0, 0.01, 0.0, 0.0), retreat($target, 0.05))"))
    out = _go(program=fixed)
    assert out.trials == 0 and out.escalate == "s1_exhausted" and out.searched == "push"
    assert _go(m=0).trials == 0
    assert _go(delta=0.0).escalate == "s1_exhausted"


def test_one_free_parameter_on_a_flat_objective(monkeypatch):
    # pycma raises in tell() when a 1-D search's step size grows past its limit (flat fitness)
    calls = _install(monkeypatch, lambda cfg, a: (0.3, ""))
    one = Program(id="f", target_cls="push_button", source="s2", root=dsl.parse(
        "Seq(approach($target, 0.03, 0.05), contact($target, 0.5, 0.01), "
        "push($target, 0.002, 5.0, 0.01, 0.0), retreat($target, 0.05))"))
    for seed in range(6):
        out = _go(program=one, m=8, delta=1.0, seed=seed)
        assert 2 <= out.trials <= 8 and out.escalate == "s1_exhausted"  # pycma may stop early on flat fitness
    assert {tuple(c.param_overrides["push"]) for c in calls["runs"] if c.param_overrides} == {("offset_v",)}


def test_reparam_ranges_override_declared_ranges(monkeypatch):
    calls = _install(monkeypatch, lambda cfg, a: (0.3, ""))
    _go(m=4, delta=1.0, reparam={"push.force": (2.0, 3.0), "offset_u": (0.0, 0.001), "rotate.angle": (0, 1)})
    for cfg in calls["runs"][2:]:
        over = cfg.param_overrides["push"]
        assert 2.0 <= over["force"] <= 3.0 and 0.0 <= over["offset_u"] <= 0.001


def test_search_space_bounds():
    sim = FakeSim()
    dims = {d.param: d for d in LS.search_space(PRESS, "push", LIB.prims, sim, "c0", _cfg(), 0.3)}
    assert set(dims) == {"depth", "force", "speed", "offset_u", "offset_v"}
    f = dims["force"]
    assert f.x0 == 9.0 and f.hi == 9.0 and f.lo == pytest.approx(9.0 - 0.3 * (15.0 - 0.5))
    d = dims["depth"]
    assert d.x0 == pytest.approx(TRAVEL * 1.25 + 3e-3)
    u = dims["offset_u"]
    assert (u.lo, u.hi) == pytest.approx((-0.3 * 0.02, 0.3 * 0.02))
    assert LS.search_space(PRESS, "release", LIB.prims, sim, "c0", _cfg(), 0.3) == []
    assert LS.search_space(PRESS, "nope", LIB.prims, sim, "c0", _cfg(), 0.3) == []
    over = _cfg(param_overrides={"push": {"force": 4.0}})
    assert {d.param: d for d in LS.search_space(PRESS, "push", LIB.prims, sim, "c0", over, 0.3)}["force"].x0 == 4.0


def test_default_values_follow_the_motor_primitives():
    sim, cfg = FakeSim(), _cfg()
    assert LS.default_value("approach", "offset", sim, "c0", cfg) == 0.03
    assert LS.default_value("contact", "speed", sim, "c0", cfg) == LIMITS.max_tip_speed_mps
    assert LS.default_value("hook_pull", "speed", sim, "c0", cfg) == 0.02
    assert LS.default_value("rotate", "angle", sim, "c0", cfg) == pytest.approx(TRAVEL + 0.08)
    assert LS.default_value("slide", "distance", sim, "c0", cfg) == pytest.approx(TRAVEL * 1.15 + 7e-3)
    assert LS.default_value("insert", "depth", sim, "c0", cfg) == pytest.approx(TRAVEL + 3e-3)
    assert LS.default_value("hook_pull", "distance", sim, "c0", cfg) is None
    sim.controls["c0"].geo.rotary, sim.controls["c0"].geo.lever_arm = True, 0.05
    assert LS.default_value("push", "depth", sim, "c0", cfg) == pytest.approx(0.05 * TRAVEL * 1.25 + 3e-3)


def test_pick_primitive_and_objective():
    tr = StepTrace(panel_id="p", control_id="c0", program_id="x", attempt=0)
    assert LS.pick_primitive(PRESS, tr, LIB.prims) == "push"  # first actuating primitive
    tr.failed_primitive = "contact"
    assert LS.pick_primitive(PRESS, tr, LIB.prims) == "contact"
    tr.failed_primitive = "release"  # no parameters to tune
    assert LS.pick_primitive(PRESS, tr, LIB.prims) == "push"
    nested = Program(id="n", target_cls="push_button", source="s2", root=dsl.parse("Seq(press_button($target), retreat())"))
    assert LS.pick_primitive(nested, StepTrace(panel_id="p", control_id="c", program_id="n", attempt=0),
                             LIB.prims) == "push"
    plain = Program(id="r", target_cls="push_button", source="s2", root=dsl.parse("Seq(release(), retreat())"))
    assert LS.pick_primitive(plain, StepTrace(panel_id="p", control_id="c", program_id="r", attempt=0),
                             LIB.prims) is None
    assert LS.progress(StepTrace(panel_id="p", control_id="c", program_id="r", attempt=0), 0.01) == 0.0
    res = _judge(StepTrace(panel_id="p", control_id="c", program_id="r", attempt=0, note="unreachable"))
    assert LS.objective(res, tr, 0.01) == 5.0


def test_relocalize_draws_noise_only_when_perception_is_noisy():
    sim = FakeSim(contact=(0.0, 0.0, 0.004))  # contact at the centre: no offset direction
    rng = np.random.default_rng(0)
    assert np.allclose(LS.relocalized_offset(sim, "c0", _cfg(), rng), 0.0)
    noisy = _cfg(perception_offset=np.array([0.0005, 0.0, 0.0]))
    off = LS.relocalized_offset(sim, "c0", noisy, rng)
    assert off[2] == 0.0 and 0 < np.linalg.norm(off) < 0.005  # in the panel plane


def test_real_sim_stuck_button_escalates():
    from sea.sim.build_model import build_model
    from sea.sim.controls import Faults
    from sea.sim.panels import reference_panel

    panel = reference_panel()
    c = panel.controls[0]
    sim = build_model(Design(id="stock"), panel, faults={c.id: Faults(stiffness_scale=10.0)})

    def judge(tr):
        return ControlResult(panel_id=panel.id, control_id=c.id, cls=c.cls,
                             verdict="pass" if tr.success_time_s is not None else "stuck", peak_force=0.0,
                             travel_reached=LS.progress(tr, c.travel), detents=0, time_s=tr.sim_seconds)

    out = LS.run_system1(sim, PRESS, c.id, _cfg(force_cap=2.0), judge, m=2, max_param_delta=0.3, seed=0,
                         panel_id=panel.id)
    assert out.result.verdict == "stuck" and out.trials == 2 and out.escalate == "s1_exhausted"
    assert out.searched == "push" and out.trace.failed_primitive == "push"
    assert [h[1] for h in out.history] == ["stuck"] * 4


def test_real_sim_pass_needs_no_search():
    from sea.sim.build_model import build_model
    from sea.sim.panels import reference_panel

    panel = reference_panel()
    c = panel.controls[0]
    sim = build_model(Design(id="stock"), panel)

    def judge(tr):
        return ControlResult(panel_id=panel.id, control_id=c.id, cls=c.cls,
                             verdict="pass" if tr.success_time_s is not None else "stuck", peak_force=0.0,
                             travel_reached=0.0, detents=0, time_s=tr.sim_seconds)

    out = LS.run_system1(sim, PRESS, c.id, _cfg(), judge, m=2, max_param_delta=0.3, seed=0, panel_id=panel.id)
    assert out.result.verdict == "pass" and out.trials == 0


def test_gripper_looks_again_after_a_failure_with_twice_the_attempts(monkeypatch):
    """D-131: a failure counts after longer looks, and a pass on a look ends them."""
    import types

    from sea.sim import env
    from sea.worlds.gripper import GripperWorld

    world = GripperWorld()
    seen = []

    def fake(panel, cfg):
        seen.append(cfg.m)
        verdict = "pass" if len(seen) == 3 else "stuck"
        return [types.SimpleNamespace(trace=None, params={}, ways=1, program="", result=types.SimpleNamespace(verdict=verdict))]

    monkeypatch.setattr(env, "run_panel", fake)
    monkeypatch.setattr(env, "attempts", lambda run: 10)
    out = world.solve(world.stock(), world.tasks()[0])
    assert out.success and out.attempts == 30
    assert seen == [seen[0], 2 * seen[0], 4 * seen[0]]
