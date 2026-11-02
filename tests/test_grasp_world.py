"""The grasp world (docs/gripper-experiment.md, D-035): objects as controls on a table."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import yaml

from sea.compose import checks
from sea.compose.library import Library
from sea.compose.typecheck import applicable_programs, typecheck
from sea.contracts import Design, HardLimits
from sea.design.build import build
from sea.design.grip import STOCK_WIDEST_MM
from sea.sim.build_model import JAW_FRICTION, build_model, pair_friction
from sea.sim.env import PanelRunner, SuiteConfig, run_suite
from sea.sim.grasp_world import grasp_panels, load_config, object_spec, phase_b_panels, table_panel
from sea.sim.panels import reference_panel
from sea.verify.verdicts import judge

ROOT = Path(__file__).resolve().parents[1]
LIMITS = HardLimits.model_validate(yaml.safe_load((ROOT / "configs" / "limits_grasp.yaml").read_text()))
PANEL_LIMITS = HardLimits.model_validate(yaml.safe_load((ROOT / "configs" / "limits.yaml").read_text()))


def _cfg(m: int = 0) -> SuiteConfig:
    built = build(Design(id="stock"), LIMITS)
    return SuiteConfig(design=built.design, tools=built.tools, library=Library.seed(), limits=LIMITS, seed=0,
                       confirm=False, m=m)


def test_phase_a_tables_are_seeded_and_inside_the_stock_envelope():
    a = grasp_panels("train", 8)
    assert a == grasp_panels("train", 8) and a != grasp_panels("val", 8)
    ranges = load_config()["phase_a"]
    for p in a:
        (c,) = p.controls
        lo, hi = ranges[c.shape]["width_mm"]
        assert c.cls == "grasp" and lo <= c.size_mm[0] <= hi <= STOCK_WIDEST_MM[c.shape]
        assert p.rules.banned_contacts == [] and checks.canonical(c.success) == "held_for($target, 1.0)"
    assert {p.split for p in phase_b_panels()} == {"phase_b"} and not LIMITS.banned_contacts


def test_an_object_is_a_free_body_whose_coordinate_is_its_height():
    sim = build_model(Design(id="stock"), table_panel("t", "train", 0, object_spec("cylinder", 30, 60, 100, 0.6)))
    h = sim.controls["obj"]
    assert sim.model.body_parentid[h.body_id] == 0  # MuJoCo allows free joints on top-level bodies only
    assert sim.control_q("obj") == pytest.approx(0.030, abs=1e-6)
    assert np.allclose(sim.model.pair_friction[:, 0], pair_friction(JAW_FRICTION, 0.6))
    sim.step(250)
    assert sim.control_q("obj") == pytest.approx(0.030, abs=5e-4)  # rests on the table, nothing pushes it


def test_the_aperture_rule_makes_wide_objects_structural():
    lib = Library.seed()
    for width, ok in ((45, True), (70, True), (75, False)):  # the stock envelope: cans to 70 mm (D-054)
        c = object_spec("cylinder", width, 70, 100, 0.8)
        errs = typecheck(lib.prims["pick_up"].body, Design(id="stock"), lib.prims, LIMITS, control=c)
        assert (not errs) == ok and {e.code for e in errs} <= {"aperture"}
    for c in reference_panel().controls:  # the grasp program never applies to a panel control
        names = [p.name for p in applicable_programs(c, Design(id="stock"), lib, PANEL_LIMITS)]
        assert "pick_up" not in names


def test_held_for_parses_and_needs_the_hold():
    e = checks.parse("held_for($target, 1.5)")
    assert checks.pretty(e) == "held_for($target, 1.5)" and checks.atoms(e)[0].value == 1.5

    class Env:
        def held_s(self, ref):
            return 1.2

    assert checks.check("held_for($target, 1.0)", Env()) and not checks.check("held_for($target, 1.5)", Env())


@pytest.fixture(scope="module")
def suite():
    panels = grasp_panels("train", 2) + [p for p in phase_b_panels() if p.id in ("gb_wide_can_60", "gb_ball_40")]
    return {r.result.panel_id: r for r in run_suite(panels, _cfg()).runs}


def test_the_stock_gripper_picks_phase_a_and_escalates_phase_b(suite):
    assert suite["gtr000"].result.verdict == "pass" and suite["gtr001"].result.verdict == "pass"
    # Phase B's objects were chosen on soft contacts, which lost cans past 55 mm and balls near the equator
    # (D-054): on rigid contacts, the stock jaws hold both.
    assert suite["gb_wide_can_60"].result.verdict == "pass" and suite["gb_ball_40"].result.verdict == "pass"


def test_the_squeeze_is_the_task_not_a_force_violation(suite):
    tr = suite["gtr000"].trace
    assert "pinch" in tr.contact_types and tr.peak_control_force > 20  # the squeeze held a whole tick (D-053)
    assert max(tr.tip_force) < LIMITS.max_force_n  # every contact but the squeeze stays under the hard limit


def test_a_lift_that_falls_back_is_not_a_pass():
    # A 60 mm, 800 g can at friction 0.25 lifts through its travel and then slides out (D-054).  Run the pick:
    # the object passed its travel for a moment, but held_for needs the full hold.
    panel = table_panel("a1", "train", 0, object_spec("cylinder", 60, 80, 800, 0.25))
    (can,) = panel.controls
    runner = PanelRunner(panel, _cfg(), 0)
    trace = runner.attempt(can, Library.seed().prims["pick_up"].body, {})
    lifted = max(q - trace.q_rest for q in trace.control_q)
    assert lifted >= can.travel * 0.3 and trace.success_time_s is None
    assert judge(trace, can, panel.rules, LIMITS).verdict == "wrong_behavior"
