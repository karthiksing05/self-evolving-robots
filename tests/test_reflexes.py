"""Sensors in the loop (D-036): System 1's reflexes, the felt features, and what System 2 is shown."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest
import yaml

from sea.compose.library import Library
from sea.contracts import Design, HardLimits, SensorSuite
from sea.control.executor import reset_episode, run_program
from sea.design.build import build
from sea.loop import trace_summary
from sea.propose.prompt import _failing_section
from sea.sim.env import ControlRun, PanelRunner, SuiteConfig, _judge
from sea.sim.grasp_world import grasp_panels, object_spec, table_panel
from sea.sim.sensors import IMPACT_ACC

ROOT = Path(__file__).resolve().parents[1]
LIMITS = HardLimits.model_validate(yaml.safe_load((ROOT / "configs" / "limits_grasp.yaml").read_text()))
PICK = Library.seed().prims["pick_up"].body


def _cfg(pads: bool = True) -> SuiteConfig:
    built = build(Design(id="stock", sensors=SensorSuite(jaw_pads=pads)), LIMITS)
    return SuiteConfig(design=built.design, tools=built.tools, library=Library.seed(), limits=LIMITS, seed=0,
                       confirm=False, m=0)


CASES = {
    "can": (grasp_panels("train", 1)[0], True),
    "heavy": (table_panel("heavy", "train", 0, object_spec("cylinder", 30, 60, 1200, 0.8)), True),
    # a 50 mm, 800 g can at friction 0.2 lifts about 10 mm and slips out of the stock jaws (D-054)
    "slips": (table_panel("slips", "train", 0, object_spec("cylinder", 50, 80, 800, 0.2)), True),
    "slips_no_pads": (table_panel("slips", "train", 0, object_spec("cylinder", 50, 80, 800, 0.2)), False),
}


@pytest.fixture(scope="module")
def picks():
    out = {}
    for name, (panel, pads) in CASES.items():
        (obj,) = panel.controls
        trace = PanelRunner(panel, _cfg(pads), 0).attempt(obj, PICK, {})
        out[name] = (panel, obj, trace, _judge(obj, panel, LIMITS)(trace))
    return out


def test_sensing_never_changes_the_physics(picks):
    panel, obj, sensed, _ = picks["can"]
    runner = PanelRunner(panel, _cfg(), 0)
    reset_episode(runner.sim)
    blind = run_program(runner.sim, PICK, obj.id, replace(runner.config(obj), sense=False), panel_id=panel.id)
    assert blind.sensors is None and sensed.sensors is not None
    assert blind.q == sensed.q and blind.control_q == sensed.control_q


def test_a_normal_pick_feels_its_object_and_no_impact(picks):
    _, obj, trace, result = picks["can"]
    s = trace.sensors
    assert result.verdict == "pass" and trace.note == ""
    assert s.impacts == 0 and s.peak_acc < IMPACT_ACC  # the tick-averaged IMU reads no vibration as impact
    assert s.payload_held_g == pytest.approx(obj.mass_g, abs=15.0)  # weighed by the servos, not looked up
    assert s.pad_drop_t is None and s.peak_pads_n > 30
    assert (result.peak_acc, result.stall_s) == (s.peak_acc, s.stall_s)  # carried to memory after the verdict


def test_too_heavy_stops_on_load_limit(picks):
    _, _, trace, result = picks["heavy"]
    assert trace.note == "load_limit" and trace.sensors.stall_joint == "shoulder_lift"
    assert result.verdict == "stuck"  # the verdict still comes from ground truth: it never lifted


def test_a_lost_grip_is_felt_only_with_pads(picks):
    _, _, felt, r_felt = picks["slips"]
    _, _, blind, r_blind = picks["slips_no_pads"]
    assert felt.note == "slip" and felt.sensors.pad_drop_t is not None
    assert blind.note == "" and "jaw_pads" not in blind.sensors.parts and blind.sensors.pad_drop_t is None
    assert r_felt.verdict == r_blind.verdict == "wrong_behavior"  # the reflex changes the diagnosis, not the verdict


def test_system_2_is_shown_what_the_robot_felt(picks):
    panel, obj, trace, result = picks["heavy"]
    s = trace_summary(ControlRun(result, trace, "pick_up", {}), obj, panel)
    assert s.felt and s.stopped_by == "load_limit" and s.stall_joint == "shoulder_lift"
    text = _failing_section([s])
    assert "felt: stopped_by=load_limit" in text and "(shoulder_lift)" in text and "payload_held_g=-" in text
    plain = s.model_copy(update={"felt": False})
    assert "felt:" not in _failing_section([plain])  # an unsensed attempt adds no line
