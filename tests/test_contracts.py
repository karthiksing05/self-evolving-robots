"""Data contracts (spec Section 6; test list 12.2): JSON round-trips, extra="forbid", schemas."""

from __future__ import annotations

import inspect

import pydantic
import pytest
from pydantic import TypeAdapter

import sea.contracts as C
from sea.compose import dsl
from sea.sim.panels import reference_panel


def _limits() -> C.HardLimits:
    return C.HardLimits(workspace_box=(C.Vec3(x=0.1, y=-0.2, z=-0.1), C.Vec3(x=0.4, y=0.2, z=0.25)))


def _program() -> C.Program:
    root = dsl.parse("Seq(approach($target), Guard(push($target, force=4.0), until=moved_at_least($target, travel), "
                     "cap=force_below(9.5)), With(0, retreat(0.05)))")
    return C.Program(id="p", target_cls="push_button", root=root, source="s2")


def _result() -> C.ControlResult:
    return C.ControlResult(panel_id="p0", control_id="c0", cls="knob", verdict="stuck", peak_force=9.4,
                           travel_reached=0.2, detents=1, time_s=3.0, surprise_z={"peak_force": 2.0})


def _proposal() -> C.Proposal:
    tool = C.ToolSpec(template="hook", params={"depth": 10, "throat": 6, "length": 40}, contact="hook")
    return C.Proposal(kind="new_tool", rung=3, rationale="r", hypothesis="h",
                      ruled_out=["rung0: no", "rung1: no", "rung2: no"],
                      design_delta=C.DesignDelta(rack_add=[tool], servo_voltage={"wrist_flex": 12.0}),
                      reparam={"push.force": (1.0, 5.0)},
                      prediction=C.Prediction(flips_to_pass=["p0:c0"], confidence=0.6))


def _decision() -> C.Decision:
    return C.Decision(accepted=True, rule="mean", delta_mean=0.1, delta_lo=-0.1, delta_hi=0.3, regressions=["r:c1"])


SAMPLES: dict[type, pydantic.BaseModel] = {}


def _samples() -> dict[type, pydantic.BaseModel]:
    if SAMPLES:
        return SAMPLES
    panel = reference_panel()
    prog = _program()
    s = [
        C.Vec3(x=1, y=2, z=3),
        C.Pose(pos=C.Vec3(x=0, y=0, z=0), quat_wxyz=(0.5, 0.5, 0.5, 0.5)),
        panel.rules,
        panel.controls[0],
        panel,
        C.ToolSpec(template="pad", params={"w": 10, "h": 10, "thickness": 4, "friction": 0.8}, contact="pad"),
        C.FingerSpec(jaw_len_mm=5.0),
        C.ArmSpec(upper_scale=1.1, servo_voltage={j: 12.0 for j in C.ARM_JOINTS}),
        C.BuildEstimate(print_minutes=12.0),
        C.SensorSuite(jaw_pads=False),
        C.Design(id="d1", parent="d0", tool_rack=[C.ToolSpec(template="probe", params={"diameter": 4, "length": 40},
                                                             contact="probe")]),
        C.PrimRef(name="push", args={"_0": "$target", "force": 3.0}),
        prog.root,
        prog,
        C.PrimitiveDef(name="x", kind="composite", doc="d", provides={"actuates": "knob"},
                       params={"p0": (0.1, 0.2)}, body=prog),
        C.StepTrace(panel_id="p0", control_id="c0", program_id="p", attempt=0, t=[0.0, 0.02],
                    q=[[0.0] * 6, [0.1] * 6], tip_force=[0.0, 1.0], control_q=[0.0, 0.001],
                    contacts=[(0.02, "edge", "c0")], failed_primitive="push"),
        _result(),
        C.CostVector(print_minutes=3, approvals=1),
        C.Report(run_id="r", iteration=0, design_id="d", panels=["p0"], results=[_result()], control_coverage=1.0,
                 pass_rate=0.0, coverage_reward=0.0, cost=C.CostVector(), reward=-0.1),
        C.Prediction(confidence=0.5, may_regress=["p0:c1"]),
        C.DesignDelta(arm={"upper_scale": 1.1}, rack_remove=[0]),
        _proposal(),
        C.ProposalBatch(proposals=[_proposal()]),
        _decision(),
        C.FixRecord(id="f", iteration=2, proposal=_proposal(), decision=_decision(), distance=0.3),
        _limits(),
        C.Posture(beta=0.5, per_area={"force": -1.0}),
        C.Budgets(print_minutes=60),
        C.StopRule(patience=2),
        C.Ablations(zero_shot=True),
        C.RunSpec(seed=1, limits=_limits(), proposer="cma", llm="mock"),
    ]
    SAMPLES.update({type(m): m for m in s})
    return SAMPLES


def _models() -> list[type]:
    return [o for o in vars(C).values() if inspect.isclass(o) and issubclass(o, pydantic.BaseModel)
            and o.__module__ == "sea.contracts" and not o.__name__.startswith("_")]


def test_every_model_has_a_sample():
    assert set(_models()) == set(_samples())


@pytest.mark.parametrize("model", _models(), ids=lambda m: m.__name__)
def test_json_round_trip(model):
    obj = _samples()[model]
    again = model.model_validate_json(obj.model_dump_json())
    assert again == obj
    assert again.model_dump(mode="json") == obj.model_dump(mode="json")


@pytest.mark.parametrize("model", _models(), ids=lambda m: m.__name__)
def test_extra_fields_rejected(model):
    data = _samples()[model].model_dump(mode="json")
    data["not_a_field"] = 1
    with pytest.raises(pydantic.ValidationError):
        model.model_validate(data)


def test_frozen_except_trace():
    with pytest.raises(pydantic.ValidationError):
        _samples()[C.Vec3].x = 5.0  # type: ignore[misc]
    tr = C.StepTrace(panel_id="p", control_id="c", program_id="x", attempt=0)
    tr.note = "mutable while recording"
    assert tr.note


def test_schemas_generate():
    schema = TypeAdapter(list[C.Proposal]).json_schema()
    assert schema["type"] == "array"
    batch = C.ProposalBatch.model_json_schema()
    assert batch["type"] == "object" and "proposals" in batch["properties"]
    assert "Node" in batch.get("$defs", {})


def test_proposal_batch_bounds_and_confidence():
    with pytest.raises(pydantic.ValidationError):
        C.ProposalBatch(proposals=[])
    with pytest.raises(pydantic.ValidationError):
        C.ProposalBatch(proposals=[_proposal()] * 4)
    with pytest.raises(pydantic.ValidationError):
        C.Prediction(confidence=1.5)


def test_helpers():
    assert C.Vec3(x=1, y=2, z=3).as_list() == [1, 2, 3]
    assert _result().key == "p0:c0"
    assert [C.recess_bin(v) for v in (0.0, 5.0, 10.0, 12.0)] == ["0", "0-10", "0-10", ">10"]
    import math

    assert [C.tilt_bin(math.radians(v)) for v in (0, 20, 30, 45, 60)] == ["0-20", "0-20", "20-45", "20-45", ">45"]
