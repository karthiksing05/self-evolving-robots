"""Design space ranges, tool spec validation, and the [0, 1]^d encoding (spec 8.8)."""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from sea.contracts import ArmSpec, Design, FingerSpec, HardLimits, ToolSpec, Vec3
from sea.design.space import (
    SLOT_CHOICES,
    TEMPLATE_CONTACT,
    TOOL_RANGES,
    DesignSpace,
    make_tool_spec,
    mid_params,
    params_at,
    tool_errors,
    tools_equivalent,
)

LIMITS = HardLimits(workspace_box=(Vec3(x=0.1, y=-0.2, z=-0.1), Vec3(x=0.4, y=0.2, z=0.25)))


def test_ranges_match_spec_table():
    assert TOOL_RANGES["pad"] == {"w": (8, 30), "h": (8, 30), "thickness": (3, 8), "friction": (0.4, 1.2)}
    assert TOOL_RANGES["hook"] == {"depth": (4, 20), "throat": (3, 12), "length": (20, 60)}
    assert TOOL_RANGES["knob_cup"] == {"inner_diameter": (8, 40), "depth": (4, 15), "ribs": (0, 8)}
    assert TOOL_RANGES["soft_tip"] == {"radius": (2, 8), "softness": (0, 1)}
    assert TOOL_RANGES["slot_inserter"] == {"blade_thickness": (0.6, 2), "width": (10, 40), "length": (20, 60)}
    assert TOOL_RANGES["probe"] == {"diameter": (3, 8), "length": (20, 80)}
    assert TEMPLATE_CONTACT == {"stock": "edge", "pad": "pad", "hook": "hook", "knob_cup": "cup",
                                "soft_tip": "pad", "slot_inserter": "edge", "probe": "probe"}


def test_make_tool_spec_fills_middle_and_contact():
    spec = make_tool_spec("probe", length=30)
    assert spec.contact == "probe"
    assert spec.params == {"diameter": 5.5, "length": 30.0}
    assert make_tool_spec("hook").params == mid_params("hook")
    assert params_at("pad", 1.0)["w"] == 30.0


@pytest.mark.parametrize("bad", [{"length": 100.0}, {"lenght": 30.0}])
def test_make_tool_spec_rejects_bad_params(bad):
    with pytest.raises(ValueError):
        make_tool_spec("probe", **bad)
    with pytest.raises(ValueError, match="unknown template"):
        make_tool_spec("spork")


def test_tool_errors():
    ok = make_tool_spec("pad")
    assert tool_errors(ok) == []
    wrong_contact = ToolSpec(template="pad", params=ok.params, contact="hook")
    assert any("contact" in e for e in tool_errors(wrong_contact))
    missing = ToolSpec(template="pad", params={"w": 10.0}, contact="pad")
    assert any("missing" in e for e in tool_errors(missing))
    extra = ToolSpec(template="pad", params={**ok.params, "colour": 1.0}, contact="pad")
    assert any("unknown parameter" in e for e in tool_errors(extra))
    low = ToolSpec(template="pad", params={**ok.params, "w": 1.0}, contact="pad")
    assert any("outside" in e for e in tool_errors(low))
    assert tool_errors(ToolSpec(template="stock", params={}, contact="edge")) == []


def test_tools_equivalent():
    a = make_tool_spec("hook", depth=10, throat=6, length=40)
    near = make_tool_spec("hook", depth=11, throat=6.5, length=43)  # within 0.1 x range width
    far = make_tool_spec("hook", depth=10, throat=6, length=50)
    assert tools_equivalent(a, near)
    assert not tools_equivalent(a, far)
    assert not tools_equivalent(a, make_tool_spec("probe"))


def _design() -> Design:
    return Design(
        id="d001",
        parent="d000",
        arm=ArmSpec(upper_scale=1.1, fore_scale=0.9, wrist_offset_mm=10.0, riser_m=0.05, base_pitch_rad=0.1,
                    servo_voltage={"shoulder_pan": 7.4, "shoulder_lift": 12.0, "elbow_flex": 12.0,
                                   "wrist_flex": 7.4, "wrist_roll": 7.4, "gripper": 7.4}),
        fingers=FingerSpec(jaw_len_mm=12.0, tip_radius_mm=3.0, softness=0.4),
        tool_rack=[make_tool_spec("hook", depth=8, throat=5, length=35), make_tool_spec("pad", w=20, h=12)],
    )


@pytest.mark.parametrize(
    ("levels", "dim"),
    [(["L0"], 20), (["L1"], 3), (["L2"], 5), (["L3"], 6), (["L0", "L1", "L2", "L3"], 34)],
)
def test_dims_and_names(levels, dim):
    space = DesignSpace(levels, rack_slots=4, limits=LIMITS)
    assert space.dim == dim == len(space.names())


def test_encode_decode_round_trip():
    space = DesignSpace(["L3", "L0", "L1", "L2"], rack_slots=4, limits=LIMITS)
    d = _design()
    vec = space.encode(d)
    assert vec.shape == (space.dim,)
    assert np.all((vec >= 0) & (vec <= 1))
    back = space.decode(vec, base=d, new_id="d002")
    assert back.id == "d002" and back.parent == "d001"
    assert [t.template for t in back.tool_rack] == ["hook", "pad"]
    for a, b in zip(back.tool_rack, d.tool_rack, strict=True):
        assert a.params == pytest.approx(b.params)
    assert back.fingers.model_dump() == pytest.approx(d.fingers.model_dump())
    arm_a, arm_b = back.arm.model_dump(), d.arm.model_dump()
    assert arm_a.pop("servo_voltage") == arm_b.pop("servo_voltage")
    assert arm_a == pytest.approx(arm_b)


def test_decode_clips_and_keeps_unsearched_levels():
    space = DesignSpace(["L0"], rack_slots=2, limits=LIMITS)
    base = _design()
    vec = np.full(space.dim, 7.0)  # out of range: clipped to 1 -> last template, max params
    d = space.decode(vec, base=base, new_id="x")
    assert [t.template for t in d.tool_rack] == [SLOT_CHOICES[-1]] * 2
    assert d.tool_rack[0].params == {"diameter": 8.0, "length": 80.0}
    assert d.fingers == base.fingers and d.arm == base.arm  # L1/L2 not searched
    empty = space.decode(np.zeros(space.dim), base=base, new_id="y")
    assert empty.tool_rack == []
    with pytest.raises(ValueError):
        space.decode(np.zeros(3), base=base, new_id="z")


def test_encode_ignores_stock_and_empty_slots():
    space = DesignSpace(["L0"], rack_slots=2, limits=LIMITS)
    d = Design(id="s", tool_rack=[ToolSpec(template="stock", params={}, contact="edge")])
    vec = space.encode(d)
    assert space.decode(vec, base=d, new_id="t").tool_rack == []


def test_arm_ranges_follow_limits():
    tight = LIMITS.model_copy(update={"max_link_scale": 1.1})
    space = DesignSpace(["L2"], rack_slots=0, limits=tight)
    d = space.decode(np.ones(space.dim), base=Design(id="a"), new_id="b")
    assert d.arm.upper_scale == pytest.approx(1.1)
    assert d.arm.riser_m == pytest.approx(0.10)


@settings(derandomize=True, max_examples=40, deadline=None)
@given(st.lists(st.floats(min_value=-1.0, max_value=2.0), min_size=34, max_size=34))
def test_decode_is_idempotent_through_encode(raw):
    space = DesignSpace(["L0", "L1", "L2", "L3"], rack_slots=4, limits=LIMITS)
    base = Design(id="base")
    first = space.decode(np.array(raw), base=base, new_id="n")
    second = space.decode(space.encode(first), base=base, new_id="n")
    assert [t.template for t in second.tool_rack] == [t.template for t in first.tool_rack]
    for a, b in zip(second.tool_rack, first.tool_rack, strict=True):
        assert a.params == pytest.approx(b.params)
        assert tool_errors(a) == []
    assert second.arm.servo_voltage == first.arm.servo_voltage


def test_sample_in_unit_cube():
    space = DesignSpace(["L0", "L1"], rack_slots=3, limits=LIMITS)
    x = space.sample(np.random.default_rng(0))
    assert x.shape == (space.dim,) and np.all((x >= 0) & (x < 1))
