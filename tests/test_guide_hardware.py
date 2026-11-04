"""A guide's rods and bushings are a choice of real parts (D-098), and research comes first (D-099)."""

import json
import math

import numpy as np
import pytest

from sea.evolve.core import Idea, Mechanism, Part, Requirement
from sea.evolve.loop import make_world
from sea.evolve.requirements import coverage
from sea.evolve.research import Note, add
from sea.evolve.tools import Server, ToolError
from sea.mech.graph import Graph
from sea.sim import jaws as J
from sea.worlds import gripper as G
from tests.test_tools import Measured, _ctx


@pytest.fixture(scope="module")
def world():
    return make_world("gripper")


def _slide(world, rod=None, radius=50.0):
    params = {"sector_radius_mm": radius, **({"rod_mm": rod} if rod else {})}
    return world.build_idea(world.stock(), Idea(name="s", description="d", slot="jaw", mechanism=Mechanism(
        module="long_slide", params=params)), "n")


def test_stiffness_goes_as_the_fourth_power_of_the_rod():
    ei3 = 200e9 * math.pi * (1.5e-3) ** 4 / 4  # EI = E pi r^4 / 4: 0.795 N m^2
    assert J.LM3.ei == pytest.approx(ei3) and ei3 == pytest.approx(0.795, abs=0.001)
    assert J.BUSHINGS["lm6uu"].ei == pytest.approx(16 * ei3) and J.BUSHINGS["lm8uu"].ei / ei3 == pytest.approx((8 / 3) ** 4)
    assert J.rod_stiffness(0.05, J.BUSHINGS["lm6uu"].ei)[0] == pytest.approx(16 * J.rod_stiffness(0.05)[0])


def test_the_default_is_exactly_the_slide_there_was(world):
    assert _slide(world).parts["jaw"].mechanism == _slide(world, 3.0).parts["jaw"].mechanism
    g = Graph.model_validate_json(_slide(world).parts["jaw"].mechanism)
    assert g.guides[0].bushing == "lm3uu" and {p.radius for b in g.bodies for p in b.parts if getattr(p, "role", "") == "rod"} == {J.ROD_R}


def test_a_size_the_catalogue_lacks_is_refused_naming_those_it_has(world):
    with pytest.raises(ValueError, match="the sizes are"):
        _slide(world, 7.0)


@pytest.mark.parametrize("rod", [3.0, 4.0, 5.0, 6.0, 8.0])
def test_every_size_builds_and_passes_the_laws_and_prices_its_own_parts(world, rod):
    body = _slide(world, rod, 45.0)
    assert world.check(body) == []
    items = {b.item for b in world.bom(body)}
    assert {f"steel_rod_{rod:.0f}x150", f"lm{rod:.0f}uu_bearing"} <= items <= {*world.manifest()["items"], "print:PLA"}


def test_a_thicker_rod_holds_what_a_thin_one_cannot(world):
    thin, thick = (G.to_design(_slide(world, d, 45.0)).fingers for d in (3.0, 6.0))
    q = 1.0
    t, k = G._rod_numbers(thin, q, 1.7), G._rod_numbers(thick, q, 1.7)
    assert t["rated_n"] == J.BUSHING_RATED_N and t["bushing_n"] > t["rated_n"]
    rod_lift = lambda r: r["lift_mm"] - r["bend_mm"]  # noqa: E731  the rods' share: the rack's own flex is D-100's
    assert k["rated_n"] == 265.0 and k["bushing_n"] < k["rated_n"] and rod_lift(k) < k["skip_mm"] < rod_lift(t)


def test_the_simulator_uses_the_parts_named():
    from sea.design.build import build as build_design
    from sea.sim.build_model import build_model

    body = _slide(make_world("gripper"), 6.0)
    sim = build_model(build_design(G.to_design(body), G._limits()).design)
    hw = J.BUSHINGS["lm6uu"]
    assert sim.rods.ei == pytest.approx(hw.ei) and sim.rods.rated == hw.rated_n and sim.rods.length == hw.length


def _mutated(world, change):
    body = _slide(world, 6.0, 45.0)
    d = json.loads(body.parts["jaw"].mechanism)
    change(d)
    return body.model_copy(update={"parts": {**body.parts, "jaw": body.parts["jaw"].model_copy(update={"mechanism": json.dumps(d)})}})


def test_laws_refuse_a_guide_whose_parts_do_not_fit(world):
    def thin_rods(d):
        for b in d["bodies"]:
            for p in b["parts"]:
                if p.get("role") == "rod":
                    p["radius"] = J.ROD_R

    def small_carriage(d):
        car = next(b for b in d["bodies"] if b["name"] == "jaw_carriage")
        car["parts"] = [p for p in car["parts"] if p["kind"] != "box" or p["name"] == "jaw_rack"]

    def wrong_bought(d):
        d["guides"][0]["bushing"] = "lm8uu"

    def nonsense(d):
        d["guides"][0]["bushing"] = "lm99"

    assert any("runs on a 6 mm rod" in e for e in world.check(_mutated(world, thin_rods)))
    assert any("no room for an LM6UU" in e for e in world.check(_mutated(world, small_carriage)))
    assert any("carries 0 LM8UU" in e for e in world.check(_mutated(world, wrong_bought)))
    assert any("no bushing 'lm99'" in e for e in world.check(_mutated(world, nonsense)))


def test_the_brief_for_system_2_lists_the_parts_and_their_ratings(world):
    text = world.modules_text()
    assert "lm6uu (6 mm rod, 12 mm outside, 19 mm long, 8.5 g, rated 265 N)" in text and "rod_mm" in text


def test_a_group_named_as_a_source_is_covered():
    r = Requirement(id="R", statement="s", metric="x()", sense=">=", marginal=1.0, source="reg_sugar_box: fell; tuna_fish_can: 85 mm")
    assert coverage([r], ["sugar_box", "tuna_fish_can", "cup"]) == ["cup"]


def test_research_comes_first_in_a_runs_first_iteration(tmp_path):
    w = Measured()
    ctx = _ctx(w)
    book, log = tmp_path / "n.jsonl", tmp_path / "r.jsonl"
    ctx = ctx.model_copy(update={"notebook": str(book), "research_log": str(log), "run": "r", "iteration": 0})
    server = Server(ctx, tmp_path / "calls.jsonl", w)
    with pytest.raises(ToolError, match="research first: this run has recorded 0 of 2"):
        server._research_first()
    for k in range(2):
        add(book, Note(claim=f"c{k}", source_url="http://a"), {"http://a"}, "w", "r", 0)
    server._research_first()
    other = Server(ctx.model_copy(update={"run": "someone else"}), tmp_path / "c2.jsonl", w)
    with pytest.raises(ToolError):  # notes from other runs are read, but this run still reads for itself
        other._research_first()
    Server(ctx.model_copy(update={"iteration": 1, "run": "someone else"}), tmp_path / "c3.jsonl", w)._research_first()


def test_a_body_that_cannot_be_priced_is_refused_and_does_not_end_the_run():
    from sea.evolve.loop import screen

    class Broken(Measured):
        def __init__(self, laws):
            super().__init__()
            self.laws = laws

        def check(self, body):
            return self.laws

        def bom(self, body):
            if body.id == "bad":
                raise ValueError("repeated name 'x' in geom")
            return super().bom(body)

    for laws, want in ((["the mechanism does not compile"], "does not compile"), ([], "cannot be priced")):
        w = Broken(laws)
        body = w.stock()
        errs, _ = screen(w, body.model_copy(update={"id": "bad"}), body, w.manifest(), _ctx(w).spent, {})
        assert any(want in e for e in errs)


def test_the_tuner_moves_only_catalogue_parameters_when_a_swap_builds_on_a_built_jaw(world):
    from sea.evolve.loop import tunable

    catalogue = {p.name: p for p in world.catalogue()}
    slide = _slide(world, 6.0)
    pads = catalogue["pads"]
    built = slide.model_copy(update={"parts": {**slide.parts, "pads": Part(type="pads", params={k: p.lo for k, p in pads.params.items()})}})
    assert built.parts["jaw"].type == "built"
    space = tunable(built, ["jaw", "pads"], catalogue)  # how the slots read: the jaw differs from the current body
    assert {slot for slot, _, _ in space} == {"pads"} and {k for _, k, _ in space} == set(pads.params)
    assert tunable(built, ["jaw"], catalogue) == []


# ---------------------------------------------------------------- printed racks and teeth (D-100)
def _beam(world, rod=6.0, radius=50.0):
    from sea.mech.build import rack_beam

    g = Graph.model_validate_json(_slide(world, rod, radius).parts["jaw"].mechanism)
    return g, rack_beam(g, g.guides[0])


def test_lewis_and_a_racks_bend_match_the_hand_numbers(world):
    assert J.lewis_y(17) == pytest.approx(0.302) and J.lewis_y(100) == pytest.approx(0.447)
    g, beam = _beam(world)
    # the module's rack is 6 wide and 5 tall: I = 6 * 5^3 / 12 = 62.5 mm^4; EI = 1.808e9 * 62.5e-12 = 0.113 N m^2
    assert beam.ei_sep == pytest.approx(J.PLA_E_PA * 62.5e-12, rel=1e-6) and beam.c_sep == pytest.approx(2.5e-3)
    # 11.2 N at 85 mm: F L^3 / 3 EI = 11.2 * 6.14e-4 / 0.339 = 20 mm, where teeth skip at 0.5 mm
    teeth = 1.62 / 0.0525
    r = beam.loads(beam.lo - 0.085, teeth)
    assert r["overhang"] == pytest.approx(0.085) and r["bend"] * 1000 == pytest.approx(20.5, abs=1.0)
    # a tooth on the 6 mm rack face, module 1: 30.9 N / (6e-3 * 1e-3 * Y(100) = 0.447) = 11.5 MPa
    assert r["tooth"] / 1e6 == pytest.approx(11.5, abs=0.3)


def test_the_racks_loads_follow_the_section_the_roller_and_the_direction(world):
    g, beam = _beam(world)
    teeth = 30.0
    behind, ahead = beam.loads(beam.lo - 0.05, teeth), beam.loads(beam.hi + 0.05, teeth)
    assert behind["buckle"] == math.inf and ahead["buckle"] == pytest.approx(math.pi**2 * beam.ei_min / (2 * 0.05) ** 2)
    backed = J.RackBeam(beam.lo, beam.hi, beam.ei_sep, beam.c_sep, beam.ei_min, beam.tooth, True).loads(beam.lo - 0.085, teeth)
    assert backed["bend"] == 0.0 and backed["stress"] == 0.0 and beam.loads((beam.lo + beam.hi) / 2, teeth)["overhang"] == 0.0

    from sea.mech.build import rack_beam
    from sea.mech.graph import Cylinder

    gd = g.guides[0]
    car = next(b for b in g.bodies if b.name == gd.carriage)
    rack = next(p for p in car.parts if getattr(p, "name", "") == "jaw_rack")
    # a rib along the rack, 20 mm tall, stiffens it by far more than the depth it adds
    rib = rack.model_copy(update={"name": "rib", "half": (rack.half[0], 0.01, 0.0015), "pos": (rack.pos[0], rack.pos[1] - rack.half[1] - 0.01, rack.pos[2])})
    deep = g.model_copy(update={"bodies": tuple(b.model_copy(update={"parts": (*b.parts, rib)}) if b.name == gd.carriage else b for b in g.bodies)})
    assert rack_beam(deep, gd).ei_sep > 20 * beam.ei_sep
    # a roller under the rack at the mesh, touching it, backs it; one elsewhere does not
    span_low = float((J.HINGE + J.R0 @ np.array(rack.pos))[2] - rack.half[1])
    y = float((J.HINGE + J.R0 @ np.array(rack.pos))[1])

    def with_roller(x, gap=0.0):
        roller = Cylinder(name="roller", pos=(x, y, span_low - 0.003 - gap), radius=0.003, half_length=0.002, material="steel", role="roller")
        return g.model_copy(update={"bodies": tuple(b.model_copy(update={"parts": (*b.parts, roller)}) if b.name == "gripper" else b for b in g.bodies)})

    assert rack_beam(with_roller(gd.mesh_x), gd).backed and not rack_beam(g, gd).backed
    assert not rack_beam(with_roller(gd.mesh_x + 0.05), gd).backed and not rack_beam(with_roller(gd.mesh_x, 0.005), gd).backed


def test_the_thin_rack_that_won_on_the_old_judge_now_fails_for_the_right_reason(world):
    from pathlib import Path

    from sea.evolve.core import Body

    summary = Path("runs/evolve/gripper_f_s0/summary.json")
    if not summary.exists():
        pytest.skip("the run's record is not here")
    body = Body.model_validate(json.loads(summary.read_text())["final_body"])
    f = G.to_design(body).fingers
    r = G._rod_numbers(f, 1.47, 1.7)
    assert r["bend_mm"] > r["skip_mm"] and r["tooth_mpa"] > J.PLA_ALLOW_PA / 1e6 and r["rack_mpa"] > J.PLA_ALLOW_PA / 1e6
    assert world.read(body, "tooth_stress_mpa", {}) == pytest.approx(r["tooth_mpa"])
    assert world.read(body, "rack_bend_mm", {"width_mm": 85}) > 1.0


def test_a_roller_is_priced_as_the_bearing_it_is(world):
    g, _ = _beam(world)
    from sea.mech.graph import Cylinder

    roller = Cylinder(name="roller", pos=(0.02, -0.03, -0.09), radius=0.003, half_length=0.002, material="steel", role="roller")
    g = g.model_copy(update={"bodies": tuple(b.model_copy(update={"parts": (*b.parts, roller)}) if b.name == "gripper" else b for b in g.bodies)})
    items = {b.item: b.qty for b in G._built_bom(g.model_dump_json())}
    assert items["mr63_bearing"] >= 1
