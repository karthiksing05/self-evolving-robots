"""The skills library (D-096): seeds on any world, promotion on evidence, and nothing touched with the switch off."""


import json

import pytest

from sea.evolve import library as LB
from sea.evolve import research as RS
from sea.evolve.core import Body, Idea, Mechanism, ProposalBatch, Requirement, Use
from sea.evolve.llm import Scripted
from sea.evolve.loop import RunConfig, run
from sea.mech.elements import ELEMENTS, fragment
from sea.mech.graph import Body as GB
from sea.mech.graph import Box, Graph, Joint
from sea.mech.patch import Op, apply_patch
from sea.worlds.me2110_kit import KitWorld, graph_of
from tests.test_concepts import Patches, _rod


def _arm(length: float, prefix: str = "a", on: str = "base", extra_parts: int = 0) -> Graph:
    """A base with one hinged arm on it, named as System 2 might name it."""
    parts = [Box(name=f"{prefix}_bar", pos=(length / 2, 0.0, 0.0), half=(length / 2, 0.005, 0.005))]
    parts += [Box(name=f"{prefix}_x{i}", pos=(0.0, 0.0, 0.0), half=(0.01, 0.01, 0.01)) for i in range(extra_parts)]
    return Graph(name="g", bodies=(
        GB(name=on, stock=True, parts=(Box(name="plate", pos=(0.0, 0.0, 0.0), half=(0.1, 0.1, 0.01)),)),
        GB(name=f"{prefix}_link", parent=on, pos=(length, 0.0, 0.02),
           joints=(Joint(name=f"{prefix}_q", type="hinge", axis=(0.0, 1.0, 0.0)),), parts=tuple(parts))),
        excludes=((on, f"{prefix}_link"),))


class _World:
    def __init__(self, name, graphs):
        self.name, self.graphs = name, graphs

    def mechanism_text(self, body):
        return self.graphs[body.id].model_dump_json() if body.id in self.graphs else ""


def _keep(root, world, length, mount="base", prefix="a", run="r1", it=0, extra=0, name="hinged arm"):
    w = _World(world, {"new": _arm(length, prefix, mount, extra)})
    return LB.keep(root, w, Body(id="old", parts={}), Body(id="new", parent="old", parts={}), name, run, it)


def test_names_and_lengths_do_not_change_the_signature_and_a_part_does(tmp_path):
    a = _keep(tmp_path, "w1", 0.1)
    assert a == _keep(tmp_path, "w1", 0.3, prefix="arm2")
    assert a != _keep(tmp_path, "w1", 0.1, extra=1)


def test_nothing_that_moves_is_entered_and_a_body_without_a_mechanism_is_not(tmp_path):
    g = Graph(name="g", bodies=(GB(name="base", parts=(Box(name="p", pos=(0.0, 0.0, 0.0), half=(0.1, 0.1, 0.01)),)),
                                GB(name="plate2", parent="base", parts=(Box(name="q", pos=(0.0, 0.0, 0.0), half=(0.1, 0.1, 0.01)),))))
    assert LB.fragments(None, g) == []
    assert LB.keep(tmp_path, _World("w", {}), Body(id="o", parts={}), Body(id="n", parts={}), "x", "r", 0) == []


def test_a_mechanism_kept_in_one_context_is_not_offered_and_in_two_it_is_with_its_varying_lengths(tmp_path):
    sig = _keep(tmp_path, "w1", 0.1)[0]
    _keep(tmp_path, "w1", 0.2, it=1)  # the same context again is not new evidence
    assert LB.promoted(tmp_path) == {}
    _keep(tmp_path, "w2", 0.3, mount="chassis", run="r2")
    entry = LB.promoted(tmp_path)[sig]
    args = entry.arguments()
    assert entry.name == "hinged arm"
    assert any(lo == pytest.approx(0.05) and hi == pytest.approx(0.15) for lo, hi in args.values())
    # every stored mechanism says which trial proved it
    assert all(i.run and i.body and i.iteration >= 0 for e in LB.load(tmp_path)[1].values() for i in e.instances)


def test_a_promoted_mechanism_can_be_built_at_new_values_on_a_new_body(tmp_path):
    sig = _keep(tmp_path, "w1", 0.1)[0]
    _keep(tmp_path, "w2", 0.3, mount="chassis", run="r2")
    entries = LB.promoted(tmp_path)
    key = next(k for k, (lo, hi) in entries[sig].arguments().items() if "half.0" in k)
    idea = Idea(name="i", description="d", mechanism=Mechanism(module="self", uses=[
        Use(element=sig, prefix="new", on="frame", at=(0.1, 0.0, 0.2), params={key: 0.4})]))
    out = LB.expand(idea, entries)
    assert out.mechanism.uses == []
    base = Graph(name="b", bodies=(GB(name="frame", parts=(Box(name="f", pos=(0.0, 0.0, 0.0), half=(0.1, 0.1, 0.01)),)),))
    g = apply_patch(base, [Op.model_validate(o) for o in out.mechanism.patch])
    link = next(b for b in g.bodies if b.parent == "frame")
    assert link.pos == (0.1, 0.0, 0.2) and link.joints[0].name.startswith("new_")
    assert link.parts[0].half[0] == 0.4
    assert ("frame", link.name) in g.excludes  # what tied it to its mount is carried over
    with pytest.raises(ValueError, match="no argument"):
        LB.expand(idea.model_copy(update={"mechanism": Mechanism(module="self", uses=[
            Use(element=sig, prefix="n", on="frame", params={"nope": 1.0})])}), entries)


def test_an_entry_unseen_for_ten_runs_is_pruned_and_a_promoted_one_stays(tmp_path):
    lonely = _keep(tmp_path, "w1", 0.1, extra=1)[0]
    proven = _keep(tmp_path, "w1", 0.1)[0]
    _keep(tmp_path, "w2", 0.2, mount="chassis")
    LB.finish_run(tmp_path)  # the run that kept them
    for _ in range(10):
        assert lonely in LB.load(tmp_path)[1]
        LB.finish_run(tmp_path)
    assert lonely not in LB.load(tmp_path)[1]
    assert proven in LB.load(tmp_path)[1]


def test_a_method_used_in_two_runs_is_shown_only_where_its_instruments_exist(tmp_path):
    from sea.evolve.requirements import Row

    world = KitWorld()
    r = Requirement(id="R1", statement="narrow", metric="width_in() * 2", sense="<=", marginal=1.0, source="t")
    bad = Requirement(id="R2", statement="broken", metric="no_such()", sense="<=", marginal=1.0, source="t")
    rows = [Row(id="R1", value=1.0, meets="ideal"), Row(id="R2", value=None, meets="unknown", error="no such")]
    LB.remember(tmp_path, [r, bad], rows, "me2110_kit", "r1")
    assert LB.methods(tmp_path, world) == []
    LB.remember(tmp_path, [r, bad], rows, "me2110_kit", "r2")
    assert [m.metric for m in LB.methods(tmp_path, world)] == ["width_in() * 2"]
    assert "no_such" not in (tmp_path / LB.METHODS).read_text()
    other = Requirement(id="R3", statement="s", metric="squeeze_n() + 1", sense="<=", marginal=1.0, source="t")
    for r_id in ("a", "b"):
        LB.remember(tmp_path, [other], [Row(id="R3", value=1.0, meets="ideal")], "gripper", r_id)
    assert "squeeze_n" not in " ".join(m.metric for m in LB.methods(tmp_path, world))


def test_a_rejected_proposal_counts_against_the_notes_it_cites(tmp_path):
    path = tmp_path / "n.jsonl"
    note = RS.add(path, RS.Note(claim="c", source_url="http://a"), {"http://a"}, "w", "r", 0)
    assert RS.against(path, f"because {note.id}") == [note.id]
    RS.credit(path, note.id)
    got = RS.load(path)[0]
    assert (got.credit, got.against) == (1, 1)
    assert "rejected 1" in RS.text([got])


def test_the_prompt_section_lists_the_elements_and_what_is_proven(tmp_path):
    world = KitWorld()
    text = LB.text(tmp_path, world)
    assert all(n in text for n in ELEMENTS)
    assert not (tmp_path / LB.MECHANISMS).exists()  # reading writes nothing
    sig = _keep(tmp_path, "w1", 0.1)[0]
    assert sig not in LB.text(tmp_path, world)
    _keep(tmp_path, "w2", 0.3, mount="chassis")
    assert sig in LB.text(tmp_path, world)


SMALL = {"lever": {"arm_m": 0.05}, "latch": {"arm_m": 0.05}, "four_bar": {"crank_m": 0.02, "coupler_m": 0.05, "rocker_m": 0.05, "ground_m": 0.05},
         "slider_crank": {"crank_m": 0.02, "rod_m": 0.06}, "rack_pinion": {"pitch_r_m": 0.01, "width_m": 0.01, "travel_m": 0.05},
         "gear_train": {"driver_r_m": 0.01, "follower_r_m": 0.02, "width_m": 0.01}, "cam": {"disc_r_m": 0.02, "eccentric_m": 0.005},
         "ratchet": {"wheel_r_m": 0.03, "teeth": 8}, "guide": {"rod_m": 0.1, "rod_gap_m": 0.03, "travel_m": 0.03}}


@pytest.mark.parametrize("name", sorted(ELEMENTS))
def test_every_seed_bolted_to_the_kits_plate_passes_its_laws(name):
    """Nothing about a seed is refused for being a seed; it needs only to be placed where it fits."""
    world = KitWorld()
    bracket = [p for p in fragment(name, "e", "robot", params=SMALL[name]).bodies[0].parts if p.kind == "box"][-1]
    x, z = 0.05, 0.14 - bracket.pos[2]  # high enough to swing clear, the post beside its bracket, behind what moves
    post = {"kind": "box", "name": "post", "pos": [x + bracket.pos[0], bracket.pos[1] + bracket.half[1] + 0.003, 0.08],
            "half": [0.003, 0.003, 0.06 + 0.002]}
    idea = Idea(name="i", description="d", mechanism=Mechanism(module="self", patch=[
        {"op": "add", "path": "bodies/robot/parts", "value": post}], uses=[
        Use(element=name, prefix="e", on="robot", at=(x, 0.0, z), params=SMALL[name])]))
    built = world.build_idea(world.stock(), idea, "n")
    assert any(b.name.startswith("e_") for b in graph_of(built).bodies)
    assert world.check(built) == [], name


def test_a_bracket_in_the_air_is_floating_and_a_joint_needs_something_to_hold_it():
    world = KitWorld()
    def build(z):
        return world.build_idea(world.stock(), Idea(name="i", description="d", mechanism=Mechanism(module="self", uses=[
            Use(element="lever", prefix="e", on="robot", at=(0.05, 0.0, z), params=SMALL["lever"])])), "n")
    assert world.check(build(0.04)) == []
    assert any("floating" in e for e in world.check(build(0.09)))
    swing = Idea(name="i", description="d", mechanism=Mechanism(module="self", patch=[{"op": "add", "path": "bodies/arm", "value": {
        "name": "arm", "parent": "robot", "pos": [0.0, 0.0, 0.2], "joints": [{"name": "arm_q", "type": "hinge", "axis": [0.0, 1.0, 0.0]}],
        "parts": [{"kind": "box", "name": "armbar", "pos": [0.0, 0.0, 0.0], "half": [0.02, 0.005, 0.005]}]}}]))
    assert any("nothing holds the joint" in e for e in world.check(world.build_idea(world.stock(), swing, "n")))


def test_compose_refuses_a_name_used_twice_and_a_mount_that_is_not_there():
    world = KitWorld()
    use = Use(element="lever", prefix="e", on="robot")
    with pytest.raises(ValueError, match="another prefix"):
        world.build_idea(world.stock(), Idea(name="i", description="d", mechanism=Mechanism(module="self", uses=[use, use])), "n")
    with pytest.raises(ValueError, match="mount on"):
        world.build_idea(world.stock(), Idea(name="i", description="d", mechanism=Mechanism(
            module="self", uses=[use.model_copy(update={"on": "nope"})])), "n")


class _Arms(Patches):
    """The toy world whose kept rods are mechanisms, mounted on a body of its choosing."""

    mount = "base"

    def mechanism_text(self, body):
        part = body.parts["stick"]
        return _arm(part.params["length"], on=self.mount).model_dump_json() if part.type == "built" else ""


def _loop(tmp_path, name, world, length, **cfg):
    p = Scripted([ProposalBatch(proposals=[_rod(length=length)])])
    s = run(RunConfig(world="toy", iterations=1, merge=False, library_dir=str(tmp_path / "lib"), **cfg), tmp_path / name,
            proposer=p, world=world, log=lambda m: None)
    return s, p, [json.loads(x) for x in (tmp_path / name / "iterations.jsonl").read_text().splitlines()]


def test_with_the_switch_off_nothing_is_read_or_written(tmp_path):
    _, p, _ = _loop(tmp_path, "off", _Arms(), 0.28)
    assert not (tmp_path / "lib").exists() and "# Library" not in p.prompts[0]


def test_a_run_enters_what_it_kept_and_a_later_run_is_shown_it_once_proven(tmp_path):
    lib = tmp_path / "lib"
    world = _Arms()
    _, p, recs = _loop(tmp_path, "a", world, 0.28, library=True)
    assert "# Library" in p.prompts[0] and "ratchet" in p.prompts[0]
    kept = next(t for t in recs[0]["trials"] if t["result"].startswith("KEPT"))
    sig = kept["library"][0]
    runs, entries = LB.load(lib)
    assert runs == 1 and entries[sig].instances[0].run == "a" and entries[sig].instances[0].body == recs[0]["accepted"]
    other = _Arms()
    other.mount = "chassis"
    _, later, _ = _loop(tmp_path, "b", other, 0.3, library=True)
    assert sig not in later.prompts[0]  # one context so far
    _, third, _ = _loop(tmp_path, "c", _Arms(), 0.28, library=True)
    assert sig in third.prompts[0]
