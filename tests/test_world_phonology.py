"""The phonology world (sea.worlds.phonology): acoustics, the stock body's limits, a gold body, the bench."""

from __future__ import annotations

import itertools
import time

import numpy as np
import pytest

from sea.evolve.core import Body, Edit, Part, World, apply
from sea.evolve.loop import RunConfig, held_groups, run
from sea.evolve.resources import check
from sea.worlds.phonology import (
    FRICATIVES,
    STOPS,
    SYLLABLES,
    VOWELS,
    PhonologyWorld,
    acoustics,
    formants,
    rig,
)

PHYSICAL = {"f1_floor_too_high", "f1_ceiling_too_low", "f2_floor_too_high", "f2_ceiling_too_low", "f3_floor_too_high",
            "no_closure_at_lips", "no_closure_at_alveolar", "no_closure_at_velar", "no_channel_at_s", "no_channel_at_sh",
            "noise_peak_too_low", "noise_peak_too_high", "transition_too_slow"}


@pytest.fixture(scope="module")
def world() -> PhonologyWorld:
    return PhonologyWorld()


def _train(w: PhonologyWorld, groups: set[str] | None = None):
    return [t for t in w.tasks() if t.split == "train" and (groups is None or t.group in groups)]


def _held(w: PhonologyWorld, body: Body, split: str = "train") -> set[str]:
    tasks = [t for t in w.tasks() if t.split == split]
    return held_groups(tasks, {t.id: w.solve(body, t) for t in tasks})


def _gold(w: PhonologyWorld) -> Body:
    cat = {p.name: p for p in w.catalogue()}
    return apply(w.stock(), [Edit(slot="tongue", type="tip_tongue", params={"rail_start": 3.5, "stroke": 1.3}),
                             Edit(slot="lips", type="round_lips", params={"max_protrusion": 1.6})], cat, "gold")


def test_uniform_tube_is_a_quarter_wave_resonator():
    f = formants(np.full((1, 26), 2.0), np.full((1, 26), 17.5 / 26))[0]
    assert f == pytest.approx([500, 1500, 2500], rel=0.05)
    short = formants(np.full((1, 26), 2.0), np.full((1, 26), 14.0 / 26))[0]
    assert short / f == pytest.approx(np.full(3, 17.5 / 14.0), rel=0.02)  # every formant scales with 1/L


def test_one_evaluation_is_fast(world):
    r = rig(world.stock())
    ctl = {k: np.array([(lo + hi) / 2]) for k, (lo, hi) in r.controls.items()}
    acoustics(r, ctl)
    t0 = time.perf_counter()
    for _ in range(50):
        acoustics(r, ctl)
    assert (time.perf_counter() - t0) / 50 < 0.005


def test_world_contract(world):
    assert isinstance(world, World)
    cat = world.catalogue()
    slots = {p.slot for p in cat}
    assert slots == {"tract", "tongue", "lips"}
    for s in slots:
        types = [p for p in cat if p.slot == s]
        assert len(types) >= 3 and sum(p.stock for p in types) == 1
        assert world.stock().parts[s].type == next(p.name for p in types if p.stock)
    words = {w for _, w, *_ in [*VOWELS.values(), *FRICATIVES.values()]} | {
        f"/{g}/" for g in [*VOWELS, *STOPS, *FRICATIVES, *SYLLABLES]}
    for p in cat:  # a part says what it does, never which sound it is for
        assert p.provides and not any(w in p.provides for w in words)
        assert all(prm.doc for prm in p.params.values())
    tasks = world.tasks()
    assert len({t.group for t in tasks}) == 17 and {t.split for t in tasks} == {"train", "test"}
    assert all(sum(t.split == "train" for t in tasks if t.group == g) == 1 for g in {t.group for t in tasks})


def test_stock_holds_some_groups_and_fails_the_rest_for_physical_reasons(world):
    stock = world.stock()
    tasks = _train(world)
    outs = {t.id: world.solve(stock, t) for t in tasks}
    held = held_groups(tasks, outs)
    assert held == {"iy", "ih", "eh", "ae", "aa", "ah"}  # front, central and open vowels
    mech = {t.group: outs[t.id].mechanism for t in tasks if t.group not in held}
    assert set(mech.values()) <= PHYSICAL, mech
    assert mech["p"] == "no_closure_at_lips" and mech["t"] == "no_closure_at_alveolar"
    assert mech["k"] == "no_closure_at_velar" and mech["er"] == "f3_floor_too_high"
    assert mech["uw"] == "f2_floor_too_high"
    uw = outs["uw_train0"].detail
    assert "target 300/870" in uw and "spans" in uw  # the numbers behind the label
    assert "0.20 cm2" in outs["k_train0"].detail  # the cam's smallest area at the velum


def test_brief_states_facts_not_advice(world):
    text = world.brief(world.stock(), world.tasks())
    for fact in ("13.5 cm glottis to teeth", "F1 350-450: F2", "lowest F3", "closure alveolar", "3 of 6 controller"):
        assert fact in text
    assert not any(w in text.lower() for w in ("swap", "should", "replace", "try ", "recommend"))
    assert len(world.principles()) >= 5


def test_a_gold_body_holds_every_group_within_budget(world):
    stock, gold = world.stock(), _gold(world)
    assert world.check(gold) == []
    res = check(gold, world.bom(gold), stock, world.bom(stock), world.manifest())
    assert res.ok and 0 < res.usd <= world.manifest()["budget"]["usd"]
    groups = {t.group for t in world.tasks()}
    assert _held(world, gold) == groups
    # other noise, targets moved 3%; the syllables sit at the servos' 30 degree edge, and the /ɑ/ moved up 3% takes
    # more than that from either closure (D-072)
    assert _held(world, gold, "test") == groups - set(SYLLABLES)


def test_no_params_only_body_holds_everything(world):
    cat = {p.name: p for p in world.catalogue()}
    stock = world.stock()
    knobs = [(s, k, p) for s, part in stock.parts.items() for k, p in cat[part.type].params.items()]
    tasks = _train(world, {"p", "t", "er"})
    for corner in itertools.product(*[(p.lo, p.hi) for _, _, p in knobs]):  # every extreme of every stock knob
        edits = [Edit(slot=s, params={k: v}) for (s, k, _), v in zip(knobs, corner, strict=True)]
        body = apply(stock, edits, cat, "corner")
        outs = {t.group: world.solve(body, t) for t in tasks}
        assert not any(o.success for o in outs.values()), body.describe()
        assert outs["p"].mechanism == "no_closure_at_lips" and outs["t"].mechanism == "no_closure_at_alveolar"
        assert outs["er"].mechanism == "f3_floor_too_high"
    # ... but parameters are the cheaper rung where they suffice: a full cam stroke seals the velum
    stroke = apply(stock, [Edit(slot="tongue", params={"stroke": 1.3})], cat, "stroke")
    assert _held(world, stroke) - _held(world, stock) == {"k", "ka", "sh"}  # a deeper cam seals the velum


def test_the_longer_cast_is_a_trap(world):
    cat = {p.name: p for p in world.catalogue()}
    long = apply(world.stock(), [Edit(slot="tract", type="long_cast")], cat, "long")
    lost = _held(world, world.stock()) - _held(world, long)
    assert {"eh", "ae", "aa", "ah"} <= lost  # a longer tube lowers the F1 and F2 ceilings too


def test_bom_and_check(world):
    stock = world.stock()
    assert world.check(stock) == []
    assert sum(b.qty for b in world.bom(stock) if b.item == "servo_mg90s") == 3
    cat = {p.name: p for p in world.catalogue()}
    seven = apply(_gold(world), [Edit(slot="tract", type="larynx_tube")], cat, "seven")
    assert any("7 servos" in e and "6 channels" in e for e in world.check(seven))
    pneu = apply(stock, [Edit(slot="tongue", type="pneumatic_tongue")], cat, "pneu")
    res = check(pneu, world.bom(pneu), stock, world.bom(stock), world.manifest())
    assert not res.ok and "cannot be bought" in res.errors[0]
    tip = apply(stock, [Edit(slot="tongue", type="tip_tongue")], cat, "tip")
    res = check(tip, world.bom(tip), stock, world.bom(stock), world.manifest())
    assert res.ok and res.usd == 12 and res.fab_minutes > 0  # one servo beyond the spare, one linkage kit
    bad = Body(parts={**stock.parts, "lips": Part(type="cam_tongue", params={"rail_start": 3, "stroke": 1})})
    assert world.check(bad)


def test_loop_runs_on_the_phonology_world(tmp_path):
    for arm in ("random", "s1"):
        s = run(RunConfig(world="phonology", arm=arm, iterations=2), tmp_path / arm, world=PhonologyWorld(),
                log=lambda m: None)
        assert s["arm"] == arm and s["train_groups"] == 17 and len(s["final_held_train"]) >= 6
        assert s["test"]["n"] == 34 and (tmp_path / arm / "summary.json").exists()
