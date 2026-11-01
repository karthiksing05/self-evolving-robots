"""Compose split (spec 7.4; test list 12.2): novel combinations of components seen in train."""

from __future__ import annotations

from sea.contracts import recess_bin, tilt_bin
from sea.sim.panels import combo, generate, held_out, load_config

N_TRAIN, N_COMPOSE, SEED = 200, 60, 1


def _combos(panels):
    return {combo(c, p.tilt_rad) for p in panels for c in p.controls}


def test_every_compose_panel_has_a_combination_absent_from_train():
    train = generate("train", N_TRAIN, SEED)
    compose = generate("compose", N_COMPOSE, SEED)
    seen = _combos(train)
    assert not any(held_out(c) for c in seen)
    for p in compose:
        novel = [c for c in p.controls if combo(c, p.tilt_rad) not in seen]
        assert novel, p.id
        assert held_out(combo(p.controls[0], p.tilt_rad)), p.id  # c0 carries the holdout


def test_every_component_appears_in_train():
    train = generate("train", N_TRAIN, SEED)
    compose = generate("compose", N_COMPOSE, SEED)
    seen = _combos(train)
    parts = [{c[i] for c in seen} for i in range(4)]
    for p in compose:
        for c in p.controls:
            k = combo(c, p.tilt_rad)
            for i in range(4):
                assert k[i] in parts[i], (p.id, c.id, k, i)


def test_every_holdout_rule_is_used_and_bins_are_consistent():
    cfg = load_config()
    compose = generate("compose", N_COMPOSE, SEED)
    rules = cfg["compose_holdouts"]
    used = [0] * len(rules)
    for p in compose:
        c0 = p.controls[0]
        k = combo(c0, p.tilt_rad)
        assert k[2] == recess_bin(c0.recess_depth_mm) and k[3] == tilt_bin(p.tilt_rad)
        for i, r in enumerate(rules):
            if all(r[f] in ("*", v) for f, v in zip(("cls", "handle", "recess", "tilt"), k, strict=True)):
                used[i] += 1
    assert all(n >= N_COMPOSE // len(rules) for n in used), used


def test_generation_is_deterministic():
    a = generate("compose", 6, 3)
    b = generate("compose", 6, 3)
    assert [p.model_dump() for p in a] == [p.model_dump() for p in b]
    assert [p.model_dump() for p in generate("train", 4, 3)] != [p.model_dump() for p in generate("train", 4, 4)]
