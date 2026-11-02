"""Stock gripper capability table (docs/gripper-experiment.md V1): the golden table and its instrument."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from sea.contracts import Design
from sea.sim.build_model import JAW_FRICTION, build_model, pair_friction
from sea.sim.grasp_probe import PROBE, pick, scene_object

GOLDEN = Path(__file__).resolve().parent / "golden" / "stock_grasp.json"


@pytest.fixture(scope="module")
def table() -> dict:
    return json.loads(GOLDEN.read_text())


def _held(table: dict, shape: str, width: float, mass: float = 50.0, mu: float = 0.8, past: float | None = None) -> bool:
    rows = [r for r in table["picks"] if (r["shape"], r["width_mm"], r["mass_g"], r["friction"]) == (shape, width, mass, mu)
            and r.get("sphere_past_mm") == past]
    assert len(rows) == 1, (shape, width, mass, mu, past)
    return rows[0]["held"]


def test_the_table_supports_the_scenario(table):
    # Phase A's object set and phase B's incompatible objects (docs/gripper-experiment.md section 1).
    # With friction that does not creep (D-048), the splayed stock faces hold cans to 55 mm and boxes to 60 mm.
    assert all(_held(table, "cylinder", w) for w in (25, 30, 35, 40, 45, 50, 55))
    assert not any(_held(table, "cylinder", w) for w in (60, 70))
    assert all(_held(table, "box", w) for w in (20, 30, 40, 50, 60))
    # The 50 mm can holds at every friction tried: before D-048 it failed at every friction alike, the mark
    # of soft-contact creep rather than of a friction cone.
    assert all(_held(table, "cylinder", 50, mu=mu) for mu in (0.3, 0.8, 1.5))
    # Mass: the 30 mm can holds to 800 g; at 1200 g the shoulder stalls and the payload reading saturates.
    assert _held(table, "cylinder", 30, 800) and _held(table, "cylinder", 30, 800, mu=0.3)
    heavy = next(r for r in table["picks"] if r["mass_g"] == 1200)
    assert not heavy["held"] and heavy["payload_g"] < 900 and min(heavy["pads_n"]) > 10
    # Spheres slip out of a grip at the equator and hold only when the tips reach well below it,
    # whatever the friction: a profile problem (V-groove or concave pads), not a pad-material one.
    assert not any(_held(table, "sphere", w, past=5.0) for w in (20, 30, 40, 50))
    assert all(_held(table, "sphere", 40, mu=mu, past=12.0) for mu in (0.3, 0.8, 1.5))
    assert table["envelope"]["cylinder_widest_held_mm"] == 55.0


def test_held_picks_read_like_a_held_object(table):
    for r in table["picks"]:
        if r["held"]:
            assert min(r["pads_n"]) > 5 and r["payload_g"] == pytest.approx(r["mass_g"], abs=40), r
        elif r["mass_g"] < 1000:
            assert max(r["pads_n"]) < 1 and abs(r["payload_g"]) < 25, r  # dropped: empty jaws, no payload


@pytest.mark.parametrize("shape,width,past", [("cylinder", 30, None), ("cylinder", 60, None), ("sphere", 40, 12.0)])
def test_the_probe_reproduces_the_table(table, shape, width, past):
    probe = replace(PROBE, sphere_past_mm=past) if past is not None else PROBE
    got = pick(scene_object(shape, width, 50.0, 0.8, probe), probe=probe)
    want = next(r for r in table["picks"] if (r["shape"], r["width_mm"], r["mass_g"], r["friction"]) ==
                (shape, width, 50.0, 0.8) and r.get("sphere_past_mm") == past)
    assert got.held == want["held"]
    assert got.hold_mm == pytest.approx(want["hold_mm"], abs=5) and got.jaw_q == pytest.approx(want["jaw_q"], abs=0.02)


def test_jaw_contacts_mix_both_surfaces_and_panels_are_untouched():
    sim = build_model(Design(id="stock"), objects=[scene_object("cylinder", 30, 50, 0.3)], table_z=0.0)
    m = sim.model
    jaws = [g for g, k in sim.geom_kind.items() if k.startswith("jaw")]
    assert m.npair == len(jaws) > 0
    assert np.allclose(m.pair_friction[:, 0], pair_friction(JAW_FRICTION, 0.3))
    plain = build_model(Design(id="stock"))
    assert plain.model.npair == 0 and {int(plain.model.geom_conaffinity[g]) for g in plain.geom_kind} == {1}
