"""The talking robot's drives (D-071): each law refuses its planted fault and passes its twin."""

import math

import numpy as np
import pytest

from sea.evolve.core import Body, Part
from sea.worlds import linkages as K
from sea.worlds.phonology import PhonologyWorld, rig

W = PhonologyWorld()


def _body(tract=("short_cast", {"length": 13.5}), tongue=("tip_tongue", {"rail_start": 3.25, "stroke": 1.3}),
          lips=("round_lips", {"max_protrusion": 1.6})) -> Body:
    return Body(parts={"tract": Part(type=tract[0], params=tract[1]), "tongue": Part(type=tongue[0], params=tongue[1]),
                       "lips": Part(type=lips[0], params=lips[1])})


def test_the_stock_and_the_evolved_talker_pass():
    assert W.check(W.stock()) == []
    assert W.check(_body()) == []  # the c1 campaign's final body on every System 2 seed


def test_a_rail_whose_sled_reaches_the_glottis_is_refused():
    short = _body(("short_cast", {"length": 12.5}), ("long_rail_tongue", {"rail_start": 2.5, "stroke": 1.0}),
                  ("jaw_lips", {"rest_area": 3.0}))
    assert any(e.startswith("tongue rail:") and "glottis" in e for e in W.check(short))
    long = _body(("long_cast", {"length": 15.5}), ("long_rail_tongue", {"rail_start": 2.5, "stroke": 1.0}),
                 ("jaw_lips", {"rest_area": 3.0}))
    assert W.check(long) == []
    # the larynx slide's carriage stands 3 mm further forward than a fixed cast's reed housing
    lar = _body(("larynx_tube", {"length": 12.5, "travel": 2.0}), ("cam_tongue", {"rail_start": 3.0, "stroke": 1.0}))
    fixed = _body(("short_cast", {"length": 12.5}), ("cam_tongue", {"rail_start": 3.0, "stroke": 1.0}))
    assert any("glottis" in e for e in W.check(lar)) and W.check(fixed) == []


def test_the_rail_crank_reaches_both_ends_of_the_rail():
    for kind, (travel, _) in K.RAIL_MM.items():
        for start in np.arange(2.5, 4.001, 0.05):
            c = K.rail_crank(start, kind)
            lo, hi = c.span()
            assert lo <= K.sled_pin(start + travel / 10) and K.sled_pin(start) <= hi
    # the old placement rounded the servo onto the grid away from the rail's back end, short of it
    good = K.rail_crank(3.25, "cam_tongue")
    bad = K.Crank(good.x_c + 8.0, good.y_c, good.y_pin, good.a, good.b)
    xs = np.array([K.sled_pin(p) for p in np.linspace(3.25, 10.25, 21)])
    assert K._crank_faults("rail", good, xs, "the sled") == []
    assert "the crank reaches" in K._crank_faults("rail", bad, xs, "the sled")[0]


def test_a_rod_that_meets_its_slide_square_on_is_refused():
    """A pin sliding along x driven by a rod standing upright transmits nothing along the slide."""
    pins = [np.array([x, 0.0]) for x in np.linspace(-5, 5, 11)]
    upright = K.Dyad((0.0, -40.0), 10.0, 40.0, 1.0)
    along = K.Dyad((-40.0, -5.0), 10.0, 38.0, 1.0)
    normal = np.array([0.0, 1.0])
    assert any("transmission angle" in e for e in K._dyad_faults("rod", upright, pins, normal, ["x"] * 11))
    assert K._dyad_faults("rod", along, pins, normal, ["x"] * 11) == []


def test_a_horn_that_cannot_reach_or_turns_too_far_is_refused():
    pins = [np.array([x, 0.0]) for x in np.linspace(-20, 20, 11)]
    short = K.Dyad((0.0, -60.0), 5.0, 40.0, 1.0)  # 40 + 5 mm cannot span 60 mm
    assert "cannot reach" in K._dyad_faults("rod", short, pins, None, ["x"] * 11)[0]
    wide = [np.array([12 * math.cos(t), 12 * math.sin(t)]) for t in np.linspace(0, 1.5 * math.pi, 31)]
    spin = K.Dyad((0.0, 0.0), 6.0, 7.0, 1.0)  # its pin circles the servo: the horn turns 270 degrees
    assert any("turns" in e for e in K._dyad_faults("rod", spin, wide, None, ["x"] * 31))


def test_the_drawstring_spool_and_the_larynx_pinion_cover_their_travel(monkeypatch):
    lips = _body(("short_cast", {"length": 13.5}), ("cam_tongue", {"rail_start": 3.25, "stroke": 0.95}))
    lar = _body(("larynx_tube", {"length": 14.0, "travel": 3.0}), ("cam_tongue", {"rail_start": 3.25, "stroke": 0.95}),
                ("jaw_lips", {"rest_area": 3.0}))
    assert W.check(lips) == [] and W.check(lar) == []
    assert K.draw_needed(-0.2, 1.45) == pytest.approx(2 * math.pi * 14.5)  # the ring's circumference, open to shut
    monkeypatch.setattr(K, "SPOOL_R", 12.0 / 2)  # the old 12 mm horn's reach, as a spool
    assert any(e.startswith("drawstring:") for e in W.check(lips))
    monkeypatch.setattr(K, "PINION_R", 8.0 / 2)  # half a turn of an 8 mm lead screw is 4 mm
    assert any(e.startswith("larynx slide:") for e in W.check(lar))


def test_the_drives_cover_every_corner_of_the_catalogue_but_the_rail():
    """Every law but the rail's holds at the catalogue's corners: the drives are built for their parts' ranges."""
    corners = []
    for tract in (("short_cast", {"length": 14.5}), ("long_cast", {"length": 17.5}),
                  ("larynx_tube", {"length": 14.5, "travel": 3.0})):
        for tongue in ("cam_tongue", "tip_tongue"):
            for start, stroke in ((2.5, 0.6), (2.5, 1.3)):
                for lips in (("jaw_lips", {"rest_area": 2.5}), ("spread_lips", {"rest_area": 3.5}),
                             ("round_lips", {"max_protrusion": 0.6}), ("round_lips", {"max_protrusion": 1.6})):
                    corners.append(_body(tract, (tongue, {"rail_start": start, "stroke": stroke}), lips))
    for b in corners:
        assert K.faults(rig(b), b.parts) == [], b


def test_the_tip_blade_cannot_press_through_the_tongue_pad():
    """The blade presses from the palate and the pad from the floor: where both narrow one section, the posture is one
    the parts cannot take, and it fails however it sounds."""
    from sea.worlds import phonology as P

    r = rig(_body())
    both = {"tongue_place": np.array([3.0]), "tongue_height": np.array([1.3]), "tip_place": np.array([3.0]),
            "tip_height": np.array([1.3]), "lip_radius": np.array([1.0]), "protrusion": np.array([0.0])}
    apart = both | {"tip_place": np.array([0.5]), "tongue_place": np.array([8.0])}
    over, at = P.clash(r, both)
    assert over[0] > 1.0 and at[0] == pytest.approx(3.0, abs=0.3)  # both seal the bore: 2.4 cm of crossing
    assert P.clash(r, apart)[0][0] <= P.CLASH
    assert P.clash(rig(W.stock()), {k: v for k, v in both.items() if k in rig(W.stock()).controls} | {
        "jaw": np.array([0.5])})[0][0] == 0.0  # one articulator: nothing to clash with
    task = next(t for t in W.tasks() if t.id == "t_train0")
    f, clo = P.acoustics(r, both)
    assert P.PhonologyWorld._errors(task, f, clo, over)[0] > 1.0  # a closure, but not one the parts can make
    bab = W.babble(_body())
    assert 0.05 < (bab.clash > P.CLASH).mean() < 0.5  # System 1's map knows which postures are impossible
