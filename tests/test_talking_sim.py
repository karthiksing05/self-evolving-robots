"""The talking robot simulated (D-121, D-122): air stepped in time, and a tube the robot has to press on."""

from __future__ import annotations

import numpy as np
import pytest

from sea.evolve.core import Part
from sea.evolve.loop import make_world
from sea.mech import graph as G
from sea.mech.world import SLOT
from sea.sim import soft_tract as ST
from sea.sim import tract_sound as T
from sea.worlds import phonology as P

L = np.r_[np.full(24, 13.5 / 24), [1.0, 1.0]][None]


def _narrow(area: float, at: int) -> np.ndarray:
    a = np.full(26, 4.5)
    a[at : at + 2], a[24:] = area, 4.0
    return a[None]


def _lever(arm_cm: float, half_y_cm: float = 1.8, palate: bool = False, second: bool = False) -> G.Graph:
    """A pad on a lever from below the tube."""
    arm, at = arm_cm / 100, 0.0675
    hinge = at - arm
    parts = [G.Box(name="base", pos=(0.04, 0.0, -0.08), half=(0.11, 0.05, 0.003)),
             G.Box(name="post", pos=(hinge, 0.0, -0.05), half=(0.003, 0.003, 0.03))]
    if palate:
        parts.append(G.Box(name="palate", pos=(at, 0.0, 0.0165), half=(0.03, 0.02, 0.002)))
    bodies = [G.Body(name="frame", parent="world", parts=tuple(parts)),
              G.Body(name="glottis", parent="frame", pos=(0.135, 0.0, 0.0)),
              G.Body(name="lip_end", parent="frame", pos=(-0.01, 0.0, 0.0)),
              G.Body(name="lev", parent="frame", pos=(hinge, 0.0, -0.02),
                     joints=(G.Joint(name="lev_q", type="hinge", axis=(0.0, 1.0, 0.0), range=(-1.2, 0.05)),),
                     parts=(G.Box(name="lev_bar", pos=(arm / 2, 0.0, 0.0), half=(arm / 2, 0.003, 0.002)),
                            G.Box(name="pad", pos=(arm, 0.0, 0.0), half=(0.008, half_y_cm / 100, 0.002))))]
    actuators = [G.Servo(name="lev", joint="lev_q", stall_nm=0.216)]
    if second:  # another moving part, above the first's way, joined to it by nothing
        bodies.append(G.Body(name="top", parent="frame", pos=(at, 0.0, 0.0165),
                             joints=(G.Joint(name="top_q", type="slide", axis=(0.0, 0.0, 1.0), range=(0.0, 1e-4)),),
                             parts=(G.Box(name="top_pad", pos=(0.0, 0.0, 0.0), half=(0.03, 0.02, 0.002)),)))
        actuators.append(G.Servo(name="top", joint="top_q", stall_nm=50.0))
    return G.Graph(name="lever", bodies=tuple(bodies), excludes=(("frame", "lev"),), actuators=tuple(actuators))


@pytest.mark.parametrize("area", [np.full(26, 4.5)[None], _narrow(0.4, 16), np.r_[np.full(10, 1.0), np.full(14, 6.0), [6.0, 6.0]][None],
                                  np.r_[np.full(10, 8.0), np.full(8, 0.6), np.full(6, 3.0), [2.0, 2.0]][None]])
def test_air_stepped_in_time_has_the_formants_of_the_transfer_function(area):
    assert T.formants(area, L)[0] == pytest.approx(P.formants(area, L)[0], rel=0.03)


def test_a_long_tube_keeps_its_own_length():
    assert T.formants(np.full(26, 4.5)[None], L * 1.3)[0] == pytest.approx(P.formants(np.full(26, 4.5)[None], L * 1.3)[0], rel=0.03)


@pytest.mark.parametrize(("at", "area"), [(21, 0.1), (16, 0.1), (12, 0.1), (21, 0.3)])
def test_a_hiss_peaks_where_the_cavity_in_front_of_it_resonates(at, area):
    a = _narrow(area, at)
    front = np.where(np.arange(26) > np.argmin(a[0]), L[0], 0.0)[None]
    assert T.hiss(T.Tube(a, L))[0] == pytest.approx(P.formants(np.maximum(a, 0.08), front, 1, P.FREQ_HI)[0, 0], rel=0.10)


def test_a_seal_holds_the_lungs_and_passes_nothing():
    shut = T.blow(T.Tube(_narrow(0.0, 21), L), seconds=0.03)
    assert abs(shut["flow"][0]) < 1.0 and shut["behind"][0] > 0.95 * T.PS


def test_flow_through_a_slit_is_a_turbulent_jet_of_about_what_bernoulli_says():
    air = T.blow(T.Tube(_narrow(0.1, 21), L), seconds=0.03)
    orifice = 0.1 * np.sqrt(2 * T.PS / T.RHO)  # lungs to air through 0.1 cm2, with the glottis's share ignored
    assert 0.5 * orifice < air["flow"][0] < orifice and air["re"][0] > T.RE_C


def test_a_tract_that_opens_bursts():
    frames = np.stack([_narrow(0.0 if t < 20 else 4.5, 21)[0] for t in range(40)])
    air = T.speak(frames, L[0], 0.002)
    i = int(0.04 / float(air["dt"]))
    assert np.abs(air["lips"][: i - 5]).max() < 50.0 < np.abs(air["lips"][i:]).max()


def test_a_tube_at_rest_is_round():
    areas = ST.SoftTract(_lever(4.0))({})[0][0]
    assert areas[:22] == pytest.approx(np.pi * ST.BORE**2, rel=0.01)


def test_a_pad_below_the_axis_leaves_most_of_the_air():
    w = make_world("phonology_kit")
    r = w.rig_of(w.stock())
    areas = r.areas({k: np.array([v]) for k, v in {"arm": 0.0, "lever": 0.0, "jaw": 0.0}.items()})[0]
    assert areas.min() > 2.0  # the concentric bore read this as 0.19 cm2


def test_a_lever_presses_the_wall_shut_and_the_wall_presses_back():
    st = ST.SoftTract(_lever(4.0))
    open_, shut = st({"lev": np.array([0.0])}), st({"lev": np.array([-1.0])})
    assert open_[0].min() > 4.0 and shut[0].min() < 0.05 and st.load[0] > 0.05


def test_a_long_lever_stalls_short_of_its_command():
    st = ST.SoftTract(_lever(8.0))
    st({"lev": np.array([-0.2])})
    assert st.short[0] < np.radians(2.0)
    st({"lev": np.array([-1.0])})
    assert st.load[0] == pytest.approx(0.216, abs=1e-3) and st.short[0] > np.radians(20.0)


def test_a_solid_stops_a_solid():
    free = ST.SoftTract(_lever(4.0, palate=False))
    braced = ST.SoftTract(_lever(4.0, palate=True))
    free({"lev": np.array([-1.2])})
    braced({"lev": np.array([-1.2])})
    assert braced.short[0] > free.short[0] + np.radians(5.0)


def test_a_part_stops_a_part_it_is_not_joined_to():
    free, blocked = ST.SoftTract(_lever(4.0)), ST.SoftTract(_lever(4.0, second=True))
    free({"lev": np.array([-1.2])})
    blocked({"lev": np.array([-1.2]), "top": np.array([0.0])})
    assert blocked.short[0] > free.short[0] + np.radians(5.0)


def test_a_narrow_pad_leaves_air_either_side_of_it():
    wide, narrow = ST.SoftTract(_lever(4.0, 1.8)), ST.SoftTract(_lever(4.0, 0.2))
    q = {"lev": np.array([-0.45])}
    assert narrow(q)[0].min() > wide(q)[0].min()


def test_the_same_command_reads_the_same_and_is_remembered():
    st = ST.SoftTract(_lever(4.0))
    q = {"lev": np.array([-0.5])}
    first = st(q)[0].copy()
    n = st.d.time
    assert (st(q)[0] == first).all() and st.d.time == n


def test_a_kit_talker_is_told_of_a_stall_and_of_what_it_held():
    w = make_world("phonology_kit")
    body = w.stock().model_copy(update={"parts": {SLOT: Part(type="built", mechanism=_lever(8.0).model_dump_json())}})
    r = w.rig_of(body)
    d = {"kind": "stop", "place": "velar"}
    note = w._felt(r, {"lev": np.array([-1.0])}, d, 0.5)
    assert "stalls" in note[1] and "lungs' pressure" in note[1] and note[0] < 1.0


def test_a_syllable_is_driven_whole_with_air_flowing():
    w = make_world("phonology_kit")
    body = w.stock().model_copy(update={"parts": {SLOT: Part(type="built", mechanism=_lever(4.0).model_dump_json())}})
    timed, how = w._timed(w.rig_of(body), body, {"lev": np.array([-0.9])}, {"lev": np.array([-0.1])}, 0.0)
    assert 0.0 < timed < 3.0 and "bursting" in how


def test_a_failing_syllable_says_it_is_only_a_guide():
    w = make_world("phonology_kit")
    body = w.stock().model_copy(update={"parts": {SLOT: Part(type="built", mechanism=_lever(4.0).model_dump_json())}})
    assert "would take" in w._timed(w.rig_of(body), body, {"lev": np.array([-0.9])}, {"lev": np.array([-0.1])}, 2.0)[1]


def test_the_kit_world_says_what_it_judges():
    w = make_world("phonology_kit")
    tasks = {t.group: t for t in w.tasks()}
    assert "lungs' pressure" in tasks["t"].facts and "as driven" in tasks["ta"].facts and "no servo turning" not in tasks["ta"].facts
    assert any("silicone" in r for r in w.brief_rules())
