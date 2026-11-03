"""The harder sounds (D-072): fricatives filtered by the cavity in front, syllables timed by the servos."""

import numpy as np
import pytest

from sea.evolve.core import Body, Part
from sea.worlds import linkages as K
from sea.worlds import phonology as P

W = P.PhonologyWorld()
TALKER = Body(parts={"tract": Part(type="short_cast", params={"length": 13.5}),
                     "tongue": Part(type="tip_tongue", params={"rail_start": 3.5, "stroke": 1.3}),
                     "lips": Part(type="round_lips", params={"max_protrusion": 1.6})})


def _task(gid: str):
    return next(t for t in W.tasks() if t.id == gid)


def test_the_front_cavity_is_a_quarter_wave_resonator():
    """A narrow channel then a uniform 2 cm2 tube of length L: the hiss peaks near c / 4L (plus the lips' end
    correction), so a front cavity twice as long peaks about half as high."""
    peaks = []
    for front in (1.5, 3.0):
        areas = np.array([[3.0] * 10 + [0.1] + [2.0] * 6])
        lens = np.array([[1.0] * 10 + [0.3] + [front / 6] * 6])
        i = 10
        mask = np.where(np.arange(areas.shape[1]) > i, lens, 0.0)
        peaks.append(P.formants(areas, mask, 1, P.FREQ_HI)[0, 0])
        end = 0.8 * np.sqrt(2.0 / np.pi)  # a flanged opening's end correction
        assert peaks[-1] == pytest.approx(P.C / (4 * (front + end)), rel=0.12)
    assert peaks[0] / peaks[1] == pytest.approx(2.0, rel=0.25)


def test_a_fricative_needs_a_channel_in_place_and_its_hiss_in_band():
    f = _task("s_train0").data
    clo = np.array([[0.10, 0.8, 1.0], [0.03, 0.8, 1.0], [0.10, 2.5, 1.0], [0.10, 0.8, 0.1], [0.10, 0.8, 1.0]])
    peak = np.array([5500.0, 5500.0, 5500.0, 5500.0, 3000.0])
    ok = P._fricative_error(clo, peak, f) <= 1.0
    assert ok.tolist() == [True, False, False, False, False]  # a channel; a seal; out of place; blocked; too low


def test_the_talker_hisses_both_fricatives_and_the_stock_neither():
    for g in ("s", "sh"):
        assert W.solve(TALKER, _task(f"{g}_train0")).success
        assert W.solve(W.stock(), _task(f"{g}_train0")).mechanism == f"no_channel_at_{g}"


def test_a_syllable_is_two_postures_a_servo_turn_apart():
    t = _task("ka_train0")
    o = W.solve(TALKER, t)
    assert o.success and {k[:2] for k in o.controls} == {"c.", "v."}
    r = P.rig(TALKER)
    c = {k[2:]: np.array([v]) for k, v in o.controls.items() if k.startswith("c.")}
    v = {k[2:]: np.array([v]) for k, v in o.controls.items() if k.startswith("v.")}
    turn = np.abs(K.servo_angles(r, TALKER.parts, c)[1] - K.servo_angles(r, TALKER.parts, v)[1]).max()
    assert turn <= P.TURN + np.radians(3)  # noise-free; each attempt adds 1% servo noise to both


def test_a_slow_release_is_heard_as_a_glide(monkeypatch):
    t = _task("ka_train0")
    monkeypatch.setattr(P, "TURN", np.radians(5.0))  # a servo ten times slower: no pair is near enough
    o = W.solve(TALKER, t)
    assert not o.success and o.mechanism == "transition_too_slow"


def test_servo_angles_cover_every_servo_and_stay_on_one_branch():
    r = P.rig(TALKER)
    names, a = K.servo_angles(r, TALKER.parts, {k: np.linspace(lo, hi, 7) for k, (lo, hi) in r.controls.items()})
    assert names == ["rail servo", "cam servo", "tip servo 1", "tip servo 2", "protrusion servo", "drawstring servo"]
    assert len(names) == r.servos and (np.ptp(a, 0) <= K.SPAN + 1e-9).all()
