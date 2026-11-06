"""A printed part's stress, solved (D-124), with beam theory as the check."""

from __future__ import annotations

import math

import pytest

from sea.sim import solid_fe as F


def _rect_i(w: float, h: float) -> float:
    return w * h**3 / 12


@pytest.mark.parametrize(("length", "w", "h"), [(0.09, 0.006, 0.004), (0.05, 0.01, 0.004), (0.1, 0.003, 0.0095)])
def test_a_slender_part_agrees_with_beam_theory(length, w, h):
    stress, move = F.Prism(length, w, h).bend()
    assert stress == pytest.approx(length * (h / 2) / _rect_i(w, h), rel=0.04)
    assert move == pytest.approx(length**3 / (3 * F.E * _rect_i(w, h)), rel=0.03)


def test_a_stubby_part_bends_more_than_beam_theory_says_from_shear():
    move = F.Prism(0.01, 0.008, 0.008).bend()[1]
    assert move > 1.3 * 0.01**3 / (3 * F.E * _rect_i(0.008, 0.008))


def test_a_slender_part_buckles_at_euler():
    assert F.buckle(0.09, 0.006, 0.004) == pytest.approx(math.pi**2 * F.E * _rect_i(0.006, 0.004) / 0.09**2, rel=0.05)


def test_it_buckles_the_thin_way_whichever_way_it_is_named():
    assert F.buckle(0.09, 0.006, 0.004) == pytest.approx(F.buckle(0.09, 0.004, 0.006), rel=1e-6)


def test_a_stubby_part_buckles_well_below_euler():
    assert F.buckle(0.01, 0.008, 0.008) < 0.6 * math.pi**2 * F.E * _rect_i(0.008, 0.008) / 0.01**2


def test_a_length_between_millimetres_is_carried():
    near = F.bend(0.03, 0.006, 0.004)
    at = F.bend_near(0.0304, 0.006, 0.004)
    assert at[0] == pytest.approx(near[0] * 0.0304 / 0.03) and at[1] == pytest.approx(near[1] * (0.0304 / 0.03) ** 3)


def test_nothing_of_no_length_carries_nothing():
    assert F.bend(0.0, 0.006, 0.004) == (0.0, 0.0) and F.buckle(0.0, 0.006, 0.004) == math.inf
