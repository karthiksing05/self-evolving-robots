"""The mechanism laws, reading the jaw's graph, say what they said reading name tables (D-078)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.usefixtures("full_roll")  # D-085: the builders knew no roll limit

sys.path.insert(0, str(Path(__file__).parent))
from capture_laws import OUT, bodies  # noqa: E402

from sea.mechanism import gripper as MG  # noqa: E402

GOLD = json.loads(OUT.read_text())
BODIES = dict(bodies())


@pytest.mark.parametrize("key", sorted(GOLD))
def test_the_laws_read_from_the_graph_as_they_read_from_names(key):
    g = MG.geometry(BODIES[key])
    now = {"laws": MG.laws(BODIES[key]), "fastened": sorted(g["fastened"]), "loose": sorted(g["loose"]),
           "clashes": sorted(g["clashes"]), "carried": {k: v["worst_gap_mm"] for k, v in g["carried"].items()}}
    assert now == GOLD[key]


def test_a_loops_transmission_is_measured_where_its_links_meet():
    """The parallelogram's angle at its loop pin is the old formula's: links against the line through the fixed pins."""
    import numpy as np

    from sea.sim import jaws

    g = MG.geometry(BODIES['parallelogram {"link_mm": 70.0}'])
    lo, hi = g["jaw_range_servo_rad"]
    assert g["loops_deg"]["jaw_follower / jaw_coupler"] == pytest.approx(
        min(jaws.transmission_deg(q) for q in np.linspace(lo, hi, 50)), abs=0.1)
