"""The D-078 equivalence gates compare the graph with the deleted hand builders, which knew no roll limit."""

from __future__ import annotations

import pytest


@pytest.fixture
def full_roll(monkeypatch):
    """The wrist's whole joint range, as before D-085: the graph-against-builders gates are about the jaw, not the
    roll limit (which ``test_roll_range.py`` checks).  Geometry cached under either range is dropped both ways."""
    from sea.mechanism import gripper as MG
    from sea.worlds import gripper as W

    monkeypatch.setattr(W, "roll_range", lambda fingers: None)
    MG._CACHE.clear()
    yield
    MG._CACHE.clear()
