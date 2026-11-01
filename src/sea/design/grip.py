"""What the gripper can hold (docs/gripper-experiment.md, finger family rule 3: capability is computed).

For the stock jaws the numbers are measured by the V1 capability table (tests/golden/stock_grasp.json,
D-032); the finger family will compute them from its parameters.  The typecheck's aperture rule
reads them, so an object wider than the gripper's widest stable grip is ``no_applicable_primitive``
(structural) before any simulation, and System 2 sees the reason.
"""

from __future__ import annotations

from sea.contracts import Design, FingerSpec

# Widest object held 30 mm up after a 1 s hold, at 50 g and friction 0.8 (V1).  Spheres hold only with
# the jaw tips about 12 mm below the equator; System 1 has to find that grip.
STOCK_WIDEST_MM = {"cylinder": 45.0, "box": 50.0, "sphere": 40.0}


def widest_mm(design: Design, shape: str) -> float:
    if design.fingers != FingerSpec():  # pragma: no cover - L1 proposals are rejected until the finger family
        raise NotImplementedError("finger-family capability is not computed yet")
    return STOCK_WIDEST_MM[shape]
