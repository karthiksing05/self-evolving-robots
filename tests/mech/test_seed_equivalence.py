"""The gripper's jaws, compiled from their mechanism graphs, are the models the hand-written builders made
(D-078): every body, joint, solid, constraint and mass, at the ends and middle of each parameter's range."""

from __future__ import annotations

import json
import sys
from functools import lru_cache
from pathlib import Path

import numpy as np
import pytest

pytestmark = pytest.mark.usefixtures("full_roll")  # D-085: the builders knew no roll limit

sys.path.insert(0, str(Path(__file__).parent))
from capture_jaws import OUT, capture  # noqa: E402

GOLD = json.loads(OUT.read_text())


def _close(a, b, path: str, errs: list[str], tol: float = 1e-6) -> None:
    if isinstance(a, dict):
        if set(a) != set(b):
            errs.append(f"{path}: keys {sorted(set(a) ^ set(b))}")
        for k in set(a) & set(b):
            _close(a[k], b[k], f"{path}.{k}", errs, tol)
    elif isinstance(a, list):
        if len(a) != len(b):
            errs.append(f"{path}: length {len(a)} != {len(b)}")
        else:
            for i, (x, y) in enumerate(zip(a, b, strict=True)):
                _close(x, y, f"{path}[{i}]", errs, tol)
    elif isinstance(a, float) or isinstance(b, float):
        if not np.isclose(a, b, rtol=1e-6, atol=tol, equal_nan=True):
            errs.append(f"{path}: {a} != {b}")
    elif a != b:
        errs.append(f"{path}: {a!r} != {b!r}")


@pytest.mark.parametrize("key", sorted(GOLD))
def test_the_graph_compiles_the_jaw_the_builders_made(key):
    errs: list[str] = []
    _close(GOLD[key], json.loads(json.dumps(_now()[key])), key, errs)
    assert not errs, "\n".join(errs[:20])


@lru_cache(maxsize=1)
def _now() -> dict:
    return capture()  # inside a test, so under ``full_roll``
