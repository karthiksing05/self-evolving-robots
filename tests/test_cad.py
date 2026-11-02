"""CAD of a body and its pictures (sea.design.cad, sea.worlds.cad, scripts/render_bodies.py)."""

from __future__ import annotations

import io
import json
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from sea.design.cad import CadModel, View, box, changed, render
from sea.evolve.core import Body, Edit, apply
from sea.worlds.cad import CAD
from sea.worlds.gripper import GripperWorld
from sea.worlds.mobile import MobileWorld
from sea.worlds.phonology import PhonologyWorld

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import render_bodies  # noqa: E402

WORLDS = [GripperWorld(), MobileWorld(), PhonologyWorld()]


def _swap(world, slot: str, type_: str) -> Body:
    cat = {p.name: p for p in world.catalogue()}
    params = {k: (p.lo + p.hi) / 2 for k, p in cat[type_].params.items()}
    return apply(world.stock(), [Edit(slot=slot, type=type_, params=params)], cat, "t")


@pytest.mark.parametrize("world", WORLDS, ids=lambda w: w.name)
def test_every_part_is_a_closed_shape_in_a_slot_of_the_body(world):
    model = CAD[world.name](world.stock())
    assert model.parts and model.views
    for part in model.parts:
        assert part.slot in set(world.stock().parts) | {""}, part.name
        assert len(part.mesh.faces) > 0 and np.isfinite(part.mesh.vertices).all(), part.name


@pytest.mark.parametrize(("world", "slot", "type_"), [(WORLDS[0], "jaw", "prismatic"), (WORLDS[1], "locomotion", "tracks"),
                                                      (WORLDS[2], "tongue", "tip_tongue")], ids=lambda x: str(x)[:12])
def test_a_swapped_part_changes_the_cad_of_its_slot(world, slot, type_):
    def shapes(body: Body) -> dict[str, list[tuple]]:
        out: dict[str, list[tuple]] = {}
        for p in CAD[world.name](body).parts:
            out.setdefault(p.slot, []).append((p.name, len(p.mesh.faces), tuple(np.round(p.mesh.bounds.ravel(), 2))))
        return {k: sorted(v) for k, v in out.items()}

    a, b = shapes(world.stock()), shapes(_swap(world, slot, type_))
    assert a[slot] != b[slot] and set(a) == set(b)  # other slots may move with it: a tub sits over its running gear


def test_slide_finger_does_not_turn_as_it_opens():
    """The prismatic jaw's finger keeps its orientation: at two openings its mesh differs by a shift along x."""
    body = _swap(WORLDS[0], "jaw", "prismatic")
    a, b = (next(p for p in CAD["gripper"](body, opening_mm=w).parts if p.name == "jaw_blade") for w in (40, 80))
    shift = b.mesh.vertices - a.mesh.vertices
    assert np.allclose(shift[:, 1:], 0.0, atol=1e-4) and np.ptp(shift[:, 0]) < 1e-4 and 35 < shift[0, 0] < 45


def test_only_what_a_change_added_or_altered_is_outlined():
    """A tongue swap outlines the tip blade and its linkage, not the rail and carriage both tongues share."""
    world = WORLDS[2]
    a, b = CAD["phonology"](world.stock()), CAD["phonology"](_swap(world, "tongue", "tip_tongue"))
    names = {b.parts[i].name for i in changed(a, b)}
    assert {"tip blade", "tip servo", "pushrod"} <= names
    assert not names & {"rail", "carriage", "cam", "cast tract", "base plate", "upper lip"}
    assert changed(a, a) == set()


def test_render_shares_one_camera_across_models():
    small = CadModel([box("a", "", (0, 0, 0), (10, 10, 10))], [View("front", (0, -1, 0)), View("top", (0, 0, 1), (0, 1, 0))],
                     tile=(80, 60))
    big = CadModel([box("a", "", (0, 0, 0), (40, 10, 10))], small.views, tile=(80, 60))
    pngs = render([small, big])
    ims = [np.asarray(Image.open(io.BytesIO(p)).convert("RGB")) for p in pngs]
    assert all(im.shape == (60, 160, 3) for im in ims)
    drawn = [(np.abs(im.astype(int) - im[0, 0].astype(int)).sum(-1) > 30).sum() for im in ims]
    assert drawn[1] > 2.5 * drawn[0]  # the same scale: the longer box covers more of the picture


def test_render_bodies_records_each_kept_change(tmp_path):
    world = MobileWorld()
    stock = world.stock().model_dump()
    kept = _swap(world, "sensor", "ultrasonic").model_dump() | {"id": "b001"}
    run = tmp_path / "mobile_x"
    run.mkdir()
    trial = {"k": "1", "move": "swap", "result": "KEPT: gained hallway, lost none; costs $2.50, 4 machine min",
             "gained": ["hallway"], "lost": [], "proposal": {"hypothesis": "a sonar sees 2 m ahead"}}
    recs = [{"iteration": 0, "body": stock | {"id": "stock"}, "held": ["a"], "trials": [trial], "accepted": "b001"},
            {"iteration": 1, "body": kept, "held": ["a", "hallway"], "stop": "all groups held"}]
    (run / "iterations.jsonl").write_text("\n".join(json.dumps(r) for r in recs) + "\n")
    (run / "summary.json").write_text(json.dumps({"world": "mobile", "train_groups": 10, "final_body": kept,
                                                  "final_held_train": ["a", "hallway"]}))
    assert render_bodies.main([str(run), "--out", str(tmp_path / "out")]) == 0
    fs = json.loads((tmp_path / "out" / "mobile_x" / "frames.json").read_text())["frames"]
    assert [f["id"] for f in fs] == ["stock", "b001"]
    assert fs[1]["changes"] == ["sensor: bump_switch → ultrasonic"]
    assert fs[1]["why"] == ["a sonar sees 2 m ahead"] and fs[1]["cost"] == "$2.50, 4 machine min" and fs[1]["held"] == 2
    assert all((tmp_path / "out" / "mobile_x" / f["file"]).exists() for f in fs)
