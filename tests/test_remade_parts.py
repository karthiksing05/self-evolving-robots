"""A change pays the machine for the parts it remakes, not for the whole body (D-125)."""

from __future__ import annotations

from sea.evolve.core import Body, BomLine, Part
from sea.evolve.resources import check
from sea.mech.build import printed_lines
from sea.mech.world import graph_of
from sea.worlds import phonology_kit as K

MANIFEST = {"budget": {"fab_minutes": 100.0}, "fabrication": {"print:PLA": {"minutes_per_unit": 2.0, "unit": "g"}}}


def _body(tag: str) -> Body:
    return Body(parts={"robot": Part(type="rigid", params={"tag": 1.0 if tag == "a" else 2.0})})


def _lines(*pieces: tuple[str, float]) -> list[BomLine]:
    return [BomLine(item="print:PLA", qty=q, slot="robot", piece=p) for p, q in pieces]


def test_only_the_changed_piece_is_charged():
    r = check(_body("b"), _lines(("x:1", 10.0), ("y:2", 5.0)), _body("a"), _lines(("x:1", 10.0), ("y:1", 5.0)), MANIFEST)
    assert r.ok and r.fab_minutes == 10.0
    assert any("reused" in ln for ln in r.lines)


def test_a_line_that_names_no_piece_is_made_again():
    r = check(_body("b"), [BomLine(item="print:PLA", qty=10.0, slot="robot")],
              _body("a"), [BomLine(item="print:PLA", qty=10.0, slot="robot")], MANIFEST)
    assert r.fab_minutes == 20.0


def test_a_new_body_over_the_machine_time_is_still_refused():
    r = check(_body("b"), _lines(("x:2", 60.0)), _body("a"), _lines(("x:1", 60.0)), MANIFEST)
    assert not r.ok and "fabrication takes 120" in r.errors[0]


def test_the_talker_recasts_its_tube_only_when_its_length_changes():
    w = K.PhonologyKit()
    stock = w.stock()
    longer = Body(parts={K.SLOT: Part(type="built", mechanism=w.module("talker", {"length_cm": 15.0, "place_cm": 6.75}).model_dump_json())})
    a, b = w.bom(stock), w.bom(longer)
    tube = lambda bom: next(ln.piece for ln in bom if ln.item == "cast:silicone")  # noqa: E731
    assert tube(a) != tube(b) and tube(a) == tube(w.bom(stock))
    assert sum(ln.qty for ln in a if ln.item == "print:PLA") > 0
    res = check(longer, b, stock, a, w.manifest())
    full = sum(ln.qty for ln in b if ln.item == "print:PLA") * w.manifest()["fabrication"]["print:PLA"]["minutes_per_unit"]
    assert res.fab_minutes < full + 2.5 * sum(ln.qty for ln in b if ln.item == "cast:silicone")


def test_the_same_graph_prints_the_same_pieces():
    w = K.PhonologyKit()
    g = graph_of(w.stock())
    assert printed_lines(g, "s") == printed_lines(g, "s")
    assert len({ln.piece for ln in printed_lines(g, "s")}) == len(printed_lines(g, "s"))
