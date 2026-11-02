"""Resource grounding (D-041): what a body change needs, and whether the bench provides it.

A world's manifest lists the items on hand, the ones that can be bought (price, lead time), and how
fabricated material (``print:PLA``, ``cast:silicone``...) is charged in machine minutes.  ``check``
compares a candidate body's bill of materials with the current body's:

- an item the current body already has is reused, so swapping a part back costs nothing new;
- a shortfall is bought if the item has a price, and refused if it has none ("not provided");
- fabricated lines are charged only for slots whose part changed: a changed part is a new print.

The report says, per item, whether it is on hand, bought, or unavailable, so System 2 is told why a
design was refused and what a different design would cost.  Manifest shape::

    budget: {usd: 30, lead_days: 7, fab_minutes: 600}
    fabrication: {print:PLA: {minutes_per_unit: 1.2, unit: g}}
    items: {steel_rod_3x100: {on_hand: 0, usd: 1.5, lead_days: 2, doc: ...}}
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence
from typing import Any

from pydantic import BaseModel, ConfigDict

from sea.evolve.core import Body, BomLine


class Spent(BaseModel):
    model_config = ConfigDict(extra="forbid")
    usd: float = 0.0
    fab_minutes: float = 0.0
    lead_days: float = 0.0  # the longest wait so far: parts for successive changes are ordered in sequence


class ResourceReport(BaseModel):
    model_config = ConfigDict(extra="forbid")
    ok: bool
    lines: list[str]
    errors: list[str]
    usd: float = 0.0
    lead_days: float = 0.0
    fab_minutes: float = 0.0


def totals(lines: Sequence[BomLine]) -> dict[str, float]:
    out: dict[str, float] = defaultdict(float)
    for ln in lines:
        out[ln.item] += ln.qty
    return out


def is_fab(item: str) -> bool:
    return ":" in item


def check(new: Body, new_bom: Sequence[BomLine], cur: Body, cur_bom: Sequence[BomLine], manifest: dict[str, Any],
          spent: Spent | None = None) -> ResourceReport:
    """Can the bench build ``new`` from ``cur``, within what the run has left to spend?"""
    spent = spent or Spent()
    items: dict[str, dict[str, Any]] = manifest.get("items", {})
    fab: dict[str, dict[str, Any]] = manifest.get("fabrication", {})
    budget: dict[str, float] = manifest.get("budget", {})
    lines: list[str] = []
    errors: list[str] = []
    usd = lead = minutes = 0.0
    need, have = totals([b for b in new_bom if not is_fab(b.item)]), totals([b for b in cur_bom if not is_fab(b.item)])
    for item, qty in sorted(need.items()):
        spec = items.get(item)
        if spec is None:
            errors.append(f"{item}: not in the resource manifest (nothing provides it)")
            continue
        avail = float(spec.get("on_hand", 0)) + have.get(item, 0.0)
        short = max(0.0, qty - avail)
        if short <= 1e-9:
            lines.append(f"{item} x{qty:g}: provided ({'reused' if have.get(item) else 'on hand'})")
            continue
        if spec.get("usd") is None:
            errors.append(f"{item} x{qty:g}: only {avail:g} provided and it cannot be bought")
            continue
        cost = short * float(spec["usd"])
        usd += cost
        lead = max(lead, float(spec.get("lead_days", 0)))
        lines.append(f"{item} x{qty:g}: {avail:g} provided, buy {short:g} at ${float(spec['usd']):g} "
                     f"(${cost:.2f}, {float(spec.get('lead_days', 0)):g} days)")
    changed = {s for s in set(new.parts) | set(cur.parts) if new.parts.get(s) != cur.parts.get(s)}
    for ln in new_bom:
        if not is_fab(ln.item) or ln.slot not in changed:
            continue
        spec = fab.get(ln.item)
        if spec is None:
            errors.append(f"{ln.item}: no machine for it on this bench")
            continue
        m = ln.qty * float(spec.get("minutes_per_unit", 0.0))
        minutes += m
        lines.append(f"{ln.item} {ln.qty:.1f} {spec.get('unit', '')} for {ln.slot}: {m:.0f} machine minutes")
    if usd and spent.usd + usd > float(budget.get("usd", 0.0)) + 1e-9:
        errors.append(f"buying costs ${usd:.2f}; the run has ${float(budget.get('usd', 0.0)) - spent.usd:.2f} left")
    if lead and lead > float(budget.get("lead_days", 0.0)) + 1e-9:
        errors.append(f"parts take {lead:g} days to arrive; the run allows {float(budget.get('lead_days', 0.0)):g}")
    if minutes and spent.fab_minutes + minutes > float(budget.get("fab_minutes", float("inf"))) + 1e-9:
        errors.append(f"fabrication takes {minutes:.0f} min; the run has "
                      f"{float(budget.get('fab_minutes', 0.0)) - spent.fab_minutes:.0f} left")
    return ResourceReport(ok=not errors, lines=lines, errors=errors, usd=round(usd, 2), lead_days=lead,
                          fab_minutes=round(minutes, 1))
