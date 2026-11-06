"""Resource grounding (D-041): what a body change needs, and whether the bench provides it.

A world's manifest lists the items on hand, the ones that can be bought (price, lead time), and how
fabricated material (``print:PLA``, ``cast:silicone``...) is charged in machine minutes.  ``check``
compares a candidate body's bill of materials with the current body's:

- an item the current body already has is reused, so swapping a part back costs nothing new;
- the bench's pool of an item is what the run owns (the stock body's parts plus everything bought, in
  ``Spent.owned``) and the spares on hand that no kept change has used, so a spare is used only once;
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
    # items the run owns besides the manifest's spares: the stock body's, then every purchase; None when a
    # single change is checked on its own (the current body's items are then the run's)
    owned: dict[str, float] | None = None


def start(stock_bom: Sequence[BomLine]) -> Spent:
    """What a run owns before it changes anything: the stock body's items."""
    return Spent(owned=dict(totals([b for b in stock_bom if not is_fab(b.item)])))


def charge(spent: Spent, res: ResourceReport) -> Spent:
    """``spent`` after building a kept change: its dollars, machine time and wait, and the items it bought."""
    owned = None if spent.owned is None else {k: spent.owned.get(k, 0.0) + res.buy.get(k, 0.0)
                                              for k in {*spent.owned, *res.buy}}
    return Spent(usd=spent.usd + res.usd, fab_minutes=spent.fab_minutes + res.fab_minutes,
                 lead_days=max(spent.lead_days, res.lead_days), owned=owned)


class ResourceReport(BaseModel):
    model_config = ConfigDict(extra="forbid")
    ok: bool
    lines: list[str]
    errors: list[str]
    usd: float = 0.0
    lead_days: float = 0.0
    fab_minutes: float = 0.0
    buy: dict[str, float] = {}  # item -> quantity bought


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
    buy: dict[str, float] = {}
    need, have = totals([b for b in new_bom if not is_fab(b.item)]), totals([b for b in cur_bom if not is_fab(b.item)])
    for item, qty in sorted(need.items()):
        spec = items.get(item)
        if spec is None:
            errors.append(f"{item}: not in the resource manifest (nothing provides it)")
            continue
        owned = have if spent.owned is None else spent.owned
        avail = float(spec.get("on_hand", 0)) + owned.get(item, 0.0)
        short = max(0.0, qty - avail)
        if short <= 1e-9:
            lines.append(f"{item} x{qty:g}: provided ({'reused' if have.get(item) else 'on hand'})")
            continue
        if spec.get("usd") is None:
            errors.append(f"{item} x{qty:g}: only {avail:g} provided and it cannot be bought")
            continue
        cost = short * float(spec["usd"])
        usd += cost
        buy[item] = short
        lead = max(lead, float(spec.get("lead_days", 0)))
        lines.append(f"{item} x{qty:g}: {avail:g} provided, buy {short:g} at ${float(spec['usd']):g} "
                     f"(${cost:.2f}, {float(spec.get('lead_days', 0)):g} days)")
    changed = {s for s in set(new.parts) | set(cur.parts) if new.parts.get(s) != cur.parts.get(s)}
    made = {(b.slot, b.item, b.piece) for b in cur_bom if b.piece}
    remade: dict[tuple[str, str], list[float]] = {}
    for ln in new_bom:
        if not is_fab(ln.item) or ln.slot not in changed:
            continue
        row = remade.setdefault((ln.slot, ln.item), [0.0, 0.0])
        if (ln.slot, ln.item, ln.piece) in made:  # the part in hand is this part
            row[1] += ln.qty
        else:
            row[0] += ln.qty
    for (slot, item), (qty, kept) in remade.items():
        spec = fab.get(item)
        if spec is None:
            errors.append(f"{item}: no machine for it on this bench")
            continue
        m = qty * float(spec.get("minutes_per_unit", 0.0))
        minutes += m
        lines.append(f"{item} {qty:.1f} {spec.get('unit', '')} for {slot}: {m:.0f} machine minutes"
                     + (f" ({kept:.1f} {spec.get('unit', '')} of parts reused)" if kept else ""))
    if usd and spent.usd + usd > float(budget.get("usd", 0.0)) + 1e-9:
        errors.append(f"buying costs ${usd:.2f}; the run has ${float(budget.get('usd', 0.0)) - spent.usd:.2f} left")
    if lead and lead > float(budget.get("lead_days", 0.0)) + 1e-9:
        errors.append(f"parts take {lead:g} days to arrive; the run allows {float(budget.get('lead_days', 0.0)):g}")
    if minutes and spent.fab_minutes + minutes > float(budget.get("fab_minutes", float("inf"))) + 1e-9:
        errors.append(f"fabrication takes {minutes:.0f} min; the run has "
                      f"{float(budget.get('fab_minutes', 0.0)) - spent.fab_minutes:.0f} left")
    return ResourceReport(ok=not errors, lines=lines, errors=errors, usd=round(usd, 2), lead_days=lead,
                          fab_minutes=round(minutes, 1), buy=buy)


def cost_idea(needs: Sequence[Any], manifest: dict[str, Any], spent: Spent) -> dict[str, Any]:
    """What an idea's bill of materials would take from this bench: provided, bought at the manifest's price, not
    provided, or a new item at System 2's own estimate.  An idea is never built, so nothing is charged."""
    items: dict[str, dict[str, Any]] = manifest.get("items", {})
    fab: dict[str, dict[str, Any]] = manifest.get("fabrication", {})
    owned = spent.owned or {}
    lines, missing, new = [], [], []
    usd = est = lead = minutes = 0.0
    for n in needs:
        if is_fab(n.item):
            spec = fab.get(n.item)
            if spec is None:
                missing.append(f"{n.item}: no machine for it on this bench")
                continue
            m = n.qty * float(spec.get("minutes_per_unit", 0.0))
            minutes += m
            lines.append(f"{n.item} {n.qty:g} {spec.get('unit', '')}: {m:.0f} machine min")
            continue
        spec = items.get(n.item)
        if spec is None:
            if n.usd_each is None:
                missing.append(f"{n.item} x{n.qty:g}: not on the manifest and unpriced")
            else:
                est += n.qty * n.usd_each
                lead = max(lead, float(n.lead_days or 0.0))
                new.append(f"{n.item} x{n.qty:g}: not on the manifest; estimated ${n.qty * n.usd_each:.2f}")
            continue
        short = max(0.0, n.qty - float(spec.get("on_hand", 0)) - owned.get(n.item, 0.0))
        if short <= 1e-9:
            lines.append(f"{n.item} x{n.qty:g}: provided")
        elif spec.get("usd") is None:
            missing.append(f"{n.item} x{n.qty:g}: cannot be bought")
        else:
            usd += short * float(spec["usd"])
            lead = max(lead, float(spec.get("lead_days", 0)))
            lines.append(f"{n.item} x{n.qty:g}: buy {short:g} at ${float(spec['usd']):g}")
    budget = manifest.get("budget", {})
    left = float(budget.get("usd", 0.0)) - spent.usd
    fits = not missing and usd + est <= left + 1e-9 and lead <= float(budget.get("lead_days", float("inf"))) + 1e-9 \
        and spent.fab_minutes + minutes <= float(budget.get("fab_minutes", float("inf"))) + 1e-9
    return {"usd": round(usd, 2), "usd_estimated": round(est, 2), "lead_days": lead, "fab_minutes": round(minutes, 1),
            "lines": lines, "new_items": new, "missing": missing, "fits_budget": fits}


def idea_text(c: dict[str, Any]) -> str:
    """One line for the prompt and the record."""
    head = (f"${c['usd']:.2f} from the manifest" + (f" + ${c['usd_estimated']:.2f} estimated for new items"
                                                     if c["usd_estimated"] else "") +
            f", {c['lead_days']:g} days, {c['fab_minutes']:.0f} machine min")
    tail = "; ".join(c["missing"] + c["new_items"])
    return head + ("; within the budget left" if c["fits_budget"] else "; NOT within what is left or provided") + \
        (f" ({tail})" if tail else "")
