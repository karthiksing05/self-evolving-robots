"""Before and after pictures of the design language (docs/design-language.md), and the audit that checks them.

    uv run python scripts/design_language.py --out results/design_language

Writes to ``--out``:
- ``before_pads_<k>.png``, ``after_pads_<k>.png``: the jaws with the simulated pad boxes, and with the pads as built,
  for a flat TPU pad and a 40 degree grooved one; ``tips_<k>.png``: each fingertip close up, before above after;
- ``after_rig_stock.png``, ``after_rig_rich.png``: the talking robot, one camera for both bodies (the rich body adds
  the tip tongue, round lips and the larynx tube), ``labelled_rig_<k>.png`` with a callout on every component, and
  ``grouped_rig_<k>.png`` with its parts coloured by what they do and one label per group (a first look);
- ``audit.json``, ``audit.md``: per assembly, the parts nothing carries to the base, the parts that share space with
  no declared joint, declared joints that do not touch, and each part's form (sharp edge, overhang, thinnest wall);
  the same laws on each rig at both ends of every control's range; and the pad parameter sets the proposal-tier
  rules would refuse.  About four minutes.

``before_rig_*.png`` were drawn by ``worlds/cad.py``'s previous ``phonology`` (commit cbfb3a2) and are not rewritten.
"""

from __future__ import annotations

import argparse
import io
import json
import sys
from pathlib import Path

import numpy as np

FLAT = {"reach_mm": 10.0, "pad_angle_deg": 0.0, "groove_mm": 0.0, "lip_mm": 0.0, "pad_tpu_mm": 2.0, "width_mm": 12.0}
GROOVED = {"reach_mm": 10.0, "pad_angle_deg": 40.0, "groove_mm": 3.0, "lip_mm": 0.0, "pad_tpu_mm": 3.0, "width_mm": 20.0}
HOOKED = {"reach_mm": 20.0, "pad_angle_deg": 20.0, "groove_mm": 1.5, "lip_mm": 3.0, "pad_tpu_mm": 1.5, "width_mm": 16.0}
PADS = {"flat": FLAT, "grooved40": GROOVED, "hooked": HOOKED}


def gripper_bodies() -> dict:
    from sea.evolve.core import Edit, apply
    from sea.worlds.gripper import GripperWorld

    w = GripperWorld()
    cat = {p.name: p for p in w.catalogue()}
    return {k: apply(w.stock(), [Edit(slot="pads", type="pads", params=v)], cat, "t") for k, v in PADS.items()}


def rig_bodies() -> dict:
    from sea.evolve.core import Edit, apply
    from sea.worlds.phonology import PhonologyWorld

    w = PhonologyWorld()
    cat = {p.name: p for p in w.catalogue()}

    def swap(*edits):
        return apply(w.stock(), [Edit(slot=s, type=t, params={k: (p.lo + p.hi) / 2 for k, p in cat[t].params.items()})
                                 for s, t in edits], cat, "t")

    return {"stock": w.stock(), "rich": swap(("tongue", "tip_tongue"), ("lips", "round_lips"), ("tract", "larynx_tube")),
            "pneumatic_spread": swap(("tongue", "pneumatic_tongue"), ("lips", "spread_lips")),
            "long_rail_long_cast": swap(("tongue", "long_rail_tongue"), ("tract", "long_cast"))}


def boxes_only(body):
    """The gripper as drawn before the design language: the simulated pad boxes."""
    from sea.design import pad_cad
    from sea.worlds import cad

    keep = pad_cad.dress
    pad_cad.dress = lambda parts, spec: parts
    try:
        return cad.gripper(body)
    finally:
        pad_cad.dress = keep


def tip(model, side: str):
    """One fingertip close up: its pad and the bottom of the jaw it sits on."""
    from sea.design.cad import CadModel, CadPart, View
    from sea.design.pad_cad import _host

    pads = [p for p in model.parts if p.slot == "pads" and (p.name.startswith(side) or f"_{side}_" in p.name)]
    host = _host(pads[0].mesh, [p for p in model.parts if p.slot != "pads"])
    lo, hi = np.min([p.mesh.bounds[0] for p in pads], 0), np.max([p.mesh.bounds[1] for p in pads], 0)
    centre, reach = (lo + hi) / 2, float(np.linalg.norm(hi - lo)) / 2 + 8.0
    m = host.mesh
    keep = np.linalg.norm(m.triangles_center - centre, axis=1) < reach
    import trimesh

    clipped = trimesh.Trimesh(m.vertices, m.faces[keep], process=False)
    clipped.remove_unreferenced_vertices()
    s = 1.0 if side == "fixed" else -1.0
    views = [View("side", (0.0, -1.0, 0.0)), View("from the object", (s, -0.75, -0.4))]
    return CadModel([CadPart(host.name, "", clipped, "printed"), *pads], views, (330, 330))


def stack(rows: list[tuple[str, bytes]]) -> bytes:
    from PIL import Image, ImageDraw, ImageFont

    ims = [(label, Image.open(io.BytesIO(png)).convert("RGB")) for label, png in rows]
    w = max(im.width for _, im in ims)
    out = Image.new("RGB", (w, sum(im.height + 28 for _, im in ims)), (250, 249, 246))
    d, y = ImageDraw.Draw(out), 0
    font = ImageFont.load_default(size=18)
    for label, im in ims:
        d.text((10, y + 5), label, fill=(25, 25, 28), font=font)
        out.paste(im, (0, y + 28))
        y += im.height + 28
    buf = io.BytesIO()
    out.save(buf, format="PNG", optimize=True)
    return buf.getvalue()


FLOW = "Air from the pump → the reed buzzes → the silicone tube shapes the buzz into a sound; servos squeeze the tube."
# the rig's parts by what they do: (name, what it does, colour, the part names that open the group)
RIG_GROUPS = [
    ("Frame", "base plate and face plate: everything is screwed to them", (0.78, 0.76, 0.72), ("base plate", "face plate")),
    ("Voice source", "air from the pump buzzes a brass reed: the raw sound, like vocal folds",
     (0.30, 0.55, 0.85), ("reed housing",)),
    ("Vocal tract", "a silicone tube, 24 mm bore; squeezing it changes the sound", (0.93, 0.55, 0.40), ("cast tract",)),
    ("Tongue position", "a servo and crank slide the tongue along a rail: where it squeezes the tube",
     (0.55, 0.40, 0.75), ("rail", "rail servo")),
    ("Tongue height", "a servo turns a cam that pushes a soft pad up into the tube: how hard it squeezes",
     (0.85, 0.35, 0.55), ("sled",)),
    ("Lips", "servos open and close the lips (the stock jaw pulls the lower lip down)", (0.25, 0.65, 0.45),
     ("upper lip", "lip sleeve")),
    ("Larynx", "a servo and lead screw slide the reed end: a longer tube, lower resonances", (0.60, 0.58, 0.22),
     ("larynx bellows",)),
    ("Tongue tip", "two servos push a thin blade up just behind the teeth: the seal for /t/", (0.95, 0.70, 0.20),
     ("palate bridge",)),
]


def rig_groups(names: list[str]) -> list:
    """RIG_GROUPS with each group's part indices: a part joins the group whose opening name came last before it
    (the assembly lists each mechanism's parts together; a servo's mount, horn and screws follow it)."""
    opens = {n: k for k, (_, _, _, first) in enumerate(RIG_GROUPS) for n in first}
    members: list[list[int]] = [[] for _ in RIG_GROUPS]
    k = 0
    for i, n in enumerate(names):
        k = opens.get(n, k)
        members[k].append(i)
    return [(name, does, colour, members[j]) for j, (name, does, colour, _) in enumerate(RIG_GROUPS) if members[j]]


def main(argv: list[str] | None = None) -> int:
    from sea.contracts import FingerSpec
    from sea.design import language as LG
    from sea.design import pad_cad
    from sea.design.cad import render
    from sea.design.callouts import grouped, labelled
    from sea.worlds import cad, rig_cad

    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path("results/design_language"))
    args = ap.parse_args(argv)
    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    report: dict = {"pads": {}, "rig": {}, "refused": {}}

    for k, body in gripper_bodies().items():
        before, after = boxes_only(body), cad.gripper(body)
        if k in ("flat", "grooved40"):
            pngs = render([before, after])
            (out / f"before_pads_{k}.png").write_bytes(pngs[0])
            (out / f"after_pads_{k}.png").write_bytes(pngs[1])
            rows = []
            for side in ("fixed", "moving"):
                b, a = render([tip(before, side), tip(after, side)])
                rows += [(f"{side} pad, before: the simulated boxes", b), (f"{side} pad, after: as built", a)]
            (out / f"tips_{k}.png").write_bytes(stack(rows))
        report["pads"][k] = {asm.base: LG.audit(asm) for asm in pad_cad.assemblies(after.parts)}
        print(f"pads {k}: drawn and audited", flush=True)
    for k, params in PADS.items():
        report["refused"][k] = pad_cad.refusals(FingerSpec(**params))

    bodies = rig_bodies()
    asms = {k: rig_cad.assembly(b) for k, b in bodies.items()}
    pngs = render([asms["stock"].model(rig_cad.VIEWS, (460, 320)), asms["rich"].model(rig_cad.VIEWS, (460, 320))])
    (out / "after_rig_stock.png").write_bytes(pngs[0])
    (out / "after_rig_rich.png").write_bytes(pngs[1])
    titles = {"stock": "The stock talking robot: short cast, cam tongue, jaw lips",
              "rich": "Tip tongue, round lips and larynx tube: the tract in section so the bore shows"}
    for k in ("stock", "rich"):
        legend: dict[str, str] = {}
        for m in asms[k].members:
            legend.setdefault(m.part.name, m.role)
        model = asms[k].model(rig_cad.VIEWS, (460, 320))
        (out / f"labelled_rig_{k}.png").write_bytes(labelled(model, legend, titles[k]))
        (out / f"grouped_rig_{k}.png").write_bytes(grouped(model, rig_groups([m.part.name for m in asms[k].members]),
                                                           titles[k], FLOW))
    for k, asm in asms.items():
        report["rig"][k] = LG.audit(asm)
        print(f"rig {k}: drawn and audited", flush=True)
    report["sweep"] = {}
    for k, b in bodies.items():
        report["sweep"][k] = rig_cad.sweep(b)
        print(f"rig {k}: swept through each control's range", flush=True)

    (out / "audit.json").write_text(json.dumps(report, indent=1) + "\n")
    (out / "audit.md").write_text(summary(report))
    print(summary(report))
    bad = [k for group in ("pads", "rig") for k, v in report[group].items()
           for r in (v.values() if group == "pads" else [v]) if r["floating"] or r["undeclared"] or r["loose"]]
    return 1 if bad else 0


def summary(report: dict) -> str:
    lines = ["# Design-language audit", "",
             "Each assembly is checked by `sea.design.language.audit` (docs/design-language.md): parts that no chain of "
             "touching parts carries to the base (`mechanism.geometry.grounded`), pairs with no declared joint whose "
             "surfaces cross (`mechanism.geometry.overlap`) and that share volume more than 0.5 mm deep (undeclared), "
             "and declared joints whose parts do not touch (loose).  The form columns are over the parts the "
             "generators print (PLA and TPU; the stock jaws are their maker's): the worst sharp edge, the worst "
             "overhang share in the part's stated print orientation, and the thinnest wall.  The pads state no print "
             "orientation yet, so their overhang is not measured.", "",
             "| assembly | parts | joints | floating | undeclared | loose | sharp edge (mm) | overhang | thinnest wall (mm) |",
             "|---|---|---|---|---|---|---|---|---|"]
    rows = [(f"pad {k}, on {base}", r) for k, v in report["pads"].items() for base, r in v.items()]
    rows += [(f"rig {k}", r) for k, r in report["rig"].items()]
    for name, r in rows:
        printed = {k: f for k, f in r["form"].items() if f["material"] in ("printed", "tpu")}

        def worst(field: str, pick, d=printed) -> str:
            vals = [(f[field], k) for k, f in d.items() if f[field] is not None]
            if not vals:
                return "-"
            v, k = pick(vals)
            return f"{v:g} ({k})"

        lines.append(f"| {name} | {r['members']} | {r['joints']} | {len(r['floating'])} | {len(r['undeclared'])} | "
                     f"{len(r['loose'])} | {worst('sharp_mm', max)} | {worst('overhang', max)} | "
                     f"{worst('wall_p01_mm', min)} |")
    lines += ["", "At each end of each control's range (`rig_cad.sweep`), the other controls at the default posture:", ""]
    for k, found in report["sweep"].items():
        if not found:
            lines.append(f"- {k}: nothing floats, interferes or comes loose")
        for f in found:
            what = [f"{a} meets {b} ({' x '.join(f'{e:g}' for e in ext)} mm)" for a, b, ext in f["undeclared"]]
            what += [f"{p} floats" for p in f["floating"]] + [f"{a} leaves {b} ({c:g} mm)" for a, b, c in f["loose"]]
            lines.append(f"- {k}, {f['control']} = {f['value']:g}: " + "; ".join(what))
    lines += ["", "Pad parameter sets the proposal-tier rules would refuse:", ""]
    for k, why in report["refused"].items():
        lines.append(f"- {k}: " + ("; ".join(why) if why else "buildable"))
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    sys.exit(main())
