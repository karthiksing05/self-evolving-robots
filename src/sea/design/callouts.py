"""A labelled picture of a model: numbered callouts with leader lines to each component, and a legend that says
what each one does (docs/design-language.md).  For people learning a body; System 2 never sees it.

The callouts point at what is visible: each part is also rendered in a flat colour of its own from the same camera,
and a leader ends at the visible pixel of the part farthest from its outline.  Parts that share a name share a number.
"""

from __future__ import annotations

import io
from collections.abc import Mapping

import numpy as np

from sea.design.cad import CadModel, View, _basis, _polydata, render


def _camera(model: CadModel, view: View, w: int, h: int, margin: float):
    """The orthographic camera ``render`` fits for one view: (right, up, back, centre, half height)."""
    back, up, right = _basis(view)
    pts = np.vstack([p.mesh.vertices for p in model.parts])
    u, v = pts @ right, pts @ up
    cu, cv = (u.min() + u.max()) / 2, (v.min() + v.max()) / 2
    half = max((v.max() - v.min()) / 2, (u.max() - u.min()) / 2 * h / w) * (1 + margin)
    centre = right * cu + up * cv + back * float((pts @ back).mean())
    return right, up, back, centre, half


def ids(model: CadModel, view: View, w: int, h: int, margin: float = 0.09) -> np.ndarray:
    """Which part (index) each pixel shows, -1 for background: every part drawn unlit in a colour of its own."""
    import vtk
    from vtk.util.numpy_support import vtk_to_numpy

    right, up, back, centre, half = _camera(model, view, w, h, margin)
    ren = vtk.vtkRenderer()
    ren.SetBackground(0, 0, 0)
    for i, part in enumerate(model.parts):
        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputData(_polydata(part.mesh))
        actor = vtk.vtkActor()
        actor.SetMapper(mapper)
        k = i + 1
        actor.GetProperty().SetColor((k & 255) / 255, ((k >> 8) & 255) / 255, 0.0)
        actor.GetProperty().LightingOff()
        ren.AddActor(actor)
    cam = ren.GetActiveCamera()
    cam.ParallelProjectionOn()
    cam.SetFocalPoint(*centre)
    cam.SetPosition(*(centre + back * 5000.0))
    cam.SetViewUp(*up)
    cam.SetParallelScale(half)
    ren.ResetCameraClippingRange()
    win = vtk.vtkRenderWindow()
    win.SetOffScreenRendering(1)
    win.SetMultiSamples(0)
    win.SetSize(w, h)
    win.AddRenderer(ren)
    win.Render()
    grab = vtk.vtkWindowToImageFilter()
    grab.SetInput(win)
    grab.ReadFrontBufferOff()
    grab.Update()
    img = vtk_to_numpy(grab.GetOutput().GetPointData().GetScalars()).reshape(h, w, -1)[::-1, :, :3].astype(int)
    win.Finalize()
    return img[:, :, 0] + 256 * img[:, :, 1] - 1


def _anchor(mask: np.ndarray) -> tuple[int, int] | None:
    """The pixel (x, y) of a region farthest from its outline."""
    from scipy.ndimage import distance_transform_edt

    if mask.sum() < 6:
        return None
    d = distance_transform_edt(np.pad(mask, 1))[1:-1, 1:-1]
    y, x = np.unravel_index(int(np.argmax(d)), d.shape)
    return int(x), int(y)


def _spread(xs: list[float], lo: float, hi: float, gap: float) -> list[float]:
    """Positions as near ``xs`` (sorted) as they can be while ``gap`` apart, inside [lo, hi]."""
    out = list(xs)
    for i in range(1, len(out)):
        out[i] = max(out[i], out[i - 1] + gap)
    if out and out[-1] > hi:
        out[-1] = hi
        for i in range(len(out) - 2, -1, -1):
            out[i] = min(out[i], out[i + 1] - gap)
    return [max(lo, x) for x in out]


def _wrap(d, text: str, font, width: float) -> list[str]:
    lines, cur = [], ""
    for word in text.split():
        trial = f"{cur} {word}".strip()
        if cur and d.textlength(trial, font=font) > width:
            lines.append(cur)
            cur = word
        else:
            cur = trial
    return [*lines, cur]


def _sides(tips: dict[str, list[tuple[int, int, int]]], w: int, h: int) -> dict[str, str]:
    """The margin each label goes to: the one nearest its part (the sides weighed for a wide picture)."""
    out = {}
    for n, t in tips.items():
        x, y = np.mean([q[1] for q in t]) / w, np.mean([q[2] for q in t]) / h
        out[n] = min((("top", y), ("bottom", 1 - y), ("left", 1.6 * x), ("right", 1.6 * (1 - x))), key=lambda kv: kv[1])[0]
    return out


def _cross(p, q, a, b) -> bool:
    """Whether segments pq and ab cross."""

    def turn(u, v, x) -> float:
        return float(np.sign((v[0] - u[0]) * (x[1] - u[1]) - (v[1] - u[1]) * (x[0] - u[0])))

    return turn(p, q, a) != turn(p, q, b) and turn(a, b, p) != turn(a, b, q)


def _uncross(spot: dict[str, float], place, ends: dict[str, list[tuple[float, float]]]) -> None:
    """Swap labels' places along one margin until no two of their leaders cross (or a pass limit is reached)."""
    names = list(spot)
    for _ in range(4 * len(names)):
        hit = next(((a, b) for i, a in enumerate(names) for b in names[i + 1:]
                    if any(_cross(place(spot[a]), e, place(spot[b]), f) for e in ends[a] for f in ends[b])), None)
        if hit is None:
            return
        a, b = hit
        spot[a], spot[b] = spot[b], spot[a]


def labelled(model: CadModel, legend: Mapping[str, str], title: str = "", view: int = 0,
             size: tuple[int, int] = (1400, 700), per_name: int = 2) -> bytes:
    """One view of ``model`` with a numbered callout on every component and a legend of what each does.  ``legend``
    maps a part's name to its line.  Parts that share a name share a number, with leaders to the ``per_name`` most
    visible of them (to one, for a name more parts than that share).  Labels sit in the margin nearest their part and
    are numbered clockwise from the top left, so the legend reads in the order the eye goes round."""
    from PIL import Image, ImageDraw, ImageFont

    w, h = size
    v = model.views[view]
    pic = Image.open(io.BytesIO(render([CadModel(model.parts, [v], (w, h))])[0])).convert("RGB")
    who = ids(model, v, w, h)
    seen: dict[str, list[tuple[int, int, int]]] = {}
    count: dict[str, int] = {}
    for i, part in enumerate(model.parts):
        count[part.name] = count.get(part.name, 0) + 1
        a = _anchor(who == i)
        if a is not None:
            seen.setdefault(part.name, []).append((int((who == i).sum()), *a))
    tips = {n: sorted(t, reverse=True)[:per_name if count[n] <= per_name else 1] for n, t in seen.items()}
    side = _sides(tips, w, h)
    band, r, gap = 48, 13, 32
    head = 36 if title else 0
    ox, oy = band, head + band  # the picture's corner on the canvas
    canvas_w = w + 2 * band
    order: list[str] = []
    at: dict[str, tuple[float, float]] = {}
    ends = {n: [(ox + x, oy + y) for _, x, y in t] for n, t in tips.items()}
    for s in ("top", "right", "bottom", "left"):
        along = 1 if s in ("top", "bottom") else 2
        key = {n: float(np.mean([q[along] for q in tips[n]])) for n in tips if side[n] == s}
        names = sorted(key, key=key.get)
        spot = dict(zip(names, _spread([key[n] for n in names], r + 2, (w if along == 1 else h) - r - 2, gap),
                        strict=True))

        def place(q: float, s: str = s) -> tuple[float, float]:
            return {"top": (ox + q, head + band / 2), "bottom": (ox + q, oy + h + band / 2),
                    "left": (band / 2, oy + q), "right": (ox + w + band / 2, oy + q)}[s]

        _uncross(spot, place, ends)
        for n in sorted(spot, key=spot.get, reverse=s in ("bottom", "left")):  # clockwise
            order.append(n)
            at[n] = place(spot[n])
    font, small, tiny = ImageFont.load_default(size=16), ImageFont.load_default(size=14), ImageFont.load_default(size=12)
    probe = ImageDraw.Draw(Image.new("RGB", (1, 1)))
    col_w = (canvas_w - 32) / 2
    entries = [_wrap(probe, f"{n} · {legend[n]}" if legend.get(n) else n, font, col_w - 52) for n in order]
    half = -(-len(entries) // 2)
    rows = max(sum(len(e) for e in entries[:half]), sum(len(e) for e in entries[half:]))
    canvas = Image.new("RGB", (canvas_w, oy + h + band + 14 + 21 * rows + 2 * half + 14), (250, 249, 246))
    canvas.paste(pic, (ox, oy))
    d = ImageDraw.Draw(canvas)
    if title:
        d.text((16, 10), title, fill=(25, 25, 28), font=ImageFont.load_default(size=20))
    for k, n in enumerate(order):
        lx, ly = at[n]
        for _, x, y in tips[n]:
            d.line([(lx, ly), (ox + x, oy + y)], fill=(70, 74, 82), width=1)
            d.ellipse([ox + x - 3, oy + y - 3, ox + x + 3, oy + y + 3], fill=(20, 90, 200))
        d.ellipse([lx - r, ly - r, lx + r, ly + r], fill=(20, 90, 200))
        d.text((lx, ly), str(k + 1), fill=(255, 255, 255), font=small, anchor="mm")
    for c in range(2):
        y = oy + h + band + 14
        for k in range(c * half, min((c + 1) * half, len(entries))):
            x = 16 + c * col_w
            d.ellipse([x, y + 1, x + 20, y + 21], fill=(20, 90, 200))
            d.text((x + 10, y + 11), str(k + 1), fill=(255, 255, 255), font=tiny, anchor="mm")
            for line in entries[k]:
                d.text((x + 30, y + 2), line, fill=(30, 30, 34), font=font)
                y += 21
            y += 2
    buf = io.BytesIO()
    canvas.save(buf, format="PNG", optimize=True)
    return buf.getvalue()
