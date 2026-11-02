"""CAD of a body, and pictures of it (docs/evolve.md: a world's optional ``cad``).

A world's ``cad(body)`` returns a ``CadModel``: the body's parts as meshes in millimetres (CAD exports
such as the SO-101's STLs, or CadQuery solids tessellated by ``build.to_mesh``), each tagged with its
slot and a material, plus the views that show it best.  ``render`` draws several models (a run's bodies,
one per kept change) with one camera per view fitted to all of them, so the pictures share a scale.

The pictures are for people reading a run; System 2 never sees them.
"""

from __future__ import annotations

import io
from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np
import trimesh

MATERIALS = {  # base colour (RGB, 0-1)
    "printed": (0.86, 0.84, 0.80),  # PLA
    "tpu": (0.93, 0.47, 0.16),
    "silicone": (0.91, 0.66, 0.62),
    "metal": (0.63, 0.65, 0.69),
    "rubber": (0.20, 0.20, 0.21),
    "servo": (0.17, 0.18, 0.20),
    "motor": (0.72, 0.62, 0.38),
    "board": (0.13, 0.42, 0.30),
    "cell": (0.24, 0.42, 0.74),
}
BACKGROUND = (0.955, 0.950, 0.940)
EDGE = (0.12, 0.12, 0.13)


@dataclass
class CadPart:
    name: str
    slot: str  # the body slot it belongs to, or "" for structure no slot changes
    mesh: trimesh.Trimesh  # millimetres, in the model's frame
    material: str = "printed"


@dataclass
class View:
    name: str
    toward: tuple[float, float, float]  # from the model toward the camera
    up: tuple[float, float, float] = (0.0, 0.0, 1.0)


@dataclass
class CadModel:
    parts: list[CadPart]
    views: list[View] = field(default_factory=lambda: [View("iso", (1.0, -1.2, 0.8))])
    tile: tuple[int, int] = (360, 300)  # pixels per view


def solid(name: str, slot: str, wp, material: str = "printed") -> CadPart:
    """A part from a CadQuery workplane or shape."""
    from sea.design.build import to_mesh

    return CadPart(name, slot, to_mesh(wp), material)


def _polydata(mesh: trimesh.Trimesh):
    import vtk
    from vtk.util.numpy_support import numpy_to_vtk, numpy_to_vtkIdTypeArray

    pts = vtk.vtkPoints()
    pts.SetData(numpy_to_vtk(np.asarray(mesh.vertices, dtype=float), deep=True))
    faces = np.asarray(mesh.faces, dtype=np.int64)
    cells = vtk.vtkCellArray()
    offsets = numpy_to_vtkIdTypeArray(np.arange(0, 3 * len(faces) + 1, 3, dtype=np.int64), deep=True)
    cells.SetData(offsets, numpy_to_vtkIdTypeArray(faces.ravel(), deep=True))
    pd = vtk.vtkPolyData()
    pd.SetPoints(pts)
    pd.SetPolys(cells)
    return pd


def _basis(view: View) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    back = np.asarray(view.toward, dtype=float)
    back /= np.linalg.norm(back)
    up = np.asarray(view.up, dtype=float)
    up = up - back * (up @ back)
    up /= np.linalg.norm(up)
    return back, up, np.cross(up, back)


def render(models: Sequence[CadModel], margin: float = 0.09) -> list[bytes]:
    """One PNG per model, its views side by side (the first model's views and tile size); each view's
    camera (orthographic) is fitted to every model, so a part that grew between two bodies is drawn larger."""
    import vtk
    from PIL import Image

    w, h = models[0].tile
    tiles: list[list[np.ndarray]] = [[] for _ in models]
    for view in models[0].views:
        back, up, right = _basis(view)
        pts = np.vstack([p.mesh.vertices for mdl in models for p in mdl.parts])
        u, v = pts @ right, pts @ up
        cu, cv = (u.min() + u.max()) / 2, (v.min() + v.max()) / 2
        half = max((v.max() - v.min()) / 2, (u.max() - u.min()) / 2 * h / w) * (1 + margin)
        centre = right * cu + up * cv + back * float((pts @ back).mean())
        for mi, mdl in enumerate(models):
            ren = vtk.vtkRenderer()
            ren.SetBackground(*BACKGROUND)
            ren.SetUseFXAA(True)
            kit = vtk.vtkLightKit()  # key, fill and back lights: faces square to the camera still shade
            kit.SetKeyLightIntensity(0.85)
            kit.AddLightsToRenderer(ren)
            for part in mdl.parts:
                normals = vtk.vtkPolyDataNormals()
                normals.SetInputData(_polydata(part.mesh))
                normals.SetFeatureAngle(35.0)
                normals.SplittingOn()
                mapper = vtk.vtkPolyDataMapper()
                mapper.SetInputConnection(normals.GetOutputPort())
                actor = vtk.vtkActor()
                actor.SetMapper(mapper)
                prop = actor.GetProperty()
                prop.SetColor(*MATERIALS.get(part.material, MATERIALS["printed"]))
                prop.SetAmbient(0.25)
                prop.SetDiffuse(0.8)
                prop.SetSpecular(0.15)
                prop.SetSpecularPower(20)
                ren.AddActor(actor)
                edges = vtk.vtkFeatureEdges()
                edges.SetInputData(_polydata(part.mesh))
                edges.SetFeatureAngle(40.0)
                edges.BoundaryEdgesOn()
                edges.FeatureEdgesOn()
                edges.ManifoldEdgesOff()
                edges.NonManifoldEdgesOff()
                emap = vtk.vtkPolyDataMapper()
                emap.SetInputConnection(edges.GetOutputPort())
                emap.ScalarVisibilityOff()
                emap.SetResolveCoincidentTopologyToPolygonOffset()
                eact = vtk.vtkActor()
                eact.SetMapper(emap)
                eact.GetProperty().SetColor(*EDGE)
                eact.GetProperty().SetLineWidth(1.0)
                eact.GetProperty().SetOpacity(0.55)
                ren.AddActor(eact)
            cam = ren.GetActiveCamera()
            cam.ParallelProjectionOn()
            cam.SetFocalPoint(*centre)
            cam.SetPosition(*(centre + back * 5000.0))
            cam.SetViewUp(*up)
            cam.SetParallelScale(half)
            ren.ResetCameraClippingRange()
            win = vtk.vtkRenderWindow()
            win.SetOffScreenRendering(1)
            win.SetSize(w, h)
            win.AddRenderer(ren)
            win.Render()
            grab = vtk.vtkWindowToImageFilter()
            grab.SetInput(win)
            grab.ReadFrontBufferOff()
            grab.Update()
            img = grab.GetOutput()
            from vtk.util.numpy_support import vtk_to_numpy

            arr = vtk_to_numpy(img.GetPointData().GetScalars()).reshape(h, w, -1)[::-1, :, :3]
            tiles[mi].append(arr.copy())
            win.Finalize()
    out = []
    for row in tiles:
        buf = io.BytesIO()
        Image.fromarray(np.hstack(row).astype(np.uint8)).save(buf, format="PNG", optimize=True)
        out.append(buf.getvalue())
    return out


def box(name: str, slot: str, lo: Sequence[float], hi: Sequence[float], material: str = "printed") -> CadPart:
    """An axis-aligned box from its corners (mm): for parts whose shape is a block."""
    lo_, hi_ = np.asarray(lo, dtype=float), np.asarray(hi, dtype=float)
    m = trimesh.creation.box(extents=hi_ - lo_)
    m.apply_translation((lo_ + hi_) / 2)
    return CadPart(name, slot, m, material)
