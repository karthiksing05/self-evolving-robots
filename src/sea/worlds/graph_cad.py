"""CAD of any robot built as one graph (D-110), for pictures of a run.

General on purpose: it reads the solids the graph compiles to, so what is drawn is what is simulated, and it names no
world.  A bought part the graph carries as mass with no solid and no place (``Bought``) has nothing to
draw: ``unplaced`` lists those, so a caption can say so rather than a picture inventing where they go.
"""

from __future__ import annotations

import mujoco
import numpy as np
import trimesh

from sea.design.cad import CadModel, CadPart, View
from sea.evolve.core import Body
from sea.mech import graph as G
from sea.mech.base import WORLD
from sea.mech.build import compile as compile_graph
from sea.mech.world import SLOT, graph_of


def _mesh(kind: int, size: np.ndarray) -> trimesh.Trimesh | None:
    t = mujoco.mjtGeom
    if kind == t.mjGEOM_BOX:
        return trimesh.creation.box(extents=2 * size[:3])
    if kind == t.mjGEOM_CYLINDER:
        return trimesh.creation.cylinder(radius=size[0], height=2 * size[1], sections=48)
    if kind == t.mjGEOM_CAPSULE:
        return trimesh.creation.capsule(radius=size[0], height=2 * size[1], count=[24, 24]).apply_translation((0, 0, -size[1]))
    if kind == t.mjGEOM_SPHERE:
        return trimesh.creation.icosphere(radius=size[0], subdivisions=3)
    return None


def graph_model(g: G.Graph) -> CadModel:
    """The graph's solids at its rest pose, in millimetres, each as the material it is made of."""
    spec = mujoco.MjSpec.from_string('<mujoco><worldbody/></mujoco>')
    compile_graph(spec, g, WORLD)
    m = spec.compile()
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    made = {p.name: "rubber" if isinstance(p, G.Wheel) else "servo" if getattr(p, "material", "pla") == "stock" else "printed"
            for b in g.bodies for p in b.parts if hasattr(p, "name")}
    parts = []
    for i in range(m.ngeom):
        mesh = _mesh(m.geom_type[i], m.geom_size[i])
        if mesh is None or m.geom_group[i] == 4:
            continue
        pose = np.eye(4)
        pose[:3, :3], pose[:3, 3] = d.geom_xmat[i].reshape(3, 3), d.geom_xpos[i]
        name = m.geom(i).name or f"geom{i}"
        parts.append(CadPart(name, SLOT, mesh.apply_transform(pose).apply_scale(1000.0), made.get(name, "printed")))
    for s in g.sensors:
        at = d.xmat[m.body(s.body).id].reshape(3, 3) @ np.array(s.pos) + d.xpos[m.body(s.body).id]
        parts.append(CadPart(s.name, SLOT, trimesh.creation.box(extents=(14.0, 14.0, 14.0)).apply_translation(at * 1000.0), "board"))
    return CadModel(parts, [View("three-quarter", (1.0, -1.25, 0.85)), View("side", (0.0, -1.0, 0.0))])


def graph(body: Body) -> CadModel:
    return graph_model(graph_of(body))


def unplaced(g: G.Graph) -> list[str]:
    """What the graph carries with nothing to draw: bought parts, by item."""
    return [p.item for b in g.bodies for p in b.parts if isinstance(p, G.Bought)]
