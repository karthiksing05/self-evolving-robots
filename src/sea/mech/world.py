"""A world whose robot is one graph (D-110): what any such world gives System 2, with nothing about its task.

A world subclasses ``GraphWorld`` and supplies its tasks, physics, stock graph and kit.  It gets what System 2 composes
with: modules, patches, the machine elements, the text that explains them, and the body's graph.  ``mech.laws`` holds the
laws, ``mech.members`` the printed members, ``mech.program`` the controls and ``evolve.search`` System 1.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, ClassVar

from sea.evolve.core import Body, Part
from sea.mech import graph as G
from sea.mech.elements import compose
from sea.mech.members import PLA_ALLOW_PA, PRINT_FACTOR
from sea.mech.patch import Op, apply_patch
from sea.mech.patch import __doc__ as PATCH_DOC

SLOT = "robot"
KINDS: tuple[type, ...] = (G.Box, G.Cylinder, G.Capsule, G.Sphere, G.Link, G.Web, G.Wheel, G.Bought, G.Body, G.Joint, G.Loop,
                           G.Servo, G.Gearmotor, G.Piston, G.TorsionSpring, G.Band, G.Latch, G.LimitSwitch, G.RangeFinder,
                           G.Encoder, G.Carry)
MEMBERS = (f"A printed bar, link or web on a loop is judged at its pin's force, held 20 ms: in stress, at most "
           f"{PLA_ALLOW_PA / 1e6:.0f} MPa, and pushed, against its Euler load over {PRINT_FACTOR:g}.")


def graph_of(body: Body) -> G.Graph:
    return G.Graph.model_validate_json(body.parts[SLOT].mechanism)


class GraphWorld:
    modules: ClassVar[dict[str, tuple[Callable[[dict[str, float]], G.Graph], dict[str, tuple[float, float]]]]] = {}
    kinds: ClassVar[tuple[type, ...]] = KINDS
    frame: ClassVar[str] = ""  # what the world's graphs are in: their units and axes, and what the first body must be

    @classmethod
    def module(cls, name: str, params: dict[str, float]) -> G.Graph:
        if name not in cls.modules:
            raise ValueError(f"no module {name!r}; the modules are {sorted(cls.modules)}")
        build, ranges = cls.modules[name]
        bad = [f"{k} {v:g} outside {ranges[k][0]:g}..{ranges[k][1]:g}" if k in ranges else f"no parameter {k!r}"
               for k, v in params.items() if k not in ranges or not ranges[k][0] <= v <= ranges[k][1]]
        if bad:
            raise ValueError("; ".join(bad))
        return build({k: params.get(k, (lo + hi) / 2) for k, (lo, hi) in ranges.items()})

    # mechanisms System 2 composes
    def build_idea(self, body: Body, idea: Any, new_id: str) -> Body:
        mech = idea.mechanism  # a robot here is one graph: whatever slot an idea names, it is that graph
        if mech.module == "self":
            if mech.params:
                raise ValueError("module self has no parameters: write the change in the patch")
            g = graph_of(body)
        else:
            g = self.module(mech.module, mech.params)
        g = apply_patch(compose(g, mech.uses), [Op.model_validate(o) for o in mech.patch]).model_copy(update={"name": idea.name})
        return Body(id=new_id, parent=body.id, parts={SLOT: Part(type="built", mechanism=g.model_dump_json())})

    def module_names(self) -> list[str]:
        return sorted(self.modules)

    def mechanism_text(self, body: Body) -> str:
        return body.parts[SLOT].mechanism

    def module_text(self, name: str, params: dict[str, float]) -> str:
        return self.module(name, params).model_dump_json()

    def modules_text(self) -> str:
        kinds = []
        for cls in self.kinds:
            fields = ", ".join(f"{n}" + ("" if f.is_required() else f"={f.default!r}") for n, f in cls.model_fields.items()
                               if n not in ("kind", "kg"))
            kinds.append(f"- {cls.__name__}: {' '.join((cls.__doc__ or '').split())} ({fields})")
        out = [f"{self.frame}  Every part fastened to a body must touch another on it.  Joints, actuators, latches and "
               f"sensors are what operate it: each gives System 1 controls, which your plan sets.  {MEMBERS}", "Kinds:", *kinds,
               (PATCH_DOC or "").strip(), "Modules:"]
        for name, (_, ranges) in self.modules.items():
            mid = {k: (lo + hi) / 2 for k, (lo, hi) in ranges.items()}
            out += [f"## {name} ({', '.join(f'{k} {lo:g}..{hi:g}' for k, (lo, hi) in ranges.items())}); at {mid}:",
                    self.module(name, mid).model_dump_json()]
        return "\n".join(out)
