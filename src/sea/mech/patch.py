"""Edit a mechanism graph by naming what changes (D-079): how System 2 writes a mechanism the catalogue lacks.

A patch is a list of operations on a module's graph, each addressed by names, never by position:

- ``bodies/<body>``: a whole body (``add`` with the body, ``remove``);
- ``bodies/<body>/<field>``: one of its fields (``set``: pos, quat, parent, inertial, joints);
- ``bodies/<body>/parts/<part>``: a part (``add`` with the part, ``set`` to replace it, ``remove``);
- ``bodies/<body>/parts/<part>/<field>``: one field of a part (``set``);
- ``guides/<carriage>/<field>``, ``meshes/<follower>/<field>``, ``loops/<a>/<field>``: a guide's, a mesh's or
  a loop's field (``set``; a loop is named by its first body);
- ``loops``, ``guides``, ``meshes``, ``excludes``, ``actuators``, ``latches``, ``sensors``, ``carried``: append one
  (``add``);
- ``actuators/<name>``, ``latches/<name>``, ``sensors/<name>``: replace or remove one (``set``, ``remove``);
- ``linked``: replace the joints that follow the servo by ratio (``set``).

A part's name is its ``name``, or for a blade ``blade`` and for a bought part its ``item``.
"""

from __future__ import annotations

import json
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from sea.mech.graph import Graph


class Op(BaseModel):
    model_config = ConfigDict(extra="forbid")

    op: Literal["add", "set", "remove"]
    path: str = Field(description="what to change, by names: bodies/<body>[/parts/<part>][/<field>], "
                      "guides/<carriage>/<field>, meshes/<follower>/<field>, loops/<a>/<field>, loops, guides, "
                      "meshes, excludes, linked, actuators[/<name>], latches[/<name>], sensors[/<name>], carried")
    value: Any = None


def _part_name(p: dict) -> str:
    return p.get("name") or ("blade" if p.get("kind") == "blade" else p.get("item", p.get("kind", "?")))


def _find(items: list[dict], name: str, key, what: str) -> int:
    for i, x in enumerate(items):
        if key(x) == name:
            return i
    raise ValueError(f"no {what} {name!r}; there are {[key(x) for x in items]}")


def apply_patch(g: Graph, ops: list[Op]) -> Graph:
    """``g`` with ``ops`` applied; a ``ValueError`` names the first that does not apply or a graph that does not
    validate."""
    d = json.loads(g.model_dump_json())
    for k, o in enumerate(ops):
        seg = [s for s in o.path.split("/") if s]
        try:
            if seg[0] in ("actuators", "latches", "sensors") and len(seg) == 2:
                i = _find(d[seg[0]], seg[1], lambda x: x["name"], seg[0][:-1])
                if o.op == "remove":
                    d[seg[0]].pop(i)
                else:
                    d[seg[0]][i] = o.value
            elif seg[0] in ("loops", "guides", "meshes", "excludes", "actuators", "latches", "sensors", "carried") \
                    and len(seg) == 1:
                if o.op != "add":
                    raise ValueError("only add to a list")
                d[seg[0]] = [*d[seg[0]], o.value]
            elif seg == ["linked"]:
                d["linked"] = o.value
            elif seg[0] in ("guides", "meshes", "loops") and len(seg) == 3:
                key = {"guides": lambda x: x["carriage"], "meshes": lambda x: x["follower"], "loops": lambda x: x["a"]}[seg[0]]
                d[seg[0]][_find(d[seg[0]], seg[1], key, seg[0][:-1])][seg[2]] = o.value
            elif seg[0] == "bodies":
                bodies = d["bodies"]
                if len(seg) == 2:
                    if o.op == "add":
                        bodies.append(o.value)
                    else:
                        i = _find(bodies, seg[1], lambda x: x["name"], "body")
                        if o.op == "remove":
                            bodies.pop(i)
                        else:
                            bodies[i] = o.value
                    continue
                b = bodies[_find(bodies, seg[1], lambda x: x["name"], "body")]
                if seg[2] != "parts":
                    b[seg[2]] = o.value
                elif len(seg) == 3:
                    if o.op != "add":
                        raise ValueError("add a part with bodies/<body>/parts")
                    b["parts"] = [*b["parts"], o.value]
                else:
                    parts = b["parts"]
                    if o.op == "add":
                        parts.append(o.value)
                        continue
                    i = _find(parts, seg[3], _part_name, f"part on {seg[1]}")
                    if o.op == "remove":
                        parts.pop(i)
                    elif len(seg) == 4:
                        parts[i] = o.value
                    else:
                        parts[i] = {**parts[i], seg[4]: o.value}
            else:
                raise ValueError("unknown path")
        except (ValueError, IndexError, KeyError, TypeError) as exc:
            raise ValueError(f"patch operation {k} ({o.op} {o.path}): {exc}") from exc
    try:
        return Graph.model_validate(d)
    except Exception as exc:  # pydantic's message names the field
        raise ValueError(f"the patched mechanism is not a valid graph: {str(exc)[:600]}") from exc
