"""Static program checks before simulation (spec 8.7 ``typecheck.py``, Section 9, D-008).

``typecheck(program, design, library, limits, reach, control=..., panel=...)`` returns a list of
``TypeCheckError`` (empty = the program may run).  It checks:

- structure (Appendix A arity), unknown primitives and argument names;
- numeric arguments against the primitive's declared range and the hard limits (forces above
  ``max_force_n``, speeds above ``max_tip_speed_mps``, ``force_below(x)`` caps above the force
  limit), including the ranges of promoted composites;
- ``With`` slots against the tool rack;
- contact typing (D-008): the tool in scope (stock = ``edge``/``pinch``, a rack tool = its
  ``ToolSpec.contact``) must offer a contact that the primitive requires *and* that the target
  class accepts (Appendix B).  Banned contacts are not rejected here; the verifier flags them;
- that some actuating primitive acting on the target provides its class (and handle);
- reach: the contact point and approach direction of the target must lie in the reach grid for
  the tool tip in use.

``applicable_programs`` and ``wrap_with_tool`` are what the suite runner uses to pick a program
and a rack slot for a control (and to emit ``no_applicable_primitive`` when nothing fits).
"""

from __future__ import annotations

import math
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field

import numpy as np

from sea.compose import checks, dsl
from sea.compose.library import provides_control, requires_contacts
from sea.contracts import (
    ControlSpec,
    Design,
    HardLimits,
    Node,
    PanelSpec,
    PrimitiveDef,
    Program,
    ToolSpec,
    Vec3,
)

STOCK_CONTACTS = frozenset({"edge", "pinch"})
FORCE_PARAMS = frozenset({"force", "normal_force"})
SPEED_PARAMS = frozenset({"speed"})

# Appendix B: contacts each class accepts (doors and drawers by handle).
ACCEPTS: dict[str, frozenset[str]] = {
    "push_button": frozenset({"pad", "probe", "edge"}),
    "recessed_button": frozenset({"pad", "probe", "edge"}),
    "toggle": frozenset({"pad", "edge"}),
    "knob": frozenset({"cup", "pinch"}),
    "lever": frozenset({"pad", "hook"}),
    "slider": frozenset({"pad", "edge"}),
    "slot": frozenset({"edge"}),
}
HANDLE_ACCEPTS: dict[str, frozenset[str]] = {
    "tab": frozenset({"pinch", "hook"}),
    "pull": frozenset({"hook"}),
    "push_latch": frozenset({"pad", "probe"}),
}


@dataclass(frozen=True)
class TypeCheckError:
    code: str  # structure | unknown | arg | range | hard_limit | tool | contact | provides | ref | reach
    message: str
    path: str

    def __str__(self) -> str:
        return f"{self.path}: [{self.code}] {self.message}"


def accepted_contacts(control: ControlSpec | None = None, cls: str | None = None,
                      handle: str | None = None) -> frozenset[str]:
    """Appendix B acceptance for a control (or a class/handle pair; handle None = any handle)."""
    if control is not None:
        cls, handle = control.cls, control.handle
    if cls in ("door", "drawer"):
        return HANDLE_ACCEPTS.get(handle or "", frozenset().union(*HANDLE_ACCEPTS.values()))
    return ACCEPTS[cls or ""]


def tool_contacts(tool: ToolSpec | None) -> frozenset[str]:
    return STOCK_CONTACTS if tool is None or tool.template == "stock" else frozenset({tool.contact})


# ---------------------------------------------------------------- geometry (no sim needed)
def control_target(panel: PanelSpec, control: ControlSpec) -> tuple[np.ndarray, np.ndarray]:
    """World point the tool engages (handle point for tab/pull handles) and the approach direction,
    with the control at rest, computed from the specs alone (same frames as ``build_model``)."""
    from sea.control.frames import quat_wxyz_to_mat
    from sea.sim.controls import build_geometry

    geo = build_geometry(control)
    p_rot = quat_wxyz_to_mat(panel.pose_world.quat_wxyz)
    p_pos = np.array(panel.pose_world.pos.as_list())
    c_rot = quat_wxyz_to_mat(control.pose_panel.quat_wxyz)
    c_pos = np.array(control.pose_panel.pos.as_list())
    r_ctl = p_rot @ c_rot
    t_ctl = p_pos + p_rot @ c_pos
    m = geo.moving
    point_body = geo.contact_body
    if control.handle in ("tab", "pull") and geo.handle_body is not None and geo.flap is None:
        point_body = geo.handle_body
    point_ctl = np.asarray(m.pos, dtype=float) + quat_wxyz_to_mat(m.quat) @ np.asarray(point_body, dtype=float)
    approach = r_ctl @ np.asarray(geo.approach_ctl, dtype=float)
    return t_ctl + r_ctl @ point_ctl, approach / np.linalg.norm(approach)


ReachSource = Callable[[np.ndarray | None], object]  # tip transform (None = stock) -> ReachGrid


class ReachProvider:
    """Tip-specific reach grids for the stock arm, computed once per tip and cached on disk."""

    def __init__(self, limits: HardLimits, cache_dir=None):
        from sea.control.frames import ROOT

        self.limits = limits
        self.cache_dir = cache_dir or ROOT / ".cache" / "reach"
        self._grids: dict[bytes, object] = {}

    def __call__(self, tip: np.ndarray | None):
        from sea.control.executor import stock_ik
        from sea.control.reach import reach_for

        key = np.round(np.eye(4) if tip is None else np.asarray(tip), 9).tobytes()
        if key not in self._grids:
            self._grids[key] = reach_for(stock_ik(), self.limits, None if tip is None else np.asarray(tip),
                                         self.cache_dir)
        return self._grids[key]


def _tip_of(tool: ToolSpec | None) -> np.ndarray | None:
    if tool is None or tool.template == "stock":
        return None
    from sea.design.tools import tip_transform

    return tip_transform(tool)


def reach_ok(reach, point: np.ndarray, approach: np.ndarray, tool: ToolSpec | None) -> bool:
    """Is the tool tip able to reach ``point`` along ``approach``?

    ``reach`` is a ``ReachGrid`` for the stock frame or a callable ``tip -> ReachGrid``.  With a
    plain grid, a mounted tool is handled by moving the query back along the approach axis by the
    tip's z offset (exact for z-only tips; a hook's lateral offset is ignored, see C-003).
    """
    tip = _tip_of(tool)
    if callable(reach) and not hasattr(reach, "reachable"):
        return bool(reach(tip).reachable(point, approach))
    if tip is not None:
        point = point - approach * float(tip[2, 3])
    return bool(reach.reachable(point, approach))


# ---------------------------------------------------------------- the checker
@dataclass
class _Ctx:
    design: Design
    library: Mapping[str, PrimitiveDef]
    limits: HardLimits
    reach: object | None
    control: ControlSpec | None
    panel: PanelSpec | None
    target_cls: str = ""
    errors: list[TypeCheckError] = field(default_factory=list)
    provides_target: bool = False
    acts_on_target: bool = False
    depth: int = 0

    def err(self, code: str, path: str, msg: str) -> None:
        self.errors.append(TypeCheckError(code, msg, path))

    def control_by_id(self, ref: str | None) -> ControlSpec | None:
        if ref in (None, checks.TARGET):
            return self.control
        if self.panel is not None:
            for c in self.panel.controls:
                if c.id == ref:
                    return c
        return None


def bind_args(node: Node, defn: PrimitiveDef) -> tuple[str | None, dict[str, float | str], list[str]]:
    """(target ref, param -> value, unknown arg names) with positional args mapped like the executor."""
    assert node.prim is not None
    params = list(defn.params)
    args: dict[str, float | str] = dict(node.bind_args) if node.op == "bind" else {}
    args.update(node.prim.args)
    pos_keys = sorted((k for k in args if k.startswith("_") and k[1:].isdigit()), key=lambda k: int(k[1:]))
    target: str | None = None
    out: dict[str, float | str] = {}
    unknown: list[str] = []
    pi = 0
    for k in pos_keys:
        v = args[k]
        if isinstance(v, str) and target is None:
            target = v
            continue
        if pi < len(params):
            out[params[pi]] = v
        else:
            unknown.append(k)
        pi += 1
    for k, v in args.items():
        if k in pos_keys:
            continue
        if k == "target" and isinstance(v, str):
            target = v
        elif k in defn.params:
            out[k] = v
        else:
            unknown.append(k)
    return target, out, unknown


def _value_range(v: float | str, scope: Mapping[str, tuple[float, float]]) -> tuple[float, float] | None:
    if isinstance(v, (int, float)):
        return float(v), float(v)
    if isinstance(v, str) and v.startswith("$") and v != checks.TARGET:
        return scope.get(v[1:])
    return None


def _check_value(ctx: _Ctx, path: str, prim: str, param: str, rng: tuple[float, float],
                 declared: tuple[float, float]) -> None:
    lo, hi = rng
    if not (math.isfinite(lo) and math.isfinite(hi)):
        ctx.err("range", path, f"{prim}.{param}: non-finite value")
        return
    if param in FORCE_PARAMS and hi > ctx.limits.max_force_n:
        ctx.err("hard_limit", path, f"{prim}.{param} up to {hi:g} N exceeds the force limit {ctx.limits.max_force_n:g} N")
    if param in SPEED_PARAMS and hi > ctx.limits.max_tip_speed_mps:
        ctx.err("hard_limit", path,
                f"{prim}.{param} up to {hi:g} m/s exceeds the speed limit {ctx.limits.max_tip_speed_mps:g} m/s")
    if lo < declared[0] - 1e-12 or hi > declared[1] + 1e-12:
        ctx.err("range", path, f"{prim}.{param} [{lo:g}, {hi:g}] outside declared [{declared[0]:g}, {declared[1]:g}]")


def _check_expr(ctx: _Ctx, path: str, expr_text: str | None) -> None:
    if not expr_text:
        return
    try:
        expr = checks.parse(expr_text)
    except checks.CheckSyntaxError as exc:  # pragma: no cover - dsl.validate reports it first
        ctx.err("structure", path, str(exc))
        return
    for a in checks.atoms(expr):
        if a.name == "force_below" and float(a.value) > ctx.limits.max_force_n:  # type: ignore[arg-type]
            ctx.err("hard_limit", path, f"force_below({a.value}) exceeds the force limit {ctx.limits.max_force_n:g} N")
        if a.ref not in (None, checks.TARGET) and ctx.panel is not None and ctx.control_by_id(a.ref) is None:
            ctx.err("ref", path, f"check refers to unknown control {a.ref!r}")


def _check_prim(ctx: _Ctx, node: Node, path: str, tool: ToolSpec | None, target_ref: str | None,
                scope: Mapping[str, tuple[float, float]]) -> None:
    assert node.prim is not None
    name = node.prim.name
    defn = ctx.library.get(name)
    if defn is None:
        ctx.err("unknown", path, f"unknown primitive {name!r}")
        return
    if defn.kind == "check":
        ctx.err("unknown", path, f"{name!r} is a check, not a callable primitive")
        return
    ref, values, unknown = bind_args(node, defn)
    for k in unknown:
        ctx.err("arg", path, f"{name}: unknown argument {k!r}")
    if ref is not None and ref != checks.TARGET:
        target_ref = ref
    target = ctx.control_by_id(target_ref)
    if target_ref not in (None, checks.TARGET) and ctx.panel is not None and target is None:
        ctx.err("ref", path, f"{name}: unknown control {target_ref!r}")
    resolved: dict[str, tuple[float, float]] = {}
    for p, v in values.items():
        rng = _value_range(v, scope)
        if rng is None:
            ctx.err("arg", path, f"{name}.{p}: unresolved value {v!r}")
            continue
        resolved[p] = rng
        _check_value(ctx, path, name, p, rng, defn.params[p])

    if defn.kind == "composite" and defn.body is not None:
        if ctx.depth > 8:
            ctx.err("structure", path, f"{name}: composites nested too deeply")
            return
        inner = {k: resolved.get(k, v) for k, v in defn.params.items()}
        ctx.depth += 1
        _walk(ctx, defn.body.root, f"{path}->{name}", tool, target_ref, inner)
        ctx.depth -= 1
        return

    # motor primitive: contact typing against the control it acts on
    on_target = target_ref in (None, checks.TARGET) or (ctx.control is not None and target_ref == ctx.control.id)
    if not defn.provides:
        return
    if target is not None:
        cls, handle, accepts = target.cls, target.handle, accepted_contacts(target)
    elif on_target and ctx.target_cls:
        cls, handle, accepts = ctx.target_cls, "?", accepted_contacts(cls=ctx.target_cls)
    else:
        return
    offered = tool_contacts(tool)
    need = requires_contacts(defn)
    usable = (offered if need is None else offered & need) & accepts
    if not usable:
        tname = "stock gripper" if tool is None else f"{tool.template} ({tool.contact})"
        wanted = sorted((need & accepts) if need is not None else accepts) or sorted(accepts)
        ctx.err("contact", path, f"{name} on {cls}[{handle}] needs one of {wanted}; {tname} offers {sorted(offered)}")
    if cls == "slot" and name == "insert" and (tool is None or tool.template != "slot_inserter"):
        ctx.err("contact", path, "insert on a slot needs the slot_inserter tool")
    if not on_target:
        return
    ctx.acts_on_target = True
    if target is None:
        return
    if provides_control(defn, target):
        ctx.provides_target = True
    if ctx.reach is not None and ctx.panel is not None:
        point, approach = control_target(ctx.panel, target)
        if not reach_ok(ctx.reach, point, approach, tool):
            ctx.err("reach", path, f"{name}: target at {np.round(point, 3).tolist()} is out of reach "
                    f"for {'the stock gripper' if tool is None else tool.template}")


def _walk(ctx: _Ctx, node: Node, path: str, tool: ToolSpec | None, target_ref: str | None,
          scope: Mapping[str, tuple[float, float]]) -> None:
    _check_expr(ctx, path, node.check)
    _check_expr(ctx, path, node.cap)
    if node.op == "with":
        idx = node.tool_index if node.tool_index is not None else -1
        if not 0 <= idx < len(ctx.design.tool_rack):
            ctx.err("tool", path, f"With({idx}): rack has {len(ctx.design.tool_rack)} slot(s)")
            return
        _walk(ctx, node.children[0], f"{path}.with[{idx}]", ctx.design.tool_rack[idx], target_ref, scope)
        return
    if node.op in ("prim", "bind"):
        _check_prim(ctx, node, path, tool, target_ref, scope)
        return
    for i, c in enumerate(node.children):
        _walk(ctx, c, f"{path}.children[{i}]", tool, target_ref, scope)


def typecheck(program: Program, design: Design, library: Mapping[str, PrimitiveDef], limits: HardLimits,
              reach=None, *, control: ControlSpec | None = None,
              panel: PanelSpec | None = None) -> list[TypeCheckError]:
    """All static errors of ``program`` for ``control`` (empty list = OK to simulate)."""
    library = getattr(library, "prims", library)
    errs = [TypeCheckError("structure", "root", e) for e in dsl.validate(program.root)]
    if errs:
        return errs
    ctx = _Ctx(design, library, limits, reach, control, panel, target_cls=program.target_cls)
    _walk(ctx, program.root, "root", None, None, {})
    if control is not None:
        if not ctx.acts_on_target:
            ctx.err("provides", "root", f"no primitive acts on the target {control.cls}")
        elif not ctx.provides_target:
            ctx.err("provides", "root", f"no primitive provides {control.cls}[{control.handle}]")
    return ctx.errors


def reparam_errors(reparam: Mapping[str, tuple[float, float]], library: Mapping[str, PrimitiveDef],
                   limits: HardLimits, primitive: str | None = None) -> list[TypeCheckError]:
    """Check System 2 ``reparam`` ranges (keys ``prim.param`` or bare ``param`` for ``primitive``)."""
    library = getattr(library, "prims", library)
    ctx = _Ctx(Design(id="_"), library, limits, None, None, None)
    for key, (lo, hi) in reparam.items():
        prim, _, param = key.rpartition(".")
        prim = prim or (primitive or "")
        defn = library.get(prim)
        path = f"reparam[{key}]"
        if defn is None or param not in defn.params:
            ctx.err("arg", path, f"unknown primitive parameter {key!r}")
            continue
        if lo > hi:
            ctx.err("range", path, f"empty range [{lo:g}, {hi:g}]")
            continue
        _check_value(ctx, path, prim, param, (float(lo), float(hi)), defn.params[param])
    return ctx.errors


# ---------------------------------------------------------------- picking programs and tools
def _contact_ok(program: Program, design: Design, library: Mapping[str, PrimitiveDef], limits: HardLimits,
                control: ControlSpec) -> bool:
    return not any(e.code in ("contact", "tool") for e in typecheck(program, design, library, limits, control=control))


def _usable_contacts(root: Node, library: Mapping[str, PrimitiveDef], tool: ToolSpec | None,
                     control: ControlSpec) -> set[str]:
    out: set[str] = set()
    for n in dsl.walk(root):
        d = library.get(n.prim.name) if n.prim is not None else None
        if d is None:
            continue
        if d.kind == "composite" and d.body is not None:
            out |= _usable_contacts(d.body.root, library, tool, control)
        elif d.provides:
            need = requires_contacts(d)
            offered = tool_contacts(tool)
            out |= (offered if need is None else offered & need) & accepted_contacts(control)
    return out


def wrap_with_tool(program: Program, design: Design, control: ControlSpec, library: Mapping[str, PrimitiveDef],
                   limits: HardLimits | None = None, banned: Iterable[str] = ()) -> Program | None:
    """The program with a ``With(slot)`` for the first tool whose contact fits (None if none fits).

    Programs that already mount tools are returned unchanged when they type-check.  Candidates are
    the stock gripper, then rack slots in order; a candidate whose only usable contacts are banned
    (e.g. a stock pinch on a tab under no-pinch) is used only when nothing else fits, so the
    verifier still reports the violation when it is the only option (D-008, C-004).
    """
    library = getattr(library, "prims", library)
    limits = limits or _PERMISSIVE
    if dsl.tool_indices(program.root):
        return program if _contact_ok(program, design, library, limits, control) else None
    banned = set(banned)
    fallback: Program | None = None
    candidates: list[tuple[int | None, ToolSpec | None]] = [(None, None)]
    candidates += [(i, t) for i, t in enumerate(design.tool_rack) if t.template != "stock"]
    for slot, tool in candidates:
        cand = program if slot is None else program.model_copy(update={"root": dsl.with_tool(slot, program.root)})
        if not _contact_ok(cand, design, library, limits, control):
            continue
        usable = _usable_contacts(program.root, library, tool, control)
        if usable and usable <= banned:
            fallback = fallback or cand
            continue
        return cand
    return fallback


_PERMISSIVE = HardLimits(max_force_n=1e9, max_tip_speed_mps=1e9,
                         workspace_box=(Vec3(x=-1e9, y=-1e9, z=-1e9), Vec3(x=1e9, y=1e9, z=1e9)))


def _laplace_pass_rate(p: PrimitiveDef) -> float:
    return (p.passes + 1) / (p.uses + 2)


def applicable_programs(control: ControlSpec, design: Design, library, limits: HardLimits, reach=None,
                        panel: PanelSpec | None = None, banned: Iterable[str] = ()) -> list[PrimitiveDef]:
    """Library programs that provide ``control`` and type-check once wrapped with a tool.

    Each returned ``PrimitiveDef`` carries the runnable (tool-wrapped) program in ``body``.  Order:
    smoothed pass rate ``(passes + 1) / (uses + 2)`` descending, then name, so untried programs rank
    above ones that keep failing (C-005).
    """
    prims = getattr(library, "prims", library)
    banned = tuple(banned) or tuple(panel.rules.banned_contacts if panel is not None else ())
    out: list[PrimitiveDef] = []
    for p in prims.values():
        if p.kind != "composite" or p.body is None or not provides_control(p, control):
            continue
        wrapped = wrap_with_tool(p.body, design, control, prims, limits, banned)
        if wrapped is None:
            continue
        if typecheck(wrapped, design, prims, limits, reach, control=control, panel=panel):
            continue
        out.append(p.model_copy(update={"body": wrapped}))
    return sorted(out, key=lambda p: (-_laplace_pass_rate(p), p.name))
