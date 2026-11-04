"""System 2's research and its engineering notebook (D-090; the general framework, F3).

An engineer finds what they need to know (a handbook's rule, a datasheet's rating, a paper's measurement)
and writes it down with its source.  System 2 does the same: its call has the CLI's ``WebSearch`` and
``WebFetch``, and the ``sea`` server's ``note`` tool records a finding in a notebook that outlives the run.

The harness owns two things here, and no knowledge:

- **the guard**, a ``PreToolUse`` hook on every search and fetch: it refuses what the world's
  ``research_blocked`` patterns match (for a benchmark, the course and its brief) and calls past the
  iteration's budget, and it logs every call.  A ``PostToolUse`` hook records the URLs each call returned;
- **provenance**: a note must cite a URL this run searched or fetched, and keeps its world, run and iteration.
  A kept change that cites a note credits it.

    python -m sea.evolve.research pre --config <research.json>    (the hooks; the event arrives on stdin)
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import time
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

TOOLS = ("WebSearch", "WebFetch")
BUDGET = 12  # searches and fetches per iteration
NOTE_ID = re.compile(r"\bN-[0-9a-f]{6}\b")
URL = re.compile(r"https?://[^\s\"'<>\\)\]]+")


class Config(BaseModel):
    """What the hooks of one System 2 call need: the patterns to refuse, the log, the iteration and budget."""

    model_config = ConfigDict(extra="forbid")
    blocked: list[str] = []
    log: str
    iteration: int
    budget: int = BUDGET


def _text(tool_input: dict[str, Any]) -> str:
    return " ".join(str(tool_input.get(k, "")) for k in ("query", "url", "prompt"))


def _calls(log: Path, iteration: int) -> int:
    if not log.exists():
        return 0
    return sum(1 for line in log.read_text().splitlines()
               if (r := json.loads(line)).get("iteration") == iteration and r.get("event") == "pre" and r.get("allowed"))


def _append(log: Path, row: dict[str, Any]) -> None:
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("a") as f:
        f.write(json.dumps(row) + "\n")


def pre(cfg: Config, event: dict[str, Any]) -> tuple[bool, str]:
    """Whether a search or fetch may run, and why not; logged either way."""
    tool, inp = event.get("tool_name", ""), event.get("tool_input") or {}
    log = Path(cfg.log)
    why = ""
    if tool in TOOLS:
        hit = next((p for p in cfg.blocked if re.search(p, _text(inp), re.IGNORECASE)), None)
        if hit:
            why = ("this topic is outside what the research may read (the task's own course or competition "
                   "material); search the engineering instead")
        elif _calls(log, cfg.iteration) >= cfg.budget:
            why = f"this iteration's {cfg.budget} searches and fetches are spent"
    _append(log, {"event": "pre", "iteration": cfg.iteration, "tool": tool, "input": inp, "allowed": not why,
                  "why": why, "t": round(time.time(), 1)})
    return not why, why


def post(cfg: Config, event: dict[str, Any]) -> None:
    """Record the URLs a search returned or a fetch read: the pages a note may cite."""
    inp = event.get("tool_input") or {}
    body = json.dumps(event.get("tool_response", ""))
    urls = sorted(set(URL.findall(body) + ([inp["url"]] if inp.get("url") else [])))
    # a hook cannot edit what a search returned: an innocent query may still surface blocked pages, so count them
    leaks = [p for p in cfg.blocked if re.search(p, body, re.IGNORECASE)]
    _append(Path(cfg.log), {"event": "post", "iteration": cfg.iteration, "tool": event.get("tool_name", ""),
                            "input": inp, "urls": urls, "leaks": leaks})


def read_urls(log: Path) -> set[str]:
    """Every URL this run's research searched or fetched."""
    if not log.exists():
        return set()
    out: set[str] = set()
    for line in log.read_text().splitlines():
        r = json.loads(line)
        if r.get("event") == "post":
            out.update(_norm(u) for u in r.get("urls", []))
    return out


def summary(log: Path, iteration: int | None = None) -> dict[str, Any]:
    """The research in ``log``: calls made, refused, and results that surfaced blocked material anyway."""
    rows = [json.loads(line) for line in log.read_text().splitlines()] if log.exists() else []
    rows = [r for r in rows if iteration is None or r.get("iteration") == iteration]
    return {"calls": sum(r["event"] == "pre" and r["allowed"] for r in rows),
            "refused": sum(r["event"] == "pre" and not r["allowed"] for r in rows),
            "leaks": sum(bool(r.get("leaks")) for r in rows)}


def _norm(url: str) -> str:
    return url.rstrip("/.,;").split("#")[0]


def settings(config: Path) -> dict[str, Any]:
    """The CLI settings that put the guard on every search and fetch."""
    cmd = f"{sys.executable} -m sea.evolve.research {{}} --config {config.resolve()}"
    return {"hooks": {
        "PreToolUse": [{"matcher": "|".join(TOOLS), "hooks": [{"type": "command", "command": cmd.format("pre")}]}],
        "PostToolUse": [{"matcher": "|".join(TOOLS), "hooks": [{"type": "command", "command": cmd.format("post")}]}]}}


# ---------------------------------------------------------------- the notebook
class Note(BaseModel):
    """A finding, with where it came from (D-090)."""

    model_config = ConfigDict(extra="forbid")
    id: str = ""
    claim: str = Field(description="what was found, in one sentence")
    source_url: str = Field(description="the page it was read on (searched or fetched in this run)")
    quote: str = Field(default="", description="the passage, verbatim, that supports it")
    numbers: dict[str, float] = Field(default_factory=dict, description="its values by name, with units in the name")
    validity: str = Field(default="", description="where it holds (sizes, materials, speeds)")
    applies_to: str = Field(default="", description="what in the design it bears on")
    world: str = ""
    run: str = ""
    iteration: int = -1
    credit: int = 0


def load(path: Path) -> list[Note]:
    if not path.exists():
        return []
    notes: dict[str, Note] = {}
    for line in path.read_text().splitlines():
        r = json.loads(line)
        if r.get("credit_for"):  # a credit row
            if r["credit_for"] in notes:
                n = notes[r["credit_for"]]
                notes[n.id] = n.model_copy(update={"credit": n.credit + 1})
            continue
        n = Note.model_validate(r)
        notes[n.id] = n
    return list(notes.values())


def add(path: Path, note: Note, read: set[str], world: str, run: str, iteration: int) -> Note:
    """Record ``note``; refused (``ValueError``) unless it cites a page this run read."""
    if not note.claim.strip():
        raise ValueError("a note needs its claim")
    url = _norm(note.source_url)
    if url not in read:
        raise ValueError(f"{note.source_url!r} is not a page this run searched or fetched: cite what you read")
    h = hashlib.sha1(f"{note.claim}|{url}|{note.quote}".encode()).hexdigest()[:6]
    note = note.model_copy(update={"id": f"N-{h}", "world": world, "run": run, "iteration": iteration, "credit": 0})
    if any(n.id == note.id for n in load(path)):
        return note
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as f:
        f.write(note.model_dump_json() + "\n")
    return note


def credit(path: Path, text: str) -> list[str]:
    """Credit every note ``text`` (a kept proposal) cites; returns their ids."""
    known = {n.id for n in load(path)}
    ids = sorted(set(NOTE_ID.findall(text)) & known)
    if ids:
        with path.open("a") as f:
            for i in ids:
                f.write(json.dumps({"credit_for": i}) + "\n")
    return ids


def text(notes: list[Note], limit: int = 40) -> str:
    """The notebook for the prompt: most credited first, then newest."""
    if not notes:
        return "The notebook is empty."
    order = sorted(notes, key=lambda n: (-n.credit, -n.iteration))[:limit]
    rows = []
    for n in order:
        nums = ", ".join(f"{k} = {v:g}" for k, v in n.numbers.items())
        rows.append(f"- {n.id}{f' (credited {n.credit})' if n.credit else ''} [{n.world}] {n.claim}"
                    + (f" ({nums})" if nums else "") + (f"; holds {n.validity}" if n.validity else "")
                    + f"; source {n.source_url}")
    more = len(notes) - len(order)
    return "\n".join(rows + ([f"... and {more} more notes"] if more > 0 else []))


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="the research guard's hooks")
    ap.add_argument("event", choices=("pre", "post"))
    ap.add_argument("--config", type=Path, required=True)
    a = ap.parse_args(argv)
    try:
        cfg = Config.model_validate_json(a.config.read_text())
        event = json.loads(sys.stdin.read() or "{}")
        if a.event == "post":
            post(cfg, event)
            return
        ok, why = pre(cfg, event)
    except Exception as exc:  # the guard fails closed: only exit 2 refuses a call, so a crash must become one
        if a.event == "post":
            print(f"research log: {exc}", file=sys.stderr)
            return
        ok, why = False, f"the research guard could not check this call: {exc}"
    if not ok:
        print(why, file=sys.stderr)
        sys.exit(2)  # the CLI refuses the call and shows the model why


if __name__ == "__main__":
    main()
