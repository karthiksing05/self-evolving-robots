"""System 2 backends for the evolution core.

``PlanClaude`` asks Claude through the logged-in Claude Code CLI, as ``sea.propose.llm.ClaudeCodeLLM``
does (D-029): calls bill against the claude.ai plan, it refuses any other login, and nothing in the repo
is read as context.  Every call is appended to ``log_path`` with its tokens and list-price cost, and the
prompt and answer are saved, so a run can be read back call by call.  ``Scripted`` replays fixed
batches, for tests and for the random baseline.
"""

from __future__ import annotations

import json
import subprocess
import tempfile
import time
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any, Protocol

from pydantic import ValidationError

from sea.evolve.core import ProposalBatch
from sea.propose.llm import (
    CLAUDE_FLAGS,
    PLAN_AUTH,
    SCRUBBED_ENV,
    PlanBillingError,
    PlanLimitReached,
    call_usd,
    cli_schema,
    find_claude,
)

# With System 2's tools (D-086): ``--safe-mode`` would drop every MCP server, the ``sea`` one too, so its parts are
# named instead: no built-in tools, no settings files (so no hooks or plugins), no skills or commands, and no MCP
# server but ``sea``.  A smoke test showed no CLAUDE.md, memory or skill in the context (2026-11-04).
TOOL_FLAGS = ("-p", "--output-format", "json", "--tools", "", "--setting-sources", "", "--disable-slash-commands",
              "--strict-mcp-config", "--no-session-persistence")

UNREACHABLE = ("can't reach the api server", "enotfound", "econnreset", "econnrefused", "etimedout", "overloaded")


def unreachable(msg: str) -> bool:
    """The CLI could not reach Claude: a network or service outage, not an answer."""
    return any(k in msg.lower() for k in UNREACHABLE)


class Proposer(Protocol):
    def propose(self, system: str, prompt: str, tag: str = "") -> ProposalBatch: ...


class PlanClaude:
    def __init__(self, model: str = "claude-opus-5-5", effort: str = "high", log_dir: Path | None = None,
                 timeout_s: float = 1800.0, run: Callable[..., subprocess.CompletedProcess] = subprocess.run,
                 outage_s: float = 1800.0, sleep: Callable[[float], None] = time.sleep):
        import os

        self.model, self.effort, self.timeout_s = model, effort, timeout_s
        self.log_dir = log_dir
        self._run = run
        self.outage_s, self._sleep = outage_s, sleep  # how long to wait out an outage, and how
        self._env = {k: v for k, v in os.environ.items() if k not in SCRUBBED_ENV}
        self._cwd = tempfile.mkdtemp(prefix="sea-evolve-")
        self._checked = False
        self.calls: list[dict[str, Any]] = []
        self.tools: tuple[Path, Path] | None = None  # the loop sets (context, log) each iteration when tools are on
        self.research: Path | None = None  # the loop sets the guard's settings file when research is on (D-090)

    def flags(self) -> list[str]:
        """The CLI's flags: no tools at all, or the ``sea`` tool server over this iteration's context."""
        if self.tools is None:
            return list(CLAUDE_FLAGS)
        import sys

        from sea.evolve.tools import NAMES

        ctx, log = self.tools
        cfg = ctx.with_suffix(".mcp.json")
        cfg.write_text(json.dumps({"mcpServers": {"sea": {
            "command": sys.executable, "args": ["-m", "sea.evolve.tools", "--context", str(ctx.resolve()),
                                                "--log", str(log.resolve())]}}}))
        allowed = [f"mcp__sea__{n}" for n in NAMES]
        flags = [*TOOL_FLAGS, "--mcp-config", str(cfg.resolve())]
        if self.research is not None:  # search and fetch, each passing the guard's hooks (D-090)
            from sea.evolve.research import TOOLS

            allowed += TOOLS
            flags[flags.index("--tools") + 1] = ",".join(TOOLS)
            flags += ["--settings", str(self.research.resolve())]
        return [*flags, "--allowedTools", ",".join(allowed)]

    def _exec(self, args: list[str], stdin: str | None = None) -> subprocess.CompletedProcess:
        binary = find_claude()
        if not binary:
            raise FileNotFoundError("no Claude Code CLI: install it, or set SEA_CLAUDE_BIN")
        return self._run([binary, *args], input=stdin, capture_output=True, text=True, timeout=self.timeout_s,
                         env=self._env, cwd=self._cwd)

    def check_plan(self) -> None:
        out = self._exec(["auth", "status"])
        try:
            status = json.loads(out.stdout)
        except json.JSONDecodeError as exc:
            raise PlanBillingError(f"`claude auth status` printed no JSON: {out.stderr[:200]}") from exc
        if not status.get("loggedIn") or status.get("authMethod") != PLAN_AUTH:
            raise PlanBillingError(f"Claude Code is logged in via {status.get('authMethod')!r}, not a claude.ai plan")
        self._checked = True

    def propose(self, system: str, prompt: str, tag: str = "") -> ProposalBatch:
        if not self._checked:
            self.check_plan()
        schema = json.dumps(cli_schema(ProposalBatch.model_json_schema()), sort_keys=True)
        args = [*self.flags(), "--model", self.model, "--effort", self.effort, "--system-prompt", system,
                "--json-schema", schema]
        text, error, batch = prompt, "", None
        t0 = time.time()
        tokens_in = tokens_out = 0
        usd = 0.0
        result: dict[str, Any] = {}
        attempts, waited, wait = 0, 0.0, 30.0
        while attempts < 2:
            try:
                out = self._exec(args, stdin=text)
            except subprocess.TimeoutExpired:  # a long answer (a mechanism) is not an outage: ask once more (D-082)
                attempts += 1
                error = f"no answer within {self.timeout_s:.0f} s"
                continue
            try:
                result = json.loads(out.stdout)
            except json.JSONDecodeError as exc:
                raise RuntimeError(f"`claude -p` exited {out.returncode} without JSON: {out.stderr[:300]}") from exc
            if result.get("is_error"):
                msg = str(result.get("result", ""))
                if result.get("api_error_status") == 429 or "usage limit" in msg.lower():
                    raise PlanLimitReached(msg[:300])
                if unreachable(msg):  # an outage is not an answer: wait it out, then ask again (2026-11-02)
                    if waited >= self.outage_s:
                        raise RuntimeError(f"Claude unreachable for {waited:.0f} s: {msg[:200]}")
                    self._sleep(wait)
                    waited, wait = waited + wait, min(2 * wait, 300.0)
                    continue
            attempts += 1
            usage = result.get("usage") or {}
            t_in = int(usage.get("input_tokens") or 0)
            c_w, c_r = int(usage.get("cache_creation_input_tokens") or 0), int(usage.get("cache_read_input_tokens") or 0)
            tokens_in += t_in + c_w + c_r
            tokens_out += int(usage.get("output_tokens") or 0)
            listed = result.get("total_cost_usd")
            usd += float(listed) if listed is not None else call_usd(self.model, t_in, int(usage.get("output_tokens") or 0), c_w, c_r)
            structured = result.get("structured_output")
            try:
                if structured is None:
                    structured = json.loads(result.get("result") or "null")
                batch = ProposalBatch.model_validate(structured)
                error = ""
                break
            except (ValidationError, json.JSONDecodeError, TypeError) as exc:
                error = str(exc)[:600]
                text = prompt + f"\n\nYour previous answer was not a valid ProposalBatch: {error}\nAnswer again."
        call = {"tag": tag, "model": self.model, "effort": self.effort, "seconds": round(time.time() - t0, 1),
                "tokens_in": tokens_in, "tokens_out": tokens_out, "usd_list_price": round(usd, 4), "billing": "plan",
                "error": error}
        if self.tools is not None:
            call["turns"] = int(result.get("num_turns") or 0)
        self.calls.append(call)
        if self.log_dir is not None:
            self.log_dir.mkdir(parents=True, exist_ok=True)
            (self.log_dir / f"{tag or len(self.calls)}_prompt.md").write_text(f"# system\n\n{system}\n\n# prompt\n\n{prompt}\n")
            (self.log_dir / f"{tag or len(self.calls)}_answer.json").write_text(
                json.dumps(batch.model_dump() if batch else {"error": error, "raw": result.get("result")}, indent=1))
            with (self.log_dir / "calls.jsonl").open("a") as f:
                f.write(json.dumps(call) + "\n")
        if batch is None:
            raise RuntimeError(f"System 2 gave no valid ProposalBatch twice: {error}")
        return batch


class Scripted:
    """Returns ``batches`` in order (the last one repeats), or calls ``fn(prompt)`` if given."""

    def __init__(self, batches: Sequence[ProposalBatch] = (), fn: Callable[[str], ProposalBatch] | None = None):
        self.batches = list(batches)
        self.fn = fn
        self.calls: list[dict[str, Any]] = []
        self.prompts: list[str] = []

    def propose(self, system: str, prompt: str, tag: str = "") -> ProposalBatch:
        self.prompts.append(prompt)
        self.calls.append({"tag": tag, "tokens_in": 0, "tokens_out": 0, "usd_list_price": 0.0, "billing": "none"})
        if self.fn is not None:
            return self.fn(prompt)
        return self.batches[min(len(self.calls) - 1, len(self.batches) - 1)]
