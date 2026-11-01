"""System 2 LLM clients (spec Section 8.6, deviations in docs/decisions.md D-002, D-029).

``AnthropicLLM`` asks Claude for a ``ProposalBatch`` through a tool named ``propose``.  Forced
tool choice and sampling parameters are rejected by the current models, so the request uses
``tool_choice: auto`` with an instruction naming the tool, and the tool input is validated
client-side; a missing or invalid call is retried once with the error appended (history stays
append-only).  ``ClaudeCodeLLM`` asks the same question through the Claude Code CLI
(``claude -p --json-schema``), billed to the claude.ai plan it is logged into instead of API
credits.  Both cache responses by prompt hash and log every call with tokens and list-price USD,
so the ledger charges a proposer the same whichever way it was billed.
``MockLLM`` replays fixtures and is the only LLM tests may use.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import tempfile
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any, Literal, Protocol

from pydantic import BaseModel, ConfigDict, ValidationError

from sea.contracts import Proposal, ProposalBatch
from sea.propose.prompt import S2Prompt, request_hash

__all__ = [
    "PRICES",
    "TOOL_NAME",
    "AnthropicLLM",
    "ClaudeCodeLLM",
    "FixtureMissing",
    "LLM",
    "LLMCall",
    "MockLLM",
    "PlanBillingError",
    "PlanLimitReached",
    "ProposalSchemaError",
    "call_usd",
    "cli_schema",
    "find_claude",
    "parse_tool_response",
    "request_hash",
    "write_fixture",
]

TOOL_NAME = "propose"
FALLBACK_BETA = "server-side-fallback-2026-07-01"
# USD per million tokens (input, output); see D-002.
PRICES: dict[str, tuple[float, float]] = {
    "claude-opus-5-5": (4.0, 20.0),
    "claude-sonnet-5-5": (2.0, 10.0),
    "claude-haiku-5-5": (0.1, 0.5),
}
CACHE_WRITE_MULT = 1.25
CACHE_READ_MULT = 0.1


class ProposalSchemaError(ValueError):
    """The response carried no valid ``propose`` tool call."""


class FixtureMissing(KeyError):
    """``MockLLM`` has no fixture for a prompt hash."""


class PlanBillingError(RuntimeError):
    """The Claude Code CLI is not logged into a claude.ai plan, so a call would bill API credits."""


class PlanLimitReached(RuntimeError):
    """The plan's usage window is spent; the run should stop, not record an empty proposal batch."""


class LLM(Protocol):
    def propose(self, prompt: S2Prompt) -> list[Proposal]: ...


class LLMCall(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    prompt_hash: str
    model: str
    input_tokens: int = 0
    output_tokens: int = 0
    usd: float = 0.0
    cached: bool = False
    attempts: int = 0
    n_proposals: int = 0
    error: str | None = None
    billing: Literal["api", "plan"] | None = None  # None: cached or replayed, nothing was billed


class _CallLog:
    """Shared bookkeeping: keeps ``calls`` and appends each one to ``llm_calls.jsonl``."""

    def __init__(self, log_path: Path | None):
        self.log_path = Path(log_path) if log_path is not None else None
        self.calls: list[LLMCall] = []

    def _log(self, call: LLMCall) -> None:
        self.calls.append(call)
        if self.log_path is not None:
            self.log_path.parent.mkdir(parents=True, exist_ok=True)
            with self.log_path.open("a") as fh:
                fh.write(call.model_dump_json() + "\n")

    @property
    def total_usd(self) -> float:
        return sum(c.usd for c in self.calls)


def call_usd(model: str, input_tokens: int, output_tokens: int, cache_write: int = 0, cache_read: int = 0) -> float:
    """Price one call.  Unknown models are priced as the most expensive known model."""
    p_in, p_out = PRICES.get(model, max(PRICES.values()))
    return (p_in * (input_tokens + CACHE_WRITE_MULT * cache_write + CACHE_READ_MULT * cache_read)
            + p_out * output_tokens) / 1e6


def _block_get(block: Any, name: str, default: Any = None) -> Any:
    if isinstance(block, dict):
        return block.get(name, default)
    return getattr(block, name, default)


def parse_tool_response(content_blocks: Sequence[Any]) -> list[Proposal]:
    """Validate the ``propose`` tool call in a response's content blocks."""
    for block in content_blocks:
        if _block_get(block, "type") == "tool_use" and _block_get(block, "name") == TOOL_NAME:
            try:
                return list(ProposalBatch.model_validate(_block_get(block, "input")).proposals)
            except ValidationError as exc:
                raise ProposalSchemaError(str(exc)) from exc
    raise ProposalSchemaError(f"no `{TOOL_NAME}` tool call in the response")


def _as_param(block: Any) -> Any:
    """Turn a response content block back into a request param, unchanged (thinking included)."""
    if hasattr(block, "model_dump"):
        return block.model_dump(mode="json", exclude_none=True)
    if isinstance(block, dict):
        return dict(block)
    return {k: v for k, v in vars(block).items() if v is not None}


def _retry_turn(content_blocks: Sequence[Any], error: str) -> list[dict[str, Any]]:
    """User turn after a bad response: a tool_result error per tool call, plus a correction note."""
    note = (f"Your previous answer did not contain a valid `{TOOL_NAME}` call: {error}\n"
            f"Call the `{TOOL_NAME}` tool once more with a corrected ProposalBatch.")
    results: list[dict[str, Any]] = [
        {"type": "tool_result", "tool_use_id": _block_get(b, "id"), "content": note, "is_error": True}
        for b in content_blocks if _block_get(b, "type") == "tool_use"
    ]
    if results:
        return results
    return [{"type": "text", "text": note}]


class _Cached(_CallLog):
    """A live client's response cache: ``cache_dir/<prompt hash>.json`` holds the validated batch."""

    def __init__(self, cache_dir: Path | None, log_path: Path | None):
        super().__init__(log_path)
        self.cache_dir = Path(cache_dir) if cache_dir is not None else None

    def _cache_path(self, h: str) -> Path | None:
        return None if self.cache_dir is None else self.cache_dir / f"{h}.json"

    def _cache_get(self, h: str) -> list[Proposal] | None:
        path = self._cache_path(h)
        if path is None or not path.exists():
            return None
        data = json.loads(path.read_text())
        return list(ProposalBatch.model_validate(data["tool_input"]).proposals)

    def _cache_put(self, h: str, model: str, proposals: list[Proposal]) -> None:
        path = self._cache_path(h)
        if path is None:
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        batch = {"proposals": [p.model_dump(mode="json") for p in proposals]}
        path.write_text(json.dumps({"model": model, "tool_input": batch}, sort_keys=True, indent=1))


class AnthropicLLM(_Cached):
    """Claude as System 2.  Pass ``client`` to inject a fake in tests."""

    def __init__(self, model: str, cache_dir: Path | None = None, log_path: Path | None = None, client: Any = None,
                 effort: str = "medium", max_tokens: int = 16000):
        super().__init__(cache_dir, log_path)
        self.model = model
        self.effort = effort
        self.max_tokens = max_tokens
        self._client = client

    @property
    def client(self) -> Any:
        if self._client is None:  # pragma: no cover - live network client
            import anthropic

            self._client = anthropic.Anthropic()
        return self._client

    def tool(self) -> dict[str, Any]:
        return {
            "name": TOOL_NAME,
            "description": "Submit 1-3 System 2 proposals for the failing controls as a ProposalBatch.",
            "input_schema": ProposalBatch.model_json_schema(),
        }

    def request_kwargs(self, prompt: S2Prompt, messages: list[dict[str, Any]]) -> dict[str, Any]:
        kwargs: dict[str, Any] = {
            "model": prompt.model or self.model,
            "max_tokens": self.max_tokens,
            "system": prompt.system,
            "messages": messages,
            "tools": [self.tool()],
            "tool_choice": {"type": "auto"},
            "output_config": {"effort": self.effort},
            "cache_control": {"type": "ephemeral"},
        }
        if not kwargs["model"].startswith("claude-haiku"):  # no server-side fallback for Haiku
            kwargs["betas"] = [FALLBACK_BETA]
            kwargs["fallbacks"] = "default"
        return kwargs

    # ------------------------------------------------------------ call
    def propose(self, prompt: S2Prompt) -> list[Proposal]:
        h = prompt.hash()
        model = prompt.model or self.model
        cached = self._cache_get(h)
        if cached is not None:
            self._log(LLMCall(prompt_hash=h, model=model, cached=True, n_proposals=len(cached)))
            return cached
        instruction = (f"\n\nRespond by calling the `{TOOL_NAME}` tool exactly once with 1 to "
                       f"{prompt.max_proposals} proposals.")
        messages: list[dict[str, Any]] = [{"role": "user", "content": prompt.render() + instruction}]
        tokens_in = tokens_out = 0
        usd = 0.0
        proposals: list[Proposal] = []
        error: str | None = None
        attempts = 0
        for _ in range(2):  # first try + one retry on a schema error
            attempts += 1
            response = self.client.beta.messages.create(**self.request_kwargs(prompt, messages))
            usage = response.usage
            t_in, t_out = int(usage.input_tokens or 0), int(usage.output_tokens or 0)
            c_w = int(getattr(usage, "cache_creation_input_tokens", 0) or 0)
            c_r = int(getattr(usage, "cache_read_input_tokens", 0) or 0)
            served_by = getattr(response, "model", None) or model
            tokens_in += t_in + c_w + c_r
            tokens_out += t_out
            usd += call_usd(served_by if served_by in PRICES else model, t_in, t_out, c_w, c_r)
            if response.stop_reason == "refusal":
                details = getattr(response, "stop_details", None)
                category = _block_get(details, "category") if details is not None else None
                error = f"refusal (category {category})"
                break
            try:
                proposals = parse_tool_response(response.content)
                error = None
                break
            except ProposalSchemaError as exc:
                error = str(exc)
                messages.append({"role": "assistant", "content": [_as_param(b) for b in response.content]})
                messages.append({"role": "user", "content": _retry_turn(response.content, error)})
        if error is None:
            self._cache_put(h, model, proposals)
        self._log(LLMCall(prompt_hash=h, model=model, input_tokens=tokens_in, output_tokens=tokens_out, usd=usd,
                          attempts=attempts, n_proposals=len(proposals), error=error, billing="api"))
        return proposals


# ---------------------------------------------------------------- Claude Code CLI (plan billing)
PLAN_AUTH = "claude.ai"
# Read the proposal from the arguments alone: no tools, no CLAUDE.md, hooks, skills or MCP servers,
# no transcript.  ``--bare`` would also do this but accepts only an API key, never the plan login.
CLAUDE_FLAGS = ("-p", "--output-format", "json", "--tools", "", "--safe-mode", "--strict-mcp-config",
                "--no-session-persistence")
# Unset in the child: an API key outranks the plan login, and CLAUDECODE marks a nested session.
SCRUBBED_ENV = ("ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN", "CLAUDECODE")
Runner = Callable[..., subprocess.CompletedProcess]


def find_claude() -> str | None:
    """``$SEA_CLAUDE_BIN``, else ``claude`` on PATH, else the newest VS Code extension's bundled CLI."""
    if os.environ.get("SEA_CLAUDE_BIN"):
        return os.environ["SEA_CLAUDE_BIN"]
    on_path = shutil.which("claude")
    if on_path:
        return on_path
    bundled = Path.home().glob(".vscode/extensions/anthropic.claude-code-*/resources/native-binary/claude")
    newest = max(bundled, key=_extension_version, default=None)
    return str(newest) if newest is not None else None


def cli_schema(schema: Any) -> Any:
    """``schema`` with tuple ``prefixItems`` folded into ``items``: the CLI's strict validator rejects it.

    The answer is validated against the pydantic model anyway, so the looser array costs nothing.
    """
    if isinstance(schema, list):
        return [cli_schema(v) for v in schema]
    if not isinstance(schema, dict):
        return schema
    out = {k: cli_schema(v) for k, v in schema.items() if k != "prefixItems"}
    if "prefixItems" in schema:
        items = [cli_schema(v) for v in schema["prefixItems"]]
        out["items"] = items[0] if all(i == items[0] for i in items) else {"anyOf": items}
    return out


def _extension_version(binary: Path) -> tuple[int, ...]:
    m = re.search(r"claude-code-(\d+(?:\.\d+)*)", str(binary))
    return tuple(int(x) for x in m.group(1).split(".")) if m else ()


class ClaudeCodeLLM(_Cached):
    """Claude as System 2 through ``claude -p``.  Pass ``run`` (a ``subprocess.run`` stand-in) in tests.

    ``require_plan`` makes the first call check ``claude auth status`` and refuse anything but a
    claude.ai login, so a run can never fall back to API billing unnoticed.
    """

    def __init__(self, model: str, cache_dir: Path | None = None, log_path: Path | None = None,
                 binary: str | None = None, effort: str = "medium", require_plan: bool = True,
                 timeout_s: float = 900.0, run: Runner = subprocess.run):
        super().__init__(cache_dir, log_path)
        self.model = model
        self.binary = binary
        self.effort = effort
        self.require_plan = require_plan
        self.timeout_s = timeout_s
        self._run = run
        self._plan_checked = False
        self._cwd: str | None = None

    def _bin(self) -> str:
        found = self.binary or find_claude()
        if not found:
            raise FileNotFoundError("no Claude Code CLI: install it, or set SEA_CLAUDE_BIN to its path")
        return found

    def _env(self) -> dict[str, str]:
        return {k: v for k, v in os.environ.items() if k not in SCRUBBED_ENV}

    def _exec(self, args: list[str], stdin: str | None = None) -> subprocess.CompletedProcess:
        if self._cwd is None:  # an empty directory, so nothing in the repo is read as context
            self._cwd = tempfile.mkdtemp(prefix="sea-claude-")
        return self._run([self._bin(), *args], input=stdin, capture_output=True, text=True,
                         timeout=self.timeout_s, env=self._env(), cwd=self._cwd)

    def check_plan(self) -> dict[str, Any]:
        """``claude auth status`` as a dict; raises ``PlanBillingError`` unless it is a claude.ai login."""
        out = self._exec(["auth", "status"])
        try:
            status = json.loads(out.stdout)
        except json.JSONDecodeError as exc:
            raise PlanBillingError(f"`claude auth status` printed no JSON: {out.stderr.strip()[:200]}") from exc
        if not status.get("loggedIn") or status.get("authMethod") != PLAN_AUTH:
            raise PlanBillingError(f"Claude Code is logged in via {status.get('authMethod')!r}, not a claude.ai "
                                   "plan; run `claude auth login`, or pass require_plan=False to bill the API")
        self._plan_checked = True
        return status

    def command(self, prompt: S2Prompt) -> list[str]:
        return [*CLAUDE_FLAGS, "--model", prompt.model or self.model, "--effort", self.effort,
                "--system-prompt", prompt.system,
                "--json-schema", json.dumps(cli_schema(ProposalBatch.model_json_schema()), sort_keys=True)]

    def _ask(self, prompt: S2Prompt, text: str) -> tuple[dict[str, Any], Any]:
        """One CLI call: (result JSON, the structured output or None)."""
        out = self._exec(self.command(prompt), stdin=text)
        try:
            result = json.loads(out.stdout)
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"`claude -p` exited {out.returncode} without JSON: {out.stderr.strip()[:300]}") from exc
        if result.get("is_error"):
            message = str(result.get("result", ""))
            if result.get("api_error_status") == 429 or "usage limit" in message.lower():
                raise PlanLimitReached(message[:300])
            return result, None
        structured = result.get("structured_output")
        if structured is None:
            try:
                structured = json.loads(result.get("result") or "null")
            except json.JSONDecodeError:
                structured = None
        return result, structured

    def propose(self, prompt: S2Prompt) -> list[Proposal]:
        h = prompt.hash()
        model = prompt.model or self.model
        cached = self._cache_get(h)
        if cached is not None:
            self._log(LLMCall(prompt_hash=h, model=model, cached=True, n_proposals=len(cached)))
            return cached
        if self.require_plan and not self._plan_checked:
            self.check_plan()
        text = prompt.render() + f"\n\nAnswer with a ProposalBatch of 1 to {prompt.max_proposals} proposals."
        tokens_in = tokens_out = attempts = 0
        usd = 0.0
        proposals: list[Proposal] = []
        error: str | None = None
        for _ in range(2):  # first try + one retry on a schema error
            attempts += 1
            result, structured = self._ask(prompt, text)
            usage = result.get("usage") or {}
            t_in, t_out = int(usage.get("input_tokens") or 0), int(usage.get("output_tokens") or 0)
            c_w = int(usage.get("cache_creation_input_tokens") or 0)
            c_r = int(usage.get("cache_read_input_tokens") or 0)
            tokens_in += t_in + c_w + c_r
            tokens_out += t_out
            listed = result.get("total_cost_usd")
            usd += float(listed) if listed is not None else call_usd(model, t_in, t_out, c_w, c_r)
            try:
                if structured is None:
                    raise ProposalSchemaError(f"{result.get('subtype')}: {str(result.get('result', ''))[:200]}")
                proposals = list(ProposalBatch.model_validate(structured).proposals)
                error = None
                break
            except (ProposalSchemaError, ValidationError) as exc:
                error = str(exc)
                text = (prompt.render() + f"\n\nYour previous answer was not a valid ProposalBatch: {error}\n"
                        f"Answer again with a corrected ProposalBatch of 1 to {prompt.max_proposals} proposals.")
        if error is None:
            self._cache_put(h, model, proposals)
        self._log(LLMCall(prompt_hash=h, model=model, input_tokens=tokens_in, output_tokens=tokens_out, usd=usd,
                          attempts=attempts, n_proposals=len(proposals), error=error, billing="plan"))
        return proposals


class MockLLM(_CallLog):
    """Replays ``fixtures_dir/<prompt hash>.json``; raises ``FixtureMissing`` when absent."""

    def __init__(self, fixtures_dir: Path, log_path: Path | None = None):
        super().__init__(log_path)
        self.fixtures_dir = Path(fixtures_dir)

    def fixture_path(self, prompt: S2Prompt) -> Path:
        return self.fixtures_dir / f"{prompt.hash()}.json"

    def propose(self, prompt: S2Prompt) -> list[Proposal]:
        path = self.fixture_path(prompt)
        if not path.exists():
            raise FixtureMissing(f"no MockLLM fixture for prompt hash {prompt.hash()} (expected {path})")
        data = json.loads(path.read_text())
        proposals = [Proposal.model_validate(p) for p in data["proposals"]]
        self._log(LLMCall(prompt_hash=prompt.hash(), model=prompt.model, attempts=1, n_proposals=len(proposals)))
        return proposals


def write_fixture(prompt: S2Prompt, proposals: Sequence[Proposal], fixtures_dir: Path) -> Path:
    """Store a ``MockLLM`` fixture for ``prompt``."""
    fixtures_dir = Path(fixtures_dir)
    fixtures_dir.mkdir(parents=True, exist_ok=True)
    path = fixtures_dir / f"{prompt.hash()}.json"
    body = {"prompt_hash": prompt.hash(), "model": prompt.model,
            "proposals": [p.model_dump(mode="json") for p in proposals]}
    path.write_text(json.dumps(body, sort_keys=True, indent=1))
    return path
