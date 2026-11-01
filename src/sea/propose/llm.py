"""System 2 LLM clients (spec Section 8.6, deviations in docs/decisions.md D-002).

``AnthropicLLM`` asks Claude for a ``ProposalBatch`` through a tool named ``propose``.  Forced
tool choice and sampling parameters are rejected by the current models, so the request uses
``tool_choice: auto`` with an instruction naming the tool, and the tool input is validated
client-side; a missing or invalid call is retried once with the error appended (history stays
append-only).  Responses are cached by prompt hash, and every call is logged with tokens and USD.
``MockLLM`` replays fixtures and is the only LLM tests may use.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Protocol

from pydantic import BaseModel, ConfigDict, ValidationError

from sea.contracts import Proposal, ProposalBatch
from sea.propose.prompt import S2Prompt, request_hash

__all__ = [
    "PRICES",
    "TOOL_NAME",
    "AnthropicLLM",
    "FixtureMissing",
    "LLM",
    "LLMCall",
    "MockLLM",
    "ProposalSchemaError",
    "call_usd",
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


class AnthropicLLM(_CallLog):
    """Claude as System 2.  Pass ``client`` to inject a fake in tests."""

    def __init__(self, model: str, cache_dir: Path | None = None, log_path: Path | None = None, client: Any = None,
                 effort: str = "medium", max_tokens: int = 16000):
        super().__init__(log_path)
        self.model = model
        self.cache_dir = Path(cache_dir) if cache_dir is not None else None
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

    # ------------------------------------------------------------ cache
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
                          attempts=attempts, n_proposals=len(proposals), error=error))
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
