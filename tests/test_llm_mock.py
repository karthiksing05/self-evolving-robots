"""System 2 LLM clients and prompt (spec 8.6; test list 12.2).  No live API calls."""

from __future__ import annotations

import copy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from anthropic.types.beta import BetaMessage

from sea.contracts import (
    Budgets,
    Decision,
    Design,
    FixRecord,
    HardLimits,
    Posture,
    Prediction,
    PrimitiveDef,
    Proposal,
    ToolSpec,
    Vec3,
)
from sea.propose import llm as L
from sea.propose.prompt import SECTION_TITLES, S2Prompt, TraceSummary, build_prompt, request_hash
from sea.spec.posture import derive

RECORDED = Path(__file__).parent / "fixtures" / "llm" / "recorded" / "propose_ok.json"
LIMITS = HardLimits(workspace_box=(Vec3(x=0.10, y=-0.20, z=-0.10), Vec3(x=0.40, y=0.20, z=0.25)))


def _recorded() -> dict:
    return json.loads(RECORDED.read_text())


def _message(data: dict) -> BetaMessage:
    return BetaMessage.model_validate(data)


def _prompt(model: str = "claude-opus-5-5", **kw) -> S2Prompt:
    lib = [PrimitiveDef(name="push", kind="motor", doc="Guarded push.", requires={"contact": "pad|probe|edge"},
                        provides={"actuates": "push_button"}, params={"depth": (0.0005, 0.08)}),
           PrimitiveDef(name="latched", kind="check", doc="Latch engaged.")]
    failing = [TraceSummary(key="p003:c2", cls="door", verdict="no_applicable_primitive", peak_force=0.0,
                            travel_reached=0.0, travel_required=0.26, handle="push_latch", tilt_deg=12.0,
                            escalation_reason="structural", contact_types=["edge"], surprise_z={"peak_force": 0.4})]
    args = dict(limits=LIMITS, eff=derive(Posture(beta=1.0)), remaining={"print_minutes": 0.9, "llm_usd": 0.75},
                budgets=Budgets(), library=lib, design=Design(id="d000"), failing=failing, fixes=[], model=model)
    args.update(kw)
    return build_prompt(**args)


class FakeMessages:
    def __init__(self, responses):
        self.responses = list(responses)
        self.requests: list[dict] = []

    def create(self, **kwargs):
        self.requests.append(copy.deepcopy(kwargs))
        return self.responses.pop(0)


def _client(*responses) -> SimpleNamespace:
    return SimpleNamespace(beta=SimpleNamespace(messages=FakeMessages(responses)))


# ---------------------------------------------------------------- MockLLM
def test_mock_llm_replays_fixtures_and_raises_on_missing(tmp_path):
    prompt = _prompt()
    proposals = L.parse_tool_response(_message(_recorded()).content)
    path = L.write_fixture(prompt, proposals, tmp_path)
    assert path.name == f"{prompt.hash()}.json"
    mock = L.MockLLM(tmp_path, log_path=tmp_path / "llm_calls.jsonl")
    assert mock.propose(prompt) == proposals
    assert mock.calls[0].prompt_hash == prompt.hash() and mock.total_usd == 0.0
    other = _prompt(remaining={"print_minutes": 0.1})
    with pytest.raises(L.FixtureMissing) as err:
        mock.propose(other)
    assert isinstance(err.value, KeyError) and other.hash() in str(err.value)
    assert len((tmp_path / "llm_calls.jsonl").read_text().splitlines()) == 1


# ---------------------------------------------------------------- AnthropicLLM
def test_anthropic_llm_parses_recorded_response(tmp_path):
    client = _client(_message(_recorded()))
    llm = L.AnthropicLLM("claude-opus-5-5", cache_dir=tmp_path / "cache", log_path=tmp_path / "calls.jsonl",
                         client=client)
    out = llm.propose(_prompt())
    assert len(out) == 1 and out[0].kind == "program" and out[0].program is not None
    assert out[0].ruled_out[0].startswith("rung0:")
    call = llm.calls[0]
    assert (call.input_tokens, call.output_tokens, call.attempts, call.error) == (5123 + 3900, 812, 1, None)
    assert call.usd == pytest.approx((4.0 * (5123 + 0.1 * 3900) + 20.0 * 812) / 1e6)
    req = client.beta.messages.requests[0]
    assert "temperature" not in req and "thinking" not in req
    assert req["tool_choice"] == {"type": "auto"}
    assert req["tools"][0]["name"] == "propose" and req["tools"][0]["input_schema"]["type"] == "object"
    assert req["fallbacks"] == "default" and req["betas"] == [L.FALLBACK_BETA]
    assert req["output_config"] == {"effort": "medium"}
    assert "propose" in req["messages"][0]["content"]
    logged = [json.loads(line) for line in (tmp_path / "calls.jsonl").read_text().splitlines()]
    assert logged[0]["n_proposals"] == 1


def test_cache_hit_skips_the_client(tmp_path):
    client = _client(_message(_recorded()))
    llm = L.AnthropicLLM("claude-opus-5-5", cache_dir=tmp_path, client=client)
    first = llm.propose(_prompt())
    second = llm.propose(_prompt())
    assert first == second and len(client.beta.messages.requests) == 1
    assert llm.calls[1].cached and llm.calls[1].usd == 0.0
    assert llm.total_usd == pytest.approx(llm.calls[0].usd)


def test_schema_error_triggers_one_retry(tmp_path):
    bad = _recorded()
    bad["content"][2]["input"] = {"proposals": [{"kind": "program"}]}
    client = _client(_message(bad), _message(_recorded()))
    llm = L.AnthropicLLM("claude-opus-5-5", client=client)
    out = llm.propose(_prompt())
    assert len(out) == 1 and llm.calls[0].attempts == 2 and llm.calls[0].error is None
    reqs = client.beta.messages.requests
    assert len(reqs) == 2
    retry_msgs = reqs[1]["messages"]
    assert [m["role"] for m in retry_msgs] == ["user", "assistant", "user"]
    assert retry_msgs[1]["content"][0]["type"] == "thinking"  # thinking passed back unchanged
    result = retry_msgs[2]["content"][0]
    assert result["type"] == "tool_result" and result["tool_use_id"] == "toolu_01Recorded" and result["is_error"]
    assert reqs[0]["messages"] == retry_msgs[:1]  # history is append-only


def test_missing_tool_call_retries_with_text_then_gives_up(tmp_path):
    text_only = _recorded()
    text_only["content"] = [{"type": "text", "text": "I would rather explain."}]
    text_only["stop_reason"] = "end_turn"
    client = _client(_message(text_only), _message(text_only))
    llm = L.AnthropicLLM("claude-opus-5-5", cache_dir=tmp_path, client=client)
    assert llm.propose(_prompt()) == []
    call = llm.calls[0]
    assert call.attempts == 2 and "no `propose` tool call" in call.error
    assert client.beta.messages.requests[1]["messages"][2]["content"][0]["type"] == "text"
    assert not list(tmp_path.iterdir())  # failures are not cached


def test_refusal_returns_no_proposals():
    refused = {"id": "msg_r", "type": "message", "role": "assistant", "model": "claude-opus-5-5",
               "stop_reason": "refusal", "stop_sequence": None, "content": [],
               "usage": {"input_tokens": 10, "output_tokens": 0}}
    msg = SimpleNamespace(**{**refused, "usage": SimpleNamespace(**refused["usage"]),
                             "stop_details": SimpleNamespace(category="cyber")})
    llm = L.AnthropicLLM("claude-opus-5-5", client=_client(msg))
    assert llm.propose(_prompt()) == []
    assert llm.calls[0].error == "refusal (category cyber)" and llm.calls[0].attempts == 1
    bare = SimpleNamespace(**{**refused, "usage": SimpleNamespace(**refused["usage"]), "model": None})
    llm2 = L.AnthropicLLM("claude-opus-5-5", client=_client(bare))
    assert llm2.propose(_prompt()) == [] and llm2.calls[0].error == "refusal (category None)"


def test_request_kwargs_per_model():
    llm = L.AnthropicLLM("claude-haiku-5-5", client=_client(), effort="low")
    kw = llm.request_kwargs(_prompt(model="claude-haiku-5-5"), [])
    assert "fallbacks" not in kw and "betas" not in kw and kw["output_config"] == {"effort": "low"}
    kw2 = llm.request_kwargs(_prompt(model="claude-sonnet-5-5"), [])
    assert kw2["model"] == "claude-sonnet-5-5" and kw2["fallbacks"] == "default"


def test_parse_tool_response_variants():
    good = _recorded()["content"]
    assert L.parse_tool_response(good)[0].kind == "program"  # plain dict blocks
    with pytest.raises(L.ProposalSchemaError):
        L.parse_tool_response([{"type": "tool_use", "name": "other", "input": {}}])
    too_many = copy.deepcopy(good)
    too_many[2]["input"]["proposals"] *= 4
    with pytest.raises(L.ProposalSchemaError):
        L.parse_tool_response(too_many)
    ns = [SimpleNamespace(type="tool_use", name="propose", id="t1", input=good[2]["input"], caller=None)]
    assert L.parse_tool_response(ns)[0].rung == 1
    assert L._as_param(ns[0]) == {"type": "tool_use", "name": "propose", "id": "t1", "input": good[2]["input"]}
    assert L._as_param({"type": "text", "text": "x"}) == {"type": "text", "text": "x"}


def test_pricing():
    assert L.call_usd("claude-sonnet-5-5", 1_000_000, 1_000_000) == pytest.approx(12.0)
    assert L.call_usd("claude-haiku-5-5", 1_000_000, 0, cache_write=1_000_000) == pytest.approx(0.1 + 0.125)
    assert L.call_usd("mystery-model", 1_000_000, 0) == pytest.approx(4.0)  # priced as the dearest known
    assert L.request_hash({"b": 1, "a": 2}) == request_hash({"a": 2, "b": 1})


# ---------------------------------------------------------------- prompt
def test_prompt_has_eight_sections_and_stable_hash():
    p = _prompt()
    assert [t for t, _ in p.sections] == list(SECTION_TITLES) and len(p.sections) == 8
    assert p.hash() == _prompt().hash()
    assert p.hash() != _prompt(remaining={"print_minutes": 0.5, "llm_usd": 0.75}).hash()
    assert p.target_keys == ["p003:c2"]
    text = p.render()
    assert text.index("## 1. Hard limits") < text.index("## 8. Output schema and rules")
    assert "banned contact types: pinch" in text and "rung<i>:" in text
    assert "push [motor] Guarded push." in text and "depth=[0.0005, 0.08]" in text
    assert "p003:c2 cls=door handle=push_latch" in text and "surprise_z=peak_force=0.4" in text
    assert "(no similar past fixes yet)" in text and "rack: empty" in text


def test_prompt_lists_fixes_rack_and_handles_empty_inputs():
    prop = Proposal(kind="new_tool", rung=3, rationale="r", hypothesis="tab needs a hook, not a pinch",
                    ruled_out=["rung0: a", "rung1: b", "rung2: c"], prediction=Prediction(confidence=0.4))
    dec = Decision(accepted=True, rule="mean", delta_mean=0.2, delta_lo=0.05, delta_hi=0.3, prediction_score=1.0)
    fixes = [FixRecord(id=f"fix{i}", iteration=i, proposal=prop, decision=dec, distance=0.1 * (7 - i))
             for i in range(7)]
    rejected = FixRecord(id="fixr", iteration=2, proposal=prop, decision=dec.model_copy(update={"accepted": False}),
                         distance=0.0)
    design = Design(id="d004", parent="d003",
                    tool_rack=[ToolSpec(template="hook", params={"depth": 8.0}, contact="hook")])
    p = _prompt(fixes=[*fixes, rejected], design=design, failing=[], library=[], remaining={})
    body = dict(p.sections)
    assert body["Similar past fixes"].count("- fix") == 5
    assert "fixr" in body["Similar past fixes"] and "-> rejected" in body["Similar past fixes"]
    assert "fix0" not in body["Similar past fixes"]  # farthest dropped
    assert "rack[0]: hook contact=hook depth=8" in body["Current design"]
    assert "parent d003" in body["Current design"]
    assert body["Failing controls"] == "(none)" and body["Primitive library"] == "(empty)"
    assert "remaining budget fractions: -" in body["Posture and remaining budgets"]
    assert p.target_keys == []
