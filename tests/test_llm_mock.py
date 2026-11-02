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
    FingerSpec,
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


# ---------------------------------------------------------------- ClaudeCodeLLM
PLAN_STATUS = {"loggedIn": True, "authMethod": "claude.ai", "subscriptionType": "max"}


def _cli_result(structured=None, **kw) -> dict:
    """The JSON ``claude -p --output-format json`` prints, trimmed to the fields the client reads."""
    out = {"type": "result", "subtype": "success", "is_error": False, "num_turns": 2, "total_cost_usd": 0.0612,
           "usage": {"input_tokens": 3, "cache_creation_input_tokens": 6100, "cache_read_input_tokens": 0,
                     "output_tokens": 840},
           "result": json.dumps(structured) if structured is not None else "", "structured_output": structured}
    out.update(kw)
    return out


class FakeCLI:
    """Stands in for ``subprocess.run``: answers ``auth status`` and then each ``-p`` call in order."""

    def __init__(self, *results, status=PLAN_STATUS):
        self.results = list(results)
        self.status = status
        self.calls: list[dict] = []

    def __call__(self, args, **kw):
        self.calls.append({"args": list(args), **kw})
        body = self.status if args[1:3] == ["auth", "status"] else self.results.pop(0)
        stdout = body if isinstance(body, str) else json.dumps(body)
        return SimpleNamespace(stdout=stdout, stderr="", returncode=0)


def _batch() -> dict:
    return _recorded()["content"][2]["input"]


def test_claude_code_llm_checks_the_plan_and_reads_structured_output(tmp_path, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-should-not-reach-the-cli")
    cli = FakeCLI(_cli_result(_batch()))
    llm = L.ClaudeCodeLLM("claude-opus-5-5", cache_dir=tmp_path / "cache", log_path=tmp_path / "calls.jsonl",
                          binary="/bin/claude", run=cli)
    out = llm.propose(_prompt())
    assert len(out) == 1 and out[0].kind == "program"
    status, ask = cli.calls
    assert status["args"] == ["/bin/claude", "auth", "status"]
    args = ask["args"]
    assert args[:2] == ["/bin/claude", "-p"] and args[args.index("--tools") + 1] == ""
    assert "--safe-mode" in args and "--no-session-persistence" in args and "--bare" not in args
    assert args[args.index("--model") + 1] == "claude-opus-5-5" and args[args.index("--effort") + 1] == "medium"
    assert json.loads(args[args.index("--json-schema") + 1])["properties"]["proposals"]["maxItems"] == 3
    assert args[args.index("--system-prompt") + 1] == _prompt().system
    assert ask["input"].startswith("## 1.") and "1 to 3 proposals" in ask["input"]
    assert "ANTHROPIC_API_KEY" not in ask["env"] and "CLAUDECODE" not in ask["env"]
    assert ask["cwd"] == status["cwd"] and not any(Path(ask["cwd"]).iterdir())
    call = llm.calls[0]
    assert (call.input_tokens, call.output_tokens, call.attempts, call.billing) == (6103, 840, 1, "plan")
    assert call.usd == pytest.approx(0.0612)  # list price, so the ledger charges it like an API call
    assert llm.propose(_prompt()) == out and len(cli.calls) == 2  # cached: no second CLI call
    assert llm.calls[1].cached and llm.calls[1].billing is None


def test_claude_code_llm_refuses_api_billing():
    cli = FakeCLI(status={"loggedIn": True, "authMethod": "api_key"})
    llm = L.ClaudeCodeLLM("claude-opus-5-5", binary="claude", run=cli)
    with pytest.raises(L.PlanBillingError, match="api_key"):
        llm.propose(_prompt())
    with pytest.raises(L.PlanBillingError, match="no JSON"):
        L.ClaudeCodeLLM("claude-opus-5-5", binary="claude", run=FakeCLI(status="not json")).check_plan()
    opted_in = L.ClaudeCodeLLM("claude-opus-5-5", binary="claude", require_plan=False,
                               run=FakeCLI(_cli_result(_batch()), status={"authMethod": "api_key"}))
    assert len(opted_in.propose(_prompt())) == 1 and len(opted_in._run.calls) == 1  # no status check


def test_claude_code_llm_retries_a_bad_batch_then_falls_back_to_result_text(tmp_path):
    bad = _cli_result({"proposals": [{"kind": "program"}]})
    text_only = _cli_result(None, result=json.dumps(_batch()), total_cost_usd=None)
    cli = FakeCLI(bad, text_only)
    llm = L.ClaudeCodeLLM("claude-sonnet-5-5", cache_dir=tmp_path, binary="claude", run=cli)
    out = llm.propose(_prompt(model="claude-sonnet-5-5"))
    assert len(out) == 1 and llm.calls[0].attempts == 2 and llm.calls[0].error is None
    assert "not a valid ProposalBatch" in cli.calls[2]["input"]
    assert llm.calls[0].usd == pytest.approx(0.0612 + L.call_usd("claude-sonnet-5-5", 3, 840, 6100, 0))


def test_claude_code_llm_errors_and_limits(tmp_path):
    failed = _cli_result(None, is_error=True, subtype="error_max_structured_output_retries", result="gave up")
    llm = L.ClaudeCodeLLM("claude-opus-5-5", cache_dir=tmp_path, binary="claude", run=FakeCLI(failed, failed))
    assert llm.propose(_prompt()) == []
    assert llm.calls[0].attempts == 2 and "error_max_structured_output_retries" in llm.calls[0].error
    assert not list(tmp_path.iterdir())  # failures are not cached
    limited = _cli_result(None, is_error=True, result="Claude AI usage limit reached|1793500000")
    with pytest.raises(L.PlanLimitReached):
        L.ClaudeCodeLLM("claude-opus-5-5", binary="claude", run=FakeCLI(limited)).propose(_prompt())
    with pytest.raises(L.PlanLimitReached):
        L.ClaudeCodeLLM("claude-opus-5-5", binary="claude",
                        run=FakeCLI(_cli_result(None, is_error=True, api_error_status=429))).propose(_prompt())
    with pytest.raises(RuntimeError, match="without JSON"):
        L.ClaudeCodeLLM("claude-opus-5-5", binary="claude", run=FakeCLI("Error: crashed")).propose(_prompt())


def test_cli_schema_folds_tuples():
    tup = {"type": "array", "minItems": 2, "maxItems": 2, "prefixItems": [{"type": "number"}, {"type": "number"}]}
    assert L.cli_schema({"a": [tup]}) == {"a": [{"type": "array", "minItems": 2, "maxItems": 2,
                                                  "items": {"type": "number"}}]}
    mixed = {"prefixItems": [{"type": "number"}, {"type": "string"}]}
    assert L.cli_schema(mixed) == {"items": {"anyOf": [{"type": "number"}, {"type": "string"}]}}
    assert "prefixItems" not in json.dumps(L.cli_schema(L.ProposalBatch.model_json_schema()))


def test_find_claude(monkeypatch, tmp_path):
    monkeypatch.setenv("SEA_CLAUDE_BIN", "/opt/claude")
    assert L.find_claude() == "/opt/claude"
    monkeypatch.delenv("SEA_CLAUDE_BIN")
    monkeypatch.setattr(L.shutil, "which", lambda name: None)
    monkeypatch.setattr(L.Path, "home", lambda: tmp_path)
    assert L.find_claude() is None
    with pytest.raises(FileNotFoundError, match="SEA_CLAUDE_BIN"):
        L.ClaudeCodeLLM("claude-opus-5-5", require_plan=False).propose(_prompt())
    for v in ("2.1.9", "2.1.314"):
        b = tmp_path / f".vscode/extensions/anthropic.claude-code-{v}-darwin-arm64/resources/native-binary/claude"
        b.parent.mkdir(parents=True)
        b.touch()
    assert L.find_claude().endswith("claude-code-2.1.314-darwin-arm64/resources/native-binary/claude")  # not by name
    monkeypatch.setattr(L.shutil, "which", lambda name: "/usr/local/bin/claude")
    assert L.find_claude() == "/usr/local/bin/claude"


# ---------------------------------------------------------------- prompt
def test_prompt_has_ten_sections_and_stable_hash():
    p = _prompt()
    assert [t for t, _ in p.sections] == list(SECTION_TITLES) and len(p.sections) == 10
    assert p.hash() == _prompt().hash()
    assert p.hash() != _prompt(remaining={"print_minutes": 0.5, "llm_usd": 0.75}).hash()
    assert p.target_keys == ["p003:c2"]
    text = p.render()
    assert text.index("## 1. Hard limits") < text.index("## 10. Output schema and rules")
    assert "jaw_pads [mounted" in text and "link_skin [NOT mounted" in text  # the sensor manifest (D-033)
    assert "banned contact types: pinch" in text and "rung<i>:" in text
    assert "push [motor] Guarded push." in text and "depth=[0.0005, 0.08]" in text
    assert "p003:c2 cls=door handle=push_latch" in text and "surprise_z=peak_force=0.4" in text
    assert "(no similar past fixes yet)" in text and "rack: empty" in text
    assert "widest grip by shape: box=60, cylinder=55, sphere=50 mm (V1 capability table)" in text
    assert "pad_angle_deg [0, 45]" in text and "## 9. Rejected before trial\n(none yet)" in text


def test_prompt_shows_what_is_touched_and_what_screening_refused():
    from sea.propose.prompt import Rejection

    s = TraceSummary(key="t:obj", cls="grasp", verdict="stuck", peak_force=9.0, travel_reached=0.0,
                     travel_required=0.03, accepts=["pinch"], object="tennis_ball", size_mm=(66.5, 66.4), mass_g=58.0)
    no = Rejection(iteration=1, kind="new_tool", hypothesis="a hook lifts the ball",
                   errors=["t:obj: no rack tool provides the contacts this program needs"])
    text = _prompt(failing=[s], rejected=[no]).render()
    assert "object=tennis_ball width_mm=66.5 height_mm=66.4 mass_g=58 accepted_contacts=pinch" in text
    assert "- iter 1 new_tool: a hook lifts the ball\n    refused: t:obj: no rack tool" in text
    fingers = _prompt(design=Design(id="f", fingers=FingerSpec(reach_mm=10, pad_tpu_mm=2))).render()
    assert "(V1 probe, measured on these fingers)" in fingers and "pad_tpu_mm=2" in fingers


def test_prompt_lists_fixes_rack_and_handles_empty_inputs():
    prop = Proposal(kind="new_tool", rung=3, rationale="r", hypothesis="tab needs a hook, not a pinch",
                    ruled_out=["rung0: a", "rung1: b", "rung2: c"], prediction=Prediction(confidence=0.4))
    dec = Decision(accepted=True, rule="mean", delta_mean=0.2, delta_lo=0.05, delta_hi=0.3, prediction_score=1.0)
    fixes = [FixRecord(id=f"fix{i}", iteration=i, proposal=prop, decision=dec, distance=0.1 * (7 - i))
             for i in range(7)]
    rejected = FixRecord(id="fixr", iteration=2, proposal=prop,
                         decision=dec.model_copy(update={"accepted": False, "regressions": ["reg_banana:obj"]}),
                         distance=0.0)
    design = Design(id="d004", parent="d003",
                    tool_rack=[ToolSpec(template="hook", params={"depth": 8.0}, contact="hook")])
    p = _prompt(fixes=[*fixes, rejected], design=design, failing=[], library=[], remaining={})
    body = dict(p.sections)
    assert body["Similar past fixes"].count("- fix") == 5
    assert "fixr" in body["Similar past fixes"] and "-> rejected" in body["Similar past fixes"]
    assert "regressions 1, prediction score 1)\n    broke: reg_banana:obj" in body["Similar past fixes"]
    assert "fix0" not in body["Similar past fixes"]  # farthest dropped
    assert "rack[0]: hook contact=hook depth=8" in body["Current design"]
    assert "parent d003" in body["Current design"]
    assert body["Failing controls"] == "(none)" and body["Primitive library"] == "(empty)"
    assert "remaining budget fractions: -" in body["Posture and remaining budgets"]
    assert p.target_keys == []
