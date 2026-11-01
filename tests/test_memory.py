"""Cobweb concept memory (spec 8.5, 12.2 test_memory)."""

from __future__ import annotations

import numpy as np
import pytest

from sea.compose import dsl
from sea.contracts import D_DUP, MEASUREMENTS, ControlResult, Decision, Prediction, Program, Proposal
from sea.memory.concepts import SD_FLOOR, ConceptMemory, _mean_sd, _Welford

DIM = 64


def _centers() -> tuple[np.ndarray, np.ndarray]:
    mu_b = np.zeros(DIM, dtype=np.float32)
    mu_b[:8] = 2.0
    return mu_b, -mu_b


def result(cid: str, cls: str, force: float, verdict: str = "pass", panel: str = "p0") -> ControlResult:
    button = cls == "push_button"
    return ControlResult(
        panel_id=panel, control_id=cid, cls=cls, verdict=verdict, peak_force=force,
        travel_reached=0.003 if button else 0.6, detents=1 if button else 0, time_s=4.0 if button else 6.0,
    )


def two_clusters(seed: int = 0):
    """60 buttons with force ~ N(2.9, 0.4) and 40 levers ~ N(6, 0.8), well separated in context."""
    rng = np.random.default_rng(seed)
    mu_b, mu_l = _centers()
    results, ctx = [], {}
    for i in range(60):
        r = result(f"b{i}", "push_button", float(rng.normal(2.9, 0.4)))
        results.append(r)
        ctx[r.key] = (mu_b + rng.normal(0, 0.3, DIM)).astype(np.float32)
    for i in range(40):
        r = result(f"l{i}", "lever", float(rng.normal(6.0, 0.8)))
        results.append(r)
        ctx[r.key] = (mu_l + rng.normal(0, 0.3, DIM)).astype(np.float32)
    order = rng.permutation(len(results))
    return [results[i] for i in order], ctx


@pytest.fixture(scope="module")
def memory():
    results, ctx = two_clusters()
    mem = ConceptMemory(seed=1)
    mem.learn(results, set(), ctx, iteration=0)
    return mem


def measured(force: float) -> dict[str, float]:
    return {"peak_force": force, "travel_reached": 0.003, "detents": 1.0, "time_s": 4.0, "ik_residual_mm": 0.0}


def test_expectation_uses_ancestor_with_n_min(memory):
    mu_b, mu_l = _centers()
    exp = memory.expect(mu_b)
    assert exp.n_results >= 20
    t = memory.trees["context"]
    leaf = t.leaf(mu_b)
    assert len(t.payloads_under(leaf)) < exp.n_results  # walked up from a small leaf
    assert exp.mean["peak_force"] == pytest.approx(2.9, abs=0.2)
    lever = memory.expect(mu_l)
    assert lever.n_results >= 20 and lever.mean["peak_force"] == pytest.approx(6.0, abs=0.4)
    assert memory.concept_id(mu_b) != memory.concept_id(mu_l)
    assert 0 < memory.concept_n(mu_b) <= 60


def test_button_at_cluster_mean_is_not_surprising(memory):
    mu_b, _ = _centers()
    z, nll = memory.surprise(mu_b, measured(2.9))
    assert abs(z["peak_force"]) < 1.0
    assert set(z) == set(MEASUREMENTS)


def test_bound_button_is_surprising(memory):
    mu_b, _ = _centers()
    z, nll = memory.surprise(mu_b, measured(8.0))
    assert z["peak_force"] > 5.0
    assert nll > memory.surprise(mu_b, measured(2.9))[1]


def test_uncertainty_tracks_spread(memory):
    mu_b, mu_l = _centers()
    ub, ul = memory.uncertainty(mu_b), memory.uncertainty(mu_l)
    assert 0 < ub < ul


def test_save_load_round_trip(memory, tmp_path):
    p = Proposal(kind="program", rung=1, rationale="push slower", hypothesis="detent missed",
                 ruled_out=["rung0: tried"], prediction=Prediction(flips_to_pass=["p0:b1"], confidence=0.6))
    memory.record_fix("fix0", p, Decision(accepted=True, rule="mean", delta_mean=0.1, delta_lo=0.0, delta_hi=0.2), 3)
    prog = Program(id="prog0", target_cls="push_button", root=dsl.parse("Seq(approach($target), retreat())"),
                   source="s2")
    memory.add_program(prog, {"push_button": 1.0})
    memory.learn([result("pend", "push_button", 9.0, verdict="stuck")], set(),
                 {"p0:pend": _centers()[0]}, iteration=4)
    memory.save(tmp_path / "mem")
    for name in ("context.json", "full.json", "fix.json", "program.json", "sidecar.json"):
        assert (tmp_path / "mem" / name).exists()
    back = ConceptMemory().load(tmp_path / "mem")
    for name in ("context", "full", "fix", "program"):
        assert back.tree_json(name) == memory.tree_json(name)
    rng = np.random.default_rng(7)
    for _ in range(5):
        q = rng.normal(0, 2, DIM).astype(np.float32)
        assert back.expect(q) == memory.expect(q)
        assert back.surprise(q, measured(4.0)) == memory.surprise(q, measured(4.0))
    assert back.similar_fixes("detent missed") == memory.similar_fixes("detent missed")
    assert back.similar_programs(prog, {"push_button": 1.0}) == memory.similar_programs(prog, {"push_button": 1.0})
    assert set(back.pending) == {"p0:pend"}


def test_unconfirmed_failures_are_not_learned():
    mu_b, _ = _centers()
    mem = ConceptMemory(seed=0)
    ok = result("a", "push_button", 3.0)
    bad = result("b", "push_button", 9.5, verdict="stuck")
    ctx = {ok.key: mu_b, bad.key: mu_b + 0.1}
    mem.learn([ok, bad], set(), ctx)
    assert mem.trees["context"].tree.root.count == 1
    assert list(mem.pending) == [bad.key]
    # A later call confirms the queued failure.
    mem.learn([], {bad.key}, ctx, iteration=1)
    assert mem.trees["context"].tree.root.count == 2 and not mem.pending
    verdicts = [p["verdict"] for p in mem.trees["context"].payloads_under(mem.trees["context"].tree.root)]
    assert sorted(verdicts) == ["pass", "stuck"]


def test_pending_failure_dropped_when_control_later_passes():
    mu_b, _ = _centers()
    mem = ConceptMemory()
    bad = result("b", "push_button", 9.5, verdict="no_actuation")
    mem.learn([bad], set(), {bad.key: mu_b})
    good = result("b", "push_button", 3.0)
    mem.learn([good], {bad.key}, {bad.key: mu_b})
    assert not mem.pending
    assert mem.trees["context"].tree.root.count == 1


def test_confirmed_failure_added_immediately():
    mu_b, _ = _centers()
    mem = ConceptMemory()
    bad = result("b", "push_button", 9.5, verdict="stuck")
    mem.learn([bad], {bad.key}, {bad.key: mu_b})
    assert mem.trees["full"].tree.root.count == 1 and not mem.pending


def test_missing_context_raises():
    with pytest.raises(KeyError):
        ConceptMemory().learn([result("x", "push_button", 3.0)], set(), {})


def test_same_seed_gives_identical_trees():
    results, ctx = two_clusters(seed=4)
    a, b = ConceptMemory(seed=2), ConceptMemory(seed=2)
    a.learn(results, set(), ctx)
    b.learn(results, set(), ctx)
    for name in ("context", "full"):
        assert a.tree_json(name) == b.tree_json(name)


def test_empty_memory_defaults():
    mem = ConceptMemory()
    q = np.zeros(DIM, dtype=np.float32)
    exp = mem.expect(q)
    assert exp.n_results == 0 and exp.concept_n == 0
    z, nll = mem.surprise(q, measured(3.0))
    assert nll == 0.0 and all(v == 0.0 for v in z.values())
    assert mem.uncertainty(q) == 2.0
    assert mem.similar_fixes("anything") == []
    prog = Program(id="p", target_cls="knob", root=dsl.parse("retreat()"), source="s2")
    assert mem.similar_programs(prog) == []


def _proposal(hyp: str, rationale: str) -> Proposal:
    return Proposal(kind="program", rung=1, rationale=rationale, hypothesis=hyp, ruled_out=["rung0: no"],
                    prediction=Prediction(confidence=0.5))


def test_similar_fixes_returns_nearest_first():
    mem = ConceptMemory(seed=3)
    texts = [
        ("knob needs cup tool", "pinch banned on knob"),
        ("door latch stiff", "pull harder with hook"),
        ("button too deep in recess", "probe reaches bottom"),
        ("slider reversed direction", "slide negative distance"),
        ("lever out of reach", "approach at 45 degrees"),
        ("toggle detent missed", "push past detent"),
    ]
    for i, (h, r) in enumerate(texts):
        dec = Decision(accepted=i % 2 == 0, rule="mean", delta_mean=0.0, delta_lo=0.0, delta_hi=0.0)
        mem.record_fix(f"f{i}", _proposal(h, r), dec, iteration=i)
    hits = mem.similar_fixes("button too deep in recess probe reaches bottom", k=3)
    assert len(hits) == 3
    assert hits[0].id == "f2" and hits[0].distance == pytest.approx(0.0, abs=1e-5)
    assert hits[0].iteration == 2 and hits[0].proposal.hypothesis == "button too deep in recess"
    assert [h.distance for h in hits] == sorted(h.distance for h in hits)
    assert mem.similar_fixes("x", k=0) == []


def test_program_tree_deduplicates():
    mem = ConceptMemory(seed=5)
    progs = {
        "press": "Seq(approach($target), contact($target), push($target), retreat())",
        "knob": "Seq(approach($target), Repeat(rotate($target), until=moved_at_least($target, travel)))",
        "door": "Seq(approach($target), With(1, hook_pull($target)), retreat())",
        "slide": "Seq(approach($target), contact($target), slide($target), retreat())",
    }
    for pid, text in progs.items():
        mem.add_program(Program(id=pid, target_cls="push_button", root=dsl.parse(text), source="seed"),
                        {"push_button": 1.0} if pid == "press" else {})
    same = Program(id="new", target_cls="push_button", root=dsl.parse(progs["press"]), source="s2")
    hits = mem.similar_programs(same, {"push_button": 1.0}, k=2)
    assert hits[0][0] == "press" and hits[0][1] < D_DUP
    other = Program(id="o", target_cls="slot", root=dsl.parse("Seq(insert($target), release(), retreat())"),
                    source="s2")
    assert mem.similar_programs(other, k=1)[0][1] > D_DUP
    # An outcome difference alone moves a program by 0.5 x |delta signature|.
    no_outcome = mem.similar_programs(same, None, k=4)
    assert dict(no_outcome)["press"] == pytest.approx(0.5, abs=1e-4)


def test_build_seed_keeps_lowest_validation_nll():
    results, ctx = two_clusters(seed=6)
    train, val = results[:80], results[80:]
    best = ConceptMemory.build_seed(train, ctx, val, ctx, seed=11)
    assert best.trees["context"].tree.root.count == 80

    def score(mem):
        return float(np.mean([mem.surprise(ctx[r.key], {m: getattr(r, m) for m in MEASUREMENTS})[1] for r in val]))

    rng = np.random.default_rng(11)
    scores = []
    for _ in range(3):
        order = rng.permutation(len(train))
        mem = ConceptMemory(seed=11)
        mem.learn([train[j] for j in order], set(), ctx)
        scores.append(score(mem))
    assert score(best) == pytest.approx(min(scores))


def test_build_seed_with_failures_and_no_validation():
    mu_b, _ = _centers()
    r = [result("a", "push_button", 3.0), result("b", "push_button", 9.0, verdict="stuck")]
    mem = ConceptMemory.build_seed(r, {x.key: mu_b for x in r}, [], {}, seed=0, n_shuffles=2)
    assert mem.trees["context"].tree.root.count == 2


def test_log_prob_fallback_far_from_everything(memory):
    far = np.full(DIM + len(MEASUREMENTS), 1e4, dtype=np.float32)
    lp = memory.trees["full"].log_prob(far)
    assert np.isfinite(lp) and lp < -1e6


def test_helpers():
    assert _mean_sd([]) == (0.0, 0.0)
    assert _mean_sd([2.0]) == (2.0, 0.0)
    w = _Welford()
    assert w.sd == 0.0
    for x in (1.0, 2.0, 3.0):
        w.add(x)
    assert w.sd == pytest.approx(1.0)
    assert _Welford.from_list(w.to_list()).mean == pytest.approx(2.0)
    assert set(SD_FLOOR) == set(MEASUREMENTS)


def test_identical_instances_share_a_leaf():
    mu_b, _ = _centers()
    mem = ConceptMemory()
    a, b = result("a", "push_button", 3.0), result("b", "push_button", 3.2)
    mem.learn([a, b], set(), {a.key: mu_b, b.key: mu_b})
    t = mem.trees["context"]
    leaf = t.leaf(mu_b)
    assert leaf is t.tree.root and len(t.payloads_under(leaf)) == 2
