"""Cobweb concept memory (spec 3.7, 8.5).

Four ``CPPCobwebTree``s, all on CPU:

======== ===================================================== ==========================
Tree     Instance                                              Leaf payload
======== ===================================================== ==========================
context  64-d whitened context vector                          ControlResult ids + measurements
full     context ⊕ z-scored measurements (69-d)                ControlResult ids
fix      text_emb(hypothesis + rationale), 384-d               (Proposal, Decision) ids
program  unit text_emb(pretty program) ⊕ 0.5 × outcome (393-d) Program ids
======== ===================================================== ==========================

Payloads live on the leaf object ``ifit`` returns; Cobweb never replaces a leaf object on insert
(docs/decisions.md D-003).  Every tree call runs inside :func:`_seeded`, which seeds Python's
``random`` (Cobweb's tie-breaks) and torch from the memory seed, the tree, and the tree's insert
count, so the same calls in the same order build byte-identical trees.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import math
import random
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from cobweb_language_embedding.CPPCobweb import CPPCobwebNode, CPPCobwebTree

from sea.compose import dsl
from sea.contracts import (
    CONTROL_CLASSES,
    MEASUREMENTS,
    N_MIN,
    ControlResult,
    Decision,
    FixRecord,
    Program,
    Proposal,
)
from sea.memory.embed import EMB_DIM, Embedder, FakeEmbedder

TREES = ("context", "full", "fix", "program")
_BASE_PRIOR_VAR = 1.0 / (2.0 * math.e * math.pi)  # Cobweb's default acuity
_CATEGORIZE_NODES = 64  # best-first expansions before descending greedily to a leaf
_LOGPROB_NODES = 32
# Floors on a concept's sd when computing z (units of each measurement): docs/decisions.md M-004.
SD_FLOOR = {"peak_force": 0.05, "travel_reached": 1e-4, "detents": 0.25, "time_s": 0.05, "ik_residual_mm": 0.1}


@dataclass
class Expectation:
    mean: dict[str, float]
    sd: dict[str, float]
    concept_n: int
    node_id: int
    n_results: int


def _measurements(r: ControlResult) -> dict[str, float]:
    return {m: float(getattr(r, m)) for m in MEASUREMENTS}


@contextlib.contextmanager
def _seeded(seed: int) -> Iterator[None]:
    py_state = random.getstate()
    torch_state = torch.get_rng_state()
    random.seed(seed)
    torch.manual_seed(seed)
    try:
        yield
    finally:
        random.setstate(py_state)
        torch.set_rng_state(torch_state)


def _dfs(tree: CPPCobwebTree) -> list[CPPCobwebNode]:
    """Nodes in ``dump_json`` order (pre-order, children in order)."""
    out: list[CPPCobwebNode] = []
    stack = [tree.root]
    while stack:
        node = stack.pop()
        out.append(node)
        stack.extend(reversed(node.children))
    return out


class _Tree:
    """One Cobweb tree plus its payloads and a per-node payload cache."""

    def __init__(self, name: str, dim: int, prior_var: float, seed: int):
        self.name = name
        self.dim = dim
        self.prior_var = prior_var
        self.seed = seed
        self.tree = CPPCobwebTree(dim, device="cpu", prior_var=prior_var)
        self.inserts = 0
        self._cache: dict[int, list[dict[str, Any]]] | None = None
        self._index: dict[int, int] | None = None

    def _rng_seed(self, n: int) -> int:
        digest = hashlib.sha256(f"{self.seed}:{self.name}:{n}".encode()).digest()
        return int.from_bytes(digest[:4], "little")

    @property
    def empty(self) -> bool:
        return self.tree.root.count <= 0

    def insert(self, vec: np.ndarray, label: str | None, payload: dict[str, Any]) -> None:
        with _seeded(self._rng_seed(self.inserts)):
            leaf = self.tree.ifit(torch.as_tensor(np.asarray(vec, dtype=np.float32)), label)
        assert not leaf.children, "Cobweb returned a non-leaf from ifit"
        if not hasattr(leaf, "sea_payloads"):
            leaf.sea_payloads = []
        leaf.sea_payloads.append(payload)
        self.inserts += 1
        self._cache = None
        self._index = None

    def leaf(self, vec: np.ndarray) -> CPPCobwebNode:
        """Categorize to a single leaf: best node of a bounded best-first search, then greedy descent."""
        x = torch.as_tensor(np.asarray(vec, dtype=np.float32))
        with _seeded(self._rng_seed(-1)), torch.no_grad():
            node = self.tree.categorize(x, use_best=True, max_nodes=_CATEGORIZE_NODES)
            labels = torch.zeros(self.tree.num_labels)
            while node.children:
                node = max(node.children, key=lambda c: c.log_prob_class_given_instance(x, labels))
        return node

    def payloads_under(self, node: CPPCobwebNode) -> list[dict[str, Any]]:
        if self._cache is None:
            self._cache = {}
            for n in reversed(_dfs(self.tree)):  # children before parents
                acc = list(getattr(n, "sea_payloads", []))
                for c in n.children:
                    acc.extend(self._cache[id(c)])
                self._cache[id(n)] = acc
        return self._cache[id(node)]

    def index_of(self, node: CPPCobwebNode) -> int:
        if self._index is None:
            self._index = {id(n): i for i, n in enumerate(_dfs(self.tree))}
        return self._index[id(node)]

    def log_prob(self, vec: np.ndarray) -> float:
        x = torch.as_tensor(np.asarray(vec, dtype=np.float32))
        with _seeded(self._rng_seed(-2)), torch.no_grad():
            labels = torch.zeros(self.tree.num_labels)
            lp = self.tree.log_prob(x, labels, _LOGPROB_NODES, False)
            if not math.isfinite(lp):  # Cobweb's running log-sum-exp underflows far from every concept
                lp = max(n.log_prob(x, labels) for n in _dfs(self.tree) if n.count > 0)
        return float(lp)

    # ---- persistence
    def save(self, directory: Path) -> dict[str, Any]:
        self.tree.dump_json(str(directory / f"{self.name}.json"))
        payloads = {
            str(i): n.sea_payloads for i, n in enumerate(_dfs(self.tree)) if getattr(n, "sea_payloads", None)
        }
        return {"labels": dict(self.tree.labels), "payloads": payloads, "inserts": self.inserts}

    def load(self, directory: Path, side: dict[str, Any]) -> None:
        self.tree = CPPCobwebTree(self.dim, device="cpu", prior_var=self.prior_var)
        self.tree.load_json(str(directory / f"{self.name}.json"))
        self.tree.labels = {k: int(v) for k, v in side["labels"].items()}
        self.tree.reverse_labels = {v: k for k, v in self.tree.labels.items()}
        nodes = _dfs(self.tree)
        for idx, plist in side["payloads"].items():
            nodes[int(idx)].sea_payloads = list(plist)
        self.inserts = int(side["inserts"])
        self._cache = None
        self._index = None


class _Welford:
    def __init__(self) -> None:
        self.n = 0
        self.mean = 0.0
        self.m2 = 0.0

    def add(self, x: float) -> None:
        self.n += 1
        d = x - self.mean
        self.mean += d / self.n
        self.m2 += d * (x - self.mean)

    @property
    def sd(self) -> float:
        return math.sqrt(self.m2 / (self.n - 1)) if self.n > 1 else 0.0

    def to_list(self) -> list[float]:
        return [self.n, self.mean, self.m2]

    @classmethod
    def from_list(cls, v: list[float]) -> _Welford:
        w = cls()
        w.n, w.mean, w.m2 = int(v[0]), float(v[1]), float(v[2])
        return w


def _mean_sd(values: list[float]) -> tuple[float, float]:
    a = np.asarray(values, dtype=np.float64)
    if a.size == 0:
        return 0.0, 0.0
    return float(a.mean()), float(a.std(ddof=1)) if a.size > 1 else 0.0


class ConceptMemory:
    """Context, full, fix and program trees behind one API (spec 8.5 wrapper)."""

    def __init__(
        self,
        seed: int = 0,
        ctx_dim: int = 64,
        text_dim: int = EMB_DIM,
        n_min: int = N_MIN,
        embedder: Embedder | None = None,
    ):
        self.seed = seed
        self.ctx_dim = ctx_dim
        self.text_dim = text_dim
        self.n_min = n_min
        self.embedder = embedder or FakeEmbedder(dim=text_dim)
        text_prior = _BASE_PRIOR_VAR / text_dim  # unit-norm vectors: per-dim variance ~1/dim (M-002)
        self.trees = {
            "context": _Tree("context", ctx_dim, _BASE_PRIOR_VAR, seed),
            "full": _Tree("full", ctx_dim + len(MEASUREMENTS), _BASE_PRIOR_VAR, seed),
            "fix": _Tree("fix", text_dim, text_prior, seed),
            "program": _Tree("program", text_dim + len(CONTROL_CLASSES), text_prior, seed),
        }
        self.stats = {m: _Welford() for m in MEASUREMENTS}
        self._pending: dict[str, dict[str, Any]] = {}
        self._fixes: dict[str, dict[str, Any]] = {}
        self._programs: dict[str, dict[str, Any]] = {}

    # ---------------------------------------------------------------- learning
    @property
    def pending(self) -> dict[str, ControlResult]:
        """Unconfirmed failures held back from the trees, by result key."""
        return {k: ControlResult.model_validate(v["result"]) for k, v in self._pending.items()}

    def learn(
        self,
        results: list[ControlResult],
        confirmed_failures: set[str],
        contexts: dict[str, np.ndarray],
        iteration: int = 0,
    ) -> None:
        """Add passes now and failures once confirmed (spec 8.5 "Learning")."""
        for r in results:
            self._pending.pop(r.key, None)
            if r.key not in contexts:
                raise KeyError(f"no context vector for {r.key}")
            ctx = np.asarray(contexts[r.key], dtype=np.float32)
            if r.verdict == "pass" or r.key in confirmed_failures:
                self._add_result(r, ctx, iteration)
            else:
                self._pending[r.key] = {"result": r.model_dump(mode="json"), "ctx": ctx.tolist(), "iter": iteration}
        for key in sorted(k for k in self._pending if k in confirmed_failures):
            entry = self._pending.pop(key)
            self._add_result(ControlResult.model_validate(entry["result"]), np.asarray(entry["ctx"]), entry["iter"])

    def _add_result(self, r: ControlResult, ctx: np.ndarray, iteration: int) -> None:
        meas = _measurements(r)
        for m, v in meas.items():
            self.stats[m].add(v)
        payload = {"id": f"{iteration}:{r.key}", "verdict": r.verdict, "measurements": meas}
        self.trees["context"].insert(ctx, r.verdict, payload)
        self.trees["full"].insert(self._full_vec(ctx, meas), r.verdict, {"id": payload["id"]})

    def _global_z(self, meas: dict[str, float]) -> np.ndarray:
        out = []
        for m in MEASUREMENTS:
            st = self.stats[m]
            out.append((meas[m] - st.mean) / max(st.sd, SD_FLOOR[m]))
        return np.asarray(out, dtype=np.float32)

    def _full_vec(self, ctx: np.ndarray, meas: dict[str, float]) -> np.ndarray:
        return np.concatenate([np.asarray(ctx, dtype=np.float32), self._global_z(meas)])

    # ---------------------------------------------------------------- expectations
    def _expectation_node(self, ctx_vec: np.ndarray) -> tuple[CPPCobwebNode, CPPCobwebNode]:
        t = self.trees["context"]
        leaf = t.leaf(ctx_vec)
        node = leaf
        while node.parent is not None and len(t.payloads_under(node)) < self.n_min:
            node = node.parent
        return leaf, node

    def expect(self, ctx_vec: np.ndarray) -> Expectation:
        """Mean and sd per measurement from the deepest ancestor holding >= n_min results."""
        t = self.trees["context"]
        if t.empty:
            return Expectation({m: 0.0 for m in MEASUREMENTS}, {m: 0.0 for m in MEASUREMENTS}, 0, 0, 0)
        leaf, node = self._expectation_node(ctx_vec)
        pls = t.payloads_under(node)
        mean, sd = {}, {}
        for m in MEASUREMENTS:
            mean[m], sd[m] = _mean_sd([p["measurements"][m] for p in pls])
        parent = leaf.parent if leaf.parent is not None else leaf
        return Expectation(mean, sd, len(t.payloads_under(parent)), t.index_of(node), len(pls))

    def concept_n(self, ctx_vec: np.ndarray) -> int:
        return self.expect(ctx_vec).concept_n

    def concept_id(self, ctx_vec: np.ndarray) -> int:
        return self.expect(ctx_vec).node_id

    def surprise(self, ctx_vec: np.ndarray, measured: dict[str, float]) -> tuple[dict[str, float], float]:
        """z per measurement against the expectation, and NLL under the full tree."""
        exp = self.expect(ctx_vec)
        if exp.n_results == 0:
            return {m: 0.0 for m in MEASUREMENTS}, 0.0
        z = {
            m: (float(measured[m]) - exp.mean[m]) / max(exp.sd[m], SD_FLOOR[m])
            for m in MEASUREMENTS
            if m in measured
        }
        nll = -self.trees["full"].log_prob(self._full_vec(np.asarray(ctx_vec), {**exp.mean, **measured}))
        return z, nll

    def uncertainty(self, ctx_vec: np.ndarray) -> float:
        """mean_a(concept sd_a / global sd_a) + 1/sqrt(concept_n + 1) (spec 8.4)."""
        exp = self.expect(ctx_vec)
        if exp.n_results == 0:
            return 1.0 + 1.0
        ratios = [exp.sd[m] / max(self.stats[m].sd, SD_FLOOR[m]) for m in MEASUREMENTS]
        return float(np.mean(ratios)) + 1.0 / math.sqrt(exp.concept_n + 1)

    # ---------------------------------------------------------------- fixes
    def _text_vec(self, text: str) -> np.ndarray:
        return np.asarray(self.embedder.text([text])[0], dtype=np.float32)

    def record_fix(self, fix_id: str, proposal: Proposal, decision: Decision, iteration: int) -> None:
        vec = self._text_vec(proposal.hypothesis + " " + proposal.rationale)
        self._fixes[fix_id] = {
            "proposal": proposal.model_dump(mode="json"),
            "decision": decision.model_dump(mode="json"),
            "iteration": iteration,
        }
        label = "accepted" if decision.accepted else "rejected"
        self.trees["fix"].insert(vec, label, {"id": fix_id, "vec": vec.tolist()})

    def similar_fixes(self, text: str, k: int = 5) -> list[FixRecord]:
        hits = self._nearest("fix", self._text_vec(text), k)
        out = []
        for fid, dist in hits:
            rec = self._fixes[fid]
            out.append(FixRecord(id=fid, iteration=rec["iteration"], proposal=rec["proposal"],
                                 decision=rec["decision"], distance=dist))
        return out

    # ---------------------------------------------------------------- programs
    def program_vec(self, program: Program, outcome: dict[str, float] | None = None) -> np.ndarray:
        text = self._text_vec(dsl.pretty(program.root))
        norm = float(np.linalg.norm(text))
        text = text / norm if norm > 0 else text
        sig = np.asarray([float((outcome or {}).get(c, 0.0)) for c in CONTROL_CLASSES], dtype=np.float32)
        return np.concatenate([text, 0.5 * sig]).astype(np.float32)

    def add_program(self, program: Program, outcome: dict[str, float]) -> None:
        vec = self.program_vec(program, outcome)
        self._programs[program.id] = {"program": program.model_dump(mode="json"), "outcome": dict(outcome)}
        self.trees["program"].insert(vec, None, {"id": program.id, "vec": vec.tolist()})

    def similar_programs(
        self, program: Program, outcome: dict[str, float] | None = None, k: int = 5
    ) -> list[tuple[str, float]]:
        """Nearest stored programs by Euclidean distance; outcome None means an all-zero signature."""
        return self._nearest("program", self.program_vec(program, outcome), k)

    def _nearest(self, tree: str, q: np.ndarray, k: int) -> list[tuple[str, float]]:
        """Distances to payload vectors under the categorized leaf and its parent (spec 8.5).

        When that subtree holds fewer than k payloads the search widens to further ancestors (M-005).
        """
        t = self.trees[tree]
        if t.empty or k <= 0:
            return []
        leaf = t.leaf(q)
        node = leaf.parent if leaf.parent is not None else leaf
        while node.parent is not None and len(t.payloads_under(node)) < k:
            node = node.parent
        scored = []
        for p in t.payloads_under(node):
            d = float(np.linalg.norm(np.asarray(p["vec"], dtype=np.float32) - q))
            scored.append((d, p["id"]))
        scored.sort()
        return [(pid, d) for d, pid in scored[:k]]

    # ---------------------------------------------------------------- persistence
    def save(self, path: str | Path) -> None:
        """Write ``<tree>.json`` (Cobweb ``dump_json``) per tree plus ``sidecar.json``."""
        directory = Path(path)
        directory.mkdir(parents=True, exist_ok=True)
        side = {
            "seed": self.seed,
            "ctx_dim": self.ctx_dim,
            "text_dim": self.text_dim,
            "n_min": self.n_min,
            "trees": {name: t.save(directory) for name, t in self.trees.items()},
            "stats": {m: w.to_list() for m, w in self.stats.items()},
            "pending": self._pending,
            "fixes": self._fixes,
            "programs": self._programs,
        }
        (directory / "sidecar.json").write_text(json.dumps(side, sort_keys=True))

    def load(self, path: str | Path) -> ConceptMemory:
        """Replace this memory's state with the one saved at ``path``; returns ``self``."""
        directory = Path(path)
        side = json.loads((directory / "sidecar.json").read_text())
        self.__init__(side["seed"], side["ctx_dim"], side["text_dim"], side["n_min"], self.embedder)  # type: ignore[misc]
        for name, t in self.trees.items():
            t.load(directory, side["trees"][name])
        self.stats = {m: _Welford.from_list(v) for m, v in side["stats"].items()}
        self._pending = side["pending"]
        self._fixes = side["fixes"]
        self._programs = side["programs"]
        return self

    def tree_json(self, name: str) -> str:
        """Canonical JSON of one tree's statistics (for determinism checks)."""
        nodes = []
        for n in _dfs(self.trees[name].tree):
            nodes.append([n.count, n.mean.tolist(), n.sum_sq.tolist(), n.label_counts.tolist(),
                          getattr(n, "sea_payloads", [])])
        return json.dumps(nodes, sort_keys=True)

    # ---------------------------------------------------------------- seeding
    @classmethod
    def build_seed(
        cls,
        results: list[ControlResult],
        contexts: dict[str, np.ndarray],
        val_results: list[ControlResult],
        val_contexts: dict[str, np.ndarray],
        seed: int = 0,
        embedder: Embedder | None = None,
        n_shuffles: int = 3,
        **kwargs: Any,
    ) -> ConceptMemory:
        """Build from ``n_shuffles`` orderings; keep the lowest mean validation NLL (spec 8.5)."""
        confirmed = {r.key for r in results if r.verdict != "pass"}
        rng = np.random.default_rng(seed)
        best: tuple[float, int, ConceptMemory] | None = None
        for i in range(n_shuffles):
            order = rng.permutation(len(results))
            mem = cls(seed=seed, embedder=embedder, **kwargs)
            mem.learn([results[j] for j in order], confirmed, contexts)
            nlls = [mem.surprise(val_contexts[r.key], _measurements(r))[1] for r in val_results]
            score = float(np.mean(nlls)) if nlls else 0.0
            if best is None or score < best[0]:
                best = (score, i, mem)
        assert best is not None
        return best[2]
