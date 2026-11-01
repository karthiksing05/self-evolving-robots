"""Embeddings for the concept memory (spec 8.5, Appendix D).

``FakeEmbedder`` is deterministic and model-free; fast tests and ``embedder: fake`` runs use it.
The real embedders (all-MiniLM-L6-v2 text, DINOv2-small CLS image) load lazily on CPU and are
exercised only by ``slow`` tests.  ``ContextEncoder`` turns a control into the 64-d context
vector: ``[image_emb(crop) ⊕ text_emb(description)]`` whitened by PCA+ICA once fitted.
"""

from __future__ import annotations

import hashlib
import math
import re
import warnings
from pathlib import Path
from typing import Protocol

import numpy as np

from sea.contracts import CONTROL_CLASSES, GRASP_CLASS, ControlSpec, PanelSpec, recess_bin, tilt_bin

EMB_DIM = 384
CTX_DIM = 64
SURROGATE_PX = 32


class Embedder(Protocol):
    def text(self, texts: list[str]) -> np.ndarray: ...
    def image(self, images: list[np.ndarray]) -> np.ndarray: ...


def _normalize(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    return np.where(n > 0, v / np.where(n > 0, n, 1.0), 0.0).astype(np.float32)


_TOKEN_RE = re.compile(r"[^\s(),\[\]{}]+")


class FakeEmbedder:
    """Deterministic stand-in: token-hash bag of words for text, a fixed random projection for images."""

    def __init__(self, dim: int = EMB_DIM, seed: int = 0):
        self.dim = dim
        self._token_cache: dict[str, np.ndarray] = {}
        rng = np.random.default_rng(seed)
        self._img_proj = rng.standard_normal((16 * 16 * 3, dim)).astype(np.float32)

    def _token_vec(self, token: str) -> np.ndarray:
        vec = self._token_cache.get(token)
        if vec is None:
            digest = hashlib.sha256(token.encode()).digest()
            rng = np.random.default_rng(int.from_bytes(digest[:8], "little"))
            vec = rng.standard_normal(self.dim).astype(np.float32)
            self._token_cache[token] = vec
        return vec

    def text(self, texts: list[str]) -> np.ndarray:
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for i, t in enumerate(texts):
            for tok in _TOKEN_RE.findall(t.lower()):
                out[i] += self._token_vec(tok)
        return _normalize(out)

    def image(self, images: list[np.ndarray]) -> np.ndarray:
        out = np.zeros((len(images), self.dim), dtype=np.float32)
        for i, img in enumerate(images):
            out[i] = downsample16(img).reshape(-1) @ self._img_proj
        return _normalize(out)


def downsample16(img: np.ndarray) -> np.ndarray:
    """Nearest-neighbour 16x16x3 float image in [0, 1]."""
    a = np.asarray(img, dtype=np.float32)
    if a.ndim == 2:
        a = np.stack([a, a, a], axis=-1)
    a = a[..., :3]
    if a.size and a.max() > 1.0:
        a = a / 255.0
    rows = np.linspace(0, a.shape[0] - 1, 16).round().astype(int)
    cols = np.linspace(0, a.shape[1] - 1, 16).round().astype(int)
    return a[np.ix_(rows, cols)]


class CompositeEmbedder:
    """Text from one embedder, images from another."""

    def __init__(self, text_embedder: Embedder, image_embedder: Embedder):
        self._text = text_embedder
        self._image = image_embedder

    def text(self, texts: list[str]) -> np.ndarray:
        return self._text.text(texts)

    def image(self, images: list[np.ndarray]) -> np.ndarray:
        return self._image.image(images)


class SentenceTransformerEmbedder:
    """all-MiniLM-L6-v2 on CPU, L2-normalized 384-d."""

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.model_name = model_name
        self._model = None

    def text(self, texts: list[str]) -> np.ndarray:  # pragma: no cover - loads a real model
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self.model_name, device="cpu")
        emb = self._model.encode(texts, normalize_embeddings=True, convert_to_numpy=True)
        return np.asarray(emb, dtype=np.float32)

    def image(self, images: list[np.ndarray]) -> np.ndarray:
        raise NotImplementedError("SentenceTransformerEmbedder embeds text only")


class DinoV2Embedder:
    """DINOv2-small CLS token at 224x224 on CPU, L2-normalized 384-d."""

    def __init__(self, model_name: str = "facebook/dinov2-small"):
        self.model_name = model_name
        self._model = None
        self._processor = None

    def text(self, texts: list[str]) -> np.ndarray:
        raise NotImplementedError("DinoV2Embedder embeds images only")

    def image(self, images: list[np.ndarray]) -> np.ndarray:  # pragma: no cover - loads a real model
        import torch
        from transformers import AutoImageProcessor, AutoModel

        if self._model is None:
            self._processor = AutoImageProcessor.from_pretrained(self.model_name)
            self._model = AutoModel.from_pretrained(self.model_name).to("cpu").eval()
        pix = []
        for img in images:
            a = np.asarray(img, dtype=np.float32)
            if a.ndim == 2:
                a = np.stack([a, a, a], axis=-1)
            if a.max() <= 1.0:
                a = a * 255.0
            pix.append(a[..., :3].astype(np.uint8))
        inputs = self._processor(images=pix, return_tensors="pt", size={"height": 224, "width": 224},
                                 do_center_crop=False)
        with torch.no_grad():
            cls = self._model(**inputs).last_hidden_state[:, 0].numpy()
        return _normalize(cls)


def real_embedder() -> CompositeEmbedder:  # pragma: no cover - constructs real-model wrappers
    return CompositeEmbedder(SentenceTransformerEmbedder(), DinoV2Embedder())


# ---------------------------------------------------------------- control descriptions
def context_text(control: ControlSpec, panel: PanelSpec) -> str:
    if control.cls == GRASP_CLASS:  # an object on a table: what it is, its grasp class, size and weight
        w, h = control.size_mm[0], control.size_mm[1]
        return (f"grasp {control.asset or control.shape} {control.shape} width={5 * round(w / 5):.0f} "
                f"height={10 * round(h / 10):.0f} mass={50 * round(control.mass_g / 50):.0f}")
    return (
        f"{control.cls} {control.handle} recess={recess_bin(control.recess_depth_mm)} "
        f"tilt={tilt_bin(panel.tilt_rad)}"
    )


_PALETTE = {
    cls: np.array(
        [0.5 + 0.45 * math.cos(2 * math.pi * k / 9 + p) for p in (0.0, 2.1, 4.2)], dtype=np.float32
    )
    for k, cls in enumerate(CONTROL_CLASSES)
}
_PALETTE[GRASP_CLASS] = np.array([0.85, 0.45, 0.2], dtype=np.float32)
_ROUND = {"push_button", "recessed_button", "knob"}


def geometry_surrogate(control: ControlSpec, panel: PanelSpec) -> np.ndarray:
    """Procedural 32x32x3 'crop' for fast mode: class colour and shape, size, recess, handle, tilt.

    Deterministic; controls that look alike in the real render (same class, similar size, same
    recess and tilt) give nearly identical images.
    """
    n = SURROGATE_PX
    yy, xx = np.mgrid[0:n, 0:n].astype(np.float32)
    cx = cy = (n - 1) / 2.0
    tilt = min(abs(panel.tilt_rad), 1.2) / 1.2
    img = np.empty((n, n, 3), dtype=np.float32)
    img[...] = (0.45 + 0.3 * tilt * (yy / (n - 1) - 0.5))[..., None]
    scale = (n - 4) / 120.0  # px per mm; a 120 mm door fills the crop
    sizes = list(control.size_mm) or [10.0]
    w = max(sizes[0] * scale, 1.5)
    h = max((sizes[1] if len(sizes) > 1 else sizes[0] * (1.0 if control.cls in _ROUND else 0.35)) * scale, 1.5)
    if control.cls in _ROUND or control.shape == "sphere":
        mask = ((xx - cx) ** 2 + (yy - cy) ** 2) <= (w / 2) ** 2
    else:
        mask = (np.abs(xx - cx) <= w / 2) & (np.abs(yy - cy) <= h / 2)
    if control.recess_depth_mm > 0:
        ring = min(1.0 + control.recess_depth_mm / 5.0, 6.0)
        if control.cls in _ROUND:
            outer = ((xx - cx) ** 2 + (yy - cy) ** 2) <= (w / 2 + ring) ** 2
        else:
            outer = (np.abs(xx - cx) <= w / 2 + ring) & (np.abs(yy - cy) <= h / 2 + ring)
        img[outer & ~mask] = 0.08
    img[mask] = _PALETTE[control.cls]
    if control.handle != "none":
        hx = int(min(cx + w / 2 - 1, n - 2))
        level = {"tab": 1.0, "pull": 0.75, "push_latch": 0.25}[control.handle]
        span = {"tab": 1, "pull": 4, "push_latch": 0}[control.handle]
        img[max(int(cy) - span, 0): int(cy) + span + 1, max(hx - 1, 0): hx + 1] = level
    return img


# ---------------------------------------------------------------- context encoder
class ContextEncoder:
    """Control -> 64-d context vector (spec 8.5, Context tree row)."""

    def __init__(self, embedder: Embedder | None = None, dim: int = CTX_DIM, seed: int = 0):
        self.embedder = embedder or FakeEmbedder()
        self.dim = dim
        self.seed = seed
        rng = np.random.default_rng(seed)
        # Inputs are two unit vectors (norm sqrt 2); the scale keeps each output dimension ~unit variance.
        self._proj = (rng.standard_normal((2 * EMB_DIM, dim)) / math.sqrt(2.0)).astype(np.float32)
        self._white: dict[str, np.ndarray] | None = None

    @property
    def fitted(self) -> bool:
        return self._white is not None

    def raw(self, control: ControlSpec, panel: PanelSpec, image: np.ndarray | None = None) -> np.ndarray:
        return self.raw_many([(control, panel)], None if image is None else [image])[0]

    def raw_many(
        self, items: list[tuple[ControlSpec, PanelSpec]], images: list[np.ndarray | None] | None = None
    ) -> np.ndarray:
        if not items:
            return np.zeros((0, 2 * EMB_DIM), dtype=np.float32)
        imgs = [
            (images[i] if images is not None and images[i] is not None else geometry_surrogate(c, p))
            for i, (c, p) in enumerate(items)
        ]
        img_emb = self.embedder.image(imgs)  # type: ignore[arg-type]
        txt_emb = self.embedder.text([context_text(c, p) for c, p in items])
        return np.concatenate([img_emb, txt_emb], axis=1).astype(np.float32)

    def fit(self, raw_matrix: np.ndarray) -> None:
        """Fit PCA+ICA whitening (pca_dim=64) on at least 64 raw rows.

        If the data has numerical rank r < 64 (identical descriptions under FakeEmbedder), whitening
        uses r components and the remaining output dimensions are zero (docs/decisions.md M-003).
        """
        from cobweb_language_embedding.preprocess_embedding import PCAICAWhiteningModel

        x = np.asarray(raw_matrix, dtype=np.float64)
        if x.ndim != 2 or x.shape[0] < self.dim:
            raise ValueError(f"whitening needs at least {self.dim} rows, got {x.shape[0] if x.ndim == 2 else 0}")
        s = np.linalg.svd(x - x.mean(axis=0), compute_uv=False)
        rank = int((s > 1e-6 * max(s[0], 1e-12)).sum())
        pca_dim = max(1, min(self.dim, rank))
        # sklearn's PCA picks randomized SVD for wide inputs and draws it from numpy's global RNG
        # (the library passes no random_state); seed it so a fit depends on the data alone.
        np_state = np.random.get_state()
        np.random.seed(self.seed)
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                model = PCAICAWhiteningModel.fit(x, pca_dim=pca_dim)
        finally:
            np.random.set_state(np_state)
        self._white = {
            "mean": np.asarray(model.mean, dtype=np.float64),
            "pca_components": np.asarray(model.pca_components, dtype=np.float64),
            "pca_explained_var": np.asarray(model.pca_explained_var, dtype=np.float64),
            "ica_unmixing": np.asarray(model.ica_unmixing, dtype=np.float64),
            "eps": np.asarray(model.eps, dtype=np.float64),
        }

    def encode_raw(self, raw: np.ndarray) -> np.ndarray:
        x = np.asarray(raw, dtype=np.float64)
        single = x.ndim == 1
        x = np.atleast_2d(x)
        if self._white is None:
            out = x @ self._proj.astype(np.float64)
        else:
            w = self._white
            z = (x - w["mean"]) @ w["pca_components"].T / np.sqrt(w["pca_explained_var"] + w["eps"])
            z = z @ w["ica_unmixing"].T
            out = np.zeros((x.shape[0], self.dim))
            out[:, : z.shape[1]] = z
        out = out.astype(np.float32)
        return out[0] if single else out

    def encode(self, control: ControlSpec, panel: PanelSpec, image: np.ndarray | None = None) -> np.ndarray:
        return self.encode_raw(self.raw(control, panel, image))

    def encode_many(
        self, items: list[tuple[ControlSpec, PanelSpec]], images: list[np.ndarray | None] | None = None
    ) -> np.ndarray:
        return self.encode_raw(self.raw_many(items, images)).reshape(len(items), self.dim)

    def save(self, path: str | Path) -> None:
        arrays: dict[str, np.ndarray] = {"proj": self._proj, "meta": np.array([self.dim, self.seed])}
        if self._white is not None:
            arrays.update({f"white_{k}": v for k, v in self._white.items()})
        with open(path, "wb") as fh:
            np.savez(fh, **arrays)

    @classmethod
    def load(cls, path: str | Path, embedder: Embedder | None = None) -> ContextEncoder:
        with np.load(path) as data:
            dim, seed = (int(v) for v in data["meta"])
            enc = cls(embedder, dim=dim, seed=seed)
            enc._proj = data["proj"].astype(np.float32)
            if "white_mean" in data:
                enc._white = {
                    k: data[f"white_{k}"]
                    for k in ("mean", "pca_components", "pca_explained_var", "ica_unmixing", "eps")
                }
        return enc
