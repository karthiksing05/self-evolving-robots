"""Embedders and the context encoder (spec 8.5, 12.1 FakeEmbedder rule)."""

from __future__ import annotations

import numpy as np
import pytest

from sea.contracts import ControlSpec, PanelSpec, Pose, Rules, Vec3
from sea.memory.embed import (
    CompositeEmbedder,
    ContextEncoder,
    DinoV2Embedder,
    FakeEmbedder,
    SentenceTransformerEmbedder,
    context_text,
    downsample16,
    geometry_surrogate,
)

RULES = Rules(force_cap_by_class={"push_button": 10.0})


def control(cls="push_button", size=(12.0,), handle="none", recess=0.0, cid="c0") -> ControlSpec:
    return ControlSpec(
        id=cid, cls=cls, pose_panel=Pose(pos=Vec3(x=0, y=0, z=0)), size_mm=size, stiffness=500.0,
        travel=0.003, friction=0.1, handle=handle, recess_depth_mm=recess, success="moved_at_least($target, travel)",
    )


def panel(tilt=0.0) -> PanelSpec:
    return PanelSpec(id="p0", seed=0, split="train", pose_world=Pose(pos=Vec3(x=0.2, y=0, z=0)),
                     tilt_rad=tilt, controls=[], rules=RULES)


def cos(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def test_fake_text_is_deterministic_and_normalized():
    a = FakeEmbedder().text(["push_button none recess=0 tilt=0-20", "knob none"])
    b = FakeEmbedder().text(["push_button none recess=0 tilt=0-20", "knob none"])
    assert a.shape == (2, 384) and a.dtype == np.float32
    assert np.array_equal(a, b)
    assert np.allclose(np.linalg.norm(a, axis=1), 1.0, atol=1e-5)


def test_fake_text_shared_tokens_are_closer():
    e = FakeEmbedder()
    base, near, far = e.text(["door tab recess=0 tilt=0-20", "door pull recess=0 tilt=0-20", "slot none"])
    assert cos(base, near) > cos(base, far) + 0.3


def test_fake_text_empty_is_zero():
    assert not FakeEmbedder().text([""]).any()


def test_fake_image_deterministic_and_similarity():
    e = FakeEmbedder()
    rng = np.random.default_rng(0)
    img = rng.random((40, 40, 3))
    noisy = np.clip(img + rng.normal(0, 0.01, img.shape), 0, 1)
    other = rng.random((40, 40, 3))
    a, b, c = e.image([img, noisy, other])
    assert np.array_equal(a, FakeEmbedder().image([img])[0])
    assert cos(a, b) > 0.99 > cos(a, c)


def test_downsample_handles_gray_and_uint8():
    gray = (np.arange(64 * 48).reshape(64, 48) % 256).astype(np.uint8)
    out = downsample16(gray)
    assert out.shape == (16, 16, 3) and out.max() <= 1.0
    rgba = np.ones((20, 20, 4), dtype=np.float32)
    assert downsample16(rgba).shape == (16, 16, 3)


def test_context_text_uses_bins():
    c = control("door", size=(80.0, 90.0), handle="push_latch", recess=20.0)
    assert context_text(c, panel(tilt=1.0)) == "door push_latch recess=>10 tilt=>45"
    assert context_text(control(), panel()) == "push_button none recess=0 tilt=0-20"


@pytest.mark.parametrize(
    "cls,size,handle,recess",
    [
        ("push_button", (12.0,), "none", 0.0),
        ("recessed_button", (8.0,), "none", 15.0),
        ("knob", (30.0,), "none", 0.0),
        ("door", (100.0, 80.0), "tab", 0.0),
        ("door", (100.0, 80.0), "pull", 5.0),
        ("drawer", (90.0, 40.0), "push_latch", 0.0),
        ("slot", (30.0, 2.0), "none", 0.0),
        ("lever", (50.0,), "none", 0.0),
    ],
)
def test_geometry_surrogate_variants(cls, size, handle, recess):
    img = geometry_surrogate(control(cls, size, handle, recess), panel(0.5))
    assert img.shape == (32, 32, 3) and np.isfinite(img).all()
    assert np.array_equal(img, geometry_surrogate(control(cls, size, handle, recess), panel(0.5)))


def test_geometry_surrogate_empty_size_uses_default():
    img = geometry_surrogate(control("slider", size=()), panel())
    assert img.shape == (32, 32, 3)


def test_surrogate_similar_controls_are_close():
    e = FakeEmbedder()
    a = e.image([geometry_surrogate(control(size=(12.0,)), panel())])[0]
    b = e.image([geometry_surrogate(control(size=(13.0,)), panel())])[0]
    c = e.image([geometry_surrogate(control("door", size=(100.0, 90.0), handle="pull"), panel())])[0]
    assert cos(a, b) > cos(a, c)


def test_composite_embedder_delegates():
    t, i = FakeEmbedder(seed=1), FakeEmbedder(seed=2)
    comp = CompositeEmbedder(t, i)
    img = np.zeros((8, 8, 3))
    img[2:5, 2:5] = 1.0
    assert np.array_equal(comp.text(["x y"]), t.text(["x y"]))
    assert np.array_equal(comp.image([img]), i.image([img]))


def test_real_embedders_reject_wrong_modality():
    with pytest.raises(NotImplementedError):
        SentenceTransformerEmbedder().image([np.zeros((4, 4, 3))])
    with pytest.raises(NotImplementedError):
        DinoV2Embedder().text(["x"])


def _dataset(n=96):
    rng = np.random.default_rng(3)
    classes = ["push_button", "knob", "door", "lever", "slider", "toggle"]
    items = []
    for k in range(n):
        cls = classes[k % len(classes)]
        size = (float(rng.uniform(8, 40)),) if cls in ("push_button", "knob", "lever", "slider") else (
            float(rng.uniform(60, 120)), float(rng.uniform(10, 120)))
        handle = "tab" if cls == "door" else "none"
        items.append((control(cls, size, handle, float(rng.choice([0.0, 6.0, 15.0])), cid=f"c{k}"),
                      panel(float(rng.uniform(0, 1.0)))))
    return items


def test_encoder_projection_before_fit():
    enc = ContextEncoder()
    c, p = control(), panel()
    raw = enc.raw(c, p)
    assert raw.shape == (768,)
    v = enc.encode(c, p)
    assert v.shape == (64,) and v.dtype == np.float32 and not enc.fitted
    assert np.array_equal(v, ContextEncoder().encode(c, p))
    assert enc.raw_many([]).shape == (0, 768)


def test_encoder_uses_given_image():
    enc = ContextEncoder()
    c, p = control(), panel()
    img = np.ones((32, 32, 3), dtype=np.float32)
    assert not np.array_equal(enc.encode(c, p, image=img), enc.encode(c, p))


def test_encoder_fit_requires_64_rows():
    enc = ContextEncoder()
    with pytest.raises(ValueError):
        enc.fit(np.zeros((10, 768)))


def test_encoder_fit_whitens(tmp_path):
    enc = ContextEncoder(seed=5)
    items = _dataset()
    raw = enc.raw_many(items)
    enc.fit(raw)
    assert enc.fitted
    z = enc.encode_many(items)
    assert z.shape == (len(items), 64) and np.isfinite(z).all()
    active = z.std(axis=0) > 1e-6
    assert active.sum() >= 8
    assert np.allclose(z[:, active].std(axis=0), 1.0, atol=0.2)
    path = tmp_path / "enc.npz"
    enc.save(path)
    back = ContextEncoder.load(path)
    assert back.fitted
    assert np.allclose(back.encode_many(items), z, atol=1e-5)


def test_encoder_fit_ignores_global_rng():
    # wide and low-rank, so sklearn's PCA takes its randomized-SVD path
    rng = np.random.default_rng(0)
    raw = rng.normal(size=(80, 40)) @ rng.normal(size=(40, 768))
    fits = []
    for g in (1, 2):
        np.random.seed(g)
        enc = ContextEncoder(seed=3)
        enc.fit(raw)
        fits.append(enc.encode_raw(raw[:5]))
        assert np.random.get_state()[1][0] == np.random.RandomState(g).get_state()[1][0]  # restored
    assert np.array_equal(fits[0], fits[1])


def test_encoder_save_load_unfitted(tmp_path):
    enc = ContextEncoder(seed=9)
    path = tmp_path / "enc.npz"
    enc.save(path)
    back = ContextEncoder.load(path)
    assert not back.fitted
    assert np.array_equal(back.encode(control(), panel()), enc.encode(control(), panel()))


@pytest.mark.slow
def test_real_embedders_shapes():  # pragma: no cover - nightly
    st = SentenceTransformerEmbedder().text(["push_button none recess=0 tilt=0-20"])
    assert st.shape == (1, 384)
    img = (geometry_surrogate(control(), panel()) * 255).astype(np.uint8)
    dv = DinoV2Embedder().image([img])
    assert dv.shape == (1, 384) and np.allclose(np.linalg.norm(dv, axis=1), 1.0, atol=1e-4)
