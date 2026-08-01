"""
MLX image models — M2 slice S1 gates (docs/mlx-m2-seam-map.md, Stage B).

Everything here is parity against the real torch classes, on CPU fp32 —
deterministic, no downloads, no GPU (contention-immune). The whole module
needs mlx (darwin-only backend); linux CI skips it at collection.

Gates enforced:
- decode value parity <= 1e-6 fp32, 64x64 canvas at scale 1 AND scale 2,
  both image models, palette-target passthrough included;
- gradient parity via mx.grad vs torch autograd on the same scalar loss,
  cosine >= 0.9999 per param group;
- argsort palette-order index equality on random palettes;
- HdrLoss / PaletteLoss value + grad parity <= 1e-6;
- state_dict import/export round-trip exactness (strict load included).
"""

import importlib.util

import numpy as np
import pytest
import torch

needs_mlx = pytest.mark.skipif(
    importlib.util.find_spec("mlx") is None,
    reason="mlx not installed (darwin-only backend)",
)
pytestmark = needs_mlx

CPU = torch.device("cpu")
SIDE = 64  # canvas edge for every case; scale=2 halves the param grid


def make_pixel(scale=1, palette_size=6, n_palettes=3, seed=0, hdr_weight=0.5):
    from pytti.image_models import PixelImage

    img = PixelImage(
        SIDE // scale,
        SIDE // scale,
        scale,
        palette_size,
        n_palettes,
        hdr_weight=hdr_weight,
        norm_weight=0.1,
        device=CPU,
    ).to(CPU)  # submodule buffers (hdr comp/weight) default elsewhere
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        # straddle every clamp: value beyond [0, 1], palette across [0, 2]
        img.value.copy_(torch.rand(img.value.shape, generator=g) * 1.4 - 0.2)
        img.tensor.copy_(torch.randn(img.tensor.shape, generator=g))
        img.palette.copy_(torch.rand(img.palette.shape, generator=g) * 2.0)
        img.palette_target.copy_(torch.rand(img.palette.shape, generator=g))
    return img


def make_rgb(scale=1, seed=0):
    from pytti.image_models import RGBImage

    img = RGBImage(SIDE // scale, SIDE // scale, scale, device=CPU).to(CPU)
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        # straddle the [0, 1] clamp on both sides
        img.tensor.copy_(torch.rand(img.tensor.shape, generator=g) * 1.6 - 0.3)
    return img


def pixel_tree(img):
    from pytti.mlx_engine import pixel_params_from_state_dict

    return pixel_params_from_state_dict(img.state_dict())


def rgb_tree(img):
    from pytti.mlx_engine import rgb_params_from_state_dict

    return rgb_params_from_state_dict(img.state_dict())


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    a = a.ravel().astype(np.float64)
    b = b.ravel().astype(np.float64)
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    assert na > 0 and nb > 0, "degenerate (zero) gradient — fixture is broken"
    return float(a @ b / (na * nb))


# ---------------------------------------------------------------------------
# PixelImage decode
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("scale", [1, 2])
@pytest.mark.parametrize("use_target", [False, True])
def test_pixel_decode_parity(scale, use_target):
    from pytti.mlx_engine import pixel_decode

    img = make_pixel(scale=scale, seed=3 + scale)
    img.use_palette_target = use_target
    ref = img.decode_tensor().detach().numpy()
    out = np.array(
        pixel_decode(pixel_tree(img), scale=scale, use_palette_target=use_target)
    )
    assert out.shape == ref.shape == (1, 3, SIDE, SIDE)
    assert np.abs(out - ref).max() <= 1e-6


@pytest.mark.parametrize("scale", [1, 2])
def test_pixel_grad_parity(scale):
    import mlx.core as mx

    from pytti.mlx_engine import pixel_decode

    img = make_pixel(scale=scale, seed=17)
    w = torch.randn(
        (1, 3, SIDE, SIDE), generator=torch.Generator().manual_seed(23)
    )
    (img.decode_tensor() * w).sum().backward()

    tree = pixel_tree(img)
    w_mx = mx.array(w.numpy())

    def loss_fn(tree):
        return mx.sum(pixel_decode(tree, scale=scale, use_palette_target=False) * w_mx)

    grads = mx.grad(loss_fn)(tree)
    for name, param in (
        ("value", img.value),
        ("tensor", img.tensor),
        ("palette", img.palette),
    ):
        assert cosine(np.array(grads[name]), param.grad.numpy()) >= 0.9999, name


def test_pixel_grad_with_palette_target_ignores_palette():
    # locked palette: torch leaves palette.grad None; mlx must give zeros
    import mlx.core as mx

    from pytti.mlx_engine import pixel_decode

    img = make_pixel(seed=29)
    img.use_palette_target = True
    w = torch.randn((1, 3, SIDE, SIDE), generator=torch.Generator().manual_seed(31))
    (img.decode_tensor() * w).sum().backward()
    assert img.palette.grad is None

    tree = pixel_tree(img)
    w_mx = mx.array(w.numpy())
    grads = mx.grad(
        lambda tr: mx.sum(pixel_decode(tr, scale=1, use_palette_target=True) * w_mx)
    )(tree)
    assert not np.array(grads["palette"]).any()
    for name, param in (("value", img.value), ("tensor", img.tensor)):
        assert cosine(np.array(grads[name]), param.grad.numpy()) >= 0.9999, name


# ---------------------------------------------------------------------------
# palette sorting
# ---------------------------------------------------------------------------


def test_palette_sort_indices_match_torch():
    import mlx.core as mx

    from pytti.mlx_engine import palette_sort_indices

    for seed in range(5):
        g = torch.Generator().manual_seed(seed)
        palette = torch.rand((8, 4, 3), generator=g) * 2.0
        p = (palette / 2.0).clamp(0, 1)
        magic = p.new_tensor([[[0.299, 0.587, 0.114]]])
        ref = p.square().mul(magic).sum(dim=-1).argsort(dim=0)  # [P, n]
        got = np.array(palette_sort_indices(mx.array(palette.numpy())))
        assert np.array_equal(got, ref.numpy()), f"seed {seed}"


def test_sort_palette_values_match_torch():
    import mlx.core as mx  # noqa: F401  (asserts the mlx path is importable)

    from pytti.mlx_engine import sort_palette

    img = make_pixel(seed=9)
    ref = img.sort_palette().detach().numpy()
    got = np.array(sort_palette(pixel_tree(img), use_palette_target=False))
    assert np.array_equal(got, ref)  # div-by-2 / clamp / gather: bit-exact

    img.use_palette_target = True
    got_t = np.array(sort_palette(pixel_tree(img), use_palette_target=True))
    assert np.array_equal(got_t, img.sort_palette().detach().numpy())


# ---------------------------------------------------------------------------
# image losses
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("use_target", [False, True])
def test_hdr_loss_parity(use_target):
    import mlx.core as mx

    from pytti.mlx_engine import hdr_loss

    img = make_pixel(seed=5)
    img.use_palette_target = use_target
    loss_t, raw_t = img.hdr_loss(img)

    tree = pixel_tree(img)
    loss_m, raw_m = hdr_loss(tree, use_palette_target=use_target)
    assert abs(float(loss_m) - float(loss_t.detach())) <= 1e-6
    assert abs(float(raw_m) - float(raw_t.detach())) <= 1e-6

    grads = mx.grad(lambda tr: hdr_loss(tr, use_palette_target=use_target)[0])(tree)
    if use_target:
        # target buffer replaces the palette: the torch loss has no grad
        # path at all, the mlx palette gradient must be identically zero
        assert not loss_t.requires_grad
        assert not np.array(grads["palette"]).any()
    else:
        loss_t.backward()
        diff = np.abs(np.array(grads["palette"]) - img.palette.grad.numpy()).max()
        assert diff <= 1e-6


def test_palette_loss_parity():
    import mlx.core as mx

    from pytti.mlx_engine import palette_loss

    img = make_pixel(seed=7)
    loss_t, raw_t = img.loss(img)
    loss_t.backward()

    tree = pixel_tree(img)
    loss_m, raw_m = palette_loss(tree)
    assert abs(float(loss_m) - float(loss_t.detach())) <= 1e-6
    assert abs(float(raw_m) - float(raw_t.detach())) <= 1e-6

    grads = mx.grad(lambda tr: palette_loss(tr)[0])(tree)
    diff = np.abs(np.array(grads["tensor"]) - img.tensor.grad.numpy()).max()
    assert diff <= 1e-6


# ---------------------------------------------------------------------------
# per-step update clamps
# ---------------------------------------------------------------------------


def test_pixel_update_clamps_match_torch():
    from pytti.mlx_engine import pixel_update

    img = make_pixel(seed=21)
    g = torch.Generator().manual_seed(22)
    with torch.no_grad():
        # push every param outside its clamp range in both directions
        img.palette.copy_(torch.rand(img.palette.shape, generator=g) * 6.0 - 2.0)
        img.value.copy_(torch.rand(img.value.shape, generator=g) * 3.0 - 1.0)
        img.tensor.copy_(torch.randn(img.tensor.shape, generator=g))
    tree = pixel_tree(img)

    img.update()
    clamped = pixel_update(tree)
    for name, param in (
        ("palette", img.palette),
        ("value", img.value),
        ("tensor", img.tensor),
    ):
        assert np.array_equal(np.array(clamped[name]), param.detach().numpy()), name
    # constants pass through untouched
    for name in ("palette_target", "hdr_comp", "hdr_weight", "norm_weight"):
        assert np.array_equal(np.array(clamped[name]), np.array(tree[name])), name


def test_rgb_update_is_identity():
    from pytti.mlx_engine import rgb_update

    tree = rgb_tree(make_rgb(seed=25))
    out = rgb_update(tree)
    assert np.array_equal(np.array(out["tensor"]), np.array(tree["tensor"]))


# ---------------------------------------------------------------------------
# RGBImage decode + clamp_with_grad
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("scale", [1, 2])
def test_rgb_decode_parity(scale):
    from pytti.mlx_engine import rgb_decode

    img = make_rgb(scale=scale, seed=2)
    ref = img.decode_tensor().detach().numpy()
    out = np.array(rgb_decode(rgb_tree(img), scale=scale))
    assert out.shape == ref.shape == (1, 3, SIDE, SIDE)
    assert np.abs(out - ref).max() <= 1e-6


@pytest.mark.parametrize("scale", [1, 2])
def test_rgb_grad_parity(scale):
    # signed weights exercise the cotangent-dependent clamp gate both ways
    import mlx.core as mx

    from pytti.mlx_engine import rgb_decode

    img = make_rgb(scale=scale, seed=4)
    w = torch.randn((1, 3, SIDE, SIDE), generator=torch.Generator().manual_seed(31))
    (img.decode_tensor() * w).sum().backward()

    tree = rgb_tree(img)
    w_mx = mx.array(w.numpy())
    grads = mx.grad(lambda tr: mx.sum(rgb_decode(tr, scale=scale) * w_mx))(tree)
    assert cosine(np.array(grads["tensor"]), img.tensor.grad.numpy()) >= 0.9999


def test_clamp_with_grad_semantics_exact():
    import mlx.core as mx

    from pytti.mlx_engine import clamp_with_grad
    from pytti.tensor_tools import clamp_with_grad as torch_cwg

    g = torch.Generator().manual_seed(7)
    x = torch.rand((256,), generator=g) * 3.0 - 1.0  # below / inside / above
    w = torch.randn((256,), generator=g)  # signed cotangents

    xt = x.clone().requires_grad_(True)
    (torch_cwg(xt, 0, 1) * w).sum().backward()

    x_mx = mx.array(x.numpy())
    w_mx = mx.array(w.numpy())
    val = np.array(clamp_with_grad(x_mx, 0.0, 1.0))
    grad = np.array(mx.grad(lambda v: mx.sum(clamp_with_grad(v, 0.0, 1.0) * w_mx))(x_mx))

    assert np.array_equal(val, x.clamp(0, 1).numpy())
    assert np.array_equal(grad, xt.grad.numpy())  # identical gate, identical values
    # the fixture must actually exercise the gate in both directions
    out_of_range = ((x < 0) | (x > 1)).numpy()
    assert (grad[out_of_range] == 0).any() and (grad[out_of_range] != 0).any()


# ---------------------------------------------------------------------------
# state_dict import/export
# ---------------------------------------------------------------------------


def test_pixel_state_dict_round_trip():
    from pytti.image_models import PixelImage
    from pytti.mlx_engine import (
        pixel_params_from_state_dict,
        pixel_state_dict_from_params,
    )

    img = make_pixel(seed=13)
    sd = img.state_dict()
    tree = pixel_params_from_state_dict(sd)
    sd2 = pixel_state_dict_from_params(tree)
    assert set(sd2) == set(sd)
    for key in sd:
        assert torch.equal(sd2[key], sd[key]), key

    fresh = PixelImage(SIDE, SIDE, 1, 6, 3, hdr_weight=0.5, norm_weight=0.1, device=CPU)
    fresh.load_state_dict(sd2)  # strict — key or shape drift fails loud

    tree2 = pixel_params_from_state_dict(sd2)
    assert set(tree2) == set(tree)
    for key in tree:
        assert np.array_equal(np.array(tree2[key]), np.array(tree[key])), key


def test_pixel_round_trip_without_hdr():
    from pytti.mlx_engine import (
        hdr_loss,
        pixel_params_from_state_dict,
        pixel_state_dict_from_params,
    )

    img = make_pixel(seed=14, hdr_weight=0)
    sd = img.state_dict()
    assert "hdr_loss.comp" not in sd
    tree = pixel_params_from_state_dict(sd)
    assert "hdr_comp" not in tree and "hdr_weight" not in tree
    sd2 = pixel_state_dict_from_params(tree)
    assert set(sd2) == set(sd)
    for key in sd:
        assert torch.equal(sd2[key], sd[key]), key
    with pytest.raises(ValueError, match="hdr"):
        hdr_loss(tree, use_palette_target=False)


def test_rgb_state_dict_round_trip():
    from pytti.image_models import RGBImage
    from pytti.mlx_engine import (
        rgb_params_from_state_dict,
        rgb_state_dict_from_params,
    )

    img = make_rgb(seed=16)
    sd = img.state_dict()
    sd2 = rgb_state_dict_from_params(rgb_params_from_state_dict(sd))
    assert set(sd2) == {"tensor"}
    assert torch.equal(sd2["tensor"], sd["tensor"])
    fresh = RGBImage(SIDE, SIDE, 1, device=CPU)
    fresh.load_state_dict(sd2)


def test_import_rejects_off_contract_state_dicts():
    from pytti.mlx_engine import (
        pixel_params_from_state_dict,
        rgb_params_from_state_dict,
    )

    sd = make_pixel(seed=15).state_dict()

    missing = dict(sd)
    del missing["value"]
    with pytest.raises(ValueError, match="value"):
        pixel_params_from_state_dict(missing)

    extra = dict(sd)
    extra["bogus"] = torch.zeros(1)
    with pytest.raises(ValueError, match="bogus"):
        pixel_params_from_state_dict(extra)

    half_hdr = dict(sd)
    del half_hdr["hdr_loss.weight"]
    with pytest.raises(ValueError, match="hdr"):
        pixel_params_from_state_dict(half_hdr)

    with pytest.raises(ValueError, match="extra"):
        rgb_params_from_state_dict(
            {"tensor": torch.zeros(1, 3, 4, 4), "extra": torch.zeros(2)}
        )
    with pytest.raises(ValueError, match="fp32"):
        rgb_params_from_state_dict(
            {"tensor": torch.zeros(1, 3, 4, 4, dtype=torch.float64)}
        )


# ---------------------------------------------------------------------------
# mx.compile compatibility (S5 assembles these into the whole-step compile)
# ---------------------------------------------------------------------------


def test_decode_functions_compile():
    import mlx.core as mx

    from pytti.mlx_engine import pixel_decode, rgb_decode

    img = make_pixel(scale=2, seed=19)
    tree = pixel_tree(img)
    eager = np.array(pixel_decode(tree, scale=2, use_palette_target=False))
    compiled = np.array(
        mx.compile(lambda tr: pixel_decode(tr, scale=2, use_palette_target=False))(tree)
    )
    assert np.abs(eager - compiled).max() <= 1e-6

    rtree = rgb_tree(make_rgb(seed=20))
    w = mx.array(
        np.random.default_rng(0).standard_normal((1, 3, SIDE, SIDE)).astype(np.float32)
    )

    def rgb_loss(tr):
        return mx.sum(rgb_decode(tr, scale=1) * w)

    g_eager = np.array(mx.grad(rgb_loss)(rtree)["tensor"])
    g_compiled = np.array(mx.compile(mx.grad(rgb_loss))(rtree)["tensor"])
    # the custom-VJP clamp must survive compile with the gate intact
    assert np.abs(g_eager - g_compiled).max() <= 1e-6
    assert (g_eager == 0).any() and (g_eager != 0).any()


def test_nearest_upsample_rejects_bad_input():
    import mlx.core as mx

    from pytti.mlx_engine import nearest_upsample

    with pytest.raises(ValueError, match="scale"):
        nearest_upsample(mx.zeros((1, 3, 4, 4)), 0)
    with pytest.raises(ValueError, match="NCHW"):
        nearest_upsample(mx.zeros((3, 4, 4)), 2)
