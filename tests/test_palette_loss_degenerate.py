"""
PaletteLoss degenerate-state regression — the "solid black pixel art" bug.

With ``palettes: 1`` the palette-normalization loss is mathematically
undefined: softmax over a single-palette axis is identically 1.0 for every
pixel, so sigma == 0 exactly and both loss terms divide by zero (0/0 ->
NaN). One optimizer step propagates the NaN through Adam into every
parameter, decode() emits all-NaN frames, and the uint8 cast writes solid
black. On the compiled MLX step the fused variance reduction could instead
accumulate rounding residue at larger N, hiding the defect behind a finite
garbage constant with a zero gradient.

The fix (torch ``PaletteLoss.forward``, mlx ``palette_loss``):
- n_palettes == 1 -> graph-connected exact-zero loss (nothing to
  decorrelate across one palette);
- n_palettes >= 2 -> sigma guard (torch ``clamp_min(1e-8)``, mlx
  ``sqrt(var + 1e-16)``) so a mid-run degenerate logit state can never
  NaN-poison a render. Both guards are bit-identical at healthy sigma
  (~0.1) — pinned here against the unguarded reference formula.
"""

import importlib.util

import numpy as np
import pytest
import torch

needs_mlx = pytest.mark.skipif(
    importlib.util.find_spec("mlx") is None,
    reason="mlx not installed (darwin-only backend)",
)

CPU = torch.device("cpu")


def make_pixel(n_palettes, side=32, scale=16, palette_size=8, seed=7):
    from pytti.image_models import PixelImage

    torch.manual_seed(seed)
    img = PixelImage(
        side, side, scale, palette_size, n_palettes, device=CPU
    ).to(CPU)
    img.encode_random()
    return img


# ---------------------------------------------------------------------------
# n_palettes == 1: exact zero, finite gradients — torch
# ---------------------------------------------------------------------------


def test_palette_loss_single_palette_zero_torch():
    img = make_pixel(n_palettes=1)
    loss, loss_raw = img.loss(img)
    assert float(loss_raw.detach()) == 0.0
    assert float(loss.detach()) == 0.0
    # graph-connected: backward must run and yield finite (zero) grads
    loss.backward()
    assert img.tensor.grad is not None
    assert torch.isfinite(img.tensor.grad).all()
    assert not img.tensor.grad.any()


# ---------------------------------------------------------------------------
# n_palettes == 1: exact zero, finite gradients — mlx, eager AND compiled
# ---------------------------------------------------------------------------


@needs_mlx
@pytest.mark.parametrize("side", [16, 32, 48])
def test_palette_loss_single_palette_zero_mlx(side):
    # the compiled step is what real renders execute; grid 16/32 were the
    # NaN cells and grid 48 the finite-garbage-constant cell pre-fix
    import mlx.core as mx

    from pytti.mlx_engine import palette_loss, pixel_params_from_state_dict

    img = make_pixel(n_palettes=1, side=side)
    tree = pixel_params_from_state_dict(img.state_dict())

    loss_m, raw_m = palette_loss(tree)
    assert float(raw_m) == 0.0
    assert float(loss_m) == 0.0

    compiled = mx.compile(lambda tr: palette_loss(tr)[1])
    assert float(compiled(tree)) == 0.0

    def scalar(t):
        tr = dict(tree)
        tr["tensor"] = t
        return palette_loss(tr)[0]

    g = mx.grad(mx.compile(scalar))(tree["tensor"])
    g_np = np.array(g)
    assert np.isfinite(g_np).all()
    assert not g_np.any()


# ---------------------------------------------------------------------------
# n_palettes >= 2 with degenerate logits (sigma == 0): eps guard holds
# ---------------------------------------------------------------------------


def test_palette_loss_degenerate_logits_no_nan_torch():
    img = make_pixel(n_palettes=2)
    with torch.no_grad():
        img.tensor.zero_()  # uniform softmax everywhere -> sigma == 0
    loss, loss_raw = img.loss(img)
    assert torch.isfinite(loss_raw).all()
    loss.backward()
    assert torch.isfinite(img.tensor.grad).all()


@needs_mlx
def test_palette_loss_degenerate_logits_no_nan_mlx():
    import mlx.core as mx

    from pytti.mlx_engine import palette_loss, pixel_params_from_state_dict

    img = make_pixel(n_palettes=2)
    with torch.no_grad():
        img.tensor.zero_()
    tree = pixel_params_from_state_dict(img.state_dict())
    loss_m, raw_m = palette_loss(tree)
    assert np.isfinite(float(raw_m))

    def scalar(t):
        tr = dict(tree)
        tr["tensor"] = t
        return palette_loss(tr)[0]

    g = np.array(mx.grad(scalar)(tree["tensor"]))
    assert np.isfinite(g).all()


# ---------------------------------------------------------------------------
# no-op proof at healthy sigma: bit-identical to the unguarded formula
# ---------------------------------------------------------------------------


def _reference_palette_loss_torch(img):
    """The pre-fix formula, verbatim, no guards."""
    tensor = (
        img.tensor.movedim(0, -1)
        .contiguous()
        .view(-1, img.n_palettes)
        .softmax(dim=-1)
    )
    N, _ = tensor.shape
    mu = tensor.mean(dim=0, keepdim=True)
    sigma = tensor.std(dim=0, keepdim=True)
    tensor = tensor.sub(mu)
    S = (tensor.transpose(0, 1) @ tensor).div(sigma * sigma.transpose(0, 1) * N)
    S.sub_(torch.diag(S.diagonal()))
    loss_raw = S.mean()
    loss_raw.add_(sigma.mul(N).pow(-1).mean())
    return loss_raw


@pytest.mark.parametrize("n_palettes", [2, 3, 9])
def test_palette_loss_healthy_bit_identical_torch(n_palettes):
    img = make_pixel(n_palettes=n_palettes, side=64, scale=2)
    with torch.no_grad():
        ref = _reference_palette_loss_torch(img)
        _, loss_raw = img.loss(img)
    assert float(ref) == float(loss_raw)  # exact, not approx


@needs_mlx
@pytest.mark.parametrize("n_palettes", [2, 3, 9])
def test_palette_loss_healthy_bit_identical_mlx(n_palettes):
    import mlx.core as mx

    from pytti.mlx_engine import palette_loss, pixel_params_from_state_dict

    img = make_pixel(n_palettes=n_palettes, side=64, scale=2)
    tree = pixel_params_from_state_dict(img.state_dict())

    t = mx.softmax(
        mx.transpose(tree["tensor"], (1, 2, 0)).reshape(-1, n_palettes), axis=-1
    )
    big_n = t.shape[0]
    mu = mx.mean(t, axis=0, keepdims=True)
    sigma = mx.std(t, axis=0, keepdims=True, ddof=1)  # unguarded reference
    centered = t - mu
    s = (centered.T @ centered) / (sigma * sigma.T * big_n)
    s = s - mx.diag(mx.diagonal(s))
    ref = mx.mean(s) + mx.mean(1.0 / (sigma * big_n))

    _, raw_m = palette_loss(tree)
    assert float(ref) == float(raw_m)  # exact, not approx


# ---------------------------------------------------------------------------
# torch/mlx parity at n=1 (both exact zero)
# ---------------------------------------------------------------------------


@needs_mlx
def test_palette_loss_single_palette_parity():
    from pytti.mlx_engine import palette_loss, pixel_params_from_state_dict

    img = make_pixel(n_palettes=1)
    loss_t, raw_t = img.loss(img)
    tree = pixel_params_from_state_dict(img.state_dict())
    loss_m, raw_m = palette_loss(tree)
    assert float(raw_t.detach()) == float(raw_m) == 0.0
    assert float(loss_t.detach()) == float(loss_m) == 0.0


# ---------------------------------------------------------------------------
# end-to-end: tiny PixelImage trains without collapsing to black
# ---------------------------------------------------------------------------


def test_pixel_image_32x32_ps16_single_palette_training_does_not_collapse():
    """
    The reproducer shape (32x32 logical, pixel_size 16, palette_size 8,
    palettes 1) trained with the REAL path: make_optimizer's Adam at the
    production lr, image_loss() modules (hdr + palette normalization)
    summed with a synthetic semantic loss, update() clamps after every
    step. Pre-fix this NaN'd every parameter by step 2 and decoded to
    solid black from the first save.
    """
    from torch.nn import functional as F

    from pytti.ImageGuide import make_optimizer

    img = make_pixel(n_palettes=1, side=32, scale=16, palette_size=8)
    optimizer = make_optimizer(img.parameters(), "adam", lr=0.02)

    torch.manual_seed(11)
    target = torch.rand(1, 3, 512, 512)

    for _ in range(10):
        optimizer.zero_grad()
        z = img.decode_training_tensor()
        loss = F.mse_loss(z, target)
        for aug in img.image_loss():
            aug_loss, _ = aug(img)
            loss = loss + aug_loss
        loss.backward()
        for p in img.parameters():
            assert torch.isfinite(p.grad).all(), "NaN/inf gradient mid-train"
        optimizer.step()
        img.update()
        for p in img.parameters():
            assert torch.isfinite(p).all(), "NaN/inf parameter mid-train"

    frame = np.asarray(img.decode_image(), dtype=np.float64) / 255.0
    assert np.isfinite(frame).all()
    # non-collapsed: not pinned at either bound, retains structure
    assert 0.05 < frame.mean() < 0.95
    assert frame.std() > 0.01
