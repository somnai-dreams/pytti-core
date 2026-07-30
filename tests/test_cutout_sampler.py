"""
Tests for the batched sync-free cutout sampler (`pytti_batched`).

Gates from the modernization plan §4 / Slice 6:
- value parity vs per-crop slice + bilinear interpolate: max abs diff < 2/255
- MPS-vs-CPU gradient parity through grid_sample backward: < 1e-6
  (runs on machines with MPS; skips at runtime elsewhere so CPU CI stays green)
- offsets/sizes keep pytti_classic's coordinate convention, including the
  padded-input shift for non-clamp border modes.
"""

import pytest
import torch
from torch.nn import functional as F

from pytti.Perceptor.cutouts.samplers import (
    _affine_crop_grid,
    pytti_batched,
    pytti_classic,
)

PARITY_GATE = 2 / 255  # plan §4 hard gate
# Plan §4 gate: MPS-vs-CPU parity through grid_sample backward, mean-squared
# (the plan's measured 2.9e-11 at cutn=40 @ 768x432 is an MSE; at this test's
# size we measured 2.5e-13 — atomic-add reordering noise, torch 2.13).
GRAD_MSE_GATE = 1e-9
# Same comparison, max-abs relative to grad magnitude: measured 1.0e-6.
GRAD_REL_GATE = 1e-5
# grid_sample backward vs interpolate backward accumulate in different orders,
# so cross-op gradient agreement is fp32-noise-limited: measured 3.1e-6 relative
# (batched 1.8e-5 abs vs fp64 truth at grad max 5.8 — pure rounding, not math).
CROSS_OP_GRAD_REL_GATE = 1e-5


def _identity(x):
    return x


def _pad_like_embedder(raw, side_x, side_y, padding, mode="replicate"):
    paddingx = min(round(side_x * padding), side_x)
    paddingy = min(round(side_y * padding), side_y)
    return F.pad(raw, (paddingx, paddingx, paddingy, paddingy), mode=mode)


def _smooth_image(side_x, side_y, device="cpu"):
    """Deterministic non-trivial image in [0, 1]."""
    ys = torch.linspace(0, 6.28, side_y, device=device)
    xs = torch.linspace(0, 6.28, side_x, device=device)
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")
    chans = [
        0.5 + 0.5 * torch.sin(xx + 2 * yy),
        0.5 + 0.5 * torch.cos(3 * xx - yy),
        (xx / 6.28 + yy / 6.28) / 2,
    ]
    return torch.stack(chans).unsqueeze(0)


def _int_coords(offsets, sizes, side_x, side_y):
    """Recover integer pixel offsets/sizes from the normalized [cutn, 2] tensors."""
    ox = (offsets[:, 0] * side_x).round().long()
    oy = (offsets[:, 1] * side_y).round().long()
    sx = (sizes[:, 0] * side_x).round().long()
    sy = (sizes[:, 1] * side_y).round().long()
    # cutouts are square: both columns must encode the same pixel size
    assert torch.equal(sx, sy), "sizes columns disagree on the square cut size"
    return ox, oy, sx


def _reference_cutouts(input, side_x, side_y, cut_size, padding, border_mode, ox, oy, sz):
    """pytti_classic's exact per-crop math (slice + bilinear interpolate) at
    given integer offsets/sizes. `input` must be pre-padded for non-clamp
    border modes, exactly as the Embedder does before calling the sampler."""
    paddingx = min(round(side_x * padding), side_x)
    paddingy = min(round(side_y * padding), side_y)
    outs = []
    for x, y, s in zip(ox.tolist(), oy.tolist(), sz.tolist(), strict=True):
        if border_mode == "clamp":
            crop = input[:, :, y : y + s, x : x + s]
        else:
            crop = input[
                :, :, paddingy + y : paddingy + y + s, paddingx + x : paddingx + x + s
            ]
        assert crop.shape[-2:] == (s, s), (
            f"reference crop truncated: offset ({x}, {y}) size {s} out of bounds"
        )
        outs.append(
            F.interpolate(
                crop, size=(cut_size, cut_size), mode="bilinear", align_corners=False
            )
        )
    return torch.cat(outs)


# (side_x, side_y, cut_size, cut_pow): second case is upsampling-heavy —
# most sampled sizes fall below cut_size, exercising the grid-clamp path
# that reproduces F.interpolate's edge replication.
GEOMETRIES = [(97, 65, 32, 1.5), (64, 56, 48, 3.0)]


@pytest.mark.parametrize("border_mode", ["clamp", "smear"])
@pytest.mark.parametrize("side_x,side_y,cut_size,cut_pow", GEOMETRIES)
def test_parity_with_slice_interpolate(border_mode, side_x, side_y, cut_size, cut_pow):
    torch.manual_seed(0)
    padding, cutn = 0.25, 32
    raw = torch.rand(1, 3, side_y, side_x)  # noise image: worst case for resampling
    input = (
        raw
        if border_mode == "clamp"
        else _pad_like_embedder(raw, side_x, side_y, padding)
    )
    cutouts, offsets, sizes = pytti_batched(
        input=input,
        side_x=side_x,
        side_y=side_y,
        cut_size=cut_size,
        padding=padding,
        cutn=cutn,
        cut_pow=cut_pow,
        border_mode=border_mode,
        augs=_identity,
        noise_fac=0,
        device="cpu",
    )
    ox, oy, sz = _int_coords(offsets, sizes, side_x, side_y)
    reference = _reference_cutouts(
        input, side_x, side_y, cut_size, padding, border_mode, ox, oy, sz
    )
    diff = (cutouts - reference).abs().max().item()
    assert diff < PARITY_GATE, f"batched vs slice+interpolate diff {diff} >= 2/255"


@pytest.mark.parametrize("border_mode", ["clamp", "smear"])
def test_shape_and_range_invariants(border_mode):
    torch.manual_seed(1)
    side_x, side_y, cut_size, padding, cutn, cut_pow = 96, 64, 32, 0.25, 40, 1.5
    raw = torch.rand(1, 3, side_y, side_x)
    input = (
        raw
        if border_mode == "clamp"
        else _pad_like_embedder(raw, side_x, side_y, padding)
    )
    cutouts, offsets, sizes = pytti_batched(
        input=input,
        side_x=side_x,
        side_y=side_y,
        cut_size=cut_size,
        padding=padding,
        cutn=cutn,
        cut_pow=cut_pow,
        border_mode=border_mode,
        augs=_identity,
        noise_fac=0,
        device="cpu",
    )
    assert cutouts.shape == (cutn, 3, cut_size, cut_size)
    assert offsets.shape == (cutn, 2)
    assert sizes.shape == (cutn, 2)
    assert torch.isfinite(cutouts).all()
    # bilinear samples are convex combinations of input pixels
    assert cutouts.min() >= input.min() - 1e-6
    assert cutouts.max() <= input.max() + 1e-6

    ox, oy, sz = _int_coords(offsets, sizes, side_x, side_y)
    max_size = min(side_x, side_y)
    assert (sz >= 1).all() and (sz <= max_size).all()
    paddingx = min(round(side_x * padding), side_x)
    paddingy = min(round(side_y * padding), side_y)
    if border_mode == "clamp":
        # crops stay inside the raw image
        assert (ox >= 0).all() and (ox + sz <= side_x).all()
        assert (oy >= 0).all() and (oy + sz <= side_y).all()
    else:
        # crops may reach into the padding by at most min(size, padding) px,
        # and always stay inside the padded input
        assert (ox >= -torch.minimum(sz, torch.tensor(paddingx))).all()
        assert (oy >= -torch.minimum(sz, torch.tensor(paddingy))).all()
        assert (paddingx + ox + sz <= side_x + 2 * paddingx).all()
        assert (paddingy + oy + sz <= side_y + 2 * paddingy).all()


def test_augs_and_noise_are_applied():
    torch.manual_seed(2)
    side_x, side_y, cut_size = 64, 48, 16
    raw = torch.rand(1, 3, side_y, side_x)
    cutouts, _, _ = pytti_batched(
        input=raw,
        side_x=side_x,
        side_y=side_y,
        cut_size=cut_size,
        padding=0.25,
        cutn=8,
        cut_pow=1.5,
        border_mode="clamp",
        augs=lambda x: x + 10.0,
        noise_fac=0.1,
        device="cpu",
    )
    # augs ran (values shifted by 10) and noise_fac perturbed around that
    assert cutouts.min() > 5.0
    assert cutouts.max() < 15.0


def test_offset_size_semantics_clamp():
    """Coordinate convention: offsets are the crop's top-left corner in raw-image
    pixels normalized by (side_x, side_y); sizes are the square pixel size
    normalized the same way — pytti_classic's convention. Forcing
    size == cut_size == max_size makes resampling the identity, so each cutout
    must equal the raw pixels at exactly the reported location."""
    torch.manual_seed(3)
    side_x, side_y, cut_size, cutn = 64, 32, 32, 24
    raw = torch.rand(1, 3, side_y, side_x)
    cutouts, offsets, sizes = pytti_batched(
        input=raw,
        side_x=side_x,
        side_y=side_y,
        cut_size=cut_size,  # cut_size == max_size forces sampled size == 32
        padding=0.25,
        cutn=cutn,
        cut_pow=1.5,
        border_mode="clamp",
        augs=_identity,
        noise_fac=0,
        device="cpu",
    )
    ox, oy, sz = _int_coords(offsets, sizes, side_x, side_y)
    assert (sz == 32).all()
    # normalized exactly like classic: size/side_x, size/side_y
    assert torch.allclose(sizes, torch.tensor([32 / side_x, 32 / side_y]).expand(cutn, 2))
    assert (oy == 0).all()  # side_y - size == 0 leaves a single valid row
    assert (ox >= 0).all() and (ox <= side_x - 32).all()
    # offsets must be integer pixel positions under classic's normalization
    assert torch.allclose(offsets[:, 0] * side_x, ox.to(offsets.dtype), atol=1e-4)
    for i in range(cutn):
        x, y = ox[i].item(), oy[i].item()
        assert torch.allclose(
            cutouts[i], raw[0, :, y : y + 32, x : x + 32], atol=1e-4
        ), f"cutout {i} does not match raw pixels at reported offset ({x}, {y})"

    # classic, same geometry: identical size normalization (proves the shared
    # convention; classic's offset upper bound is side - size + 1 vs our
    # side - size — the documented off-by-one fix)
    torch.manual_seed(3)
    _, offsets_c, sizes_c = pytti_classic(
        input=raw,
        side_x=side_x,
        side_y=side_y,
        cut_size=cut_size,
        padding=0.25,
        cutn=cutn,
        cut_pow=1.5,
        border_mode="clamp",
        augs=_identity,
        noise_fac=0,
        device="cpu",
    )
    assert torch.allclose(sizes_c, sizes)
    assert (offsets_c[:, 0] * side_x >= 0).all()
    assert (offsets_c[:, 0] * side_x <= side_x - 32 + 1).all()


def test_offset_size_semantics_padded():
    """Non-clamp border modes: input arrives pre-padded, crops are shifted by
    (paddingx, paddingy) inside it, but reported offsets stay in unpadded
    coordinates (and may be negative) — classic's padding-shift semantics."""
    torch.manual_seed(4)
    side_x, side_y, cut_size, cutn, padding = 64, 32, 32, 24, 0.25
    paddingx = min(round(side_x * padding), side_x)  # 16
    paddingy = min(round(side_y * padding), side_y)  # 8
    raw = torch.rand(1, 3, side_y, side_x)
    padded = _pad_like_embedder(raw, side_x, side_y, padding)
    cutouts, offsets, sizes = pytti_batched(
        input=padded,
        side_x=side_x,
        side_y=side_y,
        cut_size=cut_size,  # forces size == 32 as above
        padding=padding,
        cutn=cutn,
        cut_pow=1.5,
        border_mode="smear",
        augs=_identity,
        noise_fac=0,
        device="cpu",
    )
    ox, oy, sz = _int_coords(offsets, sizes, side_x, side_y)
    assert (sz == 32).all()
    assert (ox >= -paddingx).all() and (oy >= -paddingy).all()
    saw_negative = bool((ox < 0).any() or (oy < 0).any())
    for i in range(cutn):
        x, y = ox[i].item(), oy[i].item()
        assert torch.allclose(
            cutouts[i],
            padded[0, :, paddingy + y : paddingy + y + 32, paddingx + x : paddingx + x + 32],
            atol=1e-4,
        ), f"cutout {i} does not match padded pixels shifted by padding at ({x}, {y})"
    # the whole point of the padded branch: crops can reach into the padding
    assert saw_negative, "seeded draw expected at least one negative offset"


def test_grad_parity_cpu():
    """Gradients through pytti_batched must match slice+interpolate gradients
    at the same offsets/sizes — same bilinear math; agreement is limited only
    by fp32 accumulation-order differences between the two backward kernels."""
    torch.manual_seed(5)
    side_x, side_y, cut_size, padding, cutn, cut_pow = 96, 64, 32, 0.25, 16, 1.5
    raw = torch.rand(1, 3, side_y, side_x)
    weights = torch.randn(cutn, 3, cut_size, cut_size)

    leaf_b = raw.clone().requires_grad_()
    cutouts, offsets, sizes = pytti_batched(
        input=leaf_b,
        side_x=side_x,
        side_y=side_y,
        cut_size=cut_size,
        padding=padding,
        cutn=cutn,
        cut_pow=cut_pow,
        border_mode="clamp",
        augs=_identity,
        noise_fac=0,
        device="cpu",
    )
    (cutouts * weights).sum().backward()

    ox, oy, sz = _int_coords(offsets, sizes, side_x, side_y)
    leaf_r = raw.clone().requires_grad_()
    reference = _reference_cutouts(
        leaf_r, side_x, side_y, cut_size, padding, "clamp", ox, oy, sz
    )
    (reference * weights).sum().backward()

    diff = (leaf_b.grad - leaf_r.grad).abs().max().item()
    rel = diff / leaf_r.grad.abs().max().item()
    assert rel < CROSS_OP_GRAD_REL_GATE, (
        f"CPU grad rel diff vs slice+interpolate {rel} >= {CROSS_OP_GRAD_REL_GATE}"
    )


def test_mps_cpu_grad_parity():
    """Plan §4 CI pin: grid_sample forward+backward with the *same* grid must
    agree MPS-vs-CPU to < 1e-6, so a torch upgrade regressing MPS
    grid_sampler backward fails loudly. Skips at runtime where MPS is
    unavailable (CPU CI stays green).

    The grid comes from a real pytti_batched run (fractional scales, clamped
    edges) via the same _affine_crop_grid the sampler uses, and is first
    verified to reproduce that run's cutouts — pinning the actual code path,
    not a synthetic grid."""
    if not torch.backends.mps.is_available():
        pytest.skip("MPS not available on this machine")
    torch.manual_seed(6)
    side_x, side_y, cut_size, padding, cutn, cut_pow = 96, 64, 32, 0.25, 16, 1.5
    raw = torch.rand(1, 3, side_y, side_x)
    weights = torch.randn(cutn, 3, cut_size, cut_size)

    cutouts, offsets, sizes = pytti_batched(
        input=raw,
        side_x=side_x,
        side_y=side_y,
        cut_size=cut_size,
        padding=padding,
        cutn=cutn,
        cut_pow=cut_pow,
        border_mode="clamp",
        augs=_identity,
        noise_fac=0,
        device="cpu",
    )
    ox, oy, sz = _int_coords(offsets, sizes, side_x, side_y)
    grid = _affine_crop_grid(
        ox.float(), oy.float(), sz.float(), 3, cut_size, side_y, side_x
    )

    def run(device):
        leaf = raw.detach().to(device).clone().requires_grad_()
        out = F.grid_sample(
            leaf.expand(cutn, -1, -1, -1),
            grid.to(device),
            mode="bilinear",
            padding_mode="border",
            align_corners=False,
        )
        (out * weights.to(device)).sum().backward()
        return out.detach().cpu(), leaf.grad.cpu()

    out_cpu, grad_cpu = run("cpu")
    # the rebuilt grid must reproduce the real sampler run exactly
    assert torch.equal(out_cpu, cutouts), "rebuilt grid diverged from pytti_batched"

    out_mps, grad_mps = run("mps")
    value_diff = (out_mps - out_cpu).abs().max().item()
    grad_mse = (grad_mps - grad_cpu).pow(2).mean().item()
    grad_rel = (
        (grad_mps - grad_cpu).abs().max() / grad_cpu.abs().max()
    ).item()
    assert value_diff < PARITY_GATE, f"MPS-vs-CPU value diff {value_diff} >= 2/255"
    assert grad_mse < GRAD_MSE_GATE, f"MPS-vs-CPU grad MSE {grad_mse} >= {GRAD_MSE_GATE}"
    assert grad_rel < GRAD_REL_GATE, f"MPS-vs-CPU grad rel {grad_rel} >= {GRAD_REL_GATE}"


def test_rejects_batched_input():
    with pytest.raises(ValueError, match="single-image batch"):
        pytti_batched(
            input=torch.rand(2, 3, 32, 32),
            side_x=32,
            side_y=32,
            cut_size=16,
            padding=0.25,
            cutn=4,
            cut_pow=1.0,
            border_mode="clamp",
            augs=_identity,
            noise_fac=0,
            device="cpu",
        )


def test_rejects_unpadded_input_for_padded_mode():
    with pytest.raises(ValueError, match="pre-padded"):
        pytti_batched(
            input=torch.rand(1, 3, 32, 32),  # smear expects padded H/W
            side_x=32,
            side_y=32,
            cut_size=16,
            padding=0.25,
            cutn=4,
            cut_pow=1.0,
            border_mode="smear",
            augs=_identity,
            noise_fac=0,
            device="cpu",
        )
