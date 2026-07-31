"""
Tests for the designed two-population cutout sampler (`pytti_smart`).

Contract under test (lead's spec):
- identical signature / input / return conventions to `pytti_batched`
  (pre-padded input for non-clamp border modes, offsets/sizes [cutn, 2]
  normalized by (side_x, side_y), augs + noise applied last);
- population split: n_global = max(2, round(cutn * 0.25)) full-frame anchors
  at exactly the inscribed square, stratified along the free axis;
- n_detail = cutn - n_global stratified detail cuts with size fraction in
  [0.2, 0.55] of max_size (biased small), clamped to >= cut_size, one crop
  origin jittered inside each cell of a near-square grid, sizes randomly
  permuted across cells;
- fail-loud on malformed input; gradients flow; works at cutn as low as 4.
"""

import math

import pytest
import torch
from hydra import compose, initialize
from omegaconf import OmegaConf
from torch.nn import functional as F

from pytti.Perceptor.cutouts.samplers import _stratified_cells, pytti_smart

# geometry used unless a test needs something specific: landscape, max_size 64
SIDE_X, SIDE_Y, CUT_SIZE = 96, 64, 8
PADDING = 0.25
# fp32 flooring can pull a stratified origin at most one pixel below its
# cell's continuous lower bound
FLOOR_SLACK = 1 + 1e-3


def _identity(x):
    return x


def _n_global(cutn):
    return max(2, round(cutn * 0.25))


def _pad_like_embedder(raw, side_x, side_y, padding, mode="replicate"):
    paddingx = min(round(side_x * padding), side_x)
    paddingy = min(round(side_y * padding), side_y)
    return F.pad(raw, (paddingx, paddingx, paddingy, paddingy), mode=mode)


def _int_coords(offsets, sizes, side_x, side_y):
    """Recover integer pixel offsets/sizes from the normalized [cutn, 2] tensors."""
    ox = (offsets[:, 0] * side_x).round().long()
    oy = (offsets[:, 1] * side_y).round().long()
    sx = (sizes[:, 0] * side_x).round().long()
    sy = (sizes[:, 1] * side_y).round().long()
    assert torch.equal(sx, sy), "sizes columns disagree on the square cut size"
    return ox, oy, sx


def _run(
    input,
    side_x=SIDE_X,
    side_y=SIDE_Y,
    cut_size=CUT_SIZE,
    cutn=16,
    border_mode="clamp",
    padding=PADDING,
    augs=_identity,
    noise_fac=0,
):
    return pytti_smart(
        input=input,
        side_x=side_x,
        side_y=side_y,
        cut_size=cut_size,
        padding=padding,
        cutn=cutn,
        cut_pow=1.5,  # accepted but unused
        border_mode=border_mode,
        augs=augs,
        noise_fac=noise_fac,
        device="cpu",
    )


def _detail_grid_rows(n_detail, side_x, side_y):
    """The near-square grid's row count — mirrors pytti_smart's choice."""
    a = math.ceil(math.sqrt(n_detail))
    b = math.ceil(n_detail / a)
    return a if side_y > side_x else b


@pytest.mark.parametrize("border_mode", ["clamp", "smear"])
@pytest.mark.parametrize("cutn", [4, 16, 40])
def test_population_counts_and_size_bounds(border_mode, cutn):
    torch.manual_seed(0)
    raw = torch.rand(1, 3, SIDE_Y, SIDE_X)
    input = (
        raw
        if border_mode == "clamp"
        else _pad_like_embedder(raw, SIDE_X, SIDE_Y, PADDING)
    )
    cutouts, offsets, sizes = _run(input, cutn=cutn, border_mode=border_mode)
    assert cutouts.shape == (cutn, 3, CUT_SIZE, CUT_SIZE)
    assert offsets.shape == (cutn, 2)
    assert sizes.shape == (cutn, 2)
    assert torch.isfinite(cutouts).all()

    _, _, sz = _int_coords(offsets, sizes, SIDE_X, SIDE_Y)
    n_global = _n_global(cutn)
    max_size = min(SIDE_X, SIDE_Y)  # 64
    # global anchors: exactly the inscribed square, first n_global rows
    assert (sz[:n_global] == max_size).all()
    # detail: floor(f * max_size) with f in [0.2, 0.55); CUT_SIZE=8 never binds
    d_sz = sz[n_global:]
    assert d_sz.shape[0] == cutn - n_global
    assert (d_sz >= math.floor(0.2 * max_size)).all()  # >= 12
    assert (d_sz <= math.floor(0.55 * max_size)).all()  # <= 35
    assert (d_sz < max_size).all()


def test_detail_sizes_clamped_up_to_cut_size():
    # cut_size 48 > 0.55 * 64: every detail size must be raised to exactly 48
    torch.manual_seed(1)
    raw = torch.rand(1, 3, SIDE_Y, SIDE_X)
    _, offsets, sizes = _run(raw, cut_size=48, cutn=16)
    _, _, sz = _int_coords(offsets, sizes, SIDE_X, SIDE_Y)
    assert (sz[_n_global(16) :] == 48).all()


@pytest.mark.parametrize(
    "side_x,side_y,free_axis", [(128, 64, "x"), (64, 128, "y")]
)
def test_global_anchors_stratified_along_free_axis(side_x, side_y, free_axis):
    torch.manual_seed(2)
    cutn = 16
    n_global = _n_global(cutn)  # 4
    raw = torch.rand(1, 3, side_y, side_x)
    _, offsets, sizes = _run(raw, side_x=side_x, side_y=side_y, cutn=cutn)
    ox, oy, sz = _int_coords(offsets, sizes, side_x, side_y)
    max_size = min(side_x, side_y)
    free = max(side_x, side_y) - max_size  # 64
    strat, fixed = (ox, oy) if free_axis == "x" else (oy, ox)
    assert (fixed[:n_global] == 0).all()
    lane_w = free / n_global
    for i in range(n_global):
        v = strat[i].item()
        assert i * lane_w - FLOOR_SLACK < v < (i + 1) * lane_w + 1e-3, (
            f"global anchor {i} at {v} outside its lane [{i * lane_w}, {(i + 1) * lane_w})"
        )


def test_global_anchors_square_canvas_all_zero_and_identical():
    torch.manual_seed(3)
    side = 64
    raw = torch.rand(1, 3, side, side)
    cutouts, offsets, sizes = _run(raw, side_x=side, side_y=side, cutn=16)
    n_global = _n_global(16)
    assert (offsets[:n_global] == 0).all()
    ox, oy, sz = _int_coords(offsets, sizes, side, side)
    assert (sz[:n_global] == side).all()
    # same crop, identity augs, no noise: the anchors are exact copies
    for i in range(1, n_global):
        assert torch.equal(cutouts[0], cutouts[i])


@pytest.mark.parametrize("border_mode", ["clamp", "smear"])
def test_detail_stratification_covers_every_cell(border_mode):
    """Detail row i is cell i of the near-square grid (sizes are permuted,
    cells are not): with a fixed seed, every crop origin must land inside its
    own cell, mapped onto that cutout's valid offset domain."""
    torch.manual_seed(4)
    cutn = 16
    n_global = _n_global(cutn)
    n_detail = cutn - n_global  # 12 -> 4x3 grid on this landscape canvas
    raw = torch.rand(1, 3, SIDE_Y, SIDE_X)
    input = (
        raw
        if border_mode == "clamp"
        else _pad_like_embedder(raw, SIDE_X, SIDE_Y, PADDING)
    )
    _, offsets, sizes = _run(input, cutn=cutn, border_mode=border_mode)
    ox, oy, sz = _int_coords(offsets, sizes, SIDE_X, SIDE_Y)
    d_ox, d_oy, d_sz = ox[n_global:], oy[n_global:], sz[n_global:]

    n_rows = _detail_grid_rows(n_detail, SIDE_X, SIDE_Y)
    cx_lo, cx_w, cy_lo, cy_h = _stratified_cells(
        n_detail, n_rows, "cpu", torch.float32
    )
    paddingx = min(round(SIDE_X * PADDING), SIDE_X)
    paddingy = min(round(SIDE_Y * PADDING), SIDE_Y)
    for i in range(n_detail):
        s = d_sz[i].item()
        if border_mode == "clamp":
            lo_x, ext_x = 0.0, SIDE_X - s
            lo_y, ext_y = 0.0, SIDE_Y - s
        else:
            px, py = min(s, paddingx), min(s, paddingy)
            lo_x, ext_x = -px, SIDE_X - s + 2 * px
            lo_y, ext_y = -py, SIDE_Y - s + 2 * py
        x_lo = lo_x + cx_lo[i].item() * ext_x
        x_hi = lo_x + (cx_lo[i] + cx_w[i]).item() * ext_x
        y_lo = lo_y + cy_lo[i].item() * ext_y
        y_hi = lo_y + (cy_lo[i] + cy_h[i]).item() * ext_y
        assert x_lo - FLOOR_SLACK < d_ox[i] < x_hi + 1e-3, (
            f"detail {i} x-origin {d_ox[i]} outside cell [{x_lo}, {x_hi})"
        )
        assert y_lo - FLOOR_SLACK < d_oy[i] < y_hi + 1e-3, (
            f"detail {i} y-origin {d_oy[i]} outside cell [{y_lo}, {y_hi})"
        )


@pytest.mark.parametrize("n_cells,n_rows", [(1, 1), (2, 1), (5, 2), (12, 3), (13, 4), (30, 5)])
def test_stratified_cells_tile_the_unit_square(n_cells, n_rows):
    x_lo, x_w, y_lo, y_h = _stratified_cells(n_cells, n_rows, "cpu", torch.float32)
    assert x_lo.shape == x_w.shape == y_lo.shape == y_h.shape == (n_cells,)
    assert (x_lo >= 0).all() and (x_lo + x_w <= 1 + 1e-6).all()
    assert (y_lo >= 0).all() and (y_lo + y_h <= 1 + 1e-6).all()
    # exact tiling: total cell area is 1
    assert torch.isclose((x_w * y_h).sum(), torch.tensor(1.0), atol=1e-5)


def test_offset_size_semantics_clamp():
    """Same coordinate convention as pytti_batched: offsets are the crop's
    top-left corner in raw-image pixels normalized by (side_x, side_y).
    cut_size == max_size forces every size (anchor and clamped-up detail) to
    max_size, making resampling the identity, so each cutout must equal the
    raw pixels at exactly the reported location."""
    torch.manual_seed(5)
    side_x, side_y, cut_size, cutn = 64, 32, 32, 24
    raw = torch.rand(1, 3, side_y, side_x)
    cutouts, offsets, sizes = _run(
        raw, side_x=side_x, side_y=side_y, cut_size=cut_size, cutn=cutn
    )
    ox, oy, sz = _int_coords(offsets, sizes, side_x, side_y)
    assert (sz == 32).all()
    assert torch.allclose(
        sizes, torch.tensor([32 / side_x, 32 / side_y]).expand(cutn, 2)
    )
    assert (oy == 0).all()  # side_y - size == 0 leaves a single valid row
    assert (ox >= 0).all() and (ox <= side_x - 32).all()
    assert torch.allclose(offsets[:, 0] * side_x, ox.to(offsets.dtype), atol=1e-4)
    for i in range(cutn):
        x, y = ox[i].item(), oy[i].item()
        assert torch.allclose(
            cutouts[i], raw[0, :, y : y + 32, x : x + 32], atol=1e-4
        ), f"cutout {i} does not match raw pixels at reported offset ({x}, {y})"


def test_offset_size_semantics_padded():
    """Non-clamp border modes: input arrives pre-padded, crops are shifted by
    (paddingx, paddingy) inside it, reported offsets stay in unpadded
    coordinates (negative when reaching into padding) — pytti_batched's
    convention. Global anchors never leave the unpadded frame."""
    torch.manual_seed(6)
    side_x, side_y, cut_size, cutn = 64, 32, 32, 24
    paddingx = min(round(side_x * PADDING), side_x)  # 16
    paddingy = min(round(side_y * PADDING), side_y)  # 8
    raw = torch.rand(1, 3, side_y, side_x)
    padded = _pad_like_embedder(raw, side_x, side_y, PADDING)
    cutouts, offsets, sizes = _run(
        padded,
        side_x=side_x,
        side_y=side_y,
        cut_size=cut_size,
        cutn=cutn,
        border_mode="smear",
    )
    ox, oy, sz = _int_coords(offsets, sizes, side_x, side_y)
    assert (sz == 32).all()
    n_global = _n_global(cutn)
    # anchors stay inside the unpadded frame on every border mode
    assert (ox[:n_global] >= 0).all() and (ox[:n_global] + 32 <= side_x).all()
    assert (oy[:n_global] == 0).all()
    # detail may reach into the padding by at most min(size, padding) px
    assert (ox >= -paddingx).all() and (oy >= -paddingy).all()
    for i in range(cutn):
        x, y = ox[i].item(), oy[i].item()
        assert torch.allclose(
            cutouts[i],
            padded[
                0, :, paddingy + y : paddingy + y + 32, paddingx + x : paddingx + x + 32
            ],
            atol=1e-4,
        ), f"cutout {i} does not match padded pixels shifted by padding at ({x}, {y})"
    # the point of the padded branch: some detail crop reached into padding
    assert bool((ox < 0).any() or (oy < 0).any()), (
        "seeded draw expected at least one negative detail offset"
    )


def test_augs_and_noise_are_applied():
    torch.manual_seed(7)
    raw = torch.rand(1, 3, SIDE_Y, SIDE_X)
    cutouts, _, _ = _run(
        raw, cut_size=16, cutn=8, augs=lambda x: x + 10.0, noise_fac=0.1
    )
    assert cutouts.min() > 5.0
    assert cutouts.max() < 15.0


def test_gradient_flows_to_input():
    torch.manual_seed(8)
    leaf = torch.rand(1, 3, SIDE_Y, SIDE_X, requires_grad=True)
    cutouts, _, _ = _run(leaf, cut_size=32, cutn=8)
    cutouts.sum().backward()
    assert leaf.grad is not None
    assert torch.isfinite(leaf.grad).all()
    assert leaf.grad.abs().sum() > 0


def test_rejects_batched_input():
    with pytest.raises(ValueError, match="single-image batch"):
        _run(torch.rand(2, 3, SIDE_Y, SIDE_X))


def test_rejects_unpadded_input_for_padded_mode():
    with pytest.raises(ValueError, match="pre-padded"):
        _run(torch.rand(1, 3, SIDE_Y, SIDE_X), border_mode="smear")


def test_rejects_cutn_below_three():
    with pytest.raises(ValueError, match="cutn >= 3"):
        _run(torch.rand(1, 3, SIDE_Y, SIDE_X), cutn=2)


def test_embedder_dispatch_smart():
    from pytti.Perceptor.cutouts.augs import BatchedAugs
    from pytti.Perceptor.Embedder import CUTOUT_SAMPLERS, HDMultiClipEmbedder

    assert CUTOUT_SAMPLERS["smart"] is pytti_smart
    emb = HDMultiClipEmbedder(perceptors=[], cutout_sampler="smart", device="cpu")
    assert isinstance(emb.augs, BatchedAugs)
    with pytest.raises(ValueError, match="unknown cutout_sampler"):
        HDMultiClipEmbedder(perceptors=[], cutout_sampler="smrat", device="cpu")


def test_config_smart_default_and_batched_selectable():
    with initialize(config_path="config", version_base=None):
        cfg = compose(
            config_name="_structured_config",
            overrides=["scenes=x", "cutout_sampler=batched"],
        )
    assert OmegaConf.to_object(cfg).cutout_sampler == "batched"
    with initialize(config_path="config", version_base=None):
        cfg = compose(config_name="_structured_config", overrides=["scenes=x"])
    obj = OmegaConf.to_object(cfg)
    # A/B verdict 2026-07-31: smart@16 beat batched@40 on held-out
    # adherence at ~2.3x less tower compute
    assert obj.cutout_sampler == "smart"
    assert obj.cutouts == 16
