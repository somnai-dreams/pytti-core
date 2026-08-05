"""
Tests for the full-vision cutout sampler (`pytti_full`).

Contract under test (lead's spec — "exclusively full vision, no smaller cuts
than 100% resolution"):
- identical signature / input / return conventions to `pytti_batched` and
  `pytti_smart` (pre-padded input for non-clamp border modes, offsets/sizes
  [cutn, 2] normalized by (side_x, side_y), augs + noise applied last);
- EVERY cutout is exactly the inscribed square: size == min(side_x, side_y)
  for all cutn rows, both border modes;
- square canvas: all offsets 0 and the crops are byte-identical pre-aug;
- non-square canvas: offsets stratified along the free axis (smart's anchor
  lanes at n_global == cutn), always inside the unpadded frame;
- cut_pow accepted but unused; fail-loud on malformed input; gradients
  flow; works at cutn as low as 1.
"""

import pytest
import torch
from hydra import compose, initialize
from omegaconf import OmegaConf

from pytti.Perceptor.cutouts.samplers import pytti_full

# landscape geometry unless a test needs something specific: max_size 64
SIDE_X, SIDE_Y, CUT_SIZE = 96, 64, 8
PADDING = 0.25
# fp32 flooring can pull a stratified origin at most one pixel below its
# lane's continuous lower bound (mirrors test_smart_sampler.py)
FLOOR_SLACK = 1 + 1e-3


def _identity(x):
    return x


def _pad_like_embedder(raw, side_x, side_y, padding, mode="replicate"):
    from torch.nn import functional as F

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
    cutn=12,
    border_mode="clamp",
    padding=PADDING,
    augs=_identity,
    noise_fac=0,
    cut_pow=1.5,
):
    return pytti_full(
        input=input,
        side_x=side_x,
        side_y=side_y,
        cut_size=cut_size,
        padding=padding,
        cutn=cutn,
        cut_pow=cut_pow,  # accepted but unused
        border_mode=border_mode,
        augs=augs,
        noise_fac=noise_fac,
        device="cpu",
    )


@pytest.mark.parametrize("border_mode", ["clamp", "smear"])
@pytest.mark.parametrize("cutn", [1, 8, 16])
def test_every_size_is_exactly_max_size(border_mode, cutn):
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

    ox, oy, sz = _int_coords(offsets, sizes, SIDE_X, SIDE_Y)
    max_size = min(SIDE_X, SIDE_Y)  # 64
    assert (sz == max_size).all()  # no cut smaller than 100% resolution, ever
    # the size column normalization: min-side column reads exactly 1.0 —
    # coherence_weighting's anchor detector fires on every row
    assert (sizes[:, 1] == 1.0).all()
    # offsets stay inside the unpadded frame on EVERY border mode
    assert (ox >= 0).all() and (ox + max_size <= SIDE_X).all()
    assert (oy == 0).all()  # free_y == 0 on this landscape canvas


def test_square_canvas_all_zero_offsets_and_identical_crops():
    torch.manual_seed(1)
    side = 64
    raw = torch.rand(1, 3, side, side)
    cutouts, offsets, sizes = _run(raw, side_x=side, side_y=side, cutn=12)
    assert (offsets == 0).all()
    _, _, sz = _int_coords(offsets, sizes, side, side)
    assert (sz == side).all()
    # same crop, identity augs, no noise: every cutout is an exact copy —
    # augs + noise_fac are the designed diversity source
    for i in range(1, 12):
        assert torch.equal(cutouts[0], cutouts[i])


@pytest.mark.parametrize(
    "side_x,side_y,free_axis", [(128, 64, "x"), (64, 128, "y")]
)
def test_offsets_stratified_along_free_axis(side_x, side_y, free_axis):
    torch.manual_seed(2)
    cutn = 8
    raw = torch.rand(1, 3, side_y, side_x)
    _, offsets, sizes = _run(raw, side_x=side_x, side_y=side_y, cutn=cutn)
    ox, oy, sz = _int_coords(offsets, sizes, side_x, side_y)
    max_size = min(side_x, side_y)
    assert (sz == max_size).all()
    free = max(side_x, side_y) - max_size  # 64
    strat, fixed = (ox, oy) if free_axis == "x" else (oy, ox)
    assert (fixed == 0).all()
    lane_w = free / cutn
    for i in range(cutn):
        v = strat[i].item()
        assert i * lane_w - FLOOR_SLACK < v < (i + 1) * lane_w + 1e-3, (
            f"anchor {i} at {v} outside its lane [{i * lane_w}, {(i + 1) * lane_w})"
        )


def test_offset_size_semantics_clamp():
    """Same coordinate convention as pytti_batched/pytti_smart: offsets are
    the crop's top-left corner in raw-image pixels normalized by
    (side_x, side_y). cut_size == max_size makes resampling the identity, so
    each cutout must equal the raw pixels at exactly the reported location."""
    torch.manual_seed(3)
    side_x, side_y, cut_size, cutn = 64, 32, 32, 12
    raw = torch.rand(1, 3, side_y, side_x)
    cutouts, offsets, sizes = _run(
        raw, side_x=side_x, side_y=side_y, cut_size=cut_size, cutn=cutn
    )
    ox, oy, sz = _int_coords(offsets, sizes, side_x, side_y)
    assert (sz == 32).all()
    assert torch.allclose(
        sizes, torch.tensor([32 / side_x, 32 / side_y]).expand(cutn, 2)
    )
    assert (oy == 0).all()
    assert (ox >= 0).all() and (ox <= side_x - 32).all()
    for i in range(cutn):
        x, y = ox[i].item(), oy[i].item()
        assert torch.allclose(
            cutouts[i], raw[0, :, y : y + 32, x : x + 32], atol=1e-4
        ), f"cutout {i} does not match raw pixels at reported offset ({x}, {y})"


def test_offset_size_semantics_padded_never_reaches_padding():
    """Non-clamp border modes: input arrives pre-padded, crops shift by
    (paddingx, paddingy) inside it, reported offsets stay in unpadded
    coordinates. Unlike batched/smart detail cuts, full-vision crops NEVER
    reach into the padding (a full-frame view of padding is not a view of
    the image) — smart's anchor rule, applied to every row."""
    torch.manual_seed(4)
    side_x, side_y, cut_size, cutn = 64, 32, 32, 12
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
    assert (ox >= 0).all() and (ox + 32 <= side_x).all()
    assert (oy == 0).all()
    for i in range(cutn):
        x, y = ox[i].item(), oy[i].item()
        # crop lands at padding + offset in the padded frame == raw pixels
        assert torch.allclose(
            cutouts[i],
            padded[
                0, :, paddingy + y : paddingy + y + 32, paddingx + x : paddingx + x + 32
            ],
            atol=1e-4,
        ), f"cutout {i} does not match padded pixels shifted by padding at ({x}, {y})"
        assert torch.allclose(
            cutouts[i], raw[0, :, y : y + 32, x : x + 32], atol=1e-4
        ), f"cutout {i} reached outside the unpadded frame at ({x}, {y})"


def test_cut_pow_is_accepted_but_unused():
    raw = torch.rand(1, 3, SIDE_Y, SIDE_X)
    torch.manual_seed(5)
    a = _run(raw, cutn=8, cut_pow=0.5)
    torch.manual_seed(5)
    b = _run(raw, cutn=8, cut_pow=3.0)
    for ta, tb in zip(a, b, strict=True):
        assert torch.equal(ta, tb)


def test_augs_and_noise_are_applied():
    torch.manual_seed(6)
    raw = torch.rand(1, 3, SIDE_Y, SIDE_X)
    cutouts, _, _ = _run(
        raw, cut_size=16, cutn=8, augs=lambda x: x + 10.0, noise_fac=0.1
    )
    assert cutouts.min() > 5.0
    assert cutouts.max() < 15.0


def test_gradient_flows_to_input():
    torch.manual_seed(7)
    leaf = torch.rand(1, 3, SIDE_Y, SIDE_X, requires_grad=True)
    cutouts, _, _ = _run(leaf, cut_size=32, cutn=8)
    cutouts.sum().backward()
    assert leaf.grad is not None
    assert torch.isfinite(leaf.grad).all()
    assert leaf.grad.abs().sum() > 0


def test_works_at_cutn_one():
    torch.manual_seed(8)
    raw = torch.rand(1, 3, SIDE_Y, SIDE_X)
    cutouts, offsets, sizes = _run(raw, cutn=1)
    assert cutouts.shape == (1, 3, CUT_SIZE, CUT_SIZE)
    _, _, sz = _int_coords(offsets, sizes, SIDE_X, SIDE_Y)
    assert (sz == min(SIDE_X, SIDE_Y)).all()


def test_rejects_batched_input():
    with pytest.raises(ValueError, match="single-image batch"):
        _run(torch.rand(2, 3, SIDE_Y, SIDE_X))


def test_rejects_unpadded_input_for_padded_mode():
    with pytest.raises(ValueError, match="pre-padded"):
        _run(torch.rand(1, 3, SIDE_Y, SIDE_X), border_mode="smear")


def test_rejects_cutn_below_one():
    with pytest.raises(ValueError, match="cutn >= 1"):
        _run(torch.rand(1, 3, SIDE_Y, SIDE_X), cutn=0)


def test_embedder_dispatch_full():
    from pytti.Perceptor.cutouts.augs import BatchedAugs
    from pytti.Perceptor.Embedder import CUTOUT_SAMPLERS, HDMultiClipEmbedder

    assert CUTOUT_SAMPLERS["full"] is pytti_full
    emb = HDMultiClipEmbedder(perceptors=[], cutout_sampler="full", device="cpu")
    assert isinstance(emb.augs, BatchedAugs)


def test_config_full_selectable_smart_still_default():
    with initialize(config_path="config", version_base=None):
        cfg = compose(
            config_name="_structured_config",
            overrides=["scenes=x", "cutout_sampler=full"],
        )
    assert OmegaConf.to_object(cfg).cutout_sampler == "full"
    with initialize(config_path="config", version_base=None):
        cfg = compose(config_name="_structured_config", overrides=["scenes=x"])
    assert OmegaConf.to_object(cfg).cutout_sampler == "smart"


def test_coherence_weighting_all_anchor_batch_is_uniform():
    """The designed interaction: full-vision batches are all-anchors, and
    sizes_to_coherence_weights renormalizes an all-anchor batch to exactly
    uniform — coherence_weighting=true is a harmless no-op under full.
    (The definitional unit test lives in test_coherence_weighting.py::
    test_all_anchor_batch_is_exactly_uniform; this crosses it with the REAL
    sampler's sizes output.)"""
    from pytti.Perceptor.Prompt import sizes_to_coherence_weights

    torch.manual_seed(9)
    raw = torch.rand(1, 3, SIDE_Y, SIDE_X)
    _, _, sizes = _run(raw, cutn=12)
    weights = sizes_to_coherence_weights(sizes, SIDE_X, SIDE_Y)
    assert torch.equal(weights, torch.ones(12))
