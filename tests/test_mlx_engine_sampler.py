"""
MLX cutout samplers — M2 slice S3 (pytti.mlx_engine.sampler).

Gates (docs/mlx-m2-seam-map.md S3, lead's slice spec):
- with INJECTED integer sizes/offsets (the exact reference construction of
  test_cutout_sampler.py / test_smart_sampler.py): cutout values vs torch
  slice+interpolate <= 2/255; input-grad rel <= 1e-5 fp32;
- offsets/sizes coordinate-convention equality with the torch samplers;
- distribution sanity for the RNG halves (population counts, size bounds,
  stratified cell coverage) under a fixed mx seed/key;
- both border-mode families (clamp + padded), both samplers;
- pad composition (mirror/wrap gathers, smear/black native) == torch F.pad.

The whole module needs mlx (darwin): every test compares against torch on
CPU, so it is marked skipif-no-mlx and imports mlx lazily inside tests
(linux CI collects but never touches mlx).
"""

import importlib.util
import math

import numpy as np
import pytest
import torch
from torch.nn import functional as F

from pytti.Perceptor.cutouts.samplers import (
    _stratified_cells as torch_stratified_cells,
)

needs_mlx = pytest.mark.skipif(
    importlib.util.find_spec("mlx") is None,
    reason="mlx not installed (darwin-only backend)",
)
pytestmark = needs_mlx

VALUE_GATE = 2 / 255  # slice gate: cutout values vs torch
GRAD_REL_GATE = 1e-5  # slice gate: input-grad rel, fp32
# fp32 flooring can pull a stratified origin at most one pixel below its
# cell's continuous lower bound (mirrors test_smart_sampler.py)
FLOOR_SLACK = 1 + 1e-3


def _mlx():
    import mlx.core as mx

    from pytti.mlx_engine import sampler

    return mx, sampler


def _identity(x):
    return x


# ---------------------------------------------------------------------------
# torch <-> mlx conversion (sampler is NHWC, torch reference is NCHW)
# ---------------------------------------------------------------------------


def to_mx_nhwc(t: torch.Tensor):
    mx, _ = _mlx()
    return mx.array(t.detach().permute(0, 2, 3, 1).contiguous().numpy())


def to_torch_nchw(a) -> torch.Tensor:
    return torch.from_numpy(np.array(a)).permute(0, 3, 1, 2)


def to_torch(a) -> torch.Tensor:
    return torch.from_numpy(np.array(a))


# ---------------------------------------------------------------------------
# torch reference construction — the exact pattern of test_cutout_sampler.py
# ---------------------------------------------------------------------------


def _pad_like_embedder(raw, side_x, side_y, padding, mode="replicate"):
    paddingx = min(round(side_x * padding), side_x)
    paddingy = min(round(side_y * padding), side_y)
    return F.pad(raw, (paddingx, paddingx, paddingy, paddingy), mode=mode)


def _int_coords(offsets, sizes, side_x, side_y):
    """Recover integer pixel offsets/sizes from the normalized [cutn, 2]
    tensors; asserts the square-cut convention (both size columns encode the
    same pixel size)."""
    ox = (offsets[:, 0] * side_x).round().long()
    oy = (offsets[:, 1] * side_y).round().long()
    sx = (sizes[:, 0] * side_x).round().long()
    sy = (sizes[:, 1] * side_y).round().long()
    assert torch.equal(sx, sy), "sizes columns disagree on the square cut size"
    # offsets must be integral pixel positions under the torch normalization
    assert torch.allclose(offsets[:, 0] * side_x, ox.float(), atol=1e-3)
    assert torch.allclose(offsets[:, 1] * side_y, oy.float(), atol=1e-3)
    return ox, oy, sx


def _reference_cutouts(input, side_x, side_y, cut_size, padding, border_mode, ox, oy, sz):
    """pytti_classic's exact per-crop math (slice + bilinear interpolate) at
    given integer offsets/sizes. `input` (torch NCHW) must be pre-padded for
    non-clamp border modes, exactly as the Embedder does."""
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


def _smooth_image(side_x, side_y):
    """Deterministic non-trivial image in [0, 1] with nonzero gradients
    everywhere (good for grad parity: no flat regions)."""
    ys = torch.linspace(0, 6.28, side_y)
    xs = torch.linspace(0, 6.28, side_x)
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")
    chans = [
        0.5 + 0.5 * torch.sin(xx + 2 * yy),
        0.5 + 0.5 * torch.cos(3 * xx - yy),
        (xx / 6.28 + yy / 6.28) / 2,
    ]
    return torch.stack(chans).unsqueeze(0)


def _n_global(cutn):
    return max(2, round(cutn * 0.25))


def _detail_grid_rows(n_detail, side_x, side_y):
    """The near-square grid's row count — mirrors pytti_smart's choice."""
    a = math.ceil(math.sqrt(n_detail))
    b = math.ceil(n_detail / a)
    return a if side_y > side_x else b


def _run_sampler(
    sampler_name,
    input_mx,
    *,
    side_x,
    side_y,
    cut_size,
    cutn,
    border_mode="clamp",
    padding=0.25,
    cut_pow=1.5,
    augs=_identity,
    noise_fac=0,
    key=None,
):
    _, mxs = _mlx()
    fn = getattr(mxs, sampler_name)
    return fn(
        input_mx,
        side_x=side_x,
        side_y=side_y,
        cut_size=cut_size,
        padding=padding,
        cutn=cutn,
        cut_pow=cut_pow,
        border_mode=border_mode,
        augs=augs,
        noise_fac=noise_fac,
        key=key,
    )


# ---------------------------------------------------------------------------
# grid_sample_border: value + input-grad parity vs torch F.grid_sample
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("n_input", [1, 6])  # 1 = the samplers' expand path
def test_grid_sample_value_parity(n_input):
    mx, mxs = _mlx()
    torch.manual_seed(0)
    b, h, w, ho, wo = 6, 23, 31, 9, 13
    inp_t = torch.rand(n_input, 3, h, w)
    # grid reaches past [-1, 1] to exercise the border clip on all sides
    grid_t = torch.rand(b, ho, wo, 2) * 2.6 - 1.3
    ref = F.grid_sample(
        inp_t.expand(b, -1, -1, -1),
        grid_t,
        mode="bilinear",
        padding_mode="border",
        align_corners=False,
    )
    out = mxs.grid_sample_border(to_mx_nhwc(inp_t), mx.array(grid_t.numpy()))
    diff = (to_torch_nchw(out) - ref).abs().max().item()
    assert diff < VALUE_GATE, f"grid_sample value diff {diff} >= 2/255"
    assert diff < 1e-6  # fp32 same-math agreement, not just the hard gate


def test_grid_sample_input_grad_parity():
    mx, mxs = _mlx()
    torch.manual_seed(1)
    b, h, w, ho, wo = 5, 19, 27, 8, 8
    inp_t = _smooth_image(w, h).expand(1, -1, -1, -1)
    grid_t = torch.rand(b, ho, wo, 2) * 2.4 - 1.2
    weights = torch.randn(b, 3, ho, wo)

    leaf = inp_t.clone().requires_grad_()
    out = F.grid_sample(
        leaf.expand(b, -1, -1, -1),
        grid_t,
        mode="bilinear",
        padding_mode="border",
        align_corners=False,
    )
    (out * weights).sum().backward()

    grid_m = mx.array(grid_t.numpy())
    w_m = mx.array(weights.permute(0, 2, 3, 1).contiguous().numpy())

    def loss(inp_m):
        return (mxs.grid_sample_border(inp_m, grid_m) * w_m).sum()

    grad_m = mx.grad(loss)(to_mx_nhwc(inp_t))
    grad_m_t = to_torch_nchw(grad_m)
    rel = ((grad_m_t - leaf.grad).abs().max() / leaf.grad.abs().max()).item()
    assert rel < GRAD_REL_GATE, f"grid_sample input-grad rel {rel} >= {GRAD_REL_GATE}"


def test_grid_sample_rejects_bad_shapes():
    mx, mxs = _mlx()
    with pytest.raises(ValueError, match="batch mismatch"):
        mxs.grid_sample_border(mx.zeros((2, 4, 4, 3)), mx.zeros((3, 2, 2, 2)))
    with pytest.raises(ValueError, match=r"\[B, Ho, Wo, 2\]"):
        mxs.grid_sample_border(mx.zeros((1, 4, 4, 3)), mx.zeros((1, 2, 2, 3)))
    with pytest.raises(ValueError, match=r"\[N, H, W, C\]"):
        mxs.grid_sample_border(mx.zeros((4, 4, 3)), mx.zeros((1, 2, 2, 2)))


# ---------------------------------------------------------------------------
# injected integer geometry through _cut_batch: the slice's core gates
# ---------------------------------------------------------------------------

# (side_x, side_y, cut_size) = (97, 65, 32): paddingx=24, paddingy=16,
# max_size=65. Geometry covers: full-frame, interior, exact-cut_size,
# heavy upsample (size < cut_size, exercising the per-cutout grid clamp /
# edge replication), tiny crop, and domain-edge offsets.
INJECT_SIDE_X, INJECT_SIDE_Y, INJECT_CUT = 97, 65, 32
INJECT_CLAMP = {  # offsets stay inside the raw frame
    "sz": [65, 40, 32, 12, 5, 65],
    "ox": [0, 57, 33, 85, 92, 32],
    "oy": [0, 25, 17, 53, 60, 0],
}
INJECT_SMEAR = {  # offsets reach into the padding by up to min(size, pad)
    "sz": [65, 40, 32, 12, 5, 20],
    "ox": [0, -24, 89, -12, 92, 101],
    "oy": [0, -16, 49, -12, 60, 61],
}


def _injected_case(border_mode):
    spec = INJECT_CLAMP if border_mode == "clamp" else INJECT_SMEAR
    sz = torch.tensor(spec["sz"], dtype=torch.long)
    ox = torch.tensor(spec["ox"], dtype=torch.long)
    oy = torch.tensor(spec["oy"], dtype=torch.long)
    raw = _smooth_image(INJECT_SIDE_X, INJECT_SIDE_Y)
    input_t = (
        raw
        if border_mode == "clamp"
        else _pad_like_embedder(raw, INJECT_SIDE_X, INJECT_SIDE_Y, 0.25)
    )
    return raw, input_t, ox, oy, sz


def _mx_cut_batch(mxs, mx, input_m, ox, oy, sz, border_mode):
    paddingx = min(round(INJECT_SIDE_X * 0.25), INJECT_SIDE_X)
    paddingy = min(round(INJECT_SIDE_Y * 0.25), INJECT_SIDE_Y)
    return mxs._cut_batch(
        input_m,
        mx.array(sz.float().numpy()),
        mx.array(ox.float().numpy()),
        mx.array(oy.float().numpy()),
        side_x=INJECT_SIDE_X,
        side_y=INJECT_SIDE_Y,
        cut_size=INJECT_CUT,
        paddingx=paddingx,
        paddingy=paddingy,
        border_mode=border_mode,
    )


@pytest.mark.parametrize("border_mode", ["clamp", "smear"])
def test_injected_geometry_value_parity(border_mode):
    mx, mxs = _mlx()
    raw, input_t, ox, oy, sz = _injected_case(border_mode)
    cutouts, offsets, sizes = _mx_cut_batch(
        mxs, mx, to_mx_nhwc(input_t), ox, oy, sz, border_mode
    )
    reference = _reference_cutouts(
        input_t, INJECT_SIDE_X, INJECT_SIDE_Y, INJECT_CUT, 0.25, border_mode, ox, oy, sz
    )
    diff = (to_torch_nchw(cutouts) - reference).abs().max().item()
    assert diff < VALUE_GATE, f"injected-geometry value diff {diff} >= 2/255"

    # returned offsets/sizes reproduce the injected convention exactly
    ox2, oy2, sz2 = _int_coords(to_torch(offsets), to_torch(sizes), INJECT_SIDE_X, INJECT_SIDE_Y)
    assert torch.equal(ox2, ox) and torch.equal(oy2, oy) and torch.equal(sz2, sz)


@pytest.mark.parametrize("border_mode", ["clamp", "smear"])
def test_injected_geometry_input_grad_parity(border_mode):
    """Grad w.r.t. the RAW image — for smear this differentiates through the
    MLX pad composition too (pad_image), vs torch F.pad(replicate)."""
    mx, mxs = _mlx()
    raw, _, ox, oy, sz = _injected_case(border_mode)
    torch.manual_seed(2)
    weights = torch.randn(len(sz), 3, INJECT_CUT, INJECT_CUT)
    w_m = mx.array(weights.permute(0, 2, 3, 1).contiguous().numpy())

    def loss(raw_m):
        padded = mxs.pad_image(raw_m, INJECT_SIDE_X, INJECT_SIDE_Y, 0.25, border_mode)
        cutouts, _, _ = _mx_cut_batch(mxs, mx, padded, ox, oy, sz, border_mode)
        return (cutouts * w_m).sum()

    grad_m = to_torch_nchw(mx.grad(loss)(to_mx_nhwc(raw)))

    leaf = raw.clone().requires_grad_()
    input_t = (
        leaf
        if border_mode == "clamp"
        else _pad_like_embedder(leaf, INJECT_SIDE_X, INJECT_SIDE_Y, 0.25)
    )
    reference = _reference_cutouts(
        input_t, INJECT_SIDE_X, INJECT_SIDE_Y, INJECT_CUT, 0.25, border_mode, ox, oy, sz
    )
    (reference * weights).sum().backward()

    rel = ((grad_m - leaf.grad).abs().max() / leaf.grad.abs().max()).item()
    assert rel < GRAD_REL_GATE, f"injected-geometry grad rel {rel} >= {GRAD_REL_GATE}"


# ---------------------------------------------------------------------------
# full samplers: RNG geometry feeds the same resampling — parity at the
# sampler's OWN draws (recovered ints -> torch reference), both samplers,
# both border-mode families
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("border_mode", ["clamp", "smear"])
@pytest.mark.parametrize(
    "sampler_name", ["pytti_batched", "pytti_smart", "pytti_full"]
)
def test_full_sampler_value_parity(sampler_name, border_mode):
    mx, _ = _mlx()
    side_x, side_y, cut_size, cutn, padding = 97, 65, 32, 32, 0.25
    raw = _smooth_image(side_x, side_y)
    input_t = (
        raw if border_mode == "clamp" else _pad_like_embedder(raw, side_x, side_y, padding)
    )
    cutouts, offsets, sizes = _run_sampler(
        sampler_name,
        to_mx_nhwc(input_t),
        side_x=side_x,
        side_y=side_y,
        cut_size=cut_size,
        cutn=cutn,
        border_mode=border_mode,
        key=mx.random.key(0),
    )
    assert cutouts.shape == (cutn, cut_size, cut_size, 3)
    ox, oy, sz = _int_coords(to_torch(offsets), to_torch(sizes), side_x, side_y)
    reference = _reference_cutouts(
        input_t, side_x, side_y, cut_size, padding, border_mode, ox, oy, sz
    )
    diff = (to_torch_nchw(cutouts) - reference).abs().max().item()
    assert diff < VALUE_GATE, f"{sampler_name} value diff {diff} >= 2/255"


@pytest.mark.parametrize(
    "sampler_name", ["pytti_batched", "pytti_smart", "pytti_full"]
)
def test_offset_size_semantics_clamp(sampler_name):
    """Torch samplers' coordinate convention: offsets are the crop's top-left
    corner in raw-image pixels normalized by (side_x, side_y). cut_size ==
    max_size forces every size to max_size, making resampling the identity,
    so each cutout must equal the raw pixels at the reported location."""
    mx, _ = _mlx()
    side_x, side_y, cut_size, cutn = 64, 32, 32, 24
    torch.manual_seed(3)
    raw = torch.rand(1, 3, side_y, side_x)
    cutouts, offsets, sizes = _run_sampler(
        sampler_name,
        to_mx_nhwc(raw),
        side_x=side_x,
        side_y=side_y,
        cut_size=cut_size,
        cutn=cutn,
        key=mx.random.key(1),
    )
    offsets_t, sizes_t = to_torch(offsets), to_torch(sizes)
    ox, oy, sz = _int_coords(offsets_t, sizes_t, side_x, side_y)
    assert (sz == 32).all()
    assert torch.allclose(
        sizes_t, torch.tensor([32 / side_x, 32 / side_y]).expand(cutn, 2)
    )
    assert (oy == 0).all()  # side_y - size == 0 leaves a single valid row
    assert (ox >= 0).all() and (ox <= side_x - 32).all()
    cut_t = to_torch_nchw(cutouts)
    for i in range(cutn):
        x, y = ox[i].item(), oy[i].item()
        assert torch.allclose(
            cut_t[i], raw[0, :, y : y + 32, x : x + 32], atol=1e-4
        ), f"cutout {i} does not match raw pixels at reported offset ({x}, {y})"


@pytest.mark.parametrize("sampler_name", ["pytti_batched", "pytti_smart"])
def test_offset_size_semantics_padded(sampler_name):
    """Non-clamp modes: input arrives pre-padded, crops are shifted by
    (paddingx, paddingy) inside it, reported offsets stay in unpadded
    coordinates (negative when reaching into the padding)."""
    mx, mxs = _mlx()
    side_x, side_y, cut_size, cutn, padding = 64, 32, 32, 24, 0.25
    paddingx = min(round(side_x * padding), side_x)  # 16
    paddingy = min(round(side_y * padding), side_y)  # 8
    torch.manual_seed(4)
    raw = torch.rand(1, 3, side_y, side_x)
    padded_m = mxs.pad_image(to_mx_nhwc(raw), side_x, side_y, padding, "smear")
    padded_t = to_torch_nchw(padded_m)
    cutouts, offsets, sizes = _run_sampler(
        sampler_name,
        padded_m,
        side_x=side_x,
        side_y=side_y,
        cut_size=cut_size,
        cutn=cutn,
        border_mode="smear",
        key=mx.random.key(2),
    )
    ox, oy, sz = _int_coords(to_torch(offsets), to_torch(sizes), side_x, side_y)
    assert (sz == 32).all()
    assert (ox >= -paddingx).all() and (oy >= -paddingy).all()
    if sampler_name == "pytti_smart":
        n_global = _n_global(cutn)
        # anchors stay inside the unpadded frame on every border mode
        assert (ox[:n_global] >= 0).all() and (ox[:n_global] + 32 <= side_x).all()
        assert (oy[:n_global] == 0).all()
    cut_t = to_torch_nchw(cutouts)
    for i in range(cutn):
        x, y = ox[i].item(), oy[i].item()
        assert torch.allclose(
            cut_t[i],
            padded_t[
                0, :, paddingy + y : paddingy + y + 32, paddingx + x : paddingx + x + 32
            ],
            atol=1e-4,
        ), f"cutout {i} does not match padded pixels shifted by padding at ({x}, {y})"
    # the point of the padded branch: some crop reached into the padding
    assert bool((ox < 0).any() or (oy < 0).any()), (
        "seeded draw expected at least one negative offset"
    )


# ---------------------------------------------------------------------------
# distribution sanity — the RNG halves, fixed mx seed/key
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("border_mode", ["clamp", "smear"])
def test_batched_shape_and_range_invariants(border_mode):
    mx, mxs = _mlx()
    side_x, side_y, cut_size, padding, cutn = 96, 64, 32, 0.25, 40
    torch.manual_seed(5)
    raw = torch.rand(1, 3, side_y, side_x)
    input_m = mxs.pad_image(to_mx_nhwc(raw), side_x, side_y, padding, border_mode)
    cutouts, offsets, sizes = _run_sampler(
        "pytti_batched",
        input_m,
        side_x=side_x,
        side_y=side_y,
        cut_size=cut_size,
        cutn=cutn,
        border_mode=border_mode,
        key=mx.random.key(3),
    )
    assert cutouts.shape == (cutn, cut_size, cut_size, 3)
    assert offsets.shape == (cutn, 2) and sizes.shape == (cutn, 2)
    cut_t = to_torch_nchw(cutouts)
    assert torch.isfinite(cut_t).all()
    # bilinear samples are convex combinations of input pixels
    input_t = to_torch_nchw(input_m)
    assert cut_t.min() >= input_t.min() - 1e-6
    assert cut_t.max() <= input_t.max() + 1e-6

    ox, oy, sz = _int_coords(to_torch(offsets), to_torch(sizes), side_x, side_y)
    max_size = min(side_x, side_y)
    assert (sz >= 1).all() and (sz <= max_size).all()
    assert len(torch.unique(sz)) > 1, "N(0.8, 0.3) sizes degenerate at cutn=40"
    paddingx = min(round(side_x * padding), side_x)
    paddingy = min(round(side_y * padding), side_y)
    if border_mode == "clamp":
        assert (ox >= 0).all() and (ox + sz <= side_x).all()
        assert (oy >= 0).all() and (oy + sz <= side_y).all()
    else:
        # crops reach into the padding by at most min(size, padding) px and
        # always stay inside the padded input
        assert (ox >= -torch.minimum(sz, torch.tensor(paddingx))).all()
        assert (oy >= -torch.minimum(sz, torch.tensor(paddingy))).all()
        assert (paddingx + ox + sz <= side_x + 2 * paddingx).all()
        assert (paddingy + oy + sz <= side_y + 2 * paddingy).all()


@pytest.mark.parametrize("border_mode", ["clamp", "smear"])
@pytest.mark.parametrize("cutn", [4, 16, 40])
def test_smart_population_counts_and_size_bounds(border_mode, cutn):
    mx, mxs = _mlx()
    side_x, side_y, cut_size, padding = 96, 64, 8, 0.25
    torch.manual_seed(6)
    raw = torch.rand(1, 3, side_y, side_x)
    input_m = mxs.pad_image(to_mx_nhwc(raw), side_x, side_y, padding, border_mode)
    cutouts, offsets, sizes = _run_sampler(
        "pytti_smart",
        input_m,
        side_x=side_x,
        side_y=side_y,
        cut_size=cut_size,
        cutn=cutn,
        border_mode=border_mode,
        key=mx.random.key(4),
    )
    assert cutouts.shape == (cutn, cut_size, cut_size, 3)
    _, _, sz = _int_coords(to_torch(offsets), to_torch(sizes), side_x, side_y)
    n_global = _n_global(cutn)
    max_size = min(side_x, side_y)  # 64
    # global anchors: exactly the inscribed square, first n_global rows
    assert (sz[:n_global] == max_size).all()
    # detail: floor(f * max_size) with f in [0.2, 0.55); cut_size=8 never binds
    d_sz = sz[n_global:]
    assert d_sz.shape[0] == cutn - n_global
    assert (d_sz >= math.floor(0.2 * max_size)).all()  # >= 12
    assert (d_sz <= math.floor(0.55 * max_size)).all()  # <= 35
    assert (d_sz < max_size).all()


def test_smart_detail_sizes_clamped_up_to_cut_size():
    # cut_size 48 > 0.55 * 64: every detail size must be raised to exactly 48
    mx, _ = _mlx()
    side_x, side_y = 96, 64
    torch.manual_seed(7)
    raw = torch.rand(1, 3, side_y, side_x)
    _, offsets, sizes = _run_sampler(
        "pytti_smart",
        to_mx_nhwc(raw),
        side_x=side_x,
        side_y=side_y,
        cut_size=48,
        cutn=16,
        key=mx.random.key(5),
    )
    _, _, sz = _int_coords(to_torch(offsets), to_torch(sizes), side_x, side_y)
    assert (sz[_n_global(16) :] == 48).all()


@pytest.mark.parametrize("side_x,side_y,free_axis", [(128, 64, "x"), (64, 128, "y")])
def test_smart_global_anchors_stratified_along_free_axis(side_x, side_y, free_axis):
    mx, _ = _mlx()
    cutn = 16
    n_global = _n_global(cutn)  # 4
    torch.manual_seed(8)
    raw = torch.rand(1, 3, side_y, side_x)
    _, offsets, sizes = _run_sampler(
        "pytti_smart",
        to_mx_nhwc(raw),
        side_x=side_x,
        side_y=side_y,
        cut_size=8,
        cutn=cutn,
        key=mx.random.key(6),
    )
    ox, oy, _ = _int_coords(to_torch(offsets), to_torch(sizes), side_x, side_y)
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


def test_smart_global_anchors_square_canvas_identical():
    mx, _ = _mlx()
    side = 64
    torch.manual_seed(9)
    raw = torch.rand(1, 3, side, side)
    cutouts, offsets, sizes = _run_sampler(
        "pytti_smart",
        to_mx_nhwc(raw),
        side_x=side,
        side_y=side,
        cut_size=8,
        cutn=16,
        key=mx.random.key(7),
    )
    n_global = _n_global(16)
    offsets_t = to_torch(offsets)
    assert (offsets_t[:n_global] == 0).all()
    _, _, sz = _int_coords(offsets_t, to_torch(sizes), side, side)
    assert (sz[:n_global] == side).all()
    cut_t = to_torch(cutouts)
    for i in range(1, n_global):
        assert torch.equal(cut_t[0], cut_t[i])


@pytest.mark.parametrize("border_mode", ["clamp", "smear"])
def test_smart_detail_stratification_covers_every_cell(border_mode):
    """Detail row i is cell i of the near-square grid (sizes are permuted,
    cells are not): every crop origin must land inside its own cell, mapped
    onto that cutout's valid offset domain."""
    mx, mxs = _mlx()
    side_x, side_y, cut_size, padding, cutn = 96, 64, 8, 0.25, 16
    n_global = _n_global(cutn)
    n_detail = cutn - n_global  # 12 -> 4x3 grid on this landscape canvas
    torch.manual_seed(10)
    raw = torch.rand(1, 3, side_y, side_x)
    input_m = mxs.pad_image(to_mx_nhwc(raw), side_x, side_y, padding, border_mode)
    _, offsets, sizes = _run_sampler(
        "pytti_smart",
        input_m,
        side_x=side_x,
        side_y=side_y,
        cut_size=cut_size,
        cutn=cutn,
        border_mode=border_mode,
        key=mx.random.key(8),
    )
    ox, oy, sz = _int_coords(to_torch(offsets), to_torch(sizes), side_x, side_y)
    d_ox, d_oy, d_sz = ox[n_global:], oy[n_global:], sz[n_global:]

    n_rows = _detail_grid_rows(n_detail, side_x, side_y)
    cx_lo, cx_w, cy_lo, cy_h = mxs._stratified_cells(n_detail, n_rows)
    cx_lo, cx_w = to_torch(cx_lo), to_torch(cx_w)
    cy_lo, cy_h = to_torch(cy_lo), to_torch(cy_h)
    paddingx = min(round(side_x * padding), side_x)
    paddingy = min(round(side_y * padding), side_y)
    for i in range(n_detail):
        s = d_sz[i].item()
        if border_mode == "clamp":
            lo_x, ext_x = 0.0, side_x - s
            lo_y, ext_y = 0.0, side_y - s
        else:
            px, py = min(s, paddingx), min(s, paddingy)
            lo_x, ext_x = -px, side_x - s + 2 * px
            lo_y, ext_y = -py, side_y - s + 2 * py
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


# ---------------------------------------------------------------------------
# pytti_full: full-vision distribution sanity (mirrors test_full_sampler.py)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("border_mode", ["clamp", "smear"])
@pytest.mark.parametrize("cutn", [1, 8, 16])
def test_full_vision_every_size_is_max_size(border_mode, cutn):
    mx, mxs = _mlx()
    side_x, side_y, cut_size, padding = 96, 64, 8, 0.25
    torch.manual_seed(20)
    raw = torch.rand(1, 3, side_y, side_x)
    input_m = mxs.pad_image(to_mx_nhwc(raw), side_x, side_y, padding, border_mode)
    cutouts, offsets, sizes = _run_sampler(
        "pytti_full",
        input_m,
        side_x=side_x,
        side_y=side_y,
        cut_size=cut_size,
        cutn=cutn,
        border_mode=border_mode,
        key=mx.random.key(20),
    )
    assert cutouts.shape == (cutn, cut_size, cut_size, 3)
    sizes_t = to_torch(sizes)
    ox, oy, sz = _int_coords(to_torch(offsets), sizes_t, side_x, side_y)
    max_size = min(side_x, side_y)  # 64
    assert (sz == max_size).all()  # no cut below 100% resolution, ever
    # min-side size column reads exactly 1.0: the coherence-weighting anchor
    # detector (sizes_to_coherence_weights) fires on every row
    assert (sizes_t[:, 1] == 1.0).all()
    # offsets stay inside the unpadded frame on EVERY border mode
    assert (ox >= 0).all() and (ox + max_size <= side_x).all()
    assert (oy == 0).all()  # free_y == 0 on this landscape canvas


def test_full_vision_square_canvas_identical():
    mx, _ = _mlx()
    side = 64
    torch.manual_seed(21)
    raw = torch.rand(1, 3, side, side)
    cutouts, offsets, sizes = _run_sampler(
        "pytti_full",
        to_mx_nhwc(raw),
        side_x=side,
        side_y=side,
        cut_size=8,
        cutn=12,
        key=mx.random.key(21),
    )
    offsets_t = to_torch(offsets)
    assert (offsets_t == 0).all()
    _, _, sz = _int_coords(offsets_t, to_torch(sizes), side, side)
    assert (sz == side).all()
    cut_t = to_torch(cutouts)
    for i in range(1, 12):
        assert torch.equal(cut_t[0], cut_t[i])


@pytest.mark.parametrize("side_x,side_y,free_axis", [(128, 64, "x"), (64, 128, "y")])
def test_full_vision_offsets_stratified_along_free_axis(side_x, side_y, free_axis):
    mx, _ = _mlx()
    cutn = 8
    torch.manual_seed(22)
    raw = torch.rand(1, 3, side_y, side_x)
    _, offsets, sizes = _run_sampler(
        "pytti_full",
        to_mx_nhwc(raw),
        side_x=side_x,
        side_y=side_y,
        cut_size=8,
        cutn=cutn,
        key=mx.random.key(22),
    )
    ox, oy, sz = _int_coords(to_torch(offsets), to_torch(sizes), side_x, side_y)
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


def test_full_vision_padded_never_reaches_padding():
    """Unlike batched/smart detail cuts, full-vision crops never reach into
    the padding — smart's anchor rule applied to every row. cut_size ==
    max_size makes resampling the identity, so each cutout must equal the
    RAW pixels at the reported (non-negative) offset."""
    mx, mxs = _mlx()
    side_x, side_y, cut_size, cutn, padding = 64, 32, 32, 12, 0.25
    torch.manual_seed(23)
    raw = torch.rand(1, 3, side_y, side_x)
    padded_m = mxs.pad_image(to_mx_nhwc(raw), side_x, side_y, padding, "smear")
    cutouts, offsets, sizes = _run_sampler(
        "pytti_full",
        padded_m,
        side_x=side_x,
        side_y=side_y,
        cut_size=cut_size,
        cutn=cutn,
        border_mode="smear",
        key=mx.random.key(23),
    )
    ox, oy, sz = _int_coords(to_torch(offsets), to_torch(sizes), side_x, side_y)
    assert (sz == 32).all()
    assert (ox >= 0).all() and (ox + 32 <= side_x).all()
    assert (oy == 0).all()
    cut_t = to_torch_nchw(cutouts)
    for i in range(cutn):
        x, y = ox[i].item(), oy[i].item()
        assert torch.allclose(
            cut_t[i], raw[0, :, y : y + 32, x : x + 32], atol=1e-4
        ), f"cutout {i} reached outside the unpadded frame at ({x}, {y})"


def test_full_rejects_cutn_below_one():
    mx, _ = _mlx()
    with pytest.raises(ValueError, match="cutn >= 1"):
        _run_sampler(
            "pytti_full",
            mx.zeros((1, 32, 32, 3)),
            side_x=32,
            side_y=32,
            cut_size=16,
            cutn=0,
        )


@pytest.mark.parametrize(
    "n_cells,n_rows", [(1, 1), (2, 1), (5, 2), (12, 3), (13, 4), (30, 5)]
)
def test_stratified_cells_match_torch(n_cells, n_rows):
    """The MLX cell boxes equal the torch ones exactly AND tile the square."""
    _, mxs = _mlx()
    got = [to_torch(a) for a in mxs._stratified_cells(n_cells, n_rows)]
    want = torch_stratified_cells(n_cells, n_rows, "cpu", torch.float32)
    for g, w in zip(got, want, strict=True):
        assert torch.equal(g, w)
    x_lo, x_w, y_lo, y_h = got
    assert (x_lo >= 0).all() and (x_lo + x_w <= 1 + 1e-6).all()
    assert (y_lo >= 0).all() and (y_lo + y_h <= 1 + 1e-6).all()
    assert torch.isclose((x_w * y_h).sum(), torch.tensor(1.0), atol=1e-5)


def test_stratified_cells_rejects_bad_rows():
    _, mxs = _mlx()
    with pytest.raises(ValueError, match="1 <= n_rows <= n_cells"):
        mxs._stratified_cells(4, 5)


# ---------------------------------------------------------------------------
# pad composition == torch F.pad
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "border_mode,torch_mode",
    [
        ("mirror", "reflect"),
        ("smear", "replicate"),
        ("wrap", "circular"),
        ("black", "constant"),
    ],
)
def test_pad_image_matches_torch(border_mode, torch_mode):
    side_x, side_y, padding = 13, 9, 0.25  # paddingx=3, paddingy=2
    torch.manual_seed(11)
    raw = torch.rand(1, 3, side_y, side_x)
    _, mxs = _mlx()
    got = to_torch_nchw(mxs.pad_image(to_mx_nhwc(raw), side_x, side_y, padding, border_mode))
    want = _pad_like_embedder(raw, side_x, side_y, padding, mode=torch_mode)
    assert torch.equal(got, want), f"{border_mode} pad diverges from F.pad({torch_mode})"


def test_pad_image_clamp_is_identity():
    _, mxs = _mlx()
    torch.manual_seed(12)
    raw = torch.rand(1, 3, 9, 13)
    got = mxs.pad_image(to_mx_nhwc(raw), 13, 9, 0.25, "clamp")
    assert torch.equal(to_torch_nchw(got), raw)


def test_pad_image_fail_loud():
    _, mxs = _mlx()
    torch.manual_seed(13)
    raw = to_mx_nhwc(torch.rand(1, 3, 9, 13))
    with pytest.raises(ValueError, match="unknown border_mode"):
        mxs.pad_image(raw, 13, 9, 0.25, "smrat")
    with pytest.raises(ValueError, match="expected input"):
        mxs.pad_image(raw, 9, 13, 0.25, "smear")  # sides swapped
    # torch reflect contract: padding must stay below the dim size
    with pytest.raises(ValueError, match="must be < dim size"):
        mxs.pad_image(raw, 13, 9, 1.0, "mirror")


# ---------------------------------------------------------------------------
# RNG contract: explicit keys pure + deterministic, implicit mode seedable
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "sampler_name", ["pytti_batched", "pytti_smart", "pytti_full"]
)
def test_explicit_key_is_pure_and_deterministic(sampler_name):
    mx, _ = _mlx()
    torch.manual_seed(14)
    raw = to_mx_nhwc(torch.rand(1, 3, 64, 96))
    kwargs = dict(side_x=96, side_y=64, cut_size=16, cutn=8, noise_fac=0.1)
    key = mx.random.key(9)

    mx.random.seed(123)
    probe_before = mx.random.uniform(shape=(4,))
    mx.random.seed(123)
    c1, o1, s1 = _run_sampler(sampler_name, raw, key=key, **kwargs)
    probe_after = mx.random.uniform(shape=(4,))
    # keyed call neither read nor advanced the global stream
    assert mx.array_equal(probe_before, probe_after).item()

    c2, o2, s2 = _run_sampler(sampler_name, raw, key=key, **kwargs)
    assert mx.array_equal(c1, c2).item()
    assert mx.array_equal(o1, o2).item()
    assert mx.array_equal(s1, s2).item()

    _, o3, _ = _run_sampler(sampler_name, raw, key=mx.random.key(10), **kwargs)
    assert not mx.array_equal(o1, o3).item(), "different keys drew identical geometry"

    # geometry subkeys are independent of noise_fac (fixed split count)
    _, o4, s4 = _run_sampler(sampler_name, raw, key=key, **{**kwargs, "noise_fac": 0})
    assert mx.array_equal(o1, o4).item()
    assert mx.array_equal(s1, s4).item()


@pytest.mark.parametrize(
    "sampler_name", ["pytti_batched", "pytti_smart", "pytti_full"]
)
def test_implicit_state_is_seedable_and_advances(sampler_name):
    mx, _ = _mlx()
    torch.manual_seed(15)
    raw = to_mx_nhwc(torch.rand(1, 3, 64, 96))
    kwargs = dict(side_x=96, side_y=64, cut_size=16, cutn=8, noise_fac=0.1)

    mx.random.seed(42)
    c1, o1, _ = _run_sampler(sampler_name, raw, key=None, **kwargs)
    c1b, o1b, _ = _run_sampler(sampler_name, raw, key=None, **kwargs)
    mx.random.seed(42)
    c2, o2, _ = _run_sampler(sampler_name, raw, key=None, **kwargs)
    assert mx.array_equal(c1, c2).item() and mx.array_equal(o1, o2).item()
    # consecutive implicit calls advance the stream (fresh draws per step)
    assert not mx.array_equal(o1, o1b).item() or not mx.array_equal(c1, c1b).item()


@pytest.mark.parametrize(
    "sampler_name", ["pytti_batched", "pytti_smart", "pytti_full"]
)
def test_implicit_state_compiles(sampler_name):
    """The assembly contract (seam map §3): with ``mx.random.state`` in the
    compiled function's inputs/outputs, implicit-mode draws stay fresh every
    call instead of freezing into the trace, and reseeding reproduces."""
    import functools

    mx, _ = _mlx()
    torch.manual_seed(17)
    raw = to_mx_nhwc(torch.rand(1, 3, 64, 96))
    state = [mx.random.state]

    @functools.partial(mx.compile, inputs=state, outputs=state)
    def step():
        cutouts, offsets, sizes = _run_sampler(
            sampler_name,
            raw,
            side_x=96,
            side_y=64,
            cut_size=16,
            cutn=8,
            noise_fac=0.1,
            key=None,
        )
        return cutouts, offsets, sizes

    mx.random.seed(99)
    _, o1, _ = step()
    _, o2, _ = step()
    mx.random.seed(99)
    _, o3, _ = step()
    assert not mx.array_equal(o1, o2).item(), "compiled draws froze into the trace"
    assert mx.array_equal(o1, o3).item(), "compiled draws not reproducible per seed"


# ---------------------------------------------------------------------------
# augs / noise application + fail-loud input contracts
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "sampler_name", ["pytti_batched", "pytti_smart", "pytti_full"]
)
def test_augs_and_noise_are_applied(sampler_name):
    mx, _ = _mlx()
    torch.manual_seed(16)
    raw = to_mx_nhwc(torch.rand(1, 3, 48, 64))
    cutouts, _, _ = _run_sampler(
        sampler_name,
        raw,
        side_x=64,
        side_y=48,
        cut_size=16,
        cutn=8,
        augs=lambda x: x + 10.0,
        noise_fac=0.1,
        key=mx.random.key(11),
    )
    cut_t = to_torch(cutouts)
    # augs ran (values shifted by 10) and noise_fac perturbed around that
    assert cut_t.min() > 5.0
    assert cut_t.max() < 15.0


@pytest.mark.parametrize(
    "sampler_name", ["pytti_batched", "pytti_smart", "pytti_full"]
)
def test_rejects_batched_input(sampler_name):
    mx, _ = _mlx()
    with pytest.raises(ValueError, match="single-image batch"):
        _run_sampler(
            sampler_name,
            mx.zeros((2, 32, 32, 3)),
            side_x=32,
            side_y=32,
            cut_size=16,
            cutn=4,
        )


@pytest.mark.parametrize(
    "sampler_name", ["pytti_batched", "pytti_smart", "pytti_full"]
)
def test_rejects_unpadded_input_for_padded_mode(sampler_name):
    mx, _ = _mlx()
    with pytest.raises(ValueError, match="pre-padded"):
        _run_sampler(
            sampler_name,
            mx.zeros((1, 32, 32, 3)),
            side_x=32,
            side_y=32,
            cut_size=16,
            cutn=4,
            border_mode="smear",
        )


def test_smart_rejects_cutn_below_three():
    mx, _ = _mlx()
    with pytest.raises(ValueError, match="cutn >= 3"):
        _run_sampler(
            "pytti_smart",
            mx.zeros((1, 32, 32, 3)),
            side_x=32,
            side_y=32,
            cut_size=16,
            cutn=2,
        )
