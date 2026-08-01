"""
MLX direct losses (docs/mlx-m2-seam-map.md §C, slice S2 — mlx_engine/losses.py).

Every test is a parity gate against the torch LossAug classes on fixed
inputs: values AND input-grads <= 1e-6 in fp32, plus explicit cases for
negative weight, stop straddling (the straight-through gradient must match
torch's ``replace_grad``), mask on/off, and the zero-weight short-circuit
contract. No downloads; mlx-only (skipped off-darwin), torch runs on CPU.
"""

import importlib.util
import math

import pytest
import torch

from pytti.eval_tools import is_zero_weight, parametric_eval, set_t
from pytti.LossAug.EdgeLossClass import EdgeLoss
from pytti.LossAug.HSVLossClass import HSVLoss
from pytti.LossAug.MSELossClass import MSELoss
from pytti.LossAug.TVLossClass import TVLoss
from pytti.LossAug.TVLossClass import tv_loss as torch_tv_loss

needs_mlx = pytest.mark.skipif(
    importlib.util.find_spec("mlx") is None,
    reason="mlx not installed (darwin-only backend)",
)

CPU = torch.device("cpu")
ATOL = 1e-6  # the S2 gate: fp32 value + grad parity vs torch


def rand(*shape, seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.rand(shape, generator=g)


def to_mx(t):
    import mlx.core as mx

    return mx.array(t.detach().numpy())


def to_torch(a):
    import mlx.core as mx
    import numpy as np

    return torch.from_numpy(np.array(a.astype(mx.float32)))


def assert_close(mx_val, torch_val, what):
    diff = float((to_torch(mx_val) - torch_val.detach()).abs().max())
    assert diff <= ATOL, f"{what}: max abs diff {diff:.3e} > {ATOL}"


def torch_value_and_grad(fn, input):
    input = input.clone().requires_grad_(True)
    out = fn(input)
    out.sum().backward()
    return out.detach(), input.grad.detach()


def mx_value_and_grad(fn, input):
    import mlx.core as mx

    def summed(x):
        return mx.sum(fn(x))

    value = fn(to_mx(input))
    grad = mx.grad(summed)(to_mx(input))
    return value, grad


def check_parity(torch_fn, mx_fn, input, what):
    """Value + input-grad parity of scalar-or-vector losses, both <= 1e-6."""
    val_t, grad_t = torch_value_and_grad(torch_fn, input)
    val_m, grad_m = mx_value_and_grad(mx_fn, input)
    assert tuple(val_m.shape) == tuple(val_t.shape), (
        f"{what}: shape {tuple(val_m.shape)} != torch {tuple(val_t.shape)}"
    )
    assert_close(val_m, val_t, f"{what} value")
    assert_close(grad_m, grad_t, f"{what} input-grad")


# ---------------------------------------------------------------------------
# TV smoothing loss
# ---------------------------------------------------------------------------


@needs_mlx
def test_tv_loss_value_and_grad_parity():
    from pytti.mlx_engine.losses import tv_loss

    input = rand(2, 3, 16, 16, seed=1)
    check_parity(torch_tv_loss, tv_loss, input, "tv_loss")


@needs_mlx
def test_tv_loss_nonsquare_and_single():
    from pytti.mlx_engine.losses import tv_loss

    input = rand(1, 3, 8, 13, seed=2)  # asymmetric pad axes must not swap
    check_parity(torch_tv_loss, tv_loss, input, "tv_loss nonsquare")


# ---------------------------------------------------------------------------
# MSE loss, mask on/off
# ---------------------------------------------------------------------------


@needs_mlx
def test_mse_loss_parity_no_mask():
    from pytti.mlx_engine.losses import mse_loss

    input = rand(1, 3, 8, 8, seed=3)
    comp = rand(1, 3, 8, 8, seed=4)
    ref = MSELoss(comp, device=CPU)
    check_parity(
        lambda x: ref.get_loss(x, None),
        lambda x: mse_loss(x, to_mx(comp)),
        input,
        "mse_loss",
    )


@needs_mlx
def test_mse_loss_parity_with_mask():
    from pytti.mlx_engine.losses import mse_loss

    input = rand(1, 3, 8, 8, seed=5)
    comp = rand(1, 3, 8, 8, seed=6)
    mask = (rand(1, 1, 8, 8, seed=7) > 0.5).float()
    assert mask.eq(0).any() and mask.eq(1).any()
    ref = MSELoss(comp, device=CPU)
    ref.set_mask(mask)
    assert ref.use_mask
    check_parity(
        lambda x: ref.get_loss(x, None),
        lambda x: mse_loss(x, to_mx(comp), to_mx(mask)),
        input,
        "mse_loss masked",
    )
    # the mask must actually change the value (denominator stays full-size)
    unmasked = MSELoss(comp, device=CPU).get_loss(input, None)
    assert float((ref.get_loss(input, None) - unmasked).abs()) > 1e-4


@needs_mlx
def test_mse_loss_rejects_unresized_mask_and_mismatched_comp():
    """Torch resizes the mask lazily per shape (MSELossClass.py:134-138);
    that moved to setup — the MLX function must fail loud instead."""
    import mlx.core as mx

    from pytti.mlx_engine.losses import mse_loss

    input = mx.zeros((1, 3, 8, 8))
    with pytest.raises(ValueError, match="mask"):
        mse_loss(input, mx.zeros((1, 3, 8, 8)), mx.zeros((1, 1, 4, 4)))
    with pytest.raises(ValueError, match="comp"):
        mse_loss(input, mx.zeros((1, 3, 4, 4)))


@needs_mlx
def test_losses_reject_fp16_input():
    import mlx.core as mx

    from pytti.mlx_engine.losses import tv_loss

    with pytest.raises(ValueError, match="fp32"):
        tv_loss(mx.zeros((1, 3, 4, 4), dtype=mx.float16))


# ---------------------------------------------------------------------------
# HSV loss (rgb+s+v 5-channel MSE)
# ---------------------------------------------------------------------------


@needs_mlx
def test_rgbsv_conversion_parity():
    from pytti.mlx_engine.losses import rgb_to_rgbsv

    input = rand(2, 3, 8, 8, seed=8)
    out_t = HSVLoss.convert_input(input, None)
    out_m = rgb_to_rgbsv(to_mx(input))
    assert tuple(out_m.shape) == tuple(out_t.shape) == (2, 5, 8, 8)
    assert_close(out_m, out_t, "rgb_to_rgbsv value")


@needs_mlx
def test_hsv_loss_parity_mask_on_off():
    from pytti.mlx_engine.losses import hsv_loss

    input = rand(1, 3, 8, 8, seed=9)
    target = rand(1, 3, 8, 8, seed=10)
    # comp crosses exactly as the torch class stores it: already converted
    comp = HSVLoss.convert_input(target, None)
    ref = HSVLoss(comp, device=CPU)
    check_parity(
        lambda x: ref.get_loss(x, None),
        lambda x: hsv_loss(x, to_mx(comp)),
        input,
        "hsv_loss",
    )
    mask = (rand(1, 1, 8, 8, seed=11) > 0.5).float()
    ref.set_mask(mask)
    check_parity(
        lambda x: ref.get_loss(x, None),
        lambda x: hsv_loss(x, to_mx(comp), to_mx(mask)),
        input,
        "hsv_loss masked",
    )


# ---------------------------------------------------------------------------
# Edge loss (grayscale + Sobel MSE)
# ---------------------------------------------------------------------------


@needs_mlx
def test_edges_transform_parity():
    from pytti.mlx_engine.losses import edges

    input = rand(2, 3, 8, 8, seed=12)
    out_t = EdgeLoss.get_edges(input)
    out_m = edges(to_mx(input))
    assert tuple(out_m.shape) == tuple(out_t.shape) == (4, 1, 8, 8)
    assert_close(out_m, out_t, "edges value")


@needs_mlx
def test_edge_loss_parity_mask_on_off():
    from pytti.mlx_engine.losses import edge_loss

    input = rand(1, 3, 8, 8, seed=13)
    comp = EdgeLoss.convert_input(rand(1, 3, 8, 8, seed=14), None)
    ref = EdgeLoss(comp, device=CPU)
    check_parity(
        lambda x: ref.get_loss(x, None),
        lambda x: edge_loss(x, to_mx(comp)),
        input,
        "edge_loss",
    )
    mask = (rand(1, 1, 8, 8, seed=15) > 0.5).float()
    ref.set_mask(mask)
    check_parity(
        lambda x: ref.get_loss(x, None),
        lambda x: edge_loss(x, to_mx(comp), to_mx(mask)),
        input,
        "edge_loss masked",
    )


# ---------------------------------------------------------------------------
# Loss.forward wrapper: sign / stop straight-through / (loss, loss_raw)
# ---------------------------------------------------------------------------


def wrapper_case(weight, stop, seed_in=16, seed_comp=17):
    """One full-forward parity check through a torch MSELoss vs the mlx
    composition; returns the grads for gating assertions."""
    from pytti.mlx_engine.losses import loss_forward, mse_loss

    input = rand(1, 3, 8, 8, seed=seed_in)
    comp = rand(1, 3, 8, 8, seed=seed_comp)
    ref = MSELoss(comp, weight=weight, stop=stop, device=CPU)

    input_t = input.clone().requires_grad_(True)
    loss_t, raw_t = ref(input_t, None, device=CPU)
    loss_t.sum().backward()

    w = float(parametric_eval(weight))
    s = float(parametric_eval(stop))

    def mx_loss(x):
        return loss_forward(mse_loss(x, to_mx(comp)), w, s)[0]

    def mx_raw(x):
        return loss_forward(mse_loss(x, to_mx(comp)), w, s)[1]

    loss_m, grad_m = mx_value_and_grad(mx_loss, input)
    raw_m = mx_raw(to_mx(input))

    assert_close(loss_m, loss_t, f"wrapper loss (w={weight}, stop={stop})")
    assert_close(raw_m, raw_t, f"wrapper loss_raw (w={weight}, stop={stop})")
    assert_close(grad_m, input_t.grad, f"wrapper grad (w={weight}, stop={stop})")
    return input_t.grad.detach(), to_torch(grad_m)


@needs_mlx
def test_wrapper_positive_weight_no_stop():
    grad_t, _ = wrapper_case("1.5", "-inf")
    assert grad_t.abs().max() > 0


@needs_mlx
def test_wrapper_negative_weight_not_gated():
    # w=-2: loss = -raw ~ -0.17, stop=-inf -> gradient flows, scaled by -2
    grad_t, grad_m = wrapper_case("-2", "-inf")
    assert grad_t.abs().max() > 0
    assert grad_m.abs().max() > 0


@needs_mlx
def test_wrapper_negative_weight_stop_gates_gradient():
    # w=-2: loss = -raw ~ -0.17 < stop=-0.1 -> maximum picks the stop, the
    # straight-through kills the gradient while the VALUE still tracks loss
    grad_t, grad_m = wrapper_case("-2", "-0.1")
    assert grad_t.abs().max() == 0, "torch gradient must be gated"
    assert grad_m.abs().max() == 0, "mlx gradient must be gated"


@needs_mlx
def test_wrapper_stop_straddles_vector_loss():
    """TV loss is per-sample [n]; a stop between the two samples' values
    must gate exactly one — the straight-through grad must match torch
    elementwise (this is the replace_grad semantics gate)."""
    from pytti.mlx_engine.losses import loss_forward, tv_loss

    input = torch.cat(
        [rand(1, 3, 8, 8, seed=18), 0.05 * rand(1, 3, 8, 8, seed=19)]
    )
    raws = torch_tv_loss(input)
    stop = float(raws.min() + raws.max()) / 2
    assert raws.min() < stop < raws.max(), "fixture must straddle the stop"

    ref = TVLoss(weight="2", stop=stop)
    input_t = input.clone().requires_grad_(True)
    loss_t, raw_t = ref(input_t, None, device=CPU)
    loss_t.sum().backward()
    # the gated sample's image-grad must vanish, the other's must not
    per_sample = input_t.grad.abs().amax(dim=(1, 2, 3))
    gated = int(raws.argmin())
    assert per_sample[gated] == 0 and per_sample[1 - gated] > 0

    def mx_loss(x):
        return loss_forward(tv_loss(x), 2.0, stop)[0]

    loss_m, grad_m = mx_value_and_grad(mx_loss, input)
    assert_close(loss_m, loss_t, "straddle loss value")
    assert_close(grad_m, input_t.grad, "straddle straight-through grad")


@needs_mlx
def test_wrapper_weight_evaluating_to_zero_keeps_raw():
    """weight='t' at t=0 is NOT the short-circuit (is_zero_weight checks the
    raw config string): sign(0)=0 zeroes loss and grad, raw still computed."""
    assert not is_zero_weight("t")
    set_t(0.0)
    try:
        grad_t, grad_m = wrapper_case("t", "-inf")
        assert grad_t.abs().max() == 0
        assert grad_m.abs().max() == 0
        # raw parity (nonzero) is asserted inside wrapper_case
    finally:
        set_t(0.0)


@needs_mlx
def test_wrapper_accepts_mx_scalar_weight_and_stop():
    """S5 passes weight/stop as 0-dim mx inputs so per-step t never
    retraces — same math as python floats."""
    import mlx.core as mx

    from pytti.mlx_engine.losses import loss_forward

    raw = mx.array(0.25)
    for w, s in ((1.5, -math.inf), (-2.0, -0.1)):
        f_loss, f_raw = loss_forward(raw, w, s)
        a_loss, a_raw = loss_forward(raw, mx.array(w), mx.array(s))
        assert float(f_loss) == float(a_loss)
        assert float(f_raw) == float(a_raw)


@needs_mlx
def test_zero_weight_short_circuit_contract():
    """BaseLossClass.py:31-34: disabled or is_zero_weight(raw config weight)
    -> (0, 0) without evaluating the loss. Host-side decision in M2; the
    mlx pair comes from zero_loss()."""
    from pytti.mlx_engine.losses import zero_loss

    # the host-side predicate the caller must use
    for w in ("", "0", "0.0", 0, 0.0):
        assert is_zero_weight(w)

    comp = rand(1, 3, 8, 8, seed=20)
    ref = MSELoss(comp, weight="0", device=CPU)
    loss_t, raw_t = ref(rand(1, 3, 8, 8, seed=21), None, device=CPU)
    assert not loss_t.requires_grad  # torch never touched get_loss
    assert float(loss_t) == 0.0 and float(raw_t) == 0.0

    ref.weight = "1"
    ref.set_enabled(False)
    loss_t, _ = ref(rand(1, 3, 8, 8, seed=21), None, device=CPU)
    assert float(loss_t) == 0.0

    loss_m, raw_m = zero_loss()
    assert loss_m.ndim == 0 and raw_m.ndim == 0
    assert float(loss_m) == 0.0 and float(raw_m) == 0.0
