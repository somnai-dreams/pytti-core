"""
MLX direct (image-space) losses — M2 slice S2 (docs/mlx-m2-seam-map.md, §C).

Pure-function ports of the torch loss classes the still-image path uses:

- ``tv_loss``      <- LossAug/TVLossClass.tv_loss (L2 total variation,
  attached whenever ``smoothing_weight`` != 0, default 0.02)
- ``mse_loss``     <- LossAug/MSELossClass.MSELoss.get_loss (optional
  spatial mask, ``F.mse_loss(input*mask, comp*mask)`` semantics — the mask
  scales BOTH sides and the mean keeps the full denominator)
- ``rgb_to_rgbsv`` / ``hsv_loss`` <- LossAug/HSVLossClass.HSVLoss — the
  still models' ``get_preferred_loss`` (differentiable_image.py:104-107),
  so direct init weight, direct image prompts, and direct stabilization all
  route through it. Only S and V survive HSVLoss's ``hsv[:, 1:]`` slice, so
  kornia's hue/argmax cascade is not ported.
- ``edges`` / ``edge_loss`` <- LossAug/EdgeLossClass.EdgeLoss
  (``edge_stabilization_weight`` with an init image)
- ``loss_forward`` <- LossAug/BaseLossClass.Loss.forward (:30-41): weight
  sign handling, stop threshold via the straight-through ``replace_grad``,
  and the ``(loss, loss_raw)`` pair contract.

Layout & precision
------------------
All image tensors are **NCHW** fp32 — ``("n", "s", "y", "x")``, the still
path's axis order end to end (tensor_tools.named_rearrange is an identity
there). The NHWC transpose lives only inside ``edges``'s convolutions.
Non-fp32 inputs are rejected: the M2 precision ruling keeps image params
and direct losses fp32 (the fp16 cast happens at tower entry).

Constants
---------
``comp`` buffers and spatial masks are produced torch-side at setup exactly
as today — including the HSV/edge conversion of the comp (``make_comp``
stores it CONVERTED) and the once-per-shape mask resize
(MSELossClass.py:134-138) — and cross the boundary once as fp32 constants.
These functions therefore require comp/mask shapes to already match the
input; there is no lazy resize here (fail loud).

Host/trace split (Loss.forward semantics)
-----------------------------------------
The ``enabled`` flag and the ``is_zero_weight`` short-circuit
(BaseLossClass.py:31-34) inspect the RAW config weight ("", "0", 0) — host
decisions. Callers skip the loss functions entirely in that case and use
``zero_loss()`` for the record pair. ``parametric_eval`` of weight/stop
stays host-side (seam map: per-step host->device inputs); pass the results
to ``loss_forward`` as python floats (eager/tests) or as 0-dim mx arrays
(compiled-step inputs, so a new ``t`` never retraces). A weight that merely
EVALUATES to 0.0 (e.g. "t" at t=0) does not short-circuit in torch and does
not here: sign(0) = 0 zeroes the loss, but ``loss_raw`` is still computed
and recorded.

Tie-breaking: ``mx.max``/``mx.min`` in ``rgb_to_rgbsv`` may route gradients
differently from torch's channel argmax at EXACTLY tied channel values
(gray pixels); values are unaffected and ties are measure-zero on rendered
images.
"""

import mlx.core as mx

# torchvision rgb_to_grayscale luma weights (_functional_tensor.py:160)
_LUMA = (0.2989, 0.587, 0.114)
# kornia rgb_to_hsv default eps (kornia/color/hsv.py)
_HSV_EPS = 1e-8


def _require_image(x: mx.array, name: str, channels: int | None = None) -> None:
    """Fail loud at trace time on layout/precision contract violations."""
    if x.ndim != 4:
        raise ValueError(
            f"{name} must be NCHW [n, s, y, x], got shape {tuple(x.shape)}"
        )
    if x.dtype != mx.float32:
        raise ValueError(
            f"{name} must be fp32 (M2 precision ruling: direct losses run "
            f"fp32; the fp16 cast happens at tower entry), got {x.dtype}"
        )
    if channels is not None and x.shape[1] != channels:
        raise ValueError(
            f"{name} must have {channels} channels, got shape {tuple(x.shape)}"
        )


# ---------------------------------------------------------------------------
# Loss.forward wrapper semantics (BaseLossClass.py:30-41)
# ---------------------------------------------------------------------------


def straight_through(fwd: mx.array, bwd: mx.array) -> mx.array:
    """torch ``replace_grad(fwd, bwd)``: value of ``fwd``, gradient of
    ``bwd`` (plan-confirmed recipe; same as bridge.py:176-181)."""
    return bwd + mx.stop_gradient(fwd - bwd)


def loss_forward(
    loss_raw: mx.array,
    weight: "float | mx.array",
    stop: "float | mx.array",
) -> tuple[mx.array, mx.array]:
    """
    ``Loss.forward`` minus the host-side short-circuit: takes the already
    parametric-evaluated weight/stop scalars and the raw loss from one of
    the ``*_loss`` functions, returns ``(loss, loss_raw)``.

    Mirrors BaseLossClass.py:37-41: ``loss = loss_raw * sign(weight)``;
    the returned loss has the VALUE ``abs(weight) * loss`` but the GRADIENT
    of ``abs(weight) * maximum(loss, stop)`` — once ``loss`` drops below
    ``stop`` its gradient is gated off while the reported value keeps
    tracking the real loss. ``loss_raw`` passes through unweighted (it is
    what train() records).
    """
    loss = loss_raw * mx.sign(weight)
    gated = straight_through(loss, mx.maximum(loss, stop))
    return mx.abs(weight) * gated, loss_raw


def zero_loss() -> tuple[mx.array, mx.array]:
    """
    The short-circuit pair (BaseLossClass.py:31-34): callers must emit this
    — without calling the loss function — when the loss is disabled or
    ``is_zero_weight(raw_config_weight)`` holds. Both elements are 0-dim so
    the record/accumulate contract is unchanged.
    """
    zero = mx.zeros(())
    return zero, zero


# ---------------------------------------------------------------------------
# TV smoothing loss (TVLossClass.py:8-13)
# ---------------------------------------------------------------------------


def tv_loss(input: mx.array) -> mx.array:
    """L2 total variation loss, as in Mahendran et al. — [n,s,y,x] -> [n]."""
    _require_image(input, "input")
    padded = mx.pad(input, ((0, 0), (0, 0), (0, 1), (0, 1)), mode="edge")
    x_diff = padded[..., :-1, 1:] - padded[..., :-1, :-1]
    y_diff = padded[..., 1:, :-1] - padded[..., :-1, :-1]
    return mx.mean(x_diff**2 + y_diff**2, axis=(1, 2, 3))


# ---------------------------------------------------------------------------
# MSE loss with optional spatial mask (MSELossClass.py:131-141)
# ---------------------------------------------------------------------------


def mse_loss(
    input: mx.array, comp: mx.array, mask: mx.array | None = None
) -> mx.array:
    """
    ``MSELoss.get_loss`` on an already-converted input: mean over ALL
    elements of ``(input*mask - comp*mask)^2`` (mask defaults to 1) -> 0-dim.

    ``comp`` is the setup-time constant exactly as the torch class stores
    it (already ``convert_input``-ed for the HSV/edge subclasses). ``mask``
    is ``[1, 1, y, x]`` at the INPUT's resolution — the torch class's lazy
    once-per-shape resize happened at setup.
    """
    _require_image(input, "input")
    _require_image(comp, "comp")
    if tuple(comp.shape) != tuple(input.shape):
        raise ValueError(
            f"comp shape {tuple(comp.shape)} != input shape "
            f"{tuple(input.shape)} — comp constants cross the boundary "
            "pre-sized at setup; there is no lazy resize in the MLX step"
        )
    if mask is not None:
        _require_image(mask, "mask")
        if tuple(mask.shape[-2:]) != tuple(input.shape[-2:]):
            raise ValueError(
                f"mask spatial shape {tuple(mask.shape[-2:])} != input "
                f"{tuple(input.shape[-2:])} — the once-per-shape mask "
                "resize (MSELossClass.py:134-138) happens torch-side at "
                "setup; resize before crossing"
            )
        input = input * mask
        comp = comp * mask
    return mx.mean(mx.square(input - comp))


# ---------------------------------------------------------------------------
# HSV loss (HSVLossClass.py:7-13) — the still models' preferred direct loss
# ---------------------------------------------------------------------------


def rgb_to_rgbsv(input: mx.array) -> mx.array:
    """
    ``HSVLoss.convert_input``: [n,3,y,x] -> [n,5,y,x] = [r,g,b,s,v].

    Only saturation and value survive HSVLoss's ``hsv[:, 1:]`` slice, so
    just ``v = max(r,g,b)`` and ``s = (v - min) / (v + eps)`` with kornia's
    eps = 1e-8 (kornia/color/hsv.py:61-62) — the hue cascade is not needed.
    """
    _require_image(input, "input", channels=3)
    v = mx.max(input, axis=1, keepdims=True)
    lo = mx.min(input, axis=1, keepdims=True)
    s = (v - lo) / (v + _HSV_EPS)
    return mx.concatenate([input, s, v], axis=1)


def hsv_loss(
    input: mx.array, comp: mx.array, mask: mx.array | None = None
) -> mx.array:
    """``HSVLoss.get_loss``: convert the rgb input, MSE against the
    (already 5-channel) comp constant. -> 0-dim."""
    return mse_loss(rgb_to_rgbsv(input), comp, mask)


# ---------------------------------------------------------------------------
# Edge loss (EdgeLossClass.py:9-33) — edge_stabilization_weight
# ---------------------------------------------------------------------------


def edges(input: mx.array) -> mx.array:
    """
    ``EdgeLoss.get_edges``: grayscale luma dot, two 3x3 Sobel/8 convs with
    zero-padded "same", stacked along the batch axis.
    [n,3,y,x] -> [2n,1,y,x] (f_x batch first, f_y second — torch cat order).
    """
    _require_image(input, "input", channels=3)
    gray = (
        _LUMA[0] * input[:, 0]
        + _LUMA[1] * input[:, 1]
        + _LUMA[2] * input[:, 2]
    )  # [n, y, x]
    nhwc = gray[..., None]  # [n, y, x, 1]
    dx_ker = (
        mx.array(
            [[1.0, 0.0, -1.0], [2.0, 0.0, -2.0], [1.0, 0.0, -1.0]],
            dtype=mx.float32,
        ).reshape(1, 3, 3, 1)
        / 8.0
    )
    dy_ker = (
        mx.array(
            [[1.0, 2.0, 1.0], [0.0, 0.0, 0.0], [-1.0, -2.0, -1.0]],
            dtype=mx.float32,
        ).reshape(1, 3, 3, 1)
        / 8.0
    )
    f_x = mx.conv2d(nhwc, dx_ker, padding=1)
    f_y = mx.conv2d(nhwc, dy_ker, padding=1)
    out = mx.concatenate([f_x, f_y], axis=0)  # [2n, y, x, 1]
    return mx.transpose(out, (0, 3, 1, 2))


def edge_loss(
    input: mx.array, comp: mx.array, mask: mx.array | None = None
) -> mx.array:
    """``EdgeLoss.get_loss``: edge-transform the rgb input, MSE against the
    (already edge-transformed, [2n,1,y,x]) comp constant. -> 0-dim."""
    return mse_loss(edges(input), comp, mask)
