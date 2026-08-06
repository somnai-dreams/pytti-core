"""
MLX image models — M2 slice S1 (docs/mlx-m2-seam-map.md, Stage B + the
PixelImage image losses).

Functional ports of ``pytti.image_models.pixel.PixelImage`` and
``pytti.image_models.rgb_image.RGBImage``: plain functions over an explicit
params tree, no classes. The torch modules remain the source of truth for
init / save / restore; these functions are the hot-loop replacements the
whole-step compile (S5) assembles.

Params tree contract
--------------------
A tree is a flat ``dict[str, mx.array]``, every entry fp32. PixelImage::

    {
      "value":          [h, w]                        trainable
      "tensor":         [n_palettes, h, w]            trainable
      "palette":        [palette_size, n_palettes, 3] trainable
      "palette_target": [palette_size, n_palettes, 3] constant buffer
      # both present iff the torch module was built with hdr_weight != 0:
      "hdr_comp":       [palette_size, n_palettes]    constant buffer
      "hdr_weight":     []                            constant buffer
      "norm_weight":    []                            constant buffer
    }

RGBImage::

    {"tensor": [1, 3, h, w]}                          trainable

Trainable subsets are named by ``PIXEL_TRAINABLE_KEYS`` /
``RGB_TRAINABLE_KEYS`` in the package ``__init__``; sizes (palette_size,
n_palettes, h, w) are derived from the array shapes, never passed
separately. Static config — ``scale`` and ``use_palette_target`` — is a
keyword argument: a trace-time constant under ``mx.compile`` (it changes at
most between runs, so it can never silently freeze mid-render).

Decoded images are **NCHW** fp32 ``[1, 3, h*scale, w*scale]`` —
``("n", "s", "y", "x")``, matching the torch classes and slice S2's layout
contract.

Gradient semantics (each verified against torch autograd in
tests/test_mlx_engine_images.py):

- ``pixel_decode`` forward-returns the discrete image (one-hot palette
  pick, rounded value); the gradient is that of ``0.5*continuous +
  0.5*discrete`` (torch ``replace_grad``, pixel.py:352). ``value`` and
  ``tensor`` receive gradients only through the continuous path;
  ``palette`` through both.
- With ``use_palette_target=True`` the sorted target buffer replaces the
  live palette, so ``palette`` receives zero gradient — torch equivalently
  leaves ``palette.grad`` as ``None``.
- ``rgb_decode`` reproduces ``clamp_with_grad`` (tensor_tools.py:86-108):
  in-range pixels pass their gradient; out-of-range pixels pass it only
  when it points back toward the range. The gate depends on the incoming
  cotangent, so it needs a real custom VJP, not a stop-gradient recipe.

The per-step ``update()`` clamps are pure functions tree -> tree; the
assembly applies them AFTER the Adam update (ImageGuide.py:317-318 order).

Import/export: ``*_params_from_state_dict`` / ``*_state_dict_from_params``
convert torch ``state_dict`` <-> tree, both directions validated and
lossless, so ``.bak`` files stay torch-serialized and restore-compatible
(seam map §4).
"""

import functools

import mlx.core as mx
import numpy as np
import torch

from pytti.mlx_engine import PALETTE_INERTIA
from pytti.mlx_engine.losses import straight_through

# HSP brightness weights, https://alienryderflex.com/hsp.html — the exact
# constant PixelImage.sort_palette and HdrLoss use (pixel.py:136, :289)
_HSP_LUMA = (0.299, 0.587, 0.114)


# ---------------------------------------------------------------------------
# shared ops
# ---------------------------------------------------------------------------


def clip_inclusive(x: mx.array, low: float, high: float) -> mx.array:
    """
    ``torch.clamp`` for GRADIENT paths: same values as ``mx.clip``, but the
    subgradient at exactly ``low``/``high`` is 1 (torch is inclusive at both
    boundaries; ``mx.clip`` is exclusive). Load-bearing, not pedantry: the
    default init palette sits exactly ON [0, palette_inertia], and after
    every step the ``update()`` clamps park saturated params exactly on
    their boundaries — with exclusive clipping those entries would stop
    receiving gradients and trajectories diverge from torch at step 2
    (found by the S5 full-step parity gate; regression-tested in
    tests/test_mlx_engine_step.py).
    """
    return mx.where(x < low, low, mx.where(x > high, high, x))


def nearest_upsample(x: mx.array, scale: int) -> mx.array:
    """
    Nearest-neighbor upsample of an NCHW batch by an integer factor —
    exactly ``F.interpolate(mode="nearest")`` for integer scales (index map
    ``floor(i/scale)`` degenerates to block repeat); the gradient sums over
    each ``scale x scale`` block, matching torch's backward.
    """
    if not isinstance(scale, int) or scale < 1:
        raise ValueError(f"scale must be a positive int, got {scale!r}")
    if x.ndim != 4:
        raise ValueError(f"expected an NCHW batch, got shape {tuple(x.shape)}")
    if scale == 1:
        return x
    n, c, h, w = x.shape
    out = mx.broadcast_to(x[:, :, :, None, :, None], (n, c, h, scale, w, scale))
    return out.reshape(n, c, h * scale, w * scale)


@functools.cache
def _clamp_with_grad(low: float, high: float):
    @mx.custom_function
    def clamped(x: mx.array) -> mx.array:
        return mx.clip(x, low, high)

    @clamped.vjp
    def clamped_vjp(primal, cotangent, output):
        # torch ClampWithGrad.backward (tensor_tools.py:98-105): in-range
        # elements pass the gradient; out-of-range elements pass it only
        # when cotangent and overshoot share a sign (the descent step would
        # pull the value back toward the range).
        overshoot = primal - mx.clip(primal, low, high)
        gate = (cotangent * overshoot) >= 0
        return cotangent * gate.astype(cotangent.dtype)

    return clamped


def clamp_with_grad(x: mx.array, low: float, high: float) -> mx.array:
    """torch ``clamp_with_grad``: clamp forward, cotangent-gated backward."""
    return _clamp_with_grad(float(low), float(high))(x)


# ---------------------------------------------------------------------------
# tree validation (fail loud at trace time; shape checks are host-side)
# ---------------------------------------------------------------------------


def _validate_pixel_tree(params: dict) -> None:
    keys = set(params)
    required = {"value", "tensor", "palette", "palette_target", "norm_weight"}
    hdr = {"hdr_comp", "hdr_weight"}
    if keys != required and keys != required | hdr:
        raise ValueError(
            "PixelImage params tree keys off-contract: "
            f"missing {sorted(required - keys)}, "
            f"unknown {sorted(keys - required - hdr)}, "
            f"hdr entries present {sorted(keys & hdr)} (must be both or neither)"
        )
    for name, arr in params.items():
        if arr.dtype != mx.float32:
            raise ValueError(f"params[{name!r}] must be fp32, got {arr.dtype}")
    palette = params["palette"]
    if palette.ndim != 3 or palette.shape[-1] != 3:
        raise ValueError(
            f"palette must be [palette_size, n_palettes, 3], got {palette.shape}"
        )
    if params["palette_target"].shape != palette.shape:
        raise ValueError(
            f"palette_target shape {params['palette_target'].shape} != "
            f"palette shape {palette.shape}"
        )
    tensor = params["tensor"]
    if tensor.ndim != 3 or tensor.shape[0] != palette.shape[1]:
        raise ValueError(
            f"tensor must be [n_palettes={palette.shape[1]}, h, w], "
            f"got {tensor.shape}"
        )
    if params["value"].shape != tensor.shape[1:]:
        raise ValueError(
            f"value shape {params['value'].shape} != tensor spatial shape "
            f"{tensor.shape[1:]}"
        )
    if params["norm_weight"].shape != ():
        raise ValueError(f"norm_weight must be 0-dim, got {params['norm_weight'].shape}")
    if "hdr_comp" in params:
        if params["hdr_comp"].shape != palette.shape[:2]:
            raise ValueError(
                f"hdr_comp must be [palette_size, n_palettes] = "
                f"{palette.shape[:2]}, got {params['hdr_comp'].shape}"
            )
        if params["hdr_weight"].shape != ():
            raise ValueError(
                f"hdr_weight must be 0-dim, got {params['hdr_weight'].shape}"
            )


def _validate_rgb_tree(params: dict) -> None:
    if set(params) != {"tensor"}:
        raise ValueError(
            f"RGBImage params tree keys off-contract: got {sorted(params)}, "
            "expected exactly ['tensor']"
        )
    tensor = params["tensor"]
    if tensor.dtype != mx.float32:
        raise ValueError(f"params['tensor'] must be fp32, got {tensor.dtype}")
    if tensor.ndim != 4 or tensor.shape[0] != 1 or tensor.shape[1] != 3:
        raise ValueError(f"tensor must be [1, 3, h, w], got {tensor.shape}")


# ---------------------------------------------------------------------------
# PixelImage (Limited Palette)
# ---------------------------------------------------------------------------


def palette_sort_indices(palette: mx.array) -> mx.array:
    """
    HSP-brightness order of the raw palette: ``[palette_size, n_palettes]``
    integer indices — torch's ``color_norms.argsort(dim=0)`` (pixel.py:
    287-291) in its pre-transpose form.
    """
    p = mx.clip(palette / PALETTE_INERTIA, 0.0, 1.0)
    color_norms = mx.sum(mx.square(p) * mx.array(_HSP_LUMA), axis=-1)
    return mx.argsort(color_norms, axis=0)


def sort_palette(params: dict, *, use_palette_target: bool) -> mx.array:
    """
    ``PixelImage.sort_palette`` (pixel.py:280-295): the live palette scaled
    to [0, 1] and brightness-sorted per palette column — or the (already
    sorted) target buffer when the palette is locked.
    """
    if use_palette_target:
        return params["palette_target"]
    p = clip_inclusive(params["palette"] / PALETTE_INERTIA, 0.0, 1.0)
    indices = palette_sort_indices(params["palette"])
    # result[k, j, :] = p[indices[k, j], j, :] — identical to torch's
    # per-column stack/gather dance
    return mx.take_along_axis(p, indices[..., None], axis=0)


def pixel_decode(params: dict, *, scale: int, use_palette_target: bool) -> mx.array:
    """
    ``PixelImage.decode_tensor`` (pixel.py:310-352) -> ``[1, 3, h*scale,
    w*scale]``. Forward value is the discrete image; gradient is that of
    the 0.5*continuous + 0.5*discrete mix (straight-through).
    """
    _validate_pixel_tree(params)
    value = params["value"]  # [h, w]
    tensor = params["tensor"]  # [n_palettes, h, w]
    palette = sort_palette(params, use_palette_target=use_palette_target)
    palette_size, n_palettes = palette.shape[0], palette.shape[1]

    # brightness values of pixels (torch .clamp: boundary-inclusive grads —
    # update() parks saturated values exactly at 0/1 every step)
    values = clip_inclusive(value, 0.0, 1.0) * (palette_size - 1)
    floors = mx.floor(values)
    value_fracs = (values - floors)[..., None, None]  # [h, w, 1, 1]
    value_floors = floors.astype(mx.int32)
    value_ceils = mx.ceil(values).astype(mx.int32)
    value_rounds = mx.round(values).astype(mx.int32)  # half-to-even, as torch

    palette_weights = mx.transpose(tensor, (1, 2, 0))  # [h, w, n_palettes]
    one_hot = (
        mx.arange(n_palettes) == mx.argmax(palette_weights, axis=2)[..., None]
    ).astype(palette.dtype)[..., None]  # [h, w, n_palettes, 1]
    soft = mx.softmax(palette_weights, axis=2)[..., None]

    # palette[<int map>] gathers [h, w, n_palettes, 3]
    colors_disc = mx.sum(palette[value_rounds] * one_hot, axis=2)  # [h, w, 3]
    colors_cont = mx.sum(
        (palette[value_floors] * (1 - value_fracs) + palette[value_ceils] * value_fracs)
        * soft,
        axis=2,
    )
    disc = nearest_upsample(mx.transpose(colors_disc, (2, 0, 1))[None], scale)
    cont = nearest_upsample(mx.transpose(colors_cont, (2, 0, 1))[None], scale)
    return straight_through(disc, cont * 0.5 + disc * 0.5)


def pixel_update(params: dict) -> dict:
    """
    ``PixelImage.update`` (pixel.py:408-417) as a pure function: the
    per-step clamps, applied AFTER the Adam update (ImageGuide.py:317-318
    order). Constant buffers pass through untouched.
    """
    _validate_pixel_tree(params)
    out = dict(params)
    out["palette"] = mx.clip(params["palette"], 0.0, PALETTE_INERTIA)
    out["value"] = mx.clip(params["value"], 0.0, 1.0)
    out["tensor"] = mx.clip(params["tensor"], 0.0, None)
    return out


def hdr_loss(params: dict, *, use_palette_target: bool) -> tuple[mx.array, mx.array]:
    """
    ``HdrLoss.forward`` (pixel.py:127-143): sorted-palette brightness norms
    pulled toward the gamma ramp ``comp``. Returns ``(loss, loss_raw)``,
    both 0-dim. Fails loud when the tree carries no HDR buffers (torch
    builds no HdrLoss module when hdr_weight == 0 — there is nothing to
    evaluate, and image_loss() never routes here).
    """
    _validate_pixel_tree(params)
    if "hdr_comp" not in params:
        raise ValueError(
            "params tree has no 'hdr_comp'/'hdr_weight' buffers: the torch "
            "module was built with hdr_weight=0, so there is no HDR loss to "
            "evaluate"
        )
    palette = sort_palette(params, use_palette_target=use_palette_target)
    magic_root = mx.sqrt(mx.array(_HSP_LUMA))
    # torch.linalg.vector_norm defines the ZERO-vector subgradient as 0;
    # a bare mx.sqrt has a NaN gradient at 0 — and the default init palette
    # (linspace from 0) always contains an exact-zero row. The mx.where
    # pair reproduces torch: value sqrt(x), gradient 0 where x == 0.
    sq = mx.sum(mx.square(palette * magic_root), axis=-1)
    color_norms = mx.where(sq > 0, mx.sqrt(mx.where(sq > 0, sq, 1.0)), 0.0)
    loss_raw = mx.mean(mx.square(color_norms - params["hdr_comp"]))
    return loss_raw * params["hdr_weight"], loss_raw


def palette_loss(params: dict) -> tuple[mx.array, mx.array]:
    """
    ``PaletteLoss.forward`` (pixel.py:49-76), the palette normalizer:
    anticorrelate the per-pixel palette softmax across palettes and reward
    within-palette variance. Returns ``(loss, loss_raw)``, both 0-dim.
    """
    _validate_pixel_tree(params)
    tensor = params["tensor"]  # [n_palettes, h, w]
    n_palettes = tensor.shape[0]
    if n_palettes == 1:
        # Cross-palette decorrelation is undefined for a single palette:
        # softmax over a size-1 axis is identically 1.0, so sigma == 0
        # exactly and both terms below are 0/0 and 1/0. Eager mlx yields
        # NaN; the compiled step's fused variance reduction can instead
        # accumulate rounding residue at larger N, manufacturing a finite
        # garbage constant with an exactly-zero gradient. Either way the
        # loss is meaningless at n=1 -> graph-connected exact zero
        # (mirrors torch PaletteLoss.forward; shapes are static under
        # mx.compile, so this python branch resolves at trace time).
        loss_raw = mx.sum(tensor) * 0.0
        return loss_raw * params["norm_weight"], loss_raw
    t = mx.softmax(
        mx.transpose(tensor, (1, 2, 0)).reshape(-1, n_palettes), axis=-1
    )  # [N, n]
    big_n = t.shape[0]
    mu = mx.mean(t, axis=0, keepdims=True)
    # sqrt(var + 1e-16): guards a mid-run degenerate state (all logits
    # equal -> variance exactly 0) from NaN. Unlike mx.maximum(std, eps),
    # whose vjp still propagates NaN through std at zero variance, this
    # keeps the gradient finite; it is bit-identical in fp32 whenever
    # var > ~1e-9 (healthy sigma ~ 0.1) and matches the torch guard's
    # sigma == 1e-8 at the degenerate point. torch .std()/.var() and
    # ddof=1 are both unbiased.
    sigma = mx.sqrt(mx.var(t, axis=0, keepdims=True, ddof=1) + 1e-16)
    centered = t - mu
    s = (centered.T @ centered) / (sigma * sigma.T * big_n)
    s = s - mx.diag(mx.diagonal(s))
    loss_raw = mx.mean(s) + mx.mean(1.0 / (sigma * big_n))
    return loss_raw * params["norm_weight"], loss_raw


# ---------------------------------------------------------------------------
# RGBImage (Unlimited Palette)
# ---------------------------------------------------------------------------


def rgb_decode(params: dict, *, scale: int) -> mx.array:
    """``RGBImage.decode_tensor`` (rgb_image.py:30-33) -> [1, 3, h*s, w*s]."""
    _validate_rgb_tree(params)
    return clamp_with_grad(nearest_upsample(params["tensor"], scale), 0.0, 1.0)


def rgb_update(params: dict) -> dict:
    """RGBImage defines no ``update()`` clamps — identity, kept so the
    assembly applies one uniform step->clamp interface per image model."""
    _validate_rgb_tree(params)
    return dict(params)


# ---------------------------------------------------------------------------
# torch state_dict <-> params tree (both directions lossless, fail loud)
# ---------------------------------------------------------------------------

# state_dict name -> tree name
_PIXEL_KEY_MAP = {
    "value": "value",
    "tensor": "tensor",
    "palette": "palette",
    "palette_target": "palette_target",
    "loss.weight": "norm_weight",
}
_PIXEL_HDR_KEY_MAP = {
    "hdr_loss.comp": "hdr_comp",
    "hdr_loss.weight": "hdr_weight",
}


def _to_mx(name: str, tensor: torch.Tensor) -> mx.array:
    if not isinstance(tensor, torch.Tensor):
        raise ValueError(
            f"state_dict[{name!r}] is not a tensor: {type(tensor).__name__}"
        )
    if tensor.dtype != torch.float32:
        raise ValueError(f"state_dict[{name!r}] must be fp32, got {tensor.dtype}")
    return mx.array(tensor.detach().cpu().contiguous().numpy())


def _to_torch(name: str, arr: mx.array) -> torch.Tensor:
    if arr.dtype != mx.float32:
        raise ValueError(f"params[{name!r}] must be fp32, got {arr.dtype}")
    return torch.from_numpy(np.array(arr))


def pixel_params_from_state_dict(state_dict: dict) -> dict:
    """``PixelImage.state_dict()`` -> params tree (fp32 mx arrays)."""
    keys = set(state_dict)
    required = set(_PIXEL_KEY_MAP)
    hdr = set(_PIXEL_HDR_KEY_MAP)
    if keys != required and keys != required | hdr:
        raise ValueError(
            "PixelImage state_dict keys off-contract: "
            f"missing {sorted(required - keys)}, "
            f"unknown {sorted(keys - required - hdr)}, "
            f"hdr entries present {sorted(keys & hdr)} (must be both or neither)"
        )
    mapping = dict(_PIXEL_KEY_MAP)
    if hdr <= keys:
        mapping.update(_PIXEL_HDR_KEY_MAP)
    params = {
        tree_key: _to_mx(sd_key, state_dict[sd_key])
        for sd_key, tree_key in mapping.items()
    }
    _validate_pixel_tree(params)
    return params


def pixel_state_dict_from_params(params: dict) -> dict:
    """Params tree -> torch CPU fp32 tensors keyed for
    ``PixelImage.load_state_dict`` (strict) — the ``.bak`` round trip."""
    _validate_pixel_tree(params)
    mapping = dict(_PIXEL_KEY_MAP)
    if "hdr_comp" in params:
        mapping.update(_PIXEL_HDR_KEY_MAP)
    return {
        sd_key: _to_torch(tree_key, params[tree_key])
        for sd_key, tree_key in mapping.items()
    }


def rgb_params_from_state_dict(state_dict: dict) -> dict:
    """``RGBImage.state_dict()`` -> params tree (fp32 mx arrays)."""
    if set(state_dict) != {"tensor"}:
        raise ValueError(
            "RGBImage state_dict keys off-contract: "
            f"got {sorted(state_dict)}, expected exactly ['tensor']"
        )
    params = {"tensor": _to_mx("tensor", state_dict["tensor"])}
    _validate_rgb_tree(params)
    return params


def rgb_state_dict_from_params(params: dict) -> dict:
    """Params tree -> torch CPU fp32 tensors for ``RGBImage.load_state_dict``."""
    _validate_rgb_tree(params)
    return {"tensor": _to_torch("tensor", params["tensor"])}
