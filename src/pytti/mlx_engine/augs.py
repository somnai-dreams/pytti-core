"""
MLX port of the batched aug stack — M2 slice S4 (docs/mlx-m2-seam-map.md,
Stage D rows augs.py:57-189).

Functional port of ``pytti.Perceptor.cutouts.augs.BatchedAugs``: the same
composed inverse projective warp (flip / affine / perspective matrices,
pixel-center base grid in the ``align_corners=False`` convention), the same
per-sample 3x3 color matrices (saturation lerp toward luma + hue rotation
around the RGB gray axis via Rodrigues' formula), the same arange-mask
rectangle erasing, and the same per-op probability gating.

Two seams, per the slice spec:

- **Test seam** — every random quantity the torch module draws is a field
  of :class:`AugParams`, so :func:`apply_augs` is a pure deterministic
  function of ``(x, params, config)``. Parity tests re-seed the torch
  stream, replay the reference's exact draw sequence, and inject the
  recovered values here.
- **Compile seam** — :func:`draw_aug_params` produces the same bundle from
  ``mx.random``. Pass an explicit ``key`` (split into one subkey per draw)
  for fully deterministic threading, or ``key=None`` to consume the
  implicit global stream — in that case the whole-step ``mx.compile`` must
  list ``mx.random.state`` in its ``inputs=``/``outputs=`` (seam map §3),
  otherwise the draws freeze into the trace.

Precision (lead ruling): everything here is fp32 — inputs, params, and
outputs. The fp16 cast happens at tower entry (mlx_backend/vit.py), never
here. Non-fp32 inputs are rejected, not cast.

``mlx`` is imported at module level: import this module only lazily (via
the package ``__getattr__`` or inside function bodies), matching the
``mlx_backend`` pattern so linux CI never touches mlx.
"""

import math
from dataclasses import dataclass
from typing import NamedTuple

import mlx.core as mx


@dataclass(frozen=True)
class AugConfig:
    """Static aug-stack configuration. Field names, defaults, and semantics
    are exactly ``BatchedAugs.__init__`` (Perceptor/cutouts/augs.py:40-55).
    """

    p_flip: float = 0.3
    degrees: float = 30.0
    translate: float = 0.1
    p_affine: float = 0.8
    distortion: float = 0.2
    p_persp: float = 0.4
    hue: float = 0.01
    sat: float = 0.01
    p_jitter: float = 0.7
    erase_scale: tuple[float, float] = (0.1, 0.4)
    erase_ratio: tuple[float, float] = (0.3, 1 / 0.3)
    p_erase: float = 0.7

    def __post_init__(self):
        for name in ("p_flip", "p_affine", "p_persp", "p_jitter", "p_erase"):
            p = getattr(self, name)
            if not 0.0 <= p <= 1.0:
                raise ValueError(f"{name} must be in [0, 1], got {p}")
        for name in ("degrees", "translate", "distortion", "hue", "sat"):
            v = getattr(self, name)
            if v < 0:
                raise ValueError(f"{name} must be >= 0, got {v}")
        for name in ("erase_scale", "erase_ratio"):
            lo, hi = getattr(self, name)
            if not 0 < lo <= hi:
                raise ValueError(
                    f"{name} must satisfy 0 < lo <= hi, got ({lo}, {hi})"
                )


class AugParams(NamedTuple):
    """One step's random draws, all shape ``[n]`` fp32 — the injectable
    seam. Fields mirror the torch module's draw sequence one-to-one and in
    order (this order is the RNG contract of :func:`draw_aug_params`).

    ``*_u`` fields are raw U[0,1) draws that :func:`apply_augs` scales into
    the torch ranges; ``log_r`` is drawn directly in
    ``[log(erase_ratio[0]), log(erase_ratio[1])]`` (torch's ``uniform_``).

    ============== ============================== =========================
    field          torch draw (augs.py line)      derived range
    ============== ============================== =========================
    flip_u         rand [n,1]        (:64)        flips iff u < p_flip
    theta_u        rand [n]          (:72)        theta in ±degrees·π/180
    tx_u, ty_u     rand [n] ×2       (:75-76)     ±2·translate
    affine_gate_u  rand [n,1,1]      (:88)        applies iff u < p_affine
    px_u, py_u     rand [n] ×2       (:94-95)     ±distortion/2
    persp_gate_u   rand [n,1,1]      (:104)       applies iff u < p_persp
    sat_u          rand [n,1,1]      (:118)       s in 1 ± sat
    hue_u          rand [n]          (:123)       angle in ±hue·2π
    jitter_gate_u  rand [n,1,1]      (:139)       applies iff u < p_jitter
    area_u         rand [n]          (:167)       area frac in erase_scale
    log_r          uniform_ [n]      (:170)       log of aspect ratio
    y0_u, x0_u     rand [n] ×2       (:176-177)   rect origin in-frame
    erase_gate_u   rand [n,1,1]      (:187)       erases iff u < p_erase
    ============== ============================== =========================
    """

    flip_u: mx.array
    theta_u: mx.array
    tx_u: mx.array
    ty_u: mx.array
    affine_gate_u: mx.array
    px_u: mx.array
    py_u: mx.array
    persp_gate_u: mx.array
    sat_u: mx.array
    hue_u: mx.array
    jitter_gate_u: mx.array
    area_u: mx.array
    log_r: mx.array
    y0_u: mx.array
    x0_u: mx.array
    erase_gate_u: mx.array


def draw_aug_params(
    n: int, config: AugConfig, key: mx.array | None = None
) -> AugParams:
    """Draw one step's aug parameters from ``mx.random``.

    RNG contract (the compile seam):

    - ``key=None`` — 16 draws of shape ``[n]`` are consumed from the
      implicit global stream (``mx.random.state``) in ``AugParams`` field
      order. Under whole-step ``mx.compile`` the caller must pass
      ``inputs=[mx.random.state], outputs=[mx.random.state]``.
    - explicit ``key`` — ``mx.random.split(key, 16)``, one subkey per field
      in order; the global stream is untouched.

    Distributions match the torch ranges: 15 raw U[0,1) draws plus
    ``log_r`` ~ U[log(erase_ratio[0]), log(erase_ratio[1])].
    """
    if n <= 0:
        raise ValueError(f"n must be positive, got {n}")
    keys: list[mx.array | None]
    if key is None:
        keys = [None] * len(AugParams._fields)
    else:
        keys = list(mx.random.split(key, len(AugParams._fields)))

    def u(i: int, low: float = 0.0, high: float = 1.0) -> mx.array:
        return mx.random.uniform(low=low, high=high, shape=(n,), key=keys[i])

    log_lo = math.log(config.erase_ratio[0])
    log_hi = math.log(config.erase_ratio[1])
    return AugParams(
        flip_u=u(0),
        theta_u=u(1),
        tx_u=u(2),
        ty_u=u(3),
        affine_gate_u=u(4),
        px_u=u(5),
        py_u=u(6),
        persp_gate_u=u(7),
        sat_u=u(8),
        hue_u=u(9),
        jitter_gate_u=u(10),
        area_u=u(11),
        log_r=u(12, low=log_lo, high=log_hi),
        y0_u=u(13),
        x0_u=u(14),
        erase_gate_u=u(15),
    )


def warp_matrices(params: AugParams, config: AugConfig) -> mx.array:
    """Inverse (output->input) projective matrices ``[n, 3, 3]`` in
    normalized [-1, 1] coordinates — port of ``BatchedAugs._warp_matrices``
    (augs.py:57-109), gates included.
    """
    eye = mx.eye(3)

    # horizontal flip: its own inverse
    flip_vec = mx.where(
        params.flip_u[:, None] < config.p_flip,
        mx.array([-1.0, 1.0, 1.0]),
        mx.ones(3),
    )  # [n, 3]
    m_flip = flip_vec[:, :, None] * eye  # diag_embed

    # affine: rotation by U(-deg, deg) + translation U(-t, t); inverse =
    # rotate(-theta) then untranslate
    theta = (params.theta_u * 2 - 1) * (config.degrees * math.pi / 180)
    tx = (params.tx_u * 2 - 1) * (2 * config.translate)
    ty = (params.ty_u * 2 - 1) * (2 * config.translate)
    cos, sin = mx.cos(-theta), mx.sin(-theta)
    zeros = mx.zeros_like(cos)
    ones = mx.ones_like(cos)
    m_aff = mx.stack(
        [
            mx.stack([cos, -sin, -(cos * tx - sin * ty)], axis=-1),
            mx.stack([sin, cos, -(sin * tx + cos * ty)], axis=-1),
            mx.stack([zeros, zeros, ones], axis=-1),
        ],
        axis=-2,
    )
    keep_a = (params.affine_gate_u >= config.p_affine)[:, None, None]
    m_aff = mx.where(keep_a, eye, m_aff)

    # perspective: rank-1 projective perturbation (see the torch docstring)
    px = (params.px_u * 2 - 1) * (config.distortion / 2)
    py = (params.py_u * 2 - 1) * (config.distortion / 2)
    m_persp = mx.stack(
        [
            mx.stack([ones, zeros, zeros], axis=-1),
            mx.stack([zeros, ones, zeros], axis=-1),
            mx.stack([px, py, ones], axis=-1),
        ],
        axis=-2,
    )
    keep_p = (params.persp_gate_u >= config.p_persp)[:, None, None]
    m_persp = mx.where(keep_p, eye, m_persp)

    # forward order flip -> affine -> perspective; inverse composes in the
    # same order with each op inverted
    return m_flip @ m_aff @ m_persp


def color_matrices(params: AugParams, config: AugConfig) -> mx.array:
    """Per-sample 3x3 color matrices ``[n, 3, 3]``: saturation lerp toward
    luma + hue rotation around the gray axis — port of
    ``BatchedAugs._color_matrices`` (augs.py:111-140), gate included.
    """
    eye = mx.eye(3)
    luma = mx.array([[0.299, 0.587, 0.114]] * 3)
    s = (1 + (params.sat_u * 2 - 1) * config.sat)[:, None, None]
    m_sat = s * eye + (1 - s) * luma

    # hue rotation by angle h*2pi around the RGB gray axis (Rodrigues'
    # formula with unit axis (1,1,1)/sqrt(3))
    ang = (params.hue_u * 2 - 1) * (config.hue * 2 * math.pi)
    c, si = mx.cos(ang), mx.sin(ang)
    k = 1.0 / 3.0
    rt3 = 1.0 / math.sqrt(3.0)
    cc = 1 - c
    m_hue = mx.stack(
        [
            mx.stack([c + cc * k, cc * k - rt3 * si, cc * k + rt3 * si], axis=-1),
            mx.stack([cc * k + rt3 * si, c + cc * k, cc * k - rt3 * si], axis=-1),
            mx.stack([cc * k - rt3 * si, cc * k + rt3 * si, c + cc * k], axis=-1),
        ],
        axis=-2,
    )
    m = m_hue @ m_sat
    keep = (params.jitter_gate_u >= config.p_jitter)[:, None, None]
    return mx.where(keep, eye, m)


def erase_keep_mask(
    params: AugParams, config: AugConfig, h: int, w: int
) -> mx.array:
    """Multiplicative keep mask ``[n, h, w]`` fp32 (1 = keep, 0 = erased):
    one rectangle per sample, applied with p_erase — port of the erase
    section of ``BatchedAugs.forward`` (augs.py:164-189), gate included.
    """
    lo, hi = config.erase_scale
    area = (lo + params.area_u * (hi - lo)) * (h * w)
    ratio = mx.exp(params.log_r)
    eh = mx.minimum(mx.sqrt(area * ratio), h - 1)
    ew = mx.minimum(mx.sqrt(area / ratio), w - 1)
    y0 = (params.y0_u * (h - eh))[:, None, None]
    x0 = (params.x0_u * (w - ew))[:, None, None]
    eh = eh[:, None, None]
    ew = ew[:, None, None]
    yy = mx.arange(h, dtype=mx.float32).reshape(1, h, 1)
    xx = mx.arange(w, dtype=mx.float32).reshape(1, 1, w)
    inside = (yy >= y0) & (yy < y0 + eh) & (xx >= x0) & (xx < x0 + ew)
    erase = inside & (params.erase_gate_u[:, None, None] < config.p_erase)
    return mx.logical_not(erase).astype(mx.float32)


def grid_sample_border(x: mx.array, grid: mx.array) -> mx.array:
    """Bilinear grid sample, ``padding_mode="border"``,
    ``align_corners=False`` — the pure-ops gather recipe from the M2 spike
    (plan §"M2 spike results"; seam map Stage D).

    LOCAL COPY NOTE: S3's ``pytti.mlx_engine.sampler.grid_sample_border``
    was in flight when this slice was built, so S4 carries its own copy of
    the recipe rather than importing a moving target; the lead folds the
    two together at integration (both implement torch
    ``F.grid_sample(mode="bilinear", padding_mode="border",
    align_corners=False)`` exactly).

    ``x``: ``[n, c, h, w]`` fp32 (torch NCHW layout, matching the torch
    reference). ``grid``: ``[n, h_out, w_out, 2]`` fp32, normalized
    [-1, 1], ``(x, y)`` last-dim order. Differentiable w.r.t. both.
    """
    if x.ndim != 4:
        raise ValueError(f"x must be [n, c, h, w], got shape {x.shape}")
    if grid.ndim != 4 or grid.shape[-1] != 2:
        raise ValueError(f"grid must be [n, h, w, 2], got shape {grid.shape}")
    if grid.shape[0] != x.shape[0]:
        raise ValueError(
            f"batch mismatch: x has {x.shape[0]}, grid has {grid.shape[0]}"
        )
    if x.dtype != mx.float32 or grid.dtype != mx.float32:
        raise ValueError(
            f"fp32 required (lead precision ruling), got x={x.dtype}, "
            f"grid={grid.dtype}"
        )
    n, c, h, w = x.shape
    h_out, w_out = grid.shape[1], grid.shape[2]

    # unnormalize (align_corners=False), then clamp the *coordinate* into
    # the frame — torch's border padding semantics
    fx = mx.clip(((grid[..., 0] + 1) * w - 1) / 2, 0, w - 1)
    fy = mx.clip(((grid[..., 1] + 1) * h - 1) / 2, 0, h - 1)
    x0f = mx.floor(fx)
    y0f = mx.floor(fy)
    ix0 = x0f.astype(mx.int32)
    iy0 = y0f.astype(mx.int32)
    ix1 = mx.minimum(ix0 + 1, w - 1)  # weight is 0 whenever this clamps
    iy1 = mx.minimum(iy0 + 1, h - 1)

    flat = x.reshape(n, c, h * w)

    def gather(iy: mx.array, ix: mx.array) -> mx.array:
        idx = (iy * w + ix).reshape(n, 1, h_out * w_out)
        return mx.take_along_axis(flat, idx, axis=2).reshape(
            n, c, h_out, w_out
        )

    v00 = gather(iy0, ix0)
    v01 = gather(iy0, ix1)
    v10 = gather(iy1, ix0)
    v11 = gather(iy1, ix1)
    wx = (fx - x0f)[:, None, :, :]
    wy = (fy - y0f)[:, None, :, :]
    top = v00 * (1 - wx) + v01 * wx
    bot = v10 * (1 - wx) + v11 * wx
    return top * (1 - wy) + bot * wy


def apply_augs(x: mx.array, params: AugParams, config: AugConfig) -> mx.array:
    """The full aug stack — port of ``BatchedAugs.forward``
    (augs.py:142-189): one composed projective warp, per-sample linear
    color jitter, rectangle erasing.

    ``x``: ``[n, 3, h, w]`` fp32 (NCHW, matching the torch reference).
    Pure and deterministic given ``params``; differentiable w.r.t. ``x``.
    """
    if x.ndim != 4 or x.shape[1] != 3:
        raise ValueError(f"x must be [n, 3, h, w], got shape {x.shape}")
    if x.dtype != mx.float32:
        raise ValueError(
            f"fp32 required (lead precision ruling), got {x.dtype}"
        )
    n, _, h, w = x.shape
    for name, arr in zip(AugParams._fields, params, strict=True):
        if not isinstance(arr, mx.array):
            raise TypeError(f"params.{name} must be an mx.array, got {type(arr)}")
        if arr.shape != (n,):
            raise ValueError(
                f"params.{name} must have shape ({n},), got {arr.shape}"
            )
        if arr.dtype != mx.float32:
            raise ValueError(f"params.{name} must be fp32, got {arr.dtype}")

    # one composed projective warp; base grid at pixel centers in the
    # align_corners=False convention so an identity matrix reproduces the
    # input exactly (no stray subpixel resample on kept samples)
    m = warp_matrices(params, config)
    ys = (2 * mx.arange(h, dtype=mx.float32) + 1) / h - 1
    xs = (2 * mx.arange(w, dtype=mx.float32) + 1) / w - 1
    gy, gx = mx.meshgrid(ys, xs, indexing="ij")
    base = mx.stack([gx, gy, mx.ones_like(gx)], axis=-1).reshape(1, -1, 3)
    pts = base @ mx.transpose(m, (0, 2, 1))  # [n, h*w, 3]
    grid = (pts[..., :2] / mx.maximum(pts[..., 2:], 1e-6)).reshape(n, h, w, 2)
    x = grid_sample_border(x, grid)

    # per-sample linear color jitter
    cm = color_matrices(params, config)
    x = mx.einsum("nij,njhw->nihw", cm, x)

    # erasing: one rectangle per sample, applied with p_erase
    keep = erase_keep_mask(params, config, h, w)
    return x * keep[:, None, :, :]
