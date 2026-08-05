"""
MLX cutout samplers — M2 slice S3 (docs/mlx-m2-seam-map.md, Stage D).

Ports of ``pytti.Perceptor.cutouts.samplers.pytti_batched``,
``pytti_smart``, and ``pytti_full`` (the torch code is the source of truth —
every formula here is a line-for-line transcription), plus the Embedder-side
border pre-pad and a differentiable pure-ops ``grid_sample``. Parity is gated in
``tests/test_mlx_engine_sampler.py``: cutout values <= 2/255 vs torch,
input-grad rel <= 1e-5 fp32, coordinate-convention equality, distribution
sanity per population.

Layout: NHWC end to end (input ``[1, H, W, C]``, cutouts
``[cutn, cut_size, cut_size, C]``) — the layout MLX convs and the M1 towers
consume — vs the torch samplers' NCHW. Offsets/sizes keep the torch
convention exactly: ``[cutn, 2]`` fp32, columns ``(x, y)``, normalized by
``(side_x, side_y)``, offsets in *unpadded*-image coordinates (negative when
a crop reaches into the padding of a non-clamp border mode).

Precision (lead ruling): samplers run fp32 — geometry math is fp32
explicitly; resampled values follow the input dtype (fp32 in the engine;
the fp16 cast happens at tower entry, ``mlx_backend/vit.py``).

RNG contract (seam map §3) — for the whole-step assembly:

- Every draw goes through ``mx.random.*``; no numpy / python ``random`` /
  torch RNG anywhere in this module.
- Each sampler takes ``key``. With ``key=None`` (the assembly default) all
  draws use MLX's implicit global stream, in the documented draw order — a
  compiled step function then only needs ``mx.random.state`` in its
  ``inputs=``/``outputs=`` lists (plus ``mx.random.seed(params.seed)`` at
  setup) for fresh draws each step. With an explicit key the call is pure:
  the key is split ONCE into the fixed per-draw subkeys listed in each
  sampler's docstring, and the global stream is neither read nor advanced.
- Explicit-key geometry is independent of ``noise_fac`` (the split count is
  fixed); in implicit mode the noise draws simply don't happen when
  ``noise_fac`` is falsy, so global-stream consumption differs between
  noise on/off — same as torch.
- Divergence contract: MLX's Threefry streams never reproduce torch-backend
  renders for the same seed (accepted, seam map §3).

``augs`` is a plain ``Callable[[mx.array], mx.array]`` on NHWC cutouts
(S4's MLX BatchedAugs, key-bound by the caller; identity in tests).
"""

import math

import mlx.core as mx

# torch-name equivalents of the non-clamp border modes (Embedder.PADDING_MODES)
BORDER_MODES = ("clamp", "mirror", "smear", "wrap", "black")


def grid_sample_border(input: mx.array, grid: mx.array) -> mx.array:
    """
    Pure-ops bilinear grid sampling: ``F.grid_sample(mode="bilinear",
    padding_mode="border", align_corners=False)`` on NHWC input.

    ``input`` is ``[N, H, W, C]`` (N may be 1 with a larger grid batch — the
    torch samplers' ``input.expand(cutn, ...)``); ``grid`` is
    ``[B, Ho, Wo, 2]``, normalized [-1, 1], x (width) coordinate first, like
    torch. Returns ``[B, Ho, Wo, C]`` in ``input``'s dtype.

    Derivation from torch's ``grid_sampler_2d``:
    - align_corners=False unnormalization: ``ix = ((gx + 1) * W - 1) / 2``;
    - padding_mode="border" clips the *unnormalized source coordinate* to
      ``[0, size - 1]`` before interpolation, so values AND input-gradients
      match torch exactly (torch's grid-gradient zeroing outside the border
      is irrelevant here: nothing differentiates w.r.t. the grid);
    - torch skips the out-of-bounds ``floor + 1`` tap at exactly
      ``ix == W - 1``; its bilinear weight is 0 there, so clamping the index
      to ``W - 1`` and keeping the tap is identical in value and gradient.

    Coordinate math stays fp32 (fp16 cannot address 512 px: ~0.5 px
    resolution at x=511); taps and lerp run in the input dtype. The gather
    (``mx.take``) differentiates to a scatter-add — the same accumulation
    torch's backward performs.
    """
    if input.ndim != 4:
        raise ValueError(f"input must be [N, H, W, C], got {input.shape}")
    if grid.ndim != 4 or grid.shape[-1] != 2:
        raise ValueError(f"grid must be [B, Ho, Wo, 2], got {grid.shape}")
    n, h, w, c = input.shape
    b, ho, wo, _ = grid.shape
    if n not in (1, b):
        raise ValueError(f"batch mismatch: input N={n} vs grid B={b}")

    ix = mx.clip(((grid[..., 0].astype(mx.float32) + 1) * w - 1) * 0.5, 0, w - 1)
    iy = mx.clip(((grid[..., 1].astype(mx.float32) + 1) * h - 1) * 0.5, 0, h - 1)
    x0f = mx.floor(ix)
    y0f = mx.floor(iy)
    wx = (ix - x0f).astype(input.dtype)[..., None]
    wy = (iy - y0f).astype(input.dtype)[..., None]
    x0 = x0f.astype(mx.int32)
    y0 = y0f.astype(mx.int32)
    x1 = mx.minimum(x0 + 1, w - 1)
    y1 = mx.minimum(y0 + 1, h - 1)

    flat = input.reshape(n * h * w, c)
    if n == 1:
        base = mx.zeros((b, 1, 1), dtype=mx.int32)
    else:
        base = (mx.arange(b, dtype=mx.int32) * (h * w)).reshape(b, 1, 1)

    def tap(yi: mx.array, xi: mx.array) -> mx.array:
        idx = (base + yi * w + xi).reshape(-1)
        return mx.take(flat, idx, axis=0).reshape(b, ho, wo, c)

    top = tap(y0, x0) * (1 - wx) + tap(y0, x1) * wx
    bot = tap(y1, x0) * (1 - wx) + tap(y1, x1) * wx
    return top * (1 - wy) + bot * wy


def pad_image(
    input: mx.array, side_x: int, side_y: int, padding: float, border_mode: str
) -> mx.array:
    """
    The Embedder-side pre-pad (``HDMultiClipEmbedder.cutout_batches``) in
    MLX NHWC: pads ``[1, side_y, side_x, C]`` by
    ``(min(round(side * padding), side))`` per axis. ``"clamp"`` returns the
    input unchanged (the Embedder's gate lives here so callers pad
    unconditionally). Modes map to torch ``F.pad`` as mirror->reflect,
    smear->replicate, wrap->circular, black->constant(0); reflect/circular
    are composed from index gathers (absent from ``mx.pad``), replicate is
    native ``mode="edge"``, constant native.
    """
    if input.ndim != 4 or input.shape[0] != 1:
        raise ValueError(
            f"pad_image expects a single-image batch [1, H, W, C], "
            f"got {tuple(input.shape)}"
        )
    if input.shape[1] != side_y or input.shape[2] != side_x:
        raise ValueError(
            f"pad_image: expected input [1, {side_y}, {side_x}, C], "
            f"got {tuple(input.shape)}"
        )
    if border_mode not in BORDER_MODES:
        raise ValueError(
            f"unknown border_mode {border_mode!r}; expected one of {BORDER_MODES}"
        )
    if border_mode == "clamp":
        return input
    paddingx = min(round(side_x * padding), side_x)
    paddingy = min(round(side_y * padding), side_y)
    widths = [(0, 0), (paddingy, paddingy), (paddingx, paddingx), (0, 0)]
    if border_mode == "black":
        return mx.pad(input, widths, mode="constant")
    if border_mode == "smear":
        return mx.pad(input, widths, mode="edge")
    # mirror / wrap: one gather per spatial axis (separable, exactly F.pad)
    out = _index_pad(input, paddingy, axis=1, border_mode=border_mode)
    return _index_pad(out, paddingx, axis=2, border_mode=border_mode)


def _index_pad(x: mx.array, pad: int, axis: int, border_mode: str) -> mx.array:
    """Reflect ("mirror") / circular ("wrap") padding of one axis as an
    index gather, matching torch ``F.pad`` including its bound contracts
    (reflect: pad < dim; circular: pad <= dim)."""
    if pad == 0:
        return x
    n = x.shape[axis]
    if border_mode == "mirror":
        if pad >= n:
            raise ValueError(
                f"mirror (reflect) padding {pad} must be < dim size {n}"
            )
        left = pad - mx.arange(pad)  # [pad, ..., 1]
        right = n - 2 - mx.arange(pad)  # [n-2, ..., n-1-pad]
    else:  # wrap
        if pad > n:
            raise ValueError(f"wrap (circular) padding {pad} must be <= dim size {n}")
        left = n - pad + mx.arange(pad)  # last `pad` entries
        right = mx.arange(pad)  # first `pad` entries
    idx = mx.concatenate([left, mx.arange(n), right])
    return mx.take(x, idx, axis=axis)


def _affine_crop_grid(
    x0: mx.array,
    y0: mx.array,
    sizes_px: mx.array,
    cut_size: int,
    in_h: int,
    in_w: int,
) -> mx.array:
    """
    Sampling grid for a batch of square crops, ``[cutn, cut_size, cut_size,
    2]`` — the torch ``_affine_crop_grid`` exactly (samplers.py is truth).

    Output pixel j samples source pixel ``x0 + (j + 0.5) * size / cut_size
    - 0.5`` (``F.interpolate``'s align_corners=False source coordinate),
    then the grid is clamped per cutout to the crop interior
    ``[x0, x0 + size - 1]``: F.interpolate clamps source coordinates to the
    crop's own edges (edge replication when size < cut_size), and without
    the clamp grid_sample would read neighboring pixels outside the crop.
    The torch version builds thetas for ``F.affine_grid``; those thetas are
    diagonal (axis-aligned scale + translate), so the grid is written
    directly here: affine_grid's ac=False base coordinate for output index
    j of extent S is ``(2j + 1) / S - 1``.

    ``x0``, ``y0``, ``sizes_px`` are ``[cutn]`` fp32 arrays of integral
    pixel values already resolved into the coordinate frame of the input
    actually being sampled (padded frame for non-clamp border modes).
    """
    cutn = sizes_px.shape[0]
    base = (2.0 * mx.arange(cut_size, dtype=mx.float32) + 1.0) / cut_size - 1.0

    def axis_coords(p0: mx.array, extent: int) -> mx.array:
        coords = (sizes_px / extent)[:, None] * base + (
            (2 * p0 + sizes_px) / extent - 1
        )[:, None]
        lo = ((2 * p0 + 1) / extent - 1)[:, None]
        hi = ((2 * (p0 + sizes_px - 1) + 1) / extent - 1)[:, None]
        return mx.maximum(mx.minimum(coords, hi), lo)  # [cutn, cut_size]

    gx = mx.broadcast_to(
        axis_coords(x0, in_w)[:, None, :], (cutn, cut_size, cut_size)
    )
    gy = mx.broadcast_to(
        axis_coords(y0, in_h)[:, :, None], (cutn, cut_size, cut_size)
    )
    return mx.stack([gx, gy], axis=-1)


def _validate_sampler_input(
    name: str,
    input: mx.array,
    side_x: int,
    side_y: int,
    paddingx: int,
    paddingy: int,
    border_mode: str,
) -> tuple[int, int]:
    """Shared fail-loud input contract; returns (in_h, in_w) of the frame
    actually being sampled."""
    if border_mode not in BORDER_MODES:
        raise ValueError(
            f"unknown border_mode {border_mode!r}; expected one of {BORDER_MODES}"
        )
    if input.ndim != 4 or input.shape[0] != 1:
        raise ValueError(
            f"{name} expects a single-image batch [1, H, W, C], "
            f"got {tuple(input.shape)}"
        )
    if border_mode == "clamp":
        in_h, in_w = side_y, side_x
    else:
        in_h, in_w = side_y + 2 * paddingy, side_x + 2 * paddingx
    if (input.shape[1], input.shape[2]) != (in_h, in_w):
        raise ValueError(
            f"{name}: border_mode={border_mode!r} expects input {(in_h, in_w)} "
            f"(pre-padded unless 'clamp'), got {tuple(input.shape[1:3])}"
        )
    return in_h, in_w


def _cut_batch(
    input: mx.array,
    sizes_px: mx.array,
    offsetx: mx.array,
    offsety: mx.array,
    *,
    side_x: int,
    side_y: int,
    cut_size: int,
    paddingx: int,
    paddingy: int,
    border_mode: str,
) -> tuple[mx.array, mx.array, mx.array]:
    """
    Deterministic half shared by both samplers (and the tests' injection
    point): resample given integral crop geometry into ``cut_size`` squares
    and package the return convention.

    ``sizes_px``/``offsetx``/``offsety`` are ``[cutn]`` fp32 arrays of
    integral pixel values, offsets in UNPADDED image coordinates (the
    reported convention); non-clamp modes shift by (paddingx, paddingy)
    into the pre-padded input frame here, exactly like the torch samplers.
    Returns (cutouts ``[cutn, cut_size, cut_size, C]``, offsets, sizes).
    """
    if border_mode == "clamp":
        x0, y0 = offsetx, offsety
        in_h, in_w = side_y, side_x
    else:
        x0, y0 = offsetx + paddingx, offsety + paddingy
        in_h, in_w = side_y + 2 * paddingy, side_x + 2 * paddingx
    grid = _affine_crop_grid(x0, y0, sizes_px, cut_size, in_h, in_w)
    cutouts = grid_sample_border(input, grid)
    offsets = mx.stack([offsetx / side_x, offsety / side_y], axis=-1)
    sizes = mx.stack([sizes_px / side_x, sizes_px / side_y], axis=-1)
    return cutouts, offsets, sizes


def _split_keys(key: mx.array | None, num: int) -> list[mx.array | None]:
    """Explicit key -> `num` fixed subkeys; None -> implicit global stream
    (each draw advances ``mx.random.state`` in documented order)."""
    if key is None:
        return [None] * num
    return list(mx.random.split(key, num))


def _apply_noise(
    cutouts: mx.array,
    noise_fac: float,
    k_fac: mx.array | None,
    k_noise: mx.array | None,
) -> mx.array:
    """``cutouts += U(0, noise_fac)[cutn,1,1,1] * N(0,1)`` — samplers.py's
    noise_fac application (torch :255-257), NHWC broadcast."""
    cutn = cutouts.shape[0]
    facs = mx.random.uniform(
        0, noise_fac, (cutn, 1, 1, 1), dtype=cutouts.dtype, key=k_fac
    )
    field = mx.random.normal(shape=cutouts.shape, dtype=cutouts.dtype, key=k_noise)
    return cutouts + facs * field


def pytti_batched(
    input: mx.array,
    side_x: int,
    side_y: int,
    cut_size: int,
    padding: float,
    cutn: int,
    cut_pow: float,
    border_mode: str,
    augs,
    noise_fac: float,
    key: mx.array | None = None,
) -> tuple[mx.array, mx.array, mx.array]:
    """
    MLX port of ``samplers.pytti_batched`` (classic N(0.8, 0.3) size
    distribution, uniform placement). Same input contract (``[1, H, W, C]``
    NHWC, pre-padded via ``pad_image`` for non-clamp border modes) and
    return convention as the torch sampler; ``device`` is dropped (unified
    memory), ``key`` added per the module RNG contract.

    Draw order (== subkey order under an explicit key):
    1. sizes ~ N(0.8, 0.3) [cutn]   2. randx ~ U[0,1) [cutn]
    3. randy ~ U[0,1) [cutn]        4. noise facs ~ U[0, noise_fac)
    5. noise field ~ N(0, 1)        (4-5 only drawn when noise_fac)
    """
    paddingx = min(round(side_x * padding), side_x)
    paddingy = min(round(side_y * padding), side_y)
    _validate_sampler_input(
        "pytti_batched", input, side_x, side_y, paddingx, paddingy, border_mode
    )
    max_size = min(side_x, side_y)
    k_size, k_x, k_y, k_fac, k_noise = _split_keys(key, 5)

    # int(max_size * N(0.8, 0.3).clip(cut_size / max_size, 1) ** cut_pow)
    sizes_px = mx.floor(
        mx.clip(
            mx.random.normal(shape=(cutn,), loc=0.8, scale=0.3, key=k_size),
            cut_size / max_size,
            1.0,
        )
        ** cut_pow
        * max_size
    )
    offsetx_max = side_x - sizes_px + 1
    offsety_max = side_y - sizes_px + 1
    randx = mx.random.uniform(shape=(cutn,), key=k_x)
    randy = mx.random.uniform(shape=(cutn,), key=k_y)
    if border_mode == "clamp":
        offsetx = mx.floor(randx * (offsetx_max + 2 * paddingx) - paddingx)
        offsety = mx.floor(randy * (offsety_max + 2 * paddingy) - paddingy)
        # clamp to side - size (the documented off-by-one fix vs classic)
        offsetx = mx.minimum(mx.maximum(offsetx, 0.0), side_x - sizes_px)
        offsety = mx.minimum(mx.maximum(offsety, 0.0), side_y - sizes_px)
    else:
        px = mx.minimum(sizes_px, paddingx)
        py = mx.minimum(sizes_px, paddingy)
        offsetx = mx.floor(randx * (offsetx_max + 2 * px) - px)
        offsety = mx.floor(randy * (offsety_max + 2 * py) - py)

    cutouts, offsets, sizes = _cut_batch(
        input,
        sizes_px,
        offsetx,
        offsety,
        side_x=side_x,
        side_y=side_y,
        cut_size=cut_size,
        paddingx=paddingx,
        paddingy=paddingy,
        border_mode=border_mode,
    )
    cutouts = augs(cutouts)
    if noise_fac:
        cutouts = _apply_noise(cutouts, noise_fac, k_fac, k_noise)
    return cutouts, offsets, sizes


def _stratified_cells(
    n_cells: int, n_rows: int
) -> tuple[mx.array, mx.array, mx.array, mx.array]:
    """
    Fractional cell boxes for a row-major grid of exactly ``n_cells`` cells
    in ``n_rows`` rows over the unit square — ``samplers._stratified_cells``
    verbatim (rows holding one extra cell come first; every row spans the
    full [0, 1) width, so any n_cells fully tiles the square). Returns
    ``(x_lo, x_width, y_lo, y_height)``, each ``[n_cells]`` fp32; cell i is
    deterministic in i (callers permute other attributes, not the cells).
    """
    if not 1 <= n_rows <= n_cells:
        raise ValueError(
            f"_stratified_cells needs 1 <= n_rows <= n_cells, got {n_rows=} {n_cells=}"
        )
    base, rem = divmod(n_cells, n_rows)  # first `rem` rows hold base+1 cells
    idx = mx.arange(n_cells)
    split = rem * (base + 1)
    in_big = idx < split
    row = mx.where(in_big, idx // (base + 1), rem + (idx - split) // base)
    col = mx.where(in_big, idx % (base + 1), (idx - split) % base)
    ncols = base + in_big.astype(mx.float32)  # base+1 in the first `rem` rows
    x_lo = col.astype(mx.float32) / ncols
    x_width = 1.0 / ncols
    y_lo = row.astype(mx.float32) / n_rows
    y_height = mx.full((n_cells,), 1.0 / n_rows, dtype=mx.float32)
    return x_lo, x_width, y_lo, y_height


def _inscribed_anchors(
    n: int, side_x: int, side_y: int, k_x: mx.array | None, k_y: mx.array | None
) -> tuple[mx.array, mx.array, mx.array]:
    """
    ``samplers._inscribed_anchors`` verbatim: ``n`` full-frame anchor cuts —
    every size exactly the inscribed square, positions stratified along the
    free axis (evenly spaced lanes + uniform jitter; the degenerate axis of
    a square canvas yields all-zero offsets naturally), always inside the
    unpadded frame. Shared by ``pytti_smart`` (its global-anchor population)
    and ``pytti_full`` (anchors at n == cutn). Draws x jitter then y jitter
    on ``k_x``/``k_y``. Returns (sizes_px, offsetx, offsety), each ``[n]``
    fp32.
    """
    if n < 1:
        raise ValueError(f"_inscribed_anchors needs n >= 1, got {n}")
    max_size = min(side_x, side_y)
    sizes = mx.full((n,), max_size, dtype=mx.float32)
    free_x = side_x - max_size
    free_y = side_y - max_size
    lane = mx.arange(n, dtype=mx.float32)
    offx = mx.floor(
        (lane + mx.random.uniform(shape=(n,), key=k_x)) * (free_x / n)
    )
    offy = mx.floor(
        (lane + mx.random.uniform(shape=(n,), key=k_y)) * (free_y / n)
    )
    return sizes, offx, offy


def pytti_smart(
    input: mx.array,
    side_x: int,
    side_y: int,
    cut_size: int,
    padding: float,
    cutn: int,
    cut_pow: float,
    border_mode: str,
    augs,
    noise_fac: float,
    key: mx.array | None = None,
) -> tuple[mx.array, mx.array, mx.array]:
    """
    MLX port of ``samplers.pytti_smart`` (the engine default): two designed
    populations — ``n_global = max(2, round(cutn * 0.25))`` inscribed-square
    anchors stratified along the free axis, then ``n_detail = cutn -
    n_global`` stratified detail cuts (size fraction ``0.2 + 0.35 * u^1.5``
    of the inscribed square, clamped to ``[cut_size, max_size]``, one crop
    origin jittered per cell of a near-square grid, sizes randomly permuted
    across cells). Output rows are [anchors, then detail cell
    0..n_detail-1]. ``cut_pow`` is accepted but UNUSED (signature parity —
    it shapes the classic distribution this sampler replaces). Same input /
    return / border-mode contract as ``pytti_batched``; requires
    ``cutn >= 3``.

    Draw order (== subkey order under an explicit key):
    1. anchor x jitter ~ U[0,1) [n_global]  2. anchor y jitter [n_global]
    3. size u ~ U[0,1) [n_detail]           4. permutation u [n_detail]
    5. cell x jitter [n_detail]             6. cell y jitter [n_detail]
    7. noise facs ~ U[0, noise_fac)         8. noise field ~ N(0, 1)
    (7-8 only drawn when noise_fac)
    """
    if cutn < 3:
        raise ValueError(
            f"pytti_smart needs cutn >= 3 (2 global anchors + >= 1 detail cut), "
            f"got {cutn}"
        )
    paddingx = min(round(side_x * padding), side_x)
    paddingy = min(round(side_y * padding), side_y)
    _validate_sampler_input(
        "pytti_smart", input, side_x, side_y, paddingx, paddingy, border_mode
    )
    max_size = min(side_x, side_y)
    n_global = max(2, round(cutn * 0.25))
    n_detail = cutn - n_global
    k_gx, k_gy, k_u, k_perm, k_tx, k_ty, k_fac, k_noise = _split_keys(key, 8)

    # --- population 1: global anchors (shared with pytti_full) -------------
    g_sizes, g_offx, g_offy = _inscribed_anchors(n_global, side_x, side_y, k_gx, k_gy)

    # --- population 2: stratified detail -----------------------------------
    u = mx.random.uniform(shape=(n_detail,), key=k_u)
    d_sizes = mx.clip(
        mx.floor((0.2 + 0.35 * u**1.5) * max_size), cut_size, max_size
    )
    # random size-to-cell assignment (argsort of uniforms == uniform random
    # permutation): cell position independent of crop size
    perm = mx.argsort(mx.random.uniform(shape=(n_detail,), key=k_perm))
    d_sizes = d_sizes[perm]

    # near-square grid, more rows than columns on portrait canvases
    a = math.ceil(math.sqrt(n_detail))
    b = math.ceil(n_detail / a)  # a >= b
    n_rows = a if side_y > side_x else b
    cx_lo, cx_w, cy_lo, cy_h = _stratified_cells(n_detail, n_rows)
    tx = cx_lo + mx.random.uniform(shape=(n_detail,), key=k_tx) * cx_w
    ty = cy_lo + mx.random.uniform(shape=(n_detail,), key=k_ty) * cy_h

    # Map the unit-square cell coordinate onto each cutout's own valid
    # offset domain — same domain semantics as pytti_batched per border mode.
    if border_mode == "clamp":
        d_offx = mx.floor(tx * (side_x - d_sizes))
        d_offy = mx.floor(ty * (side_y - d_sizes))
    else:
        px = mx.minimum(d_sizes, paddingx)
        py = mx.minimum(d_sizes, paddingy)
        d_offx = mx.floor(tx * (side_x - d_sizes + 2 * px) - px)
        d_offy = mx.floor(ty * (side_y - d_sizes + 2 * py) - py)

    # --- assemble, one grid_sample for the whole batch ----------------------
    cutouts, offsets, sizes = _cut_batch(
        input,
        mx.concatenate([g_sizes, d_sizes]),
        mx.concatenate([g_offx, d_offx]),
        mx.concatenate([g_offy, d_offy]),
        side_x=side_x,
        side_y=side_y,
        cut_size=cut_size,
        paddingx=paddingx,
        paddingy=paddingy,
        border_mode=border_mode,
    )
    cutouts = augs(cutouts)
    if noise_fac:
        cutouts = _apply_noise(cutouts, noise_fac, k_fac, k_noise)
    return cutouts, offsets, sizes


def pytti_full(
    input: mx.array,
    side_x: int,
    side_y: int,
    cut_size: int,
    padding: float,
    cutn: int,
    cut_pow: float,
    border_mode: str,
    augs,
    noise_fac: float,
    key: mx.array | None = None,
) -> tuple[mx.array, mx.array, mx.array]:
    """
    MLX port of ``samplers.pytti_full`` (the torch code is the source of
    truth): full-vision sampling — EVERY cutout is the full inscribed square
    (``size == min(side_x, side_y)`` exactly), i.e. ``pytti_smart``'s
    global-anchor population at ``n_global == cutn`` via the shared
    ``_inscribed_anchors``. Square canvas: all offsets 0 (identical crops
    pre-aug; augs + noise are the diversity source). Non-square: offsets
    stratified along the free axis, always inside the unpadded frame. Same
    input / return / border-mode contract as ``pytti_batched``;
    ``cut_pow`` accepted but UNUSED (one size only); ``cutn >= 1``. The
    sampler choice is a trace-time constant like the others — in-graph
    geometry is pure ops, no host syncs, no retrace.

    Draw order (== subkey order under an explicit key):
    1. anchor x jitter ~ U[0,1) [cutn]   2. anchor y jitter [cutn]
    3. noise facs ~ U[0, noise_fac)      4. noise field ~ N(0, 1)
    (3-4 only drawn when noise_fac)
    """
    if cutn < 1:
        raise ValueError(f"pytti_full needs cutn >= 1, got {cutn}")
    paddingx = min(round(side_x * padding), side_x)
    paddingy = min(round(side_y * padding), side_y)
    _validate_sampler_input(
        "pytti_full", input, side_x, side_y, paddingx, paddingy, border_mode
    )
    k_gx, k_gy, k_fac, k_noise = _split_keys(key, 4)

    sizes_px, offsetx, offsety = _inscribed_anchors(cutn, side_x, side_y, k_gx, k_gy)
    cutouts, offsets, sizes = _cut_batch(
        input,
        sizes_px,
        offsetx,
        offsety,
        side_x=side_x,
        side_y=side_y,
        cut_size=cut_size,
        paddingx=paddingx,
        paddingy=paddingy,
        border_mode=border_mode,
    )
    cutouts = augs(cutouts)
    if noise_fac:
        cutouts = _apply_noise(cutouts, noise_fac, k_fac, k_noise)
    return cutouts, offsets, sizes
