"""
Methods for obtaining cutouts, agnostic to augmentations.

Cutout choices have a significant impact on the performance of the perceptors and the
overall look of the image.

The objects defined here probably are only being used in pytti.Perceptor.cutouts.Embedder.HDMultiClipEmbedder, but they
should be sufficiently general for use in notebooks without pyttitools otherwise in use.
"""


import math

import torch
from torch.nn import functional as F


def pytti_classic(
    # self,
    input: torch.Tensor,
    side_x,
    side_y,
    cut_size,
    padding,
    cutn,
    cut_pow,
    border_mode,
    augs,
    noise_fac,
    device,
) -> tuple[list, list, list]:
    """
    This is the cutout method that was already in use in the original pytti.
    """
    max_size = min(side_x, side_y)
    paddingx = min(round(side_x * padding), side_x)
    paddingy = min(round(side_y * padding), side_y)
    cutouts = []
    offsets = []
    sizes = []
    for _ in range(cutn):
        # mean is 0.8
        # varience is 0.3
        size = int(
            max_size
            * (
                torch.zeros(
                    1,
                )
                .normal_(mean=0.8, std=0.3)
                .clip(cut_size / max_size, 1.0)
                ** cut_pow
            )
        )
        offsetx_max = side_x - size + 1
        offsety_max = side_y - size + 1
        if border_mode == "clamp":
            offsetx = torch.clamp(
                (torch.rand([]) * (offsetx_max + 2 * paddingx) - paddingx)
                .floor()
                .int(),
                0,
                offsetx_max,
            )
            offsety = torch.clamp(
                (torch.rand([]) * (offsety_max + 2 * paddingy) - paddingy)
                .floor()
                .int(),
                0,
                offsety_max,
            )
            cutout = input[:, :, offsety : offsety + size, offsetx : offsetx + size]
        else:
            px = min(size, paddingx)
            py = min(size, paddingy)
            offsetx = (torch.rand([]) * (offsetx_max + 2 * px) - px).floor().int()
            offsety = (torch.rand([]) * (offsety_max + 2 * py) - py).floor().int()
            cutout = input[
                :,
                :,
                paddingy + offsety : paddingy + offsety + size,
                paddingx + offsetx : paddingx + offsetx + size,
            ]
        # Bilinear resize instead of adaptive_avg_pool2d: implemented (forward
        # AND backward) on every backend — MPS lacks non-divisible adaptive
        # pooling, and antialias=True has no MPS backward.
        cutouts.append(
            F.interpolate(
                cutout,
                size=(cut_size, cut_size),
                mode="bilinear",
                align_corners=False,
            )
        )
        offsets.append(
            torch.as_tensor([[offsetx / side_x, offsety / side_y]]).to(device)
        )
        sizes.append(torch.as_tensor([[size / side_x, size / side_y]]).to(device))
    cutouts = augs(torch.cat(cutouts))
    offsets = torch.cat(offsets)
    sizes = torch.cat(sizes)
    if noise_fac:
        facs = cutouts.new_empty([cutn, 1, 1, 1]).uniform_(0, noise_fac)
        cutouts.add_(facs * torch.randn_like(cutouts))
    return cutouts, offsets, sizes


def _affine_crop_grid(
    x0: torch.Tensor,
    y0: torch.Tensor,
    sizes_px: torch.Tensor,
    channels: int,
    cut_size: int,
    in_h: int,
    in_w: int,
) -> torch.Tensor:
    """
    Sampling grid for a batch of square crops, [cutn, cut_size, cut_size, 2].

    Affine thetas [cutn, 2, 3] map the output grid onto each crop so that
    output pixel j samples source pixel x0 + (j + 0.5) * size / cut_size - 0.5
    — exactly F.interpolate's align_corners=False source coordinate. The grid
    is then clamped per cutout to the crop interior [x0, x0 + size - 1],
    because F.interpolate clamps source coordinates to the crop's own edges
    (edge replication when size < cut_size); without the clamp, grid_sample
    would read neighboring image pixels outside the crop.

    `x0`, `y0`, `sizes_px` are [cutn] float tensors of integral pixel values
    already resolved into the coordinate frame of the input actually being
    sampled (padded frame for non-clamp border modes).
    """
    cutn = sizes_px.shape[0]
    zeros = torch.zeros_like(sizes_px)
    theta = torch.stack(
        [
            torch.stack(
                [sizes_px / in_w, zeros, (2 * x0 + sizes_px) / in_w - 1], dim=-1
            ),
            torch.stack(
                [zeros, sizes_px / in_h, (2 * y0 + sizes_px) / in_h - 1], dim=-1
            ),
        ],
        dim=-2,
    )
    grid = F.affine_grid(
        theta, (cutn, channels, cut_size, cut_size), align_corners=False
    )

    def _norm_x(px_coord):
        return (2 * px_coord + 1) / in_w - 1

    def _norm_y(px_coord):
        return (2 * px_coord + 1) / in_h - 1

    lo = torch.stack([_norm_x(x0), _norm_y(y0)], dim=-1).view(cutn, 1, 1, 2)
    hi = torch.stack(
        [_norm_x(x0 + sizes_px - 1), _norm_y(y0 + sizes_px - 1)], dim=-1
    ).view(cutn, 1, 1, 2)
    return torch.maximum(torch.minimum(grid, hi), lo)


def pytti_batched(
    input: torch.Tensor,
    side_x,
    side_y,
    cut_size,
    padding,
    cutn,
    cut_pow,
    border_mode,
    augs,
    noise_fac,
    device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Batched, sync-free reimplementation of `pytti_classic`: identical bilinear
    resampling math (one `F.grid_sample` reproduces per-crop
    slice + `F.interpolate(..., mode="bilinear", align_corners=False)` exactly),
    but all `cutn` cutouts are produced by a single kernel launch and every
    size/offset stays on-device as a tensor.

    Critical constraint: NO `int()` / `.item()` / python-float extraction
    anywhere in this function — each one is a device sync, and a naive synced
    version measured 8x *slower* than classic on MPS.

    Semantics match `pytti_classic`:
    - `border_mode == "clamp"`: `input` is the raw image `[1, C, side_y, side_x]`
      and crops are clamped inside it. (Classic clamps offsets to
      `side - size + 1`, which lets the slice overrun by one pixel and silently
      truncate to a `size-1`-wide crop; here offsets clamp to `side - size` so
      every crop is exactly `size` pixels. Same distribution otherwise.)
    - any other `border_mode`: `input` arrives pre-padded by the Embedder to
      `[1, C, side_y + 2*paddingy, side_x + 2*paddingx]`; crops are taken at
      `(paddingx + offsetx, paddingy + offsety)` in the padded image while the
      *reported* offsets stay in unpadded-image coordinates (may be negative).
    - returned `offsets` / `sizes` are `[cutn, 2]` tensors, columns `(x, y)`,
      normalized by `(side_x, side_y)` — same convention as classic, without
      classic's `2 * cutn` tiny host->device transfers.
    """
    if input.ndim != 4 or input.shape[0] != 1:
        raise ValueError(
            f"pytti_batched expects a single-image batch [1, C, H, W], got {tuple(input.shape)}"
        )
    dtype = input.dtype
    max_size = min(side_x, side_y)
    paddingx = min(round(side_x * padding), side_x)
    paddingy = min(round(side_y * padding), side_y)
    if border_mode == "clamp":
        in_h, in_w = side_y, side_x
    else:
        in_h, in_w = side_y + 2 * paddingy, side_x + 2 * paddingx
    if input.shape[-2:] != (in_h, in_w):
        raise ValueError(
            f"pytti_batched: border_mode={border_mode!r} expects input {(in_h, in_w)} "
            f"(pre-padded unless 'clamp'), got {tuple(input.shape[-2:])}"
        )

    # Per-cutout square sizes in pixels, sampled entirely on-device.
    # Same distribution as classic: int(max_size * N(0.8, 0.3).clip(lo, 1) ** cut_pow).
    sizes_px = (
        torch.empty(cutn, device=device, dtype=dtype)
        .normal_(mean=0.8, std=0.3)
        .clamp_(cut_size / max_size, 1.0)
        .pow_(cut_pow)
        .mul_(max_size)
        .floor_()
    )
    offsetx_max = side_x - sizes_px + 1
    offsety_max = side_y - sizes_px + 1
    randx = torch.rand(cutn, device=device, dtype=dtype)
    randy = torch.rand(cutn, device=device, dtype=dtype)
    if border_mode == "clamp":
        offsetx = (randx * (offsetx_max + 2 * paddingx) - paddingx).floor_()
        offsety = (randy * (offsety_max + 2 * paddingy) - paddingy).floor_()
        offsetx = torch.minimum(offsetx.clamp_(min=0), side_x - sizes_px)
        offsety = torch.minimum(offsety.clamp_(min=0), side_y - sizes_px)
        x0, y0 = offsetx, offsety  # crop origin in the (unpadded) input
    else:
        px = sizes_px.clamp(max=paddingx)
        py = sizes_px.clamp(max=paddingy)
        offsetx = (randx * (offsetx_max + 2 * px) - px).floor_()
        offsety = (randy * (offsety_max + 2 * py) - py).floor_()
        x0, y0 = offsetx + paddingx, offsety + paddingy  # shift into padded coords

    grid = _affine_crop_grid(x0, y0, sizes_px, input.shape[1], cut_size, in_h, in_w)
    cutouts = F.grid_sample(
        input.expand(cutn, -1, -1, -1),  # expand: view, no copy
        grid,
        mode="bilinear",
        padding_mode="border",
        align_corners=False,
    )
    offsets = torch.stack([offsetx / side_x, offsety / side_y], dim=-1)
    sizes = torch.stack([sizes_px / side_x, sizes_px / side_y], dim=-1)

    cutouts = augs(cutouts)
    if noise_fac:
        facs = cutouts.new_empty([cutn, 1, 1, 1]).uniform_(0, noise_fac)
        cutouts.add_(facs * torch.randn_like(cutouts))
    return cutouts, offsets, sizes


def _stratified_cells(
    n_cells: int, n_rows: int, device, dtype
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Fractional cell boxes for a row-major grid of exactly `n_cells` cells in
    `n_rows` rows over the unit square.

    Rows that must hold one extra cell come first, and every row's cells span
    the full [0, 1) width, so ANY n_cells fully tiles the unit square (no
    dropped corner cells when n_cells is not a rectangle number).

    Returns (x_lo, x_width, y_lo, y_height), each `[n_cells]` on `device`.
    Cell i of the returned arrays is deterministic in i — callers that permute
    the pairing of cells with other per-cutout attributes must permute those
    attributes, not the cells, so tests can recover cell membership by row
    index alone.
    """
    if not 1 <= n_rows <= n_cells:
        raise ValueError(
            f"_stratified_cells needs 1 <= n_rows <= n_cells, got {n_rows=} {n_cells=}"
        )
    base, rem = divmod(n_cells, n_rows)  # first `rem` rows hold base+1 cells
    idx = torch.arange(n_cells, device=device)
    split = rem * (base + 1)
    in_big = idx < split
    row = torch.where(in_big, idx // (base + 1), rem + (idx - split) // base)
    col = torch.where(in_big, idx % (base + 1), (idx - split) % base)
    ncols = base + in_big.to(dtype)  # base+1 in the first `rem` rows
    x_lo = col.to(dtype) / ncols
    x_width = 1.0 / ncols
    y_lo = row.to(dtype) / n_rows
    y_height = torch.full_like(x_lo, 1.0 / n_rows)
    return x_lo, x_width, y_lo, y_height


def _inscribed_anchors(
    n: int, side_x, side_y, device, dtype
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    `n` full-frame anchor cuts: every size is EXACTLY the inscribed square
    (min(side_x, side_y)) and positions are stratified along the free axis
    (evenly spaced lanes + uniform jitter within each). Exactly one of
    free_x/free_y is nonzero on a non-square canvas (both 0 when square), so
    one code path covers both: the degenerate axis yields all-zero offsets
    naturally. Offsets stay inside the unpadded frame for every border mode
    (a full-frame view of padding is not a view of the image).

    This is `pytti_smart`'s global-anchor population, factored so
    `pytti_full` (anchors at n == cutn) shares it verbatim. Draws 2 RNG
    vectors (x jitter, then y jitter), both `[n]` on `device`.

    Returns (sizes_px, offsetx, offsety), each `[n]`.
    """
    if n < 1:
        raise ValueError(f"_inscribed_anchors needs n >= 1, got {n}")
    max_size = min(side_x, side_y)
    sizes = torch.full((n,), max_size, device=device, dtype=dtype)
    free_x = side_x - max_size
    free_y = side_y - max_size
    lane = torch.arange(n, device=device, dtype=dtype)
    offx = (
        (lane + torch.rand(n, device=device, dtype=dtype))
        .mul_(free_x / n)
        .floor_()
    )
    offy = (
        (lane + torch.rand(n, device=device, dtype=dtype))
        .mul_(free_y / n)
        .floor_()
    )
    return sizes, offx, offy


def pytti_smart(
    input: torch.Tensor,
    side_x,
    side_y,
    cut_size,
    padding,
    cutn,
    cut_pow,
    border_mode,
    augs,
    noise_fac,
    device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Designed two-population cutout sampler, built to hold quality at cutn ~16
    where the classic N(0.8, 0.3) distribution needs ~40. Same signature,
    input contract, and return convention as `pytti_batched` (pre-padded input
    for non-clamp border modes; offsets/sizes `[cutn, 2]` normalized by
    `(side_x, side_y)`; augs + noise applied at the end; NO host syncs — no
    `int()`/`.item()`/python-float extraction from device tensors anywhere).

    `cut_pow` is accepted for signature compatibility but UNUSED — it shapes
    the classic/batched size distribution, which this sampler replaces with a
    designed one.

    Population 1 — GLOBAL ANCHORS, `n_global = max(2, round(cutn * 0.25))`:
    every cut is exactly the inscribed square (`size = min(side_x, side_y)`).
    On a non-square canvas the positions are stratified along the free axis
    (evenly spaced cells + uniform jitter within each); on a square canvas all
    offsets are 0 and the augs/noise decorrelate the copies. This is the Disco
    Diffusion "overview cut" pattern: consistent full-frame views anchor
    global composition, which the classic distribution almost never provides
    reliably. Anchor offsets stay inside the unpadded frame for every border
    mode (a full-frame view of padding is not a view of the image).

    Population 2 — STRATIFIED DETAIL, `n_detail = cutn - n_global`: size
    fraction `f = 0.2 + 0.35 * u^1.5, u ~ U(0,1)` of the inscribed square
    (range [0.2, 0.55], biased small), clamped so `size >= cut_size` (never
    upsample into CLIP) and `size <= max_size` (the clamp can only bind when
    `cut_size > max_size`). Placement is stratified, not iid: a near-square
    row-major grid of exactly `n_detail` cells tiles the per-cutout valid
    offset domain (same domain semantics as `pytti_batched` per border mode,
    including reach into padding for non-clamp modes), one crop origin
    jittered uniformly inside each cell — every step covers the whole canvas
    even at n_detail = 12. The size-to-cell assignment is randomly permuted so
    cell position and crop size are independent. Output rows are ordered
    [global anchors, then detail cell 0..n_detail-1].

    Requires `cutn >= 3` (2 global anchors + at least 1 detail cut).
    """
    if input.ndim != 4 or input.shape[0] != 1:
        raise ValueError(
            f"pytti_smart expects a single-image batch [1, C, H, W], got {tuple(input.shape)}"
        )
    if cutn < 3:
        raise ValueError(
            f"pytti_smart needs cutn >= 3 (2 global anchors + >= 1 detail cut), got {cutn}"
        )
    dtype = input.dtype
    max_size = min(side_x, side_y)
    paddingx = min(round(side_x * padding), side_x)
    paddingy = min(round(side_y * padding), side_y)
    if border_mode == "clamp":
        in_h, in_w = side_y, side_x
    else:
        in_h, in_w = side_y + 2 * paddingy, side_x + 2 * paddingx
    if input.shape[-2:] != (in_h, in_w):
        raise ValueError(
            f"pytti_smart: border_mode={border_mode!r} expects input {(in_h, in_w)} "
            f"(pre-padded unless 'clamp'), got {tuple(input.shape[-2:])}"
        )

    n_global = max(2, round(cutn * 0.25))
    n_detail = cutn - n_global

    # --- population 1: global anchors (shared with pytti_full) ------------
    g_sizes, g_offx, g_offy = _inscribed_anchors(
        n_global, side_x, side_y, device, dtype
    )

    # --- population 2: stratified detail -----------------------------------
    u = torch.rand(n_detail, device=device, dtype=dtype)
    d_sizes = (
        (0.2 + 0.35 * u.pow(1.5))
        .mul_(max_size)
        .floor_()
        .clamp_(min=cut_size, max=max_size)
    )
    # random size-to-cell assignment: cell position independent of crop size
    # (argsort of uniforms == a uniform random permutation, sync-free on
    # every backend)
    perm = torch.argsort(torch.rand(n_detail, device=device, dtype=dtype))
    d_sizes = d_sizes[perm]

    # near-square grid, more rows than columns on portrait canvases
    a = math.ceil(math.sqrt(n_detail))
    b = math.ceil(n_detail / a)  # a >= b
    n_rows = a if side_y > side_x else b
    cx_lo, cx_w, cy_lo, cy_h = _stratified_cells(n_detail, n_rows, device, dtype)
    tx = cx_lo + torch.rand(n_detail, device=device, dtype=dtype) * cx_w
    ty = cy_lo + torch.rand(n_detail, device=device, dtype=dtype) * cy_h

    # Map the unit-square cell coordinate onto each cutout's own valid offset
    # domain — same domain semantics as pytti_batched per border mode.
    if border_mode == "clamp":
        d_offx = (tx * (side_x - d_sizes)).floor_()
        d_offy = (ty * (side_y - d_sizes)).floor_()
    else:
        px = d_sizes.clamp(max=paddingx)
        py = d_sizes.clamp(max=paddingy)
        d_offx = (tx * (side_x - d_sizes + 2 * px) - px).floor_()
        d_offy = (ty * (side_y - d_sizes + 2 * py) - py).floor_()

    # --- assemble, one grid_sample for the whole batch ---------------------
    sizes_px = torch.cat([g_sizes, d_sizes])
    offsetx = torch.cat([g_offx, d_offx])
    offsety = torch.cat([g_offy, d_offy])
    if border_mode == "clamp":
        x0, y0 = offsetx, offsety
    else:
        x0, y0 = offsetx + paddingx, offsety + paddingy  # shift into padded coords

    grid = _affine_crop_grid(x0, y0, sizes_px, input.shape[1], cut_size, in_h, in_w)
    cutouts = F.grid_sample(
        input.expand(cutn, -1, -1, -1),  # expand: view, no copy
        grid,
        mode="bilinear",
        padding_mode="border",
        align_corners=False,
    )
    offsets = torch.stack([offsetx / side_x, offsety / side_y], dim=-1)
    sizes = torch.stack([sizes_px / side_x, sizes_px / side_y], dim=-1)

    cutouts = augs(cutouts)
    if noise_fac:
        facs = cutouts.new_empty([cutn, 1, 1, 1]).uniform_(0, noise_fac)
        cutouts.add_(facs * torch.randn_like(cutouts))
    return cutouts, offsets, sizes


def pytti_full(
    input: torch.Tensor,
    side_x,
    side_y,
    cut_size,
    padding,
    cutn,
    cut_pow,
    border_mode,
    augs,
    noise_fac,
    device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Full-vision sampler: EVERY cutout is the full inscribed square
    (size == max_size == min(side_x, side_y) exactly — `pytti_smart`'s
    global-anchor population at n_global == cutn; `_inscribed_anchors` is
    shared, not duplicated). No cut smaller than 100% resolution ever.

    Square canvas: all offsets are 0, so the crops are IDENTICAL pre-aug —
    the augs + noise_fac are the designed diversity source, exactly like
    smart's anchors. Non-square canvas: offsets stratified along the free
    axis (evenly spaced lanes + uniform jitter), always inside the unpadded
    frame regardless of border mode.

    Same signature, input contract, and return convention as
    `pytti_batched`/`pytti_smart` (pre-padded input for non-clamp border
    modes; offsets/sizes `[cutn, 2]` normalized by `(side_x, side_y)`; augs
    + noise applied last; NO host syncs). `cut_pow` is accepted for
    signature compatibility but UNUSED (it shapes the classic size
    distribution — here there is exactly one size). Works at any
    `cutn >= 1`; the designed pairing is LOW cutn (~8-16).
    """
    if input.ndim != 4 or input.shape[0] != 1:
        raise ValueError(
            f"pytti_full expects a single-image batch [1, C, H, W], got {tuple(input.shape)}"
        )
    if cutn < 1:
        raise ValueError(f"pytti_full needs cutn >= 1, got {cutn}")
    dtype = input.dtype
    paddingx = min(round(side_x * padding), side_x)
    paddingy = min(round(side_y * padding), side_y)
    if border_mode == "clamp":
        in_h, in_w = side_y, side_x
    else:
        in_h, in_w = side_y + 2 * paddingy, side_x + 2 * paddingx
    if input.shape[-2:] != (in_h, in_w):
        raise ValueError(
            f"pytti_full: border_mode={border_mode!r} expects input {(in_h, in_w)} "
            f"(pre-padded unless 'clamp'), got {tuple(input.shape[-2:])}"
        )

    sizes_px, offsetx, offsety = _inscribed_anchors(
        cutn, side_x, side_y, device, dtype
    )
    if border_mode == "clamp":
        x0, y0 = offsetx, offsety
    else:
        x0, y0 = offsetx + paddingx, offsety + paddingy  # shift into padded coords

    grid = _affine_crop_grid(x0, y0, sizes_px, input.shape[1], cut_size, in_h, in_w)
    cutouts = F.grid_sample(
        input.expand(cutn, -1, -1, -1),  # expand: view, no copy
        grid,
        mode="bilinear",
        padding_mode="border",
        align_corners=False,
    )
    offsets = torch.stack([offsetx / side_x, offsety / side_y], dim=-1)
    sizes = torch.stack([sizes_px / side_x, sizes_px / side_y], dim=-1)

    cutouts = augs(cutouts)
    if noise_fac:
        facs = cutouts.new_empty([cutn, 1, 1, 1]).uniform_(0, noise_fac)
        cutouts.add_(facs * torch.randn_like(cutouts))
    return cutouts, offsets, sizes
