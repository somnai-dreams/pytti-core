"""
Methods for obtaining cutouts, agnostic to augmentations.

Cutout choices have a significant impact on the performance of the perceptors and the
overall look of the image.

The objects defined here probably are only being used in pytti.Perceptor.cutouts.Embedder.HDMultiClipEmbedder, but they
should be sufficiently general for use in notebooks without pyttitools otherwise in use.
"""


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
