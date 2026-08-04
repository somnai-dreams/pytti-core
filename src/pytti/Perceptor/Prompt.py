import math
from collections.abc import Callable

import torch
from PIL import Image
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist
from torch import nn
from torch.nn import functional as F
from torchvision.transforms import functional as TF

import pytti
from pytti import (
    cat_with_pad,
    fetch,
    format_input,
    is_zero_weight,
    parametric_eval,
    replace_grad,
    vram_usage_mode,
)
from pytti.device import default_device
from pytti.image_models import RGBImage
from pytti.prompt_spec import (
    MaskAll,
    MaskGeometric,
    MaskImage,
    MaskSemantic,
    MaskSpec,
    MaskVideo,
    parse_prompt_spec,
)

# from pytti.Notebook import Rotoscoper
from pytti.rotoscoper import Rotoscoper


def spherical_dist_loss(x, y):
    x = F.normalize(x, dim=-1)
    y = F.normalize(y, dim=-1)
    return x.sub(y).norm(dim=-1).div(2).arcsin().pow(2).mul(2)


def sizes_to_coherence_weights(
    sizes: torch.Tensor, side_x: int, side_y: int
) -> torch.Tensor:
    """
    Per-cutout semantic-loss weights from the cutout samplers' ``sizes``
    tensor (config ``coherence_weighting``).

    Sampler contract (cutouts/samplers.py, mirrored by mlx_engine/sampler.py):
    crops are squares of ``size_px <= max_size = min(side_x, side_y)`` pixels
    and ``sizes[..., (0, 1)] == (size_px / side_x, size_px / side_y)``. The
    min-side column is therefore exactly ``size_px / max_size`` — the crop's
    fraction of the inscribed square (one fp division of integral values, so
    a full-frame anchor reads exactly 1.0 even on non-square canvases).

    Weights: full-frame anchors (fraction == 1.0 — the smart sampler's
    designed anchor population; batched/classic full-size draws, which their
    clamp-to-1 produces ~25% of the time, count identically) get 3.0x, and
    EVERY cutout additionally scales by its fraction; the whole vector is
    then normalized to mean ~1.0 (fp rounding leaves the fp32 mean a few
    ulp off for adverse inputs). Mass conservation is completed at the
    COMPOSITION sites (Prompt.forward / the M1 bridge): coherence composes
    multiplicatively with mask weights and is rescaled per prompt so
    mean(|mask| * coh) == mean(|mask|) — uniform masks make that a no-op,
    non-uniform image masks keep their configured strength instead of
    silently rescaling by the mask/coherence covariance. Geometric masks
    gate via STOPS rather than weights, so their interaction is inherently
    data-dependent and is not (and cannot be) renormalized. Rationale:
    with uniform weights ~75% of semantic gradient pushes small crops
    toward the FULL prompt (per-patch prompt stuffing = the tapestry
    look); anchors must outvote patches on global composition.

    ``sizes`` is ``[..., 2]`` with the cutout/perceptor axes leading (either
    order — the normalization is over all elements, so it is invariant to
    axis order and to LocationAwareMCIP's row permutation); returns
    ``sizes.shape[:-1]``, strictly positive. Pure and sync-free: safe in the
    step path on any device and inside the MLX graph's torch twin.
    """
    if side_x <= 0 or side_y <= 0:
        raise ValueError(f"canvas dims must be positive, got {(side_x, side_y)}")
    if sizes.ndim < 2 or sizes.shape[-1] != 2:
        raise ValueError(
            f"sizes must be [..., 2] (x, y) size fractions, got {tuple(sizes.shape)}"
        )
    fraction = sizes[..., 0] if side_x <= side_y else sizes[..., 1]
    raw = torch.where(fraction == 1.0, fraction * 3.0, fraction)
    return raw / raw.mean()


def make_mask(spec: MaskSpec, thresh):
    """
    Turn a typed MaskSpec into a mask callable (or a Rotoscoper for video
    masks, which the caller attaches to the prompt).
    """
    if isinstance(spec, MaskVideo):
        return Rotoscoper(spec.path, inverted=spec.inverted)
    if isinstance(spec, MaskAll):
        mask_fun = mask_all
    elif isinstance(spec, MaskGeometric):
        mask_fun = MASK_DICT[spec.key]
    elif isinstance(spec, MaskImage):
        mask_fun = mask_image(spec.path, inverted=spec.inverted)
    elif isinstance(spec, MaskSemantic):
        mask_fun = mask_semantic(spec.text)
    else:
        raise TypeError(f"Unknown mask spec: {spec!r}")

    def masker(pos, size, emb):
        return mask_fun(pos, size, emb, parametric_eval(thresh))

    # Semantic masks are the one kind that reads the image embedding; the
    # MLX bridge computes mask vectors pre-tower (no embedding exists yet)
    # and uses this to fail loudly instead of silently passing None through.
    masker.embed_dependent = isinstance(spec, MaskSemantic)
    return masker


@torch.no_grad()
def mask_right(pos, size, emb, thresh=0.5):
    cent = pos[..., 0] + size[..., 0] / 2
    return cent.lt(thresh).float(), 1


@torch.no_grad()
def mask_left(pos, size, emb, thresh=0.5):
    cent = pos[..., 0] + size[..., 0] / 2
    return cent.gt(thresh).float(), 1


@torch.no_grad()
def mask_down(pos, size, emb, thresh=0.5):
    cent = pos[..., 1] + size[..., 1] / 2
    return cent.lt(thresh).float(), 1


@torch.no_grad()
def mask_near(pos, size, emb, thresh=0.5):
    return size.min(dim=-1)[0].lt(thresh).float(), 1


@torch.no_grad()
def mask_far(pos, size, emb, thresh=0.5):
    return size.min(dim=-1)[0].gt(thresh).float(), 1


@torch.no_grad()
def mask_up(pos, size, emb, thresh=0.5):
    cent = pos[..., 1] + size[..., 1] / 2
    return cent.gt(thresh).float(), 1


@torch.no_grad()
def mask_all(pos, size, emb, thresh=0.5):
    return torch.zeros_like(size[..., 0]).fill_(float("-inf")), 1


@torch.no_grad()
def mask_image(path, inverted=False, device=None):
    if device is None:
        device = default_device()
    if isinstance(path, Image.Image):
        mask_pil = path
    else:
        if path.startswith("-"):
            path = path[1:]
            inverted = True
        mask_pil = Image.open(fetch(path)).convert("L")
    mask_tensor = TF.to_tensor(mask_pil).squeeze().to(device)
    mask_tensor = 1 - mask_tensor if inverted else mask_tensor
    mu = mask_tensor.mean()
    err_tensor = torch.zeros_like(mask_tensor)
    height, width = mask_tensor.shape[-2:]
    size_tensor = torch.as_tensor([width, height], device=device).view(1, 1, -1)

    @torch.no_grad()
    def mask(pos, size, emb, thresh=0.5):
        if mu < 0.001:
            # no use trying on a tiny mask I think
            return torch.zeros_like(size[..., 0]).fill_(float("-inf")), 0
        low, high = pos, pos + size
        low_px, high_px = (low * size_tensor).floor().long(), (
            high * size_tensor
        ).floor().long()
        low_xs, low_ys = low_px[..., 0].contiguous(), low_px[..., 1].contiguous()
        high_xs, high_ys = high_px[..., 0].contiguous(), high_px[..., 1].contiguous()

        low_xs.clamp_(0, width - 1)
        low_ys.clamp_(0, height - 1)
        high_xs.clamp_(0, width - 1)
        high_ys.clamp_(0, height - 1)
        As = (high_xs - low_xs) * (high_ys - low_ys)
        ind = torch.argsort(As.view(-1)).flip([0])

        out = torch.zeros(*pos.shape[:-1], device=device)
        N = out.numel()
        for i in ind:
            low_x = low_xs.view(-1)[i]
            low_y = low_ys.view(-1)[i]
            high_x = high_xs.view(-1)[i]
            high_y = high_ys.view(-1)[i]
            A = (high_x - low_x) * (high_y - low_y)
            if A > 0:
                weight_sample, err_sample = (
                    mask_tensor[low_y:high_y, low_x:high_x],
                    err_tensor[low_y:high_y, low_x:high_x],
                )
                err_sample = err_sample.sign() * err_sample.abs().sqrt()
                weight = (weight_sample + err_sample).mean()
                err = mask_tensor[low_y:high_y, low_x:high_x] - weight
                err_tensor[low_y:high_y, low_x:high_x] += (
                    err.square().div(N).mul(err.sign())
                )
            else:
                weight = torch.as_tensor(0, device=device)
            out.view(-1)[i] = weight
        # err_tensor.mul_(0.99)
        return torch.zeros_like(out).fill_(-math.inf), out / mu.sqrt()

    return mask


@torch.no_grad()
def mask_semantic(text, device=None):
    if device is None:
        device = default_device()
    perceptors = pytti.Perceptor.CLIP_PERCEPTORS
    embeds = cat_with_pad([p.embed_text(text, device) for p in perceptors])

    @torch.no_grad()
    def mask(pos, size, emb, thresh=0.5):
        # that's right, it's ridiculous garbage!
        if thresh == 0.5000873264:
            return spherical_dist_loss(emb, embeds), 1
        else:
            thresh = thresh * 0.3 + 0.7
            return spherical_dist_loss(emb, embeds).gt(thresh), 1

    return mask


MASK_DICT = {
    "a": mask_all,
    "r": mask_right,
    "l": mask_left,
    "d": mask_down,
    "u": mask_up,
    "n": mask_near,
    "f": mask_far,
}


@torch.no_grad()
def parse_prompt(embedder, prompt_string="", pil_image=None, device=None):
    """
    It takes a prompt string,
    parses it, and returns a Prompt object

    :param embedder: the embedder to use
    :param prompt_string: the prompt to be parsed
    :param pil_image: if you want to use an image instead of text, pass it in here
    :param device: the device to run on
    :return: A Prompt object.
    """
    if device is None:
        device = default_device()
    spec = parse_prompt_spec(prompt_string)
    mask = make_mask(spec.mask, spec.cutoff)
    if isinstance(mask, Rotoscoper):
        roto = mask
        mask = mask_all
    else:
        roto = None
    image_path = spec.image_path()
    if image_path is not None:
        pil_image = Image.open(fetch(image_path)).convert("RGB")
    if pil_image is not None:
        dummy = RGBImage(*pil_image.size)
        dummy.encode_image(pil_image)
        out = LocationAwareMCIP(
            *embedder(dummy),
            embedder,
            spec.weight,
            spec.stop,
            spec.text,
            prompt_string,
            mask=mask,
        )
    else:
        perceptors = pytti.Perceptor.CLIP_PERCEPTORS
        embeds = cat_with_pad([p.embed_text(spec.text, device) for p in perceptors])
        out = Prompt(embeds, spec.weight, spec.stop, spec.text, prompt_string, mask=mask)
    if roto is not None:
        roto.target = out
        roto.update(0)
    return out


class Prompt(nn.Module):
    @torch.no_grad()
    def __init__(
        self,
        embeds: torch.Tensor,
        weight: str,
        stop: str,
        text: str,
        prompt_string: str,
        mask: Callable = mask_all,
        device=None,
    ):
        super().__init__()
        if device is None:
            device = default_device()
        self.device = device
        if embeds is not None:
            self.register_buffer("embeds", embeds)
        self.weight = weight
        self.stop = stop
        self.input_axes = ("n", "c", "i")
        self.prompt_string = prompt_string
        self.text = text.encode("ascii", "ignore").decode("ascii")
        self.text = (
            (self.text[:20] + ".." + self.text[-5:])
            if len(self.text) > 27
            else self.text
        )
        self.mask = mask
        self.enabled = True

    def __repr__(self):
        return self.prompt_string

    def __str__(self):
        return self.text

    def set_mask(self, pil_image, inverted=False):
        """
        Given an input image, registers the image as a mask.

        :param pil_image: The image to be masked
        :param inverted: If True, the mask is inverted, so the background is black and the foreground is
        white, defaults to False (optional)
        """
        self.mask = mask_image(pil_image, inverted=inverted)

    def set_enabled(self, enabled):
        self.enabled = enabled

    def forward(
        self, embed, position, size, offset=0.0, device=None, coherence_canvas=None
    ):
        """
        input: (Tensor) input CLIP embedding
        returns the input's loss compared to the saved embedding

        coherence_canvas: None (off), or the (side_x, side_y) canvas dims the
        sampler normalized `size` by — non-None applies the config
        ``coherence_weighting`` per-cutout weights (anchors 3x, everything
        scaled by view size, mean renormalized to 1; see
        sizes_to_coherence_weights). They compose MULTIPLICATIVELY with the
        spatial/semantic mask weights below: a masked detail crop is
        down-weighted by both its mask and its size.
        """
        if device is None:
            device = self.device
        if not self.enabled or is_zero_weight(self.weight):
            # loss_raw must be a tensor: train() records loss_raw.detach()
            zero = torch.as_tensor(offset, device=device)
            return zero, zero
        dists_raw = spherical_dist_loss(embed, self.embeds) + offset
        weight = torch.as_tensor(parametric_eval(self.weight), device=device)
        stop = torch.as_tensor(parametric_eval(self.stop), device=device)

        mask_stops, mask_weights = self.mask(position, size, embed.detach())
        weight = torch.as_tensor(mask_weights, device=device) * weight
        if coherence_canvas is not None:
            coh = sizes_to_coherence_weights(size, *coherence_canvas)
            # Coherence REDISTRIBUTES this prompt's gradient across views —
            # it must not change the prompt's total strength. With uniform
            # mask weights the mean-1 coh vector already conserves mass
            # (scale == 1 to fp rounding); with non-uniform mask weights
            # (image masks / rotoscopes) the covariance between mask and
            # coh would silently rescale the prompt up to ~3x, so rescale
            # per prompt: mean(|mask| * coh * scale) == mean(|mask|).
            # No host syncs — everything stays on-device.
            mw = torch.as_tensor(mask_weights, device=device, dtype=coh.dtype).abs()
            if mw.dim() == 0:
                mw = mw.expand_as(coh)
            scale = mw.mean() / (mw * coh).mean().clamp_min(1e-8)
            weight = weight * coh * scale
        sign_offset = weight.sign().clamp(max=0)

        dists = dists_raw * weight.sign()
        stops = torch.maximum(mask_stops + sign_offset, stop)
        dists = weight.abs() * replace_grad(dists, torch.maximum(dists, stops))
        return dists.mean(), dists_raw.mean()


class MultiClipImagePrompt(Prompt):
    @torch.no_grad()
    @vram_usage_mode("Image Prompts")
    def __init__(
        self,
        embeds: torch.Tensor,
        positions,
        sizes,
        embedder,
        weight: str,
        stop: str,
        text: str,
        prompt_string: str,
        mask: Callable = mask_all,
        device=None,
    ):
        self.input_axes = ("c", "n", "i")
        super().__init__(
            format_input(embeds, embedder, self),
            weight,
            stop,
            text + " (semantic)",
            prompt_string,
            mask=mask,
            device=device,
        )
        self.input_axes = ("c", "n", "i")
        self.register_buffer("positions", format_input(positions, embedder, self))
        self.register_buffer("sizes", format_input(sizes, embedder, self))

    @torch.no_grad()
    @vram_usage_mode("Image Prompts")
    def set_image(self, embedder, pil_image):
        """
        It takes an embedder, a PIL image, and returns the embeddings, positions, and sizes

        :param embedder: the embedder function
        :param pil_image: The image to be embedded
        """
        width, height = pil_image.size
        img = RGBImage(width, height)
        img.encode_image(pil_image)
        embeds, positions, sizes = embedder(img)
        embeds = embeds.clone()
        self.positions.set_(format_input(positions, embedder, self))
        self.sizes.set_(format_input(sizes, embedder, self))
        self.embeds.set_(format_input(embeds, embedder, self))


def minimize_average_distance(tensor_a, tensor_b):
    """
    tensor_a: pytorch tensor
    tensor_b: pytorch tensor
    returns: tensor of indicies in tensor_a which will minimize the euclidian distance between the elments of the two tensors
    """
    # Why are we doing this on the CPU?
    # ....and why are we looping here?
    tensor_a = tensor_a.detach().cpu().numpy()
    tensor_b = tensor_b.detach().cpu().numpy()
    out = []
    for c in range(tensor_a.shape[0]):
        a = tensor_a[c, :, :]
        b = tensor_b[c, :, :]
        distances = cdist(a, b)
        row_ind, col_ind = linear_sum_assignment(distances)
        col_ind = torch.as_tensor(col_ind)
        out.append(col_ind)
    return out


class LocationAwareMCIP(MultiClipImagePrompt):
    def forward(self, embed, position, size, coherence_canvas=None):
        """
        input: (Tensor) input CLIP embedding
        returns the input's loss compared to the saved embedding

        coherence_canvas passes through to Prompt.forward, which computes the
        coherence weights from the PERMUTED size rows below — the weights
        follow each row's own geometry, and the mean normalization is
        permutation-invariant.
        """
        cent_a = self.positions + self.sizes / 2
        cent_b = position + size / 2
        indices = minimize_average_distance(cent_a, cent_b)
        embed = torch.stack([a[i] for a, i in zip(embed, indices, strict=True)])
        position = torch.stack([a[i] for a, i in zip(position, indices, strict=True)])
        size = torch.stack([a[i] for a, i in zip(size, indices, strict=True)])
        return super().forward(
            embed, position, size, offset=0.7, coherence_canvas=coherence_canvas
        )
