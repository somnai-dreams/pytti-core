
import torch
from torch import nn
from torch.nn import functional as F

import pytti
from pytti import cat_with_pad, format_input, format_module
from pytti.device import default_device, memory_format_for

# from pytti.ImageGuide import DirectImageGuide
from pytti.image_models import DifferentiableImage

# import .cutouts
# import .cutouts as cutouts
# import cutouts
from .cutouts import augs as cutouts_augs
from .cutouts import samplers as cutouts_samplers

PADDING_MODES = {
    "mirror": "reflect",
    "smear": "replicate",
    "wrap": "circular",
    "black": "constant",
}

CUTOUT_SAMPLERS = {
    "classic": cutouts_samplers.pytti_classic,
    "batched": cutouts_samplers.pytti_batched,
    "smart": cutouts_samplers.pytti_smart,
}


class HDMultiClipEmbedder(nn.Module):
    """
    Multi-CLIP embedder that uses cutouts to view images larger than 224x224.
    with code by Katherine Crowson (https://github.com/crowsonkb)
    and jbusted (https://twitter.com/jbusted1)
    and dribnet (https://github.com/dribnet)
    """

    def __init__(
        self,
        perceptors=None,
        cutn=40,
        cut_pow=1.5,
        padding=0.25,
        border_mode="clamp",
        noise_fac=0.1,
        cutout_sampler="smart",
        device=None,
    ):
        super().__init__()
        if device is None:
            device = default_device()
        self.device = device
        if perceptors is None:
            perceptors = pytti.Perceptor.CLIP_PERCEPTORS
        self.cut_sizes = [p.cut_size for p in perceptors]
        self.cutn = cutn
        self.noise_fac = noise_fac
        if cutout_sampler not in CUTOUT_SAMPLERS:
            raise ValueError(
                f"unknown cutout_sampler {cutout_sampler!r}; "
                f"expected one of {sorted(CUTOUT_SAMPLERS)}"
            )
        # the aug stack follows the sampler choice: "batched"/"smart" use the
        # sync-free composed-warp stack, "classic" the 2021 kornia one
        self.augs = (
            cutouts_augs.pytti_classic()
            if cutout_sampler == "classic"
            else cutouts_augs.pytti_batched()
        )
        self.input_axes = ("n", "s", "y", "x")
        self.output_axes = ("c", "n", "i")
        self.perceptors = perceptors
        self.padding = padding
        self.cut_pow = cut_pow
        self.border_mode = border_mode
        self.cutout_sampler = cutout_sampler

    def make_cutouts(
        self,
        input: torch.Tensor,
        side_x,
        side_y,
        cut_size,
        ####
        # padding,
        # cutn,
        # cut_pow,
        # border_mode,
        # augs,
        # noise_fac,
        ####
        device=None,
    ) -> tuple[list, list, list]:
        if device is None:
            device = self.device
        sampler = CUTOUT_SAMPLERS[self.cutout_sampler]
        cutouts, offsets, sizes = sampler(
            input=input,
            side_x=side_x,
            side_y=side_y,
            cut_size=cut_size,
            padding=self.padding,
            cutn=self.cutn,
            cut_pow=self.cut_pow,
            border_mode=self.border_mode,
            augs=self.augs,
            noise_fac=self.noise_fac,
            device=device,
        )
        return cutouts, offsets, sizes

    def cutout_batches(
        self,
        diff_image: DifferentiableImage,
        input=None,
        device=None,
    ) -> list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        """
        Per-perceptor (cutouts, offsets, sizes) — everything forward() does
        before normalization and encoding. Perceptors with the same input
        resolution share one cutout batch (the SAME tensor objects: sampling
        + augs + noise cost once instead of per tower). Consumes the RNG
        stream exactly as forward() does; the MLX bridge calls this so both
        backends see identical cutouts under a fixed seed.
        """
        if device is None:
            device = self.device
        side_x, side_y = diff_image.image_shape
        if input is None:
            input = format_module(diff_image, self).to(
                device=device, memory_format=memory_format_for(device)
            )
        else:
            input = format_input(input, diff_image, self).to(
                device=device, memory_format=memory_format_for(device)
            )

        paddingx = min(round(side_x * self.padding), side_x)
        paddingy = min(round(side_y * self.padding), side_y)
        if self.border_mode != "clamp":
            input = F.pad(
                input,
                (paddingx, paddingx, paddingy, paddingy),
                mode=PADDING_MODES[self.border_mode],
            )
        cutout_cache: dict[int, tuple] = {}
        batches = []
        for cut_size in self.cut_sizes:
            if cut_size not in cutout_cache:
                cutout_cache[cut_size] = self.make_cutouts(
                    input, side_x, side_y, cut_size
                )
            batches.append(cutout_cache[cut_size])
        return batches

    def forward(
        self,
        # diff_image: DirectImageGuide,
        diff_image: DifferentiableImage,
        input=None,
        device=None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        diff_image: (DifferentiableImage) input image
        returns images embeds
        """
        batches = self.cutout_batches(diff_image, input=input, device=device)
        image_embeds = []
        all_offsets = []
        all_sizes = []
        # each perceptor applies its own normalization stats (SigLIP != CLIP)
        for perceptor, (cutouts, offsets, sizes) in zip(
            self.perceptors, batches, strict=True
        ):
            clip_in = perceptor.normalize(cutouts)
            image_embeds.append(perceptor.encode_image(clip_in).float().unsqueeze(0))
            all_offsets.append(offsets)
            all_sizes.append(sizes)
        return (
            cat_with_pad(image_embeds),
            torch.stack(all_offsets),
            torch.stack(all_sizes),
        )
