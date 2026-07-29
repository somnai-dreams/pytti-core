import math
from pathlib import Path

import torch
from PIL import Image
from torch.nn import functional as F
from torchvision.transforms import functional as TF

from pytti import default_device, fetch, vram_usage_mode
from pytti.device import memory_format_for
from pytti.LossAug.BaseLossClass import Loss
from pytti.prompt_spec import MaskAll, MaskImage, MaskSpec, MaskVideo
from pytti.rotoscoper import Rotoscoper


class MSELoss(Loss):
    @torch.no_grad()
    def __init__(
        self,
        comp,
        weight=0.5,
        stop=-math.inf,
        name="direct target loss",
        image_shape=None,
        device=None,
    ):
        super().__init__(weight, stop, name, device)
        self.register_buffer("comp", comp)
        if image_shape is None:
            height, width = comp.shape[-2:]
            image_shape = (width, height)
        self.image_shape = image_shape
        self.register_buffer("mask", torch.ones(1, 1, 1, 1, device=self.device))
        self.use_mask = False

    @classmethod
    @vram_usage_mode("Loss Augs")
    @torch.no_grad()
    def build(
        cls,
        name,
        image_shape,
        *,
        weight="1",
        stop="-inf",
        mask: MaskSpec | None = None,
        pil_image=None,
        path="",
        device=None,
    ):
        """
        Construct a direct loss from typed fields — no string parsing.
        `mask` is a prompt_spec.MaskSpec (image/video masks only).
        """
        if device is None:
            device = default_device()
        if pil_image is None and path:
            pil_image = Image.open(fetch(path)).convert("RGB")
        if pil_image is not None:
            im = pil_image.resize(image_shape, Image.LANCZOS)
            comp = cls.make_comp(im, device=device)
        else:
            comp = torch.zeros(1, 1, 1, 1, device=device)
        out = cls(comp, weight, stop, name + " (direct)", image_shape, device=device)
        out.apply_mask_spec(mask)
        return out

    def apply_mask_spec(self, mask: "MaskSpec | None"):
        if mask is None or isinstance(mask, MaskAll):
            return
        if isinstance(mask, MaskImage):
            pil = Image.open(fetch(mask.path)).convert("L")
            self.set_mask(pil, inverted=mask.inverted)
        elif isinstance(mask, MaskVideo):
            Rotoscoper(mask.path, self, inverted=mask.inverted).update(0)
        else:
            raise ValueError(
                f"Direct losses only support image/video masks, got {mask!r} "
                f"for {self.name!r} — geometric and semantic masks apply to "
                "semantic (CLIP) prompts."
            )

    @torch.no_grad()
    def set_mask(self, mask, inverted=False, device=None):
        if device is None:
            device = self.device
        if isinstance(mask, str) and mask != "":
            if mask.startswith("-"):
                mask = mask[1:]
                inverted = True
            if Path(mask.strip()).suffix.lower() == ".mp4":
                # hand the (already-parsed) inversion flag to the rotoscoper —
                # it re-applies the mask every frame
                r = Rotoscoper(mask, self, inverted=inverted)
                r.update(0)
                return
            mask = Image.open(fetch(mask)).convert("L")
        if isinstance(mask, Image.Image):
            with vram_usage_mode("Masks"):
                mask = (
                    TF.to_tensor(mask)
                    .unsqueeze(0)
                    .to(device, memory_format=memory_format_for(device))
                )
        if isinstance(mask, torch.Tensor):
            self.mask.set_(mask if not inverted else (1 - mask))
            self.use_mask = True
        else:
            self.use_mask = False

    @classmethod
    def convert_input(cls, input, img):
        return input

    @classmethod
    def make_comp(cls, pil_image, device=None):
        if device is None:
            device = default_device()
        out = (
            TF.to_tensor(pil_image)
            .unsqueeze(0)
            .to(device, memory_format=memory_format_for(device))
        )
        return cls.convert_input(out, None)

    def set_comp(self, pil_image, device=None):
        if device is None:
            device = self.device
        self.comp.set_(type(self).make_comp(pil_image, device=device))

    def get_loss(self, input, img):
        input = type(self).convert_input(input, img)
        if self.use_mask:
            if self.mask.shape[-2:] != input.shape[-2:]:
                # cache the mask at the input's resolution (resize once per
                # shape change, directly — set_mask would re-apply inversion)
                with torch.no_grad():
                    self.mask.set_(TF.resize(self.mask, input.shape[-2:]))
            return F.mse_loss(input * self.mask, self.comp * self.mask)
        else:
            return F.mse_loss(input, self.comp)
