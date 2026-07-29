import math

import torch
from PIL import Image
from torchvision.transforms import functional as TF

from pytti import default_device, fetch, vram_usage_mode
from pytti.LossAug.MSELossClass import MSELoss


class LatentLoss(MSELoss):
    @torch.no_grad()
    def __init__(
        self,
        comp,
        weight=0.5,
        stop=-math.inf,
        name="direct target loss",
        image_shape=None,
    ):
        super().__init__(comp, weight, stop, name, image_shape)
        self.pil_image = None
        self.has_latent = False
        w, h = image_shape
        self.direct_loss = MSELoss(
            TF.resize(comp.clone(), (h, w)), weight, stop, name, image_shape
        )

    @torch.no_grad()
    def set_comp(self, pil_image, device=None):
        self.pil_image = pil_image
        self.has_latent = False
        self.direct_loss.set_comp(pil_image.resize(self.image_shape, Image.LANCZOS))

    @classmethod
    @vram_usage_mode("Latent Image Loss")
    @torch.no_grad()
    def build(
        cls,
        name,
        image_shape,
        *,
        weight="1",
        stop="-inf",
        mask=None,
        pil_image=None,
        path="",
        device=None,
    ):
        if device is None:
            device = default_device()
        if pil_image is None and path:
            pil_image = Image.open(fetch(path)).convert("RGB")
        # placeholder comp; the actual latent target is computed lazily in
        # get_loss once the image model exists (needs img.make_latent)
        comp = (
            MSELoss.make_comp(pil_image, device=device)
            if pil_image is not None
            else torch.zeros(1, 1, 1, 1, device=device)
        )
        out = cls(comp, weight, stop, name + " (latent)", image_shape)
        if pil_image is not None:
            out.set_comp(pil_image)
        out.apply_mask_spec(mask)
        return out

    def set_mask(self, mask, inverted=False):
        self.direct_loss.set_mask(mask, inverted)
        super().set_mask(mask, inverted)

    def get_loss(self, input, img):
        if not self.has_latent:
            latent = img.make_latent(self.pil_image)
            with torch.no_grad():
                self.comp.set_(latent.clone())
            self.has_latent = True
        l1 = super().get_loss(img.get_latent_tensor(), img) / 2
        l2 = self.direct_loss.get_loss(input, img) / 10
        return l1 + l2
