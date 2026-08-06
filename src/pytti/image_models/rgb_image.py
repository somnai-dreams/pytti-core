import torch
from PIL import Image
from torch import nn
from torch.nn import functional as F
from torchvision.transforms import functional as TF

from pytti import clamp_with_grad
from pytti.device import default_device
from pytti.image_models import DifferentiableImage
from pytti.image_models.init_noise import (
    shaped_init_field,
    validate_spectrum_chroma,
)


class RGBImage(DifferentiableImage):
    """
    Naive RGB image representation
    """

    def __init__(self, width, height, scale=1, device=None):
        super().__init__(width * scale, height * scale)
        if device is None:
            device = default_device()
        self.device = device
        self.tensor = nn.Parameter(
            torch.zeros(1, 3, height, width).to(
                device=self.device, memory_format=torch.channels_last
            )
        )
        self.output_axes = ("n", "s", "y", "x")
        self.scale = scale

    def decode_tensor(self):
        width, height = self.image_shape
        out = F.interpolate(self.tensor, (height, width), mode="nearest")
        return clamp_with_grad(out, 0, 1)

    def clone(self):
        width, height = self.image_shape
        dummy = RGBImage(width // self.scale, height // self.scale, self.scale)
        with torch.no_grad():
            dummy.tensor.set_(self.tensor.clone())
        return dummy

    def get_image_tensor(self):
        return self.tensor.squeeze(0)

    @torch.no_grad()
    def set_image_tensor(self, tensor):
        self.tensor.set_(tensor.unsqueeze(0))

    @torch.no_grad()
    def encode_image(self, pil_image, device=None, **kwargs):
        if device is None:
            device = self.device
        width, height = self.image_shape
        scale = self.scale
        pil_image = pil_image.resize((width // scale, height // scale), Image.LANCZOS)
        self.tensor.set_(
            TF.to_tensor(pil_image)
            .unsqueeze(0)
            .to(device, memory_format=torch.channels_last)
        )

    @torch.no_grad()
    def encode_random(
        self,
        init_spectrum="white",
        init_spectrum_falloff=1.0,
        init_spectrum_chroma="full",
    ):
        """
        Overwrite the image with noise shaped per config ``init_spectrum``
        (see image_models/init_noise.py). 'white' keeps the original
        in-place uniform draw bit-for-bit and ignores the chroma knob (iid
        uniform has no low-frequency chroma to remove). The shaped spectra
        draw fields on the logical grid with cross-channel structure per
        ``init_spectrum_chroma``: 'full' = independent per-RGB-channel
        fields (the pre-knob behavior, bit-for-bit), 'natural' = lucid
        ImageNet color statistics, 'mono' = one luminance field broadcast.
        """
        validate_spectrum_chroma(init_spectrum_chroma)
        if init_spectrum == "white":
            self.tensor.uniform_()
            return
        height, width = self.tensor.shape[-2:]
        field = shaped_init_field(
            3,
            height,
            width,
            init_spectrum,
            init_spectrum_falloff,
            self.device,
            init_spectrum_chroma=init_spectrum_chroma,
        )
        self.tensor.copy_(field.unsqueeze(0))
