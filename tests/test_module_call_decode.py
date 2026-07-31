"""
Image models must be callable as modules: format_module() invokes
diff_image() via nn.Module.__call__ on every image-prompt / init-image
semantic path (Embedder.forward with input=None embeds a wrapper RGBImage
directly). Deleting DifferentiableImage.forward broke all init-image
renders with NotImplementedError — this pins the contract.
"""

import torch

from pytti.image_models.pixel import PixelImage
from pytti.image_models.rgb_image import RGBImage


def _pixel():
    return PixelImage(
        width=16, height=16, scale=1, palette_size=4, n_palettes=2,
        gamma=1, hdr_weight=0.01, norm_weight=0.1, device=torch.device("cpu"),
    )


def test_rgb_image_module_call_decodes():
    img = RGBImage(16, 16)
    img.train()
    assert tuple(img().shape) == (1, 3, 16, 16)
    img.eval()
    assert tuple(img().shape) == (1, 3, 16, 16)


def test_pixel_image_module_call_decodes():
    img = _pixel()
    img.train()
    assert tuple(img().shape) == (1, 3, 16, 16)
    img.eval()
    assert tuple(img().shape) == (1, 3, 16, 16)


def test_module_call_matches_training_decode():
    img = _pixel()
    img.train()
    torch.manual_seed(0)
    a = img()
    torch.manual_seed(0)
    b = img.decode_training_tensor()
    assert torch.equal(a, b)
