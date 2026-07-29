import gc
import math

import torch
from loguru import logger
from PIL import Image
from torch.nn import functional as F
from torchvision.transforms import functional as TF

from pytti import default_device, empty_cache, vram_usage_mode
from pytti.LossAug.MSELossClass import MSELoss

infer_helper = None


def init_AdaBins(device=None):
    global infer_helper
    if infer_helper is None:
        # Deferred: adabins is only required for 3D animation / depth losses.
        from adabins.infer import InferenceHelper

        with vram_usage_mode("AdaBins"):
            logger.debug("Loading AdaBins...")
            if device is None:
                device = default_device()
            infer_helper = InferenceHelper(dataset="nyu", device=device)
            logger.debug("AdaBins loaded.")


def _model_depth(tensor):
    """Depth from the raw AdaBins model, downscaling huge inputs first."""
    height, width = tensor.shape[-2:]
    max_depth_area = 500000
    image_area = width * height
    if image_area > max_depth_area:
        depth_scale_factor = math.sqrt(max_depth_area / image_area)
        height, width = int(height * depth_scale_factor), int(width * depth_scale_factor)
        tensor = TF.resize(
            tensor, (height, width), interpolation=TF.InterpolationMode.BILINEAR
        )
    _, depth_map = infer_helper.model(tensor)
    return depth_map


class DepthLoss(MSELoss):
    @torch.no_grad()
    def set_comp(self, pil_image):
        # pil_image = pil_image.resize(self.image_shape, Image.LANCZOS)
        self.comp.set_(DepthLoss.make_comp(pil_image))
        if self.use_mask and self.mask.shape[-2:] != self.comp.shape[-2:]:
            self.mask.set_(TF.resize(self.mask, self.comp.shape[-2:]))

    def get_loss(self, input, img):
        init_AdaBins(device=input.device)
        depth_map = _model_depth(input)
        depth_map = F.interpolate(
            depth_map, self.comp.shape[-2:], mode="bilinear", align_corners=True
        )
        return super().get_loss(depth_map, img)

    @classmethod
    @vram_usage_mode("Depth Loss")
    def make_comp(cls, pil_image, device=None):
        # Must run the target image through the *same* depth pipeline the
        # per-step input uses (_model_depth) — the previous predict_pil-based
        # comp was on a different scale/normalization, so the MSE compared
        # incommensurate values.
        if device is None:
            device = default_device()
        init_AdaBins(device=device)
        tensor = TF.to_tensor(pil_image).unsqueeze(0).to(device)
        with torch.no_grad():
            return _model_depth(tensor)

    @staticmethod
    def get_depth(pil_image, device=None):
        init_AdaBins(device=device)
        width, height = pil_image.size

        # if the area of an image is above this, the depth model fails
        max_depth_area = 500000
        image_area = width * height
        if image_area > max_depth_area:
            depth_scale_factor = math.sqrt(max_depth_area / image_area)
            depth_input = pil_image.resize(
                (int(width * depth_scale_factor), int(height * depth_scale_factor)),
                Image.LANCZOS,
            )
            depth_resized = True
        else:
            depth_input = pil_image
            depth_resized = False

        gc.collect()
        empty_cache()
        _, depth_map = infer_helper.predict_pil(depth_input)
        gc.collect()
        empty_cache()

        return depth_map, depth_resized
