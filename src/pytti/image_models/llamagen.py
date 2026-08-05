"""
LlamaGen VQ image representation — a modernized taming-style VQGAN (same
conv encoder/decoder family, 8-dim L2-normalized codebook, rFID ~1.03 @512
for ds16 vs ~4-5 for taming f16). The latent is optimized exactly like
VQGANImage: raw codebook-space rows snapped through vector_quantize with a
straight-through gradient, EMA-smoothed for output frames.

torch-only: the mlx/mlx_full backends reject image_model=LlamaGen at startup
(workhorse dispatch + MLXStillEngine eligibility).

Weights: FoundationVision/LlamaGen on HF hub, revision-pinned, MIT.
Vendored architecture: pytti/vendor/llamagen/vq_model.py.
"""

import gc

import torch
from loguru import logger
from PIL import Image
from torch.nn import functional as F
from torchvision.transforms import functional as TF

from pytti import (
    clamp_with_grad,
    default_device,
    empty_cache,
    vram_usage_mode,
)
from pytti.config.model_names import LLAMAGEN_CHECKPOINT_FILES, LLAMAGEN_MODEL_NAMES
from pytti.image_models import EMAImage
from pytti.image_models.init_noise import require_white_init
from pytti.image_models.vqgan import vector_quantize

# One revision pin for the whole repo: every checkpoint file above is
# fetched at exactly this commit of FoundationVision/LlamaGen.
LLAMAGEN_HF_REPO = "FoundationVision/LlamaGen"
LLAMAGEN_HF_REVISION = "81e41139272c038412e4fe8f1c52a51ebbf95b8b"

LLAMAGEN_MODEL = None
LLAMAGEN_NAME = None


def load_llamagen_model(checkpoint_path, model_name):
    """
    Build the frozen LlamaGen VQ model for `model_name` (ds16|ds8) and load
    the published checkpoint. Published files carry the weights under a
    "model" key (training checkpoints add "ema"/optimizer state).
    """
    from pytti.vendor.llamagen.vq_model import VQ_models

    arch = {"ds16": "VQ-16", "ds8": "VQ-8"}[model_name]
    model = VQ_models[arch]()

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if "model" in checkpoint:
        state = checkpoint["model"]
    elif "ema" in checkpoint:
        state = checkpoint["ema"]
    else:
        raise RuntimeError(
            f"LlamaGen checkpoint {checkpoint_path} has neither a 'model' nor "
            f"an 'ema' key (found: {sorted(checkpoint)[:8]}) — not a published "
            "LlamaGen VQ checkpoint"
        )
    model.load_state_dict(state, strict=True)
    model.eval().requires_grad_(False)
    return model


class LlamaGenImage(EMAImage):
    """
    LlamaGen VQ latent image representation.
    width:  (positive integer) approximate image width in pixels  (rounded down to a multiple of f)
    height: (positive integer) approximate image height in pixels (rounded down to a multiple of f)
    model:  (VQModel) LlamaGen VQ model (defaults to the init_llamagen singleton)
    """

    @vram_usage_mode("LlamaGen Image")
    def __init__(self, width, height, scale=1, model=None, ema_val=0.99, device=None):
        if device is None:
            device = default_device()
        self.device = device

        if model is None:
            model = LLAMAGEN_MODEL
            if model is None:
                raise RuntimeError(
                    "ERROR: model is None and LlamaGen is not initialized — "
                    "call LlamaGenImage.init_llamagen first"
                )

        e_dim = model.quantize.e_dim  # 8
        n_toks = model.quantize.n_e  # 16384
        # the codebook convention is L2-normalized rows; snap targets and
        # random init both live on the unit sphere
        codebook = F.normalize(model.quantize.embedding.weight, dim=-1)

        f = 2 ** (model.decoder.num_resolutions - 1)  # ds16 -> 16, ds8 -> 8
        self.e_dim = e_dim
        self.n_toks = n_toks

        width *= scale
        height *= scale
        toksX, toksY = width // f, height // f
        sideX, sideY = toksX * f, toksY * f
        self.toksX, self.toksY = toksX, toksY

        z = self.rand_latent(codebook=codebook, device=device)
        super().__init__(sideX, sideY, z, ema_val)
        self.output_axes = ("n", "s", "y", "x")
        self.lr = 0.1
        self.latent_strength = 1

        self.register_buffer("llamagen_codebook", codebook, persistent=False)
        self.llamagen_decode = model.decode
        self.llamagen_encode = model.encode

    def clone(self):
        dummy = LlamaGenImage(*self.image_shape)
        with torch.no_grad():
            dummy.tensor.set_(self.tensor.clone())
            dummy.accum.set_(self.accum.clone())
            dummy.biased.set_(self.biased.clone())
            dummy.average.set_(self.average.clone())
            dummy.decay = self.decay
        return dummy

    def get_latent_tensor(self, detach=False, device=None):
        if device is None:
            device = self.device
        z = self.tensor
        if detach:
            z = z.detach()
        z_q = (
            vector_quantize(z, self.llamagen_codebook, l2_norm=True)
            .movedim(3, 1)
            .to(device)
        )
        return z_q

    @classmethod
    def get_preferred_loss(cls):
        from pytti.LossAug.LatentLossClass import LatentLoss

        return LatentLoss

    def decode(self, z, device=None):
        if device is None:
            device = self.device
        z_q = (
            vector_quantize(z, self.llamagen_codebook, l2_norm=True)
            .movedim(3, 1)
            .to(device)
        )
        out = self.llamagen_decode(z_q).add(1).div(2)
        return clamp_with_grad(out, 0, 1)

    @torch.no_grad()
    def encode_image(self, pil_image, device=None, **kwargs):
        if device is None:
            device = self.device
        pil_image = pil_image.resize(self.image_shape, Image.LANCZOS)
        pil_image = TF.to_tensor(pil_image)
        # VQModel.encode returns the quantized latent (BCHW, unit-norm rows)
        z, *_ = self.llamagen_encode(pil_image.unsqueeze(0).to(device) * 2 - 1)
        self.tensor.set_(z.movedim(1, 3))
        self.reset()

    @torch.no_grad()
    def make_latent(self, pil_image, device=None):
        if device is None:
            device = self.device
        pil_image = pil_image.resize(self.image_shape, Image.LANCZOS)
        pil_image = TF.to_tensor(pil_image)
        z, *_ = self.llamagen_encode(pil_image.unsqueeze(0).to(device) * 2 - 1)
        z_q = (
            vector_quantize(z.movedim(1, 3), self.llamagen_codebook, l2_norm=True)
            .movedim(3, 1)
            .to(device)
        )
        return z_q

    @torch.no_grad()
    def encode_random(self, init_spectrum="white", init_spectrum_falloff=1.0):
        require_white_init("LlamaGenImage", init_spectrum)
        self.tensor.set_(self.rand_latent())
        self.reset()

    def rand_latent(self, device=None, codebook=None):
        if device is None:
            device = self.device
        if codebook is None:
            codebook = self.llamagen_codebook
        n_toks = self.n_toks
        toksX, toksY = self.toksX, self.toksY
        one_hot = F.one_hot(
            torch.randint(n_toks, [toksY * toksX], device=device), n_toks
        ).to(codebook.dtype)
        z = one_hot @ codebook
        z = z.view([-1, toksY, toksX, self.e_dim])
        return z

    @staticmethod
    def init_llamagen(model_name, device=None):
        if device is None:
            device = default_device()
        global LLAMAGEN_MODEL, LLAMAGEN_NAME
        if LLAMAGEN_NAME == model_name:
            return
        if model_name not in LLAMAGEN_MODEL_NAMES:
            raise ValueError(
                f"LlamaGen model {model_name} is not supported. "
                f"Supported models are {LLAMAGEN_MODEL_NAMES}"
            )
        from huggingface_hub import hf_hub_download

        checkpoint_path = hf_hub_download(
            LLAMAGEN_HF_REPO,
            LLAMAGEN_CHECKPOINT_FILES[model_name],
            revision=LLAMAGEN_HF_REVISION,
        )
        logger.info(f"LlamaGen {model_name} checkpoint: {checkpoint_path}")
        model = load_llamagen_model(checkpoint_path, model_name)
        with vram_usage_mode("LlamaGen"):
            LLAMAGEN_MODEL = model.to(device)
        LLAMAGEN_NAME = model_name

    @staticmethod
    def free_llamagen():
        global LLAMAGEN_MODEL, LLAMAGEN_NAME
        LLAMAGEN_MODEL = None
        LLAMAGEN_NAME = None
        gc.collect()
        empty_cache()
