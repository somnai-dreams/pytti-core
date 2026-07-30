"""
Perceptor loading. open_clip is the single loader: the default tier is the
identical OpenAI CLIP weights (bit-comparability vs the retired openai/CLIP
package verified to ~6e-6), fetched from HF hub via open_clip's 'openai'
pretrained tags. OpenAI towers need the -quickgelu model configs — the plain
names use vanilla GELU and drift ~0.77 in embedding space.

Each loaded perceptor carries its own tokenizer, input resolution, and
normalization stats: ensemble members do not share preprocessing (SigLIP
uses mean=std=0.5; applying CLIP constants would silently skew gradients).
"""

import torch
from attrs import define
from loguru import logger
from torchvision import transforms

from pytti import vram_usage_mode
from pytti.device import default_device, memory_format_for

CLIP_PERCEPTORS = None

# config key -> (open_clip model name, pretrained tag)
PERCEPTOR_REGISTRY = {
    "ViTB32": ("ViT-B-32-quickgelu", "openai"),
    "ViTB16": ("ViT-B-16-quickgelu", "openai"),
    "ViTL14": ("ViT-L-14-quickgelu", "openai"),
    "ViTL14_336px": ("ViT-L-14-336-quickgelu", "openai"),
    "RN50": ("RN50-quickgelu", "openai"),
    "RN101": ("RN101-quickgelu", "openai"),
    "RN50x4": ("RN50x4-quickgelu", "openai"),
    "RN50x16": ("RN50x16-quickgelu", "openai"),
    "RN50x64": ("RN50x64-quickgelu", "openai"),
}

# names of the perceptor config keys currently loaded into CLIP_PERCEPTORS
CLIP_MODEL_NAMES = None


def _sanitize_for_config(in_str):
    # kept for compat with older callers/tests that mapped CLIP model names
    for char in ("/", "-"):
        in_str = in_str.replace(char, "")
    for char in "@":
        in_str = in_str.replace(char, "_")
    return in_str


@define(eq=False)
class LoadedPerceptor:
    """One ensemble member: model plus its own preprocessing contract."""

    key: str
    model: torch.nn.Module
    tokenizer: object
    cut_size: int
    normalize: transforms.Normalize

    def encode_image(self, batch: torch.Tensor) -> torch.Tensor:
        return self.model.encode_image(batch)

    def embed_text(self, text: str, device) -> torch.Tensor:
        tokens = self.tokenizer([text]).to(device)
        return self.model.encode_text(tokens).float()


class _ContiguousGrad(torch.autograd.Function):
    """Identity forward; forces the incoming gradient contiguous in backward."""

    @staticmethod
    def forward(ctx, x):
        return x

    @staticmethod
    def backward(ctx, grad):
        return grad.contiguous()


def _install_grad_fences(model):
    """
    MPS pathology fence: CLIP's class-token cat produces a narrowed
    (non-contiguous) gradient which, flowing back into conv2d's
    input-gradient kernel, is ~100x slower on MPS (measured 43s -> 0.4s for
    a 24x224px forward+backward on ViT-B/32). A one-node identity that makes
    the gradient contiguous at the conv/attention-pool boundary fixes it.
    Harmless on CUDA/CPU (one no-op autograd node).
    """
    fence = lambda module, inputs, output: _ContiguousGrad.apply(output)  # noqa: E731
    visual = model.visual
    if hasattr(visual, "conv1") and hasattr(visual, "transformer"):
        # ViT (OpenAI layout, shared by open_clip): fence between the
        # patchify conv and the token pipeline
        visual.conv1.register_forward_hook(fence)
    elif hasattr(visual, "attnpool"):
        # ModifiedResNet: fence between the convnet and the attention pool
        visual.layer4.register_forward_hook(fence)
    elif hasattr(visual, "trunk") and hasattr(visual.trunk, "patch_embed"):
        # timm-backed towers (SigLIP2 etc.): fence after the patch embed
        visual.trunk.patch_embed.register_forward_hook(fence)
    else:
        raise RuntimeError(
            f"Unknown visual tower layout for {type(visual).__name__} — a new "
            "perceptor architecture needs a conscious grad-fence decision "
            "(and an MPS fwd+bwd benchmark) before it can be enabled."
        )
    return model


def _load_perceptor(key: str, device) -> LoadedPerceptor:
    import open_clip

    model_name, pretrained = PERCEPTOR_REGISTRY[key]
    model, _, _ = open_clip.create_model_and_transforms(
        model_name, pretrained=pretrained
    )
    model = (
        model.eval()
        .requires_grad_(False)
        .to(device, memory_format=memory_format_for(device))
    )
    _install_grad_fences(model)

    cfg = open_clip.get_model_preprocess_cfg(model)
    mean, std = cfg.get("mean"), cfg.get("std")
    if mean is None or std is None:
        raise RuntimeError(
            f"open_clip returned no preprocess stats for {model_name!r} — "
            "refusing to guess normalization (silently wrong gradients)."
        )
    image_size = model.visual.image_size
    if isinstance(image_size, (tuple, list)):
        image_size = image_size[0]

    return LoadedPerceptor(
        key=key,
        model=model,
        tokenizer=open_clip.get_tokenizer(model_name),
        cut_size=int(image_size),
        normalize=transforms.Normalize(mean=mean, std=std),
    )


@vram_usage_mode("CLIP")
def init_clip(keys, device=None):
    if device is None:
        device = default_device()
    global CLIP_PERCEPTORS
    if CLIP_PERCEPTORS is None:
        CLIP_PERCEPTORS = [_load_perceptor(key, device) for key in keys]


def free_clip():
    global CLIP_PERCEPTORS
    CLIP_PERCEPTORS = None


def load_clip(params, device=None):
    """
    (Re)load the perceptor ensemble selected by the config's per-model flags.
    """
    if device is None:
        device = default_device()

    global CLIP_MODEL_NAMES
    last_names = CLIP_MODEL_NAMES if CLIP_MODEL_NAMES is not None else []
    CLIP_MODEL_NAMES = [key for key in PERCEPTOR_REGISTRY if params.get(key)]

    if CLIP_MODEL_NAMES == []:
        free_clip()
        raise RuntimeError("Please select at least one CLIP model")
    if last_names != CLIP_MODEL_NAMES or CLIP_PERCEPTORS is None:
        free_clip()
        logger.debug(f"Loading perceptors: {CLIP_MODEL_NAMES}")
        init_clip(CLIP_MODEL_NAMES, device=device)
        logger.debug("Perceptors loaded.")
