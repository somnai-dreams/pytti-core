import torch
from clip import clip
from loguru import logger

from pytti import vram_usage_mode
from pytti.device import default_device, memory_format_for

CLIP_PERCEPTORS = None


def _sanitize_for_config(in_str):
    for char in ("/", "-"):
        in_str = in_str.replace(char, "")
    for char in "@":
        in_str = in_str.replace(char, "_")
    return in_str


# config key -> CLIP model name, e.g. {"ViTB32": "ViT-B/32"}
SUPPORTED_CLIP_MODELS = {
    _sanitize_for_config(model_name): model_name
    for model_name in clip.available_models()
}

# names of the CLIP models currently loaded into CLIP_PERCEPTORS
CLIP_MODEL_NAMES = None


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
        # ViT: fence between the patchify conv and the token pipeline
        visual.conv1.register_forward_hook(fence)
    elif hasattr(visual, "attnpool"):
        # ModifiedResNet: fence between the convnet and the attention pool
        visual.layer4.register_forward_hook(fence)
    return model


# this should probably be a method on the multiperceptor guide
@vram_usage_mode("CLIP")
def init_clip(clip_models, device=None):
    if device is None:
        device = default_device()
    global CLIP_PERCEPTORS
    if CLIP_PERCEPTORS is None:
        CLIP_PERCEPTORS = [
            _install_grad_fences(
                clip.load(model, jit=False)[0]
                .eval()
                .requires_grad_(False)
                .to(device, memory_format=memory_format_for(device))
            )
            for model in clip_models
        ]


def free_clip():
    global CLIP_PERCEPTORS
    CLIP_PERCEPTORS = None


def load_clip(params, device=None):
    """
    (Re)load the CLIP ensemble selected by the config's per-model flags.
    """
    if device is None:
        device = default_device()

    global CLIP_MODEL_NAMES
    last_names = CLIP_MODEL_NAMES if CLIP_MODEL_NAMES is not None else []
    CLIP_MODEL_NAMES = [
        clip_name
        for config_name, clip_name in SUPPORTED_CLIP_MODELS.items()
        if params.get(config_name)
    ]

    if CLIP_MODEL_NAMES == []:
        free_clip()
        raise RuntimeError("Please select at least one CLIP model")
    if last_names != CLIP_MODEL_NAMES or CLIP_PERCEPTORS is None:
        free_clip()
        logger.debug("Loading CLIP...")
        init_clip(CLIP_MODEL_NAMES, device=device)
        logger.debug("CLIP loaded.")
