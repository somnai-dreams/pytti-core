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


# this should probably be a method on the multiperceptor guide
@vram_usage_mode("CLIP")
def init_clip(clip_models, device=None):
    if device is None:
        device = default_device()
    global CLIP_PERCEPTORS
    if CLIP_PERCEPTORS is None:
        CLIP_PERCEPTORS = [
            clip.load(model, jit=False)[0]
            .eval()
            .requires_grad_(False)
            .to(device, memory_format=memory_format_for(device))
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
