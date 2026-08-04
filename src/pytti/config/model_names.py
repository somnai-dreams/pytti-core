"""
Model-name constants shared by the config schema and the model loaders.
Deliberately import-free so the schema can be loaded without torch/taming.
"""

VQGAN_MODEL_NAMES = [
    "imagenet",
    "coco",
    "wikiart",
    "sflickr",
    "openimages",
    "faceshq",
]

# Historical config spelling: "sflckr" was the original (typo'd) name for the
# S-FLCKR checkpoint. Accept it as an alias when validating/loading.
VQGAN_MODEL_ALIASES = {"sflckr": "sflickr"}

# LlamaGen VQ tokenizer variants (image_model: LlamaGen). ds16 is the
# default look (f=16, same toks math as taming f16); ds8 (f=8) is the
# finer-texture second look at 4x the token count per canvas.
LLAMAGEN_MODEL_NAMES = [
    "ds16",
    "ds8",
]

# Checkpoint filenames inside the FoundationVision/LlamaGen HF repo
# (revision pinned in pytti/image_models/llamagen.py).
LLAMAGEN_CHECKPOINT_FILES = {
    "ds16": "vq_ds16_c2i.pt",
    "ds8": "vq_ds8_c2i.pt",
}
