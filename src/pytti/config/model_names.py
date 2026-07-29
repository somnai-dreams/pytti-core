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
