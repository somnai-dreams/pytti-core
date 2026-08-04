"""
MLX (Metal) perceptor backend — phase M1 of docs/mlx-port-plan.md.

Layout:
- ``convert``: HF CLIP checkpoint -> MLX weight cache. Its planning half
  (name mapping, registry) is pure Python; only the download/tensor half
  touches mlx.
- ``vit``: the CLIP visual tower in mlx.nn. Imports mlx at module level.

``mlx`` is a darwin-only dependency. Importing this package — and the
conversion *planning* code — is safe on any platform; everything that needs
mlx is imported lazily via ``__getattr__`` below or inside function bodies,
so the linux CI suite never touches it.

``ViTConfig`` lives here (not in ``vit``) because it is the shared, mlx-free
vocabulary of the package: the conversion planner and the tower both derive
their structure from it.
"""

from dataclasses import dataclass
from typing import Literal

# MLP activation of the tower. "quickgelu" is x * sigmoid(1.702 x) — the
# OpenAI checkpoints. "gelu" is the EXACT erf form, x * Phi(x) — what
# open_clip's plain (non-quickgelu) model configs instantiate as torch
# ``nn.GELU(approximate='none')`` for the laion/FARE lineage. NOT the tanh
# approximation: a wrong GELU variant is the QuickGELU-drift bug again.
Activation = Literal["quickgelu", "gelu"]


@dataclass(frozen=True)
class ViTConfig:
    """Geometry of one OpenAI-architecture CLIP visual tower."""

    image_size: int
    patch_size: int
    hidden_dim: int
    num_layers: int
    num_heads: int
    mlp_dim: int
    output_dim: int
    layer_norm_eps: float = 1e-5
    activation: Activation = "quickgelu"

    def __post_init__(self):
        if self.image_size % self.patch_size:
            raise ValueError(
                f"image_size {self.image_size} not divisible by "
                f"patch_size {self.patch_size}"
            )
        if self.hidden_dim % self.num_heads:
            raise ValueError(
                f"hidden_dim {self.hidden_dim} not divisible by "
                f"num_heads {self.num_heads}"
            )
        if self.activation not in ("quickgelu", "gelu"):
            raise ValueError(
                f"activation must be 'quickgelu' or 'gelu', "
                f"got {self.activation!r}"
            )

    @property
    def grid_size(self) -> int:
        return self.image_size // self.patch_size

    @property
    def num_tokens(self) -> int:
        return self.grid_size * self.grid_size + 1


@dataclass(frozen=True)
class SigLIPConfig:
    """Geometry of one SigLIP-architecture visual tower (timm layout).

    Structurally distinct from the OpenAI ViT: patch conv WITH bias, no
    class token (position embeddings cover the patch grid only, no ln_pre),
    final LayerNorm over ALL tokens, and a MAP attention-pooling head
    (latent probe -> multi-head attention over the tokens -> residual MLP)
    instead of cls-token + linear projection. The pooled output dimension
    IS the hidden dimension (timm_proj="none" in the open_clip config) —
    ``output_dim`` is derived, not stored, so it can never disagree.

    Activation and eps verified EMPIRICALLY against the loaded torch module
    (timm 1.0.28, 2026-08-03): every GELU is ``approximate='none'`` (exact
    erf form — NOT the tanh form HF's SiglipVisionModel config suggests),
    LayerNorm eps 1e-6 throughout.
    """

    image_size: int
    patch_size: int
    hidden_dim: int
    num_layers: int
    num_heads: int
    mlp_dim: int
    layer_norm_eps: float = 1e-6
    activation: Activation = "gelu"

    def __post_init__(self):
        if self.image_size % self.patch_size:
            raise ValueError(
                f"image_size {self.image_size} not divisible by "
                f"patch_size {self.patch_size}"
            )
        if self.hidden_dim % self.num_heads:
            raise ValueError(
                f"hidden_dim {self.hidden_dim} not divisible by "
                f"num_heads {self.num_heads}"
            )
        if self.activation not in ("quickgelu", "gelu"):
            raise ValueError(
                f"activation must be 'quickgelu' or 'gelu', "
                f"got {self.activation!r}"
            )

    @property
    def output_dim(self) -> int:
        return self.hidden_dim

    @property
    def grid_size(self) -> int:
        return self.image_size // self.patch_size

    @property
    def num_tokens(self) -> int:
        # no class token in this architecture
        return self.grid_size * self.grid_size


# lazy re-exports: names -> submodule. Keeps `import pytti.Perceptor.
# mlx_backend` mlx-free (vit/siglip import mlx at module level).
_LAZY_EXPORTS = {
    "VisionTower": "vit",
    "SigLIPTower": "siglip",
    "MLX_VIT_MODELS": "convert",
    "ModelSpec": "convert",
    "convert_and_cache": "convert",
    "expected_param_names": "convert",
    "load_tower": "convert",
    "plan_conversion": "convert",
}

__all__ = ["SigLIPConfig", "ViTConfig", *sorted(_LAZY_EXPORTS)]


def __getattr__(name: str):
    submodule = _LAZY_EXPORTS.get(name)
    if submodule is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    import importlib

    return getattr(importlib.import_module(f"{__name__}.{submodule}"), name)
