"""
HF CLIP checkpoints -> cached MLX visual-tower weights.

Two halves:

- **Planning** (`plan_conversion`, `expected_param_names`, `MLX_VIT_MODELS`)
  is pure Python — importable and testable on any platform, no mlx, no
  downloads. The plan is a bidirectional, total mapping: every parameter of
  the target module tree gets exactly one source recipe, and every
  checkpoint key is either consumed or explicitly recognized as dropped.
  Anything else raises — a silently ignored tensor is a silently wrong
  gradient.
- **Applying** (`convert_and_cache`, `load_tower`) imports mlx (and
  huggingface_hub) lazily inside the functions; darwin-only in practice.

Conversion follows the mlx-examples clip pattern:

- drop ``position_ids`` buffers (index arrays, not weights) and the whole
  text tower (M1 embeds text on the torch side),
- transpose the patch-embed conv OIHW -> OHWI (MLX convs are NHWC),
- fuse the q/k/v projections into one qkv matmul via row-concatenation,
- cast to fp16 (fp32 kept as a debugging option; bf16 is derived from the
  fp32 cache at load time — deriving it from fp16 would double-round).

Converted weights are cached at ``~/.cache/pytti/mlx/<model>/`` with an
atomic write (tmp file + ``os.replace``), keyed by pytti's perceptor config
names (``ViTB32`` etc., same vocabulary as ``PERCEPTOR_REGISTRY``).
"""

import os
import tempfile
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from pytti.Perceptor.mlx_backend import ViTConfig

# --------------------------------------------------------------------------
# registry: pytti perceptor key -> HF checkpoint + geometry
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class ModelSpec:
    repo_id: str
    # Which weights file the repo actually publishes (checked 2026-07-30):
    # only clip-vit-large-patch14 has model.safetensors; the rest are
    # pytorch_model.bin only. Recorded explicitly per model rather than
    # probed at runtime.
    weights_file: str
    config: ViTConfig


# Classic OpenAI tier only (the plan's M1 scope). Keys match
# pytti.Perceptor.PERCEPTOR_REGISTRY; the HF repos hold the same OpenAI
# weights open_clip's "openai" pretrained tags load.
MLX_VIT_MODELS: dict[str, ModelSpec] = {
    "ViTB32": ModelSpec(
        "openai/clip-vit-base-patch32",
        "pytorch_model.bin",
        ViTConfig(
            image_size=224,
            patch_size=32,
            hidden_dim=768,
            num_layers=12,
            num_heads=12,
            mlp_dim=3072,
            output_dim=512,
        ),
    ),
    "ViTB16": ModelSpec(
        "openai/clip-vit-base-patch16",
        "pytorch_model.bin",
        ViTConfig(
            image_size=224,
            patch_size=16,
            hidden_dim=768,
            num_layers=12,
            num_heads=12,
            mlp_dim=3072,
            output_dim=512,
        ),
    ),
    "ViTL14": ModelSpec(
        "openai/clip-vit-large-patch14",
        "model.safetensors",
        ViTConfig(
            image_size=224,
            patch_size=14,
            hidden_dim=1024,
            num_layers=24,
            num_heads=16,
            mlp_dim=4096,
            output_dim=768,
        ),
    ),
    "ViTL14_336px": ModelSpec(
        "openai/clip-vit-large-patch14-336",
        "pytorch_model.bin",
        ViTConfig(
            image_size=336,
            patch_size=14,
            hidden_dim=1024,
            num_layers=24,
            num_heads=16,
            mlp_dim=4096,
            output_dim=768,
        ),
    ),
}

CacheDtype = Literal["float16", "float32"]
TowerDtype = Literal["float16", "bfloat16", "float32"]

_CACHE_VERSION = "v1"
_DEFAULT_CACHE_ROOT = Path.home() / ".cache" / "pytti" / "mlx"


def _spec_for(key: str) -> ModelSpec:
    spec = MLX_VIT_MODELS.get(key)
    if spec is None:
        raise KeyError(
            f"{key!r} has no MLX conversion. Available: "
            f"{sorted(MLX_VIT_MODELS)} (classic OpenAI ViT tier only in M1; "
            "RN/SigLIP2/FARE towers stay on torch)."
        )
    return spec


# --------------------------------------------------------------------------
# planning (pure: names only, no tensors, no mlx)
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Copy:
    source: str


@dataclass(frozen=True)
class ConvToOHWI:
    """Patch-embed conv weight: HF OIHW -> MLX OHWI."""

    source: str


@dataclass(frozen=True)
class ConcatRows:
    """Row-concatenate (axis 0) — fuses HF q/k/v into one qkv projection."""

    sources: tuple[str, ...]


Transform = Copy | ConvToOHWI | ConcatRows


def _sources_of(transform: Transform) -> tuple[str, ...]:
    if isinstance(transform, ConcatRows):
        return transform.sources
    return (transform.source,)


def _build_plan(config: ViTConfig) -> dict[str, Transform]:
    """Target parameter name (vit.VisionTower tree) -> source recipe."""
    plan: dict[str, Transform] = {
        "patch_embed.weight": ConvToOHWI(
            "vision_model.embeddings.patch_embedding.weight"
        ),
        "class_embedding": Copy("vision_model.embeddings.class_embedding"),
        "positional_embedding": Copy(
            "vision_model.embeddings.position_embedding.weight"
        ),
        # "pre_layrnorm" [sic]: the typo is in the HF checkpoints themselves
        "ln_pre.weight": Copy("vision_model.pre_layrnorm.weight"),
        "ln_pre.bias": Copy("vision_model.pre_layrnorm.bias"),
        "ln_post.weight": Copy("vision_model.post_layernorm.weight"),
        "ln_post.bias": Copy("vision_model.post_layernorm.bias"),
        "proj.weight": Copy("visual_projection.weight"),
    }
    for i in range(config.num_layers):
        src = f"vision_model.encoder.layers.{i}"
        dst = f"blocks.{i}"
        for wb in ("weight", "bias"):
            plan[f"{dst}.ln_1.{wb}"] = Copy(f"{src}.layer_norm1.{wb}")
            plan[f"{dst}.ln_2.{wb}"] = Copy(f"{src}.layer_norm2.{wb}")
            plan[f"{dst}.attn.qkv.{wb}"] = ConcatRows(
                tuple(f"{src}.self_attn.{p}_proj.{wb}" for p in ("q", "k", "v"))
            )
            plan[f"{dst}.attn.out_proj.{wb}"] = Copy(
                f"{src}.self_attn.out_proj.{wb}"
            )
            plan[f"{dst}.mlp.fc1.{wb}"] = Copy(f"{src}.mlp.fc1.{wb}")
            plan[f"{dst}.mlp.fc2.{wb}"] = Copy(f"{src}.mlp.fc2.{wb}")
    return plan


def expected_param_names(config: ViTConfig) -> frozenset[str]:
    """Every parameter name of the target module tree (converter's view).

    vit-side tests cross-check this against the actual VisionTower parameter
    tree, so the two files cannot drift apart silently.
    """
    return frozenset(_build_plan(config))


def _is_recognized_drop(key: str) -> bool:
    return (
        key.startswith("text_model.")  # M1 embeds text via torch
        or key in ("text_projection.weight", "logit_scale")
        or key == "vision_model.embeddings.position_ids"  # index buffer
    )


def plan_conversion(
    hf_keys: Iterable[str], config: ViTConfig
) -> dict[str, Transform]:
    """
    Build the conversion plan and reconcile it against the actual checkpoint
    keys — loud in both directions.
    """
    plan = _build_plan(config)
    needed = {s for transform in plan.values() for s in _sources_of(transform)}
    have = set(hf_keys)

    missing = sorted(needed - have)
    if missing:
        raise ValueError(
            f"checkpoint is missing {len(missing)} expected vision keys "
            f"(wrong repo or geometry?): {missing[:5]}..."
        )
    leftover = sorted(k for k in have - needed if not _is_recognized_drop(k))
    if leftover:
        raise ValueError(
            f"checkpoint has {len(leftover)} keys this converter does not "
            f"understand — refusing to silently drop weights: {leftover[:5]}..."
        )
    return plan


# --------------------------------------------------------------------------
# applying (mlx + huggingface_hub, lazy imports — darwin path only)
# --------------------------------------------------------------------------


def _cache_path(key: str, dtype: CacheDtype, cache_root: Path | None) -> Path:
    root = _DEFAULT_CACHE_ROOT if cache_root is None else cache_root
    return root / key / f"visual.{dtype}.{_CACHE_VERSION}.safetensors"


def _load_checkpoint_arrays(spec: ModelSpec) -> dict:
    """Download and read the HF checkpoint as {hf_key: mx.array} (fp32)."""
    import mlx.core as mx
    from huggingface_hub import hf_hub_download

    path = hf_hub_download(spec.repo_id, spec.weights_file)
    if spec.weights_file.endswith(".safetensors"):
        return mx.load(path)
    # legacy pickle checkpoint (three of the four openai repos publish no
    # safetensors); torch is a core dependency, weights_only keeps the
    # unpickling to plain tensors
    import torch

    state = torch.load(path, map_location="cpu", weights_only=True)
    return {key: mx.array(value.numpy()) for key, value in state.items()}


def convert_and_cache(
    key: str,
    dtype: CacheDtype = "float16",
    cache_root: Path | None = None,
) -> Path:
    """
    Ensure the converted weights for `key` exist on disk; return their path.
    Downloads the HF checkpoint on first use. The write is atomic (tmp file
    + os.replace), so a crashed conversion never leaves a half-written cache.
    """
    if dtype not in ("float16", "float32"):
        raise ValueError(f"cache dtype must be float16 or float32, got {dtype!r}")
    spec = _spec_for(key)
    path = _cache_path(key, dtype, cache_root)
    if path.exists():
        return path

    import mlx.core as mx

    hf_weights = _load_checkpoint_arrays(spec)
    plan = plan_conversion(hf_weights.keys(), spec.config)

    target_dtype = getattr(mx, dtype)
    converted: dict[str, mx.array] = {}
    for target, transform in plan.items():
        if isinstance(transform, Copy):
            array = hf_weights[transform.source]
        elif isinstance(transform, ConvToOHWI):
            array = hf_weights[transform.source].transpose(0, 2, 3, 1)
        else:
            array = mx.concatenate(
                [hf_weights[s] for s in transform.sources], axis=0
            )
        converted[target] = array.astype(target_dtype)

    # strict-load into a throwaway tower BEFORE caching: validates every
    # shape against the module tree, so the cache is trustworthy by
    # construction.
    from pytti.Perceptor.mlx_backend.vit import VisionTower

    VisionTower(spec.config).load_weights(list(converted.items()), strict=True)

    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(
        dir=path.parent, prefix=".tmp-", suffix=".safetensors"
    )
    os.close(fd)
    try:
        mx.save_safetensors(
            tmp,
            converted,
            metadata={
                "format": "mlx",
                "pytti_cache_version": _CACHE_VERSION,
                "source_repo": spec.repo_id,
                "dtype": dtype,
            },
        )
        os.replace(tmp, path)
    except BaseException:
        os.unlink(tmp)
        raise
    return path


def load_tower(
    key: str,
    dtype: TowerDtype = "float16",
    *,
    ln_fp32: bool = False,
    cache_root: Path | None = None,
):
    """
    Build a `vit.VisionTower` for `key` with real weights, converting and
    caching on first use. `dtype` is the compute/weight dtype; `ln_fp32` is
    the plan's gate-2 fallback knob (see vit.py docstring).

    Returns the tower with weights frozen (they are inference constants —
    gradients flow w.r.t. the input only).
    """
    if dtype not in ("float16", "bfloat16", "float32"):
        raise ValueError(
            f"dtype must be float16, bfloat16, or float32, got {dtype!r}"
        )
    # bf16 has fewer mantissa bits than fp16 — derive it from the fp32 cache
    # rather than double-rounding through fp16.
    cache_dtype: CacheDtype = "float32" if dtype == "bfloat16" else dtype
    path = convert_and_cache(key, cache_dtype, cache_root)

    import mlx.core as mx

    from pytti.Perceptor.mlx_backend.vit import VisionTower

    tower = VisionTower(_spec_for(key).config, ln_fp32=ln_fp32)
    tower.load_weights(str(path), strict=True)
    if dtype == "bfloat16":
        tower.set_dtype(mx.bfloat16)
    tower.freeze()
    return tower
