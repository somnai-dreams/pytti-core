"""
CLIP checkpoints -> cached MLX visual-tower weights.

Two source layouts, one target vocabulary (the ``vit.VisionTower`` tree):

- ``hf_clip`` — HF ``CLIPModel`` exports (``vision_model.encoder.layers.N.
  self_attn.{q,k,v}_proj`` ...): the classic OpenAI tier.
- ``open_clip`` — open_clip state dicts (``visual.transformer.resblocks.N.
  {ln_1, attn.in_proj_*, attn.out_proj, ln_2, mlp.c_fc, mlp.c_proj}`` ...):
  hf-hub checkpoints like FARE. qkv arrives pre-fused (``in_proj_*``), the
  text tower lives at the TOP level (no prefix), and ``visual.proj`` is a
  bare ``[hidden, output]`` matmul parameter that must be transposed into
  nn.Linear's ``[output, hidden]``.
- ``open_clip_timm`` — open_clip exports whose visual side is a timm trunk
  (``visual.trunk.{patch_embed.proj, pos_embed, blocks.N.{norm1, attn.qkv,
  attn.proj, norm2, mlp}, norm, attn_pool.*}``): the SigLIP2 tier. A
  DIFFERENT architecture (``siglip.SigLIPTower``: no class token, MAP
  attention-pool head), so this format pairs with ``SigLIPConfig``, not
  ``ViTConfig``. qkv arrives pre-fused; the text tower is ``text.``-prefixed.

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

from pytti.Perceptor.mlx_backend import SigLIPConfig, ViTConfig

# --------------------------------------------------------------------------
# registry: pytti perceptor key -> HF checkpoint + geometry
# --------------------------------------------------------------------------


SourceFormat = Literal["hf_clip", "open_clip", "open_clip_timm"]

# OpenAI CLIP preprocessing stats — shared by the classic tier AND the
# laion/FARE lineage (verified against chs20/FARE4-ViT-B-32's
# open_clip_config.json preprocess_cfg, 2026-08-03).
OPENAI_CLIP_MEAN: tuple[float, float, float] = (0.48145466, 0.4578275, 0.40821073)
OPENAI_CLIP_STD: tuple[float, float, float] = (0.26862954, 0.26130258, 0.27577711)
# SigLIP preprocessing: mean=std=0.5 — pinned torch-side too
# (pytti.Perceptor.EXPECTED_NORMALIZE guards open_clip issue #1068).
SIGLIP_MEAN: tuple[float, float, float] = (0.5, 0.5, 0.5)
SIGLIP_STD: tuple[float, float, float] = (0.5, 0.5, 0.5)


@dataclass(frozen=True)
class ModelSpec:
    repo_id: str
    # Which weights file the repo actually publishes (checked 2026-07-30;
    # FARE 2026-08-03; SigLIP2 2026-08-03): only clip-vit-large-patch14 has
    # model.safetensors, the other openai repos are pytorch_model.bin only,
    # and the chs20 FARE / timm SigLIP2 repos ship
    # open_clip_model.safetensors. Recorded explicitly per model rather
    # than probed at runtime.
    weights_file: str
    config: "ViTConfig | SigLIPConfig"
    # checkpoint key layout (see module docstring)
    source_format: SourceFormat = "hf_clip"
    # Normalization the image side must apply before the tower (same values
    # LoadedPerceptor.normalize carries; recorded here so MLX-only callers
    # don't need a torch perceptor to learn them).
    image_mean: tuple[float, float, float] = OPENAI_CLIP_MEAN
    image_std: tuple[float, float, float] = OPENAI_CLIP_STD


# Classic OpenAI tier + FARE (OpenAI ViT architecture throughout). Keys
# match pytti.Perceptor.PERCEPTOR_REGISTRY; the HF repos hold the same
# weights the torch-side open_clip tags load.
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
    # FARE adversarially-robust B/32 (Schlarmann et al., fine-tuned from
    # laion2B-s34B-b79K). open_clip layout; plain erf-GELU — open_clip
    # instantiates nn.GELU(approximate='none') for this checkpoint, NOT
    # QuickGELU (verified 2026-08-03 by loading it and inspecting the act
    # layer; the repo's config has no quick_gelu flag). heads = width//64
    # per open_clip's ViT default; mlp_dim = 4 * width.
    "FARE4ViTB32": ModelSpec(
        "chs20/FARE4-ViT-B-32-laion2B-s34B-b79K",
        "open_clip_model.safetensors",
        ViTConfig(
            image_size=224,
            patch_size=32,
            hidden_dim=768,
            num_layers=12,
            num_heads=12,
            mlp_dim=3072,
            output_dim=512,
            activation="gelu",
        ),
        source_format="open_clip",
    ),
    # SigLIP2 B/16 (timm trunk inside an open_clip export). Geometry from
    # the loaded module (timm 1.0.28, 2026-08-03): 196 tokens (no cls),
    # MAP attention-pool head, LN eps 1e-6, exact erf-GELU everywhere —
    # NOT the tanh form. output_dim == hidden_dim == 768 (timm_proj none).
    # Preprocessing is mean=std=0.5, not the CLIP constants.
    "SigLIP2B16": ModelSpec(
        "timm/ViT-B-16-SigLIP2",
        "open_clip_model.safetensors",
        SigLIPConfig(
            image_size=224,
            patch_size=16,
            hidden_dim=768,
            num_layers=12,
            num_heads=12,
            mlp_dim=3072,
        ),
        source_format="open_clip_timm",
        image_mean=SIGLIP_MEAN,
        image_std=SIGLIP_STD,
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
            f"{sorted(MLX_VIT_MODELS)} (RN towers and SigLIP2-SO400M "
            "stay on torch)."
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


@dataclass(frozen=True)
class TransposeToLinear:
    """2-D ``[in, out]`` parameter applied as ``x @ W`` (open_clip's
    ``visual.proj``) -> nn.Linear weight layout ``[out, in]``."""

    source: str


Transform = Copy | ConvToOHWI | ConcatRows | TransposeToLinear


def _sources_of(transform: Transform) -> tuple[str, ...]:
    if isinstance(transform, ConcatRows):
        return transform.sources
    return (transform.source,)


def _build_hf_clip_plan(config: ViTConfig) -> dict[str, Transform]:
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


def _build_open_clip_plan(config: ViTConfig) -> dict[str, Transform]:
    """Same target vocabulary, open_clip source layout (qkv pre-fused)."""
    plan: dict[str, Transform] = {
        "patch_embed.weight": ConvToOHWI("visual.conv1.weight"),
        "class_embedding": Copy("visual.class_embedding"),
        "positional_embedding": Copy("visual.positional_embedding"),
        "ln_pre.weight": Copy("visual.ln_pre.weight"),
        "ln_pre.bias": Copy("visual.ln_pre.bias"),
        "ln_post.weight": Copy("visual.ln_post.weight"),
        "ln_post.bias": Copy("visual.ln_post.bias"),
        # open_clip applies proj as `x @ proj` ([hidden, out]); nn.Linear
        # wants [out, hidden]
        "proj.weight": TransposeToLinear("visual.proj"),
    }
    for i in range(config.num_layers):
        src = f"visual.transformer.resblocks.{i}"
        dst = f"blocks.{i}"
        for wb in ("weight", "bias"):
            plan[f"{dst}.ln_1.{wb}"] = Copy(f"{src}.ln_1.{wb}")
            plan[f"{dst}.ln_2.{wb}"] = Copy(f"{src}.ln_2.{wb}")
            # torch MultiheadAttention ships qkv already row-fused in
            # (q, k, v) order — exactly our fused layout, plain copy
            plan[f"{dst}.attn.qkv.{wb}"] = Copy(f"{src}.attn.in_proj_{wb}")
            plan[f"{dst}.attn.out_proj.{wb}"] = Copy(
                f"{src}.attn.out_proj.{wb}"
            )
            plan[f"{dst}.mlp.fc1.{wb}"] = Copy(f"{src}.mlp.c_fc.{wb}")
            plan[f"{dst}.mlp.fc2.{wb}"] = Copy(f"{src}.mlp.c_proj.{wb}")
    return plan


def _build_timm_siglip_plan(config: SigLIPConfig) -> dict[str, Transform]:
    """Target parameter names (siglip.SigLIPTower tree) -> source recipe.

    Everything is a plain Copy or the conv OIHW->OHWI transpose: qkv ships
    pre-fused, ``pos_embed``/``attn_pool.latent`` keep their checkpoint
    shapes (``[1, N, D]`` / ``[1, 1, D]``) because the tower declares its
    parameters in those shapes.
    """
    plan: dict[str, Transform] = {
        "patch_embed.weight": ConvToOHWI("visual.trunk.patch_embed.proj.weight"),
        "patch_embed.bias": Copy("visual.trunk.patch_embed.proj.bias"),
        "positional_embedding": Copy("visual.trunk.pos_embed"),
        "ln_post.weight": Copy("visual.trunk.norm.weight"),
        "ln_post.bias": Copy("visual.trunk.norm.bias"),
        "attn_pool.latent": Copy("visual.trunk.attn_pool.latent"),
    }
    for name in ("q", "kv", "proj", "norm", "mlp.fc1", "mlp.fc2"):
        for wb in ("weight", "bias"):
            plan[f"attn_pool.{name}.{wb}"] = Copy(
                f"visual.trunk.attn_pool.{name}.{wb}"
            )
    for i in range(config.num_layers):
        src = f"visual.trunk.blocks.{i}"
        dst = f"blocks.{i}"
        for wb in ("weight", "bias"):
            plan[f"{dst}.ln_1.{wb}"] = Copy(f"{src}.norm1.{wb}")
            plan[f"{dst}.ln_2.{wb}"] = Copy(f"{src}.norm2.{wb}")
            plan[f"{dst}.attn.qkv.{wb}"] = Copy(f"{src}.attn.qkv.{wb}")
            plan[f"{dst}.attn.out_proj.{wb}"] = Copy(f"{src}.attn.proj.{wb}")
            plan[f"{dst}.mlp.fc1.{wb}"] = Copy(f"{src}.mlp.fc1.{wb}")
            plan[f"{dst}.mlp.fc2.{wb}"] = Copy(f"{src}.mlp.fc2.{wb}")
    return plan


# source_format -> (plan builder, the config type it understands). A
# format/config mismatch is a registry bug — caught loudly below rather
# than surfacing as a nonsense "missing keys" error.
_PLAN_BUILDERS = {
    "hf_clip": (_build_hf_clip_plan, ViTConfig),
    "open_clip": (_build_open_clip_plan, ViTConfig),
    "open_clip_timm": (_build_timm_siglip_plan, SigLIPConfig),
}


def expected_param_names(config: "ViTConfig | SigLIPConfig") -> frozenset[str]:
    """Every parameter name of the target module tree (converter's view).

    The target vocabulary is a function of the ARCHITECTURE (the config
    type), not the checkpoint layout: both ViT source formats cover the
    same VisionTower tree (tests assert this), and the timm format covers
    the SigLIPTower tree. Tower-side tests cross-check these names against
    the actual module parameter trees, so the files cannot drift apart
    silently.
    """
    if isinstance(config, SigLIPConfig):
        return frozenset(_build_timm_siglip_plan(config))
    if isinstance(config, ViTConfig):
        return frozenset(_build_hf_clip_plan(config))
    raise TypeError(
        f"unknown tower config type {type(config).__name__} "
        "(expected ViTConfig or SigLIPConfig)"
    )


# open_clip state dicts keep the text tower at the TOP level: these exact
# keys plus the (unprefixed) text transformer. attn_mask is the text causal
# mask — a buffer some exports persist.
_OPEN_CLIP_TEXT_KEYS = frozenset(
    {
        "positional_embedding",
        "text_projection",
        "logit_scale",
        "token_embedding.weight",
        "ln_final.weight",
        "ln_final.bias",
        "attn_mask",
    }
)


def _is_recognized_drop(key: str, source_format: SourceFormat) -> bool:
    # text towers drop in every format: M1 embeds text via torch
    if source_format == "open_clip":
        return key in _OPEN_CLIP_TEXT_KEYS or key.startswith("transformer.")
    if source_format == "open_clip_timm":
        # SigLIP2 exports: the whole text tower under "text.", plus the
        # sigmoid loss's scalar temperature and bias
        return key.startswith("text.") or key in ("logit_scale", "logit_bias")
    return (
        key.startswith("text_model.")
        or key in ("text_projection.weight", "logit_scale")
        or key == "vision_model.embeddings.position_ids"  # index buffer
    )


def plan_conversion(
    hf_keys: Iterable[str],
    config: "ViTConfig | SigLIPConfig",
    source_format: SourceFormat = "hf_clip",
) -> dict[str, Transform]:
    """
    Build the conversion plan and reconcile it against the actual checkpoint
    keys — loud in both directions.
    """
    entry = _PLAN_BUILDERS.get(source_format)
    if entry is None:
        raise ValueError(
            f"unknown source_format {source_format!r} "
            f"(expected one of {sorted(_PLAN_BUILDERS)})"
        )
    builder, config_type = entry
    if not isinstance(config, config_type):
        raise TypeError(
            f"source_format {source_format!r} maps a {config_type.__name__} "
            f"tower, got {type(config).__name__} — registry entry is "
            "mismatched"
        )
    plan = builder(config)
    needed = {s for transform in plan.values() for s in _sources_of(transform)}
    have = set(hf_keys)

    missing = sorted(needed - have)
    if missing:
        raise ValueError(
            f"checkpoint is missing {len(missing)} expected vision keys "
            f"(wrong repo, geometry, or source_format?): {missing[:5]}..."
        )
    leftover = sorted(
        k for k in have - needed if not _is_recognized_drop(k, source_format)
    )
    if leftover:
        raise ValueError(
            f"checkpoint has {len(leftover)} keys this converter does not "
            f"understand — refusing to silently drop weights: {leftover[:5]}..."
        )
    return plan


# --------------------------------------------------------------------------
# applying (mlx + huggingface_hub, lazy imports — darwin path only)
# --------------------------------------------------------------------------


def _build_tower(config: "ViTConfig | SigLIPConfig", *, ln_fp32: bool = False):
    """The (weightless) tower module matching a config's architecture.

    Dispatches on the config type — the same axis ``expected_param_names``
    uses, so the strict-load validation below always checks against the
    tree the plan was built for. Imports mlx lazily via the submodules.
    """
    if isinstance(config, SigLIPConfig):
        from pytti.Perceptor.mlx_backend.siglip import SigLIPTower

        return SigLIPTower(config, ln_fp32=ln_fp32)
    if isinstance(config, ViTConfig):
        from pytti.Perceptor.mlx_backend.vit import VisionTower

        return VisionTower(config, ln_fp32=ln_fp32)
    raise TypeError(
        f"unknown tower config type {type(config).__name__} "
        "(expected ViTConfig or SigLIPConfig)"
    )


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
    plan = plan_conversion(hf_weights.keys(), spec.config, spec.source_format)

    target_dtype = getattr(mx, dtype)
    converted: dict[str, mx.array] = {}
    for target, transform in plan.items():
        if isinstance(transform, Copy):
            array = hf_weights[transform.source]
        elif isinstance(transform, ConvToOHWI):
            array = hf_weights[transform.source].transpose(0, 2, 3, 1)
        elif isinstance(transform, TransposeToLinear):
            array = hf_weights[transform.source].transpose(1, 0)
        elif isinstance(transform, ConcatRows):
            array = mx.concatenate(
                [hf_weights[s] for s in transform.sources], axis=0
            )
        else:  # exhaustiveness over the Transform union
            raise TypeError(f"unhandled transform {transform!r}")
        converted[target] = array.astype(target_dtype)

    # strict-load into a throwaway tower BEFORE caching: validates every
    # shape against the module tree, so the cache is trustworthy by
    # construction.
    _build_tower(spec.config).load_weights(list(converted.items()), strict=True)

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
    Build the tower for `key` (``vit.VisionTower`` or ``siglip.SigLIPTower``,
    by the registry config's type) with real weights, converting and caching
    on first use. `dtype` is the compute/weight dtype; `ln_fp32` is the
    plan's gate-2 fallback knob (see vit.py docstring).

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

    tower = _build_tower(_spec_for(key).config, ln_fp32=ln_fp32)
    tower.load_weights(str(path), strict=True)
    if dtype == "bfloat16":
        tower.set_dtype(mx.bfloat16)
    tower.freeze()
    return tower
