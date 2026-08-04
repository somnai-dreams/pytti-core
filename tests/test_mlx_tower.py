"""
MLX perceptor backend (docs/mlx-port-plan.md, phase M1).

Three tiers:
- conversion-plan tests: pure Python, run everywhere (linux CI included) —
  no mlx, no downloads;
- tower tests: need mlx (darwin), random weights, no downloads;
- parity tests: need mlx AND real checkpoints -> @pytest.mark.download
  (excluded from the default suite by addopts).
"""

import dataclasses
import importlib.util
from typing import cast

import pytest

from pytti.Perceptor.mlx_backend import SigLIPConfig, ViTConfig
from pytti.Perceptor.mlx_backend.convert import (
    MLX_VIT_MODELS,
    OPENAI_CLIP_MEAN,
    OPENAI_CLIP_STD,
    ConcatRows,
    ConvToOHWI,
    Copy,
    SourceFormat,
    TransposeToLinear,
    expected_param_names,
    plan_conversion,
)

needs_mlx = pytest.mark.skipif(
    importlib.util.find_spec("mlx") is None,
    reason="mlx not installed (darwin-only backend)",
)

TINY = ViTConfig(
    image_size=16,
    patch_size=8,
    hidden_dim=8,
    num_layers=2,
    num_heads=2,
    mlp_dim=32,
    output_dim=6,
)


def synthetic_hf_keys(num_layers: int) -> list[str]:
    """The key set of an HF CLIPModel checkpoint (vision side exact; text
    side and buffers representative of the recognized-drop classes)."""
    keys = [
        "vision_model.embeddings.class_embedding",
        "vision_model.embeddings.patch_embedding.weight",
        "vision_model.embeddings.position_embedding.weight",
        "vision_model.embeddings.position_ids",
        "vision_model.pre_layrnorm.weight",  # [sic] — HF checkpoint typo
        "vision_model.pre_layrnorm.bias",
        "vision_model.post_layernorm.weight",
        "vision_model.post_layernorm.bias",
        "visual_projection.weight",
        "text_projection.weight",
        "logit_scale",
        "text_model.embeddings.token_embedding.weight",
        "text_model.encoder.layers.0.self_attn.q_proj.weight",
        "text_model.final_layer_norm.weight",
    ]
    for i in range(num_layers):
        base = f"vision_model.encoder.layers.{i}"
        for wb in ("weight", "bias"):
            keys += [
                f"{base}.layer_norm1.{wb}",
                f"{base}.layer_norm2.{wb}",
                f"{base}.self_attn.q_proj.{wb}",
                f"{base}.self_attn.k_proj.{wb}",
                f"{base}.self_attn.v_proj.{wb}",
                f"{base}.self_attn.out_proj.{wb}",
                f"{base}.mlp.fc1.{wb}",
                f"{base}.mlp.fc2.{wb}",
            ]
    return keys


def synthetic_open_clip_keys(num_layers: int) -> list[str]:
    """The key set of an open_clip state dict (vision side exact; text side
    — which lives at the TOP level in this layout — representative)."""
    keys = [
        "visual.class_embedding",
        "visual.conv1.weight",
        "visual.positional_embedding",
        "visual.ln_pre.weight",
        "visual.ln_pre.bias",
        "visual.ln_post.weight",
        "visual.ln_post.bias",
        "visual.proj",
        # text tower: unprefixed top-level keys + text transformer
        "positional_embedding",
        "text_projection",
        "logit_scale",
        "token_embedding.weight",
        "ln_final.weight",
        "ln_final.bias",
        "attn_mask",
        "transformer.resblocks.0.attn.in_proj_weight",
        "transformer.resblocks.0.mlp.c_fc.weight",
    ]
    for i in range(num_layers):
        base = f"visual.transformer.resblocks.{i}"
        for wb in ("weight", "bias"):
            keys += [
                f"{base}.ln_1.{wb}",
                f"{base}.ln_2.{wb}",
                f"{base}.attn.in_proj_{wb}",  # qkv arrives pre-fused
                f"{base}.attn.out_proj.{wb}",
                f"{base}.mlp.c_fc.{wb}",
                f"{base}.mlp.c_proj.{wb}",
            ]
    return keys


class TestConversionPlan:
    """Pure structure tests — run on any platform, no mlx."""

    def test_plan_covers_module_tree_exactly(self):
        plan = plan_conversion(synthetic_hf_keys(TINY.num_layers), TINY)
        assert set(plan) == set(expected_param_names(TINY))

    def test_qkv_fusion_consumes_separate_projections(self):
        plan = plan_conversion(synthetic_hf_keys(TINY.num_layers), TINY)
        qkv = plan["blocks.0.attn.qkv.weight"]
        assert isinstance(qkv, ConcatRows)
        assert qkv.sources == (
            "vision_model.encoder.layers.0.self_attn.q_proj.weight",
            "vision_model.encoder.layers.0.self_attn.k_proj.weight",
            "vision_model.encoder.layers.0.self_attn.v_proj.weight",
        )

    def test_missing_vision_key_fails_loud(self):
        keys = synthetic_hf_keys(TINY.num_layers)
        keys.remove("vision_model.post_layernorm.bias")
        with pytest.raises(ValueError, match="missing"):
            plan_conversion(keys, TINY)

    def test_unrecognized_key_fails_loud(self):
        keys = [*synthetic_hf_keys(TINY.num_layers), "vision_model.mystery.weight"]
        with pytest.raises(ValueError, match="does not\\s+understand"):
            plan_conversion(keys, TINY)

    def test_geometry_mismatch_fails_loud(self):
        # checkpoint has 2 layers, config claims 3 -> layer 2 keys missing
        bigger = ViTConfig(
            image_size=16,
            patch_size=8,
            hidden_dim=8,
            num_layers=3,
            num_heads=2,
            mlp_dim=32,
            output_dim=6,
        )
        with pytest.raises(ValueError, match="missing"):
            plan_conversion(synthetic_hf_keys(2), bigger)

    def test_position_ids_absence_is_tolerated(self):
        # newer transformers exports omit the position_ids buffer
        keys = synthetic_hf_keys(TINY.num_layers)
        keys.remove("vision_model.embeddings.position_ids")
        plan_conversion(keys, TINY)  # must not raise

    def test_registry_geometries(self):
        # OpenAI ViT: 8 top-level params + 12 per block; tokens = (S/P)^2+1.
        # SigLIP: 18 top-level (conv bias, no cls/ln_pre/proj, 13-param MAP
        # head) + 12 per block; tokens = (S/P)^2 (no class token).
        for key, spec in MLX_VIT_MODELS.items():
            cfg = spec.config
            top = 18 if isinstance(cfg, SigLIPConfig) else 8
            assert len(expected_param_names(cfg)) == top + 12 * cfg.num_layers, key
        assert MLX_VIT_MODELS["ViTB32"].config.num_tokens == 50
        assert MLX_VIT_MODELS["ViTB16"].config.num_tokens == 197
        assert MLX_VIT_MODELS["ViTL14_336px"].config.num_tokens == 577
        assert MLX_VIT_MODELS["SigLIP2B16"].config.num_tokens == 196

    def test_config_validates_divisibility(self):
        with pytest.raises(ValueError, match="divisible"):
            ViTConfig(
                image_size=225,
                patch_size=32,
                hidden_dim=8,
                num_layers=1,
                num_heads=2,
                mlp_dim=32,
                output_dim=6,
            )

    def test_config_validates_activation(self):
        with pytest.raises(ValueError, match="activation"):
            dataclasses.replace(TINY, activation="gelu_tanh")


class TestOpenClipConversionPlan:
    """open_clip source layout (FARE lineage) — pure structure, no mlx."""

    def test_plan_covers_module_tree_exactly(self):
        # same target vocabulary as the hf_clip plan — the tower cannot
        # tell which layout its weights came from
        plan = plan_conversion(
            synthetic_open_clip_keys(TINY.num_layers), TINY, "open_clip"
        )
        assert set(plan) == set(expected_param_names(TINY))

    def test_qkv_arrives_prefused(self):
        plan = plan_conversion(
            synthetic_open_clip_keys(TINY.num_layers), TINY, "open_clip"
        )
        assert plan["blocks.0.attn.qkv.weight"] == Copy(
            "visual.transformer.resblocks.0.attn.in_proj_weight"
        )
        assert plan["blocks.0.attn.qkv.bias"] == Copy(
            "visual.transformer.resblocks.0.attn.in_proj_bias"
        )

    def test_proj_is_transposed_into_linear_layout(self):
        # visual.proj is [hidden, out] used as x @ W; nn.Linear is [out, hidden]
        plan = plan_conversion(
            synthetic_open_clip_keys(TINY.num_layers), TINY, "open_clip"
        )
        assert plan["proj.weight"] == TransposeToLinear("visual.proj")

    def test_missing_vision_key_fails_loud(self):
        keys = synthetic_open_clip_keys(TINY.num_layers)
        keys.remove("visual.ln_post.bias")
        with pytest.raises(ValueError, match="missing"):
            plan_conversion(keys, TINY, "open_clip")

    def test_unrecognized_vision_key_fails_loud(self):
        keys = [
            *synthetic_open_clip_keys(TINY.num_layers),
            "visual.mystery.weight",
        ]
        with pytest.raises(ValueError, match="does not\\s+understand"):
            plan_conversion(keys, TINY, "open_clip")

    def test_hf_drops_are_not_recognized_here(self):
        # the drop vocabulary is per-format: an HF-style text_model key in
        # an open_clip checkpoint means something is wrong
        keys = [
            *synthetic_open_clip_keys(TINY.num_layers),
            "text_model.embeddings.token_embedding.weight",
        ]
        with pytest.raises(ValueError, match="does not\\s+understand"):
            plan_conversion(keys, TINY, "open_clip")

    def test_unknown_source_format_fails_loud(self):
        # a config string that never got validated (cast defeats the static
        # Literal so the RUNTIME boundary check is what's exercised)
        bad = cast(SourceFormat, "timm")
        with pytest.raises(ValueError, match="source_format"):
            plan_conversion(synthetic_open_clip_keys(TINY.num_layers), TINY, bad)

    def test_fare_registry_entry(self):
        spec = MLX_VIT_MODELS["FARE4ViTB32"]
        assert spec.repo_id == "chs20/FARE4-ViT-B-32-laion2B-s34B-b79K"
        assert spec.weights_file == "open_clip_model.safetensors"
        assert spec.source_format == "open_clip"
        # laion/FARE lineage: exact erf-GELU, not QuickGELU, not tanh
        assert spec.config.activation == "gelu"
        assert spec.config.num_tokens == 50  # B/32 geometry
        assert spec.config.output_dim == 512
        # FARE keeps the OpenAI CLIP preprocessing stats (repo preprocess_cfg)
        assert spec.image_mean == OPENAI_CLIP_MEAN
        assert spec.image_std == OPENAI_CLIP_STD

    def test_classic_tier_unchanged(self):
        for key in ("ViTB32", "ViTB16", "ViTL14", "ViTL14_336px"):
            spec = MLX_VIT_MODELS[key]
            assert spec.source_format == "hf_clip", key
            assert spec.config.activation == "quickgelu", key
            assert spec.image_mean == OPENAI_CLIP_MEAN, key
            assert spec.image_std == OPENAI_CLIP_STD, key


@needs_mlx
class TestVisionTower:
    """Random-weight tower behavior — needs mlx, no downloads."""

    @pytest.fixture()
    def tower(self):
        import mlx.core as mx

        from pytti.Perceptor.mlx_backend.vit import VisionTower

        mx.random.seed(0)
        tower = VisionTower(TINY)
        tower.set_dtype(mx.float16)
        return tower

    def test_param_tree_matches_converter_vocabulary(self, tower):
        # the keystone: the converter's view of the module tree and the
        # actual module tree cannot drift apart silently
        from mlx.utils import tree_flatten

        names = {name for name, _ in tree_flatten(tower.parameters())}
        assert names == set(expected_param_names(TINY))

    def test_encode_shape_dtype_and_grads(self, tower):
        import mlx.core as mx

        x = mx.random.normal((3, 16, 16, 3))  # fp32, per the bridge contract
        out = tower.encode(x)
        assert out.shape == (3, TINY.output_dim)
        assert out.dtype == mx.float16
        assert bool(mx.isfinite(out).all())

        loss, grad = mx.value_and_grad(lambda x: tower.encode(x).sum())(x)
        assert grad.shape == x.shape
        assert grad.dtype == mx.float32  # grads come back in the input dtype
        assert bool(mx.isfinite(grad).all())

    def test_encode_rejects_nchw(self, tower):
        import mlx.core as mx

        with pytest.raises(ValueError, match="NHWC"):
            tower.encode(mx.zeros((3, 3, 16, 16)))

    def test_compile_matches_eager(self, tower):
        import mlx.core as mx

        x = mx.random.normal((2, 16, 16, 3))
        compiled = mx.compile(tower.encode)
        assert bool(mx.allclose(compiled(x), tower.encode(x)))

    def test_gelu_activation_is_exact_erf_form(self):
        import mlx.core as mx
        import numpy as np
        import torch

        from pytti.Perceptor.mlx_backend.vit import _ACTIVATIONS

        x = np.linspace(-6.0, 6.0, 4001, dtype=np.float32)
        ours = np.array(_ACTIVATIONS["gelu"](mx.array(x)))
        # torch nn.GELU(approximate='none') — what open_clip instantiates
        # for the laion/FARE lineage
        erf = torch.nn.functional.gelu(torch.from_numpy(x)).numpy()
        assert float(np.abs(ours - erf).max()) < 1e-5
        # ... and NOT the tanh approximation (the QuickGELU-drift lesson)
        tanh = torch.nn.functional.gelu(
            torch.from_numpy(x), approximate="tanh"
        ).numpy()
        assert float(np.abs(ours - tanh).max()) > 1e-4

    def test_activation_choice_changes_the_tower(self):
        # a tower that silently ignored config.activation would still pass
        # every gate built on quickgelu weights — force the distinction
        import mlx.core as mx
        from mlx.utils import tree_flatten

        from pytti.Perceptor.mlx_backend.vit import VisionTower

        mx.random.seed(0)
        quick = VisionTower(TINY)
        gelu = VisionTower(dataclasses.replace(TINY, activation="gelu"))
        gelu.load_weights(tree_flatten(quick.parameters()), strict=True)
        x = mx.random.normal((2, 16, 16, 3))
        assert not bool(mx.allclose(quick.encode(x), gelu.encode(x)))

    def test_gelu_tower_compiles_with_finite_grads(self):
        import mlx.core as mx

        from pytti.Perceptor.mlx_backend.vit import VisionTower

        mx.random.seed(0)
        tower = VisionTower(dataclasses.replace(TINY, activation="gelu"))
        tower.set_dtype(mx.float16)
        x = mx.random.normal((2, 16, 16, 3))
        compiled = mx.compile(tower.encode)
        assert bool(mx.allclose(compiled(x), tower.encode(x)))
        _, grad = mx.value_and_grad(lambda x: tower.encode(x).sum())(x)
        assert grad.dtype == mx.float32
        assert bool(mx.isfinite(grad).all())


@pytest.fixture(scope="module")
def torch_ref():
    import torch

    from pytti.Perceptor import _load_perceptor

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    return _load_perceptor("ViTB32", device), device


@pytest.fixture(scope="module")
def fare_ref():
    import torch

    from pytti.Perceptor import _load_perceptor

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    return _load_perceptor("FARE4ViTB32", device), device


@needs_mlx
@pytest.mark.download
class TestRealWeightParity:
    """Plan gates 1-2 in miniature (full-size gate numbers live in the plan)."""

    def test_b32_embedding_parity(self, torch_ref):
        import mlx.core as mx
        import numpy as np
        import torch

        from pytti.Perceptor.mlx_backend.convert import load_tower

        perc, device = torch_ref
        torch.manual_seed(0)
        xn = perc.normalize(torch.rand(16, 3, perc.cut_size, perc.cut_size))
        with torch.no_grad():
            ref = perc.encode_image(xn.to(device)).float().cpu().numpy()
        tower = load_tower("ViTB32")
        emb = tower.encode(mx.array(xn.permute(0, 2, 3, 1).contiguous().numpy()))
        emb = np.array(emb.astype(mx.float32))
        a = ref / np.linalg.norm(ref, axis=1, keepdims=True)
        b = emb / np.linalg.norm(emb, axis=1, keepdims=True)
        assert float((a * b).sum(1).min()) >= 0.999  # plan gate 1

    def test_b32_input_grad_parity(self, torch_ref):
        import mlx.core as mx
        import numpy as np
        import torch

        from pytti.Perceptor.mlx_backend.convert import load_tower
        from pytti.Perceptor.Prompt import spherical_dist_loss

        perc, device = torch_ref
        torch.manual_seed(0)
        target = torch.nn.functional.normalize(torch.randn(512), dim=-1)
        xn = perc.normalize(torch.rand(8, 3, perc.cut_size, perc.cut_size))
        xt = xn.clone().to(device).requires_grad_(True)
        spherical_dist_loss(
            perc.encode_image(xt), target.to(device)
        ).sum().backward()
        ref = np.transpose(xt.grad.detach().cpu().numpy(), (0, 2, 3, 1))

        tower = load_tower("ViTB32")
        tm = mx.array(target.numpy())

        def loss_fn(x):
            e = tower.encode(x).astype(mx.float32)
            e = e / mx.sqrt((e * e).sum(axis=-1, keepdims=True))
            d = mx.sqrt(((e - tm) ** 2).sum(axis=-1))
            return (2 * mx.arcsin(d / 2) ** 2).sum()

        _, grad = mx.value_and_grad(loss_fn)(
            mx.array(xn.permute(0, 2, 3, 1).contiguous().numpy())
        )
        g = np.array(grad.astype(mx.float32)).ravel()
        r = ref.ravel()
        cos = float(r @ g / (np.linalg.norm(r) * np.linalg.norm(g)))
        assert cos >= 0.995  # plan gate 2

    def test_fare_b32_embedding_parity(self, fare_ref):
        import mlx.core as mx
        import numpy as np
        import torch

        from pytti.Perceptor.mlx_backend.convert import load_tower

        perc, device = fare_ref
        torch.manual_seed(0)
        xn = perc.normalize(torch.rand(16, 3, perc.cut_size, perc.cut_size))
        with torch.no_grad():
            ref = perc.encode_image(xn.to(device)).float().cpu().numpy()
        tower = load_tower("FARE4ViTB32")
        emb = tower.encode(mx.array(xn.permute(0, 2, 3, 1).contiguous().numpy()))
        emb = np.array(emb.astype(mx.float32))
        a = ref / np.linalg.norm(ref, axis=1, keepdims=True)
        b = emb / np.linalg.norm(emb, axis=1, keepdims=True)
        assert float((a * b).sum(1).min()) >= 0.999  # plan gate 1

    def test_fare_b32_input_grad_parity(self, fare_ref):
        import mlx.core as mx
        import numpy as np
        import torch

        from pytti.Perceptor.mlx_backend.convert import load_tower
        from pytti.Perceptor.Prompt import spherical_dist_loss

        perc, device = fare_ref
        torch.manual_seed(0)
        target = torch.nn.functional.normalize(torch.randn(512), dim=-1)
        xn = perc.normalize(torch.rand(8, 3, perc.cut_size, perc.cut_size))
        xt = xn.clone().to(device).requires_grad_(True)
        spherical_dist_loss(
            perc.encode_image(xt), target.to(device)
        ).sum().backward()
        ref = np.transpose(xt.grad.detach().cpu().numpy(), (0, 2, 3, 1))

        tower = load_tower("FARE4ViTB32")
        tm = mx.array(target.numpy())

        def loss_fn(x):
            e = tower.encode(x).astype(mx.float32)
            e = e / mx.sqrt((e * e).sum(axis=-1, keepdims=True))
            d = mx.sqrt(((e - tm) ** 2).sum(axis=-1))
            return (2 * mx.arcsin(d / 2) ** 2).sum()

        _, grad = mx.value_and_grad(loss_fn)(
            mx.array(xn.permute(0, 2, 3, 1).contiguous().numpy())
        )
        g = np.array(grad.astype(mx.float32)).ravel()
        r = ref.ravel()
        cos = float(r @ g / (np.linalg.norm(r) * np.linalg.norm(g)))
        assert cos >= 0.995  # plan gate 2


# ---------------------------------------------------------------------------
# SigLIP2 (timm trunk) — plan structure, tower behavior, real-weight parity
# ---------------------------------------------------------------------------

TINY_SIGLIP = SigLIPConfig(
    image_size=16,
    patch_size=8,
    hidden_dim=8,
    num_layers=2,
    num_heads=2,
    mlp_dim=32,
)


def synthetic_open_clip_timm_keys(num_layers: int) -> list[str]:
    """The key set of a SigLIP2 open_clip export (visual side exact; text
    side — a "text."-prefixed tower plus the sigmoid-loss scalars —
    representative of the recognized-drop classes)."""
    keys = [
        "visual.trunk.patch_embed.proj.weight",
        "visual.trunk.patch_embed.proj.bias",
        "visual.trunk.pos_embed",
        "visual.trunk.norm.weight",
        "visual.trunk.norm.bias",
        "visual.trunk.attn_pool.latent",
        "logit_scale",
        "logit_bias",
        "text.positional_embedding",
        "text.token_embedding.weight",
        "text.transformer.resblocks.0.attn.in_proj_weight",
        "text.ln_final.weight",
    ]
    for name in ("q", "kv", "proj", "norm", "mlp.fc1", "mlp.fc2"):
        for wb in ("weight", "bias"):
            keys.append(f"visual.trunk.attn_pool.{name}.{wb}")
    for i in range(num_layers):
        base = f"visual.trunk.blocks.{i}"
        for wb in ("weight", "bias"):
            keys += [
                f"{base}.norm1.{wb}",
                f"{base}.norm2.{wb}",
                f"{base}.attn.qkv.{wb}",  # qkv arrives pre-fused
                f"{base}.attn.proj.{wb}",
                f"{base}.mlp.fc1.{wb}",
                f"{base}.mlp.fc2.{wb}",
            ]
    return keys


class TestTimmSigLIPConversionPlan:
    """open_clip_timm source layout (SigLIP2) — pure structure, no mlx."""

    def _plan(self, keys=None):
        return plan_conversion(
            keys
            if keys is not None
            else synthetic_open_clip_timm_keys(TINY_SIGLIP.num_layers),
            TINY_SIGLIP,
            "open_clip_timm",
        )

    def test_plan_covers_module_tree_exactly(self):
        assert set(self._plan()) == set(expected_param_names(TINY_SIGLIP))

    def test_qkv_arrives_prefused_and_conv_transposes(self):
        plan = self._plan()
        assert plan["blocks.0.attn.qkv.weight"] == Copy(
            "visual.trunk.blocks.0.attn.qkv.weight"
        )
        assert plan["blocks.0.attn.out_proj.weight"] == Copy(
            "visual.trunk.blocks.0.attn.proj.weight"
        )
        assert plan["patch_embed.weight"] == ConvToOHWI(
            "visual.trunk.patch_embed.proj.weight"
        )
        # the OpenAI towers have no conv bias; this one does
        assert plan["patch_embed.bias"] == Copy(
            "visual.trunk.patch_embed.proj.bias"
        )

    def test_missing_vision_key_fails_loud(self):
        keys = synthetic_open_clip_timm_keys(TINY_SIGLIP.num_layers)
        keys.remove("visual.trunk.attn_pool.latent")
        with pytest.raises(ValueError, match="missing"):
            self._plan(keys)

    def test_unrecognized_vision_key_fails_loud(self):
        keys = [
            *synthetic_open_clip_timm_keys(TINY_SIGLIP.num_layers),
            "visual.trunk.mystery.weight",
        ]
        with pytest.raises(ValueError, match="does not\\s+understand"):
            self._plan(keys)

    def test_other_formats_drops_are_not_recognized_here(self):
        # the drop vocabulary is per-format: an hf_clip text_model key or an
        # open_clip attn_mask buffer in a SigLIP2 export means wrong format
        for alien in ("text_model.embeddings.token_embedding.weight", "attn_mask"):
            keys = [
                *synthetic_open_clip_timm_keys(TINY_SIGLIP.num_layers),
                alien,
            ]
            with pytest.raises(ValueError, match="does not\\s+understand"):
                self._plan(keys)

    def test_config_type_mismatch_fails_loud(self):
        # registry bugs surface as a typed error, not nonsense missing-keys
        with pytest.raises(TypeError, match="mismatch"):
            plan_conversion(
                synthetic_open_clip_timm_keys(TINY.num_layers),
                TINY,  # ViTConfig with the timm format
                "open_clip_timm",
            )
        with pytest.raises(TypeError, match="mismatch"):
            plan_conversion(
                synthetic_open_clip_keys(TINY.num_layers),
                TINY_SIGLIP,  # SigLIPConfig with a ViT format
                "open_clip",
            )

    def test_siglip_registry_entry(self):
        from pytti.Perceptor.mlx_backend.convert import SIGLIP_MEAN, SIGLIP_STD

        spec = MLX_VIT_MODELS["SigLIP2B16"]
        assert spec.repo_id == "timm/ViT-B-16-SigLIP2"
        assert spec.weights_file == "open_clip_model.safetensors"
        assert spec.source_format == "open_clip_timm"
        assert isinstance(spec.config, SigLIPConfig)
        # timm geometry: 196 tokens (no cls), MAP head keeps the width
        assert spec.config.num_tokens == 196
        assert spec.config.output_dim == spec.config.hidden_dim == 768
        # empirically verified against the loaded torch trunk (2026-08-03):
        # LN eps 1e-6, exact erf-GELU, mean=std=0.5 preprocessing
        assert spec.config.layer_norm_eps == 1e-6
        assert spec.config.activation == "gelu"
        assert spec.image_mean == SIGLIP_MEAN == (0.5, 0.5, 0.5)
        assert spec.image_std == SIGLIP_STD == (0.5, 0.5, 0.5)


@needs_mlx
class TestSigLIPTower:
    """Random-weight SigLIP tower behavior — needs mlx, no downloads."""

    @pytest.fixture()
    def tower(self):
        import mlx.core as mx

        from pytti.Perceptor.mlx_backend.siglip import SigLIPTower

        mx.random.seed(0)
        tower = SigLIPTower(TINY_SIGLIP)
        tower.set_dtype(mx.float16)
        return tower

    def test_param_tree_matches_converter_vocabulary(self, tower):
        # the keystone: the converter's view of the module tree and the
        # actual module tree cannot drift apart silently
        from mlx.utils import tree_flatten

        names = {name for name, _ in tree_flatten(tower.parameters())}
        assert names == set(expected_param_names(TINY_SIGLIP))

    def test_encode_shape_dtype_and_grads(self, tower):
        import mlx.core as mx

        x = mx.random.normal((3, 16, 16, 3))  # fp32, per the bridge contract
        out = tower.encode(x)
        assert out.shape == (3, TINY_SIGLIP.output_dim)
        assert out.dtype == mx.float16
        assert bool(mx.isfinite(out).all())

        loss, grad = mx.value_and_grad(lambda x: tower.encode(x).sum())(x)
        assert grad.shape == x.shape
        assert grad.dtype == mx.float32  # grads come back in the input dtype
        assert bool(mx.isfinite(grad).all())

    def test_encode_rejects_nchw(self, tower):
        import mlx.core as mx

        with pytest.raises(ValueError, match="NHWC"):
            tower.encode(mx.zeros((3, 3, 16, 16)))

    def test_compile_matches_eager(self, tower):
        import mlx.core as mx

        x = mx.random.normal((2, 16, 16, 3))
        compiled = mx.compile(tower.encode)
        assert bool(mx.allclose(compiled(x), tower.encode(x)))


@pytest.fixture(scope="module")
def siglip_ref():
    import torch

    from pytti.Perceptor import _load_perceptor

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    return _load_perceptor("SigLIP2B16", device), device


@needs_mlx
@pytest.mark.download
class TestSigLIPRealWeightParity:
    """Plan gates 1-2 in miniature for the SigLIP2 tower (full-size numbers:
    embed cosine min 0.999998 over 256 images, input-grad cosine 0.9958 at
    batch 40 — measured 2026-08-03, M3 Max, mlx 0.32, fp16)."""

    def test_siglip_b16_embedding_parity(self, siglip_ref):
        import mlx.core as mx
        import numpy as np
        import torch

        from pytti.Perceptor.mlx_backend.convert import load_tower

        perc, device = siglip_ref
        torch.manual_seed(0)
        xn = perc.normalize(torch.rand(16, 3, perc.cut_size, perc.cut_size))
        with torch.no_grad():
            ref = perc.encode_image(xn.to(device)).float().cpu().numpy()
        tower = load_tower("SigLIP2B16")
        emb = tower.encode(mx.array(xn.permute(0, 2, 3, 1).contiguous().numpy()))
        emb = np.array(emb.astype(mx.float32))
        a = ref / np.linalg.norm(ref, axis=1, keepdims=True)
        b = emb / np.linalg.norm(emb, axis=1, keepdims=True)
        assert float((a * b).sum(1).min()) >= 0.999  # plan gate 1

    def test_siglip_b16_input_grad_parity(self, siglip_ref):
        import mlx.core as mx
        import numpy as np
        import torch

        from pytti.Perceptor.mlx_backend.convert import load_tower
        from pytti.Perceptor.Prompt import spherical_dist_loss

        perc, device = siglip_ref
        torch.manual_seed(0)
        target = torch.nn.functional.normalize(torch.randn(768), dim=-1)
        xn = perc.normalize(torch.rand(8, 3, perc.cut_size, perc.cut_size))
        xt = xn.clone().to(device).requires_grad_(True)
        spherical_dist_loss(
            perc.encode_image(xt), target.to(device)
        ).sum().backward()
        ref = np.transpose(xt.grad.detach().cpu().numpy(), (0, 2, 3, 1))

        tower = load_tower("SigLIP2B16")
        tm = mx.array(target.numpy())

        def loss_fn(x):
            e = tower.encode(x).astype(mx.float32)
            e = e / mx.sqrt((e * e).sum(axis=-1, keepdims=True))
            d = mx.sqrt(((e - tm) ** 2).sum(axis=-1))
            return (2 * mx.arcsin(d / 2) ** 2).sum()

        _, grad = mx.value_and_grad(loss_fn)(
            mx.array(xn.permute(0, 2, 3, 1).contiguous().numpy())
        )
        g = np.array(grad.astype(mx.float32)).ravel()
        r = ref.ravel()
        cos = float(r @ g / (np.linalg.norm(r) * np.linalg.norm(g)))
        assert cos >= 0.995  # plan gate 2
