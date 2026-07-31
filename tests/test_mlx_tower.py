"""
MLX perceptor backend (docs/mlx-port-plan.md, phase M1).

Three tiers:
- conversion-plan tests: pure Python, run everywhere (linux CI included) —
  no mlx, no downloads;
- tower tests: need mlx (darwin), random weights, no downloads;
- parity tests: need mlx AND real checkpoints -> @pytest.mark.download
  (excluded from the default suite by addopts).
"""

import importlib.util

import pytest

from pytti.Perceptor.mlx_backend import ViTConfig
from pytti.Perceptor.mlx_backend.convert import (
    MLX_VIT_MODELS,
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


class TestConversionPlan:
    """Pure structure tests — run on any platform, no mlx."""

    def test_plan_covers_module_tree_exactly(self):
        plan = plan_conversion(synthetic_hf_keys(TINY.num_layers), TINY)
        assert set(plan) == set(expected_param_names(TINY))

    def test_qkv_fusion_consumes_separate_projections(self):
        plan = plan_conversion(synthetic_hf_keys(TINY.num_layers), TINY)
        qkv = plan["blocks.0.attn.qkv.weight"]
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
        # 8 top-level params + 12 per block; tokens = (S/P)^2 + 1
        for key, spec in MLX_VIT_MODELS.items():
            cfg = spec.config
            assert len(expected_param_names(cfg)) == 8 + 12 * cfg.num_layers, key
        assert MLX_VIT_MODELS["ViTB32"].config.num_tokens == 50
        assert MLX_VIT_MODELS["ViTB16"].config.num_tokens == 197
        assert MLX_VIT_MODELS["ViTL14_336px"].config.num_tokens == 577

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


@pytest.fixture(scope="module")
def torch_ref():
    import torch

    from pytti.Perceptor import _load_perceptor

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    return _load_perceptor("ViTB32", device), device


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
