"""
MLX bridge (docs/mlx-port-plan.md, phase M1 — bridge.py + integration).

Three tiers, matching test_mlx_tower.py:
- constants/validation tests: pure torch on CPU, run everywhere;
- bridge-math tests: need mlx (darwin), no downloads — the plan's gate 3
  in its exact-arithmetic (fp32 embeddings) form, values AND gradients;
- full-path parity: needs mlx AND a real checkpoint -> @pytest.mark.download.
"""

import importlib.util
import math

import pytest
import torch
from omegaconf import OmegaConf

from pytti.Perceptor.mlx_backend.bridge import (
    semantic_prompt_constants,
)
from pytti.Perceptor.Prompt import Prompt, make_mask
from pytti.prompt_spec import MaskGeometric, MaskSemantic

needs_mlx = pytest.mark.skipif(
    importlib.util.find_spec("mlx") is None,
    reason="mlx not installed (darwin-only backend)",
)

CPU = torch.device("cpu")
N, C, DIM = 5, 2, 8  # cutouts, perceptors, embedding width


class StubEmbedder:
    """format_input only needs the axis vocabulary of HDMultiClipEmbedder."""

    output_axes = ("c", "n", "i")


def make_geometry(seed=0):
    """[C, n, 2] offsets/sizes straddling the 0.5 mask threshold."""
    g = torch.Generator().manual_seed(seed)
    offsets = torch.rand((C, N, 2), generator=g) * 0.8
    sizes = torch.rand((C, N, 2), generator=g) * 0.2 + 0.1
    return offsets, sizes


def make_prompt(weight="1", stop="-inf", mask=None, seed=1):
    g = torch.Generator().manual_seed(seed)
    embeds = torch.randn((C, DIM), generator=g)
    kwargs = {} if mask is None else {"mask": mask}
    return Prompt(embeds, weight, stop, "test prompt", "test prompt:1", device=CPU, **kwargs)


# ---------------------------------------------------------------------------
# constants helper (pure torch, CPU)
# ---------------------------------------------------------------------------


class TestPromptConstants:
    def test_scalar_weight_default_mask(self):
        offsets, sizes = make_geometry()
        consts = semantic_prompt_constants(
            make_prompt(weight="1.5"), offsets, sizes, StubEmbedder()
        )
        assert consts.weight.shape == (N, C)
        assert torch.equal(consts.weight, torch.full((N, C), 1.5))
        # mask_all stops are -inf, scalar stop is -inf, weight positive
        assert torch.equal(consts.stops, torch.full((N, C), -math.inf))
        assert torch.equal(consts.embeds, make_prompt(weight="1.5").embeds)

    def test_negative_weight_folds_sign_offset_into_stops(self):
        # Prompt.forward: stops = max(mask_stops + sign_offset, stop); for a
        # negative weight sign_offset is -1, so max(-inf - 1, -0.9) = -0.9
        offsets, sizes = make_geometry()
        consts = semantic_prompt_constants(
            make_prompt(weight="-2", stop="-0.9"), offsets, sizes, StubEmbedder()
        )
        assert torch.equal(consts.weight, torch.full((N, C), -2.0))
        assert torch.equal(consts.stops, torch.full((N, C), -0.9))

    def test_geometric_mask_reproduces_mask_right(self):
        offsets, sizes = make_geometry()
        mask = make_mask(MaskGeometric(key="r"), "0.5")
        consts = semantic_prompt_constants(
            make_prompt(mask=mask), offsets, sizes, StubEmbedder()
        )
        # mask_right emits stop=1.0 for crops centered left of thresh (their
        # gradient is gated off once dists < 1.0), stop=0.0 otherwise
        cent = (offsets[..., 0] + sizes[..., 0] / 2).T  # [n, C]
        expected = cent.lt(0.5).float()
        assert torch.equal(consts.stops, expected)
        assert (expected == 0).any() and (expected == 1).any(), "seed must straddle"

    def test_parametric_weight_is_reevaluated(self):
        from pytti import set_t

        offsets, sizes = make_geometry()
        prompt = make_prompt(weight="2*t")
        set_t(3.0)
        try:
            consts = semantic_prompt_constants(prompt, offsets, sizes, StubEmbedder())
            assert torch.equal(consts.weight, torch.full((N, C), 6.0))
        finally:
            set_t(0.0)

    def test_zero_weight_and_disabled_return_none(self):
        offsets, sizes = make_geometry()
        assert (
            semantic_prompt_constants(
                make_prompt(weight="0"), offsets, sizes, StubEmbedder()
            )
            is None
        )
        prompt = make_prompt(weight="1")
        prompt.set_enabled(False)
        assert (
            semantic_prompt_constants(prompt, offsets, sizes, StubEmbedder()) is None
        )

    def test_prompt_subclasses_fail_loud(self):
        class FancyPrompt(Prompt):
            pass

        offsets, sizes = make_geometry()
        g = torch.Generator().manual_seed(1)
        prompt = FancyPrompt(
            torch.randn((C, DIM), generator=g), "1", "-inf", "x", "x:1", device=CPU
        )
        with pytest.raises(RuntimeError, match="FancyPrompt"):
            semantic_prompt_constants(prompt, offsets, sizes, StubEmbedder())

    def test_embed_dependent_mask_fails_loud(self):
        offsets, sizes = make_geometry()

        def fake_semantic_mask(pos, size, emb):
            raise AssertionError("must not be called")

        fake_semantic_mask.embed_dependent = True
        prompt = make_prompt(mask=fake_semantic_mask)
        with pytest.raises(RuntimeError, match="semantic"):
            semantic_prompt_constants(prompt, offsets, sizes, StubEmbedder())


def test_make_mask_tags_semantic_masks(monkeypatch):
    import pytti.Perceptor as perceptor_module

    class StubTextPerceptor:
        def embed_text(self, text, device):
            return torch.ones(1, 4)

    monkeypatch.setattr(
        perceptor_module, "CLIP_PERCEPTORS", [StubTextPerceptor()]
    )
    assert make_mask(MaskSemantic(text="a dog"), "0.5").embed_dependent is True
    assert make_mask(MaskGeometric(key="r"), "0.5").embed_dependent is False


# ---------------------------------------------------------------------------
# backend selection (fail loud, no silent mixed engine)
# ---------------------------------------------------------------------------


def test_load_clip_mlx_rejects_non_vit_towers(monkeypatch):
    import pytti.Perceptor as perceptor_module

    monkeypatch.setattr(perceptor_module, "CLIP_MODEL_NAMES", None)
    params = OmegaConf.create(
        {"perceptor_backend": "mlx", "ViTB32": True, "RN50": True}
    )
    with pytest.raises(RuntimeError, match="RN50"):
        perceptor_module.load_clip(params)


def test_schema_rejects_unknown_perceptor_backend():
    from pytti.config.structured_config import ConfigSchema

    with pytest.raises(ValueError, match="perceptor_backend"):
        ConfigSchema(scenes="x", perceptor_backend="metal")


# ---------------------------------------------------------------------------
# bridge math (mlx, no downloads) — plan gate 3 in exact-arithmetic form
# ---------------------------------------------------------------------------


def torch_reference(prompts_scales, embed, offsets, sizes, embedder):
    """The torch path's semantic loss, verbatim from ImageGuide.train."""
    from pytti import format_input

    total = torch.zeros(())
    raws = []
    for prompt, scale in prompts_scales:
        loss, loss_raw = prompt(
            embed,
            format_input(offsets, embedder, prompt),
            format_input(sizes, embedder, prompt),
        )
        total = total + loss * scale
        raws.append(loss_raw.detach())
    return total, torch.stack(raws)


def gate3_fixture():
    """Prompts exercising weights, negative weights, masks, and stops."""
    offsets, sizes = make_geometry(seed=7)
    prompts_scales = [
        (make_prompt(weight="1.5", seed=11), 1.0),
        (make_prompt(weight="-1", stop="-0.9", seed=12), 0.7),
        (make_prompt(weight="2", stop="1.2", seed=13), 0.3),
        (make_prompt(mask=make_mask(MaskGeometric(key="r"), "0.5"), seed=14), 0.5),
    ]
    g = torch.Generator().manual_seed(99)
    embed = torch.randn((N, C, DIM), generator=g)
    return prompts_scales, embed, offsets, sizes


@needs_mlx
def test_reduction_matches_prompt_forward_values_and_grads():
    import mlx.core as mx
    import numpy as np

    from pytti.Perceptor.mlx_backend.bridge import mlx_semantic_reduction

    prompts_scales, embed, offsets, sizes = gate3_fixture()
    embedder = StubEmbedder()

    embed_t = embed.clone().requires_grad_(True)
    total_ref, raws_ref = torch_reference(
        prompts_scales, embed_t, offsets, sizes, embedder
    )
    total_ref.backward()

    consts = [
        semantic_prompt_constants(p, offsets, sizes, embedder)
        for p, _ in prompts_scales
    ]
    assert all(c is not None for c in consts)
    # the stop=1.2 prompt must actually gate a strict subset of elements,
    # otherwise this fixture doesn't exercise replace_grad semantics
    from pytti.Perceptor.Prompt import spherical_dist_loss

    dists3 = spherical_dist_loss(embed, prompts_scales[2][0].embeds)
    assert dists3.lt(1.2).any() and dists3.ge(1.2).any()

    text = mx.array(torch.stack([c.embeds for c in consts]).numpy())
    weights = mx.array(torch.stack([c.weight for c in consts]).numpy())
    stops = mx.array(torch.stack([c.stops for c in consts]).numpy())
    scales = mx.array([s for _, s in prompts_scales])

    def total_fn(e):
        return mlx_semantic_reduction(e, text, weights, stops, scales)[0]

    embed_mx = mx.array(embed.numpy())
    total_mx, grad_mx = mx.value_and_grad(total_fn)(embed_mx)
    _, _, raws_mx = mlx_semantic_reduction(embed_mx, text, weights, stops, scales)

    assert float(total_mx) == pytest.approx(float(total_ref.detach()), rel=1e-5)
    raws_t = torch.from_numpy(np.array(raws_mx))
    assert torch.allclose(raws_t, raws_ref, rtol=1e-5, atol=1e-7)
    grad_t = torch.from_numpy(np.array(grad_mx.astype(mx.float32)))
    assert torch.allclose(grad_t, embed_t.grad, rtol=1e-4, atol=1e-6)


@needs_mlx
def test_bridge_function_round_trip_and_backward():
    """DLPack round trip, NHWC<->NCHW, clone-before-drop, cotangent scaling —
    with a trivial mlx step so no weights are needed."""
    import mlx.core as mx

    from pytti.Perceptor.mlx_backend.bridge import _MLXBridgeFunction

    def fake_step(mx_batches, *consts):
        def f(bs):
            total = mx.array(0.0)
            for b in bs:
                total = total + (b * b).sum()
            return total, mx.stack([b.mean() for b in bs])

        return mx.value_and_grad(f)(mx_batches)

    x = (torch.arange(96, dtype=torch.float32).reshape(2, 3, 4, 4) / 96.0)
    x = x.clone().requires_grad_(True)
    total, raws = _MLXBridgeFunction.apply(fake_step, (), x)

    assert total.requires_grad
    assert not raws.requires_grad
    assert float(total.detach()) == pytest.approx(
        float((x * x).sum().detach()), rel=1e-6
    )
    assert float(raws[0]) == pytest.approx(float(x.mean().detach()), rel=1e-6)

    (2.0 * total).backward()  # cotangent 2 must scale the stored grad
    assert torch.allclose(x.grad, 4.0 * x.detach(), rtol=1e-6, atol=1e-7)


# ---------------------------------------------------------------------------
# full path against a real tower (plan gate 3, fp16 tolerance)
# ---------------------------------------------------------------------------


@needs_mlx
@pytest.mark.download
def test_full_bridge_matches_torch_semantic_loss():
    import numpy as np
    from PIL import Image

    from pytti import cat_with_pad, format_input
    from pytti.device import default_device
    from pytti.image_models import RGBImage
    from pytti.Perceptor import _load_perceptor
    from pytti.Perceptor.Embedder import HDMultiClipEmbedder
    from pytti.Perceptor.mlx_backend.bridge import MLXSemanticLoss
    from pytti.Perceptor.Prompt import mask_image

    device = default_device()
    perceptor = _load_perceptor("ViTB32", device)
    embedder = HDMultiClipEmbedder(
        perceptors=[perceptor], cutn=8, cutout_sampler="batched", device=device
    )
    embeds = cat_with_pad([perceptor.embed_text("a bioluminescent forest", device)])
    embeds2 = cat_with_pad([perceptor.embed_text("night sky", device)])
    # image-masked prompt: its VALUE depends on mask weights, which depend on
    # crop sizes — locks in the sizes-vs-offsets wiring of the bridge.
    # mask_image is stateful (error diffusion advances per call), so each
    # path gets a fresh instance built from the same PIL image.
    ramp_pil = Image.fromarray(
        np.linspace(0, 255, 64, dtype=np.uint8)[None, :].repeat(64, axis=0), "L"
    )
    prompt = Prompt(embeds, "1", "-inf", "forest", "forest:1", device=device)

    img = RGBImage(128, 128, device=device)
    torch.manual_seed(5)
    img.encode_random()

    def torch_path():
        masked = Prompt(
            embeds2, "0.7", "-inf", "sky", "sky:0.7",
            mask=mask_image(ramp_pil), device=device,
        )
        z = img.decode_training_tensor()
        torch.manual_seed(42)
        image_embeds, offsets, sizes = embedder(img, input=z)
        total = torch.zeros((), device=device)
        raws = []
        for p in (prompt, masked):
            loss, loss_raw = p(
                format_input(image_embeds, embedder, p),
                format_input(offsets, embedder, p),
                format_input(sizes, embedder, p),
            )
            total = total + loss * 1.0
            raws.append(loss_raw.detach())
        total.backward()
        grads = torch.cat([p.grad.flatten().clone() for p in img.parameters()])
        return total.detach(), raws[0], grads

    def mlx_path():
        masked = Prompt(
            embeds2, "0.7", "-inf", "sky", "sky:0.7",
            mask=mask_image(ramp_pil), device=device,
        )
        bridge = MLXSemanticLoss(embedder)
        z = img.decode_training_tensor()
        torch.manual_seed(42)  # same seed -> identical cutouts (RNG parity)
        total, records = bridge(img, z, [prompt, masked], [], ramp=1.0)
        total.backward()
        grads = torch.cat([p.grad.flatten().clone() for p in img.parameters()])
        return total.detach(), records[str(prompt)], grads

    loss_t, raw_t, grads_t = torch_path()
    for p in img.parameters():
        p.grad = None
    loss_m, raw_m, grads_m = mlx_path()

    assert float(loss_m) == pytest.approx(float(loss_t), rel=1e-2)  # plan gate 3
    assert float(raw_m) == pytest.approx(float(raw_t), rel=1e-2)
    cos = torch.nn.functional.cosine_similarity(grads_t, grads_m, dim=0)
    assert float(cos) >= 0.99  # plan gate 2's expected class, full path
