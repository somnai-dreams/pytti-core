"""
Slice 9: LlamaGen VQ image model.

Default-suite tests exercise the l2_norm vector_quantize extension, the
vendored architecture at tiny scale, the schema/dispatch wiring, and the
backend eligibility fences — no downloads. The @pytest.mark.download tests
load the real ds16 checkpoint and gate the encode->decode round trip and
grad flow on CPU and MPS.
"""

import sys
from types import SimpleNamespace

import pytest
import torch
from torch.nn import functional as F

from pytti.image_models.llamagen import (
    LLAMAGEN_HF_REPO,
    LLAMAGEN_HF_REVISION,
    LlamaGenImage,
    load_llamagen_model,
)
from pytti.image_models.vqgan import vector_quantize
from pytti.vendor.llamagen.vq_model import ModelArgs, VQModel

# ---------------------------------------------------------------------------
# vector_quantize l2_norm extension
# ---------------------------------------------------------------------------


def test_vector_quantize_default_path_unchanged():
    # l2_norm defaults off: byte-identical to the historical implementation
    torch.manual_seed(0)
    x = torch.randn(1, 4, 4, 8)
    codebook = torch.randn(32, 8)

    d = (
        x.pow(2).sum(dim=-1, keepdim=True)
        + codebook.pow(2).sum(dim=1)
        - 2 * x @ codebook.T
    )
    expected = F.one_hot(d.argmin(-1), 32).to(d.dtype) @ codebook

    assert torch.equal(vector_quantize(x, codebook), expected)


def test_vector_quantize_l2_norm_matches_upstream_quantizer():
    # our module-level quantizer with l2_norm=True must reproduce the
    # vendored LlamaGen VectorQuantizer (eval mode) exactly
    from pytti.vendor.llamagen.vq_model import VectorQuantizer

    torch.manual_seed(1)
    quant = VectorQuantizer(
        n_e=64, e_dim=8, beta=0.25, entropy_loss_ratio=0.0, l2_norm=True, show_usage=False
    )
    quant.eval()
    z_bchw = torch.randn(1, 8, 4, 4)

    upstream, _, _ = quant(z_bchw)
    ours = vector_quantize(
        z_bchw.movedim(1, 3), quant.embedding.weight, l2_norm=True
    ).movedim(3, 1)

    assert torch.allclose(ours, upstream, atol=1e-6)


def test_vector_quantize_l2_norm_grad_flows_through_normalize():
    torch.manual_seed(2)
    x = torch.randn(1, 4, 4, 8, requires_grad=True)
    codebook = torch.randn(32, 8)

    out = vector_quantize(x, codebook, l2_norm=True)
    # unit-norm rows: quantized values are normalized codebook rows
    assert torch.allclose(out.norm(dim=-1), torch.ones(1, 4, 4), atol=1e-5)

    out.sum().backward()
    g = x.grad
    assert g is not None
    assert torch.isfinite(g).all()
    assert g.abs().sum() > 0
    # F.normalize backward projects out the radial component: every grad row
    # must be orthogonal to its input row. This fails if the straight-through
    # gradient were (incorrectly) attached to the raw latent instead of the
    # normalized one.
    radial = (g * x.detach()).sum(-1).abs().max()
    assert radial < 1e-5, f"radial grad component {radial} — normalize bypassed"


# ---------------------------------------------------------------------------
# vendored model + image rep at tiny scale (no download)
# ---------------------------------------------------------------------------


def tiny_model():
    torch.manual_seed(3)
    model = VQModel(
        ModelArgs(
            codebook_size=32,
            codebook_embed_dim=8,
            encoder_ch_mult=[1, 1],
            decoder_ch_mult=[1, 1],
            z_channels=8,
        )
    )
    return model.eval().requires_grad_(False)


def test_llamagen_image_geometry_and_grad():
    model = tiny_model()  # f = 2
    img = LlamaGenImage(33, 33, model=model, device="cpu")

    assert img.image_shape == (32, 32)  # rounded down to a multiple of f
    assert tuple(img.tensor.shape) == (1, 16, 16, 8)
    # latent rows live on the codebook's unit sphere
    assert torch.allclose(
        img.tensor.detach().norm(dim=-1), torch.ones(1, 16, 16), atol=1e-5
    )

    out = img.decode_training_tensor()
    assert tuple(out.shape) == (1, 3, 32, 32)
    detached = out.detach()
    assert float(detached.min()) >= 0 and float(detached.max()) <= 1

    out.mean().backward()
    g = img.tensor.grad
    assert g is not None and torch.isfinite(g).all() and g.abs().sum() > 0


def test_llamagen_image_encode_clone_and_latent(monkeypatch):
    from PIL import Image

    import pytti.image_models.llamagen as llamagen_module

    model = tiny_model()
    img = LlamaGenImage(32, 32, model=model, device="cpu")

    pil = Image.new("RGB", (32, 32), (200, 30, 90))
    img.encode_image(pil)
    assert tuple(img.tensor.shape) == (1, 16, 16, 8)

    latent = img.make_latent(pil)
    assert tuple(latent.shape) == (1, 8, 16, 16)  # BCHW for LatentLoss
    assert torch.allclose(
        latent, img.get_latent_tensor(detach=True), atol=1e-6
    )

    # clone() re-resolves the module-level singleton and the default device,
    # like VQGANImage
    monkeypatch.setattr(llamagen_module, "LLAMAGEN_MODEL", model)
    monkeypatch.setattr(llamagen_module, "default_device", lambda: "cpu")
    dupe = img.clone()
    assert torch.equal(dupe.tensor.detach(), img.tensor.detach())
    assert dupe.decay == img.decay
    assert img.get_preferred_loss().__name__ == "LatentLoss"


def test_checkpoint_key_validation_fails_loud(tmp_path):
    bogus = tmp_path / "bogus.pt"
    torch.save({"optimizer": {}}, bogus)
    with pytest.raises(RuntimeError, match="neither a 'model' nor an 'ema'"):
        load_llamagen_model(bogus, "ds16")


# ---------------------------------------------------------------------------
# schema + dispatch wiring
# ---------------------------------------------------------------------------


def test_schema_gains_llamagen_choices():
    import attrs

    from pytti.config.model_names import (
        LLAMAGEN_CHECKPOINT_FILES,
        LLAMAGEN_MODEL_NAMES,
    )
    from pytti.config.structured_config import ConfigSchema

    fields = {f.name: f for f in attrs.fields(ConfigSchema)}
    ConfigSchema(scenes="x", image_model="LlamaGen", llamagen_model="ds8")
    with pytest.raises(ValueError):
        ConfigSchema(scenes="x", image_model="NotAModel")
    with pytest.raises(ValueError):
        ConfigSchema(scenes="x", llamagen_model="ds32")
    assert fields["llamagen_model"].default == "ds16"
    assert sorted(LLAMAGEN_CHECKPOINT_FILES) == sorted(LLAMAGEN_MODEL_NAMES)
    # the pin is a full commit sha, not a branch name
    assert len(LLAMAGEN_HF_REVISION) == 40
    assert LLAMAGEN_HF_REPO == "FoundationVision/LlamaGen"


def test_workhorse_rejects_llamagen_on_mlx_backends():
    from pytti.workhorse import configure_pass

    for backend in ("mlx", "mlx_full"):
        # minimal stub of the schema fields configure_pass reads before the
        # backend check: the init_spectrum resolve happens at function top
        params = SimpleNamespace(
            image_model="LlamaGen",
            perceptor_backend=backend,
            init_spectrum="white",
            init_spectrum_falloff=1.0,
            init_spectrum_chroma="full",
        )
        with pytest.raises(ValueError, match="torch-only"):
            configure_pass(
                params,
                device="cpu",
                embedder=None,
                prompts=None,
                init_image_pil=None,
                video_frames=None,
                restore=False,
            )


@pytest.mark.skipif(sys.platform != "darwin", reason="mlx is Metal-only")
def test_mlx_full_engine_rejects_llamagen_image_rep():
    pytest.importorskip("mlx")
    from pytti.mlx_engine.engine import MLXStillEngine

    img = LlamaGenImage(32, 32, model=tiny_model(), device="cpu")
    params = SimpleNamespace(animation_mode="off")
    with pytest.raises(RuntimeError, match="torch for LlamaGen"):
        MLXStillEngine._validate_config(params, img, embedder=None)


# ---------------------------------------------------------------------------
# real checkpoint: round trip + grad on CPU and MPS
# ---------------------------------------------------------------------------


def _real_image(device):
    from huggingface_hub import hf_hub_download

    from pytti.config.model_names import LLAMAGEN_CHECKPOINT_FILES

    path = hf_hub_download(
        LLAMAGEN_HF_REPO,
        LLAMAGEN_CHECKPOINT_FILES["ds16"],
        revision=LLAMAGEN_HF_REVISION,
    )
    model = load_llamagen_model(path, "ds16").to(device)
    return LlamaGenImage(256, 256, model=model, device=device)


@pytest.mark.download
@pytest.mark.parametrize(
    "device",
    [
        "cpu",
        pytest.param(
            "mps",
            marks=pytest.mark.skipif(
                not torch.backends.mps.is_available(), reason="no MPS"
            ),
        ),
    ],
)
def test_ds16_roundtrip_and_grad(device):
    from PIL import Image

    img = _real_image(device)
    assert tuple(img.tensor.shape) == (1, 16, 16, 8)  # (1, toksY, toksX, 8)

    # encode -> decode round trip on a smooth synthetic image
    grad_img = Image.new("RGB", (256, 256))
    grad_img.putdata(
        [(x, y, (x + y) // 2) for y in range(256) for x in range(256)]
    )
    img.encode_image(grad_img)
    out = img.decode_training_tensor()
    assert tuple(out.shape) == (1, 3, 256, 256)

    import numpy as np
    from torchvision.transforms import functional as TF

    target = TF.to_tensor(grad_img).unsqueeze(0).to(device)
    mse = float(F.mse_loss(out.detach(), target))
    assert mse < 0.02, f"reconstruction MSE {mse} — decode is not round-tripping"
    assert float(out.detach().std()) > 0.05, "decode collapsed to a flat image"
    assert np.isfinite(mse)

    # grad reaches z through decoder + post_quant_conv + STE + F.normalize
    out.mean().backward()
    g = img.tensor.grad
    assert g is not None and torch.isfinite(g).all() and g.abs().sum() > 0


@pytest.mark.download
@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="no MPS")
def test_ds16_mps_cpu_decode_parity():
    torch.manual_seed(4)
    cpu_img = _real_image("cpu")
    with torch.no_grad():
        z = cpu_img.tensor.detach().clone()
        out_cpu = cpu_img.decode(z)

    mps_img = _real_image("mps")
    with torch.no_grad():
        mps_img.tensor.set_(z.to("mps"))
        out_mps = mps_img.decode(mps_img.tensor)

    diff = float((out_mps.cpu() - out_cpu).abs().max())
    # fp32 conv/GroupNorm/attention stack: MPS should track CPU closely
    assert diff < 5e-3, f"MPS-vs-CPU decode diff {diff}"
