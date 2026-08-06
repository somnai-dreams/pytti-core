"""
init_spectrum_chroma (structured_config): cross-channel structure of the
shaped inits. Per-channel-independent shaped fields ('full') leave
low-frequency COLOR blobs that CLIP never cleans up and that steer the
final palette; 'natural' follows lucid's ImageNet color statistics (mostly
luma, faint chroma); 'mono' is one luminance field, zero chroma.

'full' must stay bit-for-bit the pre-knob behavior INCLUDING the RNG
stream: pinned against tensors captured from the pristine code at 08ab910
(tests/fixtures/init_chroma_full_pin.pt). Chroma shapes only the color
axis — the spatial spectrum contract of test_init_spectrum.py holds for
every mode.
"""

import math
from pathlib import Path

import pytest
import torch
from hydra import compose, initialize
from omegaconf import OmegaConf

from pytti.config.structured_config import ConfigSchema
from pytti.image_models import LlamaGenImage, VQGANImage
from pytti.image_models.init_noise import (
    NATURAL_TENSOR_AMPLITUDE,
    shaped_init_field,
)
from tests.test_init_spectrum import _loglog_amplitude_slope, make_pixel, make_rgb

DEVICE = "cpu"
SHAPED = ["pink", "fractal"]
CHROMA = ["mono", "natural", "full"]

# captured from the pristine pre-knob code (commit 08ab910) under
# torch.manual_seed(1234): encode_random on a 64x48 RGBImage and a 64x48/3-
# palette/4-color PixelImage (random_palette=True), for pink and fractal
_PIN_PATH = Path(__file__).parent / "fixtures" / "init_chroma_full_pin.pt"
_PIN = torch.load(_PIN_PATH, weights_only=True)


def _pairwise_channel_corr(field: torch.Tensor) -> list[float]:
    """[corr(R,G), corr(R,B), corr(G,B)] of a [3, H, W] field."""
    corr = torch.corrcoef(field.reshape(3, -1))
    return [corr[0, 1].item(), corr[0, 2].item(), corr[1, 2].item()]


# ----------------------------------------------------------------------
# 'full' is the pre-knob behavior bit-for-bit, RNG stream included
# ----------------------------------------------------------------------


@pytest.mark.parametrize("spectrum", SHAPED)
def test_full_rgb_is_bitforbit_the_preknob_init(spectrum):
    img = make_rgb()
    torch.manual_seed(1234)
    img.encode_random(init_spectrum=spectrum, init_spectrum_chroma="full")
    assert torch.equal(img.tensor.detach(), _PIN[f"rgb_{spectrum}"])

    # and 'full' is the default, so the running battery's legs reproduce
    torch.manual_seed(1234)
    img.encode_random(init_spectrum=spectrum)
    assert torch.equal(img.tensor.detach(), _PIN[f"rgb_{spectrum}"])


@pytest.mark.parametrize("spectrum", SHAPED)
def test_full_pixel_is_bitforbit_the_preknob_init(spectrum):
    img = make_pixel()
    torch.manual_seed(1234)
    img.encode_random(
        random_palette=True, init_spectrum=spectrum, init_spectrum_chroma="full"
    )
    assert torch.equal(img.value.detach(), _PIN[f"pixel_{spectrum}_value"])
    assert torch.equal(img.tensor.detach(), _PIN[f"pixel_{spectrum}_tensor"])
    # the palette draw comes AFTER the field draws: equality proves the
    # whole RNG stream is consumed identically, not just the fields
    assert torch.equal(img.palette.detach(), _PIN[f"pixel_{spectrum}_palette"])


# ----------------------------------------------------------------------
# 'mono': one luminance field, zero chroma
# ----------------------------------------------------------------------


@pytest.mark.parametrize("spectrum", SHAPED)
def test_mono_rgb_has_exactly_zero_chroma(spectrum):
    img = make_rgb()
    torch.manual_seed(4)
    img.encode_random(init_spectrum=spectrum, init_spectrum_chroma="mono")
    t = img.tensor.detach().squeeze(0)
    assert torch.equal(t[0], t[1])
    assert torch.equal(t[1], t[2])
    # a real shaped field around mid-gray, not a constant plane
    assert t[0].std().item() > 0.01
    assert abs(t[0].mean().item() - 0.5) < 0.05


# ----------------------------------------------------------------------
# 'natural': lucid color statistics — mostly luma, faint chroma
# ----------------------------------------------------------------------


def test_natural_rgb_channel_correlation_is_high_while_full_is_near_zero():
    torch.manual_seed(6)
    natural = shaped_init_field(
        3, 256, 256, "pink", 1.0, DEVICE, init_spectrum_chroma="natural"
    )
    pairs = _pairwise_channel_corr(natural)
    # lucid's covariance puts the expected correlations at ~0.91 (R-G),
    # ~0.79 (R-B), ~0.91 (G-B): mean is the headline bound, min is slack
    # for the R-B pair plus pink-field sampling noise
    assert sum(pairs) / 3 > 0.8
    assert min(pairs) > 0.65

    torch.manual_seed(6)
    full = shaped_init_field(
        3, 256, 256, "pink", 1.0, DEVICE, init_spectrum_chroma="full"
    )
    # independent fields: correlation is sampling noise around 0 (pink
    # fields have few effective low-frequency samples, hence the loose cap)
    assert max(abs(c) for c in _pairwise_channel_corr(full)) < 0.4


def test_chroma_modes_produce_distinct_rgb_fields():
    fields = []
    for chroma in CHROMA:
        img = make_rgb()
        torch.manual_seed(17)
        img.encode_random(init_spectrum="pink", init_spectrum_chroma=chroma)
        fields.append(img.tensor.detach().clone())
    assert not torch.equal(fields[0], fields[1])
    assert not torch.equal(fields[1], fields[2])
    assert not torch.equal(fields[0], fields[2])


# ----------------------------------------------------------------------
# chroma never touches the spatial spectrum
# ----------------------------------------------------------------------


@pytest.mark.parametrize("chroma", CHROMA)
def test_pink_slope_is_unaffected_by_chroma_mode(chroma):
    torch.manual_seed(5)
    field = shaped_init_field(
        3, 256, 256, "pink", 1.0, DEVICE, init_spectrum_chroma=chroma
    )
    for c in range(3):
        slope = _loglog_amplitude_slope(field[c : c + 1])
        assert abs(slope + 1.0) < 0.25


@pytest.mark.parametrize("chroma", CHROMA)
def test_shaped_field_moments_hold_in_every_chroma_mode(chroma):
    torch.manual_seed(5)
    field = shaped_init_field(
        3, 256, 256, "fractal", 1.0, DEVICE, init_spectrum_chroma=chroma
    )
    assert field.shape == (3, 256, 256)
    assert field.min().item() >= 0.0
    assert field.max().item() <= 1.0
    assert abs(field.mean().item() - 0.5) < 0.02
    assert abs(field.std().item() - 1 / math.sqrt(12)) < 0.05


# ----------------------------------------------------------------------
# PixelImage: chroma governs selection-logit amplitude, not a color basis
# ----------------------------------------------------------------------


def test_pixel_mono_keeps_the_original_uniform_tensor_draw():
    img = make_pixel()
    torch.manual_seed(31)
    img.encode_random(init_spectrum="pink", init_spectrum_chroma="mono")

    # replicate the stream: the (kept, full-strength) value field consumes
    # the RNG first, then the ORIGINAL uniform draw fills the logits
    torch.manual_seed(31)
    value_ref = shaped_init_field(1, 48, 64, "pink", 1.0, DEVICE)
    tensor_ref = torch.zeros(3, 48, 64)
    tensor_ref.uniform_()
    assert torch.equal(img.value.detach(), value_ref.squeeze(0))
    assert torch.equal(img.tensor.detach(), tensor_ref)


def test_pixel_mono_tensor_distribution_is_plain_uniform():
    img = make_pixel()
    torch.manual_seed(31)
    img.encode_random(init_spectrum="pink", init_spectrum_chroma="mono")
    t = img.tensor.detach().flatten()
    assert abs(t.mean().item() - 0.5) < 0.02
    assert abs(t.std().item() - 1 / math.sqrt(12)) < 0.01
    # KS-style sanity bound: sup |empirical CDF - U(0,1) CDF|. The uniform
    # noise floor at n=9216 is ~0.014 at the 1% level; a shaped
    # (clamped-gaussian) plane sneaking in here would show D ~ 0.055
    n = t.numel()
    grid = (torch.arange(1, n + 1, dtype=t.dtype) - 0.5) / n
    d = (t.sort().values - grid).abs().max().item() + 0.5 / n
    assert d < 0.03


def test_pixel_natural_is_the_full_logits_blended_toward_mid_gray():
    img = make_pixel()
    torch.manual_seed(31)
    img.encode_random(init_spectrum="pink", init_spectrum_chroma="full")
    full_tensor = img.tensor.detach().clone()
    full_value = img.value.detach().clone()

    torch.manual_seed(31)
    img.encode_random(init_spectrum="pink", init_spectrum_chroma="natural")
    # natural consumes the exact same RNG stream as full: the logits are
    # the identical draw shrunk toward flat mid-gray, the value plane is
    # identical (full spatial prior on brightness in every mode)
    assert torch.equal(
        img.tensor.detach(),
        0.5 + NATURAL_TENSOR_AMPLITUDE * (full_tensor - 0.5),
    )
    assert torch.equal(img.value.detach(), full_value)


def test_pixel_natural_tensor_amplitude_is_reduced():
    img = make_pixel()
    torch.manual_seed(31)
    img.encode_random(init_spectrum="pink", init_spectrum_chroma="natural")
    t = img.tensor.detach()
    for plane in t:
        assert abs(plane.mean().item() - 0.5) < 0.02
        # per-plane std ~ NATURAL_TENSOR_AMPLITUDE / sqrt(12) (the full
        # field is clamped before the blend, so slightly below)
        assert abs(plane.std().item() - NATURAL_TENSOR_AMPLITUDE / math.sqrt(12)) < 0.02


@pytest.mark.parametrize("chroma", CHROMA)
def test_pixel_random_palette_still_draws_under_every_chroma(chroma):
    img = make_pixel()
    before = img.palette.detach().clone()
    torch.manual_seed(9)
    img.encode_random(
        random_palette=True, init_spectrum="pink", init_spectrum_chroma=chroma
    )
    assert not torch.equal(img.palette.detach(), before)
    assert img.palette.detach().max().item() <= img.palette_inertia


# ----------------------------------------------------------------------
# 'white' and 'gray' ignore the knob (documented): nothing to de-chroma
# ----------------------------------------------------------------------


@pytest.mark.parametrize("spectrum", ["white", "gray"])
def test_white_and_gray_are_identical_across_chroma_modes(spectrum):
    outs = []
    for chroma in CHROMA:
        img = make_rgb()
        torch.manual_seed(13)
        img.encode_random(init_spectrum=spectrum, init_spectrum_chroma=chroma)
        outs.append(img.tensor.detach().clone())
    assert torch.equal(outs[0], outs[1])
    assert torch.equal(outs[1], outs[2])


# ----------------------------------------------------------------------
# determinism: global-generator draws only, per-seed variation preserved
# ----------------------------------------------------------------------


@pytest.mark.parametrize("chroma", ["mono", "natural"])
@pytest.mark.parametrize("spectrum", SHAPED)
def test_rgb_chroma_variants_are_deterministic(spectrum, chroma):
    img = make_rgb()
    torch.manual_seed(21)
    img.encode_random(init_spectrum=spectrum, init_spectrum_chroma=chroma)
    first = img.tensor.detach().clone()

    torch.manual_seed(21)
    img.encode_random(init_spectrum=spectrum, init_spectrum_chroma=chroma)
    assert torch.equal(img.tensor.detach(), first)

    torch.manual_seed(22)
    img.encode_random(init_spectrum=spectrum, init_spectrum_chroma=chroma)
    assert not torch.equal(img.tensor.detach(), first)


@pytest.mark.parametrize("chroma", ["mono", "natural"])
def test_pixel_chroma_variants_are_deterministic(chroma):
    img = make_pixel()
    torch.manual_seed(21)
    img.encode_random(init_spectrum="pink", init_spectrum_chroma=chroma)
    value = img.value.detach().clone()
    logits = img.tensor.detach().clone()

    torch.manual_seed(21)
    img.encode_random(init_spectrum="pink", init_spectrum_chroma=chroma)
    assert torch.equal(img.value.detach(), value)
    assert torch.equal(img.tensor.detach(), logits)

    torch.manual_seed(22)
    img.encode_random(init_spectrum="pink", init_spectrum_chroma=chroma)
    assert not torch.equal(img.value.detach(), value)


# ----------------------------------------------------------------------
# fail-loud validation: unknown chroma, wrong channel count, token models
# ----------------------------------------------------------------------


def test_shaped_init_field_rejects_unknown_chroma():
    with pytest.raises(ValueError, match="init_spectrum_chroma='vivid'"):
        shaped_init_field(3, 8, 8, "pink", 1.0, DEVICE, init_spectrum_chroma="vivid")


def test_natural_demands_exactly_three_channels():
    with pytest.raises(ValueError, match=r"natural.*3.*channels"):
        shaped_init_field(4, 8, 8, "pink", 1.0, DEVICE, init_spectrum_chroma="natural")


def test_encode_random_rejects_unknown_chroma_even_for_white():
    img = make_rgb()
    with pytest.raises(ValueError, match="init_spectrum_chroma='vivid'"):
        img.encode_random(init_spectrum="white", init_spectrum_chroma="vivid")
    pix = make_pixel()
    with pytest.raises(ValueError, match="init_spectrum_chroma='vivid'"):
        pix.encode_random(init_spectrum="white", init_spectrum_chroma="vivid")


def test_token_models_accept_the_kwarg_and_still_reject_shaped_init():
    # __new__ skips weight loading: the guard must fire before any state
    img = object.__new__(VQGANImage)
    with pytest.raises(ValueError, match=r"VQGANImage.*'white'"):
        VQGANImage.encode_random(img, init_spectrum="pink", init_spectrum_chroma="mono")
    img = object.__new__(LlamaGenImage)
    with pytest.raises(ValueError, match=r"LlamaGenImage.*'white'"):
        LlamaGenImage.encode_random(
            img, init_spectrum="fractal", init_spectrum_chroma="natural"
        )


# ----------------------------------------------------------------------
# config schema
# ----------------------------------------------------------------------


def test_config_default_chroma_is_full():
    with initialize(config_path="config", version_base=None):
        cfg = compose(config_name="_structured_config", overrides=["scenes=x"])
    obj = OmegaConf.to_object(cfg)
    assert isinstance(obj, ConfigSchema)
    assert obj.init_spectrum_chroma == "full"


@pytest.mark.parametrize("chroma", CHROMA)
def test_config_accepts_every_documented_chroma(chroma):
    with initialize(config_path="config", version_base=None):
        cfg = compose(
            config_name="_structured_config",
            overrides=["scenes=x", f"init_spectrum_chroma={chroma}"],
        )
    obj = OmegaConf.to_object(cfg)
    assert isinstance(obj, ConfigSchema)
    assert obj.init_spectrum_chroma == chroma


def test_config_rejects_unknown_chroma():
    with initialize(config_path="config", version_base=None):
        cfg = compose(
            config_name="_structured_config",
            overrides=["scenes=x", "init_spectrum_chroma=vivid"],
        )
    with pytest.raises(ValueError, match="init_spectrum_chroma"):
        OmegaConf.to_object(cfg)
