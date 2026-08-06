"""
init_spectrum (structured_config): the init-noise spectrum knob for the
no-init_image start. 'white' must stay bit-for-bit the historical uniform
init; 'gray'/'pink'/'fractal' are shaped fields from
image_models/init_noise.py; VQGAN/LlamaGen reject shaped init loudly
(categorical token init has no spectrum to shape).

Spectral checks are sanity bounds, not metrology: coarse tolerances on the
radially-fit log-log slope and band-energy monotonicity.
"""

import math

import pytest
import torch
from hydra import compose, initialize
from omegaconf import OmegaConf

from pytti.config.structured_config import ConfigSchema
from pytti.image_models import LlamaGenImage, PixelImage, RGBImage, VQGANImage
from pytti.image_models.init_noise import (
    MAX_SPECTRUM_FALLOFF,
    require_white_init,
    resolve_init_spectrum,
    shaped_init_field,
)

DEVICE = "cpu"
SHAPED = ["gray", "pink", "fractal"]


def make_rgb(width=64, height=48):
    return RGBImage(width, height, scale=1, device=DEVICE)


def make_pixel(width=64, height=48, n_palettes=3):
    return PixelImage(
        width=width,
        height=height,
        scale=1,
        palette_size=4,
        n_palettes=n_palettes,
        device=DEVICE,
    )


# ----------------------------------------------------------------------
# 'white' is the historical init, bit-for-bit
# ----------------------------------------------------------------------


def test_white_rgb_is_bitforbit_the_historical_uniform_init():
    img = make_rgb()
    torch.manual_seed(11)
    img.encode_random()  # default is white

    # the pre-knob code: a bare uniform_() on the same tensor, same layout
    reference = torch.zeros(1, 3, 48, 64).to(memory_format=torch.channels_last)
    torch.manual_seed(11)
    reference.uniform_()
    assert torch.equal(img.tensor.detach(), reference)

    # explicit 'white' is the same path as the default
    torch.manual_seed(11)
    img.encode_random(init_spectrum="white")
    assert torch.equal(img.tensor.detach(), reference)


def test_white_pixel_is_bitforbit_the_historical_uniform_init():
    img = make_pixel()
    torch.manual_seed(11)
    img.encode_random(random_palette=True)

    # the pre-knob code: value.uniform_() then tensor.uniform_() then the
    # palette draw, in that RNG order
    value_ref = torch.zeros(48, 64)
    tensor_ref = torch.zeros(3, 48, 64)
    palette_ref = torch.zeros_like(img.palette.detach())
    torch.manual_seed(11)
    value_ref.uniform_()
    tensor_ref.uniform_()
    palette_ref.uniform_(to=img.palette_inertia)
    assert torch.equal(img.value.detach(), value_ref)
    assert torch.equal(img.tensor.detach(), tensor_ref)
    assert torch.equal(img.palette.detach(), palette_ref)


# ----------------------------------------------------------------------
# 'gray': mid-gray plus one-bit symmetry breaking
# ----------------------------------------------------------------------


def test_gray_stays_in_its_tiny_band_around_mid_gray():
    torch.manual_seed(0)
    img = make_rgb()
    img.encode_random(init_spectrum="gray")
    t = img.tensor.detach()
    assert t.min().item() >= 0.5 - 0.5 / 255
    assert t.max().item() <= 0.5 + 0.5 / 255
    # symmetry breaking: not a constant plane
    assert t.std().item() > 0


# ----------------------------------------------------------------------
# 'pink': radially-averaged log-log amplitude slope ~ -alpha
# ----------------------------------------------------------------------


def _loglog_amplitude_slope(field: torch.Tensor, lo=4 / 256, hi=0.25) -> float:
    """Least-squares slope of log(|F|) vs log(f) over a mid-frequency band."""
    height, width = field.shape[-2:]
    amp = torch.fft.rfft2(field - field.mean()).abs().squeeze(0)
    fy = torch.fft.fftfreq(height)
    fx = torch.fft.rfftfreq(width)
    freqs = torch.sqrt(fy[:, None] ** 2 + fx[None, :] ** 2)
    mask = (freqs >= lo) & (freqs <= hi)
    logf = freqs[mask].log()
    loga = amp[mask].clamp_min(1e-12).log()
    x = logf - logf.mean()
    y = loga - loga.mean()
    return float((x * y).sum() / (x * x).sum())


@pytest.mark.parametrize("alpha", [0.5, 1.0, 2.0])
def test_pink_spectrum_slope_tracks_minus_alpha(alpha):
    torch.manual_seed(5)
    field = shaped_init_field(1, 256, 256, "pink", alpha, DEVICE)
    slope = _loglog_amplitude_slope(field)
    assert slope < 0  # amplitude decreases with frequency
    assert abs(slope + alpha) < 0.25  # coarse sanity bound (measured ~0.05)


# ----------------------------------------------------------------------
# 'fractal': power-law-ish band energy without an FFT synthesis
# ----------------------------------------------------------------------


def test_fractal_band_energy_decreases_monotonically():
    torch.manual_seed(5)
    field = shaped_init_field(1, 256, 256, "fractal", 1.0, DEVICE)
    power = torch.fft.rfft2(field - field.mean()).abs().squeeze(0) ** 2
    fy = torch.fft.fftfreq(256)
    fx = torch.fft.rfftfreq(256)
    freqs = torch.sqrt(fy[:, None] ** 2 + fx[None, :] ** 2)
    edges = [1 / 128, 1 / 32, 1 / 8, 1 / 4, 1 / 2]
    bands = []
    for lo, hi in zip(edges[:-1], edges[1:], strict=True):
        mask = (freqs >= lo) & (freqs < hi)
        bands.append(power[mask].mean().item())
    assert bands == sorted(bands, reverse=True)
    # measured margins are ~10x per band; require at least 2x
    assert all(a > 2 * b for a, b in zip(bands[:-1], bands[1:], strict=True))


def test_fractal_matches_uniform_moments():
    torch.manual_seed(5)
    field = shaped_init_field(3, 256, 256, "fractal", 1.0, DEVICE)
    assert field.min().item() >= 0.0
    assert field.max().item() <= 1.0
    assert abs(field.mean().item() - 0.5) < 0.02
    assert abs(field.std().item() - 1 / math.sqrt(12)) < 0.05


# ----------------------------------------------------------------------
# shapes, range, determinism — both trainable image models
# ----------------------------------------------------------------------


@pytest.mark.parametrize("spectrum", SHAPED)
def test_rgb_shaped_init_shape_range_determinism(spectrum):
    img = make_rgb()
    torch.manual_seed(21)
    img.encode_random(init_spectrum=spectrum)
    first = img.tensor.detach().clone()
    assert first.shape == (1, 3, 48, 64)
    assert first.min().item() >= 0.0
    assert first.max().item() <= 1.0

    torch.manual_seed(21)
    img.encode_random(init_spectrum=spectrum)
    assert torch.equal(img.tensor.detach(), first)

    torch.manual_seed(22)
    img.encode_random(init_spectrum=spectrum)
    assert not torch.equal(img.tensor.detach(), first)


@pytest.mark.parametrize("spectrum", SHAPED)
def test_pixel_shaped_init_shape_range_determinism(spectrum):
    img = make_pixel()
    torch.manual_seed(21)
    img.encode_random(init_spectrum=spectrum)
    value = img.value.detach().clone()
    logits = img.tensor.detach().clone()
    assert value.shape == (48, 64)
    assert logits.shape == (3, 48, 64)
    for t in (value, logits):
        assert t.min().item() >= 0.0
        assert t.max().item() <= 1.0

    torch.manual_seed(21)
    img.encode_random(init_spectrum=spectrum)
    assert torch.equal(img.value.detach(), value)
    assert torch.equal(img.tensor.detach(), logits)

    torch.manual_seed(22)
    img.encode_random(init_spectrum=spectrum)
    assert not torch.equal(img.value.detach(), value)


def test_pixel_logit_planes_are_independent_fields():
    img = make_pixel()
    torch.manual_seed(3)
    img.encode_random(init_spectrum="pink")
    logits = img.tensor.detach()
    assert not torch.equal(logits[0], logits[1])
    assert not torch.equal(logits[1], logits[2])
    # and the visible value plane is not any logit plane
    assert not torch.equal(img.value.detach(), logits[0])


def test_pixel_random_palette_still_draws_the_palette_under_shaped_init():
    img = make_pixel()
    before = img.palette.detach().clone()
    torch.manual_seed(9)
    img.encode_random(random_palette=True, init_spectrum="pink")
    assert not torch.equal(img.palette.detach(), before)
    assert img.palette.detach().max().item() <= img.palette_inertia


# ----------------------------------------------------------------------
# categorical-token models refuse shaped init loudly
# ----------------------------------------------------------------------


def test_vqgan_rejects_shaped_init_by_name():
    # __new__ skips weight loading: the guard must fire before any state
    img = object.__new__(VQGANImage)
    with pytest.raises(ValueError, match=r"VQGANImage.*'white'"):
        VQGANImage.encode_random(img, init_spectrum="pink")


def test_llamagen_rejects_shaped_init_by_name():
    img = object.__new__(LlamaGenImage)
    with pytest.raises(ValueError, match=r"LlamaGenImage.*'white'"):
        LlamaGenImage.encode_random(img, init_spectrum="fractal")


def test_require_white_init_passes_white():
    require_white_init("VQGANImage", "white")  # no raise


def test_shaped_init_field_rejects_white_and_unknown():
    with pytest.raises(ValueError, match="init_spectrum='white'"):
        shaped_init_field(1, 8, 8, "white", 1.0, DEVICE)
    with pytest.raises(ValueError, match="init_spectrum='magenta'"):
        shaped_init_field(1, 8, 8, "magenta", 1.0, DEVICE)


# ----------------------------------------------------------------------
# config schema
# ----------------------------------------------------------------------


def test_config_defaults_are_white_alpha_one():
    with initialize(config_path="config", version_base=None):
        cfg = compose(config_name="_structured_config", overrides=["scenes=x"])
    obj = OmegaConf.to_object(cfg)
    assert isinstance(obj, ConfigSchema)
    assert obj.init_spectrum == "white"
    assert obj.init_spectrum_falloff == 1.0


@pytest.mark.parametrize("spectrum", ["white", "gray", "pink", "fractal"])
def test_config_accepts_every_documented_spectrum(spectrum):
    with initialize(config_path="config", version_base=None):
        cfg = compose(
            config_name="_structured_config",
            overrides=["scenes=x", f"init_spectrum={spectrum}"],
        )
    obj = OmegaConf.to_object(cfg)
    assert isinstance(obj, ConfigSchema)
    assert obj.init_spectrum == spectrum


def test_config_rejects_unknown_spectrum():
    with initialize(config_path="config", version_base=None):
        cfg = compose(
            config_name="_structured_config",
            overrides=["scenes=x", "init_spectrum=magenta"],
        )
    with pytest.raises(ValueError, match="init_spectrum"):
        OmegaConf.to_object(cfg)


# ----------------------------------------------------------------------
# falloff range: validated at compose time, finite fields at the extremes
# ----------------------------------------------------------------------


@pytest.mark.parametrize("alpha", ["-0.5", "8.5", "20"])
def test_config_rejects_out_of_range_falloff(alpha):
    # 20 was the reviewer repro: it used to overflow float32 amplitudes and
    # irfft2 an all-NaN init field with no error raised
    with initialize(config_path="config", version_base=None):
        cfg = compose(
            config_name="_structured_config",
            overrides=["scenes=x", f"init_spectrum_falloff={alpha}"],
        )
    with pytest.raises(ValueError, match="init_spectrum_falloff"):
        OmegaConf.to_object(cfg)


@pytest.mark.parametrize("alpha", ["0", "8"])
def test_config_accepts_boundary_falloff(alpha):
    with initialize(config_path="config", version_base=None):
        cfg = compose(
            config_name="_structured_config",
            overrides=["scenes=x", f"init_spectrum_falloff={alpha}"],
        )
    obj = OmegaConf.to_object(cfg)
    assert isinstance(obj, ConfigSchema)
    assert obj.init_spectrum_falloff == float(alpha)


def test_pink_synthesis_guards_the_same_range_for_direct_callers():
    with pytest.raises(ValueError, match="init_spectrum_falloff"):
        shaped_init_field(1, 32, 32, "pink", -1.0, DEVICE)
    with pytest.raises(ValueError, match="init_spectrum_falloff"):
        shaped_init_field(1, 32, 32, "pink", MAX_SPECTRUM_FALLOFF + 1, DEVICE)


def test_pink_field_is_finite_and_nonconstant_at_the_max_falloff():
    # locks the peak-1 amplitude normalization: unnormalized f_min^-alpha
    # used to overflow float32 variance on big canvases at high alpha,
    # collapsing the field to a silently constant 0.5
    torch.manual_seed(7)
    field = shaped_init_field(1, 256, 1024, "pink", MAX_SPECTRUM_FALLOFF, DEVICE)
    assert torch.isfinite(field).all()
    assert field.std().item() > 0.01
    assert field.min().item() >= 0.0
    assert field.max().item() <= 1.0


# ----------------------------------------------------------------------
# the knob only governs a visible random start (no init_image, no restore)
# ----------------------------------------------------------------------


def test_resolve_passes_the_knob_through_when_the_start_is_visible():
    assert resolve_init_spectrum(
        "pink", 2.0, has_init_image=False, restore=False
    ) == ("pink", 2.0)


def test_resolve_collapses_to_white_when_an_init_image_takes_over():
    # a VQGAN + init_image + pink config is valid: the shaped start would
    # never be shown, so the categorical-model guard must not fire
    assert resolve_init_spectrum(
        "pink", 2.0, has_init_image=True, restore=False
    ) == ("white", 1.0)


def test_resolve_collapses_to_white_on_restore():
    assert resolve_init_spectrum(
        "fractal", 1.0, has_init_image=False, restore=True
    ) == ("white", 1.0)


# ----------------------------------------------------------------------
# workhorse rejects shaped init for token models BEFORE any model download
# ----------------------------------------------------------------------


def _minimal_params(**overrides):
    # deliberately omits models_parent_dir / vqgan_model / llamagen_model:
    # if the guard fired after the model-load code, these tests would die
    # on AttributeError instead of the guard's ValueError
    from types import SimpleNamespace

    fields = dict(
        init_spectrum="pink",
        init_spectrum_falloff=1.0,
        init_spectrum_chroma="full",
        # read by the fourier validation at configure_pass top (fires
        # before the token-model guard these tests target)
        fourier_parameterization=False,
        fourier_decay=1.0,
        perceptor_backend="torch",
        structure_annealing=False,
        animation_mode="off",
    )
    fields.update(overrides)
    return SimpleNamespace(**fields)


def test_workhorse_rejects_vqgan_shaped_init_before_model_load():
    from pytti.workhorse import configure_pass

    with pytest.raises(ValueError, match=r"VQGANImage.*'white'"):
        configure_pass(
            _minimal_params(image_model="VQGAN"),
            device="cpu",
            embedder=None,
            prompts=None,
            init_image_pil=None,
            video_frames=None,
            restore=False,
        )


def test_workhorse_rejects_llamagen_shaped_init_before_model_load():
    from pytti.workhorse import configure_pass

    with pytest.raises(ValueError, match=r"LlamaGenImage.*'white'"):
        configure_pass(
            _minimal_params(image_model="LlamaGen", perceptor_backend="torch"),
            device="cpu",
            embedder=None,
            prompts=None,
            init_image_pil=None,
            video_frames=None,
            restore=False,
        )
