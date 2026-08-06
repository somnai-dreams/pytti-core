"""
fourier_parameterization (structured_config): optimize the Unlimited
Palette image as a 1/f-scaled Fourier spectrum (lucid fft_image) instead of
raw pixels — src/pytti/image_models/fourier.py. v1 scope is Unlimited
Palette + torch|mlx_full backend + stills + white init; everything else fails loud.

The numeric contracts tested here: decode stays in [0, 1]; the scale grid
is the lucid formula's SHAPE at every decay with its induced energy pinned
to the decay-1 (lucid-exact) level, so any decay in range starts near-gray
and keeps optimizing (the 2026-08 saturation-freeze regression); the color
matrix is THE shared init_noise constant with basis->RGB orientation (a
pure-luma basis vector decodes near-gray, not rainbow); encode/decode
round-trips at multiple sizes including non-square; seeds are
deterministic; .bak state_dicts round-trip the spectrum bit-for-bit.
"""

import math
from typing import Any, cast

import pytest
import torch
from hydra import compose, initialize
from omegaconf import OmegaConf

from pytti.config.structured_config import ConfigSchema
from pytti.image_models import FourierImage
from pytti.image_models.fourier import (
    FOURIER_DECAY_MAX,
    FOURIER_DECAY_MIN,
    FOURIER_INIT_SD,
    LUCID_OUTPUT_DIVISOR,
    _hermitian_variance_weights,
    fourier_scale,
    validate_fourier_decay,
    validate_fourier_parameterization,
)
from pytti.image_models.init_noise import imagenet_color_matrix

DEVICE = "cpu"

# measured (2026-08, cpu float32): tensor-level encode->decode round-trip
# max error is ~3e-7 across all tested sizes; 1e-5 gives float-noise
# headroom without hiding a real regression
ROUNDTRIP_ATOL = 1e-5


def make(width=64, height=48, scale=1, decay=1.0):
    return FourierImage(width, height, scale=scale, decay=decay, device=DEVICE)


def valid_config(**overrides: Any) -> dict[str, Any]:
    """The full keyword surface of validate_fourier_parameterization."""
    config: dict[str, Any] = dict(
        fourier_parameterization=True,
        fourier_decay=1.0,
        image_model="Unlimited Palette",
        perceptor_backend="torch",
        init_spectrum="white",
        structure_annealing=False,
        animation_mode="off",
    )
    config.update(overrides)
    return config


# ----------------------------------------------------------------------
# shapes, dtypes, decode range
# ----------------------------------------------------------------------


def test_spectrum_shapes_and_dtypes():
    img = make(width=64, height=48)
    assert tuple(img.spectrum_real.shape) == (3, 48, 33)  # rfft2 layout
    assert tuple(img.spectrum_imag.shape) == (3, 48, 33)
    assert img.spectrum_real.dtype == torch.float32
    assert img.spectrum_imag.dtype == torch.float32
    assert img.spectrum_real.requires_grad and img.spectrum_imag.requires_grad
    assert tuple(img.spectrum_scale.shape) == (48, 33)


def test_decode_shape_and_range():
    img = make(width=64, height=48)
    torch.manual_seed(0)
    img.encode_random()
    out = img.decode_tensor()
    assert tuple(out.shape) == (1, 3, 48, 64)
    assert out.dtype == torch.float32
    assert out.min().item() >= 0.0 and out.max().item() <= 1.0


def test_pixel_size_upsamples_like_rgb_image():
    img = make(width=32, height=24, scale=2)
    assert img.image_shape == (64, 48)
    torch.manual_seed(0)
    img.encode_random()
    out = img.decode_tensor()
    assert tuple(out.shape) == (1, 3, 48, 64)
    # spectrum stays on the logical grid
    assert tuple(img.spectrum_real.shape) == (3, 24, 17)


def test_decode_is_differentiable_to_the_spectrum():
    img = make()
    torch.manual_seed(0)
    img.encode_random()
    img.decode_tensor().mean().backward()
    assert img.spectrum_real.grad is not None
    assert img.spectrum_imag.grad is not None
    assert img.spectrum_real.grad.abs().sum().item() > 0


# ----------------------------------------------------------------------
# the 1/f scale grid
# ----------------------------------------------------------------------


def test_scale_grid_matches_lucid_formula_at_decay_one_non_square():
    # decay 1 is the energy anchor: the normalization gain is exactly 1
    # there, so these are lucid's raw formula values
    height, width, decay = 48, 96, 1.0
    scale = fourier_scale(height, width, decay, DEVICE)
    assert tuple(scale.shape) == (height, width // 2 + 1)
    f_min = 1.0 / max(height, width)
    norm = math.sqrt(height * width)
    # DC is clamped to the lowest representable frequency
    assert scale[0, 0].item() == pytest.approx(f_min**-decay * norm, rel=1e-5)
    # a pure-fy bin: freq = fy = 1/height
    assert scale[1, 0].item() == pytest.approx((1 / height) ** -decay * norm, rel=1e-5)
    # a pure-fx bin: freq = fx = 1/width, BELOW f_min on this non-square
    # canvas? no: 1/96 < 1/96 is false — 1/width == f_min exactly here
    assert scale[0, 1].item() == pytest.approx((1 / width) ** -decay * norm, rel=1e-5)
    # the Nyquist corner: freq = sqrt(0.5^2 + 0.5^2)
    assert scale[height // 2, -1].item() == pytest.approx(
        math.sqrt(0.5) ** -decay * norm, rel=1e-5
    )
    # monotone: lower frequency -> strictly larger amplitude scale
    assert scale[1, 0].item() > scale[2, 0].item() > scale[height // 2, 0].item()


def test_scale_grid_is_lucid_shape_times_one_common_gain():
    # at decay != 1 every bin is lucid's raw formula value times a SINGLE
    # energy-normalization scalar — shape preserved exactly
    height, width, decay = 48, 96, 1.5
    scale = fourier_scale(height, width, decay, DEVICE)
    f_min = 1.0 / max(height, width)
    norm = math.sqrt(height * width)
    gain = scale[0, 0].item() / (f_min**-decay * norm)
    # decay > 1 concentrates raw energy into low frequencies, so the
    # normalizing gain must shrink it back
    assert 0.0 < gain < 1.0
    for (ky, kx), freq in [
        ((1, 0), 1 / height),
        ((0, 1), 1 / width),
        ((height // 2, -1), math.sqrt(0.5)),
    ]:
        assert scale[ky, kx].item() == pytest.approx(
            gain * freq**-decay * norm, rel=1e-5
        )
    assert scale[1, 0].item() > scale[2, 0].item() > scale[height // 2, 0].item()


def test_scale_energy_is_decay_invariant():
    # the root-cause fix for the 2026-08 saturation-freeze finding: the
    # Hermitian-weighted energy (which sets the induced init variance AND
    # the optimizer's total image-space step budget) is pinned to the
    # decay-1 level across the whole validated range
    height, width = 48, 96
    weights = _hermitian_variance_weights(height, width, DEVICE).double()

    def energy(decay: float) -> float:
        scale = fourier_scale(height, width, decay, DEVICE)
        return (weights * scale.double() ** 2).sum().item()

    reference = energy(1.0)
    for decay in (FOURIER_DECAY_MIN, 0.5, 2.0, 3.0, FOURIER_DECAY_MAX):
        assert energy(decay) == pytest.approx(reference, rel=1e-6)


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="MPS-only")
def test_scale_builds_on_mps():
    # regression (2026-08 smoke): the float64 normalization sums must run
    # on CPU — MPS has no float64, and construction crashed on device
    scale = fourier_scale(64, 48, 3.0, "mps")
    assert scale.device.type == "mps"
    assert torch.allclose(scale.cpu(), fourier_scale(64, 48, 3.0, "cpu"), rtol=1e-5)


def test_higher_decay_boosts_low_frequencies_more():
    soft = fourier_scale(64, 64, 2.0, DEVICE)
    hard = fourier_scale(64, 64, 1.0, DEVICE)
    # relative low/high frequency ratio grows with decay (the global
    # energy normalization cancels in the ratio)
    assert (soft[1, 0] / soft[32, 0]).item() > (hard[1, 0] / hard[32, 0]).item()


# ----------------------------------------------------------------------
# color matrix: shared constant, basis->RGB orientation
# ----------------------------------------------------------------------


def test_color_matrix_is_the_shared_init_noise_constant():
    img = make()
    expected = imagenet_color_matrix(torch.float32, DEVICE)
    assert torch.equal(img.color_matrix, expected)
    assert torch.allclose(
        img.color_matrix_inv @ img.color_matrix, torch.eye(3), atol=1e-6
    )


def test_pure_luma_basis_vector_decodes_near_gray():
    img = make(width=64, height=64)
    with torch.no_grad():
        img.spectrum_real.zero_()
        img.spectrum_imag.zero_()
        img.spectrum_real[0, 0, 0] = 0.5  # DC of basis axis 0 = luma
    decoded = img._decode_logical()
    channel_means = decoded.mean(dim=(1, 2))
    spread = (channel_means.max() - channel_means.min()).item()
    shift = (channel_means.mean() - 0.5).abs().item()
    # the luma axis moved brightness well clear of the residual chroma
    assert shift > 0.01
    assert spread < shift / 10  # near-gray: chroma is a rounding error


def test_chroma_basis_vector_decodes_colorful():
    img = make(width=64, height=64)
    with torch.no_grad():
        img.spectrum_real.zero_()
        img.spectrum_imag.zero_()
        img.spectrum_real[1, 0, 0] = 0.5  # a chroma axis
    channel_means = img._decode_logical().mean(dim=(1, 2))
    spread = (channel_means.max() - channel_means.min()).item()
    assert spread > 0.01  # visibly not gray — orientation is basis->RGB


# ----------------------------------------------------------------------
# encode/decode round-trip
# ----------------------------------------------------------------------


@pytest.mark.parametrize("width,height", [(128, 128), (96, 160), (224, 224)])
def test_tensor_roundtrip(width, height):
    img = make(width=width, height=height)
    torch.manual_seed(3)
    # stay clear of the logit clamp: content in [0.05, 0.95]
    pixels = torch.rand(3, height, width) * 0.9 + 0.05
    img.set_image_tensor(pixels)
    decoded = img.get_image_tensor()
    assert tuple(decoded.shape) == (3, height, width)
    assert torch.allclose(decoded, pixels, atol=ROUNDTRIP_ATOL)


@pytest.mark.parametrize("width,height", [(128, 128), (96, 160)])
def test_decode_encode_decode_is_stable(width, height):
    # the coarse_to_fine seam semantics: re-encoding a decoded image must
    # reproduce the image (the free spectrum's redundant Hermitian pairs
    # are projected by irfft2, so the IMAGE round-trips, canonically)
    img = make(width=width, height=height)
    torch.manual_seed(5)
    img.encode_random()
    first = img.get_image_tensor().detach().clone()
    img.set_image_tensor(first)
    second = img.get_image_tensor()
    assert torch.allclose(first, second, atol=ROUNDTRIP_ATOL)


def test_pil_roundtrip_within_8bit_tolerance():
    img = make(width=96, height=64)
    torch.manual_seed(4)
    img.encode_random()
    frame = img.decode_image()  # PIL, 8-bit quantized
    img.encode_image(frame)
    redecoded = img.decode_image()
    a = torch.frombuffer(bytearray(frame.tobytes()), dtype=torch.uint8).float()
    b = torch.frombuffer(bytearray(redecoded.tobytes()), dtype=torch.uint8).float()
    # one quantization step of slack on top of the logit clamp
    assert (a - b).abs().max().item() <= 2.0


def test_extreme_pixels_survive_the_logit_clamp():
    img = make(width=32, height=32)
    pixels = torch.zeros(3, 32, 32)
    pixels[1] = 1.0
    img.set_image_tensor(pixels)
    decoded = img.get_image_tensor()
    assert torch.allclose(decoded, pixels, atol=1e-3)  # eps-clamp bound


def test_write_pixels_rejects_wrong_shape():
    img = make(width=64, height=48)
    with pytest.raises(ValueError, match=r"\[3, 48, 64\]"):
        img.set_image_tensor(torch.rand(3, 64, 48))


# ----------------------------------------------------------------------
# encode_random: lucid init, white-only, seed determinism
# ----------------------------------------------------------------------


def test_encode_random_is_seed_deterministic():
    a, b = make(), make()
    torch.manual_seed(99)
    a.encode_random()
    torch.manual_seed(99)
    b.encode_random()
    assert torch.equal(a.spectrum_real, b.spectrum_real)
    assert torch.equal(a.spectrum_imag, b.spectrum_imag)
    assert torch.equal(a.decode_tensor(), b.decode_tensor())


def test_encode_random_draws_lucid_sd():
    img = make(width=256, height=256)
    torch.manual_seed(1)
    img.encode_random()
    assert img.spectrum_real.std().item() == pytest.approx(FOURIER_INIT_SD, rel=0.05)
    assert img.spectrum_imag.std().item() == pytest.approx(FOURIER_INIT_SD, rel=0.05)


def test_encode_random_rejects_shaped_init_spectrum():
    img = make()
    with pytest.raises(ValueError, match="1/f-shaped by construction"):
        img.encode_random(init_spectrum="pink")


def test_encode_random_validates_chroma_like_every_model():
    img = make()
    with pytest.raises(ValueError, match="init_spectrum_chroma"):
        img.encode_random(init_spectrum_chroma="bogus")


@pytest.mark.parametrize(
    "decay", [FOURIER_DECAY_MIN, 1.0, 2.0, 3.0, FOURIER_DECAY_MAX]
)
@pytest.mark.parametrize("size", [128, 256])
def test_init_decodes_near_gray(decay, size):
    # the documented lucid start: sd 0.01 spectra decode to a near-gray
    # field, composition head-start comes from the scale not the init —
    # and (2026-08 regression) it must hold across the WHOLE validated
    # decay range and across sizes: pre-normalization, decay 3+ initialized
    # 94-100% sigmoid-saturated
    img = make(width=size, height=size, decay=decay)
    torch.manual_seed(2)
    img.encode_random()
    decoded = img.decode_tensor()
    assert (decoded - 0.5).abs().max().item() < 0.1


@pytest.mark.parametrize("decay", [3.0, FOURIER_DECAY_MAX])
def test_high_decay_optimizes_instead_of_freezing(decay):
    # THE 2026-08 review finding: without the scale-energy normalization,
    # decay >= 3 initialized fully sigmoid-saturated and no optimizer step
    # could escape — 200 Adam steps left the loss frozen at 0.2500 while
    # the run 'succeeded'. Guard the fix at the model's own default lr: a
    # short Adam run must make real progress and stay unsaturated.
    size = 128
    img = make(width=size, height=size, decay=decay)
    torch.manual_seed(42)
    img.encode_random()
    optimizer = torch.optim.Adam(img.parameters(), lr=img.lr)
    target = torch.full((3, size, size), 0.75)
    first = math.inf
    for step in range(50):
        optimizer.zero_grad()
        loss = ((img._decode_logical() - target) ** 2).mean()
        if step == 0:
            first = loss.item()
        loss.backward()
        optimizer.step()
    with torch.no_grad():
        decoded = img._decode_logical()
    final = ((decoded - target) ** 2).mean().item()
    saturated = ((decoded < 1e-3) | (decoded > 1 - 1e-3)).float().mean().item()
    assert final < first * 0.5  # frozen would be final == first
    assert saturated < 0.01


# ----------------------------------------------------------------------
# clone, save/restore
# ----------------------------------------------------------------------


def test_clone_copies_spectrum_and_decay():
    img = make(width=64, height=48, decay=2.0)
    torch.manual_seed(6)
    img.encode_random()
    dup = img.clone()
    assert torch.equal(dup.spectrum_real, img.spectrum_real)
    assert torch.equal(dup.spectrum_imag, img.spectrum_imag)
    assert dup.decay == img.decay
    assert torch.equal(dup.decode_tensor(), img.decode_tensor())


def test_state_dict_roundtrips_bit_for_bit(tmp_path):
    img = make(width=64, height=48, decay=1.5)
    torch.manual_seed(8)
    img.encode_random()
    bak = tmp_path / "fourier.bak"
    torch.save(img.state_dict(), bak)

    restored = make(width=64, height=48, decay=1.5)
    restored.load_state_dict(torch.load(bak))
    assert torch.equal(restored.spectrum_real, img.spectrum_real)
    assert torch.equal(restored.spectrum_imag, img.spectrum_imag)
    assert torch.equal(restored.decode_tensor(), img.decode_tensor())


def test_state_dict_carries_only_the_trainable_spectrum():
    # derived constants (scale, color matrices) are persistent=False:
    # recomputed at construction, never serialized into .bak files
    keys = set(make().state_dict().keys())
    assert keys == {"spectrum_real", "spectrum_imag"}


# ----------------------------------------------------------------------
# per-model default lr
# ----------------------------------------------------------------------


def test_default_lr_flows_through_the_guide():
    from pytti.ImageGuide import DirectImageGuide

    img = make()
    guide = DirectImageGuide(img, embedder=None)
    assert guide.lr == img.lr
    assert guide.optimizer.param_groups[0]["lr"] == img.lr
    assert img.lr != 0.02  # the pixel default would be wrong here


# ----------------------------------------------------------------------
# fail-loud config surface
# ----------------------------------------------------------------------


def test_valid_v1_config_passes():
    validate_fourier_parameterization(**valid_config())


def test_off_with_default_decay_passes():
    validate_fourier_parameterization(
        **valid_config(fourier_parameterization=False)
    )


def test_mlx_full_backend_passes():
    # the whole-step engine has its own Fourier graph (mlx_engine, gated in
    # tests/test_mlx_engine_images.py / test_mlx_engine_step.py)
    validate_fourier_parameterization(
        **valid_config(perceptor_backend="mlx_full")
    )


def test_mlx_hybrid_backend_fails_loud():
    # the M1 hybrid is the animation path; fourier is stills-only — refused
    with pytest.raises(ValueError, match="mlx_full"):
        validate_fourier_parameterization(**valid_config(perceptor_backend="mlx"))


@pytest.mark.parametrize("model", ["Limited Palette", "VQGAN", "LlamaGen"])
def test_other_image_models_fail_loud(model):
    with pytest.raises(ValueError, match="Unlimited Palette"):
        validate_fourier_parameterization(**valid_config(image_model=model))


@pytest.mark.parametrize("spectrum", ["gray", "pink", "fractal"])
def test_non_white_init_spectrum_fails_loud(spectrum):
    with pytest.raises(ValueError, match="1/f-shaped by construction"):
        validate_fourier_parameterization(**valid_config(init_spectrum=spectrum))


def test_structure_annealing_fails_loud():
    with pytest.raises(ValueError, match="structure_annealing"):
        validate_fourier_parameterization(
            **valid_config(structure_annealing=True)
        )


@pytest.mark.parametrize("mode", ["2D", "3D", "Video Source"])
def test_animation_fails_loud(mode):
    with pytest.raises(ValueError, match="stills-only"):
        validate_fourier_parameterization(**valid_config(animation_mode=mode))


# ----------------------------------------------------------------------
# fourier_decay bounds + inert-knob rule
# ----------------------------------------------------------------------


@pytest.mark.parametrize("decay", [FOURIER_DECAY_MIN, 1.0, FOURIER_DECAY_MAX])
def test_decay_bounds_accept_valid(decay):
    validate_fourier_decay(decay)


@pytest.mark.parametrize("decay", [0.05, 0.0, -1.0, 4.5])
def test_decay_bounds_reject_invalid(decay):
    with pytest.raises(ValueError, match="fourier_decay"):
        validate_fourier_decay(decay)


def test_decay_is_inert_knob_checked_when_fourier_off():
    with pytest.raises(ValueError, match="only has meaning"):
        validate_fourier_decay(2.0, fourier_parameterization=False)


def test_constructor_rejects_out_of_range_decay():
    with pytest.raises(ValueError, match="fourier_decay"):
        make(decay=9.0)


# ----------------------------------------------------------------------
# hydra compose accepts (and validates) the schema fields
# ----------------------------------------------------------------------


def test_hydra_composes_both_fields():
    with initialize(config_path="config", version_base=None):
        cfg = compose(
            config_name="_structured_config",
            overrides=[
                "scenes=x",
                "fourier_parameterization=true",
                "fourier_decay=2.0",
            ],
        )
    obj = cast(ConfigSchema, OmegaConf.to_object(cfg))
    assert obj.fourier_parameterization is True
    assert obj.fourier_decay == 2.0


def test_hydra_defaults_are_off_and_one():
    with initialize(config_path="config", version_base=None):
        cfg = compose(config_name="_structured_config", overrides=["scenes=x"])
    obj = cast(ConfigSchema, OmegaConf.to_object(cfg))
    assert obj.fourier_parameterization is False
    assert obj.fourier_decay == 1.0


def test_hydra_rejects_out_of_range_decay():
    with initialize(config_path="config", version_base=None):
        cfg = compose(
            config_name="_structured_config",
            overrides=[
                "scenes=x",
                "fourier_parameterization=true",
                "fourier_decay=5.0",
            ],
        )
    with pytest.raises(ValueError, match="fourier_decay"):
        OmegaConf.to_object(cfg)


def test_hydra_rejects_inert_decay():
    with initialize(config_path="config", version_base=None):
        cfg = compose(
            config_name="_structured_config",
            overrides=["scenes=x", "fourier_decay=2.0"],
        )
    with pytest.raises(ValueError, match="only has meaning"):
        OmegaConf.to_object(cfg)


def test_divisor_is_lucids():
    assert LUCID_OUTPUT_DIVISOR == 4.0
