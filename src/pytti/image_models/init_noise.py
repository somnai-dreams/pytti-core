"""
Shaped init-noise fields for the no-init_image start (config
``init_spectrum``).

Every image model used to open on iid uniform white noise
(``encode_random``). Natural images have ~1/f amplitude spectra, so white
init is maximally wrong and the TV/smoothing losses spend their first steps
fighting it. This module generates the alternative starting fields:

- ``gray``   — mid-gray plus a hair of symmetry-breaking uniform noise
  (0.5 +/- 0.5/255, Crowson style-transfer style).
- ``pink``   — FFT-shaped noise: a white gaussian spectrum multiplied by a
  1/max(f, f_min)^alpha amplitude falloff (f_min = the lowest representable
  frequency, so DC stays finite — lucid-style), irfft2 back to space.
  alpha (config ``init_spectrum_falloff``) = 1 gives the natural 1/f^2
  POWER spectrum.
- ``fractal`` — multi-octave upsampled white noise (Whitaker pyramid-noise
  recipe): white gaussian at the full logical grid, plus bilinear-upsampled
  octaves halving resolution each time, discounted 0.9^i, down to a ~4px
  side. A second, differently-shaped power-law family, no FFT.

pink/fractal fields are rescaled to the moments a uniform init would give
(mean 0.5, std 1/sqrt(12)) and clamped to [0, 1].

``white`` is intentionally NOT generated here: the callers keep their
original in-place ``uniform_()`` draws so the default stays bit-for-bit
identical to the historical init.

Shaping happens on the LOGICAL grid (height x width) — ``pixel_size``
upsampling happens downstream in ``decode_tensor``.

Seeding contract: every draw consumes only the global torch RNG (no
generator objects), so ``torch.manual_seed(params.seed)`` upstream gives
per-seed variation exactly like the white path.

The MLX engine (``mlx_engine``) needs no counterpart: it imports the image
params from the torch module's ``state_dict`` at engine construction, so a
shaped torch init flows through automatically.
"""

import math

import torch
from torch.nn import functional as F

# what the config validator and the per-model guards agree on
INIT_SPECTRUM_CHOICES = ("white", "gray", "pink", "fractal")

# valid range of the pink amplitude decay power alpha. Negative would invert
# the documented decay into rising blue noise; anything past ~4 is already a
# near-DC cloud, so 8 is generous headroom rather than a useful setting.
MAX_SPECTRUM_FALLOFF = 8.0

# std of U[0,1] — the moments the white init would have given
_UNIFORM_STD = 1.0 / math.sqrt(12.0)

# fractal: per-octave amplitude discount and the smallest octave side
_FRACTAL_DISCOUNT = 0.9
_FRACTAL_MIN_SIDE = 4


def validate_spectrum_falloff(value: float) -> None:
    """
    Config-boundary range check for ``init_spectrum_falloff`` (called from
    the schema validator at compose time and again by the pink synthesis).
    """
    if not 0.0 <= value <= MAX_SPECTRUM_FALLOFF:
        raise ValueError(
            f"init_spectrum_falloff={value!r} is out of range "
            f"[0, {MAX_SPECTRUM_FALLOFF:g}]. It is the pink-init amplitude "
            "decay power alpha (0 = flat/white, 1 = the natural 1/f, ~4+ is "
            "already a near-DC cloud); negative values would invert the "
            "documented decay into rising blue noise."
        )


def resolve_init_spectrum(
    init_spectrum: str,
    init_spectrum_falloff: float,
    *,
    has_init_image: bool,
    restore: bool,
) -> tuple[str, float]:
    """
    Collapse the config knob to what actually governs this pass. The knob
    only shapes the visible no-init start: with an init_image the random
    init is immediately overwritten by ``encode_image``, and on restore the
    whole image state is reloaded from the ``.bak`` — in both cases the knob
    is documented as never consulted, so it resolves to the plain white
    draw. That skips the wasted shaping work on the trainable models and
    keeps VQGAN/LlamaGen from rejecting a shaped start they never show.
    """
    if has_init_image or restore:
        return "white", 1.0
    return init_spectrum, init_spectrum_falloff


def require_white_init(model_name: str, init_spectrum: str) -> None:
    """
    Guard for models whose random init is a categorical draw over codebook
    tokens (VQGAN, LlamaGen): there is no spectrum to shape, so any
    non-'white' request fails loudly instead of silently falling back. The
    honest path to a low-frequency start for those models is an init_image
    through the frozen encoder.
    """
    if init_spectrum != "white":
        raise ValueError(
            f"init_spectrum={init_spectrum!r} is not supported for "
            f"{model_name}: its random init is a categorical draw over "
            "codebook tokens — there is no spectrum to shape. Supported: "
            "'white'. For a low-frequency start, use an init_image (it is "
            "encoded through the model's frozen encoder)."
        )


def _rescale_to_uniform_moments(field: torch.Tensor) -> torch.Tensor:
    """Per-plane: mean 0.5, std 1/sqrt(12), tails clamped into [0, 1]."""
    mu = field.mean(dim=(-2, -1), keepdim=True)
    sigma = field.std(dim=(-2, -1), keepdim=True)
    field = (field - mu) / sigma.clamp_min(1e-12)
    return (field * _UNIFORM_STD + 0.5).clamp_(0.0, 1.0)


def _gray_field(
    channels: int, height: int, width: int, device: torch.device | str
) -> torch.Tensor:
    # 0.5 + uniform*(1/255) - 0.5/255: mid-gray with one-bit symmetry breaking
    return torch.rand(channels, height, width, device=device) * (1.0 / 255.0) + (
        0.5 - 0.5 / 255.0
    )


def _pink_field(
    channels: int,
    height: int,
    width: int,
    falloff: float,
    device: torch.device | str,
) -> torch.Tensor:
    validate_spectrum_falloff(falloff)
    noise = torch.randn(channels, height, width, device=device)
    spectrum = torch.fft.rfft2(noise)
    fy = torch.fft.fftfreq(height, device=device)
    fx = torch.fft.rfftfreq(width, device=device)
    freqs = torch.sqrt(fy[:, None] ** 2 + fx[None, :] ** 2)
    # lowest representable nonzero frequency: keeps the DC gain finite
    f_min = 1.0 / max(height, width)
    # normalize to (f/f_min)^-alpha so the curve peaks at 1 instead of
    # f_min^-alpha: the moment rescale below erases any global scale, and
    # unnormalized amplitudes overflow float32 variance on large canvases
    # at high alpha (inf sigma -> a silently constant 0.5 field)
    amplitude = (freqs.clamp_min(f_min) / f_min).pow(-falloff)
    shaped = torch.fft.irfft2(spectrum * amplitude, s=(height, width))
    return _rescale_to_uniform_moments(shaped)


def _fractal_field(
    channels: int, height: int, width: int, device: torch.device | str
) -> torch.Tensor:
    noise = torch.randn(channels, height, width, device=device)
    h, w = height // 2, width // 2
    octave = 1
    while min(h, w) >= _FRACTAL_MIN_SIDE:
        band = torch.randn(1, channels, h, w, device=device)
        up = F.interpolate(
            band, size=(height, width), mode="bilinear", align_corners=False
        ).squeeze(0)
        noise = noise + up * _FRACTAL_DISCOUNT**octave
        h, w = h // 2, w // 2
        octave += 1
    return _rescale_to_uniform_moments(noise)


def shaped_init_field(
    channels: int,
    height: int,
    width: int,
    init_spectrum: str,
    init_spectrum_falloff: float,
    device: torch.device | str,
) -> torch.Tensor:
    """
    A [channels, height, width] float field in [0, 1] with independent
    channels, shaped per ``init_spectrum``. 'white' is refused on purpose —
    callers keep their original in-place ``uniform_()`` so the default init
    stays bit-for-bit identical.
    """
    if init_spectrum == "gray":
        return _gray_field(channels, height, width, device)
    if init_spectrum == "pink":
        return _pink_field(channels, height, width, init_spectrum_falloff, device)
    if init_spectrum == "fractal":
        return _fractal_field(channels, height, width, device)
    raise ValueError(
        f"init_spectrum={init_spectrum!r} has no shaped field. Shaped "
        "spectra: ['gray', 'pink', 'fractal'] ('white' is the caller's "
        "in-place uniform_() path)."
    )
