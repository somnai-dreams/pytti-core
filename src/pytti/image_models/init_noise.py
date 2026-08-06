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

Chroma structure (config ``init_spectrum_chroma``): independent per-channel
shaped fields leave low-frequency COLOR blobs in the init that CLIP never
cleans up and that steer the final palette. The knob picks how much chroma
the shaped inits carry:

- ``full``    — the original behavior, bit-for-bit: one independent shaped
  field per channel (full chroma). The default, so existing seeds stay
  reproducible.
- ``natural`` — three shaped fields drawn in a decorrelated color basis and
  mapped to RGB through lucid's ImageNet color matrix
  (``_COLOR_CORRELATION_SVD_SQRT``), so channel covariance follows natural
  image statistics: mostly luma, faint chroma.
- ``mono``    — ONE shaped luminance field broadcast to every channel
  (around mid-gray after the moment rescale): full spatial prior, zero
  chroma.

The knob only shapes color: ``white`` never reaches this module (the
callers' uniform draw has no low-frequency structure of any kind) and
``gray`` ignores it (its channel deviations are <= 1/255 and spatially
white — there are no low-frequency chroma blobs to remove, and one code
path keeps gray bit-identical across chroma values). PixelImage has no RGB
channels at init (palettes start as gray ramps), so there the knob governs
the per-palette-plane selection-logit amplitude instead — see
``PixelImage.encode_random``.

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
INIT_SPECTRUM_CHROMA_CHOICES = ("mono", "natural", "full")

# lucid's ImageNet color decorrelation matrix (the sqrt of the empirical
# RGB covariance of ImageNet): rows map a decorrelated 3-vector to RGB, so
# unit-variance basis fields come out with natural cross-channel
# correlations (~0.9 R-G, ~0.8 R-B) — mostly luma, faint chroma. Normalized
# by the max column norm exactly as lucid does; the per-plane moment
# rescale downstream erases the global scale either way.
_COLOR_CORRELATION_SVD_SQRT = (
    (0.26, 0.09, 0.02),
    (0.27, 0.00, -0.05),
    (0.27, -0.09, 0.03),
)

# PixelImage 'natural' chroma: the shaped selection-logit planes are blended
# toward flat mid-gray by this factor (0.5 + A*(field - 0.5)), shrinking
# their std to A/sqrt(12). Full-strength shaped logits pre-commit palette
# REGIONS (coherent selection areas become color areas as palettes
# diverge); 0.3 keeps a faint spatial bias on palette selection without
# letting the init dominate the softmax at step 0 — 'natural' sits between
# mono's uniform logits (no pre-commitment) and full's.
NATURAL_TENSOR_AMPLITUDE = 0.3

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


def validate_spectrum_chroma(value: str) -> None:
    """
    Config-boundary membership check for ``init_spectrum_chroma`` (called
    from the schema validator at compose time and again by every
    ``encode_random`` for direct callers).
    """
    if value not in INIT_SPECTRUM_CHROMA_CHOICES:
        raise ValueError(
            f"init_spectrum_chroma={value!r} is not a valid chroma mode. "
            f"Valid values: {list(INIT_SPECTRUM_CHROMA_CHOICES)} (mono = one "
            "luminance field, zero chroma; natural = lucid ImageNet color "
            "statistics, faint chroma; full = independent per-channel "
            "fields, the original behavior)."
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


def _raw_pink(
    channels: int,
    height: int,
    width: int,
    falloff: float,
    device: torch.device | str,
) -> torch.Tensor:
    """Shaped but not yet moment-rescaled — chroma mapping happens between."""
    validate_spectrum_falloff(falloff)
    noise = torch.randn(channels, height, width, device=device)
    spectrum = torch.fft.rfft2(noise)
    fy = torch.fft.fftfreq(height, device=device)
    fx = torch.fft.rfftfreq(width, device=device)
    freqs = torch.sqrt(fy[:, None] ** 2 + fx[None, :] ** 2)
    # lowest representable nonzero frequency: keeps the DC gain finite
    f_min = 1.0 / max(height, width)
    # normalize to (f/f_min)^-alpha so the curve peaks at 1 instead of
    # f_min^-alpha: the moment rescale downstream erases any global scale,
    # and unnormalized amplitudes overflow float32 variance on large
    # canvases at high alpha (inf sigma -> a silently constant 0.5 field)
    amplitude = (freqs.clamp_min(f_min) / f_min).pow(-falloff)
    return torch.fft.irfft2(spectrum * amplitude, s=(height, width))


def _raw_fractal(
    channels: int, height: int, width: int, device: torch.device | str
) -> torch.Tensor:
    """Shaped but not yet moment-rescaled — chroma mapping happens between."""
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
    return noise


def _natural_rgb(raw: torch.Tensor) -> torch.Tensor:
    """
    Map three shaped fields from a decorrelated color basis to RGB through
    lucid's ImageNet color matrix, then apply the same moment rescale as
    the other paths. Per-plane standardization before the matrix gives each
    basis axis exactly unit variance; the per-plane rescale after preserves
    the cross-channel CORRELATIONS the matrix installs (correlation is
    scale-invariant per variable) while restoring uniform-init moments.
    """
    mu = raw.mean(dim=(-2, -1), keepdim=True)
    sigma = raw.std(dim=(-2, -1), keepdim=True)
    basis = (raw - mu) / sigma.clamp_min(1e-12)
    matrix = torch.tensor(
        _COLOR_CORRELATION_SVD_SQRT, dtype=raw.dtype, device=raw.device
    )
    matrix = matrix / matrix.norm(dim=0).max()  # lucid's normalization
    rgb = torch.einsum("ck,khw->chw", matrix, basis)
    return _rescale_to_uniform_moments(rgb)


def shaped_init_field(
    channels: int,
    height: int,
    width: int,
    init_spectrum: str,
    init_spectrum_falloff: float,
    device: torch.device | str,
    init_spectrum_chroma: str = "full",
) -> torch.Tensor:
    """
    A [channels, height, width] float field in [0, 1] shaped per
    ``init_spectrum``, with cross-channel structure per
    ``init_spectrum_chroma`` (see module docstring; 'full' = independent
    channels, verbatim the pre-knob behavior AND RNG stream). 'white' is
    refused on purpose — callers keep their original in-place ``uniform_()``
    so the default init stays bit-for-bit identical. 'natural' is a color
    mapping, so it demands exactly 3 channels; 'gray' ignores the chroma
    knob (documented in the module docstring).
    """
    validate_spectrum_chroma(init_spectrum_chroma)
    if init_spectrum == "gray":
        return _gray_field(channels, height, width, device)
    if init_spectrum == "pink":

        def raw(c: int) -> torch.Tensor:
            return _raw_pink(c, height, width, init_spectrum_falloff, device)

    elif init_spectrum == "fractal":

        def raw(c: int) -> torch.Tensor:
            return _raw_fractal(c, height, width, device)

    else:
        raise ValueError(
            f"init_spectrum={init_spectrum!r} has no shaped field. Shaped "
            "spectra: ['gray', 'pink', 'fractal'] ('white' is the caller's "
            "in-place uniform_() path)."
        )
    if init_spectrum_chroma == "full":
        return _rescale_to_uniform_moments(raw(channels))
    if init_spectrum_chroma == "mono":
        # one luminance field around mid-gray, broadcast: R==G==B exactly
        field = _rescale_to_uniform_moments(raw(1))
        return field.expand(channels, height, width).contiguous()
    # natural: a 3x3 color mapping — only defined for RGB
    if channels != 3:
        raise ValueError(
            f"init_spectrum_chroma='natural' maps a decorrelated basis to "
            f"RGB through a 3x3 color matrix, so it requires exactly 3 "
            f"channels; got {channels}. Non-RGB planes (e.g. PixelImage "
            "selection logits) have their own chroma handling in their "
            "encode_random."
        )
    return _natural_rgb(raw(3))
