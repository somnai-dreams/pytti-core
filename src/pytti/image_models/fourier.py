"""
Fourier-parameterized image (config ``fourier_parameterization``): the
strongest literature-backed structure play. Instead of raw pixels, the
trainable state is a 1/f-SCALED FOURIER SPECTRUM (distill.pub 2018
"Differentiable Image Parameterizations" / lucid's fft_image; Aphantasia's
torch port proved the same preconditioning for CLIP-guided generation).

Why it changes composition: with pixel parameters, one optimizer step moves
every pixel independently — CLIP's gradient arrives as per-pixel salt and
low-frequency structure only emerges by slow accumulation. Here each
trainable coefficient is one FREQUENCY, and the fixed 1/f^decay scale means
a unit optimizer step moves low frequencies (composition: big masses,
placement) with large image-space amplitude and high frequencies (texture)
with small amplitude. Composition forms first BY CONSTRUCTION; texture
arrives later. ``fourier_decay`` is Aphantasia's "compositional softness"
knob: higher = the low-frequency head start grows.

The decode pipeline, adopted from lucid verbatim:

    spectrum (trainable real+imag, [3, H, W//2+1] rfft2 layout, init
    ~ N(0, 0.01) — lucid's sd)
      -> * scale, where scale = (1/max(f, f_min))^decay * sqrt(H*W),
         renormalized so its induced image-space energy equals the
         decay-1 (lucid) grid's — see fourier_scale; at decay 1 the
         factor is exactly 1, i.e. lucid verbatim
         (f from the fftfreq(H) x rfftfreq(W) grid, f_min = 1/max(H, W))
      -> irfft2 (torch default backward-normalized, matching TF's irfft2d)
      -> / 4  (lucid's magic constant — see LUCID_OUTPUT_DIVISOR)
      -> decorrelated basis -> RGB through lucid's ImageNet color matrix
         (imagenet_color_matrix — the SAME constant the init-chroma
         'natural' path uses)
      -> sigmoid -> [0, 1]

``encode_image`` (init images, coarse_to_fine stage seams, restore-reencode)
is the exact inverse: clamp to (eps, 1-eps), logit, inverse color matrix,
* 4, rfft2, / scale. decode(encode(image)) round-trips to within logit
clamp + float error (tests measure the tolerance). The free spectrum
carries redundant Hermitian-pair coefficients (the fx=0 / Nyquist columns);
irfft2 projects them, so encode(decode(spectrum)) canonicalizes the
spectrum while decode output is preserved — image-domain round-trips are
the contract.

Scope (validate_fourier_parameterization, checked at config time in
workhorse BEFORE any model loads and again in configure_pass for direct
callers):
- image_model='Unlimited Palette', perceptor_backend='torch' or
  'mlx_full'. The mlx_full whole-step engine compiles its own Fourier
  decode graph (mlx_engine/image_models.fourier_decode — mx.fft.irfft2's
  VJP is correct on Metal, eager and compiled; probed + gate-tested
  2026-08). The M1 'mlx' hybrid stays refused: it is the ANIMATION path
  (mlx_full owns stills since M2) and fourier_parameterization is
  stills-only, so the combination has no use case — refusing keeps it
  unmeasured rather than silently blessed.
- init_spectrum must stay 'white': the Fourier init IS 1/f-shaped by
  construction (white spectrum coefficients x the 1/f scale), so a shaped
  pixel-domain init request can never be honored — any non-white value is
  an inert knob, rejected loudly.
- structure_annealing is rejected: its re-liquify cycles blend
  pixel-domain planes through get/set_image_tensor, an interaction with a
  spectral optimizer state nobody has measured — and annealing itself was
  falsified on good starts (2026-08-06 battery). Don't compose them.
- animation_mode must be 'off': warps write pixels back through the
  saturating logit clamp every frame; repeated round-trip accumulation is
  unmeasured. Stills only in v1.

``get_image_tensor``/``set_image_tensor`` still work (they route through
the same decode / inverse-FFT pipeline), so anything pixel-domain that
reaches them behaves honestly rather than crashing — but no v1 config path
uses them (the animation guard above).

Seeding contract: ``encode_random`` draws real then imag via the global
torch RNG only, so ``torch.manual_seed(params.seed)`` upstream gives
seed-deterministic starts exactly like the other image models.
"""

import math

import torch
from PIL import Image
from torch import nn
from torch.nn import functional as F
from torchvision.transforms import functional as TF

from pytti.device import default_device
from pytti.image_models.differentiable_image import DifferentiableImage
from pytti.image_models.init_noise import (
    imagenet_color_matrix,
    validate_spectrum_chroma,
)

# valid range of the amplitude decay power (config ``fourier_decay``).
# decay 1 = lucid's default (~natural 1/f amplitude); toward 0.1 the
# preconditioning fades back toward pixel-like behavior; higher = more
# low-frequency-dominant (composition over texture). The knob is SHAPE
# only: fourier_scale renormalizes every grid to the decay-1 energy
# (anchored exactly at lucid's), so the whole range starts near-gray and
# optimizes at any resolution — without that normalization the raw lucid
# grid's low-frequency gain grows as max(H,W)^decay against a fixed init
# sd and the sigmoid saturates into a dead render by decay ~2-3
# (size-dependent; measured 2026-08). 4.0 is generous headroom, matching
# the init_spectrum_falloff philosophy (its moment-rescale is the same
# shape-not-gain move).
FOURIER_DECAY_MIN = 0.1
FOURIER_DECAY_MAX = 4.0
FOURIER_DECAY_DEFAULT = 1.0

# lucid's init: spectrum coefficients ~ N(0, sd) with sd = 0.01
FOURIER_INIT_SD = 0.01

# lucid's post-irfft2 magic constant ("4.0 to make it up" in the lucid
# source). DECISION: adopt lucid's exact pipeline including this divisor
# rather than compensating through lr. It scales the pre-sigmoid logits,
# so it sets both the image contrast a given spectrum magnitude produces
# AND (with the sigmoid near-linear at init) the image-space size of an
# optimizer step — dropping it while keeping lucid's sd/lr would start 4x
# hotter than the regime lucid (and Aphantasia) tuned around. Measured at
# init (sd 0.01, decay 1.0, 128px, seed 42): pre-sigmoid logit std 0.012
# with the divisor (decode range ~[0.49, 0.51], the near-gray lucid start)
# vs 0.047 without.
LUCID_OUTPUT_DIVISOR = 4.0

# per-model default lr (the learning_rate=None -> image_rep.lr mechanism in
# DirectImageGuide). Lucid used Adam lr 0.05 in this decorrelated space,
# and the smoke confirms it transfers: 'a red rowboat beneath a stone
# bridge' @128px/20 steps/seed 424242 (torch/MPS, ViT-B/32), decay 1.0 at
# 0.05 shows composition as large soft masses by step 4 (no gray stall, no
# blowout) and ends BELOW the pixel leg's semantic loss at equal steps
# (0.753 vs 0.787; pixel default lr 0.02, its own tuned value). The decay
# interaction is tame by construction: fourier_scale holds every grid at
# the decay-1 energy, so 0.05 is a sane default across the whole
# fourier_decay range — higher decay reallocates the same step budget
# toward low frequencies (bolder composition, slower texture) instead of
# inflating it.
FOURIER_LR_DEFAULT = 0.05

# encode-side clamp before the logit: 8-bit inputs bottom out at 1/255 ≈
# 4e-3, so 1e-4 never distorts real image content, and logit(1e-4) ≈ -9.2
# stays comfortably inside float32.
_LOGIT_EPS = 1e-4


def validate_fourier_decay(
    value: float, *, fourier_parameterization: bool = True
) -> None:
    """
    Config-boundary check for ``fourier_decay`` (called from the schema
    validator at compose time and again at FourierImage construction).
    Bounds, plus the inert-knob rule: a non-default decay on a run with
    fourier_parameterization off is a config lie.
    """
    if not FOURIER_DECAY_MIN <= value <= FOURIER_DECAY_MAX:
        raise ValueError(
            f"fourier_decay={value!r} is out of range "
            f"[{FOURIER_DECAY_MIN:g}, {FOURIER_DECAY_MAX:g}]. It is the "
            "Fourier-parameterization amplitude decay power (1.0 = lucid's "
            "natural-image default; lower = closer to pixel behavior, "
            "higher = softer/more composition-dominant)."
        )
    if not fourier_parameterization and value != FOURIER_DECAY_DEFAULT:
        raise ValueError(
            f"fourier_decay={value!r} only has meaning with "
            "fourier_parameterization: true — set fourier_parameterization: "
            "true or remove fourier_decay."
        )


def validate_fourier_parameterization(
    *,
    fourier_parameterization: bool,
    fourier_decay: float,
    image_model: str,
    perceptor_backend: str,
    init_spectrum: str,
    structure_annealing: bool,
    animation_mode: str,
) -> None:
    """
    Reject every config fourier_parameterization has no v1 behavior for —
    LOUDLY, before any model loads (the validate_structure_annealing
    pattern). Called from workhorse.do_run and again at the top of
    configure_pass (covering direct callers and every coarse_to_fine stage
    pass). Checks run on the CONFIGURED values: e.g. a non-white
    init_spectrum is rejected even when an init_image would have resolved
    the knob inert, because under Fourier parameterization the request can
    NEVER be honored — the init is 1/f by construction.
    """
    validate_fourier_decay(
        fourier_decay, fourier_parameterization=fourier_parameterization
    )
    if not fourier_parameterization:
        return
    if image_model != "Unlimited Palette":
        raise ValueError(
            f"fourier_parameterization has no v1 behavior for image_model="
            f"{image_model!r}: it re-parameterizes the Unlimited Palette "
            "RGB canvas as a 1/f-scaled spectrum, and the other models hold "
            "palette/codebook state with no spectral form. Set image_model: "
            "'Unlimited Palette' or fourier_parameterization: false."
        )
    if perceptor_backend not in ("torch", "mlx_full"):
        raise ValueError(
            f"fourier_parameterization runs on perceptor_backend torch or "
            f"mlx_full; perceptor_backend={perceptor_backend!r} (the M1 "
            "hybrid) is the animation path and this feature is stills-only "
            "— the combination has no use case and stays unmeasured. Set "
            "perceptor_backend: torch or mlx_full, or "
            "fourier_parameterization: false."
        )
    if init_spectrum != "white":
        raise ValueError(
            f"init_spectrum={init_spectrum!r} cannot be honored with "
            "fourier_parameterization: the Fourier init IS 1/f-shaped by "
            "construction (white spectrum coefficients x the 1/f^decay "
            "scale), so a shaped pixel-domain init would be an inert knob. "
            "Use fourier_decay to shape the start instead, or set "
            "init_spectrum: white."
        )
    if structure_annealing:
        raise ValueError(
            "structure_annealing + fourier_parameterization is unsupported: "
            "the anneal cycles blend pixel-domain planes into the image "
            "mid-run, an unmeasured interaction with spectral optimizer "
            "state — and annealing was falsified on good starts anyway "
            "(2026-08-06 battery). Set structure_annealing: false or "
            "fourier_parameterization: false."
        )
    if animation_mode != "off":
        raise ValueError(
            "fourier_parameterization is stills-only in v1: animation warps "
            f"(animation_mode={animation_mode!r}) write pixels back through "
            "a saturating logit clamp every frame, and that repeated "
            "round-trip is unmeasured. Set animation_mode: off or "
            "fourier_parameterization: false."
        )


def _hermitian_variance_weights(
    height: int, width: int, device: torch.device | str
) -> torch.Tensor:
    """
    Per-bin variance weights of the free rfft2 layout under irfft2, for
    i.i.d. real+imag coefficients: the mean image-space second moment is
    (sd^2 / (H*W)^2) * sum_k w_k * scale_k^2. Paired columns count their
    Hermitian mirror (w=4: |c|^2 = 2 components, x2 for the conjugate
    bin); the self-paired fx=0 column — and the Nyquist column when width
    is even — is Hermitian-projected by irfft2, which halves its energy
    (w=1). Verified bin-by-bin against torch.fft.irfft2 (2026-08).
    """
    weights = torch.full((height, width // 2 + 1), 4.0, device=device)
    weights[:, 0] = 1.0
    if width % 2 == 0:
        weights[:, -1] = 1.0
    return weights


def fourier_scale(
    height: int, width: int, decay: float, device: torch.device | str
) -> torch.Tensor:
    """
    Lucid's per-frequency amplitude scale for an rfft2 layout, energy-
    normalized: a [height, width//2+1] tensor built as raw(f) =
    (1/max(f, f_min))^decay * sqrt(height*width), with f = sqrt(fy^2 +
    fx^2) over the fftfreq(height) x rfftfreq(width) grid (cycles/pixel)
    and f_min = 1/max(height, width) the lowest representable nonzero
    frequency (keeps the DC gain finite, exactly like the pink-init
    synthesis in init_noise.py) — then multiplied by the scalar that makes
    its induced image-space energy (the Hermitian-weighted sum of squares,
    _hermitian_variance_weights) equal the decay-1 grid's.

    The normalization makes ``decay`` a pure SHAPE knob: it reallocates
    init variance and optimizer step amplitude across frequencies without
    changing their total, so every decay in the validated range starts
    near-gray and keeps live sigmoid gradients at any resolution. The
    anchor is lucid's decay-1 grid: at decay == 1 the factor is exactly
    1.0 (identical tensors, ratio 1), i.e. lucid verbatim. Without it the
    raw grid's low-frequency gain grows as max(H,W)^decay against the
    fixed FOURIER_INIT_SD and lr, saturating the sigmoid into a dead,
    gradient-free render by decay ~2-3 (size-dependent; measured 2026-08:
    decay 3 at 128px inits 94% saturated and 200 Adam steps make zero
    progress; decay 2 at 256px already regresses).
    """
    if height < 1 or width < 1:
        raise ValueError(f"dims must be positive, got {width}x{height}")
    validate_fourier_decay(decay)
    fy = torch.fft.fftfreq(height, device=device)[:, None]
    fx = torch.fft.rfftfreq(width, device=device)[None, :]
    freqs = torch.sqrt(fy**2 + fx**2)
    f_min = 1.0 / max(height, width)
    clamped = freqs.clamp_min(f_min)
    raw = clamped.pow(-decay) * math.sqrt(height * width)
    # The gain: sqrt of the energy ratio between the decay-1 reference grid
    # and this one. Anchored at -2.0, i.e. decay 1.0 — lucid's pipeline, a
    # fixed external reference, NOT FOURIER_DECAY_DEFAULT (the config
    # default must not move the anchor); at decay == 1 the exponents are
    # identical so the gain is exactly 1.0. The sqrt(H*W) amplitude factor
    # is common to both grids and cancels in the ratio. Sums run on CPU in
    # float64: the squared gains reach ~1e29 at decay 4 / 4k-px grids
    # (float32 loses the small-bin tail there) and MPS has no float64 —
    # this is a one-off [H, W//2+1] reduction at construction time.
    clamped64 = clamped.cpu().double()
    weights = _hermitian_variance_weights(height, width, "cpu").double()
    energy = (weights * clamped64.pow(-2.0 * decay)).sum()
    reference_energy = (weights * clamped64.pow(-2.0)).sum()
    gain = math.sqrt(float(reference_energy / energy))
    return raw * gain


class FourierImage(DifferentiableImage):
    """
    Unlimited-Palette look, Fourier-parameterized (module docstring). The
    trainable state is spectrum_real/spectrum_imag ([3, H, W//2+1] each, on
    the LOGICAL grid); pixel_size upsampling happens in decode_tensor
    exactly like RGBImage (nearest).
    """

    # registered buffers (declared so type checkers see Tensor, not Module)
    spectrum_scale: torch.Tensor
    color_matrix: torch.Tensor
    color_matrix_inv: torch.Tensor

    def __init__(self, width, height, scale=1, decay=FOURIER_DECAY_DEFAULT, device=None):
        super().__init__(width * scale, height * scale)
        validate_fourier_decay(decay)
        if device is None:
            device = default_device()
        self.device = device
        self.decay = decay
        self.spectrum_real = nn.Parameter(
            torch.zeros(3, height, width // 2 + 1).to(device=self.device)
        )
        self.spectrum_imag = nn.Parameter(
            torch.zeros(3, height, width // 2 + 1).to(device=self.device)
        )
        # derived constants — recomputed at construction, excluded from
        # state_dict (persistent=False) so .bak files carry only the
        # trainable spectrum
        self.register_buffer(
            "spectrum_scale",
            fourier_scale(height, width, decay, self.device),
            persistent=False,
        )
        matrix = imagenet_color_matrix(torch.float32, self.device)
        self.register_buffer("color_matrix", matrix, persistent=False)
        self.register_buffer(
            "color_matrix_inv", torch.linalg.inv(matrix), persistent=False
        )
        self.output_axes = ("n", "s", "y", "x")
        self.scale = scale
        self.lr = FOURIER_LR_DEFAULT

    def _logical_grid(self) -> tuple[int, int]:
        """(height, width) of the trainable grid, pre-pixel_size upsample."""
        width, height = self.image_shape
        return height // self.scale, width // self.scale

    def _decode_logical(self) -> torch.Tensor:
        """The differentiable decode to a [3, H, W] image in (0, 1)."""
        spectrum = torch.complex(self.spectrum_real, self.spectrum_imag)
        pixels = torch.fft.irfft2(spectrum * self.spectrum_scale, s=self._logical_grid())
        basis = pixels / LUCID_OUTPUT_DIVISOR
        rgb = torch.einsum("ck,khw->chw", self.color_matrix, basis)
        return torch.sigmoid(rgb)

    def decode_tensor(self):
        width, height = self.image_shape
        out = self._decode_logical().unsqueeze(0)
        return F.interpolate(out, (height, width), mode="nearest")

    def clone(self) -> "FourierImage":
        width, height = self.image_shape
        dummy = FourierImage(
            width // self.scale,
            height // self.scale,
            self.scale,
            decay=self.decay,
            device=self.device,
        )
        with torch.no_grad():
            dummy.spectrum_real.copy_(self.spectrum_real)
            dummy.spectrum_imag.copy_(self.spectrum_imag)
        return dummy

    def get_image_tensor(self):
        # pixel-domain view, differentiable — anything that reads image
        # planes (latent losses, hypothetical warps) sees honest pixels
        return self._decode_logical()

    @torch.no_grad()
    def set_image_tensor(self, tensor):
        self._write_pixels(tensor)

    @torch.no_grad()
    def _write_pixels(self, pixels: torch.Tensor) -> None:
        """
        The inverse pipeline: [3, H, W] pixels in [0, 1] -> spectrum params.
        clamp (eps, 1-eps) -> logit -> inverse color matrix -> * 4 ->
        rfft2 -> / scale. Exact inverse of _decode_logical up to the clamp
        and float error (rfft2 of a real signal is fully recoverable by
        irfft2 at the same s).
        """
        h, w = self._logical_grid()
        if tuple(pixels.shape) != (3, h, w):
            raise ValueError(
                f"FourierImage expects a [3, {h}, {w}] pixel tensor (the "
                f"logical grid), got {tuple(pixels.shape)}"
            )
        logits = torch.logit(
            pixels.to(device=self.device, dtype=torch.float32).clamp(
                _LOGIT_EPS, 1.0 - _LOGIT_EPS
            )
        )
        basis = torch.einsum("kc,chw->khw", self.color_matrix_inv, logits)
        spectrum = torch.fft.rfft2(basis * LUCID_OUTPUT_DIVISOR)
        spectrum = spectrum / self.spectrum_scale
        self.spectrum_real.copy_(spectrum.real)
        self.spectrum_imag.copy_(spectrum.imag)

    @torch.no_grad()
    def encode_image(self, pil_image, device=None, **kwargs):
        width, height = self.image_shape
        pil_image = pil_image.resize(
            (width // self.scale, height // self.scale), Image.Resampling.LANCZOS
        )
        self._write_pixels(TF.to_tensor(pil_image).to(self.device))

    @torch.no_grad()
    def encode_random(
        self,
        init_spectrum="white",
        init_spectrum_falloff=1.0,
        init_spectrum_chroma="full",
    ):
        """
        Lucid's init: spectrum ~ N(0, FOURIER_INIT_SD), real drawn before
        imag, global torch RNG only (seed-deterministic). Only
        init_spectrum='white' is accepted: the Fourier start is 1/f-shaped
        by construction, so a shaped pixel-domain init request is an inert
        knob and fails loudly (config validation rejects it earlier; this
        repeats the guard for direct callers). The chroma knob is ignored
        exactly like every other model's white path (documented in
        init_noise.py: 'white' never consults it).
        """
        validate_spectrum_chroma(init_spectrum_chroma)
        if init_spectrum != "white":
            raise ValueError(
                f"init_spectrum={init_spectrum!r} is not supported for "
                "FourierImage: the Fourier init IS 1/f-shaped by "
                "construction (white spectrum coefficients x the 1/f^decay "
                "scale). Supported: 'white'; shape the start with "
                "fourier_decay instead."
            )
        self.spectrum_real.normal_(mean=0.0, std=FOURIER_INIT_SD)
        self.spectrum_imag.normal_(mean=0.0, std=FOURIER_INIT_SD)
