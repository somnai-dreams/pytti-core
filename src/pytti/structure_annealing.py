"""
Structure annealing — periodic low-frequency re-liquification (config
``structure_annealing`` + the ``anneal_*`` knobs in structured_config).

The problem it exists for: gradient descent can refine but not RESTRUCTURE.
Composition freezes within the first steps (measured 0.963 cross-stage
composition correlation on the coarse_to_fine work) because the low
frequencies of the image receive almost no gradient per step and moving a
shape means walking through high-loss intermediate states. coarse_to_fine
is exactly ONE structural reset (each stage re-encodes a liquified small
canvas); this module generalizes that to a schedule: several times during a
run, the LOW-frequency band of the image is re-liquified — blended toward
shaped noise or toward its own mean — so CLIP gets fresh votes on
composition while the accumulated mid/high-frequency detail survives
untouched. A diffusion-style structure schedule, without a denoiser.

The operation, per cycle (``anneal_plane``)
-------------------------------------------
Value-domain plane -> ``rfft2`` -> blend the soft-masked low band toward
the source -> ``irfft2`` -> clamp back to the plane's own parameter
domain ([0, 1] for the PixelImage value plane, whose per-step ``update()``
clamps there anyway; NO clamp for the RGBImage tensor — its domain is
unbounded, decode clamps, and mid-run params legitimately hold
out-of-[0, 1] "headroom" the anneal must not flatten in regions the band
never touched):

    out = F^-1[ S + strength_k * M * (target - S) ],  S = F[plane]

- ``M`` (``band_mask``): a radial low-pass over the rfft2 grid. Frequencies
  are measured as a fraction of the Nyquist radius (0.5 cycles/sample);
  ``anneal_band`` is the cutoff fraction. The edge is a raised cosine
  spanning ``band*(1 - ANNEAL_EDGE)`` .. ``band*(1 + ANNEAL_EDGE)`` — hard
  masks ring (Gibbs), a half-band-wide cosine rolloff does not. The DC bin
  is ALWAYS excluded (``M[0,0] = 0``): global mean brightness/color is
  exposure, not structure — re-liquifying it just flashes the frame.
- ``target`` for ``anneal_source='noise'``: the pink machinery of
  ``shaped_init_field`` (image_models/init_noise.py) — the injected field
  is ALWAYS pink-spectrum regardless of ``init_spectrum`` (white lows carry
  no structure per mode: injecting a flat spectrum into a band that spans a
  handful of low bins is votes-for-nothing), with
  ``init_spectrum_falloff`` as its decay power. Chroma on RGB images: an
  explicit ``init_spectrum_chroma`` of 'mono' or 'natural' is honored;
  'full' — the schema's back-compat DEFAULT, kept only so init seeds stay
  reproducible — maps to 'mono', because independent per-channel
  low-frequency fields are exactly the measured chroma-leak pathology
  (color blobs CLIP never cleans up), and an anneal injects once per CYCLE,
  compounding what a full-chroma init does once. There is no full-chroma
  injection in v1. Single-plane images (the PixelImage value plane) are
  inherently mono. See ``injection_profile``.
- ``target`` for ``anneal_source='blur'``: zero in the band (DC excluded),
  i.e. the band decays toward the image mean. This IS "the band pulled
  toward its blurred self, scaled down": a target of ``d * blur(plane)``
  has the same lows as the plane itself, so the general form
  ``S + a*M*(d*S - S)`` is band gain ``1 - a*(1-d)`` — every choice of
  ``d`` is this operation at a reparameterized strength. Zero foreign
  content, softer than noise.

Cycle timing (``cycle_steps``)
------------------------------
``anneal_cycles`` events, evenly spaced through the run's scene-local steps
EXCLUDING a protected tail: the last ``PROTECTED_TAIL_FRACTION`` of the
steps is always anneal-free so detail settles on the final composition.
With ``usable = round(steps * (1 - tail))`` the cycles land at
``round((k+1) * usable / cycles)`` — the last cycle sits exactly at the
tail boundary, never at step 0 (the init is already liquid). Runs whose
budget can't give every cycle at least ``MIN_STEPS_PER_CYCLE`` steps of
optimization fail LOUDLY (coarse_to_fine's no-silent-degradation rule):
a cycle whose votes are reset before CLIP can act on them is shredding,
not annealing. The LAST cycle's settle steps ARE the protected tail
(it sits on the boundary), so the tail itself must also hold at least
``MIN_STEPS_PER_CYCLE`` steps — otherwise the final frame keeps most of
the injected low band.

Strength decay (``cycle_strengths``)
------------------------------------
Geometric: cycle ``k`` (0-based, K total) blends at

    strength_k = anneal_strength * ANNEAL_DECAY_FLOOR ** (k / (K - 1))

so the first cycle uses the configured strength and the last uses
``ANNEAL_DECAY_FLOOR`` (0.1) times it, log-linearly in between — the same
shape as diffusion sigma schedules: bold restructuring early, gentle
nudges late. K = 1 uses the configured strength once. No extra knob in v1.

Where it applies (``anneal_image_rep``)
---------------------------------------
- PixelImage (Limited Palette): the ``value`` plane ONLY — brightness is
  what carries composition. The palette is untouched (its identity IS the
  look) and the selection logits (``tensor``) are untouched (selection
  regions are mid/high-frequency structure — the thing annealing is
  supposed to preserve).
- RGBImage (Unlimited Palette): all three channels, chroma per
  ``injection_profile``.
- VQGAN/LlamaGen: rejected loudly at config time
  (``validate_structure_annealing``) and again here — a latent-space
  re-encode is out of scope in v1.

Backends: the torch path applies cycles in ``DirectImageGuide.run_steps``
right before the cycle step's ``train()``. Under ``mlx_full`` the SAME
torch-side operation runs as a host intervention BETWEEN compiled steps:
engine ``write_back`` -> anneal the torch module -> engine
``import_params`` (never an FFT inside the mx.compile graph in v1). Adam
moments are KEPT through a cycle on both backends: the moments for the
re-liquified band are ~10 steps stale (beta1 0.9 forgets fast) while a
reset would also zero the still-valid high-band moments and, under MLX,
poke at the compiled step's threaded state — keep is both simpler and
less wrong. ``optimizer=adamw_sf`` is rejected in v1: its parameters hold
a fast iterate entangled with an internal Polyak average, and a host-side
overwrite desynchronizes the average frames are decoded from.

Interactions, validated loudly (``validate_structure_annealing``):
still mode only; ``auto_stop`` is rejected (a deliberate mid-run loss
reset has no defined plateau semantics); with ``coarse_to_fine`` the
cycles run in the FINAL stage only, scheduled within that stage's own
step budget — earlier stages are already structure-liquid (thumbnail
canvases, and every stage transition re-encodes into a fresh rep, itself
a structural reset). Multi-scene runs anneal each scene independently
(the schedule is scene-local, exactly like phase_scheduling's t_hat),
and the first cycle must land AFTER the interpolation crossfade: a cycle
inside the ramp would spend its recomposition budget converging toward
the OUTGOING scene's fading prompts (the same reason auto_stop never
samples mid-crossfade), so that overlap is rejected at config time.

Direct init holds: a ``direct_init_weight`` loss is a full-band pull
toward a PRE-anneal image — left enabled, it drags the re-liquified band
straight back within a few steps and the cycle is inert (coarse_to_fine
holds the previous stage's composition at weight 2 in exactly the stage
the cycles run in). ``DirectImageGuide`` therefore RELEASES every direct
init hold at the first cycle (logged): the hold does its settling work up
to that step, and from the first re-liquify on CLIP owns composition.

Determinism: the injected noise consumes only the GLOBAL torch RNG (on
both backends — the anneal op is torch-side even under mlx_full, and
``mx.random.state`` is never touched), and the schedule is a pure function
of the config, so a fixed seed reproduces the whole annealed run.
"""

import math

import torch

from pytti.image_models import PixelImage, RGBImage
from pytti.image_models.init_noise import (
    shaped_init_field,
    validate_spectrum_chroma,
)

# valid ranges / choices (single source for the schema validators and the
# runtime checks)
ANNEAL_CYCLES_RANGE = (1, 12)
# strength is (lo, hi]: the low bound is EXCLUSIVE — a zero-strength cycle
# is structure_annealing that does nothing while still consuming RNG draws
# and FFT round-trips (an inert-knob config lie, the coarse_stages rule)
ANNEAL_STRENGTH_RANGE = (0.0, 1.0)
ANNEAL_BAND_RANGE = (0.02, 0.5)
ANNEAL_SOURCE_CHOICES = ("noise", "blur")

# schema defaults, mirrored here so the inert-knob check (a non-default
# anneal_* value on a run with structure_annealing off is a config lie —
# the coarse_stages rule) has one authority
ANNEAL_CYCLES_DEFAULT = 3
ANNEAL_STRENGTH_DEFAULT = 0.5
ANNEAL_BAND_DEFAULT = 0.15
ANNEAL_SOURCE_DEFAULT = "noise"

# the final fraction of a scene's steps that is always anneal-free, so
# detail settles on the final composition (~ the last third; 0.35 leaves
# the default 100-step scene 35 settle steps after a cycle at step 65)
PROTECTED_TAIL_FRACTION = 0.35

# geometric decay floor: the LAST cycle blends at this fraction of the
# configured anneal_strength (see module docstring; no knob in v1)
ANNEAL_DECAY_FLOOR = 0.1

# every cycle must get at least this many optimization steps before the
# next cycle or the end of the run — fewer means CLIP never votes on the
# re-liquified band before it is reset again (or the render ends on it).
# The LAST cycle settles entirely inside the protected tail, so the tail
# must hold at least this many steps too (cycle_steps enforces both).
MIN_STEPS_PER_CYCLE = 5

# raised-cosine mask edge half-width as a fraction of the cutoff: the
# rolloff spans band*(1-EDGE) .. band*(1+EDGE). 0.5 = a half-band-wide
# transition — soft enough not to ring, tight enough that anneal_band
# still means what it says
ANNEAL_EDGE = 0.5


# ---------------------------------------------------------------------------
# config-boundary validators (schema validators defer-import these; the
# schedule/apply functions call them again for direct callers)
# ---------------------------------------------------------------------------


def validate_anneal_cycles(value: int, *, structure_annealing: bool = True) -> None:
    lo, hi = ANNEAL_CYCLES_RANGE
    if not isinstance(value, int) or isinstance(value, bool) or not lo <= value <= hi:
        raise ValueError(
            f"anneal_cycles={value!r} must be an int in [{lo}, {hi}] — the "
            "number of low-frequency re-liquify events across the run."
        )
    if not structure_annealing and value != ANNEAL_CYCLES_DEFAULT:
        raise ValueError(
            f"anneal_cycles={value} only has meaning with "
            "structure_annealing: true — set structure_annealing: true or "
            "remove anneal_cycles."
        )


def validate_anneal_strength(value: float, *, structure_annealing: bool = True) -> None:
    lo, hi = ANNEAL_STRENGTH_RANGE
    if not lo < value <= hi:
        raise ValueError(
            f"anneal_strength={value!r} must be in ({lo:g}, {hi:g}] — the "
            "low-band blend factor at the FIRST cycle (later cycles decay "
            f"geometrically to {ANNEAL_DECAY_FLOOR:g}x of it). Zero is "
            "rejected: a zero-strength cycle changes nothing while still "
            "consuming RNG draws — set structure_annealing: false instead."
        )
    if not structure_annealing and value != ANNEAL_STRENGTH_DEFAULT:
        raise ValueError(
            f"anneal_strength={value} only has meaning with "
            "structure_annealing: true — set structure_annealing: true or "
            "remove anneal_strength."
        )


def validate_anneal_band(value: float, *, structure_annealing: bool = True) -> None:
    lo, hi = ANNEAL_BAND_RANGE
    if not lo <= value <= hi:
        raise ValueError(
            f"anneal_band={value!r} must be in [{lo:g}, {hi:g}] — the "
            "fraction of the Nyquist radius below which frequencies are "
            "re-liquified (soft cosine edge)."
        )
    if not structure_annealing and value != ANNEAL_BAND_DEFAULT:
        raise ValueError(
            f"anneal_band={value} only has meaning with "
            "structure_annealing: true — set structure_annealing: true or "
            "remove anneal_band."
        )


def validate_anneal_source(value: str, *, structure_annealing: bool = True) -> None:
    if value not in ANNEAL_SOURCE_CHOICES:
        raise ValueError(
            f"anneal_source={value!r} is not a valid source. Valid values: "
            f"{list(ANNEAL_SOURCE_CHOICES)} (noise = replace the band with "
            "shaped pink noise; blur = decay the band toward the image "
            "mean — zero foreign content)."
        )
    if not structure_annealing and value != ANNEAL_SOURCE_DEFAULT:
        raise ValueError(
            f"anneal_source={value!r} only has meaning with "
            "structure_annealing: true — set structure_annealing: true or "
            "remove anneal_source."
        )


def validate_structure_annealing(
    *,
    structure_annealing: bool,
    anneal_cycles: int,
    anneal_strength: float,
    anneal_band: float,
    anneal_source: str,
    image_model: str,
    animation_mode: str,
    auto_stop: bool,
    optimizer: str,
    steps_budget: int,
    interpolation_steps: int,
    n_scenes: int,
) -> None:
    """
    Reject every config structure_annealing has no defined behavior for —
    LOUDLY, before any model loads (the validate_coarse_to_fine pattern).
    ``steps_budget`` is the step budget the cycles are scheduled within:
    steps_per_scene on a plain run, the FINAL stage's split under
    coarse_to_fine (workhorse passes the right one). ``n_scenes`` and
    ``interpolation_steps`` gate the crossfade overlap: on a multi-scene
    run the first cycle must land at or after the end of the interp ramp.
    Also enforces the inert-knob rule when structure_annealing is off.
    """
    validate_anneal_cycles(anneal_cycles, structure_annealing=structure_annealing)
    validate_anneal_strength(anneal_strength, structure_annealing=structure_annealing)
    validate_anneal_band(anneal_band, structure_annealing=structure_annealing)
    validate_anneal_source(anneal_source, structure_annealing=structure_annealing)
    if n_scenes < 1:
        raise ValueError(f"n_scenes must be >= 1, got {n_scenes}")
    if interpolation_steps < 0:
        raise ValueError(
            f"interpolation_steps must be >= 0, got {interpolation_steps}"
        )
    if not structure_annealing:
        return
    if image_model in ("VQGAN", "LlamaGen"):
        raise ValueError(
            f"structure_annealing has no defined behavior for image_model="
            f"{image_model!r}: its image state is a codebook-token latent, "
            "and re-liquifying a pixel-domain band would need a latent "
            "re-encode every cycle (out of scope in v1). Use image_model "
            "'Limited Palette' or 'Unlimited Palette', or "
            "structure_annealing: false."
        )
    if animation_mode != "off":
        raise ValueError(
            "structure_annealing schedules re-liquify cycles over a scene's "
            f"steps; animation_mode={animation_mode!r} re-anchors the image "
            "every frame, so the schedule has no defined meaning there. Set "
            "animation_mode: off or structure_annealing: false."
        )
    if auto_stop:
        raise ValueError(
            "auto_stop judges TOTAL-loss plateaus; structure_annealing "
            "deliberately resets the loss mid-run, so a plateau verdict has "
            "no defined meaning (and an early stop would skip scheduled "
            "cycles). Set auto_stop: false or structure_annealing: false."
        )
    if optimizer != "adam":
        raise ValueError(
            f"structure_annealing supports optimizer=adam only; "
            f"{optimizer!r} (schedule-free) holds a fast iterate entangled "
            "with an internal Polyak average, and a host-side parameter "
            "overwrite desynchronizes the average output frames decode "
            "from. Set optimizer: adam or structure_annealing: false."
        )
    # fail loud on a schedule that can't fit
    schedule = cycle_steps(steps_budget, anneal_cycles)
    # a cycle at step s applies before step s trains; the crossfade is
    # active for steps < interpolation_steps (t = i / interp_steps), so a
    # first cycle at s >= interpolation_steps is clear of the ramp. Scene 1
    # crossfades from itself (workhorse passes the same prompt list), so
    # single-scene runs have no wrong-scene ramp to protect.
    if n_scenes > 1 and schedule[0] < interpolation_steps:
        raise ValueError(
            "structure_annealing re-liquifies composition so CLIP can "
            f"re-vote, but the first scheduled cycle (scene step "
            f"{schedule[0]}) lands inside the interpolation_steps="
            f"{interpolation_steps} crossfade, where the OUTGOING scene's "
            "prompts still dominate the loss — the recomposition budget "
            "would converge toward the wrong scene (auto_stop refuses to "
            "sample mid-crossfade for the same reason). Lower "
            f"interpolation_steps to <= {schedule[0]}, lower anneal_cycles, "
            "raise steps_per_scene, or set structure_annealing: false."
        )


# ---------------------------------------------------------------------------
# schedule — pure math, testable without a render
# ---------------------------------------------------------------------------


def cycle_steps(steps_budget: int, anneal_cycles: int) -> tuple[int, ...]:
    """
    The scene-local steps the cycles fire at: ``anneal_cycles`` events
    evenly spaced through ``round(steps_budget * (1 - tail))``, the last
    exactly at the protected-tail boundary, the first never at step 0. A
    cycle at step ``s`` applies BEFORE step ``s`` trains, so a cycle at the
    boundary still leaves the full tail to settle. Fails loud when any
    cycle would get fewer than ``MIN_STEPS_PER_CYCLE`` steps — INCLUDING
    the last cycle, whose settle steps are exactly the protected tail, so
    the tail itself must hold at least ``MIN_STEPS_PER_CYCLE`` steps.
    """
    validate_anneal_cycles(anneal_cycles)
    if steps_budget < 1:
        raise ValueError(f"steps budget must be >= 1, got {steps_budget}")
    usable = round(steps_budget * (1.0 - PROTECTED_TAIL_FRACTION))
    tail = steps_budget - usable
    if usable < anneal_cycles * MIN_STEPS_PER_CYCLE or tail < MIN_STEPS_PER_CYCLE:
        minimum = max(
            math.ceil(
                anneal_cycles
                * MIN_STEPS_PER_CYCLE
                / (1.0 - PROTECTED_TAIL_FRACTION)
            ),
            math.ceil(MIN_STEPS_PER_CYCLE / PROTECTED_TAIL_FRACTION),
        )
        raise ValueError(
            f"structure_annealing at anneal_cycles={anneal_cycles} needs a "
            f"step budget of >= ~{minimum} steps "
            f"({MIN_STEPS_PER_CYCLE} optimization steps per cycle, between "
            f"cycles AND after the last one — whose settle steps are the "
            f"{PROTECTED_TAIL_FRACTION:.0%} protected tail); got "
            f"{steps_budget}. Raise steps_per_scene or lower anneal_cycles."
        )
    return tuple(
        round((k + 1) * usable / anneal_cycles) for k in range(anneal_cycles)
    )


def cycle_strengths(anneal_strength: float, anneal_cycles: int) -> tuple[float, ...]:
    """
    Per-cycle blend factors: geometric decay from ``anneal_strength`` at
    the first cycle to ``ANNEAL_DECAY_FLOOR * anneal_strength`` at the
    last (log-linear in between; a single cycle uses the full strength).
    """
    validate_anneal_cycles(anneal_cycles)
    validate_anneal_strength(anneal_strength)
    if anneal_cycles == 1:
        return (anneal_strength,)
    return tuple(
        anneal_strength * ANNEAL_DECAY_FLOOR ** (k / (anneal_cycles - 1))
        for k in range(anneal_cycles)
    )


def anneal_schedule(
    steps_budget: int, anneal_cycles: int, anneal_strength: float
) -> dict[int, float]:
    """Scene-local step -> blend strength for every scheduled cycle."""
    steps = cycle_steps(steps_budget, anneal_cycles)
    strengths = cycle_strengths(anneal_strength, anneal_cycles)
    return dict(zip(steps, strengths, strict=True))


# ---------------------------------------------------------------------------
# the operation
# ---------------------------------------------------------------------------


def band_mask(
    height: int, width: int, band: float, device: torch.device | str
) -> torch.Tensor:
    """
    ``[height, width // 2 + 1]`` rfft2-grid low-pass mask in [0, 1]: 1 for
    radial frequencies below ``band * (1 - ANNEAL_EDGE)`` of Nyquist,
    raised-cosine rolloff to 0 at ``band * (1 + ANNEAL_EDGE)``. The DC bin
    is forced to 0 (mean brightness is exposure, not structure).
    """
    validate_anneal_band(band)
    if height < 1 or width < 1:
        raise ValueError(f"dims must be positive, got {height}x{width}")
    fy = torch.fft.fftfreq(height, device=device)
    fx = torch.fft.rfftfreq(width, device=device)
    # radial frequency as a fraction of the Nyquist radius (0.5 c/sample)
    radius = torch.sqrt(fy[:, None] ** 2 + fx[None, :] ** 2) / 0.5
    lo = band * (1.0 - ANNEAL_EDGE)
    hi = band * (1.0 + ANNEAL_EDGE)
    t = ((hi - radius) / (hi - lo)).clamp(0.0, 1.0)
    mask = 0.5 - 0.5 * torch.cos(math.pi * t)
    mask[0, 0] = 0.0
    return mask


def injection_profile(
    init_spectrum_falloff: float, init_spectrum_chroma: str
) -> tuple[float, str]:
    """
    (falloff, chroma) for the injected noise field. The spectrum is ALWAYS
    pink (white lows are meaningless as structure votes — see the module
    docstring); the falloff is the user's ``init_spectrum_falloff``
    verbatim (schema default 1.0 = the natural 1/f). Chroma: an explicit
    ``init_spectrum_chroma`` of 'mono' or 'natural' is honored; 'full' —
    the schema's back-compat DEFAULT, kept only so init seeds stay
    reproducible, so it never distinguishes a choice from an untouched
    knob — maps to 'mono': independent per-channel low-frequency fields
    are the measured chroma-leak pathology (color blobs CLIP never cleans
    up), and an anneal injects once per CYCLE, compounding what a
    full-chroma init does once. No full-chroma injection exists in v1.
    """
    validate_spectrum_chroma(init_spectrum_chroma)
    if init_spectrum_chroma == "full":
        return init_spectrum_falloff, "mono"
    return init_spectrum_falloff, init_spectrum_chroma


def anneal_plane(
    plane: torch.Tensor,
    *,
    strength: float,
    band: float,
    source: str,
    noise_falloff: float = 1.0,
    noise_chroma: str = "mono",
    clamp: bool = True,
) -> torch.Tensor:
    """
    One re-liquify pass over a value-domain ``[channels, height, width]``
    plane (see the module docstring for the math). Returns a new tensor.
    ``noise_falloff``/``noise_chroma`` parameterize the injected pink field
    for ``source='noise'`` (resolve them with ``injection_profile``);
    ``source='blur'`` ignores them.

    ``clamp`` must match the plane's own PARAMETER domain: True for planes
    clamped to [0, 1] every step anyway (the PixelImage value plane —
    ``PixelImage.update``), False for unbounded parameter domains (the
    RGBImage tensor: decode clamps, and mid-run params legitimately hold
    out-of-[0, 1] headroom that a whole-plane clamp would silently flatten
    in regions the band mask never touched).
    """
    validate_anneal_strength(strength)
    validate_anneal_source(source)
    if plane.ndim != 3:
        raise ValueError(
            f"anneal_plane wants [channels, height, width], got shape "
            f"{tuple(plane.shape)}"
        )
    channels, height, width = plane.shape
    spectrum = torch.fft.rfft2(plane)
    mask = band_mask(height, width, band, plane.device)
    if source == "noise":
        field = shaped_init_field(
            channels,
            height,
            width,
            "pink",
            noise_falloff,
            plane.device,
            init_spectrum_chroma=noise_chroma,
        )
        target = torch.fft.rfft2(field)
    else:  # "blur" — the band decays toward the image mean (DC is masked)
        target = torch.zeros_like(spectrum)
    blended = spectrum + strength * mask * (target - spectrum)
    out = torch.fft.irfft2(blended, s=(height, width))
    return out.clamp_(0.0, 1.0) if clamp else out


@torch.no_grad()
def anneal_image_rep(
    img,
    *,
    strength: float,
    band: float,
    source: str,
    init_spectrum_falloff: float,
    init_spectrum_chroma: str,
) -> None:
    """
    Apply one cycle to a live image rep, in place (parameter identity is
    preserved — ``copy_`` under no_grad, the ``encode_random`` mechanism —
    so the optimizer's per-param state stays attached).

    PixelImage: the value plane only (brightness carries composition; the
    palette is the look and the selection logits ARE mid/high-frequency
    structure — both stay untouched in v1), clamped to [0, 1] like its own
    per-step ``update()``. RGBImage: all channels, chroma per
    ``injection_profile``, NOT clamped — its parameter domain is unbounded
    (see ``anneal_plane``). Anything else fails loud — config validation
    rejects VQGAN/LlamaGen long before this, so reaching here with one is
    a wiring bug.
    """
    if isinstance(img, PixelImage):
        falloff, _chroma = injection_profile(
            init_spectrum_falloff, init_spectrum_chroma
        )
        annealed = anneal_plane(
            img.value.unsqueeze(0),
            strength=strength,
            band=band,
            source=source,
            noise_falloff=falloff,
            # a single luminance plane is inherently mono
            noise_chroma="mono",
            clamp=True,
        )
        img.value.copy_(annealed.squeeze(0))
    elif isinstance(img, RGBImage):
        falloff, chroma = injection_profile(
            init_spectrum_falloff, init_spectrum_chroma
        )
        annealed = anneal_plane(
            img.tensor.squeeze(0),
            strength=strength,
            band=band,
            source=source,
            noise_falloff=falloff,
            noise_chroma=chroma,
            clamp=False,
        )
        img.tensor.copy_(annealed.unsqueeze(0))
    else:
        raise ValueError(
            f"structure_annealing has no re-liquify path for "
            f"{type(img).__name__}: only PixelImage and RGBImage hold "
            "pixel-domain state (VQGAN/LlamaGen are rejected at config "
            "time — latent re-encode is out of scope in v1)."
        )
