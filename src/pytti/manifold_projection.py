"""
Manifold projection — periodic projection of the canvas onto a frozen VQ
decoder's natural-image manifold (config ``manifold_projection`` + the
``projection_*`` knobs in structured_config).

Max's ask: "what happens too if you alternate them each step or something
like that" — alternate pytti's CLIP-gradient optimization with the LlamaGen
decoder prior. The mechanism: every ``projection_every`` steps, decode the
current canvas to pixels, run it through the FROZEN LlamaGen VQ tokenizer
round-trip (encoder -> quantize -> decoder), and blend the projected image
back at ``projection_strength``:

    out = (1 - strength) * x + strength * VQ(x)

Projected gradient descent onto the decoder manifold. This is the OPPOSITE
intervention to structure_annealing (falsified 2026-08-06: injecting
foreign low-frequency noise destroyed already-good composition): the
projection carries ZERO foreign content — VQ(x) is the nearest
natural-image rendering of the canvas's OWN structure, so composition
survives by construction and only the accumulated speckle / unclean color
that has no representation on the decoder manifold gets pulled off. Target:
kill accumulated noise while keeping the pytti texture character. Because
the projection preserves structure, direct init holds are NOT released at
the first cycle (annealing releases them because a hold cancels a
re-liquify; a hold merely disagrees with the projection about the noise
being removed, and the projection reapplies every cycle), and a cycle
inside a multi-scene interpolation crossfade needs no rejection (there is
no recomposition budget to spend on the wrong scene's prompts).

Cycle timing (``projection_steps``)
-----------------------------------
Scene-local steps ``projection_every, 2*projection_every, ...`` up to the
protected-tail boundary: the last ``PROTECTED_TAIL_FRACTION`` (the
structure-annealing constant, reused — same reasoning: character settles
on the final state) of each scene's steps is always projection-free. A
projection at step ``s`` applies BEFORE step ``s`` trains (the annealing
convention), so a projection exactly at the boundary still leaves the full
tail. A budget too small for even ONE projection fails LOUDLY (an inert
``manifold_projection: true`` is a config lie — the coarse_stages rule).
No strength decay in v1: the projection is gentle by construction (it
converges — projecting an already-on-manifold image is a near-no-op), so
every cycle blends at the configured strength.

Where it applies (``project_image_rep``)
----------------------------------------
Pixel-domain canvases only; the blend happens in pixel space at the
LOGICAL grid (config width x height — pixel_size upsampling is nearest and
carries no extra information), then re-encodes into the image model:

- RGBImage (Unlimited Palette): the raw parameter plane blends directly.
  The VQ input is clamped to [0, 1] (the decoder's domain); the blend
  rides on the UNclamped parameters, so mid-run out-of-[0, 1] "headroom"
  decays toward the in-gamut projection at rate ``strength`` — deliberate:
  accumulated out-of-gamut push is part of the unclean-color pathology
  this feature exists to remove.
- FourierImage (fourier_parameterization): decode the spectrum to pixels
  (``get_image_tensor``), blend, and re-encode through ``set_image_tensor``
  (clamp-eps -> logit -> inverse color matrix -> rfft2 -> unscale) — the
  documented exact-inverse pipeline, cheap (two FFTs) and image-domain
  exact up to the logit clamp + float error.
- PixelImage (Limited Palette): the LIGHT re-encode. ``encode_image``'s
  smart_encode runs a 201-step Adam palette fit — far too heavy per cycle.
  Instead: decode the VISIBLE canvas (the discrete decode branch, strided
  back down from pixel_size), VQ-project it, and land ONLY the projection's
  brightness DELTA on the value plane, at rate ``strength``::

      value += strength * (hsp(VQ(canvas)) - hsp(canvas)),  clamped [0, 1]

  where ``hsp`` is the exact HSP-luma formula encode_image uses
  (``hsp_value``: sqrt(0.299 r^2 + 0.587 g^2 + 0.114 b^2)). The DELTA form
  matters: ``hsp(decode(value))`` is NOT ``value`` (decode quantizes the
  value plane through the palette's brightness levels), so writing
  ``hsp(blended)`` back wholesale would land that remap at FULL strength
  regardless of ``projection_strength`` — measured mean |Δvalue| ~0.15 per
  cycle against an IDENTITY tokenizer at strength 0.05, with a persistent
  two-cycle oscillation under repeated projection. The delta form is
  exactly zero when VQ(canvas) has the canvas's own brightness (a true
  projection: identity on its fixed points) and scales linearly with
  ``strength``. The palette (its identity IS the look) and the selection
  logits (selection regions are the structure the projection preserves
  anyway) stay untouched — the same value-plane-only decision
  structure_annealing made, for the same reasons. Cost: one decode + one
  VQ round-trip + two channel norms, ZERO optimization steps.
  Approximation, documented: the projection's CHROMA corrections are
  discarded (they cannot land without moving the palette or logits); its
  brightness-domain cleaning (speckle, value-plane dithering junk) lands
  at ``strength``.
- VQGAN/LlamaGen image models: rejected loudly at config time and again
  here — the latent state already lives on a decoder manifold, so
  "projecting" it is a no-op at best and, across tokenizer variants, an
  incoherent look-transplant. Use the underpainting workflow instead.

The tokenizer (``load_projection_model``)
-----------------------------------------
The frozen LlamaGen VQ model (``projection_model``: ds8 stride 8 — the
battery's structure king, the default — or ds16 stride 16) loads ONCE,
lazily, at the first projection: eval mode, requires_grad False, no
optimizer state, ~281 MB (ds8) / ~288 MB (ds16) of fp32 weights on the
render device, freed by workhorse at run end (``free_projection_model``).
It is a SEPARATE singleton from image_models.llamagen's LLAMAGEN_MODEL —
image_model=LlamaGen is rejected alongside manifold_projection, so the two
never coexist. The canvas dims must satisfy the tokenizer stride (the conv
ladder downsamples by exactly 2^(levels-1); a non-multiple canvas would
decode to different dims than it encoded from): validated at config time
against width/height (``validate_projection_dims`` — AUTO-aspect dims are
re-checked in workhorse after the init image resolves them, and
``vq_project`` re-checks the live plane at runtime).

Backends: the torch path applies cycles in ``DirectImageGuide.run_steps``
right before the cycle step's ``train()`` (the annealing slot). Under
``mlx_full`` the SAME torch-side operation runs as a host intervention
BETWEEN compiled steps through the existing structure-annealing seam:
engine ``write_back`` -> project the torch module -> ``import_params``.
Adam moments are KEPT through a cycle on both backends (the annealing
decision, and even less wrong here: the projection moves parameters far
less than a re-liquify). ``optimizer=adamw_sf`` is rejected in v1 for the
annealing reason verbatim: a host-side parameter overwrite desynchronizes
its internal Polyak average. ``auto_stop`` is rejected: a periodic
deliberate image edit perturbs the TOTAL loss on a schedule, so a plateau
verdict has no defined meaning. ``structure_annealing`` +
``manifold_projection`` is rejected: one between-steps intervention at a
time in v1. With ``coarse_to_fine`` the projections run in the FINAL stage
only (the annealing precedent: earlier thumbnail stages re-encode at every
transition — already a projection-like reset — and their budgets are for
rendering), scheduled within that stage's own step budget.

Determinism: the projection consumes NO RNG (the VQ round-trip is
deterministic in eval mode and the schedule is a pure function of the
config), so seeded runs reproduce bit-for-bit and the torch/MLX RNG
streams stay in lockstep with a projection-free run until the first cycle.
"""

import gc
import math

import torch

from pytti import empty_cache, vram_usage_mode
from pytti.image_models import FourierImage, PixelImage, RGBImage
from pytti.structure_annealing import PROTECTED_TAIL_FRACTION

# valid ranges / choices (single source for the schema validators and the
# runtime checks)
PROJECTION_EVERY_RANGE = (5, 200)
# strength is (lo, hi]: the low bound is EXCLUSIVE — a zero-strength
# projection changes nothing while still paying a full VQ round-trip per
# cycle (an inert-knob config lie, the coarse_stages rule)
PROJECTION_STRENGTH_RANGE = (0.0, 1.0)
PROJECTION_MODEL_CHOICES = ("ds16", "ds8")

# schema defaults, mirrored here so the inert-knob check has one authority
PROJECTION_EVERY_DEFAULT = 30
PROJECTION_STRENGTH_DEFAULT = 0.5
PROJECTION_MODEL_DEFAULT = "ds8"

# conv-ladder downsample factor per tokenizer variant: the canvas dims
# must be multiples of this or the decoder round-trip changes dims
PROJECTION_STRIDES = {"ds16": 16, "ds8": 8}

_PROJECTION_MODEL = None
_PROJECTION_NAME: str | None = None


# ---------------------------------------------------------------------------
# config-boundary validators (schema validators defer-import these; the
# schedule/apply functions call them again for direct callers)
# ---------------------------------------------------------------------------


def validate_projection_every(value: int, *, manifold_projection: bool = True) -> None:
    lo, hi = PROJECTION_EVERY_RANGE
    if not isinstance(value, int) or isinstance(value, bool) or not lo <= value <= hi:
        raise ValueError(
            f"projection_every={value!r} must be an int in [{lo}, {hi}] — "
            "the number of optimization steps between manifold projections."
        )
    if not manifold_projection and value != PROJECTION_EVERY_DEFAULT:
        raise ValueError(
            f"projection_every={value} only has meaning with "
            "manifold_projection: true — set manifold_projection: true or "
            "remove projection_every."
        )


def validate_projection_strength(
    value: float, *, manifold_projection: bool = True
) -> None:
    lo, hi = PROJECTION_STRENGTH_RANGE
    if not lo < value <= hi:
        raise ValueError(
            f"projection_strength={value!r} must be in ({lo:g}, {hi:g}] — "
            "the blend factor toward the VQ-projected image (1 = full "
            "replace). Zero is rejected: a zero-strength projection changes "
            "nothing while still paying a full tokenizer round-trip per "
            "cycle — set manifold_projection: false instead."
        )
    if not manifold_projection and value != PROJECTION_STRENGTH_DEFAULT:
        raise ValueError(
            f"projection_strength={value} only has meaning with "
            "manifold_projection: true — set manifold_projection: true or "
            "remove projection_strength."
        )


def validate_projection_model(value: str, *, manifold_projection: bool = True) -> None:
    if value not in PROJECTION_MODEL_CHOICES:
        raise ValueError(
            f"projection_model={value!r} is not a valid tokenizer. Valid "
            f"values: {list(PROJECTION_MODEL_CHOICES)} (ds8 = stride 8, "
            "finer texture; ds16 = stride 16)."
        )
    if not manifold_projection and value != PROJECTION_MODEL_DEFAULT:
        raise ValueError(
            f"projection_model={value!r} only has meaning with "
            "manifold_projection: true — set manifold_projection: true or "
            "remove projection_model."
        )


def validate_projection_dims(width: int, height: int, projection_model: str) -> None:
    """
    The tokenizer's conv ladder downsamples by exactly its stride; a
    non-multiple canvas would decode to different dims than it encoded
    from, so the blend has no defined meaning. Checked against the LOGICAL
    grid (config width/height — that is where the projection runs).
    AUTO-aspect dims from Create are /8-rounded, so ds8 usually fits; ds16
    needs /16 dims.
    """
    validate_projection_model(projection_model)
    stride = PROJECTION_STRIDES[projection_model]
    if width < 1 or height < 1:
        raise ValueError(f"dims must be positive, got {width}x{height}")
    if width % stride or height % stride:
        raise ValueError(
            f"manifold_projection with projection_model="
            f"{projection_model!r} needs width and height to be multiples "
            f"of {stride} (the tokenizer's conv-ladder stride), got "
            f"{width}x{height}. Round the dims to /{stride}, or use "
            + (
                "projection_model: ds8 (stride 8)."
                if projection_model == "ds16" and width % 8 == 0 and height % 8 == 0
                else "different dims."
            )
        )


def validate_manifold_projection(
    *,
    manifold_projection: bool,
    projection_every: int,
    projection_strength: float,
    projection_model: str,
    image_model: str,
    animation_mode: str,
    structure_annealing: bool,
    auto_stop: bool,
    optimizer: str,
    steps_budget: int,
    width: int,
    height: int,
) -> None:
    """
    Reject every config manifold_projection has no defined behavior for —
    LOUDLY, before any model loads (the validate_structure_annealing
    pattern). ``steps_budget`` is the budget the projections are scheduled
    within: steps_per_scene on a plain run, the FINAL stage's split under
    coarse_to_fine (workhorse passes the right one). ``width``/``height``
    are the LOGICAL canvas dims; -1 means AUTO-aspect (resolved from the
    init image later) — workhorse re-validates the stride after resolution
    and ``vq_project`` re-checks the live plane at runtime, so a bad
    resolved dim still fails loud before the first projection. Also
    enforces the inert-knob rule when manifold_projection is off.
    """
    validate_projection_every(projection_every, manifold_projection=manifold_projection)
    validate_projection_strength(
        projection_strength, manifold_projection=manifold_projection
    )
    validate_projection_model(projection_model, manifold_projection=manifold_projection)
    if not manifold_projection:
        return
    if image_model in ("VQGAN", "LlamaGen"):
        raise ValueError(
            f"manifold_projection has no defined behavior for image_model="
            f"{image_model!r}: its image state is already a codebook-token "
            "latent living on a decoder manifold — projecting it onto the "
            "LlamaGen manifold is a no-op at best and a look-transplant at "
            "worst. Use image_model 'Limited Palette' or 'Unlimited "
            "Palette' (the underpainting workflow covers latent looks), or "
            "manifold_projection: false."
        )
    if animation_mode != "off":
        raise ValueError(
            "manifold_projection schedules projection cycles over a "
            f"scene's steps; animation_mode={animation_mode!r} re-anchors "
            "the image every frame, so the schedule has no defined meaning "
            "there. Set animation_mode: off or manifold_projection: false."
        )
    if structure_annealing:
        raise ValueError(
            "structure_annealing + manifold_projection is unsupported: one "
            "between-steps intervention at a time in v1 (their interaction "
            "— re-liquify then project the injected noise — is unmeasured). "
            "Set structure_annealing: false or manifold_projection: false."
        )
    if auto_stop:
        raise ValueError(
            "auto_stop judges TOTAL-loss plateaus; manifold_projection "
            "deliberately edits the image every projection_every steps, "
            "perturbing the loss on a schedule, so a plateau verdict has "
            "no defined meaning. Set auto_stop: false or "
            "manifold_projection: false."
        )
    if optimizer != "adam":
        raise ValueError(
            f"manifold_projection supports optimizer=adam only; "
            f"{optimizer!r} (schedule-free) holds a fast iterate entangled "
            "with an internal Polyak average, and a host-side parameter "
            "overwrite desynchronizes the average output frames decode "
            "from. Set optimizer: adam or manifold_projection: false."
        )
    # fail loud on a schedule that can't fit even one projection
    projection_steps(steps_budget, projection_every)
    if width > 0 and height > 0:
        validate_projection_dims(width, height, projection_model)


# ---------------------------------------------------------------------------
# schedule — pure math, testable without a render
# ---------------------------------------------------------------------------


def projection_steps(steps_budget: int, projection_every: int) -> tuple[int, ...]:
    """
    The scene-local steps projections fire at: every ``projection_every``
    steps through ``round(steps_budget * (1 - tail))`` — the last
    ``PROTECTED_TAIL_FRACTION`` of the steps (the structure-annealing
    constant, reused) stays projection-free so character settles. A
    projection at step ``s`` applies BEFORE step ``s`` trains, so a
    projection exactly at the boundary still leaves the full tail; step 0
    is never scheduled by construction (the first cycle sits at
    ``projection_every``). Fails loud when not even ONE projection fits —
    manifold_projection: true that never projects is a config lie.
    """
    validate_projection_every(projection_every)
    if steps_budget < 1:
        raise ValueError(f"steps budget must be >= 1, got {steps_budget}")
    usable = round(steps_budget * (1.0 - PROTECTED_TAIL_FRACTION))
    steps = tuple(range(projection_every, usable + 1, projection_every))
    if not steps:
        minimum = math.ceil(projection_every / (1.0 - PROTECTED_TAIL_FRACTION))
        raise ValueError(
            f"manifold_projection at projection_every={projection_every} "
            f"needs a step budget of >= ~{minimum} steps for even one "
            f"projection to land before the {PROTECTED_TAIL_FRACTION:.0%} "
            f"protected tail; got {steps_budget}. Raise steps_per_scene, "
            "lower projection_every, or set manifold_projection: false."
        )
    return steps


# ---------------------------------------------------------------------------
# the frozen tokenizer (lazy singleton, freed at run end)
# ---------------------------------------------------------------------------


def load_projection_model(model_name: str, device=None):
    """
    Load the frozen LlamaGen VQ tokenizer for ``model_name`` (eval mode,
    requires_grad False — load_llamagen_model guarantees both), once:
    repeat calls with the same name return the singleton. Weights are the
    revision-pinned FoundationVision/LlamaGen checkpoints (~281 MB fp32
    ds8 / ~288 MB ds16 on the render device). Separate singleton from
    image_models.llamagen's LLAMAGEN_MODEL: image_model=LlamaGen is
    rejected alongside manifold_projection, so the two never coexist.
    """
    validate_projection_model(model_name)
    global _PROJECTION_MODEL, _PROJECTION_NAME
    if _PROJECTION_NAME == model_name and _PROJECTION_MODEL is not None:
        return _PROJECTION_MODEL
    from huggingface_hub import hf_hub_download

    from pytti.config.model_names import LLAMAGEN_CHECKPOINT_FILES
    from pytti.device import default_device
    from pytti.image_models.llamagen import (
        LLAMAGEN_HF_REPO,
        LLAMAGEN_HF_REVISION,
        load_llamagen_model,
    )

    if device is None:
        device = default_device()
    checkpoint_path = hf_hub_download(
        LLAMAGEN_HF_REPO,
        LLAMAGEN_CHECKPOINT_FILES[model_name],
        revision=LLAMAGEN_HF_REVISION,
    )
    model = load_llamagen_model(checkpoint_path, model_name)
    with vram_usage_mode("Manifold Projection"):
        _PROJECTION_MODEL = model.to(device)
    _PROJECTION_NAME = model_name
    return _PROJECTION_MODEL


def free_projection_model() -> None:
    """Release the tokenizer at run end (workhorse's finally)."""
    global _PROJECTION_MODEL, _PROJECTION_NAME
    _PROJECTION_MODEL = None
    _PROJECTION_NAME = None
    gc.collect()
    empty_cache()


# ---------------------------------------------------------------------------
# the operation
# ---------------------------------------------------------------------------


@torch.no_grad()
def vq_project(pixels: torch.Tensor, model) -> torch.Tensor:
    """
    One frozen VQ tokenizer round-trip: ``[3, height, width]`` pixels in
    [0, 1] -> encoder -> quantize -> decoder -> ``[3, height, width]`` in
    [0, 1] (the decoder's [-1, 1] output remapped and clamped, exactly the
    LlamaGenImage.decode convention). Deterministic — consumes no RNG.
    Fails loud when the dims don't satisfy the model's conv-ladder stride
    (the runtime backstop behind validate_projection_dims).
    """
    if pixels.ndim != 3 or pixels.shape[0] != 3:
        raise ValueError(
            f"vq_project wants [3, height, width] pixels, got shape "
            f"{tuple(pixels.shape)}"
        )
    stride = 2 ** (model.decoder.num_resolutions - 1)
    _, height, width = pixels.shape
    if height % stride or width % stride:
        raise ValueError(
            f"vq_project: canvas {width}x{height} is not a multiple of the "
            f"tokenizer stride {stride} — the decoder round-trip would "
            "change dims (validate_projection_dims should have rejected "
            "this at config time)."
        )
    quant, _, _ = model.encode(pixels.unsqueeze(0) * 2 - 1)
    out = model.decode(quant).squeeze(0)
    return out.add_(1).div_(2).clamp_(0.0, 1.0)


@torch.no_grad()
def project_pixels(pixels: torch.Tensor, model, strength: float) -> torch.Tensor:
    """
    The projection blend: ``(1 - strength) * pixels + strength *
    VQ(clamp(pixels))``. The clamp only guards RGBImage's out-of-[0, 1]
    parameter headroom at the VQ INPUT (the decoder's domain); the blend
    rides on the raw values, so headroom decays toward the in-gamut
    projection at rate ``strength`` — deliberate (module docstring).
    Returns a new tensor.
    """
    validate_projection_strength(strength)
    projected = vq_project(pixels.clamp(0.0, 1.0), model)
    return (1.0 - strength) * pixels + strength * projected


def hsp_value(pixels: torch.Tensor) -> torch.Tensor:
    """
    The HSP brightness PixelImage.encode_image derives its value plane
    from (https://alienryderflex.com/hsp.html): ``[3, height, width]``
    pixels -> ``[height, width]`` in [0, 1]. Kept formula-identical to
    encode_image so the light re-encode and the full fit agree on what
    "value" means.
    """
    if pixels.ndim != 3 or pixels.shape[0] != 3:
        raise ValueError(
            f"hsp_value wants [3, height, width] pixels, got shape "
            f"{tuple(pixels.shape)}"
        )
    magic_color = pixels.new_tensor([0.299, 0.587, 0.114]).view(3, 1, 1)
    return torch.linalg.vector_norm(pixels * magic_color.sqrt(), dim=0)


@torch.no_grad()
def pixel_canvas(img: PixelImage) -> torch.Tensor:
    """
    The VISIBLE Limited Palette canvas at the logical grid: decode_tensor's
    forward value (the discrete palette decode — what the saved PNG shows),
    strided back down from the nearest pixel_size upsample (``[::scale]``
    picks each block's source sample exactly).
    """
    frame = img.decode_tensor().squeeze(0)  # [3, H*scale, W*scale]
    scale = int(img.scale)
    if scale > 1:
        frame = frame[:, ::scale, ::scale]
    return frame


@torch.no_grad()
def project_image_rep(img, *, strength: float, model) -> None:
    """
    Apply one projection cycle to a live image rep, in place (parameter
    identity preserved — ``copy_`` under no_grad — so Adam's per-param
    state stays attached; the KEEP-moments decision, shared with
    annealing).

    PixelImage: the LIGHT re-encode (module docstring) — decode the
    visible canvas, VQ-project it, and add the projection's HSP-brightness
    DELTA to the value plane at rate ``strength`` (never a wholesale
    ``hsp(blended)`` replace: ``hsp(decode(value))`` != ``value``, so a
    replace would land the decode's palette-quantization remap at full
    strength no matter how small ``strength`` is — the delta form is
    exactly zero on on-manifold canvases). Palette + selection logits
    untouched. RGBImage: blend on the raw parameter plane (headroom
    decays toward gamut — deliberate). FourierImage: pixel-domain blend
    re-encoded through the exact-inverse spectrum pipeline. Anything else
    fails loud — config validation rejects VQGAN/LlamaGen long before
    this, so reaching here with one is a wiring bug.
    """
    if isinstance(img, PixelImage):
        validate_projection_strength(strength)
        canvas = pixel_canvas(img)
        delta = hsp_value(vq_project(canvas, model)) - hsp_value(canvas)
        img.value.add_(strength * delta).clamp_(0.0, 1.0)
    elif isinstance(img, FourierImage):
        pixels = img.get_image_tensor().detach()
        img.set_image_tensor(project_pixels(pixels, model, strength))
    elif isinstance(img, RGBImage):
        state = img.tensor.detach().squeeze(0)
        img.tensor.copy_(project_pixels(state, model, strength).unsqueeze(0))
    else:
        raise ValueError(
            f"manifold_projection has no projection path for "
            f"{type(img).__name__}: only PixelImage, RGBImage, and "
            "FourierImage hold pixel-domain canvases (VQGAN/LlamaGen are "
            "rejected at config time — their state already lives on a "
            "decoder manifold)."
        )
