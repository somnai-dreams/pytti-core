"""
Whole-step assembly — M2 slice S5 (docs/mlx-m2-seam-map.md, Stage A + §4).

One ``mx.compile``'d function runs everything ``DirectImageGuide.train``
does between "decode" and "post-Adam clamps": S1 decode -> S3 sampler per
cut-size group -> S4 augs + noise -> per-tower CLIP normalize -> fp16
towers (M1's ``VisionTower``) -> the M1 semantic reduction with
geometric-mask weights computed in-step -> S2 direct losses + the in-tree
image losses -> ``mx.value_and_grad`` over the trainable params ->
``mlx.optimizers.Adam`` (bias-corrected, torch semantics) -> S1 clamps.

Compile-state contract (seam map §4)
------------------------------------
The compiled callable threads ``[params, opt.state, mx.random.state]``
through ``inputs=``/``outputs=``:

- ``params`` — the S1 image tree (fp32), mutated in place per step;
- ``opt.state`` — Adam step count + per-param ``m``/``v`` moments;
- ``mx.random.state`` — every sampler/aug/noise draw uses the implicit
  global stream, so draws are fresh each step and ``mx.random.seed`` at
  setup makes runs reproducible.

Per-step host values (parametric weights/stops/thresholds, the
interpolation ramp folded into per-prompt scales) enter as fp32 ``mx.array``
ARGUMENTS — never as captured Python scalars, so a new ``t`` can never
freeze into the trace or force a retrace. Retraces happen only when the
step's *structure* changes (prompt set, gradient accumulation, resolution),
by rebuilding via :func:`build_step` — the engine (S6) owns that keying.

Torch-cadence equivalences (test-gated in tests/test_mlx_engine_step.py):

- Adam: ``mlx.optimizers.Adam(bias_correction=True)`` with torch defaults
  is torch.optim.Adam's exact update (same eps placement) — lockstep gate.
- ``gradient_accumulation_steps`` is an in-step loop: direct/image losses
  ride microbatch 0 undivided, the semantic total is divided by ``gas``
  each microbatch, and one gradient of the summed total equals torch's
  summed per-microbatch backwards (gradients are linear).
- frame-boundary ``set_optim(None)`` (fires every ``steps_per_frame`` even
  in still mode, ImageGuide.py:408-425) == :func:`reset_adam_state`: zero
  moments, zero step count, in-state.
- update order: Adam step, THEN the S1 ``update()`` clamps
  (ImageGuide.py:317-318).

``mlx`` is imported at module level: import this module only lazily (via
the package ``__getattr__`` or inside function bodies) so linux CI never
touches mlx.
"""

import math
from dataclasses import dataclass

import mlx.core as mx
import mlx.optimizers

from pytti.mlx_engine import PIXEL_TRAINABLE_KEYS, RGB_TRAINABLE_KEYS
from pytti.mlx_engine.augs import AugConfig, apply_augs, draw_aug_params
from pytti.mlx_engine.image_models import (
    hdr_loss,
    palette_loss,
    pixel_decode,
    pixel_update,
    rgb_decode,
    rgb_update,
)
from pytti.mlx_engine.losses import (
    edge_loss,
    hsv_loss,
    loss_forward,
    mse_loss,
    tv_loss,
)
from pytti.mlx_engine.sampler import (
    pad_image,
    pytti_batched,
    pytti_full,
    pytti_smart,
)
from pytti.Perceptor.mlx_backend.bridge import mlx_semantic_reduction

IMAGE_KINDS = ("pixel", "rgb")
SAMPLERS = {"batched": pytti_batched, "smart": pytti_smart, "full": pytti_full}
DIRECT_LOSS_KINDS = ("tv", "mse", "hsv", "edge")
# geometric mask keys (prompt_spec.GEOMETRIC_MASK_KEYS): "a" == mask_all
MASK_KINDS = ("a", "r", "l", "d", "u", "n", "f")


# ---------------------------------------------------------------------------
# optimizer — torch.optim.Adam semantics (seam map Stage A "Adam semantics")
# ---------------------------------------------------------------------------


def make_adam(lr: float) -> mlx.optimizers.Adam:
    """torch ``make_optimizer(..., "adam", lr)`` equivalent: torch defaults
    betas=(0.9, 0.999), eps=1e-8, weight decay 0, bias-corrected (mlx
    defaults ``bias_correction=False`` — MUST be True for torch parity)."""
    if not (isinstance(lr, float) and lr > 0):
        raise ValueError(f"lr must be a positive float, got {lr!r}")
    return mlx.optimizers.Adam(
        learning_rate=lr, betas=[0.9, 0.999], eps=1e-8, bias_correction=True
    )


def reset_adam_state(opt: mlx.optimizers.Adam) -> None:
    """
    The MLX equivalent of ``DirectImageGuide.set_optim(None)`` rebuilding a
    fresh ``torch.optim.Adam``: zero every ``m``/``v`` moment and the step
    count, writing INTO the existing state tree (its arrays are listed in
    the compiled step's ``inputs=``/``outputs=`` — values may change,
    structure must not).
    """
    state = opt.state
    state["step"] = mx.array(0, mx.uint64)
    moments = 0
    for key, entry in state.items():
        if key in ("step", "learning_rate"):
            continue
        if not isinstance(entry, dict) or set(entry) != {"m", "v"}:
            raise ValueError(
                f"unexpected Adam state entry {key!r}: {type(entry).__name__} "
                f"with keys {sorted(entry) if isinstance(entry, dict) else '-'}"
            )
        entry["m"] = mx.zeros_like(entry["m"])
        entry["v"] = mx.zeros_like(entry["v"])
        moments += 1
    if moments == 0:
        raise ValueError(
            "optimizer state has no per-param moments — reset_adam_state must "
            "be called after opt.init(trainable) (the engine inits at setup)"
        )


# ---------------------------------------------------------------------------
# step plans — the mlx-side halves of the engine's parsed torch objects
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DirectLossPlan:
    """One active direct loss (S2): static constants only. The parametric
    weight/stop are per-step host inputs, not stored here."""

    name: str  # str(aug) — the torch loss-record key
    kind: str  # "tv" | "mse" | "hsv" | "edge"
    comp: "mx.array | None"  # pre-converted comp constant (None for tv)
    mask: "mx.array | None"  # [1,1,H,W] at input resolution, or None

    def __post_init__(self):
        if self.kind not in DIRECT_LOSS_KINDS:
            raise ValueError(
                f"unknown direct loss kind {self.kind!r}; "
                f"expected one of {DIRECT_LOSS_KINDS}"
            )
        if (self.comp is None) != (self.kind == "tv"):
            raise ValueError(
                f"{self.name!r}: comp must be None iff kind == 'tv' "
                f"(got kind={self.kind!r}, comp is "
                f"{'None' if self.comp is None else 'set'})"
            )


@dataclass(frozen=True)
class PromptPlan:
    """One active semantic prompt: embeddings + mask structure. Parametric
    weight/stop/cutoff and the interp scale are per-step host inputs."""

    name: str  # str(prompt) — the torch loss-record key
    embeds: mx.array  # [C, i_max] fp32, ensemble-padded (as registered)
    mask_kind: str  # one of MASK_KINDS
    recorded: bool  # prompts record raws; interp prompts don't

    def __post_init__(self):
        if self.mask_kind not in MASK_KINDS:
            raise ValueError(
                f"prompt {self.name!r}: unknown mask kind {self.mask_kind!r}; "
                f"expected one of {MASK_KINDS}"
            )


@dataclass(frozen=True)
class StepConfig:
    """Trace-time constants of one compiled step."""

    image_kind: str  # "pixel" | "rgb"
    scale: int
    use_palette_target: bool  # pixel only (False for rgb)
    image_loss_names: tuple  # image-loss kinds ("hdr", "palette"), in
    # PixelImage.image_loss() order; () for rgb. The engine maps these to
    # the torch record names ("HDR normalization", "Palette normalization").
    side_x: int
    side_y: int
    cutn: int
    cut_pow: float
    padding: float
    border_mode: str
    noise_fac: float
    sampler: str  # "batched" | "smart" | "full"
    gas: int  # gradient_accumulation_steps
    # config coherence_weighting: per-cutout semantic weights from crop
    # geometry (sizes_to_coherence_weights). Trace-time constant BY DESIGN:
    # False must compile the exact pre-feature graph (never an all-ones
    # multiply), and the flag is fixed for a run so it can never retrace.
    coherence_weighting: bool = False

    def __post_init__(self):
        if self.image_kind not in IMAGE_KINDS:
            raise ValueError(
                f"unknown image kind {self.image_kind!r}; "
                f"expected one of {IMAGE_KINDS}"
            )
        if self.sampler not in SAMPLERS:
            raise ValueError(
                f"unknown sampler {self.sampler!r}; expected one of "
                f"{sorted(SAMPLERS)} (classic is torch-only)"
            )
        if self.gas < 1:
            raise ValueError(f"gas must be >= 1, got {self.gas}")
        if self.image_kind == "rgb" and self.image_loss_names != ():
            raise ValueError("rgb images have no image losses")
        if unknown := set(self.image_loss_names) - {"hdr", "palette"}:
            raise ValueError(f"unknown image losses {sorted(unknown)}")


def trainable_keys_for(image_kind: str) -> tuple:
    if image_kind == "pixel":
        return PIXEL_TRAINABLE_KEYS
    if image_kind == "rgb":
        return RGB_TRAINABLE_KEYS
    raise ValueError(f"unknown image kind {image_kind!r}")


# ---------------------------------------------------------------------------
# geometric prompt masks, in-step (seam map Stage E "Masks")
# ---------------------------------------------------------------------------


def geometric_mask_stops(
    kind: str, pos: mx.array, size: mx.array, thresh: mx.array
) -> mx.array:
    """
    The first return of Prompt.py's geometric mask functions (mask_all,
    mask_right/left/down/up/near/far) — the ``mask_stops`` tensor — computed
    from MLX-side crop geometry. ``pos``/``size`` are ``[n, C, 2]`` fp32
    ``(x, y)`` in the samplers' normalized convention; ``thresh`` is the
    parametric-eval'd cutoff (0-dim). Returns ``[n, C]`` fp32. The second
    torch return (``mask_weights``) is the constant 1 for every geometric
    mask — callers fold it implicitly.
    """
    if kind == "a":
        # mask_all: stops = -inf everywhere (Prompt.py:107-108)
        return mx.full(pos.shape[:-1], -math.inf)
    if kind in ("r", "l"):
        cent = pos[..., 0] + size[..., 0] / 2
        gate = (cent < thresh) if kind == "r" else (cent > thresh)
    elif kind in ("d", "u"):
        cent = pos[..., 1] + size[..., 1] / 2
        gate = (cent < thresh) if kind == "d" else (cent > thresh)
    elif kind in ("n", "f"):
        smallest = mx.min(size, axis=-1)
        gate = (smallest < thresh) if kind == "n" else (smallest > thresh)
    else:
        raise ValueError(
            f"unknown geometric mask kind {kind!r}; expected one of {MASK_KINDS}"
        )
    return gate.astype(mx.float32)


def sizes_to_coherence_weights(
    sizes: mx.array, side_x: int, side_y: int
) -> mx.array:
    """
    The MLX mirror of ``Prompt.sizes_to_coherence_weights`` (that docstring
    is the contract; parity is test-gated): full-frame anchors (min-side
    size column == 1.0 exactly, per the sampler contract both samplers
    share) weigh 3x, every cutout scales by its inscribed-square fraction,
    and the whole vector renormalizes to mean 1 so only the gradient's
    DISTRIBUTION changes. ``sizes`` is the step's ``[n, C, 2]`` per-cutout
    geometry — a per-step tensor already in-graph, so this is pure graph
    math: no host sync, no retrace.
    """
    if side_x <= 0 or side_y <= 0:
        raise ValueError(f"canvas dims must be positive, got {(side_x, side_y)}")
    if sizes.ndim < 2 or sizes.shape[-1] != 2:
        raise ValueError(
            f"sizes must be [..., 2] (x, y) size fractions, got {tuple(sizes.shape)}"
        )
    fraction = sizes[..., 0] if side_x <= side_y else sizes[..., 1]
    raw = mx.where(fraction == 1.0, fraction * 3.0, fraction)
    return raw / mx.mean(raw)


# ---------------------------------------------------------------------------
# the cutout stage (default production cutter; tests inject their own)
# ---------------------------------------------------------------------------


def make_cutter(cfg: StepConfig, aug_config: AugConfig):
    """
    The production cutout stage: one S3 sampler call per cut-size group with
    S4 augs and sampler-side noise, every draw on the implicit global stream
    (in torch's draw ordering: geometry, then augs, then noise).

    Returns ``cutter(padded_nhwc, cut_size) -> (cutouts, offsets, sizes)``
    — the injection seam gate (b) uses to factor ALL RNG out of the step.
    """
    sampler_fn = SAMPLERS[cfg.sampler]

    def augs(cutouts_nhwc: mx.array) -> mx.array:
        params = draw_aug_params(cutouts_nhwc.shape[0], aug_config)
        nchw = mx.transpose(cutouts_nhwc, (0, 3, 1, 2))
        return mx.transpose(apply_augs(nchw, params, aug_config), (0, 2, 3, 1))

    def cutter(padded_nhwc: mx.array, cut_size: int):
        return sampler_fn(
            input=padded_nhwc,
            side_x=cfg.side_x,
            side_y=cfg.side_y,
            cut_size=cut_size,
            padding=cfg.padding,
            cutn=cfg.cutn,
            cut_pow=cfg.cut_pow,
            border_mode=cfg.border_mode,
            augs=augs,
            noise_fac=cfg.noise_fac,
        )

    return cutter


# ---------------------------------------------------------------------------
# the whole step
# ---------------------------------------------------------------------------


def build_step(
    *,
    params: dict,
    opt: mlx.optimizers.Adam,
    cfg: StepConfig,
    direct_plans: tuple,
    prompt_plans: tuple,
    towers: tuple,
    batch_index: tuple,
    group_cut_sizes: tuple,
    tower_means: tuple,
    tower_stds: tuple,
    cutter=None,
):
    """
    Compile ONE whole-step function over the live ``params`` tree and
    ``opt`` state (both mutated in place per call; ``opt`` must already be
    ``init``'d on the trainable keys).

    ``towers[c]`` is perceptor ``c``'s M1 tower; ``batch_index[c]`` maps it
    to its cut-size group (same-resolution towers share ONE sampler draw,
    exactly like the torch embedder shares cutout tensors by cut_size);
    ``group_cut_sizes`` is the per-group input resolution.
    ``tower_means``/``tower_stds`` are per-TOWER normalization stats
    (``[3]`` fp32) — per-tower rather than per-group because a mixed
    ensemble can share a resolution but not stats (FARE = CLIP constants,
    SigLIP2 = 0.5s, both 224): each tower normalizes its group's shared raw
    cutouts with its own stats, mirroring the torch path's per-perceptor
    ``normalize`` (Embedder.forward). ``cutter`` defaults to
    :func:`make_cutter`; tests inject a deterministic one.

    Returns ``step(aug_w, aug_s, p_w, p_s, p_thresh, p_scale,
    palette_gate) -> (total, aug_raws, image_raws, prompt_raws)``:

    - ``aug_w``/``aug_s``: ``[A]`` fp32 — parametric-eval'd weight/stop per
      active direct loss (``aug_w`` includes the phase-scheduling
      ``weight_scale``, folded host-side by the engine);
    - ``p_w``/``p_s``/``p_thresh``/``p_scale``: ``[P]`` fp32 — per active
      prompt: weight, stop, mask cutoff, interp scale (t or 1-t);
    - ``palette_gate``: 0-dim fp32, 1.0 or 0.0 — phase scheduling's
      Limited Palette lock. It multiplies BOTH the palette gradient (so
      Adam's moments stop accumulating, matching torch's lock where the
      palette leaves the graph) and the palette's applied update (a
      per-param lr gate, so residual momentum cannot drift the palette
      after the lock). Unused for rgb images. Arg-driven by design: the
      lock can never be a graph-structure change / retrace;
    - ``total``: 0-dim — the torch step's TOTAL (sum over microbatches);
    - ``*_raws``: ``[A]`` / ``[len(image_loss_names)]`` / ``[P]`` raw loss
      records (last microbatch's, matching train()'s per-mb overwrite).
    """
    trainable = trainable_keys_for(cfg.image_kind)
    n_groups = len(group_cut_sizes)
    if not (len(tower_means) == len(tower_stds) == len(towers)):
        raise ValueError("tower stats must align with towers (one per tower)")
    if len(towers) != len(batch_index):
        raise ValueError("one batch_index entry per tower")
    if any(not 0 <= b < n_groups for b in batch_index):
        raise ValueError(f"batch_index {batch_index} out of range for {n_groups}")
    if cutter is None:
        cutter = make_cutter(cfg, AugConfig())

    if cfg.image_kind == "pixel":

        def decode(tree):
            return pixel_decode(
                tree, scale=cfg.scale, use_palette_target=cfg.use_palette_target
            )

        update_fn = pixel_update
    else:

        def decode(tree):
            return rgb_decode(tree, scale=cfg.scale)

        update_fn = rgb_update

    def image_losses(tree):
        # PixelImage.image_loss() order: [hdr (iff built), palette]
        out = []
        for name in cfg.image_loss_names:
            if name == "hdr":
                out.append(hdr_loss(tree, use_palette_target=cfg.use_palette_target))
            elif name == "palette":
                out.append(palette_loss(tree))
            else:
                raise ValueError(f"unknown image loss {name!r}")
        return out

    def direct_raw(plan: DirectLossPlan, z: mx.array) -> mx.array:
        if plan.kind == "tv":
            return tv_loss(z)  # [n] vector, torch shape
        if plan.kind == "mse":
            return mse_loss(z, plan.comp, plan.mask)
        if plan.kind == "hsv":
            return hsv_loss(z, plan.comp, plan.mask)
        if plan.kind == "edge":
            return edge_loss(z, plan.comp, plan.mask)
        raise ValueError(f"unknown direct loss kind {plan.kind!r}")

    n_prompts = len(prompt_plans)
    if n_prompts:
        text = mx.stack([plan.embeds for plan in prompt_plans])  # [P, C, i]
        expected = (len(towers), max(t.config.output_dim for t in towers))
        if tuple(text.shape[1:]) != expected:
            raise ValueError(
                f"prompt embeddings are {tuple(text.shape[1:])}, expected "
                f"{expected} — parsed under a different perceptor ensemble?"
            )
        out_dim_max = expected[1]

    def semantic(z, p_w, p_s, p_thresh, p_scale):
        """One microbatch of semantic loss: cutouts -> towers -> reduction
        with in-step geometric mask weights. Returns (total, raws [P])."""
        z_nhwc = mx.transpose(z, (0, 2, 3, 1))
        padded = pad_image(
            z_nhwc, cfg.side_x, cfg.side_y, cfg.padding, cfg.border_mode
        )
        groups = [cutter(padded, cut_size) for cut_size in group_cut_sizes]
        embs = []
        for tower, b, mean, std in zip(
            towers, batch_index, tower_means, tower_stds, strict=True
        ):
            # per-TOWER normalize over the group's shared raw cutouts
            # (stats can differ within a resolution group — see build_step)
            e = tower.encode((groups[b][0] - mean) / std)
            e = e.astype(mx.float32)  # [n, d_c]
            pad = out_dim_max - e.shape[-1]
            if pad:
                e = mx.pad(e, ((0, 0), (0, pad)))
            embs.append(e)
        emb = mx.stack(embs, axis=1)  # [n, C, i_max]

        # crop geometry per perceptor, [n, C, 2] — Prompt.forward's
        # format_input(offsets/sizes) with MLX-side geometry
        pos = mx.stack([groups[b][1] for b in batch_index], axis=1)
        size = mx.stack([groups[b][2] for b in batch_index], axis=1)

        # coherence weights are shared by every prompt (geometry-only);
        # computed IN-GRAPH from the sizes already flowing here — per-step
        # tensors, so the compiled structure is untouched
        coherence = (
            sizes_to_coherence_weights(size, cfg.side_x, cfg.side_y)
            if cfg.coherence_weighting
            else None
        )

        weights, stops = [], []
        for j, plan in enumerate(prompt_plans):
            # Prompt.forward:314-322 with geometric mask_weights == 1
            weight = p_w[j]
            if coherence is not None:
                # multiplies exactly where Prompt.forward composes its mask
                # weights (coherence > 0, so the sign gymnastics below see
                # the same signs as the unweighted path)
                weight = weight * coherence
            mask_stops = geometric_mask_stops(plan.mask_kind, pos, size, p_thresh[j])
            sign_offset = mx.minimum(mx.sign(weight), 0.0)  # sign().clamp(max=0)
            stops.append(mx.maximum(mask_stops + sign_offset, p_s[j]))
            weights.append(mx.broadcast_to(weight, mask_stops.shape))
        total, _losses, raws = mlx_semantic_reduction(
            emb, text, mx.stack(weights), mx.stack(stops), p_scale
        )
        return total, raws

    def loss_fn(tree, aug_w, aug_s, p_w, p_s, p_thresh, p_scale):
        full = {**params, **tree}
        total = mx.zeros(())
        aug_raws = []
        image_raws = []
        prompt_raws = mx.zeros((0,))
        for mb in range(cfg.gas):
            z = decode(full)
            mb_total = mx.zeros(())
            if mb == 0:
                for j, plan in enumerate(direct_plans):
                    loss, raw = loss_forward(direct_raw(plan, z), aug_w[j], aug_s[j])
                    mb_total = mb_total + mx.sum(loss)
                    aug_raws.append(mx.mean(raw))
                for loss, raw in image_losses(full):
                    mb_total = mb_total + loss
                    image_raws.append(raw)
            if n_prompts:
                sem_total, prompt_raws = semantic(z, p_w, p_s, p_thresh, p_scale)
                mb_total = mb_total + sem_total / cfg.gas
            total = total + mb_total
        aug_stack = mx.stack(aug_raws) if aug_raws else mx.zeros((0,))
        image_stack = mx.stack(image_raws) if image_raws else mx.zeros((0,))
        return total, (aug_stack, image_stack, prompt_raws)

    grad_fn = mx.value_and_grad(loss_fn)

    def step(aug_w, aug_s, p_w, p_s, p_thresh, p_scale, palette_gate):
        tree = {key: params[key] for key in trainable}
        (total, (aug_raws, image_raws, prompt_raws)), grads = grad_fn(
            tree, aug_w, aug_s, p_w, p_s, p_thresh, p_scale
        )
        if cfg.image_kind == "pixel":
            # phase-scheduling palette lock (see build_step docstring):
            # gate the gradient AND the applied update — palette exactly
            # frozen at gate 0, bit-identical step at gate 1 (mul by the
            # 0/1 gate is exact; the select keeps the open path untouched)
            grads["palette"] = grads["palette"] * palette_gate
        new_tree = opt.apply_gradients(grads, tree)
        if cfg.image_kind == "pixel":
            new_tree["palette"] = mx.where(
                palette_gate > 0.5, new_tree["palette"], tree["palette"]
            )
        params.update(new_tree)
        params.update(update_fn(params))  # step THEN clamp (ImageGuide:317-318)
        return total, aug_raws, image_raws, prompt_raws

    state = [params, opt.state, mx.random.state]
    return mx.compile(step, inputs=state, outputs=state)
