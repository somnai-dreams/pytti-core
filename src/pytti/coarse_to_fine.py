"""
Coarse-to-fine still rendering — the pure math and validation.

Two config knobs (structured_config, still mode only):

- ``coarse_to_fine: bool`` (default off) turns the staged render on;
- ``coarse_stages: int`` (default 2, valid 2..4) sets the number of stages —
  the pyramid: compose the image as a small thumbnail first, then repeatedly
  scale up and keep rendering.

The stage ladder is geometric and ends at the configured canvas
(``stage_dims``): stages=2 -> 1/2, 1; stages=3 -> 1/4, 1/2, 1;
stages=4 -> 1/8, 1/4, 1/2, 1. Every non-final stage is rounded to the
nearest multiple of 8 with a 64px floor; the final stage is EXACTLY the
configured dims. Earlier stages are cheaper in steps too
(``stage_steps``, the ``STAGE_STEP_PERCENTS`` table): stages=2 -> 40/60,
stages=3 -> 25/25/50, stages=4 -> 15/20/25/40 percent of
``steps_per_scene``.

Each stage transition is the same seam: the finished stage is decoded to
PIL, bicubic-upscaled to the full canvas, persisted (the resume seam), and
re-encoded into a FRESH image rep at the next stage's dims via
``encode_image`` (the same path init images use; Limited Palette carries
its learned palette across — see ``PixelImage.copy_palette_from`` and the
palette-lock dance in ``workhorse.configure_pass``). The next stage then
runs with the previous stage's image as a weight-2 direct init hold plus
the normal prompts.

Frame numbering and ``.bak`` backups continue monotonically across every
stage boundary (each stage's global step offset starts where the previous
ended, and non-final-stage frames are saved upscaled to the full canvas so
the numbered sequence is uniform in size). ``auto_stop`` judges each stage
independently; ``phase_scheduling``'s t_hat spans each stage's own step
range (each stage is its own schedule pass — there is no cross-stage
schedule).

Thumbnail-stage sampling (``is_thumbnail_stage``): at any stage whose
canvas short side is at or below the largest perceptor input resolution,
random cutout sampling degenerates — every crop the samplers can draw
upsamples (nearly) the whole frame into the tower anyway, so the size
distribution only adds noise. The orchestrator forces the designed
full-frame sampler (``cutout_sampler=full``) for exactly those stages,
regardless of the configured sampler.

Pure functions only — no torch, no PIL — so the stage math is testable
without a render (tests/test_coarse_to_fine.py). The orchestration lives in
``workhorse.run_coarse_to_fine``.
"""

from collections.abc import Sequence

# Per-stage step-split percentages, keyed by coarse_stages: earlier stages
# are cheaper (composition needs fewer steps at thumbnail cost than detail
# does at full cost). All rows sum to 100; the last stage takes the
# integer-floor remainder so the split is always exact.
STAGE_STEP_PERCENTS: dict[int, tuple[int, ...]] = {
    2: (40, 60),
    3: (25, 25, 50),
    4: (15, 20, 25, 40),
}
# every stage must get at least this many steps (guard: steps_per_scene >=
# coarse_stages * MIN_STEPS_PER_STAGE — a run that can't give each stage a
# few steps must not silently degrade)
MIN_STEPS_PER_STAGE = 3
# non-final stage dims never drop below this (and never exceed the
# configured dim)
COARSE_DIM_FLOOR = 64


def validate_coarse_stages(*, coarse_to_fine: bool, coarse_stages: int) -> None:
    """
    The coarse_stages knob's own validity: in range, and never set to a
    non-default value on a run that ignores it — a silently inert setting
    is a lie in the config.
    """
    if coarse_stages not in STAGE_STEP_PERCENTS:
        raise ValueError(
            f"coarse_stages must be one of {sorted(STAGE_STEP_PERCENTS)}, "
            f"got {coarse_stages}"
        )
    if coarse_stages != 2 and not coarse_to_fine:
        raise ValueError(
            f"coarse_stages={coarse_stages} only has meaning with "
            "coarse_to_fine: true — set coarse_to_fine: true or remove "
            "coarse_stages."
        )


def stage_steps(steps_per_scene: int, coarse_stages: int) -> tuple[int, ...]:
    """
    Split ``steps_per_scene`` across the stages by the
    ``STAGE_STEP_PERCENTS`` table (integer floors, remainder to the final
    stage, so the split always sums exactly). Fails loud when the budget
    can't give every stage at least ``MIN_STEPS_PER_STAGE`` steps — a run
    that can't fit its stages must not silently degrade.
    """
    if coarse_stages not in STAGE_STEP_PERCENTS:
        raise ValueError(
            f"coarse_stages must be one of {sorted(STAGE_STEP_PERCENTS)}, "
            f"got {coarse_stages}"
        )
    minimum = coarse_stages * MIN_STEPS_PER_STAGE
    if steps_per_scene < minimum:
        raise ValueError(
            f"coarse_to_fine at coarse_stages={coarse_stages} needs "
            f"steps_per_scene >= {minimum} "
            f"({MIN_STEPS_PER_STAGE} per stage), got "
            f"steps_per_scene={steps_per_scene}"
        )
    percents = STAGE_STEP_PERCENTS[coarse_stages]
    splits = [steps_per_scene * p // 100 for p in percents[:-1]]
    splits.append(steps_per_scene - sum(splits))
    return tuple(splits)


def stage_dims(
    width: int, height: int, coarse_stages: int
) -> tuple[tuple[int, int], ...]:
    """
    The dims ladder for a configured (width, height): geometric, ending at
    the EXACT configured dims (stages=3 -> 1/4, 1/2, 1; stages=4 -> 1/8,
    1/4, 1/2, 1). Every non-final stage is rounded to the nearest multiple
    of 8 (half rounds up), floored at ``COARSE_DIM_FLOOR``, and capped at
    the configured dim (a canvas at or below the floor runs its early
    stages at full size rather than upscaling).
    """
    if coarse_stages not in STAGE_STEP_PERCENTS:
        raise ValueError(
            f"coarse_stages must be one of {sorted(STAGE_STEP_PERCENTS)}, "
            f"got {coarse_stages}"
        )
    if width < 1 or height < 1:
        raise ValueError(f"dims must be positive, got {width}x{height}")

    def scaled(d: int, div: int) -> int:
        # nearest multiple of 8 to d/div, half rounding up
        return min(d, max(COARSE_DIM_FLOOR, 8 * ((d + 4 * div) // (8 * div))))

    ladder = [
        (scaled(width, 2**e), scaled(height, 2**e))
        for e in range(coarse_stages - 1, 0, -1)
    ]
    ladder.append((width, height))
    return tuple(ladder)


def resume_stage(i_restore: int, splits: Sequence[int]) -> tuple[int, int]:
    """
    Map a restored global step onto the stage ladder: (stage number,
    1-based; steps already done within that stage). A step exactly on a
    stage boundary resumes as the EARLIER stage with nothing left to run,
    so the upscale transition replays deterministically from the restored
    state (the ``.bak`` at that slot was written by the earlier stage and
    holds its dims). Steps past the final boundary land in the final stage
    (its window then runs zero steps).
    """
    if i_restore < 0:
        raise ValueError(f"restore step must be >= 0, got {i_restore}")
    if len(splits) < 2 or any(s < 1 for s in splits):
        raise ValueError(
            f"splits must be >= 2 stages of >= 1 step each, got {tuple(splits)}"
        )
    boundary = 0
    for stage_number, width in enumerate(splits[:-1], start=1):
        boundary += width
        if i_restore <= boundary:
            return stage_number, i_restore - (boundary - width)
    return len(splits), i_restore - boundary


def is_thumbnail_stage(canvas: tuple[int, int], max_cut_size: int) -> bool:
    """
    True when a stage's canvas (pixels, the shape the cutout samplers see)
    is small enough that cutout sampling degenerates: with the short side
    at or below the largest perceptor input resolution, every crop any
    sampler can draw gets UPSAMPLED into the tower and (nearly) every crop
    covers the whole frame anyway — the size distribution adds noise, not
    views. The orchestrator forces ``cutout_sampler=full`` (the designed
    full-frame sampler) for exactly these stages.
    """
    if min(canvas) < 1:
        raise ValueError(f"canvas dims must be positive, got {canvas}")
    if max_cut_size < 1:
        raise ValueError(f"max_cut_size must be positive, got {max_cut_size}")
    return min(canvas) <= max_cut_size


def validate_coarse_to_fine(
    *,
    animation_mode: str,
    n_scenes: int,
    breath_mode: bool,
    semantic_init: bool,
    semantic_stabilization: bool,
) -> None:
    """
    Reject every config coarse_to_fine has no defined behavior for — LOUDLY,
    before any model loads. Each message names the setting to change.
    """
    if animation_mode != "off":
        raise ValueError(
            "coarse_to_fine is a staged STILL renderer; "
            f"animation_mode={animation_mode!r} re-anchors the image every "
            "frame and has no stage mapping. Set animation_mode: off or "
            "coarse_to_fine: false."
        )
    if n_scenes != 1:
        raise ValueError(
            f"coarse_to_fine splits ONE scene's steps_per_scene into "
            f"stages; {n_scenes} scenes ('||' separator) have no defined "
            "stage mapping (deferred). Use a single scene or "
            "coarse_to_fine: false."
        )
    if breath_mode:
        raise ValueError(
            "breath_mode crossfades saved frames from the init image over "
            "the whole render; under coarse_to_fine each later stage's init "
            "image is the previous stage's result, so the blend has no "
            "defined meaning. Set breath_mode: false or "
            "coarse_to_fine: false."
        )
    if semantic_init:
        raise ValueError(
            "semantic_init_weight holds the image toward the init image's "
            "CLIP embedding; under coarse_to_fine the stages disagree about "
            "which image that is (user init vs previous stage's result), so "
            "the combination is unsupported. Set semantic_init_weight: '' "
            "or coarse_to_fine: false."
        )
    if semantic_stabilization:
        raise ValueError(
            "semantic_stabilization_weight appends a per-pass stabilization "
            "prompt; under coarse_to_fine the stages would stack "
            "prompts targeting different images, so the combination is "
            "unsupported. Set semantic_stabilization_weight: '' or "
            "coarse_to_fine: false."
        )
