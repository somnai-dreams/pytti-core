"""
Coarse-to-fine still rendering — the pure math and validation.

One config knob: ``coarse_to_fine: bool`` (structured_config, default off;
still mode only). When on, a still render runs in two stages:

- **stage 1** renders at half the configured dims (each dim halved, rounded
  to the nearest multiple of 8, floored at 64) for the first 40% of
  ``steps_per_scene`` — composition forms fast at quarter cost;
- **transition**: the stage-1 image is decoded to PIL, bicubic-upscaled to
  the full canvas, and re-encoded into a FRESH image rep at full dims via
  ``encode_image`` (the same path init images use; Limited Palette carries
  its learned palette across — see ``PixelImage.copy_palette_from`` and the
  palette-lock dance in ``workhorse.configure_pass``);
- **stage 2** runs the remaining 60% at full dims with the stage-1 image as
  a weight-2 direct init hold plus the normal prompts.

Frame numbering and ``.bak`` backups continue monotonically across the
stage boundary (stage 2's global step offset starts where stage 1 ended,
and stage-1 frames are saved upscaled to the full canvas so the numbered
sequence is uniform in size). ``auto_stop`` judges each stage
independently; ``phase_scheduling``'s t_hat spans each stage's own step
range (each stage is its own schedule pass — there is no cross-stage
schedule).

Pure functions only — no torch, no PIL — so the stage math is testable
without a render (tests/test_coarse_to_fine.py). The orchestration lives in
``workhorse.run_coarse_to_fine``.
"""

# stage 1 gets 40% of steps_per_scene (exact: steps * 2 // 5)
COARSE_STEP_NUMERATOR = 2
COARSE_STEP_DENOMINATOR = 5
# stage-1 dims never drop below this (and never exceed the configured dim)
COARSE_DIM_FLOOR = 64


def stage_steps(steps_per_scene: int) -> tuple[int, int]:
    """
    Split ``steps_per_scene`` into (stage-1 steps, stage-2 steps): stage 1
    gets 40% (floor), stage 2 the rest. Fails loud when either stage would
    be empty — a run that can't fit both stages must not silently degrade
    into a single-stage render.
    """
    stage1 = (steps_per_scene * COARSE_STEP_NUMERATOR) // COARSE_STEP_DENOMINATOR
    stage2 = steps_per_scene - stage1
    if stage1 < 1 or stage2 < 1:
        raise ValueError(
            "coarse_to_fine needs steps_per_scene >= 3 so both stages get "
            f"at least one step, got steps_per_scene={steps_per_scene}"
        )
    return stage1, stage2


def coarse_dims(width: int, height: int) -> tuple[int, int]:
    """
    Stage-1 dims for a configured (width, height): each dim halved, rounded
    to the nearest multiple of 8 (half rounds up), floored at
    ``COARSE_DIM_FLOOR``, and capped at the configured dim (a canvas at or
    below the floor runs stage 1 at full size rather than upscaling).
    """

    def half(d: int) -> int:
        if d < 1:
            raise ValueError(f"dims must be positive, got {d}")
        return min(d, max(COARSE_DIM_FLOOR, 8 * ((d + 8) // 16)))

    return half(width), half(height)


def resume_stage(i_restore: int, stage1_steps: int) -> tuple[int, int]:
    """
    Map a restored global step to (stage, steps already done within that
    stage). ``i_restore == stage1_steps`` resumes at the boundary as stage 1
    with nothing left to run, so the upscale transition replays
    deterministically from the restored coarse state (the ``.bak`` at that
    slot was written by stage 1 and holds coarse dims).
    """
    if i_restore < 0:
        raise ValueError(f"restore step must be >= 0, got {i_restore}")
    if stage1_steps < 1:
        raise ValueError(f"stage1_steps must be >= 1, got {stage1_steps}")
    if i_restore <= stage1_steps:
        return 1, i_restore
    return 2, i_restore - stage1_steps


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
            "coarse_to_fine is a two-stage STILL renderer; "
            f"animation_mode={animation_mode!r} re-anchors the image every "
            "frame and has no stage mapping. Set animation_mode: off or "
            "coarse_to_fine: false."
        )
    if n_scenes != 1:
        raise ValueError(
            f"coarse_to_fine splits ONE scene's steps_per_scene into two "
            f"stages; {n_scenes} scenes ('||' separator) have no defined "
            "stage mapping (deferred). Use a single scene or "
            "coarse_to_fine: false."
        )
    if breath_mode:
        raise ValueError(
            "breath_mode crossfades saved frames from the init image over "
            "the whole render; under coarse_to_fine stage 2's init image is "
            "the stage-1 result, so the blend has no defined meaning. Set "
            "breath_mode: false or coarse_to_fine: false."
        )
    if semantic_init:
        raise ValueError(
            "semantic_init_weight holds the image toward the init image's "
            "CLIP embedding; under coarse_to_fine the stages disagree about "
            "which image that is (user init vs stage-1 result), so the "
            "combination is unsupported. Set semantic_init_weight: '' or "
            "coarse_to_fine: false."
        )
    if semantic_stabilization:
        raise ValueError(
            "semantic_stabilization_weight appends a per-pass stabilization "
            "prompt; under coarse_to_fine the two stages would stack "
            "prompts targeting different images, so the combination is "
            "unsupported. Set semantic_stabilization_weight: '' or "
            "coarse_to_fine: false."
        )
