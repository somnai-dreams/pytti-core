"""
Phase scheduling — quality-phase behavior over normalized scene time.

One config knob: ``phase_scheduling: bool`` (structured_config, default
off; still mode only — DirectImageGuide fails loud under any
animation_mode). When on, three fixed schedules run over

    t_hat = step / steps_per_scene

the scene's normalized time (0 = scene start, approaching 1 at the scene
cap). ``run_steps`` passes scene-local steps and workhorse runs one scene
per ``run_steps`` call, so every scene — including a restored one — moves
through its own t_hat.

Schedule table (the constants below are the single source of truth):

    schedule           | applies to                        | t_hat: 0    1/3    2/3     1
    -------------------|-----------------------------------|---------------------------------
    TV/smoothing scale | the TVLoss from `smoothing_weight`| 2.0    1.5    1.0     0.5
    (structure early,  | (linear ramp)                     |
    detail late)       |                                   |
    init-hold scale    | the `direct_init_weight` loss,    | 1.0    0.833  0.667   0.5
    (anchored start,   | when an init image is present     |
    freer finish)      | (linear ramp)                     |
    palette lock       | Limited Palette `palette` updates | open   open   open    locked
    (crystallization)  | (locked iff t_hat > 2/3)          |

The scales multiply the USER'S CONFIGURED WEIGHT after parametric
evaluation (``Loss.weight_scale``): parametric expressions keep working —
their evaluated value is wrapped, the expression string is never rewritten.

All three schedules are per-step host scalars. Under
``perceptor_backend=mlx_full`` they enter the compiled step as arguments
(the direct-loss weight vector and the 0/1 palette gate), never as new
constants, so a phase change can never force a retrace
(docs/mlx-m2-seam-map.md retrace-key discipline). The torch and mlx-bridge
paths consume the same values in ``Loss.forward`` /
``PixelImage.lock_palette``.

Pure functions only — no torch, no mlx — so the whole table is testable
without a render (tests/test_phase_scheduling.py).
"""

# smoothing/TV weight: structure early (2x), detail late (0.5x)
TV_SCALE_START = 2.0
TV_SCALE_END = 0.5
# direct init-hold weight: anchored start (1x), freer finish (0.5x)
INIT_SCALE_START = 1.0
INIT_SCALE_END = 0.5
# Limited Palette: palette locked for the final third of the scene
PALETTE_LOCK_T_HAT = 2.0 / 3.0


def scene_t_hat(step: int, steps_per_scene: int) -> float:
    """
    Normalized scene time for a scene-local ``step``.

    ``step`` is the scene-local step index (what run_steps passes to
    train(): ``i + skipped_steps``), so it must lie in
    ``[0, steps_per_scene]`` — anything else means the caller handed a
    global step or a mis-sized scene, which must not be silently clamped.
    """
    if steps_per_scene <= 0:
        raise ValueError(
            f"steps_per_scene must be positive, got {steps_per_scene}"
        )
    if not 0 <= step <= steps_per_scene:
        raise ValueError(
            f"scene-local step {step} outside [0, {steps_per_scene}] — "
            "phase schedules take scene-local steps, not global ones"
        )
    return step / steps_per_scene


def _check_t_hat(t_hat: float) -> None:
    if not 0.0 <= t_hat <= 1.0:
        raise ValueError(f"t_hat must be in [0, 1], got {t_hat!r}")


def tv_weight_scale(t_hat: float) -> float:
    """Multiplier for the smoothing/TV weight: 2.0 -> 0.5, linear."""
    _check_t_hat(t_hat)
    return TV_SCALE_START + (TV_SCALE_END - TV_SCALE_START) * t_hat


def init_weight_scale(t_hat: float) -> float:
    """Multiplier for the direct init-hold weight: 1.0 -> 0.5, linear."""
    _check_t_hat(t_hat)
    return INIT_SCALE_START + (INIT_SCALE_END - INIT_SCALE_START) * t_hat


def palette_locked(t_hat: float) -> bool:
    """Whether the Limited Palette palette is locked (final third)."""
    _check_t_hat(t_hat)
    return t_hat > PALETTE_LOCK_T_HAT
