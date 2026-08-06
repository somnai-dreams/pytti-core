"""
Structure annealing (src/pytti/structure_annealing.py): cycle schedule +
decay law, band mask shape/softness, blend math, per-model application
(value-plane-only for Limited Palette), determinism, config validation,
guide integration on the torch path, and the mlx_full host-intervention
seam (state round-trip identity + torch-vs-MLX cycle equivalence).

Tiers: everything is CPU; the MLX seam tests skip when mlx is not
installed (darwin-only backend), using the stub towers from
tests/test_mlx_engine_step.py — no downloads.
"""

import importlib.util
import math
from typing import Any

import numpy as np
import pytest
import torch
from hydra import compose, initialize
from omegaconf import OmegaConf

from pytti.coarse_to_fine import stage_steps
from pytti.image_models import DifferentiableImage, PixelImage, RGBImage
from pytti.ImageGuide import DirectImageGuide
from pytti.structure_annealing import (
    ANNEAL_DECAY_FLOOR,
    ANNEAL_EDGE,
    MIN_STEPS_PER_CYCLE,
    PROTECTED_TAIL_FRACTION,
    anneal_image_rep,
    anneal_plane,
    anneal_schedule,
    band_mask,
    cycle_steps,
    cycle_strengths,
    injection_profile,
    validate_anneal_band,
    validate_anneal_cycles,
    validate_anneal_source,
    validate_anneal_strength,
    validate_structure_annealing,
)

needs_mlx = pytest.mark.skipif(
    importlib.util.find_spec("mlx") is None,
    reason="mlx not installed (darwin-only backend)",
)

CPU = torch.device("cpu")
SEED = 4242


# ---------------------------------------------------------------------------
# schedule derivation
# ---------------------------------------------------------------------------


class TestCycleSchedule:
    def test_default_run_lands_on_documented_steps(self):
        # 100 steps, 3 cycles: usable = round(65) -> 22, 43, 65
        assert cycle_steps(100, 3) == (22, 43, 65)

    @pytest.mark.parametrize(
        "steps,cycles", [(100, 3), (60, 3), (100, 1), (100, 12), (240, 5), (24, 3)]
    )
    def test_tail_is_protected_and_steps_are_sane(self, steps, cycles):
        schedule = cycle_steps(steps, cycles)
        usable = round(steps * (1.0 - PROTECTED_TAIL_FRACTION))
        assert len(schedule) == cycles
        assert len(set(schedule)) == cycles  # distinct
        assert list(schedule) == sorted(schedule)  # increasing
        assert schedule[0] > 0  # never step 0 — the init is already liquid
        assert schedule[-1] <= usable  # the last ~35% stays anneal-free
        # EVERY cycle gets its settle steps — the last cycle's come from
        # the protected tail (it sits exactly on the boundary)
        assert steps - schedule[-1] >= MIN_STEPS_PER_CYCLE

    def test_last_cycle_sits_exactly_at_the_tail_boundary(self):
        assert cycle_steps(100, 3)[-1] == round(100 * (1 - PROTECTED_TAIL_FRACTION))
        assert cycle_steps(60, 3)[-1] == round(60 * (1 - PROTECTED_TAIL_FRACTION))

    def test_c2f_final_stage_budget(self):
        # coarse_to_fine schedules within the FINAL stage's split: the
        # default 100-step / 2-stage ladder gives the final stage 60 steps
        final_budget = stage_steps(100, 2)[-1]
        assert final_budget == 60
        assert cycle_steps(final_budget, 3) == (13, 26, 39)

    def test_too_small_budget_fails_loud(self):
        with pytest.raises(ValueError, match="steps"):
            cycle_steps(20, 3)
        with pytest.raises(ValueError, match="anneal_cycles"):
            # 12 cycles need round(steps*0.65) >= 60
            cycle_steps(76, 12)

    def test_too_small_tail_fails_loud(self):
        # budgets whose inter-cycle spacing fits but whose protected tail
        # can't give the LAST cycle its settle steps: (8, 1) -> cycle at 5
        # with only 3 steps left; (12, 1) -> cycle at 8 with 4 left. The
        # final frame would keep most of the injected low band — shredding,
        # not annealing.
        for steps in (8, 10, 12):
            with pytest.raises(ValueError, match="settle"):
                cycle_steps(steps, 1)
        # the advertised single-cycle minimum leaves exactly enough tail
        minimum = math.ceil(MIN_STEPS_PER_CYCLE / PROTECTED_TAIL_FRACTION)
        schedule = cycle_steps(minimum, 1)
        assert minimum - schedule[-1] >= MIN_STEPS_PER_CYCLE

    def test_min_steps_per_cycle_boundary(self):
        # the smallest budget the message advertises for 3 cycles works
        minimum = max(
            math.ceil(3 * MIN_STEPS_PER_CYCLE / (1.0 - PROTECTED_TAIL_FRACTION)),
            math.ceil(MIN_STEPS_PER_CYCLE / PROTECTED_TAIL_FRACTION),
        )
        schedule = cycle_steps(minimum, 3)
        assert len(schedule) == 3

    def test_invalid_cycles_fail(self):
        for bad in (0, 13, -1, True):
            with pytest.raises(ValueError, match="anneal_cycles"):
                cycle_steps(100, bad)

    def test_invalid_budget_fails(self):
        with pytest.raises(ValueError, match="budget"):
            cycle_steps(0, 1)


class TestDecayLaw:
    def test_geometric_from_strength_to_floor(self):
        strengths = cycle_strengths(0.5, 3)
        assert strengths[0] == 0.5
        assert strengths[-1] == pytest.approx(0.5 * ANNEAL_DECAY_FLOOR)
        # log-linear: constant ratio between consecutive cycles
        ratios = [b / a for a, b in zip(strengths, strengths[1:], strict=False)]
        assert ratios[0] == pytest.approx(ratios[-1])

    def test_five_cycles_ratio_constant(self):
        strengths = cycle_strengths(1.0, 5)
        ratios = [b / a for a, b in zip(strengths, strengths[1:], strict=False)]
        for r in ratios[1:]:
            assert r == pytest.approx(ratios[0])
        assert strengths[-1] == pytest.approx(ANNEAL_DECAY_FLOOR)

    def test_single_cycle_uses_full_strength(self):
        assert cycle_strengths(0.7, 1) == (0.7,)

    def test_zero_strength_is_rejected(self):
        # a zero-strength cycle is annealing that does nothing — but NOT a
        # no-op: it still consumes torch RNG draws (reordering every later
        # draw at a fixed seed) and perturbs the plane by FFT round-trip
        # error. The inert-knob rule rejects it at the boundary.
        with pytest.raises(ValueError, match="anneal_strength"):
            cycle_strengths(0.0, 3)
        with pytest.raises(ValueError, match="anneal_strength"):
            validate_anneal_strength(0.0)

    def test_schedule_maps_steps_to_strengths(self):
        schedule = anneal_schedule(100, 3, 0.5)
        assert set(schedule) == set(cycle_steps(100, 3))
        assert tuple(schedule[s] for s in sorted(schedule)) == cycle_strengths(
            0.5, 3
        )


# ---------------------------------------------------------------------------
# band mask
# ---------------------------------------------------------------------------


class TestBandMask:
    def test_shape_and_range(self):
        mask = band_mask(64, 48, 0.15, CPU)
        assert mask.shape == (64, 48 // 2 + 1)
        assert float(mask.min()) >= 0.0
        assert float(mask.max()) <= 1.0

    def test_dc_is_always_preserved(self):
        # mean brightness is exposure, not structure
        assert float(band_mask(64, 64, 0.5, CPU)[0, 0]) == 0.0

    def test_low_band_is_one_high_band_is_zero(self):
        mask = band_mask(128, 128, 0.15, CPU)
        # bin (0, 1): r = (1/128)/0.5 ~ 0.016 < 0.075 = band*(1-EDGE)
        assert float(mask[0, 1]) == pytest.approx(1.0)
        # bin (0, 32): r = 0.5 > 0.225 = band*(1+EDGE)
        assert float(mask[0, 32]) == 0.0

    def test_soft_edge_no_hard_jump(self):
        # the raised-cosine edge must never step 1 -> 0 between adjacent
        # bins (hard masks ring). At 128px/band 0.15 the rolloff spans
        # bins 4.8..14.4 (radius 0.075..0.225 of Nyquist ~ 9.6 bins), so
        # the steepest adjacent-bin diff is the raised cosine's max slope
        # pi/2 over 9.6 bins ~ 0.164 — gate at 0.2, tight enough that a
        # near-cliff mask (single-bin transition) cannot pass
        mask = band_mask(128, 128, 0.15, CPU)
        row = mask[0, 1:]  # skip the forced-zero DC bin
        diffs = (row[1:] - row[:-1]).abs()
        assert float(diffs.max()) < 0.2

    def test_radially_monotone_along_axis(self):
        row = band_mask(128, 128, 0.15, CPU)[0, 1:]
        assert bool((row[1:] <= row[:-1] + 1e-6).all())

    def test_band_bounds_validated(self):
        with pytest.raises(ValueError, match="anneal_band"):
            band_mask(64, 64, 0.01, CPU)
        with pytest.raises(ValueError, match="anneal_band"):
            band_mask(64, 64, 0.51, CPU)
        with pytest.raises(ValueError, match="dims"):
            band_mask(0, 64, 0.15, CPU)


# ---------------------------------------------------------------------------
# blend math
# ---------------------------------------------------------------------------


def _sinusoid(freq_cycles: int, size: int = 64, amplitude: float = 0.2):
    """[1, size, size] plane: 0.5 + amplitude * cos(2 pi freq y / size)."""
    y = torch.arange(size, dtype=torch.float32)
    wave = 0.5 + amplitude * torch.cos(2 * math.pi * freq_cycles * y / size)
    return wave[None, :, None].expand(1, size, size).contiguous()


class TestBlendMath:
    # band 0.2 at 64px: fully-liquid below r = 0.1 (f <= 0.05 c/sample,
    # i.e. <= 3.2 cycles across 64px); untouched above r = 0.3

    def test_blur_full_strength_kills_the_low_band(self):
        plane = _sinusoid(2)  # r = 0.0625, fully inside the band
        out = anneal_plane(plane, strength=1.0, band=0.2, source="blur")
        assert float((out - 0.5).abs().max()) < 1e-4

    def test_blur_preserves_the_mean(self):
        plane = _sinusoid(2) + 0.1  # mean 0.6
        out = anneal_plane(plane, strength=1.0, band=0.2, source="blur")
        assert float(out.mean()) == pytest.approx(float(plane.mean()), abs=1e-5)

    def test_high_band_survives_untouched(self):
        plane = _sinusoid(16, amplitude=0.15)  # r = 0.5 > band*(1+EDGE)
        out = anneal_plane(plane, strength=1.0, band=0.2, source="blur")
        assert torch.allclose(out, plane, atol=1e-5)

    def test_mixed_bands_split_correctly(self):
        low, high = _sinusoid(2, amplitude=0.15), _sinusoid(16, amplitude=0.15)
        plane = low + high - 0.5
        out = anneal_plane(plane, strength=1.0, band=0.2, source="blur")
        assert torch.allclose(out, high, atol=1e-4)

    def test_strength_scales_the_blend_linearly(self):
        plane = _sinusoid(2)
        out = anneal_plane(plane, strength=0.5, band=0.2, source="blur")
        # amplitude halves: 0.5 + 0.1 cos
        residual = out - 0.5
        assert float(residual.abs().max()) == pytest.approx(0.1, abs=1e-4)

    def test_noise_source_replaces_band_and_keeps_mean(self):
        torch.manual_seed(SEED)
        plane = _sinusoid(2, amplitude=0.1)
        out = anneal_plane(
            plane, strength=1.0, band=0.2, source="noise", noise_falloff=1.0
        )
        assert float(out.mean()) == pytest.approx(float(plane.mean()), abs=0.02)
        assert not torch.allclose(out, plane, atol=1e-3)
        assert float(out.min()) >= 0.0 and float(out.max()) <= 1.0

    def test_noise_is_deterministic_under_a_seed(self):
        plane = _sinusoid(3)
        torch.manual_seed(SEED)
        first = anneal_plane(plane, strength=0.8, band=0.2, source="noise")
        torch.manual_seed(SEED)
        second = anneal_plane(plane, strength=0.8, band=0.2, source="noise")
        assert torch.equal(first, second)
        torch.manual_seed(SEED + 1)
        third = anneal_plane(plane, strength=0.8, band=0.2, source="noise")
        assert not torch.equal(first, third)

    def test_shape_validated(self):
        with pytest.raises(ValueError, match="channels"):
            anneal_plane(
                torch.zeros(8, 8), strength=0.5, band=0.2, source="blur"
            )

    def test_clamp_false_preserves_out_of_range_headroom(self):
        # RGBImage params legitimately sit outside [0, 1] mid-run (no
        # per-step clamp; decode clamps) — the anneal must not flatten
        # that headroom in regions the band never touched
        torch.manual_seed(SEED)
        plane = torch.rand(3, 64, 64)
        plane[:, 40:, 40:] = plane[:, 40:, 40:] * 0.2 + 1.2  # up to ~1.4
        plane[:, :8, :8] = -0.3
        out = anneal_plane(
            plane, strength=0.5, band=0.05, source="noise", clamp=False
        )
        assert float(out.max()) > 1.2
        assert float(out.min()) < -0.1

    def test_clamp_true_stays_in_unit_range(self):
        # the PixelImage value plane's domain IS [0, 1] (its per-step
        # update() clamps there) — clamp=True matches it
        torch.manual_seed(SEED)
        plane = torch.rand(1, 64, 64)
        out = anneal_plane(
            plane, strength=1.0, band=0.2, source="noise", clamp=True
        )
        assert float(out.min()) >= 0.0 and float(out.max()) <= 1.0

    def test_edge_constant_is_a_half_band(self):
        # documented rolloff span: band*(1-EDGE) .. band*(1+EDGE)
        assert ANNEAL_EDGE == 0.5


# ---------------------------------------------------------------------------
# per-model application
# ---------------------------------------------------------------------------


def _pixel_image(width=32, height=24):
    img = PixelImage(
        width=width,
        height=height,
        scale=1,
        palette_size=4,
        n_palettes=3,
        gamma=1,
        hdr_weight=0.01,
        norm_weight=0.2,
        device=CPU,
    ).to(CPU)
    return img


# Any-valued: **-unpacked into typed keyword params (pyright)
_ANNEAL_KWARGS: dict[str, Any] = dict(
    strength=0.8,
    band=0.15,
    source="noise",
    init_spectrum_falloff=1.0,
    init_spectrum_chroma="full",
)


class TestAnnealImageRep:
    def test_pixel_value_plane_only(self):
        torch.manual_seed(SEED)
        img = _pixel_image()
        img.encode_random()
        before = {k: v.clone() for k, v in img.state_dict().items()}
        anneal_image_rep(img, **_ANNEAL_KWARGS)
        after = img.state_dict()
        assert not torch.equal(after["value"], before["value"])
        # palette identity is the look; selection regions ARE mid/high
        # frequency structure — both untouched in v1
        for key in before:
            if key != "value":
                assert torch.equal(after[key], before[key]), key

    def test_parameter_identity_preserved(self):
        # in-place copy_ keeps the Parameter object, so optimizer state
        # (Adam moments) stays attached — the KEEP-moments decision
        torch.manual_seed(SEED)
        img = _pixel_image()
        img.encode_random()
        value_param = img.value
        anneal_image_rep(img, **_ANNEAL_KWARGS)
        assert img.value is value_param

    def test_rgb_all_channels(self):
        torch.manual_seed(SEED)
        img = RGBImage(32, 24, 1, device=CPU)
        img.encode_random()
        before = img.tensor.detach().clone()
        anneal_image_rep(img, **_ANNEAL_KWARGS)
        after = img.tensor.detach()
        for c in range(3):
            assert not torch.equal(after[0, c], before[0, c])

    def test_rgb_headroom_survives(self):
        # RGBImage has no per-step clamp (decode clamps), so mid-run Adam
        # legitimately parks saturated regions outside [0, 1] — an anneal
        # cycle must not flatten that headroom to the boundary
        torch.manual_seed(SEED)
        img = RGBImage(64, 64, 1, device=CPU)
        img.encode_random()
        with torch.no_grad():
            img.tensor[0, :, 40:, 40:] = 1.4
            img.tensor[0, :, :8, :8] = -0.3
        anneal_image_rep(img, **dict(_ANNEAL_KWARGS, band=0.05, strength=0.5))
        after = img.tensor.detach()
        assert float(after.max()) > 1.2
        assert float(after.min()) < -0.1

    def test_pixel_value_plane_stays_in_unit_range(self):
        # the value plane's own per-step update() clamps to [0, 1] — the
        # anneal keeps the same domain
        torch.manual_seed(SEED)
        img = _pixel_image()
        img.encode_random()
        anneal_image_rep(img, **_ANNEAL_KWARGS)
        value = img.value.detach()
        assert float(value.min()) >= 0.0 and float(value.max()) <= 1.0

    def test_unknown_model_fails_loud(self):
        with pytest.raises(ValueError, match="no re-liquify path"):
            anneal_image_rep(object(), **_ANNEAL_KWARGS)

    def test_injection_profile_decisions(self):
        # always pink; an explicit mono/natural chroma is honored, and the
        # back-compat default 'full' maps to mono — repeated per-cycle
        # injection of independent per-channel fields is the measured
        # chroma-leak pathology, so no full-chroma injection exists in v1
        assert injection_profile(1.0, "full") == (1.0, "mono")
        assert injection_profile(2.0, "natural") == (2.0, "natural")
        assert injection_profile(1.5, "mono") == (1.5, "mono")
        with pytest.raises(ValueError, match="chroma"):
            injection_profile(1.0, "rainbow")


# ---------------------------------------------------------------------------
# config validation
# ---------------------------------------------------------------------------


def _valid_config(**overrides) -> dict[str, Any]:
    cfg: dict[str, Any] = dict(
        structure_annealing=True,
        anneal_cycles=3,
        anneal_strength=0.5,
        anneal_band=0.15,
        anneal_source="noise",
        image_model="Limited Palette",
        animation_mode="off",
        auto_stop=False,
        optimizer="adam",
        steps_budget=100,
        interpolation_steps=0,
        n_scenes=1,
    )
    cfg.update(overrides)
    return cfg


class TestConfigValidation:
    def test_every_bound(self):
        for ok in (1, 12):
            validate_anneal_cycles(ok)
        for bad in (0, 13):
            with pytest.raises(ValueError, match="anneal_cycles"):
                validate_anneal_cycles(bad)
        for ok in (0.01, 1.0, 0.5):
            validate_anneal_strength(ok)
        for bad in (0.0, -0.01, 1.01):
            with pytest.raises(ValueError, match="anneal_strength"):
                validate_anneal_strength(bad)
        for ok in (0.02, 0.5, 0.15):
            validate_anneal_band(ok)
        for bad in (0.019, 0.51):
            with pytest.raises(ValueError, match="anneal_band"):
                validate_anneal_band(bad)
        for ok in ("noise", "blur"):
            validate_anneal_source(ok)
        with pytest.raises(ValueError, match="anneal_source"):
            validate_anneal_source("gauss")

    def test_inert_knob_rule(self):
        # a non-default anneal_* value on a run with structure_annealing
        # off is a config lie (the coarse_stages rule)
        validate_anneal_cycles(3, structure_annealing=False)
        with pytest.raises(ValueError, match="structure_annealing"):
            validate_anneal_cycles(5, structure_annealing=False)
        with pytest.raises(ValueError, match="structure_annealing"):
            validate_anneal_strength(0.7, structure_annealing=False)
        with pytest.raises(ValueError, match="structure_annealing"):
            validate_anneal_band(0.2, structure_annealing=False)
        with pytest.raises(ValueError, match="structure_annealing"):
            validate_anneal_source("blur", structure_annealing=False)

    def test_valid_config_passes(self):
        validate_structure_annealing(**_valid_config())
        validate_structure_annealing(
            **_valid_config(structure_annealing=False)
        )

    @pytest.mark.parametrize("model", ["VQGAN", "LlamaGen"])
    def test_latent_models_fail_loud(self, model):
        with pytest.raises(ValueError, match="latent"):
            validate_structure_annealing(**_valid_config(image_model=model))

    def test_animation_rejected(self):
        with pytest.raises(ValueError, match="animation_mode"):
            validate_structure_annealing(**_valid_config(animation_mode="2D"))

    def test_auto_stop_rejected(self):
        with pytest.raises(ValueError, match="auto_stop"):
            validate_structure_annealing(**_valid_config(auto_stop=True))

    def test_adamw_sf_rejected(self):
        with pytest.raises(ValueError, match="optimizer"):
            validate_structure_annealing(**_valid_config(optimizer="adamw_sf"))

    def test_budget_too_small_rejected(self):
        with pytest.raises(ValueError, match="steps"):
            validate_structure_annealing(**_valid_config(steps_budget=20))

    def test_multi_scene_crossfade_overlap_rejected(self):
        # 100 steps / 3 cycles -> first cycle at 22; a 30-step crossfade
        # would spend its recomposition on the OUTGOING scene's prompts
        with pytest.raises(ValueError, match="crossfade"):
            validate_structure_annealing(
                **_valid_config(n_scenes=2, interpolation_steps=30)
            )
        # a ramp ending at or before the first cycle is fine
        validate_structure_annealing(
            **_valid_config(n_scenes=2, interpolation_steps=22)
        )
        # single-scene runs self-crossfade — no wrong scene to protect
        validate_structure_annealing(
            **_valid_config(n_scenes=1, interpolation_steps=30)
        )

    def test_scene_and_interp_bounds(self):
        with pytest.raises(ValueError, match="n_scenes"):
            validate_structure_annealing(**_valid_config(n_scenes=0))
        with pytest.raises(ValueError, match="interpolation_steps"):
            validate_structure_annealing(**_valid_config(interpolation_steps=-1))

    def test_schema_wiring_rejects_bad_source(self):
        with initialize(config_path="config", version_base=None):
            cfg = compose(
                config_name="_structured_config",
                overrides=[
                    "scenes=x",
                    "structure_annealing=true",
                    "anneal_source=gauss",
                ],
            )
        with pytest.raises(ValueError, match="anneal_source"):
            OmegaConf.to_object(cfg)

    def test_schema_wiring_rejects_inert_knob(self):
        with initialize(config_path="config", version_base=None):
            cfg = compose(
                config_name="_structured_config",
                overrides=["scenes=x", "anneal_cycles=5"],
            )
        with pytest.raises(ValueError, match="structure_annealing"):
            OmegaConf.to_object(cfg)

    def test_schema_wiring_accepts_valid_annealing(self):
        with initialize(config_path="config", version_base=None):
            cfg = compose(
                config_name="_structured_config",
                overrides=[
                    "scenes=x",
                    "structure_annealing=true",
                    "anneal_cycles=4",
                    "anneal_strength=0.8",
                ],
            )
        obj = OmegaConf.to_object(cfg)
        assert obj.structure_annealing is True
        assert obj.anneal_cycles == 4
        assert obj.anneal_strength == 0.8


# ---------------------------------------------------------------------------
# guide integration (torch path)
# ---------------------------------------------------------------------------


def _guide_params(**overrides):
    cfg = dict(
        animation_mode="off",
        optimizer="adam",
        perceptor_backend="torch",
        structure_annealing=True,
        anneal_cycles=1,
        anneal_strength=1.0,
        anneal_band=0.15,
        anneal_source="noise",
        init_spectrum="white",
        init_spectrum_falloff=1.0,
        init_spectrum_chroma="full",
        steps_per_scene=16,
        steps_per_frame=1000,
        frames_per_second=12,
        pre_animation_steps=1000,
        save_every=1000,
        display_every=0,
        reset_lr_each_frame=False,
        auto_stop=False,
        phase_scheduling=False,
        coherence_weighting=False,
        input_audio="",
        input_audio_filters=None,
        approximate_vram_usage=False,
    )
    cfg.update(overrides)
    return OmegaConf.create(cfg)


def _rgb_guide(init_augs=None, **param_overrides):
    img = RGBImage(24, 16, 1, device=CPU)
    img.encode_random()
    guide = DirectImageGuide(
        image_rep=img,
        embedder=None,
        params=_guide_params(**param_overrides),
        init_augs=init_augs,
    )
    return guide, img


class TestGuideIntegration:
    def test_schedule_matches_the_pure_function(self):
        guide, _ = _rgb_guide(steps_per_scene=100, anneal_cycles=3)
        assert guide.anneal_schedule == anneal_schedule(100, 3, 1.0)

    def test_cycle_fires_during_run_steps(self):
        torch.manual_seed(SEED)
        guide, img = _rgb_guide()  # 16 steps, 1 cycle at round(10.4) = 10
        assert guide.anneal_schedule == {10: 1.0}
        before = img.tensor.detach().clone()
        guide.run_steps(16, [], [], [])
        # no losses, no grads: only the anneal cycle can change the image
        assert not torch.equal(img.tensor.detach(), before)

    def test_annealing_off_is_a_no_op(self):
        torch.manual_seed(SEED)
        guide, img = _rgb_guide(
            structure_annealing=False,
            anneal_cycles=3,
            anneal_strength=0.5,
        )
        assert guide.anneal_schedule is None
        before = img.tensor.detach().clone()
        guide.run_steps(16, [], [], [])
        assert torch.equal(img.tensor.detach(), before)

    def test_fixed_seed_reproduces_the_annealed_run(self):
        def run():
            torch.manual_seed(SEED)
            guide, img = _rgb_guide()
            guide.run_steps(16, [], [], [])
            return img.tensor.detach().clone()

        assert torch.equal(run(), run())

    def test_direct_init_hold_released_at_first_cycle(self):
        # a direct init hold is a full-band pull toward the PRE-anneal
        # image — left enabled it would drag the re-liquified band straight
        # back (coarse_to_fine holds the previous stage at weight 2 in
        # exactly the stage the cycles run in), so the guide releases it
        # when the first cycle fires
        from pytti.LossAug.BaseLossClass import Loss

        torch.manual_seed(SEED)
        hold = Loss(weight="2", stop=-math.inf, name="init hold", device=CPU)
        guide, _img = _rgb_guide(init_augs=[hold])
        assert hold.enabled
        guide.run_steps(16, [], [], [])
        assert not hold.enabled

    def test_restore_past_first_cycle_is_hold_free(self):
        # the release is position-based: a restore that resumes past the
        # first cycle's scene step replays NO cycle, but must still run
        # hold-free exactly like the uninterrupted run did
        from pytti.LossAug.BaseLossClass import Loss

        torch.manual_seed(SEED)
        hold = Loss(weight="2", stop=-math.inf, name="init hold", device=CPU)
        guide, _img = _rgb_guide(init_augs=[hold])  # cycle at 10
        guide.run_steps(4, [], [], [], skipped_steps=12)
        assert not hold.enabled

    def test_init_hold_untouched_without_annealing(self):
        from pytti.LossAug.BaseLossClass import Loss

        torch.manual_seed(SEED)
        hold = Loss(weight="2", stop=-math.inf, name="init hold", device=CPU)
        guide, _img = _rgb_guide(init_augs=[hold], structure_annealing=False)
        guide.run_steps(16, [], [], [])
        assert hold.enabled

    def test_cycle_inside_crossfade_rejected_in_run_steps(self):
        # backstop for directly-constructed guides (workhorse rejects the
        # overlap at config time): a first cycle inside a real crossfade —
        # DIFFERENT outgoing prompts — must fail loud, not recompose
        # toward the fading scene
        guide, _img = _rgb_guide()  # first cycle at 10
        with pytest.raises(ValueError, match="crossfade"):
            guide.run_steps(16, [], [], [], interp_steps=12)

    def test_cycle_at_crossfade_end_and_self_crossfade_run(self):
        torch.manual_seed(SEED)
        guide, img = _rgb_guide()
        before = img.tensor.detach().clone()
        # ramp ends exactly at the cycle step: fine
        guide.run_steps(16, [], [], [], interp_steps=10)
        assert not torch.equal(img.tensor.detach(), before)
        # scene 1 / coarse_to_fine pass the SAME list for both prompt
        # args (self-crossfade) — no wrong scene to protect
        torch.manual_seed(SEED)
        guide2, _ = _rgb_guide()
        prompts: list = []
        guide2.run_steps(16, prompts, prompts, [], interp_steps=12)

    def test_animation_rejected_at_construction(self):
        with pytest.raises(ValueError, match="animation_mode"):
            _rgb_guide(animation_mode="2D")

    def test_auto_stop_rejected_at_construction(self):
        with pytest.raises(ValueError, match="auto_stop"):
            _rgb_guide(auto_stop=True)

    def test_adamw_sf_rejected_at_construction(self):
        with pytest.raises(ValueError, match="optimizer"):
            _rgb_guide(optimizer="adamw_sf")

    def test_latent_model_rejected_at_construction(self):
        class TokenImage(DifferentiableImage):
            def __init__(self):
                super().__init__(8, 8)

        with pytest.raises(ValueError, match="no re-liquify path"):
            DirectImageGuide(
                image_rep=TokenImage(), embedder=None, params=_guide_params()
            )

    def test_too_small_scene_fails_at_construction(self):
        with pytest.raises(ValueError, match="steps"):
            _rgb_guide(steps_per_scene=5, anneal_cycles=3)
        # spacing fits but the protected tail can't settle the last cycle
        with pytest.raises(ValueError, match="settle"):
            _rgb_guide(steps_per_scene=8, anneal_cycles=1)


# ---------------------------------------------------------------------------
# the mlx_full host-intervention seam
# ---------------------------------------------------------------------------


@needs_mlx
class TestMlxSeam:
    def _fixture(self):
        # stub towers / embedder from the engine test module (no downloads)
        from tests.test_mlx_engine_step import _make_engine, _seeded_pixel_image

        img = _seeded_pixel_image()
        engine = _make_engine(img=img)
        return engine, img

    def _full_state(self, engine):
        """EVERYTHING the seam promises not to disturb: the params tree,
        the Adam moments + step count, and mx's global RNG state."""
        import mlx.core as mx
        from mlx.utils import tree_flatten

        return (
            {k: np.array(v) for k, v in engine._params.items()},
            {k: np.array(v) for k, v in tree_flatten(engine._opt.state)},
            np.array(mx.random.state[0]),
        )

    @staticmethod
    def _assert_state_equal(before, after):
        params_b, opt_b, rng_b = before
        params_a, opt_a, rng_a = after
        assert set(params_b) == set(params_a)
        for key in params_b:
            assert np.array_equal(params_b[key], params_a[key]), key
        assert set(opt_b) == set(opt_a)
        for key in opt_b:
            assert np.array_equal(opt_b[key], opt_a[key]), f"opt.{key}"
        assert np.array_equal(rng_b, rng_a), "mx.random.state"

    def test_round_trip_identity_when_annealing_is_off(self):
        # bit-identity of the WHOLE seam surface, not just the params tree:
        # a future import_params that rebuilt the opt state, consumed mx
        # RNG, or advanced torch RNG would silently break mlx_full anneal
        # reproducibility while a params-only check kept passing
        engine, img = self._fixture()
        torch_rng_before = torch.get_rng_state().clone()
        before = self._full_state(engine)
        engine.write_back(img)
        engine.import_params(img)
        self._assert_state_equal(before, self._full_state(engine))
        assert torch.equal(torch_rng_before, torch.get_rng_state())

    def test_torch_vs_mlx_cycle_equivalence(self):
        from pytti.mlx_engine.image_models import pixel_state_dict_from_params
        from tests.test_mlx_engine_step import _pixel_image

        engine, img = self._fixture()
        twin = _pixel_image()
        twin.load_state_dict(
            {k: v.clone() for k, v in img.state_dict().items()}
        )
        before = pixel_state_dict_from_params(engine._params)

        # mlx path: export -> anneal in torch -> import (what
        # DirectImageGuide._apply_structure_anneal does between steps)
        torch.manual_seed(123)
        engine.write_back(img)
        anneal_image_rep(img, **_ANNEAL_KWARGS)
        engine.import_params(img)

        # torch path: the same operation on the same starting state
        torch.manual_seed(123)
        anneal_image_rep(twin, **_ANNEAL_KWARGS)

        exported = pixel_state_dict_from_params(engine._params)
        reference = twin.state_dict()
        assert set(exported) == set(reference)
        for key in exported:
            # BIT equality: the seam is numpy copies both ways and the
            # anneal runs torch-side on identical fp32 values — anything
            # short of torch.equal would hide a lossy conversion
            assert torch.equal(exported[key], reference[key]), key
        # the anneal touched the value plane and nothing else
        assert not torch.equal(exported["value"], before["value"])
        for key in exported:
            if key != "value":
                assert torch.equal(exported[key], before[key]), key

    def test_intervention_diverges_exactly_at_the_cycle_step(self):
        # the strong form of "the step still runs": a k=4 host
        # intervention leaves steps 0..3 BIT-identical to the no-anneal
        # run (params AND the mx RNG stream stay in lockstep, because the
        # anneal consumes only torch RNG) and diverges exactly at step 4.
        # Pinned to the mx CPU device: Metal float reductions are not
        # run-to-run bit-deterministic, and this test's whole point is
        # bit-level lockstep
        import mlx.core as mx

        from tests.test_mlx_engine_step import _prompt, _tv_loss

        def run(anneal_at):
            mx.random.seed(SEED)
            torch.manual_seed(SEED)
            engine, img = self._fixture()
            frames = []
            for i in range(6):
                if i == anneal_at:
                    engine.write_back(img)
                    anneal_image_rep(img, **_ANNEAL_KWARGS)
                    engine.import_params(img)
                record = engine.train_step(
                    i, [_prompt("a test prompt:1")], [], [_tv_loss(0.1)]
                )
                assert math.isfinite(float(record["TOTAL"]))
                frames.append(
                    {k: np.array(v) for k, v in engine._params.items()}
                )
            return frames

        device_before = mx.default_device()
        mx.set_default_device(mx.cpu)
        try:
            base = run(anneal_at=None)
            intervened = run(anneal_at=4)
        finally:
            mx.set_default_device(device_before)
        for i in range(4):
            for key in base[i]:
                assert np.array_equal(
                    base[i][key], intervened[i][key]
                ), (i, key)
        assert any(
            not np.array_equal(base[4][key], intervened[4][key])
            for key in base[4]
        )

    def test_import_params_rejects_foreign_image(self):
        from tests.test_mlx_engine_step import _pixel_image

        engine, _img = self._fixture()
        with pytest.raises(ValueError, match="different image_rep"):
            engine.import_params(_pixel_image())
