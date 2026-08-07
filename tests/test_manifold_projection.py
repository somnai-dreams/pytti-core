"""
Manifold projection (src/pytti/manifold_projection.py): cycle schedule +
protected tail reuse, config validation (stride-vs-dims + every fail-loud
combo + inert-knob rejections), projection-op correctness (exact blend
math against a stub VQ tokenizer), the PixelImage light re-encode
(palette identity preserved), determinism (no RNG consumed), guide
integration on the torch path, and the mlx_full host-intervention seam
(bit-identity of the untouched state surfaces + torch-vs-MLX cycle
equivalence).

Tiers: everything is CPU and uses a stub VQ model (no downloads); the MLX
seam tests skip when mlx is not installed (darwin-only backend), using the
stub towers from tests/test_mlx_engine_step.py.
"""

import importlib.util
import math
from typing import Any

import numpy as np
import pytest
import torch
from hydra import compose, initialize
from omegaconf import OmegaConf
from torch.nn import functional as F

from pytti.coarse_to_fine import stage_steps
from pytti.image_models import (
    DifferentiableImage,
    FourierImage,
    PixelImage,
    RGBImage,
)
from pytti.ImageGuide import DirectImageGuide
from pytti.manifold_projection import (
    PROJECTION_EVERY_RANGE,
    hsp_value,
    pixel_canvas,
    project_image_rep,
    project_pixels,
    projection_steps,
    validate_manifold_projection,
    validate_projection_dims,
    validate_projection_every,
    validate_projection_model,
    validate_projection_strength,
    vq_project,
)
from pytti.structure_annealing import PROTECTED_TAIL_FRACTION

needs_mlx = pytest.mark.skipif(
    importlib.util.find_spec("mlx") is None,
    reason="mlx not installed (darwin-only backend)",
)

CPU = torch.device("cpu")
SEED = 4242


class _StubDecoder:
    # vq_project reads the conv-ladder depth off the decoder, exactly like
    # LlamaGenImage does: stride = 2 ** (num_resolutions - 1) = 4
    num_resolutions = 3


class _StubVQ:
    """
    A deterministic stand-in with the vendored VQModel's encode/decode
    surface: encode = 4x average pool ("quantize" to block means), decode =
    nearest 4x upsample. A genuine non-identity projection (kills
    high-frequency content) with zero RNG and zero downloads.
    """

    decoder = _StubDecoder()

    def encode(self, x):
        return F.avg_pool2d(x, 4), None, None

    def decode(self, quant):
        return F.interpolate(quant, scale_factor=4, mode="nearest")


class _IdentityDecoder:
    num_resolutions = 1  # stride 1: any dims pass


class _IdentityVQ:
    """VQ(x) = x — models a canvas already ON the decoder manifold. A true
    projection must be (near-)idempotent there: repeated cycles at any
    strength may not walk the image."""

    decoder = _IdentityDecoder()

    def encode(self, x):
        return x, None, None

    def decode(self, quant):
        return quant


# ---------------------------------------------------------------------------
# schedule derivation
# ---------------------------------------------------------------------------


class TestProjectionSchedule:
    def test_documented_smoke_schedule(self):
        # 60 steps, every 15: usable = round(39) -> 15, 30 (45 is in the tail)
        assert projection_steps(60, 15) == (15, 30)

    def test_default_run_schedule(self):
        # 100 steps, every 30: usable = 65 -> 30, 60 (90 is in the tail)
        assert projection_steps(100, 30) == (30, 60)

    @pytest.mark.parametrize(
        "steps,every", [(100, 30), (60, 15), (100, 5), (320, 200), (47, 30)]
    )
    def test_tail_is_protected_and_steps_are_sane(self, steps, every):
        schedule = projection_steps(steps, every)
        usable = round(steps * (1.0 - PROTECTED_TAIL_FRACTION))
        assert schedule  # at least one projection or fail loud
        assert schedule[0] == every  # never step 0
        assert schedule[-1] <= usable  # the last ~35% stays projection-free
        assert all(
            b - a == every for a, b in zip(schedule, schedule[1:], strict=False)
        )

    def test_reuses_the_annealing_tail_constant(self):
        # the SAME protected tail as structure_annealing, by design: the
        # final stretch settles character on both interventions
        assert PROTECTED_TAIL_FRACTION == 0.35
        assert projection_steps(100, 30)[-1] <= round(
            100 * (1 - PROTECTED_TAIL_FRACTION)
        )

    def test_c2f_final_stage_budget(self):
        # coarse_to_fine schedules within the FINAL stage's split: the
        # default 100-step / 2-stage ladder gives the final stage 60 steps
        final_budget = stage_steps(100, 2)[-1]
        assert final_budget == 60
        assert projection_steps(final_budget, 15) == (15, 30)

    def test_too_small_budget_fails_loud(self):
        # usable = round(30 * 0.65) = 20 < 30: not even one projection
        with pytest.raises(ValueError, match="manifold_projection"):
            projection_steps(30, 30)
        # the advertised minimum fits exactly one
        minimum = math.ceil(30 / (1.0 - PROTECTED_TAIL_FRACTION))
        assert projection_steps(minimum, 30) == (30,)

    def test_invalid_every_fails(self):
        lo, hi = PROJECTION_EVERY_RANGE
        for bad in (lo - 1, hi + 1, 0, -5, True):
            with pytest.raises(ValueError, match="projection_every"):
                projection_steps(100, bad)

    def test_invalid_budget_fails(self):
        with pytest.raises(ValueError, match="budget"):
            projection_steps(0, 30)


# ---------------------------------------------------------------------------
# config validation
# ---------------------------------------------------------------------------


def _valid_config(**overrides) -> dict[str, Any]:
    cfg: dict[str, Any] = dict(
        manifold_projection=True,
        projection_every=30,
        projection_strength=0.5,
        projection_model="ds8",
        image_model="Limited Palette",
        animation_mode="off",
        structure_annealing=False,
        auto_stop=False,
        optimizer="adam",
        steps_budget=100,
        width=256,
        height=256,
    )
    cfg.update(overrides)
    return cfg


class TestConfigValidation:
    def test_every_bound(self):
        for ok in PROJECTION_EVERY_RANGE:
            validate_projection_every(ok)
        for bad in (4, 201, True):
            with pytest.raises(ValueError, match="projection_every"):
                validate_projection_every(bad)
        for ok in (0.01, 0.5, 1.0):
            validate_projection_strength(ok)
        for bad in (0.0, -0.1, 1.01):
            with pytest.raises(ValueError, match="projection_strength"):
                validate_projection_strength(bad)
        for ok in ("ds16", "ds8"):
            validate_projection_model(ok)
        with pytest.raises(ValueError, match="projection_model"):
            validate_projection_model("ds4")

    def test_inert_knob_rule(self):
        # a non-default projection_* value on a run with
        # manifold_projection off is a config lie (the coarse_stages rule)
        validate_projection_every(30, manifold_projection=False)
        validate_projection_strength(0.5, manifold_projection=False)
        validate_projection_model("ds8", manifold_projection=False)
        with pytest.raises(ValueError, match="manifold_projection"):
            validate_projection_every(15, manifold_projection=False)
        with pytest.raises(ValueError, match="manifold_projection"):
            validate_projection_strength(0.7, manifold_projection=False)
        with pytest.raises(ValueError, match="manifold_projection"):
            validate_projection_model("ds16", manifold_projection=False)

    def test_valid_config_passes(self):
        validate_manifold_projection(**_valid_config())
        validate_manifold_projection(**_valid_config(manifold_projection=False))
        # fourier rides Unlimited Palette; the projection composes with it
        validate_manifold_projection(
            **_valid_config(image_model="Unlimited Palette")
        )

    @pytest.mark.parametrize("model", ["VQGAN", "LlamaGen"])
    def test_latent_models_fail_loud(self, model):
        with pytest.raises(ValueError, match="decoder manifold"):
            validate_manifold_projection(**_valid_config(image_model=model))

    def test_animation_rejected(self):
        with pytest.raises(ValueError, match="animation_mode"):
            validate_manifold_projection(**_valid_config(animation_mode="2D"))

    def test_structure_annealing_rejected(self):
        # one between-steps intervention at a time in v1
        with pytest.raises(ValueError, match="one between-steps"):
            validate_manifold_projection(
                **_valid_config(structure_annealing=True)
            )

    def test_auto_stop_rejected(self):
        with pytest.raises(ValueError, match="auto_stop"):
            validate_manifold_projection(**_valid_config(auto_stop=True))

    def test_adamw_sf_rejected(self):
        with pytest.raises(ValueError, match="optimizer"):
            validate_manifold_projection(**_valid_config(optimizer="adamw_sf"))

    def test_budget_too_small_rejected(self):
        with pytest.raises(ValueError, match="step budget"):
            validate_manifold_projection(**_valid_config(steps_budget=30))

    def test_stride_vs_dims(self):
        validate_projection_dims(256, 256, "ds16")
        validate_projection_dims(256, 192, "ds8")
        with pytest.raises(ValueError, match="multiples of 16"):
            validate_projection_dims(250, 256, "ds16")
        with pytest.raises(ValueError, match="multiples of 8"):
            validate_projection_dims(252, 252, "ds8")
        with pytest.raises(ValueError, match="positive"):
            validate_projection_dims(0, 256, "ds8")
        # /8-but-not-/16 dims get pointed at ds8
        with pytest.raises(ValueError, match="ds8"):
            validate_projection_dims(264, 256, "ds16")

    def test_stride_checked_through_the_full_validator(self):
        with pytest.raises(ValueError, match="multiples of 16"):
            validate_manifold_projection(
                **_valid_config(projection_model="ds16", width=250)
            )

    def test_auto_aspect_dims_defer_the_stride_check(self):
        # -1 = AUTO aspect, resolved from the init image later: workhorse
        # re-validates after resolution and vq_project re-checks the live
        # plane, so the deferral can never become a silent skip
        validate_manifold_projection(**_valid_config(width=-1))
        validate_manifold_projection(**_valid_config(width=-1, height=-1))

    def test_schema_wiring_rejects_bad_model(self):
        with initialize(config_path="config", version_base=None):
            cfg = compose(
                config_name="_structured_config",
                overrides=[
                    "scenes=x",
                    "manifold_projection=true",
                    "projection_model=ds4",
                ],
            )
        with pytest.raises(ValueError, match="projection_model"):
            OmegaConf.to_object(cfg)

    def test_schema_wiring_rejects_inert_knob(self):
        with initialize(config_path="config", version_base=None):
            cfg = compose(
                config_name="_structured_config",
                overrides=["scenes=x", "projection_every=15"],
            )
        with pytest.raises(ValueError, match="manifold_projection"):
            OmegaConf.to_object(cfg)

    def test_schema_wiring_accepts_valid_projection(self):
        with initialize(config_path="config", version_base=None):
            cfg = compose(
                config_name="_structured_config",
                overrides=[
                    "scenes=x",
                    "manifold_projection=true",
                    "projection_every=15",
                    "projection_strength=0.7",
                    "projection_model=ds16",
                ],
            )
        obj = OmegaConf.to_object(cfg)
        assert obj.manifold_projection is True
        assert obj.projection_every == 15
        assert obj.projection_strength == 0.7
        assert obj.projection_model == "ds16"


# ---------------------------------------------------------------------------
# the projection op
# ---------------------------------------------------------------------------


class TestProjectionOp:
    def test_blend_is_exact(self):
        # the documented contract: out = (1 - l) * x + l * VQ(x), EXACT
        torch.manual_seed(SEED)
        x = torch.rand(3, 32, 32)
        stub = _StubVQ()
        for strength in (0.25, 0.5, 1.0):
            out = project_pixels(x, stub, strength)
            expected = (1.0 - strength) * x + strength * vq_project(x, stub)
            assert torch.equal(out, expected)

    def test_full_strength_is_full_replace(self):
        torch.manual_seed(SEED)
        x = torch.rand(3, 16, 16)
        stub = _StubVQ()
        assert torch.equal(project_pixels(x, stub, 1.0), vq_project(x, stub))

    def test_vq_round_trip_stays_in_range_and_projects(self):
        torch.manual_seed(SEED)
        x = torch.rand(3, 32, 32)
        out = vq_project(x, _StubVQ())
        assert out.shape == x.shape
        assert float(out.min()) >= 0.0 and float(out.max()) <= 1.0
        assert not torch.equal(out, x)  # a genuine projection, not identity

    def test_stride_mismatch_fails_loud_at_runtime(self):
        with pytest.raises(ValueError, match="stride"):
            vq_project(torch.rand(3, 30, 32), _StubVQ())

    def test_shape_validated(self):
        with pytest.raises(ValueError, match="height"):
            vq_project(torch.rand(32, 32), _StubVQ())
        with pytest.raises(ValueError, match="height"):
            hsp_value(torch.rand(32, 32))

    def test_zero_strength_rejected(self):
        with pytest.raises(ValueError, match="projection_strength"):
            project_pixels(torch.rand(3, 16, 16), _StubVQ(), 0.0)

    def test_headroom_decays_toward_gamut(self):
        # the VQ input is clamped; the blend rides on the raw values, so
        # out-of-[0, 1] headroom decays at rate strength — deliberate
        # (accumulated out-of-gamut push is unclean color)
        x = torch.full((3, 16, 16), 0.5)
        x[:, :4, :4] = 1.4
        out = project_pixels(x, _StubVQ(), 0.5)
        assert float(out.max()) < 1.4
        assert float(out.max()) > 1.0  # decays, not slams to the boundary

    def test_no_rng_consumed_and_deterministic(self):
        torch.manual_seed(SEED)
        x = torch.rand(3, 32, 32)
        state_before = torch.get_rng_state().clone()
        first = project_pixels(x, _StubVQ(), 0.5)
        second = project_pixels(x, _StubVQ(), 0.5)
        assert torch.equal(first, second)
        assert torch.equal(state_before, torch.get_rng_state())


# ---------------------------------------------------------------------------
# per-model application
# ---------------------------------------------------------------------------


def _pixel_image(width=32, height=24, scale=1):
    return PixelImage(
        width=width,
        height=height,
        scale=scale,
        palette_size=4,
        n_palettes=3,
        gamma=1,
        hdr_weight=0.01,
        norm_weight=0.2,
        device=CPU,
    ).to(CPU)


class TestProjectImageRep:
    def test_pixel_light_reencode_preserves_palette_identity(self):
        # the light re-encode touches ONLY the value plane: the palette
        # (its identity IS the look) and the selection logits (the
        # structure the projection preserves) stay bit-identical — and no
        # 201-step smart_encode fit runs (this whole test is milliseconds)
        torch.manual_seed(SEED)
        img = _pixel_image()
        img.encode_random()
        before = {k: v.clone() for k, v in img.state_dict().items()}
        project_image_rep(img, strength=0.8, model=_StubVQ())
        after = img.state_dict()
        assert not torch.equal(after["value"], before["value"])
        for key in before:
            if key != "value":
                assert torch.equal(after[key], before[key]), key

    def test_pixel_value_matches_the_documented_formula(self):
        # value <- clamp(value + l * (hsp(VQ(canvas)) - hsp(canvas))): the
        # projection's brightness DELTA lands at rate l — never a
        # wholesale hsp(blended) replace (that would land the decode's
        # palette-quantization remap at full strength regardless of l)
        torch.manual_seed(SEED)
        img = _pixel_image()
        img.encode_random()
        stub = _StubVQ()
        canvas = pixel_canvas(img)
        value_before = img.value.detach().clone()
        expected = (
            value_before
            + 0.5 * (hsp_value(vq_project(canvas, stub)) - hsp_value(canvas))
        ).clamp(0.0, 1.0)
        project_image_rep(img, strength=0.5, model=stub)
        assert torch.equal(img.value.detach(), expected)

    def test_pixel_projection_is_identity_on_manifold_fixed_points(self):
        # an on-manifold canvas (VQ(x) = x) is a FIXED POINT: ten cycles at
        # any strength leave every parameter bit-identical. The pre-fix
        # hsp(blended) replace failed this by mean |dvalue| ~0.15 per
        # cycle at strength 0.05 — a strength-independent remap
        torch.manual_seed(SEED)
        img = _pixel_image()
        img.encode_random()
        before = {k: v.clone() for k, v in img.state_dict().items()}
        for strength in (0.05, 0.5, 1.0):
            for _ in range(10):
                project_image_rep(img, strength=strength, model=_IdentityVQ())
        for key in before:
            assert torch.equal(img.state_dict()[key], before[key]), key

    def test_pixel_delta_scales_linearly_with_strength(self):
        # projection_strength must MEAN something on Limited Palette: the
        # landed value delta is linear in l (up to the [0, 1] clamp)
        def drift(strength):
            torch.manual_seed(SEED)
            img = _pixel_image()
            img.encode_random()
            with torch.no_grad():
                # keep the fixture clear of the clamp so linearity is exact
                img.value.mul_(0.5).add_(0.25)
            v0 = img.value.detach().clone()
            project_image_rep(img, strength=strength, model=_StubVQ())
            return img.value.detach() - v0

        assert torch.allclose(drift(0.5), 2.0 * drift(0.25), atol=1e-6)

    def test_hsp_value_matches_encode_image_math(self):
        # formula-identical to PixelImage.encode_image's value_ref
        torch.manual_seed(SEED)
        pixels = torch.rand(3, 8, 8)
        magic_color = torch.tensor([[[0.299]], [[0.587]], [[0.114]]])
        reference = torch.linalg.vector_norm(
            pixels * (magic_color.sqrt()), dim=0
        )
        assert torch.equal(hsp_value(pixels), reference)

    def test_pixel_canvas_strides_back_from_pixel_size(self):
        torch.manual_seed(SEED)
        img = _pixel_image(width=16, height=12, scale=2)
        img.encode_random()
        canvas = pixel_canvas(img)
        assert canvas.shape == (3, 12, 16)  # the LOGICAL grid
        full = img.decode_tensor().squeeze(0)
        assert torch.equal(canvas, full[:, ::2, ::2])

    def test_pixel_value_stays_in_unit_range(self):
        torch.manual_seed(SEED)
        img = _pixel_image()
        img.encode_random()
        project_image_rep(img, strength=1.0, model=_StubVQ())
        value = img.value.detach()
        assert float(value.min()) >= 0.0 and float(value.max()) <= 1.0

    def test_rgb_blends_on_the_raw_plane(self):
        torch.manual_seed(SEED)
        img = RGBImage(32, 24, 1, device=CPU)
        img.encode_random()
        stub = _StubVQ()
        expected = project_pixels(
            img.tensor.detach().squeeze(0), stub, 0.5
        ).unsqueeze(0)
        project_image_rep(img, strength=0.5, model=stub)
        assert torch.equal(img.tensor.detach(), expected)

    def test_rgb_headroom_decays(self):
        torch.manual_seed(SEED)
        img = RGBImage(32, 24, 1, device=CPU)
        img.encode_random()
        with torch.no_grad():
            img.tensor[0, :, :4, :4] = 1.4
        project_image_rep(img, strength=0.5, model=_StubVQ())
        after = img.tensor.detach()
        assert float(after.max()) < 1.4
        assert float(after.max()) > 1.0

    def test_fourier_round_trips_through_the_spectrum(self):
        torch.manual_seed(SEED)
        img = FourierImage(32, 24, 1, device=CPU)
        img.encode_random()
        stub = _StubVQ()
        spectrum_before = img.spectrum_real.detach().clone()
        blended = project_pixels(img.get_image_tensor().detach(), stub, 0.5)
        project_image_rep(img, strength=0.5, model=stub)
        assert not torch.equal(img.spectrum_real.detach(), spectrum_before)
        # image-domain round-trips are FourierImage's contract: the decode
        # reproduces the blended pixels up to logit clamp + float error
        assert torch.allclose(
            img.get_image_tensor().detach(), blended, atol=1e-4
        )

    def test_parameter_identity_preserved(self):
        # in-place copy_ keeps the Parameter objects, so optimizer state
        # (Adam moments) stays attached — the KEEP-moments decision
        torch.manual_seed(SEED)
        img = _pixel_image()
        img.encode_random()
        value_param = img.value
        project_image_rep(img, strength=0.5, model=_StubVQ())
        assert img.value is value_param

    def test_unknown_model_fails_loud(self):
        with pytest.raises(ValueError, match="no projection path"):
            project_image_rep(object(), strength=0.5, model=_StubVQ())

    def test_deterministic_and_rng_free(self):
        def run():
            torch.manual_seed(SEED)
            img = RGBImage(32, 24, 1, device=CPU)
            img.encode_random()
            rng_before = torch.get_rng_state().clone()
            project_image_rep(img, strength=0.5, model=_StubVQ())
            assert torch.equal(rng_before, torch.get_rng_state())
            return img.tensor.detach().clone()

        assert torch.equal(run(), run())


# ---------------------------------------------------------------------------
# the real tokenizer (checkpoint tier — default-deselected)
# ---------------------------------------------------------------------------


@pytest.mark.download
class TestRealTokenizerFrozen:
    def test_loaded_tokenizer_is_fully_frozen(self):
        # the frozen contract, on the REAL checkpoint: eval mode and
        # requires_grad=False on every parameter AND buffer (the vendored
        # quantizer registers codebook_used as an nn.Parameter buffer,
        # which Module.requires_grad_ alone does not touch)
        from pytti.manifold_projection import (
            free_projection_model,
            load_projection_model,
        )

        try:
            model = load_projection_model("ds8", device=CPU)
            assert not model.training
            assert not any(t.requires_grad for t in model.parameters())
            assert not any(t.requires_grad for t in model.buffers())
        finally:
            free_projection_model()


# ---------------------------------------------------------------------------
# guide integration (torch path)
# ---------------------------------------------------------------------------


def _guide_params(**overrides):
    cfg = dict(
        animation_mode="off",
        optimizer="adam",
        perceptor_backend="torch",
        structure_annealing=False,
        manifold_projection=True,
        projection_every=5,
        projection_strength=0.5,
        projection_model="ds8",
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


def _rgb_guide(init_augs=None, width=24, height=16, **param_overrides):
    img = RGBImage(width, height, 1, device=CPU)
    img.encode_random()
    guide = DirectImageGuide(
        image_rep=img,
        embedder=None,
        params=_guide_params(**param_overrides),
        init_augs=init_augs,
    )
    # never download in tests: the lazily-loaded tokenizer slot takes the
    # deterministic stub instead
    guide._projection_model = _StubVQ()
    return guide, img


class TestGuideIntegration:
    def test_schedule_matches_the_pure_function(self):
        guide, _ = _rgb_guide(steps_per_scene=100, projection_every=30)
        assert guide.projection_schedule == projection_steps(100, 30)

    def test_projection_fires_during_run_steps(self):
        torch.manual_seed(SEED)
        # 16 steps, every 5: usable = round(10.4) = 10 -> cycles at 5, 10
        guide, img = _rgb_guide()
        assert guide.projection_schedule == (5, 10)
        before = img.tensor.detach().clone()
        guide.run_steps(16, [], [], [])
        # no losses, no grads: only the projection can change the image
        assert not torch.equal(img.tensor.detach(), before)

    def test_projection_off_is_a_no_op(self):
        torch.manual_seed(SEED)
        guide, img = _rgb_guide(manifold_projection=False)
        assert guide.projection_schedule is None
        before = img.tensor.detach().clone()
        guide.run_steps(16, [], [], [])
        assert torch.equal(img.tensor.detach(), before)

    def test_fixed_seed_reproduces_the_projected_run(self):
        def run():
            torch.manual_seed(SEED)
            guide, img = _rgb_guide()
            guide.run_steps(16, [], [], [])
            return img.tensor.detach().clone()

        assert torch.equal(run(), run())

    def test_init_holds_are_NOT_released(self):
        # the projection preserves the canvas's own structure — there is
        # no re-liquification for a hold to cancel, so unlike annealing
        # the direct init holds stay enabled through every cycle
        from pytti.LossAug.BaseLossClass import Loss

        torch.manual_seed(SEED)
        hold = Loss(weight="2", stop=-math.inf, name="init hold", device=CPU)
        guide, _img = _rgb_guide(init_augs=[hold])
        guide.run_steps(16, [], [], [])
        assert hold.enabled

    def test_restore_replays_remaining_cycles(self):
        # the schedule is a pure function of the config: a resume past the
        # first cycle still projects at the second cycle's scene step
        torch.manual_seed(SEED)
        guide, img = _rgb_guide()  # cycles at 5, 10
        before = img.tensor.detach().clone()
        guide.run_steps(8, [], [], [], skipped_steps=8)  # covers step 10
        assert not torch.equal(img.tensor.detach(), before)

    def test_animation_rejected_at_construction(self):
        with pytest.raises(ValueError, match="animation_mode"):
            _rgb_guide(animation_mode="2D")

    def test_auto_stop_rejected_at_construction(self):
        with pytest.raises(ValueError, match="auto_stop"):
            _rgb_guide(auto_stop=True)

    def test_adamw_sf_rejected_at_construction(self):
        with pytest.raises(ValueError, match="optimizer"):
            _rgb_guide(optimizer="adamw_sf")

    def test_annealing_combo_rejected_at_construction(self):
        with pytest.raises(ValueError, match="one between-steps"):
            _rgb_guide(
                structure_annealing=True,
                anneal_cycles=3,
                anneal_strength=0.5,
                anneal_band=0.15,
                anneal_source="noise",
            )

    def test_latent_model_rejected_at_construction(self):
        class TokenImage(DifferentiableImage):
            def __init__(self):
                super().__init__(8, 8)

        with pytest.raises(ValueError, match="no projection path"):
            DirectImageGuide(
                image_rep=TokenImage(), embedder=None, params=_guide_params()
            )

    def test_stride_mismatch_rejected_at_construction(self):
        # 20 is not a multiple of ds8's stride 8
        with pytest.raises(ValueError, match="multiple of 8"):
            _rgb_guide(width=20)

    def test_too_small_scene_fails_at_construction(self):
        # usable = round(6 * 0.65) = 4 < 5: no projection would ever fire
        with pytest.raises(ValueError, match="step budget"):
            _rgb_guide(steps_per_scene=6)


# ---------------------------------------------------------------------------
# the mlx_full host-intervention seam
# ---------------------------------------------------------------------------


@needs_mlx
class TestMlxSeam:
    def _fixture(self):
        # stub towers / embedder from the engine test module (no downloads)
        from tests.test_mlx_engine_step import _make_engine, _seeded_pixel_image

        img = _seeded_pixel_image()  # 64x48, stride-4 stub divides both
        engine = _make_engine(img=img)
        return engine, img

    def _untouched_state(self, engine):
        """Everything a projection cycle promises NOT to disturb: the Adam
        moments + step count and mx's global RNG state (the params tree's
        value plane is the one thing it edits)."""
        import mlx.core as mx
        from mlx.utils import tree_flatten

        return (
            {k: np.array(v) for k, v in tree_flatten(engine._opt.state)},
            np.array(mx.random.state[0]),
        )

    def test_projection_cycle_leaves_opt_and_rng_bit_identical(self):
        # extends the structure-annealing seam bit-identity tests to the
        # projection intervention: write_back -> project -> import_params
        # must keep the Adam moments, the mx RNG stream, AND the torch RNG
        # stream untouched (the projection consumes no RNG), while editing
        # ONLY the value plane of the params tree
        engine, img = self._fixture()
        params_before = {k: np.array(v) for k, v in engine._params.items()}
        opt_before, rng_before = self._untouched_state(engine)
        torch_rng_before = torch.get_rng_state().clone()

        engine.write_back(img)
        project_image_rep(img, strength=0.8, model=_StubVQ())
        engine.import_params(img)

        opt_after, rng_after = self._untouched_state(engine)
        assert set(opt_before) == set(opt_after)
        for key in opt_before:
            assert np.array_equal(opt_before[key], opt_after[key]), f"opt.{key}"
        assert np.array_equal(rng_before, rng_after), "mx.random.state"
        assert torch.equal(torch_rng_before, torch.get_rng_state())
        assert not np.array_equal(
            params_before["value"], np.array(engine._params["value"])
        )
        for key in params_before:
            if key != "value":
                assert np.array_equal(
                    params_before[key], np.array(engine._params[key])
                ), key

    def test_torch_vs_mlx_cycle_equivalence(self):
        from pytti.mlx_engine.image_models import pixel_state_dict_from_params
        from tests.test_mlx_engine_step import _pixel_image

        engine, img = self._fixture()
        twin = _pixel_image()
        twin.load_state_dict(
            {k: v.clone() for k, v in img.state_dict().items()}
        )

        # mlx path: export -> project in torch -> import (what
        # DirectImageGuide._apply_manifold_projection does between steps)
        engine.write_back(img)
        project_image_rep(img, strength=0.6, model=_StubVQ())
        engine.import_params(img)

        # torch path: the same operation on the same starting state
        project_image_rep(twin, strength=0.6, model=_StubVQ())

        exported = pixel_state_dict_from_params(engine._params)
        reference = twin.state_dict()
        assert set(exported) == set(reference)
        for key in exported:
            # BIT equality: the seam is numpy copies both ways and the
            # projection runs torch-side on identical fp32 values
            assert torch.equal(exported[key], reference[key]), key

    def test_intervention_diverges_exactly_at_the_cycle_step(self):
        # a step-4 host projection leaves steps 0..3 BIT-identical to the
        # projection-free run (params AND the mx RNG stream stay in
        # lockstep — the projection consumes NO RNG at all) and diverges
        # exactly at step 4. Pinned to the mx CPU device: Metal float
        # reductions are not run-to-run bit-deterministic
        import mlx.core as mx

        from tests.test_mlx_engine_step import _prompt, _tv_loss

        def run(project_at):
            mx.random.seed(SEED)
            torch.manual_seed(SEED)
            engine, img = self._fixture()
            frames = []
            for i in range(6):
                if i == project_at:
                    engine.write_back(img)
                    project_image_rep(img, strength=0.8, model=_StubVQ())
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
            base = run(project_at=None)
            intervened = run(project_at=4)
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
