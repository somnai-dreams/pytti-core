"""
Phase scheduling (phase_scheduling: true) — quality-phase behavior over
normalized scene time t_hat = step/steps_per_scene.

Four layers:
- the schedule table as pure functions (values at t_hat 0 / 0.33 / 0.67 / 1,
  domain validation);
- the torch seams: Loss.weight_scale wraps the parametric-eval'd weight,
  DirectImageGuide sets the per-step scales and flips the Limited Palette
  lock (PixelImage.lock_palette — palette grads vanish) for the final third;
- the mlx_full seams (stub towers, no downloads): the same guide wiring
  drives the compiled step's per-step host args — weight_scale folds into
  aug_w, the 0/1 palette gate freezes the palette tree exactly;
- a live A/B smoke (`download` marker): 60-step Limited Palette renders,
  phase on vs off, judged by LPIPS against a same-seed null pair.
"""

import importlib.util
import math
from pathlib import Path

import numpy as np
import pytest
import torch
from hydra import compose, initialize
from omegaconf import OmegaConf
from PIL import Image

from pytti import set_t
from pytti.config.structured_config import ConfigSchema
from pytti.image_models import PixelImage, RGBImage
from pytti.ImageGuide import DirectImageGuide
from pytti.LossAug.TVLossClass import TVLoss
from pytti.phase_scheduling import (
    INIT_SCALE_END,
    PALETTE_LOCK_T_HAT,
    TV_SCALE_START,
    init_weight_scale,
    palette_locked,
    scene_t_hat,
    tv_weight_scale,
)
from tests.test_mlx_engine_step import (
    SEED,
    StubEmbedder,
    StubPerceptor,
    _base_params,
    _make_engine,
    _prompt,
    _seeded_pixel_image,
    _stub_loader,
    _tv_loss,
    rel_err,
)

needs_mlx = pytest.mark.skipif(
    importlib.util.find_spec("mlx") is None,
    reason="mlx not installed (darwin-only backend)",
)

CPU = torch.device("cpu")


# ---------------------------------------------------------------------------
# the schedule table as pure functions
# ---------------------------------------------------------------------------


def test_tv_scale_ramps_2x_to_half():
    assert tv_weight_scale(0.0) == pytest.approx(2.0)
    assert tv_weight_scale(0.33) == pytest.approx(2.0 - 1.5 * 0.33)  # 1.505
    assert tv_weight_scale(0.67) == pytest.approx(2.0 - 1.5 * 0.67)  # 0.995
    assert tv_weight_scale(1.0) == pytest.approx(0.5)
    assert TV_SCALE_START == 2.0


def test_init_scale_decays_1x_to_half():
    assert init_weight_scale(0.0) == pytest.approx(1.0)
    assert init_weight_scale(0.33) == pytest.approx(1.0 - 0.5 * 0.33)  # 0.835
    assert init_weight_scale(0.67) == pytest.approx(1.0 - 0.5 * 0.67)  # 0.665
    assert init_weight_scale(1.0) == pytest.approx(0.5)
    assert INIT_SCALE_END == 0.5


def test_palette_locks_strictly_after_two_thirds():
    assert palette_locked(0.0) is False
    assert palette_locked(0.33) is False
    assert palette_locked(PALETTE_LOCK_T_HAT) is False  # exactly 2/3: open
    assert palette_locked(0.67) is True
    assert palette_locked(1.0) is True


def test_scene_t_hat():
    assert scene_t_hat(0, 90) == 0.0
    assert scene_t_hat(30, 90) == pytest.approx(1 / 3)
    assert scene_t_hat(90, 90) == 1.0


def test_scene_t_hat_fails_loud_outside_the_scene():
    with pytest.raises(ValueError, match="steps_per_scene"):
        scene_t_hat(0, 0)
    with pytest.raises(ValueError, match="scene-local"):
        scene_t_hat(-1, 90)
    with pytest.raises(ValueError, match="scene-local"):
        scene_t_hat(91, 90)  # a global step leaked in


@pytest.mark.parametrize(
    "fn", [tv_weight_scale, init_weight_scale, palette_locked]
)
def test_schedules_reject_t_hat_outside_unit_interval(fn):
    with pytest.raises(ValueError, match="t_hat"):
        fn(-0.1)
    with pytest.raises(ValueError, match="t_hat"):
        fn(1.1)


# ---------------------------------------------------------------------------
# torch seams: Loss.weight_scale
# ---------------------------------------------------------------------------


def _cpu_tv(weight):
    tv = TVLoss(weight=weight)
    tv.device = CPU
    return tv


def test_weight_scale_multiplies_the_evaluated_weight():
    tv = _cpu_tv(0.5)
    x = torch.rand((1, 3, 8, 8), generator=torch.Generator().manual_seed(0))
    loss_1, raw_1 = tv(x, None)
    tv.weight_scale = 2.0
    loss_2, raw_2 = tv(x, None)
    assert torch.allclose(loss_2, loss_1 * 2.0)
    assert torch.equal(raw_1, raw_2)  # raws are never scaled


def test_weight_scale_composes_with_parametric_expressions():
    set_t(0.4)
    try:
        tv = _cpu_tv("t")  # evaluates to 0.4
        x = torch.rand((1, 3, 8, 8), generator=torch.Generator().manual_seed(1))
        loss_1, raw = tv(x, None)
        assert torch.allclose(loss_1, raw * 0.4)
        tv.weight_scale = 0.5  # wraps the VALUE; the expression is untouched
        loss_2, _ = tv(x, None)
        assert tv.weight == "t"
        assert torch.allclose(loss_2, loss_1 * 0.5)
    finally:
        set_t(0)


# ---------------------------------------------------------------------------
# torch wiring: DirectImageGuide sets scales + manages the palette lock
# ---------------------------------------------------------------------------


def make_params(**overrides):
    cfg = OmegaConf.structured(ConfigSchema)
    for key, value in overrides.items():
        setattr(cfg, key, value)
    return cfg


def _pixel_image():
    torch.manual_seed(SEED)
    img = PixelImage(
        16, 16, scale=1, palette_size=4, n_palettes=3, gamma=1,
        hdr_weight=0.01, norm_weight=0.2, device=CPU,
    ).to(CPU)
    img.encode_random()
    return img


def _init_aug(img):
    rng = np.random.default_rng(7)
    pil = Image.fromarray(
        rng.integers(0, 255, (16, 16, 3), dtype=np.uint8), "RGB"
    )
    aug = type(img).get_preferred_loss().build(
        "direct init image (x)", img.image_shape,
        weight="1", pil_image=pil, device=CPU,
    )
    return aug


def _torch_guide(img, init_augs=(), steps_per_scene=90, **overrides):
    overrides.setdefault("phase_scheduling", True)
    overrides.setdefault("steps_per_scene", steps_per_scene)
    return DirectImageGuide(
        img, None, params=make_params(**overrides), init_augs=list(init_augs)
    )


def test_animation_mode_fails_loud():
    with pytest.raises(ValueError, match="phase_scheduling"):
        _torch_guide(_pixel_image(), animation_mode="2D")


def test_guide_sets_per_step_scales():
    img = _pixel_image()
    tv = _cpu_tv(0.02)
    init_aug = _init_aug(img)
    guide = _torch_guide(img, init_augs=[init_aug], steps_per_scene=90)
    guide.train(30, [], [], [tv, init_aug])
    assert tv.weight_scale == pytest.approx(tv_weight_scale(30 / 90))
    assert init_aug.weight_scale == pytest.approx(init_weight_scale(30 / 90))
    guide.train(89, [], [], [tv, init_aug])
    assert tv.weight_scale == pytest.approx(tv_weight_scale(89 / 90))
    assert init_aug.weight_scale == pytest.approx(init_weight_scale(89 / 90))


def test_phase_off_leaves_scales_neutral():
    img = _pixel_image()
    tv = _cpu_tv(0.02)
    guide = _torch_guide(img, phase_scheduling=False)
    guide.train(70, [], [], [tv])
    assert tv.weight_scale == 1.0
    assert img.use_palette_target is False


def test_torch_palette_locks_for_the_final_third_and_reopens():
    img = _pixel_image()
    tv = _cpu_tv(0.02)
    guide = _torch_guide(img, steps_per_scene=90)

    # open phase: the palette trains
    palette_0 = img.palette.detach().clone()
    guide.train(30, [], [], [tv])
    assert img.use_palette_target is False
    assert not torch.equal(img.palette.detach(), palette_0)

    # past 2/3: locked BEFORE the optimizer step — the palette is frozen
    # while value/tensor keep moving
    palette_1 = img.palette.detach().clone()
    value_1 = img.value.detach().clone()
    tensor_1 = img.tensor.detach().clone()
    guide.train(61, [], [], [tv])  # t_hat = 61/90 > 2/3
    assert img.use_palette_target is True
    assert torch.equal(img.palette.detach(), palette_1)
    assert not torch.equal(img.value.detach(), value_1)
    assert not torch.equal(img.tensor.detach(), tensor_1)

    # the lock is the lock_palette mechanism: the palette leaves the graph
    z = img.decode_training_tensor()
    z.sum().backward()
    assert img.palette.grad is None
    assert img.value.grad is not None
    img.zero_grad(set_to_none=True)

    # a new scene resets t_hat: the palette reopens
    guide.train(0, [], [], [tv])
    assert img.use_palette_target is False


def test_torch_config_locked_palette_is_never_touched():
    img = _pixel_image()
    img.lock_palette(True)  # the user's lock_palette config
    tv = _cpu_tv(0.02)
    guide = _torch_guide(img, steps_per_scene=90)
    for step in (0, 70, 0):  # open, locked, open again
        guide.train(step, [], [], [tv])
        assert img.use_palette_target is True


# ---------------------------------------------------------------------------
# mlx_full seams (stub towers, no downloads)
# ---------------------------------------------------------------------------


@needs_mlx
def test_mlx_guide_palette_gate_freezes_the_palette(monkeypatch):
    import pytti.mlx_engine.engine as engine_module

    def loader(key, dtype):
        return _stub_loader({"ViTB32": "T32", "ViTB16": "T16"}[key], dtype)

    monkeypatch.setattr(engine_module, "load_tower", loader)
    embedder = StubEmbedder()
    embedder.perceptors = [
        StubPerceptor("ViTB32", 32),
        StubPerceptor("ViTB16", 16),
    ]
    guide = DirectImageGuide(
        image_rep=_seeded_pixel_image(),
        embedder=embedder,
        params=_base_params(phase_scheduling=True, steps_per_scene=90),
    )
    engine = guide.mlx_engine
    assert engine is not None
    prompts = [_prompt("a test prompt:1")]
    augs = [_tv_loss(0.02)]

    # open phase: the palette trains, and the guide set the TV scale
    palette_0 = np.array(engine._params["palette"])
    guide.train(30, prompts, [], augs)
    assert augs[0].weight_scale == pytest.approx(tv_weight_scale(30 / 90))
    assert not np.array_equal(np.array(engine._params["palette"]), palette_0)

    # past 2/3: gate 0 — palette bit-frozen, value/tensor keep moving
    palette_1 = np.array(engine._params["palette"])
    value_1 = np.array(engine._params["value"])
    tensor_1 = np.array(engine._params["tensor"])
    guide.train(70, prompts, [], augs)  # t_hat = 7/9 > 2/3
    assert np.array_equal(np.array(engine._params["palette"]), palette_1)
    assert not np.array_equal(np.array(engine._params["value"]), value_1)
    assert not np.array_equal(np.array(engine._params["tensor"]), tensor_1)

    # a new scene resets t_hat: gate 1 — the palette trains again
    palette_2 = np.array(engine._params["palette"])
    guide.train(0, prompts, [], augs)
    assert not np.array_equal(np.array(engine._params["palette"]), palette_2)


@needs_mlx
def test_mlx_weight_scale_equals_scaling_the_weight():
    # engineA(weight 0.02, scale 2) and engineB(weight 0.04, scale 1) run
    # the identical seeded step: every record must match (within the
    # measured same-seed scatter-add noise, test_deterministic_given_seed)
    tv_scaled = _tv_loss(0.02)
    tv_scaled.weight_scale = 2.0
    rec_a = _make_engine().train_step(0, [_prompt("a test prompt:1")], [], [tv_scaled])
    rec_b = _make_engine().train_step(0, [_prompt("a test prompt:1")], [], [_tv_loss(0.04)])
    assert set(rec_a) == set(rec_b)
    for name in rec_a:
        assert rel_err(float(rec_a[name]), float(rec_b[name])) <= 1e-5, name


@needs_mlx
def test_mlx_gate_rejects_soft_values():
    engine = _make_engine()
    with pytest.raises(ValueError, match="palette_gate"):
        engine.train_step(
            0, [_prompt("a test prompt:1")], [], [], palette_gate=0.5
        )


@needs_mlx
def test_mlx_rgb_step_takes_the_gate_arg():
    # rgb has no palette: the gate is an unused compiled-step argument and
    # must be accepted without effect
    torch.manual_seed(SEED)
    img = RGBImage(64, 48, 1, device=CPU)
    img.encode_random()
    engine = _make_engine(img=img)
    rec = engine.train_step(
        0, [_prompt("a test prompt:1")], [], [_tv_loss(0.02)],
        palette_gate=0.0,
    )
    assert math.isfinite(float(rec["TOTAL"]))


# ---------------------------------------------------------------------------
# live A/B smoke: phase on vs off changes the final frame (LPIPS > null)
# ---------------------------------------------------------------------------

CONFIG_BASE_PATH = "config"
CONFIG_DEFAULTS = "default.yaml"
FIXTURE_IMAGE = str(
    Path(__file__).parent / "fixtures" / "01-velo-header-seattle-needle.jpg"
)


@pytest.mark.download
def test_phase_scheduling_ab_smoke(tmp_path, monkeypatch):
    """Four 60-step Limited Palette renders.

    LPIPS A/B runs on the bit-deterministic path — perceptor_backend=torch,
    device: cpu (the conf), one torch thread (here): off/off is the
    same-seed null pair (measured exactly 0.0, asserted tiny), so the
    on/off final-frame LPIPS is pure schedule effect.

    mlx_full CANNOT be LPIPS-gated against a same-seed null: its
    gather-bilinear backward is a GPU scatter-add with nondeterministic
    accumulation order, and Limited Palette's discrete argmax decode
    amplifies that into palette flips — measured same-seed null LPIPS ~0.13
    at 60 steps regardless of anchoring, swamping any A/B. Its arm here is
    a live phase-on completion check (schedule fires, render completes);
    the mlx seams themselves are bit-exactly gated by the stub-tower tests
    above.
    """
    from loguru import logger

    from pytti.Perceptor import load_clip
    from pytti.workhorse import _hydra_main as render_frames

    monkeypatch.chdir(tmp_path)
    threads_before = torch.get_num_threads()
    torch.set_num_threads(1)

    # Warm the in-process perceptor cache BEFORE the measured arms: building
    # the CLIP module consumes torch RNG (module init draws before the
    # weights load), and workhorse seeds before load_clip — so the arm that
    # builds the model gets a different RNG cursor than the arms that reuse
    # it. Warmed, every arm skips the build and same-seed runs are
    # bit-identical (scripts/ab_render.py's cpu/1-thread floor).
    load_clip(
        OmegaConf.create({"ViTB32": True, "perceptor_backend": "torch"}),
        device=CPU,
    )

    def render(namespace, phase, backend="torch"):
        with initialize(config_path=CONFIG_BASE_PATH, version_base=None):
            cfg = compose(
                config_name=CONFIG_DEFAULTS,
                overrides=[
                    "conf=_test_phase_scheduling",
                    f"init_image={FIXTURE_IMAGE}",
                    f"phase_scheduling={'true' if phase else 'false'}",
                    f"file_namespace={namespace}",
                    f"perceptor_backend={backend}",
                ],
            )
            render_frames(cfg)
        frames = sorted((tmp_path / "images_out" / namespace).glob("*.png"))
        assert frames, f"{namespace} rendered no frames"
        # every arm completes: 60 steps / save_every 30 -> final frame 0002
        assert frames[-1].name == f"{namespace}_0002.png"
        return frames[-1]

    try:
        final_off_a = render("phase_off_a", phase=False)
        final_off_b = render("phase_off_b", phase=False)
        final_on = render("phase_on", phase=True)

        # live mlx_full phase-on arm: completes AND the palette lock fired
        if importlib.util.find_spec("mlx") is not None:
            messages = []
            sink_id = logger.add(lambda m: messages.append(m), level="INFO")
            try:
                render("phase_on_mlx", phase=True, backend="mlx_full")
            finally:
                logger.remove(sink_id)
            assert any(
                "phase_scheduling: palette locked" in m for m in messages
            ), "the palette lock never fired on the live mlx_full render"
    finally:
        torch.set_num_threads(threads_before)

    import lpips

    loss_fn = lpips.LPIPS(net="alex", verbose=False)
    loss_fn.eval()

    def to_tensor(path):
        arr = np.asarray(Image.open(path).convert("RGB"), dtype=np.float32)
        return torch.from_numpy(arr / 255.0).permute(2, 0, 1).unsqueeze(0)

    with torch.no_grad():
        null = float(
            loss_fn(to_tensor(final_off_a) * 2 - 1, to_tensor(final_off_b) * 2 - 1)
        )
        ab = float(
            loss_fn(to_tensor(final_off_a) * 2 - 1, to_tensor(final_on) * 2 - 1)
        )
    # deterministic arms: the null pair must be (near-)identical, and the
    # schedule effect must dwarf whatever residue remains
    assert null < 1e-4, f"same-seed deterministic null pair drifted: {null:.6f}"
    assert ab > max(10 * null, 1e-3), (
        f"phase on/off LPIPS {ab:.6f} does not clear the null floor "
        f"{null:.6f}"
    )
