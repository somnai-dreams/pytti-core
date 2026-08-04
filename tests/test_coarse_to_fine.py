"""
Coarse-to-fine still rendering.

Four layers:
- the stage math (dims halving with /8 rounding + the 64 floor, the 40/60
  step split, restore->stage mapping) and config validation as pure
  functions;
- the Limited Palette carry: copy_palette_from mechanics and the full
  configure_pass carry (lock through the encode fit, release after);
- frame-numbering continuity across the two run_steps windows the
  orchestrator issues — scripted guides, no render — including a stage-1
  auto_stop convergence;
- a live 256->512 mlx_full smoke (`download` marker): frames from both
  stages, uniform size, and the stage-2 first frame pixel-correlated
  against the persisted stage-1 transition image (not eyeballed).
"""

from pathlib import Path

import numpy as np
import pytest
import torch
from hydra import compose, initialize
from loguru import logger
from omegaconf import OmegaConf
from PIL import Image
from torch import nn

from pytti.coarse_to_fine import (
    coarse_dims,
    resume_stage,
    stage_steps,
    validate_coarse_to_fine,
)
from pytti.config.structured_config import ConfigSchema
from pytti.device import default_device
from pytti.image_models.pixel import PixelImage
from pytti.ImageGuide import DirectImageGuide

# ---------------------------------------------------------------------------
# stage math (pure)
# ---------------------------------------------------------------------------


def test_coarse_dims_halves_and_stays_on_8():
    assert coarse_dims(512, 512) == (256, 256)
    assert coarse_dims(448, 256) == (224, 128)
    # non-/16 dims round half-up to the nearest multiple of 8
    assert coarse_dims(300, 300) == (152, 152)  # 150 -> 152
    assert coarse_dims(200, 200) == (104, 104)  # 100 -> 104
    assert coarse_dims(432, 432) == (216, 216)  # exact


def test_coarse_dims_floors_at_64():
    assert coarse_dims(100, 100) == (64, 64)  # 50 -> 48 -> floor
    assert coarse_dims(128, 128) == (64, 64)
    assert coarse_dims(512, 100) == (256, 64)  # per-dim


def test_coarse_dims_never_exceeds_the_configured_dim():
    # canvases at/below the floor run stage 1 at full size, never upscaled
    assert coarse_dims(64, 64) == (64, 64)
    assert coarse_dims(48, 48) == (48, 48)


def test_coarse_dims_fails_loud_on_nonpositive():
    with pytest.raises(ValueError, match="positive"):
        coarse_dims(0, 512)
    with pytest.raises(ValueError, match="positive"):
        coarse_dims(512, -1)


def test_stage_steps_split_is_40_60():
    assert stage_steps(100) == (40, 60)
    assert stage_steps(50) == (20, 30)
    assert stage_steps(13) == (5, 8)  # 40% floors
    assert stage_steps(3) == (1, 2)  # smallest legal split


def test_stage_steps_sum_is_total():
    for total in range(3, 500):
        s1, s2 = stage_steps(total)
        assert s1 + s2 == total
        assert s1 >= 1 and s2 >= 1


def test_stage_steps_fails_loud_when_a_stage_would_be_empty():
    for total in (0, 1, 2):
        with pytest.raises(ValueError, match="steps_per_scene"):
            stage_steps(total)


def test_resume_stage_maps_restored_steps_onto_stages():
    assert resume_stage(0, 20) == (1, 0)
    assert resume_stage(15, 20) == (1, 15)
    # the boundary resumes as stage 1 with nothing left: the .bak there was
    # written by stage 1 (coarse dims) and the transition replays from it
    assert resume_stage(20, 20) == (1, 20)
    assert resume_stage(25, 20) == (2, 5)
    assert resume_stage(50, 20) == (2, 30)


def test_resume_stage_fails_loud_on_bad_inputs():
    with pytest.raises(ValueError, match=">= 0"):
        resume_stage(-1, 20)
    with pytest.raises(ValueError, match="stage1_steps"):
        resume_stage(0, 0)


# ---------------------------------------------------------------------------
# config validation (pure)
# ---------------------------------------------------------------------------

_VALID = dict(
    animation_mode="off",
    n_scenes=1,
    breath_mode=False,
    semantic_init=False,
    semantic_stabilization=False,
)


def test_validate_accepts_a_plain_still():
    validate_coarse_to_fine(**_VALID)


@pytest.mark.parametrize(
    ("override", "match"),
    [
        (dict(animation_mode="2D"), "STILL"),
        (dict(animation_mode="Video Source"), "STILL"),
        (dict(n_scenes=2), "scenes"),
        (dict(breath_mode=True), "breath_mode"),
        (dict(semantic_init=True), "semantic_init_weight"),
        (dict(semantic_stabilization=True), "semantic_stabilization_weight"),
    ],
)
def test_validate_fails_loud_per_unsupported_setting(override, match):
    with pytest.raises(ValueError, match=match):
        validate_coarse_to_fine(**{**_VALID, **override})


# ---------------------------------------------------------------------------
# Limited Palette carry
# ---------------------------------------------------------------------------


def _pixel_image(width, height, device):
    return PixelImage(
        width=width,
        height=height,
        scale=1,
        palette_size=4,
        n_palettes=2,
        gamma=1,
        hdr_weight=0,
        norm_weight=0.1,
        device=device,
    )


def test_copy_palette_from_carries_the_raw_parameter():
    device = default_device()
    src = _pixel_image(32, 32, device)
    with torch.no_grad():
        src.palette.set_(torch.rand_like(src.palette) * 2)
    dst = _pixel_image(64, 64, device)
    dst.copy_palette_from(src)
    assert torch.equal(dst.palette, src.palette)
    # spatial params untouched (still the constructor zeros)
    assert dst.tensor.abs().sum() == 0


def test_copy_palette_from_fails_loud_on_shape_mismatch():
    device = default_device()
    src = _pixel_image(32, 32, device)
    dst = PixelImage(32, 32, 1, palette_size=5, n_palettes=2, device=device)
    with pytest.raises(ValueError, match="palette_size"):
        dst.copy_palette_from(src)


def test_locked_encode_fit_preserves_the_carried_palette_exactly():
    """The stage-2 encode fit with lock_palette(True): pixels are re-fit
    against exactly the carried colors, and the raw palette never moves."""
    device = default_device()
    src = _pixel_image(32, 32, device)
    with torch.no_grad():
        src.palette.set_(torch.rand_like(src.palette) * 2)
    src.encode_random()

    dst = _pixel_image(64, 64, device)
    dst.copy_palette_from(src)
    dst.lock_palette(True)
    upscaled = src.decode_image().resize((64, 64), Image.BICUBIC)
    dst.encode_image(upscaled)  # the 201-step fit, palette locked
    dst.lock_palette(False)

    assert not dst.use_palette_target
    assert torch.equal(dst.palette, src.palette)
    assert torch.allclose(dst.sort_palette(), src.sort_palette())


def test_configure_pass_carry_release_matches_config_lock():
    """End-to-end through workhorse.configure_pass: unlocked config releases
    the palette after the fit; lock_palette=true keeps it locked to the
    carried colors."""
    from pytti.workhorse import configure_pass

    device = default_device()
    src = _pixel_image(16, 16, device)
    with torch.no_grad():
        src.palette.set_(torch.rand_like(src.palette) * 2)
    src.encode_random()
    init_pil = src.decode_image().resize((32, 32), Image.BICUBIC)

    def params(**overrides):
        cfg = OmegaConf.structured(ConfigSchema)
        cfg.scenes = "x"
        cfg.image_model = "Limited Palette"
        cfg.width = 32
        cfg.height = 32
        cfg.palette_size = 4
        cfg.palettes = 2
        cfg.hdr_weight = 0.0
        cfg.direct_init_weight = "2"
        for key, value in overrides.items():
            setattr(cfg, key, value)
        return cfg

    setup = configure_pass(
        params(), device, None, [[]], init_pil, None, False, palette_source=src
    )
    assert not setup.img.use_palette_target
    assert torch.equal(setup.img.palette, src.palette)
    # the weight-2 hold was built
    assert len(setup.init_augs) == 1

    setup_locked = configure_pass(
        params(lock_palette=True),
        device,
        None,
        [[]],
        init_pil,
        None,
        False,
        palette_source=src,
    )
    assert setup_locked.img.use_palette_target
    assert torch.allclose(setup_locked.img.palette_target, src.sort_palette())


# ---------------------------------------------------------------------------
# frame-numbering continuity across the two stage windows (no render)
# ---------------------------------------------------------------------------


class _StubImage(nn.Module):
    lr = 0.1

    def __init__(self):
        super().__init__()
        self.p = nn.Parameter(torch.zeros(1))


class _RecordingGuide(DirectImageGuide):
    """train() replays a scripted TOTAL loss; _save_frame records the save
    slot n exactly as the real save path computes it."""

    def __init__(self, params, script=(1.0,)):
        super().__init__(_StubImage(), None, params=params)
        self.script = list(script)
        self.train_calls = 0
        self.saved_slots = []

    def train(self, i, prompts, interp_prompts, loss_augs, **kwargs):
        value = self.script[min(self.train_calls, len(self.script) - 1)]
        self.train_calls += 1
        return {"TOTAL": torch.tensor(float(value))}

    def _save_frame(self, i):
        self.saved_slots.append((i + 1) // self.params.save_every)


def make_params(**overrides):
    cfg = OmegaConf.structured(ConfigSchema)
    cfg.scenes = "x"
    for key, value in overrides.items():
        setattr(cfg, key, value)
    return cfg


def run_two_stages(total_steps, save_every, script1=(0.9, 0.8, 0.7, 0.6), **overrides):
    """Issue exactly the two run_steps windows run_coarse_to_fine issues."""
    s1, s2 = stage_steps(total_steps)
    g1 = _RecordingGuide(
        make_params(save_every=save_every, steps_per_scene=s1, **overrides),
        script=script1,
    )
    ran1 = g1.run_steps(s1, [], [], [], i_offset=0, skipped_steps=0)
    g2 = _RecordingGuide(
        make_params(save_every=save_every, steps_per_scene=s2, **overrides)
    )
    ran2 = g2.run_steps(s2, [], [], [], i_offset=s1, skipped_steps=0)
    return g1, g2, ran1, ran2


def test_stage_windows_produce_one_contiguous_slot_sequence():
    g1, g2, ran1, ran2 = run_two_stages(50, 5)
    assert (ran1, ran2) == (20, 30)
    # stage 1 saves slots 1-4 (i=4,9,14,19), stage 2 slots 5-10 (i=24..49)
    assert g1.saved_slots == [1, 2, 3, 4]
    assert g2.saved_slots == [5, 6, 7, 8, 9, 10]


def test_stage_boundary_off_save_grid_still_contiguous():
    # 40/60 split of 33 = (13, 20); save_every 4 -> boundary mid-interval
    g1, g2, _, _ = run_two_stages(33, 4)
    assert g1.saved_slots + g2.saved_slots == list(range(1, 9))


def test_stage1_auto_stop_keeps_numbering_aligned():
    """A converged stage 1 re-saves its final slot and still returns the
    full stage width, so stage 2's offset (and its slots) are unchanged."""
    s1, s2 = stage_steps(50)  # 20, 30
    g1 = _RecordingGuide(
        make_params(
            save_every=5,
            steps_per_scene=s1,
            auto_stop=True,
            auto_stop_window=20,
            auto_stop_threshold=0.01,
        ),
        script=(1.0,),  # flat: converges at the first full window (step 20)
    )
    ran1 = g1.run_steps(s1, [], [], [], i_offset=0, skipped_steps=0)
    assert ran1 == s1  # scene-aligned return even though it stopped early
    # slots 1-4 from update(), then the converged state re-saves slot 4
    assert g1.saved_slots == [1, 2, 3, 4, 4]
    g2 = _RecordingGuide(make_params(save_every=5, steps_per_scene=s2))
    g2.run_steps(s2, [], [], [], i_offset=s1, skipped_steps=0)
    assert g2.saved_slots == [5, 6, 7, 8, 9, 10]


def test_resumed_stage2_window_continues_the_sequence():
    """Restore into stage 2 (i_restore=35 of 50): the resumed window saves
    exactly the remaining slots."""
    s1, s2 = stage_steps(50)  # 20, 30
    stage, done = resume_stage(35, s1)
    assert (stage, done) == (2, 15)
    g2 = _RecordingGuide(make_params(save_every=5, steps_per_scene=s2))
    g2.run_steps(s2 - done, [], [], [], i_offset=s1 + done, skipped_steps=done)
    assert g2.saved_slots == [8, 9, 10]


# ---------------------------------------------------------------------------
# live mlx_full smoke
# ---------------------------------------------------------------------------

CONFIG_BASE_PATH = "config"
CONFIG_DEFAULTS = "default.yaml"


def _gray(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("L"), dtype=np.float64)


def _pearson(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.corrcoef(a.ravel(), b.ravel())[0, 1])


@pytest.mark.download
def test_mlx_full_coarse_to_fine_live(tmp_path, monkeypatch):
    """512px Limited Palette c2f render on mlx_full: both stages complete,
    the frame sequence is contiguous and uniformly 512px, and the first
    stage-2 frame is pixel-correlated with the persisted stage-1 transition
    image (the weight-2 hold visibly derives stage 2 from stage 1)."""
    pytest.importorskip("mlx.core", reason="needs Apple Silicon + mlx")
    from pytti.workhorse import _hydra_main as render_frames

    monkeypatch.chdir(tmp_path)
    messages = []
    sink_id = logger.add(lambda m: messages.append(m), level="INFO")
    try:
        with initialize(config_path=CONFIG_BASE_PATH, version_base=None):
            cfg = compose(
                config_name=CONFIG_DEFAULTS,
                overrides=["conf=_test_coarse_to_fine"],
            )
            render_frames(cfg)
    finally:
        logger.remove(sink_id)

    assert any("coarse_to_fine stage 1/2: 256x256" in m for m in messages)
    assert any("coarse_to_fine stage 2/2: 512x512" in m for m in messages)

    frames_dir = tmp_path / "images_out" / "c2f_smoke"
    frames = sorted(frames_dir.glob("*.png"))
    indices = [int(f.stem.rsplit("_", 1)[1]) for f in frames]
    # steps 50, save_every 5: slots 1-4 from stage 1, 5-10 from stage 2
    assert indices == list(range(1, 11))
    for frame in frames:
        assert Image.open(frame).size == (512, 512), frame.name

    coarse_png = tmp_path / "backup" / "c2f_smoke" / "c2f_smoke_coarse.png"
    assert coarse_png.is_file()
    assert Image.open(coarse_png).size == (512, 512)

    # stage-2 first frame (4 steps past the transition) must be pixel-level
    # derived from the stage-1 result: a fresh random start would correlate
    # ~0 (independent init), so aligned r >> the spatial null (the same
    # frame against the ROTATED stage-1 image) is decisive evidence and also
    # rules out a degenerate everything-correlates image.
    # measured 2026-08-03 (seed 12345): aligned 0.938, null -0.001
    coarse = _gray(coarse_png)
    stage2_first = _gray(frames_dir / "c2f_smoke_0005.png")
    r_aligned = _pearson(stage2_first, coarse)
    r_null = _pearson(stage2_first, np.rot90(coarse, 2))
    assert r_aligned > 0.8, f"stage-2 start decorrelated: r={r_aligned:.3f}"
    assert r_aligned > r_null + 0.5, f"aligned {r_aligned:.3f} vs null {r_null:.3f}"
