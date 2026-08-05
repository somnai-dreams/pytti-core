"""
Coarse-to-fine still rendering (2..4 pyramid stages).

Four layers:
- the stage math (the geometric dims ladder with /8 rounding + the 64
  floor, the per-stage-count step-split table, restore->stage mapping,
  thumbnail-stage detection) and config validation as pure functions;
- the Limited Palette carry: copy_palette_from mechanics and the full
  configure_pass carry (lock through the encode fit, release after);
- frame-numbering continuity across the N run_steps windows the
  orchestrator issues — scripted guides, no render — including a stage-1
  auto_stop convergence;
- live mlx_full smokes (`download` marker): a 512px stages=2 run with the
  stage-2 first frame pixel-correlated against the persisted transition
  image (not eyeballed), and a 256px stages=3 run with frames from all
  three stages, the stage boundaries in the log, and the thumbnail-stage
  sampler forcing visible for exactly the sub-perceptor-size stages.
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
    is_thumbnail_stage,
    resume_stage,
    stage_dims,
    stage_steps,
    validate_coarse_stages,
    validate_coarse_to_fine,
)
from pytti.config.structured_config import ConfigSchema
from pytti.device import default_device
from pytti.image_models.pixel import PixelImage
from pytti.ImageGuide import DirectImageGuide

# ---------------------------------------------------------------------------
# stage math (pure)
# ---------------------------------------------------------------------------


def test_stage_dims_two_stage_ladder_halves_and_stays_on_8():
    assert stage_dims(512, 512, 2) == ((256, 256), (512, 512))
    assert stage_dims(448, 256, 2) == ((224, 128), (448, 256))
    # non-/16 dims round half-up to the nearest multiple of 8
    assert stage_dims(300, 300, 2) == ((152, 152), (300, 300))  # 150 -> 152
    assert stage_dims(200, 200, 2) == ((104, 104), (200, 200))  # 100 -> 104
    assert stage_dims(432, 432, 2) == ((216, 216), (432, 432))  # exact


def test_stage_dims_ladder_is_geometric_ending_at_full():
    assert stage_dims(512, 512, 3) == ((128, 128), (256, 256), (512, 512))
    assert stage_dims(512, 512, 4) == (
        (64, 64),
        (128, 128),
        (256, 256),
        (512, 512),
    )
    # non-square: each dim scales independently
    assert stage_dims(512, 256, 3) == ((128, 64), (256, 128), (512, 256))


def test_stage_dims_final_stage_is_the_exact_configured_dims():
    # the final stage is never /8-rounded — it IS the configured canvas
    assert stage_dims(300, 300, 3)[-1] == (300, 300)
    assert stage_dims(444, 300, 4)[-1] == (444, 300)


def test_stage_dims_intermediate_stages_round_to_8():
    for dims in stage_dims(444, 300, 4)[:-1]:
        assert dims[0] % 8 == 0 and dims[1] % 8 == 0


def test_stage_dims_floors_at_64():
    assert stage_dims(100, 100, 2)[0] == (64, 64)  # 50 -> 48 -> floor
    assert stage_dims(128, 128, 2)[0] == (64, 64)
    assert stage_dims(512, 100, 2)[0] == (256, 64)  # per-dim
    # deep ladders floor their early stages: 256/8 = 32 -> 64 and
    # 256/4 = 64 both land on the floor
    assert stage_dims(256, 256, 4) == (
        (64, 64),
        (64, 64),
        (128, 128),
        (256, 256),
    )


def test_stage_dims_never_exceeds_the_configured_dim():
    # canvases at/below the floor run early stages at full size, never
    # upscaled
    assert stage_dims(64, 64, 3) == ((64, 64), (64, 64), (64, 64))
    assert stage_dims(48, 48, 2) == ((48, 48), (48, 48))


def test_stage_dims_fails_loud_on_bad_inputs():
    with pytest.raises(ValueError, match="positive"):
        stage_dims(0, 512, 2)
    with pytest.raises(ValueError, match="positive"):
        stage_dims(512, -1, 3)
    with pytest.raises(ValueError, match="coarse_stages"):
        stage_dims(512, 512, 1)
    with pytest.raises(ValueError, match="coarse_stages"):
        stage_dims(512, 512, 5)


def test_stage_steps_two_stage_split_is_40_60():
    assert stage_steps(100, 2) == (40, 60)
    assert stage_steps(50, 2) == (20, 30)
    assert stage_steps(13, 2) == (5, 8)  # 40% floors
    assert stage_steps(6, 2) == (2, 4)  # smallest legal split


def test_stage_steps_table_earlier_stages_cheaper():
    assert stage_steps(100, 3) == (25, 25, 50)
    assert stage_steps(100, 4) == (15, 20, 25, 40)
    assert stage_steps(48, 3) == (12, 12, 24)
    # floors go to the early stages, the remainder to the final stage
    assert stage_steps(13, 4) == (1, 2, 3, 7)


def test_stage_steps_sum_is_total_and_every_stage_runs():
    for n_stages in (2, 3, 4):
        for total in range(3 * n_stages, 500):
            splits = stage_steps(total, n_stages)
            assert len(splits) == n_stages
            assert sum(splits) == total
            assert all(s >= 1 for s in splits)


def test_stage_steps_fails_loud_below_the_per_stage_minimum():
    # guard: steps_per_scene >= 3 * coarse_stages
    for n_stages in (2, 3, 4):
        with pytest.raises(ValueError, match="steps_per_scene"):
            stage_steps(3 * n_stages - 1, n_stages)


def test_stage_steps_fails_loud_on_bad_stage_count():
    with pytest.raises(ValueError, match="coarse_stages"):
        stage_steps(100, 1)
    with pytest.raises(ValueError, match="coarse_stages"):
        stage_steps(100, 5)


def test_resume_stage_maps_restored_steps_onto_stages():
    splits = stage_steps(50, 2)  # (20, 30)
    assert resume_stage(0, splits) == (1, 0)
    assert resume_stage(15, splits) == (1, 15)
    # a boundary resumes as the earlier stage with nothing left: the .bak
    # there was written by that stage (its dims) and the transition replays
    assert resume_stage(20, splits) == (1, 20)
    assert resume_stage(25, splits) == (2, 5)
    assert resume_stage(50, splits) == (2, 30)


def test_resume_stage_generalizes_to_the_full_ladder():
    splits = stage_steps(48, 3)  # (12, 12, 24)
    assert resume_stage(0, splits) == (1, 0)
    assert resume_stage(12, splits) == (1, 12)  # boundary -> earlier stage
    assert resume_stage(13, splits) == (2, 1)
    assert resume_stage(24, splits) == (2, 12)  # boundary -> earlier stage
    assert resume_stage(25, splits) == (3, 1)
    assert resume_stage(48, splits) == (3, 24)
    # past the end: the final stage absorbs it (its window runs 0 steps)
    assert resume_stage(60, splits) == (3, 36)


def test_resume_stage_fails_loud_on_bad_inputs():
    with pytest.raises(ValueError, match=">= 0"):
        resume_stage(-1, (20, 30))
    with pytest.raises(ValueError, match="splits"):
        resume_stage(0, (20,))
    with pytest.raises(ValueError, match="splits"):
        resume_stage(0, (20, 0))


def test_is_thumbnail_stage_short_side_at_or_below_the_tower_input():
    assert is_thumbnail_stage((64, 64), 224)
    assert is_thumbnail_stage((224, 224), 224)  # equal counts: crops still
    # upsample or cover the whole frame
    assert is_thumbnail_stage((512, 100), 224)  # short side decides
    assert not is_thumbnail_stage((256, 256), 224)
    assert not is_thumbnail_stage((225, 512), 224)


def test_is_thumbnail_stage_fails_loud_on_bad_inputs():
    with pytest.raises(ValueError, match="positive"):
        is_thumbnail_stage((0, 512), 224)
    with pytest.raises(ValueError, match="max_cut_size"):
        is_thumbnail_stage((512, 512), 0)


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


def test_validate_coarse_stages_accepts_the_meaningful_combinations():
    validate_coarse_stages(coarse_to_fine=False, coarse_stages=2)  # default
    for n_stages in (2, 3, 4):
        validate_coarse_stages(coarse_to_fine=True, coarse_stages=n_stages)


def test_validate_coarse_stages_rejects_an_inert_setting():
    # a non-default stage count on a run that ignores it is a config lie
    with pytest.raises(ValueError, match="coarse_to_fine"):
        validate_coarse_stages(coarse_to_fine=False, coarse_stages=3)


def test_validate_coarse_stages_rejects_out_of_range():
    for bad in (0, 1, 5):
        with pytest.raises(ValueError, match="coarse_stages"):
            validate_coarse_stages(coarse_to_fine=True, coarse_stages=bad)


def test_schema_rejects_coarse_stages_without_coarse_to_fine():
    cfg = OmegaConf.structured(ConfigSchema)
    cfg.scenes = "x"
    cfg.coarse_stages = 3
    with pytest.raises(ValueError, match="coarse_to_fine"):
        OmegaConf.to_object(cfg)
    cfg.coarse_to_fine = True
    assert OmegaConf.to_object(cfg).coarse_stages == 3


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
    """The next stage's encode fit with lock_palette(True): pixels are re-fit
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
# frame-numbering continuity across the stage windows (no render)
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


def run_stage_windows(total_steps, save_every, coarse_stages, **overrides):
    """Issue exactly the run_steps windows run_coarse_to_fine issues (one
    per stage, each at the previous stages' cumulative global offset)."""
    splits = stage_steps(total_steps, coarse_stages)
    guides = []
    offset = 0
    for split in splits:
        guide = _RecordingGuide(
            make_params(save_every=save_every, steps_per_scene=split, **overrides)
        )
        ran = guide.run_steps(split, [], [], [], i_offset=offset, skipped_steps=0)
        assert ran == split
        guides.append(guide)
        offset += split
    return guides


def test_stage_windows_produce_one_contiguous_slot_sequence():
    g1, g2 = run_stage_windows(50, 5, 2)
    # stage 1 saves slots 1-4 (i=4,9,14,19), stage 2 slots 5-10 (i=24..49)
    assert g1.saved_slots == [1, 2, 3, 4]
    assert g2.saved_slots == [5, 6, 7, 8, 9, 10]


def test_three_stage_windows_produce_one_contiguous_slot_sequence():
    # splits(48, 3) = (12, 12, 24); save_every 4 -> 12 slots, boundaries
    # exactly on the save grid (i=11 -> slot 3, i=23 -> slot 6)
    g1, g2, g3 = run_stage_windows(48, 4, 3)
    assert g1.saved_slots == [1, 2, 3]
    assert g2.saved_slots == [4, 5, 6]
    assert g3.saved_slots == [7, 8, 9, 10, 11, 12]


def test_stage_boundaries_off_save_grid_still_contiguous():
    # 40/60 split of 33 = (13, 20); save_every 4 -> boundary mid-interval
    g1, g2 = run_stage_windows(33, 4, 2)
    assert g1.saved_slots + g2.saved_slots == list(range(1, 9))
    # 25/25/50 of 50 = (12, 12, 26); save_every 5 -> both boundaries (12,
    # 24) land mid-interval
    slots = []
    for guide in run_stage_windows(50, 5, 3):
        slots += guide.saved_slots
    assert slots == list(range(1, 11))


def test_stage1_auto_stop_keeps_numbering_aligned():
    """A converged stage 1 re-saves its final slot and still returns the
    full stage width, so later stages' offsets (and slots) are unchanged."""
    splits = stage_steps(50, 2)  # (20, 30)
    g1 = _RecordingGuide(
        make_params(
            save_every=5,
            steps_per_scene=splits[0],
            auto_stop=True,
            auto_stop_window=20,
            auto_stop_threshold=0.01,
        ),
        script=(1.0,),  # flat: converges at the first full window (step 20)
    )
    ran1 = g1.run_steps(splits[0], [], [], [], i_offset=0, skipped_steps=0)
    assert ran1 == splits[0]  # scene-aligned return even though it stopped
    # slots 1-4 from update(), then the converged state re-saves slot 4
    assert g1.saved_slots == [1, 2, 3, 4, 4]
    g2 = _RecordingGuide(make_params(save_every=5, steps_per_scene=splits[1]))
    g2.run_steps(splits[1], [], [], [], i_offset=splits[0], skipped_steps=0)
    assert g2.saved_slots == [5, 6, 7, 8, 9, 10]


def test_resumed_stage2_window_continues_the_sequence():
    """Restore into stage 2 (i_restore=35 of 50): the resumed window saves
    exactly the remaining slots."""
    splits = stage_steps(50, 2)  # (20, 30)
    stage, done = resume_stage(35, splits)
    assert (stage, done) == (2, 15)
    g2 = _RecordingGuide(make_params(save_every=5, steps_per_scene=splits[1]))
    g2.run_steps(
        splits[1] - done, [], [], [], i_offset=splits[0] + done, skipped_steps=done
    )
    assert g2.saved_slots == [8, 9, 10]


def test_resumed_stage3_window_continues_the_sequence():
    """Restore into the FINAL stage of a 3-stage ladder (i_restore=32 of
    48): the resumed window saves exactly the remaining slots and the
    stage-3 offset generalizes."""
    splits = stage_steps(48, 3)  # (12, 12, 24)
    stage, done = resume_stage(32, splits)
    assert (stage, done) == (3, 8)
    g3 = _RecordingGuide(make_params(save_every=4, steps_per_scene=splits[2]))
    g3.run_steps(
        splits[2] - done,
        [],
        [],
        [],
        i_offset=sum(splits[:2]) + done,
        skipped_steps=done,
    )
    # a full run saves slots 1-12; steps 0..31 already saved slots 1-8
    assert g3.saved_slots == [9, 10, 11, 12]


# ---------------------------------------------------------------------------
# live mlx_full smokes
# ---------------------------------------------------------------------------

CONFIG_BASE_PATH = "config"
CONFIG_DEFAULTS = "default.yaml"


def _gray(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("L"), dtype=np.float64)


def _pearson(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.corrcoef(a.ravel(), b.ravel())[0, 1])


def _render_live(tmp_path, monkeypatch, conf_name):
    pytest.importorskip("mlx.core", reason="needs Apple Silicon + mlx")
    from pytti.workhorse import _hydra_main as render_frames

    monkeypatch.chdir(tmp_path)
    messages = []
    sink_id = logger.add(lambda m: messages.append(m), level="INFO")
    try:
        with initialize(config_path=CONFIG_BASE_PATH, version_base=None):
            cfg = compose(
                config_name=CONFIG_DEFAULTS,
                overrides=[f"conf={conf_name}"],
            )
            render_frames(cfg)
    finally:
        logger.remove(sink_id)
    return messages


@pytest.mark.download
def test_mlx_full_coarse_to_fine_live(tmp_path, monkeypatch):
    """512px Limited Palette c2f render on mlx_full (default 2 stages): both
    stages complete, the frame sequence is contiguous and uniformly 512px,
    and the first stage-2 frame is pixel-correlated with the persisted
    stage-1 transition image (the weight-2 hold visibly derives stage 2
    from stage 1)."""
    messages = _render_live(tmp_path, monkeypatch, "_test_coarse_to_fine")

    assert any("coarse_to_fine stage 1/2: 256x256" in m for m in messages)
    assert any("coarse_to_fine stage 2/2: 512x512" in m for m in messages)

    frames_dir = tmp_path / "images_out" / "c2f_smoke"
    frames = sorted(frames_dir.glob("*.png"))
    indices = [int(f.stem.rsplit("_", 1)[1]) for f in frames]
    # steps 50, save_every 5: slots 1-4 from stage 1, 5-10 from stage 2
    assert indices == list(range(1, 11))
    for frame in frames:
        assert Image.open(frame).size == (512, 512), frame.name

    coarse_png = tmp_path / "backup" / "c2f_smoke" / "c2f_smoke_coarse_1.png"
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


@pytest.mark.download
def test_mlx_full_three_stage_pyramid_live(tmp_path, monkeypatch):
    """256px stages=3 pyramid on mlx_full: all three stages complete with
    frames on disk from each, the two transitions persist, and the
    thumbnail-stage sampler forcing fires for exactly the stages whose
    canvas is at or below the 224px tower input (64 and 128 — not 256)."""
    messages = _render_live(tmp_path, monkeypatch, "_test_coarse_stages3")

    # splits(48, 3) = (12, 12, 24); ladder = 64, 128, 256
    assert any("coarse_to_fine stage 1/3: 64x64" in m for m in messages)
    assert any("coarse_to_fine stage 2/3: 128x128" in m for m in messages)
    assert any("coarse_to_fine stage 3/3: 256x256" in m for m in messages)

    # thumbnail-stage sampler forcing: stages 1 and 2 only
    forcing = [m for m in messages if "forcing cutout_sampler=full" in m]
    assert any("stage 1/3" in m for m in forcing)
    assert any("stage 2/3" in m for m in forcing)
    assert not any("stage 3/3" in m for m in forcing)

    # frames: save_every 4 -> slots 1-3 (stage 1), 4-6 (stage 2), 7-12
    # (stage 3), all at the full 256px canvas
    frames_dir = tmp_path / "images_out" / "c2f_pyramid"
    frames = sorted(frames_dir.glob("*.png"))
    indices = [int(f.stem.rsplit("_", 1)[1]) for f in frames]
    assert indices == list(range(1, 13))
    for frame in frames:
        assert Image.open(frame).size == (256, 256), frame.name

    backup_dir = tmp_path / "backup" / "c2f_pyramid"
    seam1 = backup_dir / "c2f_pyramid_coarse_1.png"
    seam2 = backup_dir / "c2f_pyramid_coarse_2.png"
    assert seam1.is_file() and seam2.is_file()
    assert Image.open(seam1).size == (256, 256)
    assert Image.open(seam2).size == (256, 256)

    # stage 3's first frame (4 steps past the second transition) must be
    # pixel-level derived from the stage-2 result (weight-2 hold), and each
    # transition must derive from the previous one (the pyramid is one
    # continuous composition, not three renders).
    # measured 2026-08-05 (seed 12345): aligned 0.963, null 0.024,
    # seam-to-seam 0.931
    seam2_gray = _gray(seam2)
    stage3_first = _gray(frames_dir / "c2f_pyramid_0007.png")
    r_aligned = _pearson(stage3_first, seam2_gray)
    r_null = _pearson(stage3_first, np.rot90(seam2_gray, 2))
    assert r_aligned > 0.8, f"stage-3 start decorrelated: r={r_aligned:.3f}"
    assert r_aligned > r_null + 0.5, f"aligned {r_aligned:.3f} vs null {r_null:.3f}"
    r_seams = _pearson(_gray(seam1), seam2_gray)
    assert r_seams > 0.5, f"stage 2 abandoned stage 1: r={r_seams:.3f}"
