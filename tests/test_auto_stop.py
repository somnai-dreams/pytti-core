"""
Convergence auto-stop.

Three layers:
- the plateau detector as a pure function against synthetic loss sequences
  (converging, noisy-converging, still-improving, plateau-then-improve);
- run_steps wiring against a scripted guide — no render, no embedder:
  sampling cadence, full-window guard, interp-ramp guard, per-scene window
  reset, scene-aligned return value, and the final-frame save slot;
- a live mlx_full smoke (`download` marker) where a trivially-convergent
  config stops early and saves its final frame.
"""

import math
from pathlib import Path

import pytest
import torch
from hydra import compose, initialize
from loguru import logger
from omegaconf import OmegaConf
from torch import nn

from pytti.config.structured_config import ConfigSchema
from pytti.ImageGuide import (
    AUTO_STOP_CHECK_INTERVAL,
    DirectImageGuide,
    plateau_improvement,
)

# ---------------------------------------------------------------------------
# the detector as a pure function
# ---------------------------------------------------------------------------


def first_stop_index(samples, window_samples, threshold):
    """Index (1-based sample count) at which the detector first fires."""
    for k in range(1, len(samples) + 1):
        improvement = plateau_improvement(samples[:k], window_samples)
        if improvement is not None and improvement < threshold:
            return k
    return None


def test_partial_window_is_never_judged():
    assert plateau_improvement([], 2) is None
    assert plateau_improvement([1.0], 2) is None
    assert plateau_improvement([1.0, 0.9, 0.8], 4) is None


def test_window_of_one_fails_loud():
    with pytest.raises(ValueError, match="window_samples"):
        plateau_improvement([1.0, 0.9], 1)


def test_converging_sequence_stops_after_transient():
    # decays to a nonzero floor: relative improvement dies out
    samples = [0.5 + 0.5 * math.exp(-k / 2) for k in range(40)]
    fired = first_stop_index(samples, 5, 0.002)
    assert fired is not None
    assert fired >= 5  # never before one full window
    # and not while the transient still dominates
    early = plateau_improvement(samples[:6], 5)
    assert early is not None and early > 0.002


def test_noisy_converging_sequence_still_stops():
    samples = [
        0.5 + 0.5 * math.exp(-k / 2) + 0.002 * math.sin(1.7 * k)
        for k in range(40)
    ]
    fired = first_stop_index(samples, 5, 0.002)
    assert fired is not None
    assert fired >= 5


def test_still_improving_sequence_never_stops():
    # steady linear descent: relative improvement stays well above threshold
    samples = [1.0 - 0.01 * k for k in range(50)]
    assert first_stop_index(samples, 5, 0.002) is None


def test_plateau_shorter_than_window_then_improve_does_not_stop():
    samples = [3.0, 2.7, 2.4, 2.1, 2.1, 2.1, 1.8, 1.5, 1.2]
    assert first_stop_index(samples, 4, 0.01) is None


def test_plateau_lasting_a_full_window_stops():
    samples = [3.0, 2.7, 2.4, 2.1, 2.1, 2.1, 2.1, 2.1]
    fired = first_stop_index(samples, 4, 0.01)
    assert fired == 7  # first index whose trailing 4 samples are all flat


def test_regressing_sequence_counts_as_plateaued():
    # loss going UP is "no longer converging" — improvement is negative
    improvement = plateau_improvement([1.0, 1.0, 1.1, 1.2], 4)
    assert improvement is not None and improvement < 0


# ---------------------------------------------------------------------------
# run_steps wiring (no render)
# ---------------------------------------------------------------------------


class _StubImage(nn.Module):
    lr = 0.1

    def __init__(self):
        super().__init__()
        self.p = nn.Parameter(torch.zeros(1))


class _ScriptedGuide(DirectImageGuide):
    """train() replays a scripted TOTAL-loss sequence; update() and
    _save_frame() only record their calls."""

    def __init__(self, params, script):
        super().__init__(_StubImage(), None, params=params)
        self.script = list(script)
        self.train_calls = 0
        self.saved_frames = []

    def update(self, i, stage_i):
        pass

    def train(self, i, prompts, interp_prompts, loss_augs, **kwargs):
        value = self.script[min(self.train_calls, len(self.script) - 1)]
        self.train_calls += 1
        return {"TOTAL": torch.tensor(float(value))}

    def _save_frame(self, i):
        self.saved_frames.append(i)


def make_params(**overrides):
    cfg = OmegaConf.structured(ConfigSchema)
    for key, value in overrides.items():
        setattr(cfg, key, value)
    return cfg


def make_guide(script, **overrides):
    overrides.setdefault("auto_stop", True)
    overrides.setdefault("auto_stop_window", 20)  # 2 samples
    overrides.setdefault("auto_stop_threshold", 0.01)
    overrides.setdefault("save_every", 10)
    return _ScriptedGuide(make_params(**overrides), script)


def test_flat_loss_stops_at_first_full_window_and_saves():
    guide = make_guide([1.0])
    ran = guide.run_steps(100, [], [], [])
    # samples at steps 10 and 20 -> converged at step 20
    assert guide.train_calls == 2 * AUTO_STOP_CHECK_INTERVAL
    # scene-aligned return: the caller's step counter advances to the cap
    assert ran == 100
    # stop at global step 19: slots 1,2 already written by update();
    # the converged state lands in the NEXT slot (frame 3, index 29)
    assert guide.saved_frames == [29]


def test_improving_loss_runs_to_the_cap():
    guide = make_guide([0.9**k for k in range(60)])
    ran = guide.run_steps(60, [], [], [])
    assert ran == 60
    assert guide.train_calls == 60
    assert guide.saved_frames == []


def test_never_stops_before_one_full_window():
    guide = make_guide([1.0], auto_stop_window=50)  # 5 samples
    ran = guide.run_steps(45, [], [], [])  # only 4 samples fit
    assert ran == 45
    assert guide.train_calls == 45
    assert guide.saved_frames == []
    guide = make_guide([1.0], auto_stop_window=50)
    guide.run_steps(100, [], [], [])
    assert guide.train_calls == 50  # fires exactly at the first full window


def test_never_stops_during_interp_ramp():
    guide = make_guide([1.0])
    guide.run_steps(100, [], [], [], interp_steps=25)
    # sampling starts at step 30 (first interval boundary past the ramp);
    # second sample at step 40 completes the window
    assert guide.train_calls == 40


def test_multi_scene_resets_the_window_and_stays_frame_aligned():
    guide = make_guide([1.0])
    ran1 = guide.run_steps(100, [], [], [], i_offset=0)
    ran2 = guide.run_steps(100, [], [], [], i_offset=100)
    assert (ran1, ran2) == (100, 100)
    # scene 2 needs its own full window (stops at its local step 20, global
    # 120), NOT at its first sample — the window did not carry over
    assert guide.train_calls == 40
    # scene 1 final frame: slot 3 (index 29); scene 2 stopped at global step
    # 119 -> slots 1..12 written, final frame slot 13 (index 129)
    assert guide.saved_frames == [29, 129]


def test_stop_past_last_scene_slot_resaves_that_slot():
    # cap 20, save_every 10: slots 1,2 only. Stop fires at step 20, after
    # slot 2 was written -> the converged state re-saves slot 2 (index 19)
    guide = make_guide([1.0])
    ran = guide.run_steps(20, [], [], [])
    assert ran == 20
    assert guide.saved_frames == [19]


def test_scene_shorter_than_save_interval_saves_nothing():
    guide = make_guide([1.0], save_every=50)
    guide.run_steps(20, [], [], [])
    assert guide.train_calls == 20
    assert guide.saved_frames == []


def test_too_short_window_fails_loud():
    guide = make_guide([1.0], auto_stop_window=AUTO_STOP_CHECK_INTERVAL)
    with pytest.raises(ValueError, match="auto_stop_window"):
        guide.run_steps(100, [], [], [])


def test_non_finite_threshold_fails_loud():
    guide = make_guide([1.0], auto_stop_threshold=float("nan"))
    with pytest.raises(ValueError, match="auto_stop_threshold"):
        guide.run_steps(100, [], [], [])


def test_auto_stop_off_by_default_and_legacy_stop_unchanged():
    guide = _ScriptedGuide(make_params(save_every=10), [1.0, 0.4, 0.3])
    ran = guide.run_steps(100, [], [], [], stop=0.5)
    assert ran == 2  # legacy stop keeps its actual-steps-run return
    assert guide.saved_frames == []


def test_bare_guide_params_none_is_unaffected():
    guide = _ScriptedGuide(None, [1.0])
    assert guide.run_steps(30, [], [], []) == 30
    assert guide.train_calls == 30


# ---------------------------------------------------------------------------
# live mlx_full smoke
# ---------------------------------------------------------------------------

CONFIG_BASE_PATH = "config"
CONFIG_DEFAULTS = "default.yaml"
FIXTURE_IMAGE = str(
    Path(__file__).parent / "fixtures" / "01-velo-header-seattle-needle.jpg"
)


@pytest.mark.download
def test_mlx_full_auto_stop_live(tmp_path, monkeypatch):
    """A strongly init-held still render converges, stops before the
    steps_per_scene cap, and saves a final frame past the last natural
    save slot."""
    pytest.importorskip("mlx.core", reason="needs Apple Silicon + mlx")
    from pytti.workhorse import _hydra_main as render_frames

    monkeypatch.chdir(tmp_path)
    messages = []
    sink_id = logger.add(lambda m: messages.append(m), level="INFO")
    try:
        with initialize(config_path=CONFIG_BASE_PATH, version_base=None):
            cfg = compose(
                config_name=CONFIG_DEFAULTS,
                overrides=[
                    "conf=_test_auto_stop",
                    f"init_image={FIXTURE_IMAGE}",
                ],
            )
            render_frames(cfg)
    finally:
        logger.remove(sink_id)

    assert any("auto_stop: converged" in m for m in messages), (
        "the render ran to the cap without converging"
    )

    frames = sorted((tmp_path / "images_out" / "auto_stop_smoke").glob("*.png"))
    # full-length run would write 150/10 = 15 frames; earliest possible stop
    # (one full 50-step window) writes 5 natural frames + the final frame
    assert 6 <= len(frames) < 15
    # contiguous frame numbers, so the final save landed in the next slot
    indices = [int(f.stem.rsplit("_", 1)[1]) for f in frames]
    assert indices == list(range(1, len(frames) + 1))
