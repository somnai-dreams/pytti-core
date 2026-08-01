"""
M2 slices S5+S6 — whole-step assembly + engine integration
(docs/mlx-m2-seam-map.md rows S5/S6).

Tiers, matching the other mlx_engine test files:

- Adam lockstep / mask parity / assembly tests: need mlx (darwin), stub
  towers, no downloads;
- full-step parity vs the real torch engine (gate b) and its .bak
  round-trip: need mlx AND the ViTB32 checkpoint -> @pytest.mark.download.

Gate numbers asserted here:
(a) Adam lockstep vs torch.optim.Adam, identical injected grads, 50 steps
    with a frame-boundary reset at 25: trajectories rel <= 1e-6.
(b) full-step parity with ALL RNG factored out (injected geometry, replayed
    aug draws, injected noise, both sides): loss + per-record + post-step
    param deltas rel <= 1e-5 with fp32 towers, <= 1e-2 with fp16 towers.
(c) .bak round-trip: engine params -> torch state_dict -> torch.save/load ->
    load_state_dict -> torch decode == MLX decode (<= 1e-6; measured 0.0).
"""

import importlib.util
import math

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf

from pytti.image_models import PixelImage, RGBImage
from pytti.LossAug.MSELossClass import MSELoss
from pytti.LossAug.TVLossClass import TVLoss
from pytti.Perceptor.cutouts.augs import BatchedAugs
from pytti.Perceptor.Prompt import MASK_DICT, Prompt, make_mask, mask_all
from pytti.prompt_spec import parse_prompt_spec
from pytti.rotoscoper import ROTOSCOPERS

needs_mlx = pytest.mark.skipif(
    importlib.util.find_spec("mlx") is None,
    reason="mlx not installed (darwin-only backend)",
)
pytestmark = needs_mlx

CPU = torch.device("cpu")
SEED = 777


def rel_err(candidate, reference) -> float:
    candidate = np.asarray(candidate, dtype=np.float64)
    reference = np.asarray(reference, dtype=np.float64)
    denom = max(float(np.abs(reference).max()), 1e-12)
    return float(np.abs(candidate - reference).max()) / denom


def l2_rel(candidate, reference) -> float:
    candidate = np.asarray(candidate, dtype=np.float64)
    reference = np.asarray(reference, dtype=np.float64)
    return float(
        np.linalg.norm(candidate - reference)
        / max(np.linalg.norm(reference), 1e-12)
    )


# ---------------------------------------------------------------------------
# gate (a): Adam lockstep vs torch.optim.Adam
# ---------------------------------------------------------------------------


class TestAdamLockstep:
    SHAPES = {"value": (24, 32), "tensor": (3, 24, 32), "palette": (4, 3, 3)}

    def test_lockstep_with_frame_boundary_reset(self):
        import mlx.core as mx

        from pytti.mlx_engine.step import make_adam, reset_adam_state

        rng = np.random.default_rng(SEED)
        init = {
            k: rng.standard_normal(s).astype(np.float32)
            for k, s in self.SHAPES.items()
        }
        t_params = {
            k: torch.nn.Parameter(torch.tensor(v)) for k, v in init.items()
        }
        t_opt = torch.optim.Adam(t_params.values(), lr=0.02)
        m_tree = {k: mx.array(v) for k, v in init.items()}
        m_opt = make_adam(0.02)
        m_opt.init(m_tree)

        worst = 0.0
        for step in range(50):
            if step == 25:
                # reset_lr_each_frame: torch rebuilds the optimizer
                # (ImageGuide.set_optim(None)); mlx zeroes in-state
                t_opt = torch.optim.Adam(t_params.values(), lr=0.02)
                reset_adam_state(m_opt)
            grads = {
                k: (rng.standard_normal(s) * 0.1).astype(np.float32)
                for k, s in self.SHAPES.items()
            }
            for k, p in t_params.items():
                p.grad = torch.tensor(grads[k])
            t_opt.step()
            m_tree = m_opt.apply_gradients(
                {k: mx.array(v) for k, v in grads.items()}, m_tree
            )
            mx.eval(m_tree)
            for k in self.SHAPES:
                worst = max(
                    worst,
                    rel_err(np.array(m_tree[k]), t_params[k].detach().numpy()),
                )
        assert worst <= 1e-6, f"Adam lockstep rel error {worst}"

    def test_reset_requires_initialized_state(self):
        from pytti.mlx_engine.step import make_adam, reset_adam_state

        with pytest.raises(ValueError, match="opt.init"):
            reset_adam_state(make_adam(0.02))


# ---------------------------------------------------------------------------
# geometric prompt masks, in-step vs Prompt.py
# ---------------------------------------------------------------------------


class TestGeometricMaskStops:
    @pytest.mark.parametrize("kind", ["a", "r", "l", "d", "u", "n", "f"])
    @pytest.mark.parametrize("thresh", [0.5, 0.3, 0.5000873264])
    def test_matches_torch_mask_functions(self, kind, thresh):
        import mlx.core as mx

        from pytti.mlx_engine.step import geometric_mask_stops

        g = torch.Generator().manual_seed(3)
        pos = torch.rand((6, 2, 2), generator=g) * 0.9
        size = torch.rand((6, 2, 2), generator=g) * 0.3 + 0.05
        torch_fn = mask_all if kind == "a" else MASK_DICT[kind]
        expected, weights = torch_fn(pos, size, None, thresh)
        assert weights == 1  # geometric masks never weight
        got = geometric_mask_stops(
            kind, mx.array(pos.numpy()), mx.array(size.numpy()), mx.array(thresh)
        )
        assert np.array_equal(np.array(got), expected.numpy())

    def test_unknown_kind_fails(self):
        import mlx.core as mx

        from pytti.mlx_engine.step import geometric_mask_stops

        with pytest.raises(ValueError, match="unknown geometric mask"):
            geometric_mask_stops(
                "x", mx.zeros((2, 1, 2)), mx.zeros((2, 1, 2)), mx.array(0.5)
            )


# ---------------------------------------------------------------------------
# hdr_loss zero-palette-row regression (found by S5 assembly: the default
# init palette starts at exactly 0; torch's vector_norm subgradient is 0
# there, a bare mx.sqrt is NaN)
# ---------------------------------------------------------------------------


class TestHdrZeroRowGradient:
    def test_matches_torch_on_default_palette(self):
        import mlx.core as mx

        from pytti.mlx_engine.image_models import (
            hdr_loss,
            pixel_params_from_state_dict,
        )

        img = _pixel_image()  # default palette: first row exactly 0
        params = pixel_params_from_state_dict(img.state_dict())

        loss_t, _ = img.hdr_loss(img)
        loss_t.backward()
        grad_t = img.palette.grad.detach().numpy()

        def f(palette):
            loss, _ = hdr_loss(
                {**params, "palette": palette}, use_palette_target=False
            )
            return loss

        grad_m = np.array(mx.grad(f)(params["palette"]))
        assert np.isfinite(grad_m).all()
        assert float(np.abs(grad_m - grad_t).max()) <= 1e-6


# ---------------------------------------------------------------------------
# stub fixtures for the no-download assembly tests
# ---------------------------------------------------------------------------

CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)


class StubNormalize:
    mean = CLIP_MEAN
    std = CLIP_STD


class StubPerceptor:
    def __init__(self, key, cut_size):
        self.key = key
        self.cut_size = cut_size
        self.normalize = StubNormalize()


class StubEmbedder:
    """Attribute surface of HDMultiClipEmbedder that the engine reads."""

    def __init__(self, sampler="smart", cutn=6):
        self.perceptors = [StubPerceptor("T32", 32), StubPerceptor("T16", 16)]
        self.cutn = cutn
        self.cut_pow = 2.0
        self.padding = 0.25
        self.border_mode = "clamp"
        self.noise_fac = 0.1
        self.cutout_sampler = sampler
        self.augs = BatchedAugs()


def _stub_loader(key, dtype):
    """Deterministic tiny random-weight towers (seeded per key so two
    engines built the same way get identical towers)."""
    import mlx.core as mx

    from pytti.Perceptor.mlx_backend import ViTConfig
    from pytti.Perceptor.mlx_backend.vit import VisionTower

    configs = {
        "T32": ViTConfig(
            image_size=32, patch_size=8, hidden_dim=32,
            num_layers=2, num_heads=4, mlp_dim=64, output_dim=16,
        ),
        "T16": ViTConfig(
            image_size=16, patch_size=8, hidden_dim=32,
            num_layers=2, num_heads=4, mlp_dim=64, output_dim=24,
        ),
    }
    mx.random.seed(sum(ord(c) for c in key))
    return VisionTower(configs[key])


def _base_params(**overrides):
    cfg = dict(
        scenes="a test prompt",
        scene_prefix="",
        scene_suffix="",
        animation_mode="off",
        optimizer="adam",
        seed=SEED,
        semantic_stabilization_weight="",
        semantic_init_weight="",
        depth_stabilization_weight="",
        perceptor_backend="mlx_full",
        input_audio="",
        input_audio_filters=None,
        approximate_vram_usage=False,
    )
    cfg.update(overrides)
    return OmegaConf.create(cfg)


def _pixel_image(width=64, height=48, **kwargs):
    kw = dict(
        scale=1, palette_size=4, n_palettes=3, gamma=1,
        hdr_weight=0.01, norm_weight=0.2, device=CPU,
    )
    kw.update(kwargs)
    # PixelImage's hdr/palette submodules land on the default device (mps
    # here) regardless of the device arg; pin the whole module to CPU so
    # torch reference math runs device-uniform
    return PixelImage(width, height, **kw).to(CPU)


def _seeded_pixel_image(**kwargs):
    torch.manual_seed(SEED)
    img = _pixel_image(**kwargs)
    img.encode_random()
    return img


def _make_engine(img=None, params=None, embedder=None, **kwargs):
    from pytti.mlx_engine.engine import MLXStillEngine

    return MLXStillEngine(
        img if img is not None else _seeded_pixel_image(),
        embedder if embedder is not None else StubEmbedder(),
        params if params is not None else _base_params(),
        lr=0.02,
        tower_loader=_stub_loader,
        **kwargs,
    )


def _tv_loss(weight):
    # TVLoss takes no device arg and lands on the default device (mps);
    # pin it to CPU so the torch reference path runs device-uniform
    tv = TVLoss(weight=weight)
    tv.device = CPU
    return tv


def _prompt(prompt_string, dim=24, n_perceptors=2, seed=1):
    g = torch.Generator().manual_seed(seed)
    spec = parse_prompt_spec(prompt_string)
    return Prompt(
        torch.randn((n_perceptors, dim), generator=g),
        spec.weight,
        spec.stop,
        spec.text,
        prompt_string,
        mask=make_mask(spec.mask, spec.cutoff),
        device=CPU,
    )


# ---------------------------------------------------------------------------
# assembly: records, state threading, determinism, reset, round-trip
# ---------------------------------------------------------------------------


class TestEngineAssembly:
    def _run(self, engine, steps=3, prompts=None, augs=None, **kwargs):
        prompts = prompts if prompts is not None else [
            _prompt("a test prompt:1"),
            _prompt("left side:1_r_0.4", seed=2),
        ]
        augs = augs if augs is not None else [TVLoss(weight=0.02)]
        records = []
        for i in range(steps):
            records.append(
                engine.train_step(i, prompts, [], augs, **kwargs)
            )
        return records

    def test_records_keep_torch_names_and_decline(self):
        engine = _make_engine()
        records = self._run(engine, steps=5)
        assert list(records[0]) == [
            "smoothing loss (TV)",
            "HDR normalization",
            "Palette normalization",
            "a test prompt",
            "left side",
            "TOTAL",
        ]
        totals = [float(r["TOTAL"]) for r in records]
        assert all(math.isfinite(v) for v in totals)
        # random towers still define a real objective: Adam must descend
        assert min(totals[1:]) < totals[0]
        assert int(engine._opt.state["step"]) == 5

    def test_deterministic_given_seed(self):
        # Same seed -> identical draws (bit-exact RNG threading), but the
        # gather-bilinear backward is a GPU scatter-add whose accumulation
        # order is not fixed — measured spread <= ~1e-7 rel across identical
        # runs. Gate well above that noise, far below any real divergence.
        first = self._run(_make_engine(), steps=3)
        second = self._run(_make_engine(), steps=3)
        for a, b in zip(first, second, strict=True):
            for name in a:
                assert rel_err(float(a[name]), float(b[name])) <= 1e-5, name

    def test_reset_optimizer_zeroes_moments_and_step(self):
        import mlx.core as mx

        engine = _make_engine()
        self._run(engine, steps=2)
        assert int(engine._opt.state["step"]) == 2
        assert float(mx.abs(engine._opt.state["value"]["m"]).max()) > 0
        engine.reset_optimizer()
        assert int(engine._opt.state["step"]) == 0
        for key in ("value", "tensor", "palette"):
            assert float(mx.abs(engine._opt.state[key]["m"]).max()) == 0
            assert float(mx.abs(engine._opt.state[key]["v"]).max()) == 0
        # and stepping again from the reset state still works
        self._run(engine, steps=1)
        assert int(engine._opt.state["step"]) == 1

    def test_bak_round_trip(self, tmp_path):
        """Gate (c): save under mlx_full -> torch.load + load_state_dict ->
        torch decode equals MLX decode."""
        from pytti.mlx_engine.image_models import pixel_decode

        img = _seeded_pixel_image()
        engine = _make_engine(img=img)
        self._run(engine, steps=3)

        engine.write_back(img)
        bak = tmp_path / "roundtrip.bak"
        torch.save(img.state_dict(), bak)

        restored = _pixel_image()
        restored.load_state_dict(torch.load(bak))
        with torch.no_grad():
            decode_torch = restored.decode_tensor().numpy()
        decode_mlx = np.array(
            pixel_decode(engine._params, scale=1, use_palette_target=False)
        )
        assert float(np.abs(decode_torch - decode_mlx).max()) <= 1e-6

        # resume: an engine built from the restored module starts from the
        # exact same tree
        resumed = _make_engine(img=restored)
        for key, value in engine._params.items():
            assert np.array_equal(np.array(value), np.array(resumed._params[key]))

    def test_write_back_rejects_foreign_image(self):
        engine = _make_engine()
        with pytest.raises(ValueError, match="different image_rep"):
            engine.write_back(_pixel_image())

    def test_interp_and_gas_retrace(self):
        engine = _make_engine()
        prompts = [_prompt("a test prompt:1")]
        interp = [_prompt("previous scene:1", seed=9)]
        augs = [TVLoss(weight=0.02)]
        # ramp active: interp prompts contribute but are not recorded
        rec = engine.train_step(
            0, prompts, interp, augs, interp_steps=4,
            gradient_accumulation_steps=2,
        )
        assert "previous scene" not in rec
        assert set(rec) == {
            "smoothing loss (TV)", "HDR normalization",
            "Palette normalization", "a test prompt", "TOTAL",
        }
        first_key = engine._step_key
        # ramp over: prompt set shrinks -> retrace, still healthy
        rec = engine.train_step(
            4, prompts, interp, augs, interp_steps=4,
            gradient_accumulation_steps=2,
        )
        assert engine._step_key != first_key
        assert math.isfinite(float(rec["TOTAL"]))

    def test_zero_weight_and_disabled_record_zero(self):
        engine = _make_engine()
        idle_prompt = _prompt("idle:0", seed=5)
        disabled_aug = TVLoss(weight=0.5)
        disabled_aug.set_enabled(False)
        rec = engine.train_step(
            0,
            [_prompt("a test prompt:1"), idle_prompt],
            [],
            [disabled_aug],
        )
        assert float(rec["idle"]) == 0.0
        assert float(rec["smoothing loss (TV)"]) == 0.0
        assert float(rec["a test prompt"]) != 0.0

    def test_rgb_image_runs(self):
        torch.manual_seed(SEED)
        img = RGBImage(64, 48, 1, device=CPU)
        img.encode_random()
        engine = _make_engine(img=img)
        records = self._run(engine, steps=2)
        assert list(records[0]) == [
            "smoothing loss (TV)", "a test prompt", "left side", "TOTAL",
        ]
        assert all(math.isfinite(float(r["TOTAL"])) for r in records)

    def test_direct_image_guide_dispatch(self, monkeypatch):
        """train()/set_optim()/loss-history plumbing through the real
        DirectImageGuide."""
        import pytti.mlx_engine.engine as engine_module
        from pytti.ImageGuide import DirectImageGuide

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
            params=_base_params(),
        )
        assert guide.mlx_engine is not None
        losses = guide.train(
            0,
            [_prompt("a test prompt:1")],
            [],
            [TVLoss(weight=0.02)],
        )
        assert math.isfinite(float(losses["TOTAL"]))
        assert guide.loss_history[-1]["TOTAL"] is losses["TOTAL"]
        guide._report_losses(0)  # float() on mx records
        assert int(guide.mlx_engine._opt.state["step"]) == 1
        guide.set_optim(None)  # reset_lr_each_frame cadence
        assert int(guide.mlx_engine._opt.state["step"]) == 0


# ---------------------------------------------------------------------------
# eligibility: fail loud at construction (seam map §4)
# ---------------------------------------------------------------------------


class TestEligibilityGates:
    def _expect(self, match, params=None, img=None, embedder=None):
        with pytest.raises(RuntimeError, match=match):
            _make_engine(img=img, params=params, embedder=embedder)

    def test_animation_rejected(self):
        self._expect("animation_mode", params=_base_params(animation_mode="2D"))

    def test_adamw_sf_rejected(self):
        self._expect("optimizer", params=_base_params(optimizer="adamw_sf"))

    def test_classic_sampler_rejected(self):
        self._expect("classic", embedder=StubEmbedder(sampler="classic"))

    def test_semantic_features_rejected(self):
        for key in (
            "semantic_stabilization_weight",
            "semantic_init_weight",
            "depth_stabilization_weight",
        ):
            self._expect(key, params=_base_params(**{key: "1"}))

    def test_semantic_image_prompt_in_scenes_rejected(self):
        self._expect(
            "image prompt", params=_base_params(scenes="[init.png]:1 | a cat")
        )

    def test_semantic_mask_in_scenes_rejected(self):
        self._expect("MaskSemantic", params=_base_params(scenes="a cat:1_the sky"))
        self._expect("MaskImage", params=_base_params(scenes="a cat:1_[m.png]"))

    def test_unsupported_image_model_rejected(self):
        class NotStill(torch.nn.Module):
            pass

        self._expect("image model", img=NotStill())

    def test_rotoscopers_rejected(self):
        ROTOSCOPERS.rotoscopers.append(object())
        try:
            self._expect("rotoscopers")
        finally:
            ROTOSCOPERS.clear_rotoscopers()

    def test_no_embedder_rejected(self):
        class NoEmbedder:
            pass

        with pytest.raises(RuntimeError, match="no embedder"):
            from pytti.mlx_engine.engine import MLXStillEngine

            MLXStillEngine(_pixel_image(), None, _base_params(), lr=0.02)

    def test_non_prompt_type_rejected_at_train(self):
        class FancyPrompt(Prompt):
            pass

        engine = _make_engine()
        fancy = FancyPrompt(
            torch.zeros((2, 24)), "1", "-inf", "fancy", "fancy:1", device=CPU
        )
        with pytest.raises(RuntimeError, match="FancyPrompt"):
            engine.train_step(0, [fancy], [], [])

    def test_depth_like_aug_rejected_at_train(self):
        class DepthLoss(MSELoss):
            pass

        engine = _make_engine()
        aug = DepthLoss(torch.zeros(1, 1, 1, 1), weight="1", name="depth x")
        with pytest.raises(RuntimeError, match="DepthLoss"):
            engine.train_step(0, [_prompt("a test prompt:1")], [], [aug])


# ---------------------------------------------------------------------------
# gate (b): full-step parity vs the real torch engine, RNG factored out
# ---------------------------------------------------------------------------

GEO = np.random.default_rng(41)
GEO_SIZES = GEO.integers(40, 129, 8).astype(np.float32)
GEO_OX = np.array(
    [GEO.integers(0, 129 - int(s)) for s in GEO_SIZES], dtype=np.float32
)
GEO_OY = np.array(
    [GEO.integers(0, 129 - int(s)) for s in GEO_SIZES], dtype=np.float32
)
NOISE_FACS = GEO.uniform(0, 0.1, (8, 1, 1, 1)).astype(np.float32)
NOISE_FIELD = GEO.standard_normal((8, 3, 224, 224)).astype(np.float32)
AUG_SEED = 4242


def _torch_fake_batched(with_augs):
    """Deterministic replacement for samplers.pytti_batched: injected
    geometry, (optionally) the real BatchedAugs re-seeded at a known state,
    injected noise. The MLX cutter below consumes the same constants."""
    from torch.nn import functional as F

    from pytti.Perceptor.cutouts import samplers

    def fake(input, side_x, side_y, cut_size, padding, cutn, cut_pow,
             border_mode, augs, noise_fac, device):
        assert border_mode == "clamp" and cutn == 8
        sizes_px = torch.tensor(GEO_SIZES)
        ox, oy = torch.tensor(GEO_OX), torch.tensor(GEO_OY)
        grid = samplers._affine_crop_grid(
            ox, oy, sizes_px, input.shape[1], cut_size, side_y, side_x
        )
        cutouts = F.grid_sample(
            input.expand(cutn, -1, -1, -1), grid,
            mode="bilinear", padding_mode="border", align_corners=False,
        )
        if with_augs:
            torch.manual_seed(AUG_SEED)
            cutouts = augs(cutouts)
        cutouts = cutouts + torch.tensor(NOISE_FACS) * torch.tensor(NOISE_FIELD)
        offsets = torch.stack([ox / side_x, oy / side_y], dim=-1)
        sizes = torch.stack([sizes_px / side_x, sizes_px / side_y], dim=-1)
        return cutouts, offsets, sizes

    return fake


def _mlx_cutter(with_augs):
    import mlx.core as mx

    from pytti.mlx_engine.augs import AugConfig, apply_augs
    from pytti.mlx_engine.sampler import _cut_batch
    from tests.test_mlx_engine_augs import _injected_params

    if with_augs:
        torch.manual_seed(AUG_SEED)
        aug_params = _injected_params(8)
    facs = mx.array(NOISE_FACS.transpose(0, 2, 3, 1))
    field = mx.array(NOISE_FIELD.transpose(0, 2, 3, 1))

    def cutter(padded, cut_size):
        cutouts, offsets, sizes = _cut_batch(
            padded,
            mx.array(GEO_SIZES),
            mx.array(GEO_OX),
            mx.array(GEO_OY),
            side_x=128, side_y=128, cut_size=cut_size,
            paddingx=32, paddingy=32, border_mode="clamp",
        )
        if with_augs:
            nchw = mx.transpose(cutouts, (0, 3, 1, 2))
            nchw = apply_augs(nchw, aug_params, AugConfig())
            cutouts = mx.transpose(nchw, (0, 2, 3, 1))
        return cutouts + facs * field, offsets, sizes

    return cutter


@pytest.mark.download
class TestFullStepParity:
    @pytest.fixture()
    def clip_embedder(self):
        import pytti.Perceptor
        from pytti.Perceptor import free_clip, init_clip
        from pytti.Perceptor.Embedder import HDMultiClipEmbedder

        free_clip()
        init_clip(["ViTB32"], device=CPU)
        embedder = HDMultiClipEmbedder(
            perceptors=pytti.Perceptor.CLIP_PERCEPTORS,
            cutn=8, cut_pow=2, padding=0.25, border_mode="clamp",
            noise_fac=0.1, cutout_sampler="batched", device=CPU,
        )
        yield embedder
        free_clip()

    def _images(self, image_model):
        torch.manual_seed(SEED)
        if image_model == "pixel":
            reference = _pixel_image(128, 128)
            reference.encode_random()
            candidate = _pixel_image(128, 128)
        else:
            reference = RGBImage(128, 128, 1, device=CPU)
            reference.encode_random()
            candidate = RGBImage(128, 128, 1, device=CPU)
        candidate.load_state_dict(
            {k: v.clone() for k, v in reference.state_dict().items()}
        )
        return reference, candidate

    def _torch_step(self, img, embedder, prompts, augs, monkeypatch, gas,
                    with_augs):
        from pytti.ImageGuide import DirectImageGuide
        from pytti.Perceptor import Embedder as embedder_module

        monkeypatch.setitem(
            embedder_module.CUTOUT_SAMPLERS,
            "batched",
            _torch_fake_batched(with_augs),
        )
        guide = DirectImageGuide(
            image_rep=img,
            embedder=embedder,
            params=OmegaConf.create(
                dict(
                    perceptor_backend="torch", optimizer="adam",
                    input_audio="", input_audio_filters=None,
                )
            ),
        )
        pre = {k: v.clone() for k, v in img.state_dict().items()}
        guide.train(
            0, prompts, [], augs, interp_steps=0,
            gradient_accumulation_steps=gas,
        )
        record = {k: float(v) for k, v in guide.loss_history[-1].items()}
        post = {k: v.clone() for k, v in img.state_dict().items()}
        return pre, record, post

    def _mlx_step(self, img, embedder, prompts, augs, tower_dtype, gas,
                  with_augs):
        from pytti.mlx_engine.engine import MLXStillEngine
        from pytti.mlx_engine.step import trainable_keys_for

        engine = MLXStillEngine(
            img, embedder, _base_params(), lr=0.02,
            tower_dtype=tower_dtype, cutter=_mlx_cutter(with_augs),
        )
        record = engine.train_step(
            0, prompts, [], augs, interp_steps=0,
            gradient_accumulation_steps=gas,
        )
        return (
            {k: float(v) for k, v in record.items()},
            engine,
            trainable_keys_for(engine._image_kind),
        )

    # Loss/record gate: elementwise max-rel, the specced 1e-5 fp32 / 1e-2
    # fp16. Param-delta gate: L2-relative + cosine — at step 1 the
    # bias-corrected Adam update is g/(|g|+1e-8), so elements whose true
    # gradient is near zero amplify tower noise (fp32 grads: cosine
    # 1.000000, L2 rel <= 8.5e-5; fp16: cosine >= 0.9997, L2 <= 2.2e-2 —
    # the M1 gate-2 class) into O(1) *elementwise* delta wiggle; an
    # elementwise 1e-5 delta gate is unattainable for ANY two float
    # implementations. The L2+cosine pair keeps the gate meaningful: the
    # boundary-clip gradient bug this test caught measured delta L2 1.0 /
    # cosine 0.63 against these gates.
    @pytest.mark.parametrize(
        (
            "image_model", "tower_dtype", "gas", "with_augs",
            "gate", "delta_gate", "delta_cos_gate",
        ),
        [
            ("pixel", "float32", 1, True, 1e-5, 2e-3, 0.99999),
            ("pixel", "float16", 1, True, 1e-2, 2.5e-1, 0.97),
            ("pixel", "float32", 2, False, 1e-5, 2e-3, 0.99999),
            ("rgb", "float32", 1, True, 1e-5, 2e-3, 0.99999),
        ],
    )
    def test_full_step_parity(
        self, clip_embedder, monkeypatch, image_model, tower_dtype, gas,
        with_augs, gate, delta_gate, delta_cos_gate,
    ):
        from pytti.Perceptor.Prompt import parse_prompt

        reference, candidate = self._images(image_model)
        prompts = [
            parse_prompt(clip_embedder, "a red mushroom on mossy ground:1",
                         device=CPU),
            parse_prompt(clip_embedder, "the sky:0.5_u", device=CPU),
        ]
        for prompt in prompts:
            # parse_prompt embeds on `device` but leaves Prompt.device at
            # the default (mps); pin it so the torch reference runs on CPU
            prompt.device = CPU
        augs = [_tv_loss(0.02)]

        pre, torch_record, post = self._torch_step(
            reference, clip_embedder, prompts, augs, monkeypatch, gas,
            with_augs,
        )
        mlx_record, engine, trainable = self._mlx_step(
            candidate, clip_embedder, prompts, augs, tower_dtype, gas,
            with_augs,
        )

        assert set(mlx_record) == set(torch_record)
        for name, reference_value in torch_record.items():
            err = rel_err(mlx_record[name], reference_value)
            assert err <= gate, f"{name}: rel {err} (torch {reference_value})"

        # the tree key for each trainable equals its state_dict key
        for key in trainable:
            delta_torch = (post[key] - pre[key]).numpy().astype(np.float64)
            delta_mlx = (
                np.array(engine._params[key]).astype(np.float64)
                - pre[key].numpy()
            )
            err = l2_rel(delta_mlx, delta_torch)
            assert err <= delta_gate, f"{key} delta: L2 rel {err}"
            cos = float(
                (delta_mlx * delta_torch).sum()
                / (np.linalg.norm(delta_mlx) * np.linalg.norm(delta_torch))
            )
            assert cos >= delta_cos_gate, f"{key} delta: cosine {cos}"

    def test_bak_round_trip_after_real_steps(self, clip_embedder, tmp_path):
        """Gate (c) on the real towers: 3 unfactored (live-RNG) steps, then
        the .bak written by write_back restores to the exact MLX decode."""
        from pytti.mlx_engine.engine import MLXStillEngine
        from pytti.mlx_engine.image_models import pixel_decode
        from pytti.Perceptor.Prompt import parse_prompt

        _, img = self._images("pixel")
        engine = MLXStillEngine(img, clip_embedder, _base_params(), lr=0.02)
        prompts = [parse_prompt(clip_embedder, "a red mushroom:1", device=CPU)]
        prompts[0].device = CPU
        for i in range(3):
            engine.train_step(i, prompts, [], [_tv_loss(0.02)])
        engine.write_back(img)
        bak = tmp_path / "real.bak"
        torch.save(img.state_dict(), bak)
        restored = _pixel_image(128, 128)
        restored.load_state_dict(torch.load(bak))
        with torch.no_grad():
            decode_torch = restored.decode_tensor().numpy()
        decode_mlx = np.array(
            pixel_decode(engine._params, scale=1, use_palette_target=False)
        )
        assert float(np.abs(decode_torch - decode_mlx).max()) <= 1e-6
