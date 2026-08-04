"""
coherence_weighting gates (structured_config.coherence_weighting).

Gate 1 — sizes_to_coherence_weights pure-function contract (torch, CPU):
    anchor rows (min-side size column == 1.0 exactly) weigh 3x before the
    mean normalization, detail rows scale by their inscribed-square
    fraction, mean(weights) == 1 (bit-exact for degenerate uniform batches;
    within 1 fp32 ulp for mixed batches — fp rounding makes bit-exact 1.0
    unattainable for arbitrary fractions), non-square canvas conventions,
    degenerate all-anchor / all-detail batches, and the REAL smart sampler's
    anchor population hitting the exact-1.0 path end to end.

Gate 3 (torch half) — Prompt.forward with coherence_canvas=None is
    BIT-EXACT against a verbatim copy of the pre-feature forward math, and
    the on-path equals the same math with the coherence weights composed
    multiplicatively after the mask weights (including tensor mask weights
    and LocationAwareMCIP's Hungarian row permutation).

Gate 3 (mlx_full half) — flag-off engine vs the pre-feature golden
    (tests/fixtures/coherence_off_golden_mlx.npz, generated at commit
    b839cfe by tests/fixtures/gen_coherence_off_golden.py): step-1 records
    bit-exact; later records / params gated at 10x the measured same-code
    rebuild determinism floor (MLX's gather-bilinear backward is a GPU
    scatter-add with unfixed accumulation order — records floor ~2e-7,
    params floor ~5e-5; see test_mlx_engine_step.test_deterministic_given
    _seed). Flag-on must move the semantic records and leave step-1
    direct-loss records bit-identical (the knob touches ONLY semantic
    weights).

Gate 2 (backend parity with the flag ON) lives in
    tests/test_mlx_engine_step.py::TestFullStepParity (coherence rows).
"""

import importlib.util
import math

import numpy as np
import pytest
import torch

from pytti import replace_grad
from pytti.eval_tools import is_zero_weight, parametric_eval
from pytti.Perceptor.Prompt import (
    LocationAwareMCIP,
    Prompt,
    minimize_average_distance,
    sizes_to_coherence_weights,
    spherical_dist_loss,
)

needs_mlx = pytest.mark.skipif(
    importlib.util.find_spec("mlx") is None,
    reason="mlx not installed (darwin-only backend)",
)

CPU = torch.device("cpu")
FP32_ULP_AT_1 = float(np.spacing(np.float32(1.0)))  # 2**-23


def sizes_from_px(sizes_px, side_x: int, side_y: int) -> torch.Tensor:
    """Package pixel sizes exactly as both samplers do (samplers.py:253-254,
    mlx_engine/sampler.py:_cut_batch): columns (px/side_x, px/side_y)."""
    px = torch.as_tensor(sizes_px, dtype=torch.float32)
    return torch.stack([px / side_x, px / side_y], dim=-1)


def reference_weights(sizes_px, side_x: int, side_y: int) -> np.ndarray:
    """The definition, independently in fp64: anchors (px == min side) 3x,
    everything scaled by px/min_side, normalized to mean 1."""
    px = np.asarray(sizes_px, dtype=np.float64)
    frac = px / min(side_x, side_y)
    raw = np.where(px == min(side_x, side_y), frac * 3.0, frac)
    return raw / raw.mean()


class TestSizesToCoherenceWeights:
    def test_anchor_rows_3x_then_normalized(self):
        px = [512, 512, 100, 205, 350, 128]
        weights = sizes_to_coherence_weights(sizes_from_px(px, 512, 512), 512, 512)
        expected = reference_weights(px, 512, 512)
        np.testing.assert_allclose(weights.numpy(), expected, rtol=1e-6)
        # before normalization the anchor:detail mass ratio is 3 / fraction;
        # normalization preserves ratios exactly
        ratio = weights[0] / weights[2]
        assert math.isclose(float(ratio), 3.0 / (100 / 512), rel_tol=1e-6)

    def test_detail_rows_scale_by_fraction(self):
        px = [100, 200, 400]
        weights = sizes_to_coherence_weights(sizes_from_px(px, 512, 512), 512, 512)
        assert math.isclose(float(weights[1] / weights[0]), 2.0, rel_tol=1e-6)
        assert math.isclose(float(weights[2] / weights[0]), 4.0, rel_tol=1e-6)

    @pytest.mark.parametrize(
        ("side_x", "side_y"),
        [(512, 512), (512, 384), (384, 512), (640, 360), (127, 255)],
    )
    def test_mean_is_one(self, side_x, side_y):
        max_size = min(side_x, side_y)
        rng = np.random.default_rng(7)
        px = np.concatenate(
            [
                np.full(4, max_size),
                np.floor((0.2 + 0.35 * rng.random(28) ** 1.5) * max_size),
            ]
        )
        weights = sizes_to_coherence_weights(
            sizes_from_px(px, side_x, side_y), side_x, side_y
        )
        # fp32 mean lands within 1 ulp of 1.0 (exactly 1.0 for many inputs;
        # arbitrary fractions can round the renormalized mean 1 ulp off)
        assert abs(float(weights.mean()) - 1.0) <= FP32_ULP_AT_1
        assert abs(float(weights.double().mean()) - 1.0) <= 1e-6

    def test_non_square_canvas_conventions(self):
        # anchors are the inscribed square: size == min(side), and the
        # sampler convention stores size/side per column — only the MIN-side
        # column reads exactly 1.0. Same pixel geometry must weigh the same
        # whichever way the canvas is oriented.
        px = [384, 384, 120, 256, 300]
        landscape = sizes_to_coherence_weights(
            sizes_from_px(px, 512, 384), 512, 384
        )
        portrait = sizes_to_coherence_weights(
            sizes_from_px(px, 384, 512), 384, 512
        )
        assert torch.equal(landscape, portrait)
        np.testing.assert_allclose(
            landscape.numpy(), reference_weights(px, 512, 384), rtol=1e-6
        )
        # the max-side column of an anchor is < 1 (384/512): it must NOT
        # be mistaken for the fraction
        assert float(sizes_from_px(px, 512, 384)[0, 0]) < 1.0

    def test_all_anchor_batch_is_exactly_uniform(self):
        weights = sizes_to_coherence_weights(
            sizes_from_px([384] * 5, 512, 384), 512, 384
        )
        assert torch.equal(weights, torch.ones(5))

    def test_all_detail_batch_has_no_anchor_boost(self):
        px = [100, 150, 200, 250]
        weights = sizes_to_coherence_weights(sizes_from_px(px, 512, 512), 512, 512)
        np.testing.assert_allclose(
            weights.numpy(), reference_weights(px, 512, 512), rtol=1e-6
        )
        # equal-size details normalize to exactly 1.0 (x / x == 1 in fp)
        uniform = sizes_to_coherence_weights(
            sizes_from_px([100] * 4, 512, 512), 512, 512
        )
        assert torch.equal(uniform, torch.ones(4))

    def test_perceptor_axis_and_orientation_invariance(self):
        # [n, C, 2] (text prompts) and [C, n, 2] (image prompts) carry the
        # same elements; global normalization makes the weights transposes
        # of each other
        rng = np.random.default_rng(3)
        px = np.floor(rng.uniform(64, 512, (6, 2)))
        px[0, :] = 512
        n_c = sizes_to_coherence_weights(sizes_from_px(px, 512, 512), 512, 512)
        c_n = sizes_to_coherence_weights(
            sizes_from_px(px.T, 512, 512), 512, 512
        )
        np.testing.assert_allclose(n_c.numpy(), c_n.numpy().T, rtol=1e-6)

    def test_real_smart_sampler_anchors_hit_the_exact_path(self):
        from pytti.Perceptor.cutouts.samplers import pytti_smart

        side_x, side_y, cutn = 128, 96, 16
        torch.manual_seed(11)
        _, _, sizes = pytti_smart(
            input=torch.rand(1, 3, side_y, side_x),
            side_x=side_x,
            side_y=side_y,
            cut_size=32,
            padding=0.25,
            cutn=cutn,
            cut_pow=2.0,
            border_mode="clamp",
            augs=lambda x: x,
            noise_fac=0.0,
            device=CPU,
        )
        weights = sizes_to_coherence_weights(sizes, side_x, side_y)
        n_global = max(2, round(cutn * 0.25))
        anchors = weights[:n_global]
        details = weights[n_global:]
        assert torch.equal(anchors, anchors[0].expand_as(anchors))
        assert float(anchors[0]) > float(details.max())
        assert abs(float(weights.mean()) - 1.0) <= FP32_ULP_AT_1

    def test_fail_loud_contracts(self):
        good = sizes_from_px([512, 100], 512, 512)
        with pytest.raises(ValueError, match="canvas dims"):
            sizes_to_coherence_weights(good, 0, 512)
        with pytest.raises(ValueError, match=r"\[\.\.\., 2\]"):
            sizes_to_coherence_weights(good[..., :1], 512, 512)
        with pytest.raises(ValueError, match=r"\[\.\.\., 2\]"):
            sizes_to_coherence_weights(good[0], 512, 512)


# ---------------------------------------------------------------------------
# gate 3, torch half: the off path is the old math, bit for bit
# ---------------------------------------------------------------------------


def old_forward(prompt, embed, position, size, offset=0.0, coherence_canvas=None):
    """Reference copy of Prompt.forward: the pre-feature math (b839cfe)
    plus the spec'd coherence composition — multiplied in right after the
    mask weights WITH the per-prompt mass rescale (review finding: without
    it, non-uniform mask weights silently rescale the prompt by the
    mask/coherence covariance)."""
    device = prompt.device
    if not prompt.enabled or is_zero_weight(prompt.weight):
        zero = torch.as_tensor(offset, device=device)
        return zero, zero
    dists_raw = spherical_dist_loss(embed, prompt.embeds) + offset
    weight = torch.as_tensor(parametric_eval(prompt.weight), device=device)
    stop = torch.as_tensor(parametric_eval(prompt.stop), device=device)

    mask_stops, mask_weights = prompt.mask(position, size, embed.detach())
    weight = torch.as_tensor(mask_weights, device=device) * weight
    if coherence_canvas is not None:
        coh = sizes_to_coherence_weights(size, *coherence_canvas)
        mw = torch.as_tensor(mask_weights, device=device, dtype=coh.dtype).abs()
        if mw.dim() == 0:
            mw = mw.expand_as(coh)
        scale = mw.mean() / (mw * coh).mean().clamp_min(1e-8)
        weight = weight * coh * scale
    sign_offset = weight.sign().clamp(max=0)

    dists = dists_raw * weight.sign()
    stops = torch.maximum(mask_stops + sign_offset, stop)
    dists = weight.abs() * replace_grad(dists, torch.maximum(dists, stops))
    return dists.mean(), dists_raw.mean()


def make_geometry(n=8, c=2, side_x=128, side_y=96, seed=5):
    rng = np.random.default_rng(seed)
    px = np.floor(rng.uniform(32, min(side_x, side_y), (n, c)))
    px[:2, :] = min(side_x, side_y)  # two anchor rows
    size = sizes_from_px(px, side_x, side_y)
    position = torch.rand(
        (n, c, 2), generator=torch.Generator().manual_seed(seed)
    ) * (1 - size)
    return position, size


class TensorWeightMask:
    """A mask whose mask_weights are a TENSOR (like image masks), to prove
    the multiplicative composition with coherence weights."""

    def __init__(self, weights):
        self.weights = weights

    def __call__(self, pos, size, emb):
        return torch.zeros(size.shape[:-1]).fill_(-math.inf), self.weights


class TestPromptForwardCoherence:
    def _prompt(self, prompt_string="a test:1", mask=None, seed=1, dim=24, c=2):
        from pytti.Perceptor.Prompt import make_mask
        from pytti.prompt_spec import parse_prompt_spec

        g = torch.Generator().manual_seed(seed)
        spec = parse_prompt_spec(prompt_string)
        return Prompt(
            torch.randn((c, dim), generator=g),
            spec.weight,
            spec.stop,
            spec.text,
            prompt_string,
            mask=mask if mask is not None else make_mask(spec.mask, spec.cutoff),
            device=CPU,
        )

    def _embed(self, n=8, c=2, dim=24, seed=9):
        g = torch.Generator().manual_seed(seed)
        return torch.randn((n, c, dim), generator=g)

    @pytest.mark.parametrize(
        "prompt_string", ["a test:1", "negative:-0.5", "left side:1_r_0.4"]
    )
    def test_off_is_bit_exact_old_math(self, prompt_string):
        prompt = self._prompt(prompt_string)
        embed, (position, size) = self._embed(), make_geometry()
        got = prompt(embed, position, size)
        want = old_forward(prompt, embed, position, size)
        assert torch.equal(got[0], want[0]) and torch.equal(got[1], want[1])

    @pytest.mark.parametrize(
        "prompt_string", ["a test:1", "negative:-0.5", "left side:1_r_0.4"]
    )
    def test_on_composes_after_mask_weights(self, prompt_string):
        prompt = self._prompt(prompt_string)
        embed, (position, size) = self._embed(), make_geometry()
        got = prompt(embed, position, size, coherence_canvas=(128, 96))
        want = old_forward(
            prompt, embed, position, size, coherence_canvas=(128, 96)
        )
        assert torch.equal(got[0], want[0]) and torch.equal(got[1], want[1])
        off = prompt(embed, position, size)
        assert not torch.equal(got[0], off[0])  # the knob actually acts
        assert torch.equal(got[1], off[1])  # raw records stay unweighted

    def test_composes_multiplicatively_with_tensor_mask_weights(self):
        position, size = make_geometry()
        g = torch.Generator().manual_seed(13)
        mask_weights = torch.rand(size.shape[:-1], generator=g) + 0.5
        prompt = self._prompt("a test:1", mask=TensorWeightMask(mask_weights))
        embed = self._embed()
        got = prompt(embed, position, size, coherence_canvas=(128, 96))
        want = old_forward(
            prompt, embed, position, size, coherence_canvas=(128, 96)
        )
        assert torch.equal(got[0], want[0])

    def test_location_aware_mcip_permutes_then_weighs(self):
        class StubEmbedder:
            output_axes = ("c", "n", "i")

        n, c, dim = 6, 2, 24
        g = torch.Generator().manual_seed(21)
        stored_pos, stored_size = make_geometry(n=n, c=c, seed=31)
        prompt = LocationAwareMCIP(
            torch.randn((c, n, dim), generator=g),
            stored_pos.permute(1, 0, 2),
            stored_size.permute(1, 0, 2),
            StubEmbedder(),
            "1",
            "-inf",
            "image prompt",
            "image prompt",
        )
        prompt.device = CPU
        embed = torch.randn((c, n, dim), generator=g)
        position, size = make_geometry(n=n, c=c, seed=32)
        position, size = position.permute(1, 0, 2), size.permute(1, 0, 2)

        got = prompt(embed, position, size, coherence_canvas=(128, 96))

        # reproduce: Hungarian match on centers, permute, then old math with
        # coherence computed from the PERMUTED sizes
        cent_a = prompt.positions + prompt.sizes / 2
        cent_b = position + size / 2
        indices = minimize_average_distance(cent_a, cent_b)
        p = [
            torch.stack([a[i] for a, i in zip(t, indices, strict=True)])
            for t in (embed, position, size)
        ]
        want = old_forward(
            prompt, p[0], p[1], p[2], offset=0.7, coherence_canvas=(128, 96)
        )
        assert torch.equal(got[0], want[0])


# ---------------------------------------------------------------------------
# torch <-> mlx mirror parity of the pure function
# ---------------------------------------------------------------------------


@needs_mlx
class TestMlxMirror:
    @pytest.mark.parametrize(
        ("side_x", "side_y"), [(512, 512), (512, 384), (384, 512)]
    )
    def test_matches_torch(self, side_x, side_y):
        import mlx.core as mx

        from pytti.mlx_engine.step import (
            sizes_to_coherence_weights as mlx_weights,
        )

        rng = np.random.default_rng(17)
        max_size = min(side_x, side_y)
        px = np.floor(rng.uniform(32, max_size, (16, 2)))
        px[:3, :] = max_size
        sizes = sizes_from_px(px, side_x, side_y)
        torch_w = sizes_to_coherence_weights(sizes, side_x, side_y)
        mlx_w = np.array(mlx_weights(mx.array(sizes.numpy()), side_x, side_y))
        np.testing.assert_allclose(mlx_w, torch_w.numpy(), rtol=1e-6, atol=0)
        assert abs(float(np.mean(mlx_w.astype(np.float64))) - 1.0) <= 1e-6

    def test_fail_loud_contracts(self):
        import mlx.core as mx

        from pytti.mlx_engine.step import (
            sizes_to_coherence_weights as mlx_weights,
        )

        good = mx.ones((4, 2, 2))
        with pytest.raises(ValueError, match="canvas dims"):
            mlx_weights(good, -1, 512)
        with pytest.raises(ValueError, match=r"\[\.\.\., 2\]"):
            mlx_weights(mx.ones((4, 2, 3)), 512, 512)


# ---------------------------------------------------------------------------
# gate 3, mlx_full half: flag off == the pre-feature engine (golden)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def golden():
    from pathlib import Path

    path = Path(__file__).parent / "fixtures" / "coherence_off_golden_mlx.npz"
    with np.load(path) as data:
        return dict(data.items())


@needs_mlx
class TestMlxOffPathGolden:
    STEPS = 3
    # 10x the measured same-code rebuild determinism floors (see module
    # docstring): records ~2e-7, params ~5e-5
    RECORD_GATE = 2e-6
    PARAMS_GATE = 5e-4

    def _run(self, **param_overrides):
        from tests.test_mlx_engine_step import (
            _base_params,
            _make_engine,
            _prompt,
            _tv_loss,
        )

        engine = _make_engine(
            params=_base_params(**param_overrides) if param_overrides else None
        )
        prompts = [
            _prompt("a test prompt:1"),
            _prompt("left side:1_r_0.4", seed=2),
        ]
        records = []
        for i in range(self.STEPS):
            records.append(engine.train_step(i, prompts, [], [_tv_loss(0.02)]))
        return records, engine

    @staticmethod
    def _rel(candidate: np.ndarray, reference: np.ndarray) -> float:
        candidate = candidate.astype(np.float64)
        reference = reference.astype(np.float64)
        denom = max(float(np.abs(reference).max()), 1e-12)
        return float(np.abs(candidate - reference).max()) / denom

    @pytest.mark.parametrize("explicit_false", [False, True])
    def test_off_matches_pre_feature_golden(self, golden, explicit_false):
        overrides = {"coherence_weighting": False} if explicit_false else {}
        records, engine = self._run(**overrides)
        # step 1 is deterministic (the nondeterministic scatter-add enters
        # via backward, i.e. from the step-1 PARAM update onward): bit-exact
        for name, value in records[0].items():
            assert np.array_equal(np.array(value), golden[f"record/0/{name}"]), name
        for i in (1, 2):
            for name, value in records[i].items():
                err = self._rel(np.array(value), golden[f"record/{i}/{name}"])
                assert err <= self.RECORD_GATE, f"step {i} {name}: rel {err}"
        for key, value in engine._params.items():
            err = self._rel(np.array(value), golden[f"params/{key}"])
            assert err <= self.PARAMS_GATE, f"params/{key}: rel {err}"

    def test_on_moves_only_semantic_weights(self, golden):
        records, _ = self._run(coherence_weighting=True)
        # step-1 direct/image losses see the same params + draws: bit-exact
        for name in ("smoothing loss (TV)", "HDR normalization", "Palette normalization"):
            assert np.array_equal(
                np.array(records[0][name]), golden[f"record/0/{name}"]
            ), name
        # RAW prompt records stay unweighted, but TOTAL folds the weights in
        assert not np.array_equal(
            np.array(records[0]["TOTAL"]), golden["record/0/TOTAL"]
        )


class TestMassRenormalization:
    """Review finding: coherence must not rescale a prompt whose mask
    weights are non-uniform (image masks) — the composition site rescales
    per prompt so mean(|mask| * coh * scale) == mean(|mask|)."""

    def test_uniform_mask_scale_is_identity_class(self):
        import torch

        from pytti.Perceptor.Prompt import sizes_to_coherence_weights

        torch.manual_seed(0)
        n = 40
        frac = torch.rand(n).clamp(0.2, 1.0)
        frac[:10] = 1.0  # anchors
        sizes = torch.stack([frac, frac], dim=-1)  # square canvas
        coh = sizes_to_coherence_weights(sizes, 512, 512)
        mw = torch.ones(n)
        scale = mw.mean() / (mw * coh).mean().clamp_min(1e-8)
        assert abs(float(scale) - 1.0) < 1e-5

    def test_nonuniform_mask_mass_is_conserved(self):
        import torch

        from pytti.Perceptor.Prompt import sizes_to_coherence_weights

        torch.manual_seed(1)
        n = 40
        frac = torch.rand(n).clamp(0.2, 1.0)
        frac[:10] = 1.0
        sizes = torch.stack([frac, frac], dim=-1)
        coh = sizes_to_coherence_weights(sizes, 512, 512)
        # adversarial mask: zero on anchors, full on details — the case the
        # review measured at 0.35x without renormalization
        mw = torch.ones(n)
        mw[:10] = 0.0
        scale = mw.mean() / (mw * coh).mean().clamp_min(1e-8)
        composed = mw * coh * scale
        assert abs(float(composed.mean()) - float(mw.mean())) < 1e-6
        # and the unnormalized composition really was off by ~3x, proving
        # the rescale has teeth
        assert abs(float((mw * coh).mean()) / float(mw.mean()) - 1.0) > 0.3
