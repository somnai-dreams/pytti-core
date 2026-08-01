"""
MLX aug stack (docs/mlx-m2-seam-map.md, slice S4).

Parity strategy: torch ``BatchedAugs`` draws its parameters internally from
the global torch stream, so the tests seed that stream, run the reference,
then re-seed and REPLAY its exact draw sequence (same call order, shapes,
dtypes — see ``_replay_torch_draws``) to recover the values it consumed.
Those values are injected into the MLX port through the ``AugParams`` test
seam and outputs/grads are compared directly.

Every test needs mlx (darwin, ``pytestmark`` below); none download.
Contention-immune: correctness only, no timing.
"""

import importlib.util
import math

import pytest
import torch

from pytti.Perceptor.cutouts.augs import BatchedAugs

needs_mlx = pytest.mark.skipif(
    importlib.util.find_spec("mlx") is None,
    reason="mlx not installed (darwin-only backend)",
)
pytestmark = needs_mlx

SEED = 1234


def _off():
    return dict(p_flip=0, p_affine=0, p_persp=0, p_jitter=0, p_erase=0)


def _all_on():
    return dict(p_flip=1, p_affine=1, p_persp=1, p_jitter=1, p_erase=1)


def _replay_torch_draws(n, erase_ratio=(0.3, 1 / 0.3)):
    """Replicate the exact RNG call sequence of ``BatchedAugs.forward``
    (same order, shapes, dtypes) so a re-seeded default stream yields the
    values the reference consumed. Returns AugParams kwargs as [n] fp32
    torch tensors. Must be kept in lockstep with
    ``pytti/Perceptor/cutouts/augs.py`` draw order.
    """
    d = {}
    d["flip_u"] = torch.rand(n, 1).flatten()  # _warp_matrices:64
    d["theta_u"] = torch.rand(n)  # :72
    d["tx_u"] = torch.rand(n)  # :75
    d["ty_u"] = torch.rand(n)  # :76
    d["affine_gate_u"] = torch.rand(n, 1, 1).flatten()  # :88
    d["px_u"] = torch.rand(n)  # :94
    d["py_u"] = torch.rand(n)  # :95
    d["persp_gate_u"] = torch.rand(n, 1, 1).flatten()  # :104
    d["sat_u"] = torch.rand(n, 1, 1).flatten()  # _color_matrices:118
    d["hue_u"] = torch.rand(n)  # :123
    d["jitter_gate_u"] = torch.rand(n, 1, 1).flatten()  # :139
    d["area_u"] = torch.rand(n)  # forward:167
    d["log_r"] = torch.empty(n).uniform_(  # :170
        math.log(erase_ratio[0]), math.log(erase_ratio[1])
    )
    d["y0_u"] = torch.rand(n)  # :176
    d["x0_u"] = torch.rand(n)  # :177
    d["erase_gate_u"] = torch.rand(n, 1, 1).flatten()  # :187
    return d


def _injected_params(n, **erase_ratio_kw):
    """Re-seeded replay -> AugParams. Call with the SAME seed state the
    reference consumed (i.e. torch.manual_seed(SEED) immediately before).
    """
    import mlx.core as mx

    from pytti.mlx_engine.augs import AugParams

    draws = _replay_torch_draws(n, **erase_ratio_kw)
    return AugParams(**{k: mx.array(v.numpy()) for k, v in draws.items()})


def _max_abs(a):
    import mlx.core as mx

    return float(mx.abs(a).max())


def _cosine(a, b):
    import mlx.core as mx

    a, b = a.flatten(), b.flatten()
    return float(
        (a * b).sum() / (mx.sqrt((a * a).sum()) * mx.sqrt((b * b).sum()))
    )


# ---------------------------------------------------------------------------
# gate: injected-parameter parity vs torch BatchedAugs
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("n,h,w", [(16, 32, 32), (12, 24, 40)])
@pytest.mark.parametrize("kwargs", [{}, _all_on()], ids=["defaults", "all_on"])
def test_injected_value_parity_vs_torch(n, h, w, kwargs):
    import mlx.core as mx

    from pytti.mlx_engine.augs import AugConfig, apply_augs

    g = torch.Generator().manual_seed(7)
    x_t = torch.rand(n, 3, h, w, generator=g)
    torch.manual_seed(SEED)
    ref = BatchedAugs(**kwargs)(x_t)
    torch.manual_seed(SEED)
    params = _injected_params(n)
    out = apply_augs(mx.array(x_t.numpy()), params, AugConfig(**kwargs))

    assert _max_abs(out - mx.array(ref.numpy())) <= 1e-5
    # the comparison must not be vacuous: the stack actually transformed
    assert _max_abs(out - mx.array(x_t.numpy())) > 1e-3


@pytest.mark.parametrize("kwargs", [{}, _all_on()], ids=["defaults", "all_on"])
def test_injected_grad_parity_vs_torch(kwargs):
    import mlx.core as mx

    from pytti.mlx_engine.augs import AugConfig, apply_augs

    n, h, w = 16, 32, 32
    g = torch.Generator().manual_seed(11)
    x_t = torch.rand(n, 3, h, w, generator=g).requires_grad_()
    r_t = torch.randn(n, 3, h, w, generator=g)

    torch.manual_seed(SEED)
    (BatchedAugs(**kwargs)(x_t) * r_t).sum().backward()
    torch.manual_seed(SEED)
    params = _injected_params(n)

    r_mx = mx.array(r_t.numpy())
    cfg = AugConfig(**kwargs)
    grad = mx.grad(lambda xi: (apply_augs(xi, params, cfg) * r_mx).sum())(
        mx.array(x_t.detach().numpy())
    )
    grad_ref = mx.array(x_t.grad.numpy())

    assert _max_abs(grad) > 0
    assert _cosine(grad, grad_ref) >= 0.9999


def test_local_grid_sample_matches_torch():
    """The documented local copy of the S3 gather-bilinear recipe vs torch
    F.grid_sample (bilinear, border, align_corners=False), including
    out-of-frame grid points to exercise the border clamp."""
    import mlx.core as mx

    from pytti.mlx_engine.augs import grid_sample_border

    g = torch.Generator().manual_seed(3)
    x_t = torch.rand(4, 3, 24, 40, generator=g)
    grid_t = (torch.rand(4, 17, 23, 2, generator=g) * 2 - 1) * 1.3
    ref = torch.nn.functional.grid_sample(
        x_t, grid_t, mode="bilinear", padding_mode="border", align_corners=False
    )
    out = grid_sample_border(mx.array(x_t.numpy()), mx.array(grid_t.numpy()))
    assert _max_abs(out - mx.array(ref.numpy())) <= 1e-6


# ---------------------------------------------------------------------------
# invariants ported from tests/test_batched_augs.py
# ---------------------------------------------------------------------------


def test_all_ops_off_is_identity_exact():
    import mlx.core as mx

    from pytti.mlx_engine.augs import AugConfig, apply_augs, draw_aug_params

    cfg = AugConfig(**_off())
    x = mx.random.uniform(shape=(8, 3, 32, 32), key=mx.random.key(0))
    out = apply_augs(x, draw_aug_params(8, cfg, key=mx.random.key(1)), cfg)
    # power-of-two sizes make the pixel-center grid math exact in fp32
    assert bool(mx.array_equal(out, x))


def test_flip_only_is_exact_horizontal_flip():
    import mlx.core as mx

    from pytti.mlx_engine.augs import AugConfig, apply_augs, draw_aug_params

    cfg = AugConfig(**{**_off(), "p_flip": 1.0})
    x = mx.random.uniform(shape=(8, 3, 32, 32), key=mx.random.key(0))
    out = apply_augs(x, draw_aug_params(8, cfg, key=mx.random.key(1)), cfg)
    assert bool(mx.array_equal(out, x[..., ::-1]))


def test_erase_only_zeroes_a_rectangle_per_sample():
    import mlx.core as mx

    from pytti.mlx_engine.augs import AugConfig, apply_augs, draw_aug_params

    cfg = AugConfig(**{**_off(), "p_erase": 1.0})
    x = mx.ones((8, 3, 64, 64))
    out = apply_augs(x, draw_aug_params(8, cfg, key=mx.random.key(2)), cfg)
    # erase-only output is binary: pixels are kept (1) or erased (0)
    assert bool(((out == 0) | (out == 1)).all())
    zeroed = (out == 0).reshape(8, -1).sum(axis=1)
    # scale (0.1, 0.4) of the area, all three channels
    assert bool((zeroed >= 0.05 * 3 * 64 * 64).all())
    assert bool((zeroed <= 0.5 * 3 * 64 * 64).all())


def test_jitter_only_is_small_and_preserves_gray():
    import mlx.core as mx

    from pytti.mlx_engine.augs import AugConfig, apply_augs, draw_aug_params

    cfg = AugConfig(**{**_off(), "p_jitter": 1.0})
    x = mx.random.uniform(shape=(8, 3, 32, 32), key=mx.random.key(0))
    out = apply_augs(x, draw_aug_params(8, cfg, key=mx.random.key(1)), cfg)
    # hue 0.01 / sat 0.01 are tiny perturbations
    assert _max_abs(out - x) < 0.05
    gray = mx.full((2, 3, 8, 8), 0.5)
    out_gray = apply_augs(gray, draw_aug_params(2, cfg, key=mx.random.key(3)), cfg)
    # the gray axis is invariant under both hue rotation and saturation
    assert bool(mx.allclose(out_gray, gray, atol=1e-4))


def test_default_stack_actually_transforms():
    import mlx.core as mx

    from pytti.mlx_engine.augs import AugConfig, apply_augs, draw_aug_params

    cfg = AugConfig()
    x = mx.random.uniform(shape=(32, 3, 32, 32), key=mx.random.key(0))
    out = apply_augs(x, draw_aug_params(32, cfg, key=mx.random.key(1)), cfg)
    assert out.shape == x.shape
    assert not bool(mx.isnan(out).any())
    # with default probabilities, most samples must differ from the input
    changed = mx.abs(out - x).reshape(32, -1).max(axis=1) > 1e-3
    assert int(changed.sum()) >= 24


# ---------------------------------------------------------------------------
# gate: RNG-drawn parameter distributions match the torch ranges
# ---------------------------------------------------------------------------


def test_drawn_distributions_match_torch_ranges():
    import mlx.core as mx

    from pytti.mlx_engine.augs import AugConfig, draw_aug_params

    cfg = AugConfig()
    n = 4096
    p = draw_aug_params(n, cfg, key=mx.random.key(0))

    uniforms = [f for f in p._fields if f != "log_r"]
    for name in uniforms:
        arr = getattr(p, name)
        assert arr.shape == (n,) and arr.dtype == mx.float32
        lo, hi = float(arr.min()), float(arr.max())
        assert 0.0 <= lo and hi < 1.0, f"{name} out of U[0,1): [{lo}, {hi}]"
        # spans the range and is centered like a uniform
        assert lo < 0.05 and hi > 0.95, f"{name} does not span [0,1)"
        assert abs(float(arr.mean()) - 0.5) < 0.05, f"{name} mean off"

    log_lo, log_hi = math.log(cfg.erase_ratio[0]), math.log(cfg.erase_ratio[1])
    assert float(p.log_r.min()) >= log_lo
    assert float(p.log_r.max()) <= log_hi

    # derived quantities land in the torch module's sampling ranges
    deg_rad = cfg.degrees * math.pi / 180
    theta = (p.theta_u * 2 - 1) * deg_rad
    assert -deg_rad <= float(theta.min()) and float(theta.max()) <= deg_rad
    tx = (p.tx_u * 2 - 1) * (2 * cfg.translate)
    assert _max_abs(tx) <= 2 * cfg.translate
    px = (p.px_u * 2 - 1) * (cfg.distortion / 2)
    assert _max_abs(px) <= cfg.distortion / 2
    s = 1 + (p.sat_u * 2 - 1) * cfg.sat
    assert 1 - cfg.sat <= float(s.min()) and float(s.max()) <= 1 + cfg.sat
    ang = (p.hue_u * 2 - 1) * (cfg.hue * 2 * math.pi)
    assert _max_abs(ang) <= cfg.hue * 2 * math.pi
    frac = cfg.erase_scale[0] + p.area_u * (
        cfg.erase_scale[1] - cfg.erase_scale[0]
    )
    assert cfg.erase_scale[0] <= float(frac.min())
    assert float(frac.max()) <= cfg.erase_scale[1]


# ---------------------------------------------------------------------------
# RNG contract: key threading and global-state behavior (the compile seam)
# ---------------------------------------------------------------------------


def test_explicit_key_is_deterministic_and_leaves_global_state_alone():
    import mlx.core as mx

    from pytti.mlx_engine.augs import AugConfig, draw_aug_params

    cfg = AugConfig()
    a = draw_aug_params(8, cfg, key=mx.random.key(42))
    mx.random.seed(0)
    before = mx.random.uniform(shape=(4,))
    b = draw_aug_params(8, cfg, key=mx.random.key(42))
    c = draw_aug_params(8, cfg, key=mx.random.key(43))
    for fa, fb in zip(a, b, strict=True):
        assert bool(mx.array_equal(fa, fb))
    assert not bool(mx.array_equal(a.flip_u, c.flip_u))
    # keyed draws must not have consumed the global stream
    mx.random.seed(0)
    assert bool(mx.array_equal(before, mx.random.uniform(shape=(4,))))


def test_global_stream_draws_advance_and_reseed_reproduces():
    import mlx.core as mx

    from pytti.mlx_engine.augs import AugConfig, draw_aug_params

    cfg = AugConfig()
    mx.random.seed(5)
    a = draw_aug_params(8, cfg)
    b = draw_aug_params(8, cfg)
    assert not bool(mx.array_equal(a.flip_u, b.flip_u))  # state advanced
    mx.random.seed(5)
    a2 = draw_aug_params(8, cfg)
    for fa, fa2 in zip(a, a2, strict=True):
        assert bool(mx.array_equal(fa, fa2))


def test_compile_seam_with_global_rng_state():
    """apply_augs with in-step draws composes under mx.compile with
    mx.random.state threaded through inputs/outputs — the exact S5
    whole-step contract (seam map §3)."""
    import mlx.core as mx

    from pytti.mlx_engine.augs import AugConfig, apply_augs, draw_aug_params

    cfg = AugConfig()

    def step(xi):
        return apply_augs(xi, draw_aug_params(8, cfg), cfg)

    compiled = mx.compile(
        step, inputs=[mx.random.state], outputs=[mx.random.state]
    )
    x = mx.random.uniform(shape=(8, 3, 32, 32), key=mx.random.key(9))
    mx.random.seed(5)
    a = compiled(x)
    b = compiled(x)
    assert not bool(mx.array_equal(a, b))  # RNG advanced inside the compile
    mx.random.seed(5)
    a2 = compiled(x)
    assert bool(mx.array_equal(a, a2))  # reseed reproduces, bit-exact
    # compiled vs eager: same draws, but kernel fusion reorders fp ops
    # (fma), so parity is ulp-level rather than bit-exact
    mx.random.seed(5)
    assert _max_abs(a - step(x)) <= 1e-5


# ---------------------------------------------------------------------------
# fail-loud boundaries
# ---------------------------------------------------------------------------


def test_rejects_non_fp32_and_bad_shapes():
    import mlx.core as mx

    from pytti.mlx_engine.augs import AugConfig, apply_augs, draw_aug_params

    cfg = AugConfig()
    params = draw_aug_params(4, cfg, key=mx.random.key(0))
    with pytest.raises(ValueError, match="fp32"):
        apply_augs(mx.ones((4, 3, 8, 8), dtype=mx.float16), params, cfg)
    with pytest.raises(ValueError, match=r"\[n, 3, h, w\]"):
        apply_augs(mx.ones((4, 1, 8, 8)), params, cfg)
    with pytest.raises(ValueError, match="shape"):
        apply_augs(mx.ones((5, 3, 8, 8)), params, cfg)  # n mismatch


def test_config_validation_fails_loud():
    from pytti.mlx_engine.augs import AugConfig, draw_aug_params

    with pytest.raises(ValueError, match="p_flip"):
        AugConfig(p_flip=1.5)
    with pytest.raises(ValueError, match="erase_scale"):
        AugConfig(erase_scale=(0.4, 0.1))
    with pytest.raises(ValueError, match="degrees"):
        AugConfig(degrees=-1)
    with pytest.raises(ValueError, match="n must be positive"):
        draw_aug_params(0, AugConfig())
