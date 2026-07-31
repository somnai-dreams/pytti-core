import torch

from pytti.Perceptor.cutouts.augs import BatchedAugs, pytti_batched


def _off():
    return dict(p_flip=0, p_affine=0, p_persp=0, p_jitter=0, p_erase=0)


def test_all_ops_off_is_identity():
    torch.manual_seed(0)
    x = torch.rand(8, 3, 32, 32)
    out = BatchedAugs(**_off())(x)
    assert torch.allclose(out, x, atol=1e-5)


def test_flip_only_is_exact_horizontal_flip():
    torch.manual_seed(0)
    x = torch.rand(8, 3, 32, 32)
    out = BatchedAugs(**{**_off(), "p_flip": 1.0})(x)
    assert torch.allclose(out, x.flip(-1), atol=1e-5)


def test_erase_only_zeroes_a_rectangle_per_sample():
    torch.manual_seed(0)
    x = torch.ones(8, 3, 64, 64)
    out = BatchedAugs(**{**_off(), "p_erase": 1.0})(x)
    zeroed = (out == 0).flatten(1).sum(dim=1)
    # scale (0.1, 0.4) of the area, all three channels
    assert bool((zeroed >= 0.05 * 3 * 64 * 64).all())
    assert bool((zeroed <= 0.5 * 3 * 64 * 64).all())


def test_jitter_only_is_small_and_preserves_gray():
    torch.manual_seed(0)
    x = torch.rand(8, 3, 32, 32)
    out = BatchedAugs(**{**_off(), "p_jitter": 1.0})(x)
    # hue 0.01 / sat 0.01 are tiny perturbations
    assert float((out - x).abs().max()) < 0.05
    gray = torch.full((2, 3, 8, 8), 0.5)
    out_gray = BatchedAugs(**{**_off(), "p_jitter": 1.0})(gray)
    # the gray axis is invariant under both hue rotation and saturation
    assert torch.allclose(out_gray, gray, atol=1e-4)


def test_warp_stays_in_range_and_grads_flow():
    torch.manual_seed(0)
    x = torch.rand(16, 3, 32, 32, requires_grad=True)
    out = pytti_batched()(x)
    assert out.shape == x.shape
    out.sum().backward()
    assert x.grad is not None
    assert bool((x.grad != 0).any())
    assert not bool(out.isnan().any())


def test_default_stack_actually_transforms():
    torch.manual_seed(0)
    x = torch.rand(32, 3, 32, 32)
    out = pytti_batched()(x)
    # with default probabilities, most samples must differ from the input
    changed = (out - x).abs().flatten(1).max(dim=1).values > 1e-3
    assert int(changed.sum()) >= 24
