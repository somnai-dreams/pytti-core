"""
Slice 7: make_optimizer factory + schedule-free eval/train save semantics.

All tests run on CPU — the optimizer math is device-agnostic; MPS behavior is
covered by the workhorse smoke runs.
"""

import pytest
import schedulefree
import torch
from omegaconf import OmegaConf
from torch import nn, optim

from pytti.ImageGuide import DirectImageGuide, make_optimizer


class TinyImageRep(nn.Module):
    """Minimal stand-in for DifferentiableImage: parameters + lr."""

    lr = 0.1

    def __init__(self):
        super().__init__()
        self.tensor = nn.Parameter(torch.zeros(4))


def make_guide(optimizer_name: str) -> DirectImageGuide:
    params = OmegaConf.create(
        {
            "input_audio": "",
            "input_audio_filters": "",
            "optimizer": optimizer_name,
        }
    )
    return DirectImageGuide(TinyImageRep(), None, params=params)


# ---------------------------------------------------------------------------
# factory
# ---------------------------------------------------------------------------


def test_factory_adam_matches_legacy():
    rep = TinyImageRep()
    opt = make_optimizer(rep.parameters(), "adam", 0.05)
    assert type(opt) is optim.Adam
    assert opt.param_groups[0]["lr"] == 0.05


def test_factory_adamw_sf_warmup_and_train_mode():
    rep = TinyImageRep()
    opt = make_optimizer(rep.parameters(), "adamw_sf", 0.05)
    assert isinstance(opt, schedulefree.AdamWScheduleFree)
    assert opt.param_groups[0]["lr"] == 0.05
    assert opt.param_groups[0]["warmup_steps"] == 10
    # must come back ready to step: schedulefree refuses to step in eval mode
    assert opt.param_groups[0]["train_mode"] is True


def test_factory_unknown_name_fails_loud():
    rep = TinyImageRep()
    with pytest.raises(ValueError, match="sgd"):
        make_optimizer(rep.parameters(), "sgd", 0.05)


def test_adamw_sf_converges_on_quadratic():
    # same regime as the plan's convergence smoke: 60 steps, initial loss
    # ~0.33, measured final 1.6e-4 on this machine
    torch.manual_seed(0)
    target = torch.tensor([0.5, -0.8, 0.3, 0.6])
    p = nn.Parameter(torch.zeros(4))
    opt = make_optimizer([p], "adamw_sf", 0.1)
    first_loss = ((p - target) ** 2).mean().item()
    for _ in range(60):
        opt.zero_grad()
        loss = ((p - target) ** 2).mean()
        loss.backward()
        opt.step()
    # readout must use the averaged iterate
    opt.eval()
    final_loss = ((p - target) ** 2).mean().item()
    assert final_loss < first_loss / 100
    assert final_loss < 1e-3


# ---------------------------------------------------------------------------
# DirectImageGuide integration
# ---------------------------------------------------------------------------


def test_guide_reads_optimizer_from_params():
    guide = make_guide("adamw_sf")
    assert isinstance(guide.optimizer, schedulefree.AdamWScheduleFree)
    guide = make_guide("adam")
    assert type(guide.optimizer) is optim.Adam


def test_params_none_guide_defaults_to_adam():
    # bare guide, e.g. PixelImage.encode_image's palette fit
    guide = DirectImageGuide(TinyImageRep(), None, params=None)
    assert type(guide.optimizer) is optim.Adam
    with guide.optimizer_eval():
        pass  # no-op passthrough must not raise
    guide.update(0, 0)  # params=None early-return path


def test_explicit_optimizer_kwarg_bypasses_factory():
    rep = TinyImageRep()
    prebuilt = optim.Adam(rep.parameters(), lr=0.1)
    guide = DirectImageGuide(rep, None, optimizer=prebuilt, params=None)
    assert guide.optimizer is prebuilt


def test_set_optim_reconstructs_in_train_mode():
    guide = make_guide("adamw_sf")
    first = guide.optimizer
    guide.set_optim(None)  # what reset_lr_each_frame does per frame
    assert guide.optimizer is not first
    assert isinstance(guide.optimizer, schedulefree.AdamWScheduleFree)
    assert guide.optimizer.param_groups[0]["train_mode"] is True


# ---------------------------------------------------------------------------
# eval/train save-path wrapper
# ---------------------------------------------------------------------------


def test_optimizer_eval_round_trips_mode():
    guide = make_guide("adamw_sf")
    opt = guide.optimizer
    assert opt.param_groups[0]["train_mode"] is True
    with guide.optimizer_eval():
        assert opt.param_groups[0]["train_mode"] is False
    assert opt.param_groups[0]["train_mode"] is True


def test_optimizer_eval_restores_train_mode_on_exception():
    guide = make_guide("adamw_sf")
    with pytest.raises(RuntimeError, match="boom"):
        with guide.optimizer_eval():
            raise RuntimeError("boom")
    assert guide.optimizer.param_groups[0]["train_mode"] is True


def test_optimizer_eval_swaps_averaged_iterate_and_restores():
    guide = make_guide("adamw_sf")
    p = guide.image_rep.tensor
    opt = guide.optimizer
    target = torch.full((4,), 5.0)
    for _ in range(20):
        opt.zero_grad()
        ((p - target) ** 2).mean().backward()
        opt.step()
    y = p.detach().clone()  # fast (train) iterate
    with guide.optimizer_eval():
        x = p.detach().clone()  # averaged (eval) iterate
    assert not torch.allclose(x, y), "eval must expose a different iterate"
    # the swap is a lerp round-trip, so exact bit equality isn't guaranteed
    assert torch.allclose(p.detach(), y, atol=1e-6), "train iterate must be restored"
