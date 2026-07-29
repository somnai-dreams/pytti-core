import torch

from pytti.Perceptor.Prompt import (
    Prompt,
    make_mask,
    mask_down,
    mask_far,
    mask_left,
    mask_near,
    mask_right,
    mask_up,
)
from pytti.prompt_spec import MaskGeometric

# a cutout at x∈[0.1, 0.5], y∈[0.6, 0.8]: pos=(0.1, 0.6), size=(0.4, 0.2)
POS = torch.tensor([[0.1, 0.6]])
SIZE = torch.tensor([[0.4, 0.2]])
EMB = torch.zeros(1, 4)


def test_directional_mask_semantics():
    # x-center = 0.1 + 0.4/2 = 0.3 ; y-center = 0.6 + 0.2/2 = 0.7
    assert mask_right(POS, SIZE, EMB, thresh=0.4)[0].item() == 1.0  # 0.3 < 0.4
    assert mask_right(POS, SIZE, EMB, thresh=0.2)[0].item() == 0.0
    assert mask_left(POS, SIZE, EMB, thresh=0.2)[0].item() == 1.0  # 0.3 > 0.2
    assert mask_down(POS, SIZE, EMB, thresh=0.8)[0].item() == 1.0  # 0.7 < 0.8
    assert mask_up(POS, SIZE, EMB, thresh=0.5)[0].item() == 1.0  # 0.7 > 0.5


def test_near_far_use_size_not_position():
    # min cutout size = 0.2
    assert mask_near(POS, SIZE, EMB, thresh=0.3)[0].item() == 1.0  # 0.2 < 0.3
    assert mask_far(POS, SIZE, EMB, thresh=0.1)[0].item() == 1.0  # 0.2 > 0.1


def test_make_mask_passes_pos_and_size_in_order():
    # regression: make_mask's lambda used to swap pos and size, so every
    # geometric mask computed the wrong quantity. With pos.x-center=0.3 and
    # a swap the center would be size.x + pos.x/2 = 0.45 — thresh 0.4
    # distinguishes the two.
    masked = make_mask(MaskGeometric(key="r"), "0.4")
    expected_weight = mask_right(POS, SIZE, EMB, thresh=0.4)
    stops, weights = masked(POS, SIZE, EMB)
    assert weights == expected_weight[1] or torch.equal(
        torch.as_tensor(weights), torch.as_tensor(expected_weight[1])
    )
    assert torch.equal(stops, expected_weight[0])
    assert stops.item() == 1.0  # correct order: 0.3 < 0.4


def test_make_mask_near_uses_size():
    # with the historical swap, "n" tested position instead of size
    masked = make_mask(MaskGeometric(key="n"), "0.3")
    stops, _ = masked(POS, SIZE, EMB)
    assert stops.item() == 1.0  # min size 0.2 < 0.3


def test_make_mask_threshold_is_parametric():
    masked = make_mask(MaskGeometric(key="r"), "0.2 + 0.2")
    stops, _ = masked(POS, SIZE, EMB)
    assert stops.item() == 1.0


def test_zero_weight_prompt_short_circuits():
    for weight in ("0", "0.0", "", 0, 0.0):
        p = Prompt(torch.zeros(1, 4), weight, "-inf", "x", "x", device="cpu")
        loss, loss_raw = p.forward(torch.zeros(1, 4), POS, SIZE)
        assert float(loss) == 0.0
        assert loss_raw == 0.0
