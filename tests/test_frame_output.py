import pytest

from pytti.ImageGuide import breath_alpha, frame_filename


def test_frame_filename_zero_pads():
    assert frame_filename("run", 1) == "run_0001.png"
    assert frame_filename("run", 42) == "run_0042.png"
    assert frame_filename("run", 12345) == "run_12345.png"


def test_frame_filenames_sort_lexically():
    names = [frame_filename("x", n) for n in (1, 2, 10, 100, 1000)]
    assert names == sorted(names)


def test_breath_alpha_ramps_zero_to_one():
    # 1 scene x 1000 steps, save every 100 -> 10 frames
    assert breath_alpha(0, 1, 1000, 100) == 0.0
    assert breath_alpha(5, 1, 1000, 100) == pytest.approx(0.5)
    assert breath_alpha(10, 1, 1000, 100) == 1.0


def test_breath_alpha_clamps_past_end():
    assert breath_alpha(15, 1, 1000, 100) == 1.0


def test_breath_alpha_multiple_scenes():
    # 2 scenes x 500 steps, save every 100 -> 10 frames total
    assert breath_alpha(5, 2, 500, 100) == pytest.approx(0.5)


def test_breath_alpha_never_divides_by_zero():
    assert breath_alpha(1, 1, 10, 100) == 1.0  # fewer steps than save interval
