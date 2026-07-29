import pytest

from pytti.prompt_spec import (
    DEFAULT_CUTOFF,
    MaskAll,
    MaskGeometric,
    MaskImage,
    MaskSemantic,
    MaskVideo,
    parse_prompt_spec,
    parse_weight_spec,
    split_toplevel,
)


def test_bare_text():
    spec = parse_prompt_spec("a forest")
    assert spec.text == "a forest"
    assert spec.weight == "1"
    assert spec.stop == "-inf"
    assert spec.mask == MaskAll()
    assert spec.cutoff == DEFAULT_CUTOFF


def test_text_and_weight():
    spec = parse_prompt_spec("a forest:2")
    assert spec.text == "a forest"
    assert spec.weight == "2"


def test_negative_weight_and_stop():
    spec = parse_prompt_spec("blurry:-1.0:-.95")
    assert spec.text == "blurry"
    assert spec.weight == "-1.0"
    assert spec.stop == "-.95"


def test_stop_weight_parsing_expressions():
    spec = parse_prompt_spec("pulse:10*sin(t/2):0.5")
    assert spec.weight == "10*sin(t/2)"
    assert spec.stop == "0.5"


def test_geometric_mask():
    spec = parse_prompt_spec("the sky:1_u")
    assert spec.mask == MaskGeometric(key="u")


def test_mask_with_cutoff():
    spec = parse_prompt_spec("the sky:1_u_0.3")
    assert spec.mask == MaskGeometric(key="u")
    assert spec.cutoff == "0.3"


def test_bracketed_image_mask():
    spec = parse_prompt_spec("a face:2_[mask.png]")
    assert spec.mask == MaskImage(path="mask.png", inverted=False)


def test_inverted_image_mask():
    spec = parse_prompt_spec("a face:2_[-mask.png]")
    assert spec.mask == MaskImage(path="mask.png", inverted=True)


def test_video_mask_case_insensitive():
    spec = parse_prompt_spec("a face:2_[roto.MP4]")
    assert spec.mask == MaskVideo(path="roto.MP4", inverted=False)


def test_semantic_mask():
    spec = parse_prompt_spec("fire:1_the left half of the image")
    assert spec.mask == MaskSemantic(text="the left half of the image")


def test_image_prompt_text():
    spec = parse_prompt_spec("[init.png]:3")
    assert spec.image_path() == "init.png"
    assert spec.weight == "3"


# --- the Windows/URL cases that motivated the unified parser ---------------


def test_windows_path_bare():
    # direct image prompts pass bare paths; the drive-letter colon must not
    # be treated as a weight separator (this was broken upstream)
    spec = parse_prompt_spec(r"C:\Users\max\img.png:0.5")
    assert spec.text == r"C:\Users\max\img.png"
    assert spec.weight == "0.5"


def test_windows_path_forward_slashes():
    spec = parse_prompt_spec("C:/Users/max/img.png:0.5")
    assert spec.text == "C:/Users/max/img.png"
    assert spec.weight == "0.5"


def test_windows_path_bracketed():
    spec = parse_prompt_spec(r"init image [C:\Users\max\img.png]:0.5")
    assert spec.text == r"init image [C:\Users\max\img.png]"
    assert spec.image_path() is None  # text isn't *only* the bracket
    assert spec.weight == "0.5"


def test_windows_path_in_mask():
    spec = parse_prompt_spec(r"a face:2_[C:\masks\face_mask.png]")
    assert spec.mask == MaskImage(path=r"C:\masks\face_mask.png", inverted=False)


def test_underscores_in_bracketed_mask_survive():
    spec = parse_prompt_spec("a face:2_[my_mask_file.png]")
    assert spec.mask == MaskImage(path="my_mask_file.png", inverted=False)


def test_url_prompt():
    spec = parse_prompt_spec("https://example.com/a.png:0.5")
    assert spec.text == "https://example.com/a.png"
    assert spec.weight == "0.5"


def test_url_without_weight():
    spec = parse_prompt_spec("http://example.com/img.png")
    assert spec.text == "http://example.com/img.png"
    assert spec.weight == "1"


def test_bracketed_url():
    spec = parse_prompt_spec("[https://example.com/a.png]:2")
    assert spec.image_path() == "https://example.com/a.png"


# --- errors and edge cases --------------------------------------------------


def test_empty_prompt_raises():
    with pytest.raises(ValueError, match="no text"):
        parse_prompt_spec(":1")
    with pytest.raises(ValueError, match="no text"):
        parse_prompt_spec("")


def test_weight_spec_bare_paths():
    # direct-prompt style: image.png:1.5_mask.png (no brackets)
    weight, mask, cutoff = parse_weight_spec("1.5_mask.png")
    assert weight == "1.5"
    assert mask == MaskImage(path="mask.png", inverted=False)


def test_weight_spec_video_mask_inverted():
    weight, mask, _ = parse_weight_spec("2_-roto.mp4")
    assert weight == "2"
    assert mask == MaskVideo(path="roto.mp4", inverted=True)


def test_weight_spec_defaults():
    weight, mask, cutoff = parse_weight_spec("")
    assert weight == "1"
    assert mask == MaskAll()
    assert cutoff == DEFAULT_CUTOFF


def test_split_toplevel_respects_brackets():
    assert split_toplevel("a[b:c]d:e", ":", 2) == ["a[b:c]d", "e"]
    assert split_toplevel("x_[a_b]_y", "_", 2) == ["x", "[a_b]", "y"]


def test_split_toplevel_maxsplit():
    assert split_toplevel("a:b:c:d", ":", 2) == ["a", "b", "c:d"]
