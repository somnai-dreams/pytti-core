"""
End-to-end render smokes: tiny configs, real model downloads (CLIP; VQGAN
for the vqgan cases). Marked `download` so the default CPU test run skips
them; run explicitly with `pytest -m download`.

3D and Video Source rows additionally need the optional adabins/gma extras
and are marked `extras`.
"""

from pathlib import Path

import pytest
from hydra import compose, initialize

from pytti.config.model_names import VQGAN_MODEL_NAMES

pytestmark = pytest.mark.download

CONFIG_BASE_PATH = "config"
CONFIG_DEFAULTS = "default.yaml"
FIXTURE_VIDEO = str(Path(__file__).parent / "fixtures" / "HebyMorgongava_512kb.mp4")


def render(**overrides):
    from pytti.workhorse import _hydra_main as render_frames

    with initialize(config_path=CONFIG_BASE_PATH, version_base=None):
        cfg = compose(
            config_name=CONFIG_DEFAULTS,
            overrides=[f"{k}={v}" for k, v in overrides.items()],
        )
        render_frames(cfg)


@pytest.mark.parametrize(
    "conf",
    ["_test_limited_palette", "_test_unlimited_palette"],
)
def test_image_models(conf):
    render(conf=conf)


def test_vqgan():
    render(conf="_test_vqgan")


@pytest.mark.parametrize(
    "vqgan_model",
    VQGAN_MODEL_NAMES,
)
def test_vqgan_checkpoints(vqgan_model):
    render(conf="_test_vqgan", vqgan_model=vqgan_model)


def test_animation_2d():
    render(conf="_test_limited_palette", animation_mode="2D", translate_x="'3'")


@pytest.mark.extras
def test_animation_3d():
    pytest.importorskip("adabins", reason="needs the [threed] extra")
    render(conf="_test_limited_palette", animation_mode="3D")


@pytest.mark.extras
def test_animation_video_source():
    pytest.importorskip("gma", reason="needs the [video] extra")
    render(
        conf="_test_limited_palette",
        animation_mode="Video Source",
        video_path=FIXTURE_VIDEO,
    )


@pytest.mark.parametrize(
    "weight_key",
    [
        "direct_stabilization_weight",
        "semantic_stabilization_weight",
        "edge_stabilization_weight",
    ],
)
def test_stabilization_modes(weight_key):
    render(
        conf="_test_limited_palette",
        init_image=str(Path(__file__).parent / "fixtures" / "01-velo-header-seattle-needle.jpg"),
        **{weight_key: "1"},
    )
