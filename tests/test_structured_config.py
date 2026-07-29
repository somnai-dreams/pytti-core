import pytest
from hydra import compose, initialize
from hydra.errors import ConfigCompositionException
from omegaconf import OmegaConf, errors


def test_scenes_is_mandatory():
    with initialize(config_path="config", version_base=None):
        cfg = compose(config_name="_structured_config")
    with pytest.raises(errors.MissingMandatoryValue):
        OmegaConf.to_object(cfg)


def test_initialization_of_default_structured_config():
    with initialize(config_path="config", version_base=None):
        cfg = compose(
            config_name="_structured_config", overrides=["scenes=a test scene"]
        )
    obj = OmegaConf.to_object(cfg)
    assert obj.scenes == "a test scene"
    assert obj.image_model == "Unlimited Palette"


def test_acceptance_of_overwrite_with_valid_config():
    with initialize(config_path="config", version_base=None):
        cfg = compose(
            config_name="_structured_config",
            overrides=[
                "scenes=a test scene",
                "+conf=_test_structured_config/_valid_animation",
            ],  # animation_mode = 2D
        )
    assert OmegaConf.to_object(cfg).animation_mode == "2D"


def test_rejection_of_overwrite_with_invalid_config():
    with initialize(config_path="config", version_base=None):
        cfg = compose(
            config_name="_structured_config",
            overrides=[
                "scenes=a test scene",
                "+conf=_test_structured_config/_invalid_animation",
            ],  # animation_mode = 1D
        )
    with pytest.raises(ValueError, match="animation_mode"):
        OmegaConf.to_object(cfg)


def test_correct_spelling_of_limited_palette_is_accepted():
    # regression: the validator used to list 'Limimted Palette' and reject
    # the spelling the renderer requires
    with initialize(config_path="config", version_base=None):
        cfg = compose(
            config_name="_structured_config",
            overrides=["scenes=x", "image_model=Limited Palette"],
        )
    assert OmegaConf.to_object(cfg).image_model == "Limited Palette"


def test_unknown_key_is_rejected():
    with initialize(config_path="config", version_base=None):
        with pytest.raises(ConfigCompositionException, match="show_palette"):
            compose(
                config_name="_structured_config",
                overrides=["scenes=x", "show_palette=true"],
            )
