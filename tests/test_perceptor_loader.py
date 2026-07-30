import pytest
import torch

from pytti.Perceptor import PERCEPTOR_REGISTRY, _load_perceptor


def test_registry_covers_all_schema_flags():
    import attrs

    from pytti.config.structured_config import ConfigSchema

    clip_flags = {
        f.name
        for f in attrs.fields(ConfigSchema)
        if f.name.startswith(("ViT", "RN"))
    }
    assert clip_flags == set(PERCEPTOR_REGISTRY)


def test_registry_names_exist_in_open_clip():
    import open_clip

    available = {(name, tag) for name, tag in open_clip.list_pretrained()}
    for key, (model_name, pretrained) in PERCEPTOR_REGISTRY.items():
        assert (model_name, pretrained) in available, (key, model_name, pretrained)


def test_openai_registry_uses_quickgelu():
    # OpenAI weights were trained with QuickGELU; the plain open_clip configs
    # use vanilla GELU and drift ~0.77 in embedding space (measured)
    for key, (model_name, pretrained) in PERCEPTOR_REGISTRY.items():
        if pretrained == "openai":
            assert "quickgelu" in model_name, (key, model_name)


@pytest.mark.download
def test_loaded_perceptor_contract():
    p = _load_perceptor("ViTB32", torch.device("cpu"))
    assert p.cut_size == 224
    img = torch.rand(2, 3, 224, 224)
    with torch.no_grad():
        e = p.encode_image(p.normalize(img))
        t = p.embed_text("a test prompt", "cpu")
    assert e.shape[0] == 2 and t.shape[0] == 1
