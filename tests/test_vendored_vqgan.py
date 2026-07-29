import torch
from omegaconf import OmegaConf

from pytti.vendor.taming.vqgan_models import GumbelVQ, VQModel, build_vqgan

TINY_DDCONFIG = {
    "double_z": False,
    "z_channels": 4,
    "resolution": 32,
    "in_channels": 3,
    "out_ch": 3,
    "ch": 32,  # GroupNorm in taming uses 32 groups
    "ch_mult": [1, 2],
    "num_res_blocks": 1,
    "attn_resolutions": [],
    "dropout": 0.0,
}


def test_vqmodel_roundtrip():
    model = VQModel(ddconfig=TINY_DDCONFIG, n_embed=16, embed_dim=4)
    x = torch.randn(1, 3, 32, 32)
    quant, _, _ = model.encode(x)
    out = model.decode(quant)
    assert out.shape == (1, 3, 32, 32)


def test_training_only_config_keys_are_ignored():
    # published checkpoint configs carry lossconfig/ckpt_path/etc.
    model = VQModel(
        ddconfig=TINY_DDCONFIG,
        n_embed=16,
        embed_dim=4,
        lossconfig={"target": "taming.modules.losses.DummyLoss"},
        ckpt_path=None,
        monitor="val/rec_loss",
    )
    assert not hasattr(model, "loss")


def test_gumbel_variant():
    model = GumbelVQ(
        ddconfig=TINY_DDCONFIG,
        n_embed=16,
        embed_dim=4,
        temperature_scheduler_config={"target": "x"},
    )
    assert hasattr(model.quantize, "embed")  # the gumbel signature pytti sniffs


def test_build_vqgan_dispatch():
    cfg = OmegaConf.create(
        {
            "model": {
                "target": "taming.models.vqgan.VQModel",
                "params": {"ddconfig": TINY_DDCONFIG, "n_embed": 16, "embed_dim": 4},
            }
        }
    )
    model, gumbel = build_vqgan(cfg)
    assert isinstance(model, VQModel)
    assert gumbel is False


def test_build_vqgan_net2net_unwraps_first_stage():
    cfg = OmegaConf.create(
        {
            "model": {
                "target": "taming.models.cond_transformer.Net2NetTransformer",
                "params": {
                    "first_stage_config": {
                        "target": "taming.models.vqgan.VQModel",
                        "params": {
                            "ddconfig": TINY_DDCONFIG,
                            "n_embed": 16,
                            "embed_dim": 4,
                        },
                    }
                },
            }
        }
    )
    model, gumbel = build_vqgan(cfg)
    assert isinstance(model, VQModel)
    assert gumbel is False
