"""
Inference-only VQGAN models, adapted from CompVis/taming-transformers
(taming/models/vqgan.py) with the pytorch-lightning base class, training
methods, and perceptual-loss construction removed — pytti only decodes and
encodes through the frozen first stage. See LICENSE in this directory.

Checkpoint configs carry training-only parameters (lossconfig, ckpt_path,
image_key, monitors, temperature schedules); constructors accept and ignore
them so the published config files load unmodified.
"""

import torch
from torch import nn

from pytti.vendor.taming.diffusion_model import Decoder, Encoder
from pytti.vendor.taming.quantize import GumbelQuantize
from pytti.vendor.taming.quantize import VectorQuantizer2 as VectorQuantizer


class VQModel(nn.Module):
    def __init__(
        self,
        ddconfig,
        n_embed,
        embed_dim,
        remap=None,
        sane_index_shape=False,
        **_training_only,
    ):
        super().__init__()
        self.encoder = Encoder(**ddconfig)
        self.decoder = Decoder(**ddconfig)
        self.quantize = VectorQuantizer(
            n_embed, embed_dim, beta=0.25, remap=remap, sane_index_shape=sane_index_shape
        )
        self.quant_conv = nn.Conv2d(ddconfig["z_channels"], embed_dim, 1)
        self.post_quant_conv = nn.Conv2d(embed_dim, ddconfig["z_channels"], 1)

    def encode(self, x):
        h = self.encoder(x)
        h = self.quant_conv(h)
        quant, emb_loss, info = self.quantize(h)
        return quant, emb_loss, info

    def decode(self, quant):
        quant = self.post_quant_conv(quant)
        return self.decoder(quant)

    def forward(self, input):
        quant, diff, _ = self.encode(input)
        return self.decode(quant), diff


class GumbelVQ(VQModel):
    def __init__(
        self,
        ddconfig,
        n_embed,
        embed_dim,
        kl_weight=1e-8,
        remap=None,
        **_training_only,
    ):
        super().__init__(ddconfig, n_embed, embed_dim, remap=remap)
        self.vocab_size = n_embed
        self.quantize = GumbelQuantize(
            ddconfig["z_channels"],
            embed_dim,
            n_embed=n_embed,
            kl_weight=kl_weight,
            temp_init=1.0,
            remap=remap,
        )


def load_checkpoint_state(checkpoint_path) -> dict:
    """
    Load a taming checkpoint's state_dict. Tries the safe weights-only path
    first; 2020-era Lightning checkpoints can carry pickled hyperparameter
    objects, in which case we fall back to a full unpickle of this
    explicitly-configured, locally-cached file.
    """
    try:
        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    except Exception:
        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    return ckpt["state_dict"] if "state_dict" in ckpt else ckpt


def build_vqgan(config) -> tuple[VQModel, bool]:
    """
    Instantiate the (frozen, loss-free) VQGAN described by a taming config
    node (`config.model`). Net2NetTransformer configs resolve to their
    first-stage model. Returns (model, is_gumbel).
    """
    target = config.model.target
    params = config.model.params
    if target.endswith("Net2NetTransformer"):
        first = params.first_stage_config
        target, params = first.target, first.params
    if target.endswith(".VQModel"):
        return VQModel(**params), False
    if target.endswith(".GumbelVQ"):
        return GumbelVQ(**params), True
    raise ValueError(f"unknown model type: {config.model.target}")
