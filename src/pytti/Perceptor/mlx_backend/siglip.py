"""
SigLIP visual tower (timm layout) in mlx.nn.

Architecture, matching timm's ``vit_base_patch16_siglip_224`` exactly as
open_clip instantiates it for the SigLIP2 checkpoints (structure and
activation verified against the LOADED torch module, timm 1.0.28,
2026-08-03 — see ``SigLIPConfig``'s docstring):

- patch conv WITH bias -> [n, tokens, D] (row-major over the grid, the same
  order timm's ``flatten(2).transpose(1, 2)`` produces);
- + learned position embedding (no class token, no pre-LN);
- pre-LN transformer blocks identical in structure to the OpenAI ViT's
  (fused qkv + sdpa + MLP), but LayerNorm eps 1e-6 and exact erf-GELU;
- final LayerNorm over ALL tokens;
- MAP attention-pooling head (:class:`MAPHead`): a learned latent probe
  attends over the tokens, then a residual pre-LN MLP; the probe's output
  IS the image embedding (``output_dim == hidden_dim``, no projection).

Input contract is the same as ``vit.VisionTower.encode``: an **NHWC** fp32
batch, already resized to ``config.image_size`` and normalized with THIS
perceptor's stats (SigLIP is mean=std=0.5 — not the CLIP constants). The
embedding comes back **unnormalized**, matching open_clip ``encode_image``.

The transformer primitives (LayerNorm/Attention/MLP) are vit.py's — one
implementation, two towers. The block is re-declared here rather than
reusing ``vit.ResidualAttentionBlock`` only because that class is
constructed from a ``ViTConfig``; the math is identical.
"""

import mlx.core as mx
import mlx.nn as nn

from pytti.Perceptor.mlx_backend import SigLIPConfig
from pytti.Perceptor.mlx_backend.vit import MLP, Attention, LayerNorm


class SigLIPBlock(nn.Module):
    """Pre-LN transformer block (timm Block with identity ls/drop-path)."""

    def __init__(self, config: SigLIPConfig, ln_fp32: bool):
        super().__init__()
        self.ln_1 = LayerNorm(config.hidden_dim, config.layer_norm_eps, ln_fp32)
        self.attn = Attention(config.hidden_dim, config.num_heads)
        self.ln_2 = LayerNorm(config.hidden_dim, config.layer_norm_eps, ln_fp32)
        self.mlp = MLP(config.hidden_dim, config.mlp_dim, config.activation)

    def __call__(self, x: mx.array) -> mx.array:
        x = x + self.attn(self.ln_1(x))
        return x + self.mlp(self.ln_2(x))


class MAPHead(nn.Module):
    """timm ``AttentionPoolLatent`` with latent_len=1, pool='token'.

    A learned [1, 1, D] latent is the (single) query; keys/values come from
    a fused kv projection of the tokens (k = first D channels, v = the
    last D — timm's ``reshape(B, N, 2, heads, head_dim)`` split order);
    then ``x = proj(attn); x = x + mlp(norm(x)); return x[:, 0]``.
    """

    def __init__(self, config: SigLIPConfig, ln_fp32: bool):
        super().__init__()
        dims = config.hidden_dim
        self.latent = mx.zeros((1, 1, dims))
        self.q = nn.Linear(dims, dims, bias=True)
        self.kv = nn.Linear(dims, 2 * dims, bias=True)
        self.proj = nn.Linear(dims, dims, bias=True)
        self.norm = LayerNorm(dims, config.layer_norm_eps, ln_fp32)
        self.mlp = MLP(dims, config.mlp_dim, config.activation)
        self._num_heads = config.num_heads
        self._scale = (dims // config.num_heads) ** -0.5

    def __call__(self, x: mx.array) -> mx.array:
        batch, tokens, dims = x.shape
        heads = self._num_heads
        q = self.q(mx.broadcast_to(self.latent, (batch, 1, dims)))
        q = q.reshape(batch, 1, heads, -1).transpose(0, 2, 1, 3)
        k, v = mx.split(self.kv(x), 2, axis=-1)
        k = k.reshape(batch, tokens, heads, -1).transpose(0, 2, 1, 3)
        v = v.reshape(batch, tokens, heads, -1).transpose(0, 2, 1, 3)
        out = mx.fast.scaled_dot_product_attention(q, k, v, scale=self._scale)
        out = self.proj(out.transpose(0, 2, 1, 3).reshape(batch, 1, dims))
        out = out + self.mlp(self.norm(out))
        return out[:, 0]


class SigLIPTower(nn.Module):
    """
    The full SigLIP visual tower. Parameter names are the target vocabulary
    of ``convert``'s timm plan — renaming anything here without updating the
    planner fails loudly at (strict) weight load.
    """

    def __init__(self, config: SigLIPConfig, *, ln_fp32: bool = False):
        super().__init__()
        self.patch_embed = nn.Conv2d(
            3,
            config.hidden_dim,
            kernel_size=config.patch_size,
            stride=config.patch_size,
            bias=True,
        )
        # kept [1, tokens, D] — the checkpoint's own shape; broadcasting
        # over the batch is identical either way
        self.positional_embedding = mx.zeros(
            (1, config.num_tokens, config.hidden_dim)
        )
        self.blocks = [
            SigLIPBlock(config, ln_fp32) for _ in range(config.num_layers)
        ]
        self.ln_post = LayerNorm(config.hidden_dim, config.layer_norm_eps, ln_fp32)
        self.attn_pool = MAPHead(config, ln_fp32)
        self._config = config

    @property
    def config(self) -> SigLIPConfig:
        return self._config

    @property
    def dtype(self) -> mx.Dtype:
        """Compute dtype, derived from the loaded weights (single source)."""
        return self.patch_embed.weight.dtype

    def encode(self, images_nhwc: mx.array) -> mx.array:
        """[n, S, S, 3] normalized fp32 -> [n, hidden_dim] unnormalized."""
        size = self._config.image_size
        if images_nhwc.ndim != 4 or images_nhwc.shape[1:] != (size, size, 3):
            raise ValueError(
                f"expected NHWC batch [n, {size}, {size}, 3] "
                f"(channels LAST — MLX convs are NHWC), got {images_nhwc.shape}"
            )
        x = images_nhwc.astype(self.dtype)
        h = self.patch_embed(x)  # [n, grid, grid, D]
        h = h.reshape(h.shape[0], -1, self._config.hidden_dim)
        h = h + self.positional_embedding
        for block in self.blocks:
            h = block(h)
        h = self.ln_post(h)  # ALL tokens (no class token in this lineage)
        return self.attn_pool(h)

    def __call__(self, images_nhwc: mx.array) -> mx.array:
        return self.encode(images_nhwc)
