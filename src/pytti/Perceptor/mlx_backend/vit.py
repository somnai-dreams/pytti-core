"""
CLIP visual tower (OpenAI ViT architecture) in mlx.nn.

Architecture, matching the OpenAI checkpoints exactly: patch conv (no bias),
class token + learned position embedding, pre-LN transformer blocks
(`mx.fast.scaled_dot_product_attention` + `mx.fast.layer_norm`), MLP with
the checkpoint's activation (``ViTConfig.activation``: QuickGELU
``x * sigmoid(1.702 x)`` for the OpenAI tier, exact erf-GELU for the
open_clip laion/FARE lineage), final LN on the class token + linear
projection. The q/k/v projections are fused into one matmul (the converter
row-concatenates the HF weights; open_clip checkpoints ship them pre-fused);
this is numerically identical to separate projections.

Input contract (the bridge boundary)
------------------------------------
``encode`` takes an **NHWC** fp32 batch that the torch side has ALREADY
preprocessed: cutouts resized to ``config.image_size`` and normalized with
that perceptor's CLIP mean/std. This module does no resizing and no
normalization. The input is explicitly cast to the tower's weight dtype on
entry (mixed-dtype ops in MLX would otherwise silently promote the whole
tower to fp32); gradients w.r.t. the input flow back through that cast, so
``mx.value_and_grad`` over an fp32 input returns fp32 gradients even when
the tower computes in fp16.

Embeddings are returned **unnormalized**, same as open_clip's
``encode_image``; prompt losses normalize downstream.

Precision knobs (the plan's gate-2 fallback ladder): weight dtype is set by
the loaded weights (see ``convert.load_tower``); ``ln_fp32`` accumulates
every LayerNorm in fp32. The ladder's "precise softmax" rung needs no knob:
``mx.fast.scaled_dot_product_attention`` on mlx 0.32 already accumulates its
softmax in fp32 — verified bit-identical to a manual attention with
``mx.softmax(..., precise=True)`` on fp16 inputs (2026-07-30).

``encode`` is pure (no mutable state, weights frozen after load), so it —
or any loss built on it — can be wrapped in ``mx.compile`` directly.
"""

import mlx.core as mx
import mlx.nn as nn

from pytti.Perceptor.mlx_backend import ViTConfig


def quick_gelu(x: mx.array) -> mx.array:
    return x * mx.sigmoid(1.702 * x)


# ViTConfig.activation -> callable. ``nn.gelu`` is the exact erf form
# (x * (1 + erf(x / sqrt 2)) / 2) — parity with torch
# ``nn.GELU(approximate='none')`` measured at 6e-7 max abs diff over
# [-6, 6] fp32 (mlx 0.32). Do NOT swap in gelu_approx/gelu_fast_approx:
# those are the tanh/sigmoid approximations, a different function.
_ACTIVATIONS = {"quickgelu": quick_gelu, "gelu": nn.gelu}


class LayerNorm(nn.Module):
    """LayerNorm via mx.fast.layer_norm, with optional fp32 accumulation."""

    def __init__(self, dims: int, eps: float, upcast: bool):
        super().__init__()
        self.weight = mx.ones((dims,))
        self.bias = mx.zeros((dims,))
        self._eps = eps
        self._upcast = upcast

    def __call__(self, x: mx.array) -> mx.array:
        if self._upcast and x.dtype != mx.float32:
            y = mx.fast.layer_norm(
                x.astype(mx.float32),
                self.weight.astype(mx.float32),
                self.bias.astype(mx.float32),
                self._eps,
            )
            return y.astype(x.dtype)
        return mx.fast.layer_norm(x, self.weight, self.bias, self._eps)


class Attention(nn.Module):
    """Multi-head self-attention with a fused qkv projection."""

    def __init__(self, dims: int, num_heads: int):
        super().__init__()
        self.qkv = nn.Linear(dims, 3 * dims, bias=True)
        self.out_proj = nn.Linear(dims, dims, bias=True)
        self._num_heads = num_heads
        self._scale = (dims // num_heads) ** -0.5

    def __call__(self, x: mx.array) -> mx.array:
        batch, tokens, dims = x.shape
        q, k, v = mx.split(self.qkv(x), 3, axis=-1)
        q = q.reshape(batch, tokens, self._num_heads, -1).transpose(0, 2, 1, 3)
        k = k.reshape(batch, tokens, self._num_heads, -1).transpose(0, 2, 1, 3)
        v = v.reshape(batch, tokens, self._num_heads, -1).transpose(0, 2, 1, 3)
        out = mx.fast.scaled_dot_product_attention(q, k, v, scale=self._scale)
        return self.out_proj(out.transpose(0, 2, 1, 3).reshape(batch, tokens, dims))


class MLP(nn.Module):
    def __init__(self, dims: int, hidden_dims: int, activation: str):
        super().__init__()
        self.fc1 = nn.Linear(dims, hidden_dims, bias=True)
        self.fc2 = nn.Linear(hidden_dims, dims, bias=True)
        # KeyError here is unreachable through ViTConfig (validated at
        # construction) but still loud for any direct caller
        self._act = _ACTIVATIONS[activation]

    def __call__(self, x: mx.array) -> mx.array:
        return self.fc2(self._act(self.fc1(x)))


class ResidualAttentionBlock(nn.Module):
    """Pre-LN transformer block (OpenAI CLIP layout)."""

    def __init__(self, config: ViTConfig, ln_fp32: bool):
        super().__init__()
        self.ln_1 = LayerNorm(config.hidden_dim, config.layer_norm_eps, ln_fp32)
        self.attn = Attention(config.hidden_dim, config.num_heads)
        self.ln_2 = LayerNorm(config.hidden_dim, config.layer_norm_eps, ln_fp32)
        self.mlp = MLP(config.hidden_dim, config.mlp_dim, config.activation)

    def __call__(self, x: mx.array) -> mx.array:
        x = x + self.attn(self.ln_1(x))
        return x + self.mlp(self.ln_2(x))


class VisionTower(nn.Module):
    """
    The full visual tower. Parameter names are the target vocabulary of
    ``convert.plan_conversion`` — renaming anything here without updating the
    planner will fail loudly at (strict) weight load.
    """

    def __init__(self, config: ViTConfig, *, ln_fp32: bool = False):
        super().__init__()
        self.patch_embed = nn.Conv2d(
            3,
            config.hidden_dim,
            kernel_size=config.patch_size,
            stride=config.patch_size,
            bias=False,
        )
        self.class_embedding = mx.zeros((config.hidden_dim,))
        self.positional_embedding = mx.zeros((config.num_tokens, config.hidden_dim))
        self.ln_pre = LayerNorm(config.hidden_dim, config.layer_norm_eps, ln_fp32)
        self.blocks = [
            ResidualAttentionBlock(config, ln_fp32)
            for _ in range(config.num_layers)
        ]
        self.ln_post = LayerNorm(config.hidden_dim, config.layer_norm_eps, ln_fp32)
        self.proj = nn.Linear(config.hidden_dim, config.output_dim, bias=False)
        self._config = config

    @property
    def config(self) -> ViTConfig:
        return self._config

    @property
    def dtype(self) -> mx.Dtype:
        """Compute dtype, derived from the loaded weights (single source)."""
        return self.patch_embed.weight.dtype

    def encode(self, images_nhwc: mx.array) -> mx.array:
        """[n, S, S, 3] normalized fp32 -> [n, output_dim] unnormalized."""
        size = self._config.image_size
        if images_nhwc.ndim != 4 or images_nhwc.shape[1:] != (size, size, 3):
            raise ValueError(
                f"expected NHWC batch [n, {size}, {size}, 3] "
                f"(channels LAST — MLX convs are NHWC), got {images_nhwc.shape}"
            )
        x = images_nhwc.astype(self.dtype)
        h = self.patch_embed(x)  # [n, grid, grid, D]
        batch = h.shape[0]
        h = h.reshape(batch, -1, self._config.hidden_dim)
        cls = mx.broadcast_to(
            self.class_embedding, (batch, 1, self._config.hidden_dim)
        )
        h = mx.concatenate([cls, h], axis=1)  # [n, tokens, D]
        h = h + self.positional_embedding
        h = self.ln_pre(h)
        for block in self.blocks:
            h = block(h)
        h = self.ln_post(h[:, 0, :])  # class token
        return self.proj(h)

    def __call__(self, images_nhwc: mx.array) -> mx.array:
        return self.encode(images_nhwc)
