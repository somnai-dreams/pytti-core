"""
The torch <-> MLX boundary for phase M1 (docs/mlx-port-plan.md).

`MLXSemanticLoss` replaces the semantic-loss half of a train step: instead of
torch `encode_image` + `Prompt.forward`, the (already sampled + augmented)
cutout batches cross into MLX once per microbatch, the towers and the
per-prompt weighted reduction run there, and a single fused fwd+bwd produces
both the loss values and the input-gradient. Everything image-side (samplers,
augs, normalization, the optimizer) stays torch and is untouched.

Semantics contract
------------------
The MLX reduction reproduces ``Prompt.forward`` exactly (see
`mlx_semantic_reduction`): spherical distance on padded embeddings, weight
sign handling, stop thresholds via the straight-through ``replace_grad``
trick. Everything in that reduction that does *not* depend on the image —
parametric weight/stop evaluation and the spatial mask vectors, which are
functions of crop geometry only — is computed torch-side under ``no_grad``
by `semantic_prompt_constants` and crosses the boundary as constants.

M1 scope (fail loud, no silent engine mixing): plain semantic text prompts
only. `LocationAwareMCIP` (image prompts, semantic stabilization/init) needs
per-cutout embeddings on the torch side for Hungarian matching, and semantic
``[mask]`` prompts need the embedding *inside* the mask — both raise
``RuntimeError`` telling the user to use ``perceptor_backend=torch``.

Gradient decomposition
----------------------
Per-prompt input-gradients would cost one tower backward per prompt; instead
the bridge differentiates the *pre-scaled sum* of the prompt losses (the
interpolation-ramp scale of every prompt is folded in as a constant before
the boundary), stores ONE gradient per cutout batch, and `backward` returns
``stored_grad * grad_total`` — exact, because the total is the only
differentiable output and gradients are linear in its cotangent. Per-prompt
raw losses come back as non-differentiable extras for the loss records.

Sync discipline
---------------
DLPack does not synchronize pending work, so the bridge owns exactly two
runtime syncs per crossing: ``torch.mps.synchronize()`` before MLX imports
the torch buffers, and ``mx.eval`` + ``mx.synchronize()`` before torch reads
the MLX results. ``torch.from_dlpack`` of an MLX array is a zero-copy VIEW
of MLX memory — every result is ``.clone()``d into torch-owned memory before
the view is dropped. These are the ONLY device syncs in the step path (see
the ``ImageGuide.train`` header comment); nothing here calls ``float()`` or
``.item()``.

``mlx`` is imported lazily inside functions: importing this module is safe
on any platform (the constants helpers are pure torch and tested on linux).
"""

from dataclasses import dataclass

import torch
from loguru import logger

from pytti import format_input
from pytti.eval_tools import is_zero_weight, parametric_eval
from pytti.Perceptor.mlx_backend.convert import MLX_VIT_MODELS
from pytti.Perceptor.Prompt import Prompt


def _sync_torch(device: torch.device) -> None:
    """Flush torch's pending GPU work before another runtime reads its
    buffers (DLPack does not do this for us)."""
    if device.type == "mps":
        torch.mps.synchronize()
    elif device.type == "cuda":  # pragma: no cover - MLX never pairs with cuda
        torch.cuda.synchronize(device)


# --------------------------------------------------------------------------
# per-prompt constants (pure torch, no mlx — testable anywhere)
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class PromptConstants:
    """Everything Prompt.forward computes that does not depend on the image,
    evaluated for one step (parametric weights are time-dependent)."""

    name: str
    # [C, i_max] fp32 — the prompt's text embeddings, exactly as registered
    # (cat_with_pad zero-padding included; spherical distance is invariant
    # to zero-padding, so the reduction pads tower outputs to match instead
    # of slicing)
    embeds: torch.Tensor
    # [n, C] fp32 — mask weights x parametric weight, signed
    weight: torch.Tensor
    # [n, C] fp32 — maximum(mask_stops + sign_offset, parametric stop)
    stops: torch.Tensor


@torch.no_grad()
def semantic_prompt_constants(
    prompt: Prompt,
    offsets: torch.Tensor,
    sizes: torch.Tensor,
    embedder,
) -> PromptConstants | None:
    """
    Recompute the constant half of ``Prompt.forward`` for one step.

    `offsets`/`sizes` are the embedder-stacked ``[C, n, 2]`` crop geometry
    tensors. Returns None for prompts that contribute nothing this step
    (disabled or zero weight — matching the early-out in Prompt.forward).
    """
    if type(prompt) is not Prompt:
        raise RuntimeError(
            f"perceptor_backend=mlx cannot evaluate {type(prompt).__name__} "
            f"({prompt!r}): M1 supports plain semantic text prompts only. "
            "Image prompts, semantic stabilization, and semantic init "
            "weights need perceptor_backend=torch."
        )
    if getattr(prompt.mask, "embed_dependent", False):
        raise RuntimeError(
            f"perceptor_backend=mlx cannot evaluate the semantic [mask] on "
            f"{prompt!r}: it needs the image embedding inside the mask. "
            "Use perceptor_backend=torch."
        )
    if not prompt.enabled or is_zero_weight(prompt.weight):
        return None

    device = offsets.device
    position = format_input(offsets, embedder, prompt)  # [n, C, 2]
    size = format_input(sizes, embedder, prompt)

    # mirrors Prompt.forward line for line (emb=None: every non-semantic
    # mask ignores the embedding argument)
    weight = torch.as_tensor(parametric_eval(prompt.weight), device=device)
    stop = torch.as_tensor(parametric_eval(prompt.stop), device=device)
    mask_stops, mask_weights = prompt.mask(position, size, None)
    weight = torch.as_tensor(mask_weights, device=device) * weight
    sign_offset = weight.sign().clamp(max=0)
    stops = torch.maximum(mask_stops + sign_offset, stop)

    shape = mask_stops.shape  # [n, C]
    return PromptConstants(
        name=str(prompt),
        embeds=prompt.embeds.detach(),
        weight=weight.to(torch.float32).broadcast_to(shape).contiguous(),
        stops=stops.to(torch.float32).broadcast_to(shape).contiguous(),
    )


# --------------------------------------------------------------------------
# the MLX reduction (lazy mlx import; pure function of mx arrays)
# --------------------------------------------------------------------------


def mlx_semantic_reduction(embeds, text, weights, stops, scales):
    """
    The MLX mirror of ``Prompt.forward``'s math, fused over all prompts.

    embeds:  [n, C, i] fp32 — tower embeddings, zero-padded to a common width
    text:    [P, C, i] fp32 — per-prompt text embeddings (same padding)
    weights: [P, n, C] fp32 — signed, mask-weighted (PromptConstants.weight)
    stops:   [P, n, C] fp32 — folded stop thresholds (PromptConstants.stops)
    scales:  [P]       fp32 — caller-side multiplier (interpolation ramp)

    Returns (total, losses [P], raws [P]). ``total`` is the differentiable
    output: sum of scales * losses. The stop threshold is applied with the
    straight-through recipe ``maxed + stop_gradient(dists - maxed)`` — value
    ``dists``, gradient of ``maximum(dists, stops)`` — exactly torch's
    ``replace_grad(dists, torch.maximum(dists, stops))``.
    """
    import mlx.core as mx

    def normalize(x):  # torch F.normalize: x / max(||x||, 1e-12)
        norm = mx.sqrt(mx.sum(x * x, axis=-1, keepdims=True))
        return x / mx.maximum(norm, 1e-12)

    en = normalize(embeds)[None]  # [1, n, C, i]
    tn = normalize(text)[:, None]  # [P, 1, C, i]
    dist = mx.sqrt(mx.sum((en - tn) ** 2, axis=-1))  # [P, n, C]
    raw = 2.0 * mx.arcsin(dist / 2.0) ** 2  # spherical_dist_loss

    dists = raw * mx.sign(weights)
    maxed = mx.maximum(dists, stops)
    gated = maxed + mx.stop_gradient(dists - maxed)
    losses = mx.mean(mx.abs(weights) * gated, axis=(1, 2))  # [P]
    raws = mx.mean(raw, axis=(1, 2))  # [P]
    total = mx.sum(scales * losses)
    return total, losses, raws


def _make_step(towers, batch_index: list[int], out_dim_max: int):
    """Compile ONE fused fwd+bwd: batches -> (total, raws), d(total)/d(batches).

    Retraces only when shapes change (prompt count changes at scene
    boundaries); per-step constant *values* reuse the trace.
    """
    import mlx.core as mx

    def loss_fn(batches, text, weights, stops, scales):
        embs = []
        for tower, b in zip(towers, batch_index, strict=True):
            e = tower.encode(batches[b]).astype(mx.float32)  # [n, d_c]
            pad = out_dim_max - e.shape[-1]
            if pad:
                e = mx.pad(e, ((0, 0), (0, pad)))
            embs.append(e)
        emb = mx.stack(embs, axis=1)  # [n, C, i_max]
        total, _losses, raws = mlx_semantic_reduction(
            emb, text, weights, stops, scales
        )
        return total, raws

    return mx.compile(mx.value_and_grad(loss_fn))


# --------------------------------------------------------------------------
# the autograd boundary
# --------------------------------------------------------------------------


class _MLXBridgeFunction(torch.autograd.Function):
    """
    forward: cutout batches (torch, fp32, NCHW, CLIP-normalized) -> NHWC ->
    DLPack -> MLX fused fwd+bwd -> (total loss, per-prompt raw losses) with
    the input-gradients stashed. backward: stored gradient x the incoming
    cotangent of ``total`` (exact — see module docstring). NO tower work at
    backward time.
    """

    @staticmethod
    def forward(ctx, mlx_step, constants, *batches):
        import mlx.core as mx
        from torch.utils.dlpack import to_dlpack

        device = batches[0].device
        # materialize every crossing tensor BEFORE the sync: contiguous() is
        # itself a kernel launch, and MLX must not read a buffer whose fill
        # is still queued (measured: skipping this order reads stale memory)
        nhwc = [b.permute(0, 2, 3, 1).contiguous() for b in batches]
        consts = [c.contiguous() for c in constants]
        _sync_torch(device)
        # torch -> MLX copies (safe); MLX convs want NHWC
        mx_batches = [mx.array(to_dlpack(t)) for t in nhwc]
        mx_consts = [mx.array(to_dlpack(c)) for c in consts]

        (total, raws), grads = mlx_step(mx_batches, *mx_consts)
        mx.eval(total, raws, *grads)
        mx.synchronize()

        # MLX -> torch lands as a zero-copy view of MLX memory (on mps):
        # copy everything into torch-owned buffers on the INPUT's device
        # before the views are dropped (copy=True forces the copy even when
        # the view is already on `device`)
        total_t = torch.from_dlpack(total).to(device, copy=True)
        raws_t = torch.from_dlpack(raws).to(device, copy=True)
        grads_t = [
            torch.from_dlpack(g).to(device, copy=True).permute(0, 3, 1, 2).contiguous()
            for g in grads
        ]
        ctx.save_for_backward(*grads_t)
        ctx.mark_non_differentiable(raws_t)
        return total_t, raws_t

    @staticmethod
    def backward(ctx, grad_total, _grad_raws):
        return (None, None, *(g * grad_total for g in ctx.saved_tensors))


class MLXSemanticLoss:
    """
    The engine-facing object: one per DirectImageGuide when
    ``perceptor_backend=mlx``. Owns the MLX towers (fp16, converted/cached on
    first use) and the compiled fused step; consumes the SAME cutout batches
    the torch path would (``HDMultiClipEmbedder.cutout_batches`` — samplers,
    augs, RNG stream, and normalization all unchanged).
    """

    def __init__(self, embedder, dtype: str = "float16"):
        from pytti.Perceptor.mlx_backend.convert import load_tower

        perceptors = list(embedder.perceptors)
        unsupported = sorted(
            {p.key for p in perceptors} - set(MLX_VIT_MODELS)
        )
        if unsupported:
            raise RuntimeError(
                f"perceptor_backend=mlx supports only {sorted(MLX_VIT_MODELS)} "
                f"in M1; got {unsupported}. Deselect them or use "
                "perceptor_backend=torch."
            )
        self.embedder = embedder
        self._perceptors = perceptors

        # towers with the same input resolution share one cutout batch (the
        # Embedder shares them by cut_size) — group them so each batch
        # crosses the boundary once
        group_of_size: dict[int, int] = {}
        self._group_leader: list[int] = []  # perceptor index per batch group
        batch_index: list[int] = []  # perceptor -> batch group
        for idx, perceptor in enumerate(perceptors):
            size = perceptor.cut_size
            if size not in group_of_size:
                group_of_size[size] = len(self._group_leader)
                self._group_leader.append(idx)
            batch_index.append(group_of_size[size])
        for idx, perceptor in enumerate(perceptors):
            leader = perceptors[self._group_leader[batch_index[idx]]]
            same_stats = tuple(leader.normalize.mean) == tuple(
                perceptor.normalize.mean
            ) and tuple(leader.normalize.std) == tuple(perceptor.normalize.std)
            if not same_stats:
                raise RuntimeError(
                    f"{perceptor.key} and {leader.key} share a cutout batch "
                    "(same input resolution) but disagree on normalization "
                    "stats — the shared-batch bridge cannot represent that."
                )

        self._towers = [load_tower(p.key, dtype) for p in perceptors]
        self._out_dim_max = max(t.config.output_dim for t in self._towers)
        self._step = _make_step(self._towers, batch_index, self._out_dim_max)
        logger.info(
            f"MLX bridge ready: {[p.key for p in perceptors]} ({dtype}), "
            f"{len(self._group_leader)} cutout batch(es) per microbatch."
        )

    def __call__(
        self,
        diff_image,
        input: torch.Tensor,
        prompts: list,
        interp_prompts: list,
        ramp: float,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """
        One microbatch of semantic loss. `prompts` enter scaled by ``ramp``
        and are recorded under ``str(prompt)`` (raw, unweighted — same names
        and values as the torch path's loss records); `interp_prompts` enter
        scaled by ``1 - ramp`` and are not recorded (matching train()).

        Returns (total, records): ``total`` is differentiable w.r.t. the
        image; record values are detached 0-dim device tensors.
        """
        embedder = self.embedder
        batches = embedder.cutout_batches(diff_image, input=input)
        device = batches[0][0].device

        with torch.no_grad():
            offsets = torch.stack([o for _, o, _ in batches])  # [C, n, 2]
            sizes = torch.stack([s for _, _, s in batches])  # [C, n, 2]

            records: dict[str, torch.Tensor] = {}
            active: list[tuple[bool, float, PromptConstants]] = []
            for prompt_list, scale, recorded in (
                (prompts, ramp, True),
                (interp_prompts, 1.0 - ramp, False),
            ):
                for prompt in prompt_list:
                    consts = semantic_prompt_constants(
                        prompt, offsets, sizes, embedder
                    )
                    if consts is None:
                        # Prompt.forward's early-out contributes offset=0
                        if recorded:
                            records[str(prompt)] = torch.zeros((), device=device)
                        continue
                    active.append((recorded, float(scale), consts))
            if not active:
                return torch.zeros((), device=device), records

            expected = (len(self._perceptors), self._out_dim_max)
            for _, _, consts in active:
                if tuple(consts.embeds.shape) != expected:
                    raise RuntimeError(
                        f"prompt {consts.name!r} carries embeddings of shape "
                        f"{tuple(consts.embeds.shape)}, expected {expected} — "
                        "it was parsed under a different perceptor ensemble."
                    )
            text = torch.stack(
                [c.embeds for _, _, c in active]
            ).float().contiguous()  # [P, C, i_max]
            weights = torch.stack([c.weight for _, _, c in active]).contiguous()
            stops = torch.stack([c.stops for _, _, c in active]).contiguous()
            scales = torch.tensor(
                [s for _, s, _ in active], dtype=torch.float32, device=device
            )

        # one normalized batch per resolution group, differentiable w.r.t.
        # the image (Normalize is affine; autograd chains the bridge's
        # returned gradient back through it to the samplers)
        group_batches = [
            self._perceptors[i].normalize(batches[i][0])
            for i in self._group_leader
        ]
        total, raws = _MLXBridgeFunction.apply(
            self._step, (text, weights, stops, scales), *group_batches
        )
        for idx, (recorded, _, consts) in enumerate(active):
            if recorded:
                records[consts.name] = raws[idx]
        return total, records
