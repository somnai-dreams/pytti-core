"""
Engine integration — M2 slice S6 (docs/mlx-m2-seam-map.md, §4-5).

``MLXStillEngine`` is what ``DirectImageGuide`` delegates ``train()`` to
when ``perceptor_backend == "mlx_full"``: it owns the MLX side of the run —
the S1 image params tree, the bias-corrected Adam state, and the compiled
whole-step function (S5) — and speaks torch at exactly the boundaries the
seam map draws:

- **in**, once at setup: the image ``state_dict``, tower weights (M1's
  converter cache), loss-aug comp/mask constants, prompt text embeddings;
- **in**, per step: parametric weight/stop/cutoff scalars and the interp
  ramp (host floats -> tiny fp32 arrays, never retrace triggers);
- **out**, per step: the torch-named loss records as LAZY mx scalars
  (``float()`` happens at ``_report_losses``/early-stop only — the
  no-mid-step-sync rule in ``train()``'s header comment);
- **out**, at ``save_every``: :meth:`write_back` loads the params tree into
  the torch module's ``state_dict`` so PNG/breath/``.bak`` stay
  torch-serialized and ``restore=True`` round-trips (gate c).

Eligibility is validated LOUDLY at construction (seam map §4 "Proposed M2
torch boundary"): still mode, PixelImage/RGBImage, batched|smart sampler,
plain Adam, and no semantic-mask / semantic-image-prompt / depth / video
features — each rejection names the backend that does support the config.
The torch and M1-mlx paths are untouched.

Structure changes (scene boundaries changing the prompt set, the interp
ramp ending) rebuild the compiled step; per-step values never do.

``mlx`` is imported at module level: import this module only lazily (via
the package ``__getattr__`` or inside function bodies) so linux CI never
touches mlx.
"""

import sys

import mlx.core as mx
import numpy as np
import torch
from loguru import logger
from torchvision.transforms import functional as TF

from pytti.eval_tools import is_zero_weight, parametric_eval
from pytti.image_models import PixelImage, RGBImage
from pytti.image_models.pixel import HdrLoss, PaletteLoss
from pytti.LossAug.EdgeLossClass import EdgeLoss
from pytti.LossAug.HSVLossClass import HSVLoss
from pytti.LossAug.MSELossClass import MSELoss
from pytti.LossAug.TVLossClass import TVLoss
from pytti.mlx_engine.augs import AugConfig
from pytti.mlx_engine.image_models import (
    pixel_params_from_state_dict,
    pixel_state_dict_from_params,
    rgb_params_from_state_dict,
    rgb_state_dict_from_params,
)
from pytti.mlx_engine.step import (
    DirectLossPlan,
    PromptPlan,
    StepConfig,
    build_step,
    make_adam,
    make_cutter,
    reset_adam_state,
    trainable_keys_for,
)
from pytti.Perceptor.mlx_backend.convert import MLX_VIT_MODELS, load_tower
from pytti.Perceptor.Prompt import Prompt
from pytti.prompt_spec import MaskAll, MaskGeometric, parse_prompt_spec
from pytti.rotoscoper import ROTOSCOPERS

_DIRECT_LOSS_KINDS = {TVLoss: "tv", MSELoss: "mse", HSVLoss: "hsv", EdgeLoss: "edge"}


def _reject(reason: str, alternative: str) -> "RuntimeError":
    return RuntimeError(
        f"perceptor_backend=mlx_full cannot run this config: {reason}. "
        f"{alternative}"
    )


def _to_mx_constant(name: str, tensor: torch.Tensor) -> mx.array:
    if not isinstance(tensor, torch.Tensor):
        raise ValueError(f"{name} is not a tensor: {type(tensor).__name__}")
    if tensor.dtype != torch.float32:
        raise ValueError(f"{name} must be fp32, got {tensor.dtype}")
    return mx.array(tensor.detach().cpu().contiguous().numpy())


def _scene_prompt_specs(params):
    """Re-run parse_scenes' splitting over the config's scene strings and
    yield the typed PromptSpec of every semantic prompt the run will parse."""
    prefix = params.scene_prefix or ""
    suffix = params.scene_suffix or ""
    for stage in str(params.scenes).split("||"):
        if not stage:
            continue
        for piece in (prefix + stage + suffix).strip().split("|"):
            if piece.strip():
                yield parse_prompt_spec(piece.strip())


class MLXStillEngine:
    """
    The whole-step MLX engine for one still-image run. One instance per
    ``DirectImageGuide``; owns params/Adam/RNG state end to end. The torch
    ``image_rep`` module stays the serialization vessel — its tensors are
    only read at construction and written by :meth:`write_back`.
    """

    def __init__(
        self,
        image_rep,
        embedder,
        params,
        lr: float,
        *,
        tower_dtype: str = "float16",
        cutter=None,
        tower_loader=None,
    ):
        if embedder is None:
            raise _reject(
                "no embedder (bare palette-fit guides stay torch)",
                "Use perceptor_backend=torch.",
            )
        self._validate_config(params, image_rep, embedder)

        self._image_rep = image_rep
        self._params_config = params
        self._lr = float(lr)
        self._cutter = cutter

        # ---- image tree (S1) ------------------------------------------------
        state_dict = image_rep.state_dict()
        if isinstance(image_rep, PixelImage):
            image_kind = "pixel"
            self._params = pixel_params_from_state_dict(state_dict)
            use_palette_target = bool(image_rep.use_palette_target)
            image_loss_kinds, image_loss_names = [], []
            for module in image_rep.image_loss():
                if isinstance(module, HdrLoss):
                    image_loss_kinds.append("hdr")
                elif isinstance(module, PaletteLoss):
                    image_loss_kinds.append("palette")
                else:
                    raise _reject(
                        f"unknown image loss {type(module).__name__}",
                        "Use perceptor_backend=mlx or torch.",
                    )
                image_loss_names.append(str(module))
            if ("hdr" in image_loss_kinds) != ("hdr_comp" in self._params):
                raise ValueError(
                    "hdr loss module and hdr tree buffers disagree — "
                    "the torch module is off-contract"
                )
        else:
            image_kind = "rgb"
            self._params = rgb_params_from_state_dict(state_dict)
            use_palette_target = False
            image_loss_kinds, image_loss_names = [], []
        self._image_kind = image_kind
        self._image_loss_names = tuple(image_loss_names)
        side_x, side_y = image_rep.image_shape

        # ---- towers (reuse M1 conversion cache + grouping) ------------------
        loader = tower_loader
        if loader is None:
            perceptor_keys = [p.key for p in embedder.perceptors]
            unsupported = sorted(set(perceptor_keys) - set(MLX_VIT_MODELS))
            if unsupported:
                raise _reject(
                    f"perceptors {unsupported} have no MLX towers "
                    f"(supported: {sorted(MLX_VIT_MODELS)})",
                    "Deselect them or use perceptor_backend=torch.",
                )
            loader = load_tower
        perceptors = list(embedder.perceptors)
        # same-resolution towers share ONE sampler group (the torch
        # embedder's cut_size sharing); normalization is per-TOWER inside
        # the step because stats can differ within a group (FARE = CLIP
        # constants, SigLIP2 = 0.5s, both 224 — see build_step's docstring)
        group_of_size: dict[int, int] = {}
        group_leader: list[int] = []
        batch_index: list[int] = []
        for idx, perceptor in enumerate(perceptors):
            size = perceptor.cut_size
            if size not in group_of_size:
                group_of_size[size] = len(group_leader)
                group_leader.append(idx)
            batch_index.append(group_of_size[size])
        self._perceptors = perceptors
        self._towers = tuple(loader(p.key, tower_dtype) for p in perceptors)
        self._batch_index = tuple(batch_index)
        self._group_cut_sizes = tuple(perceptors[i].cut_size for i in group_leader)
        self._tower_means = tuple(
            mx.array(np.asarray(p.normalize.mean, dtype=np.float32))
            for p in perceptors
        )
        self._tower_stds = tuple(
            mx.array(np.asarray(p.normalize.std, dtype=np.float32))
            for p in perceptors
        )
        self._out_dim_max = max(t.config.output_dim for t in self._towers)

        # ---- static step config ---------------------------------------------
        self._base_cfg = dict(
            image_kind=image_kind,
            scale=int(image_rep.scale),
            use_palette_target=use_palette_target,
            image_loss_names=tuple(image_loss_kinds),
            side_x=int(side_x),
            side_y=int(side_y),
            cutn=int(embedder.cutn),
            cut_pow=float(embedder.cut_pow),
            padding=float(embedder.padding),
            border_mode=str(embedder.border_mode),
            noise_fac=float(embedder.noise_fac),
            sampler=str(embedder.cutout_sampler),
            coherence_weighting=bool(params.get("coherence_weighting", False)),
        )
        # aug stack config: parsed from the live torch module, not assumed
        augs = embedder.augs
        self._aug_config = AugConfig(
            p_flip=augs.p_flip,
            degrees=augs.degrees,
            translate=augs.translate,
            p_affine=augs.p_affine,
            distortion=augs.distortion,
            p_persp=augs.p_persp,
            hue=augs.hue,
            sat=augs.sat,
            p_jitter=augs.p_jitter,
            erase_scale=tuple(augs.erase_scale),
            erase_ratio=tuple(augs.erase_ratio),
            p_erase=augs.p_erase,
        )

        # ---- optimizer (S5) ---------------------------------------------------
        self._opt = make_adam(self._lr)
        self._opt.init(
            {k: self._params[k] for k in trainable_keys_for(image_kind)}
        )

        # lazily-built step machinery (structure follows the prompt set)
        self._step = None
        self._step_key = None
        self._direct_plans: tuple = ()
        self._direct_augs: tuple = ()
        self._prompt_meta: dict[int, tuple] = {}  # id -> (ref, embeds, kind, cutoff)

        seed = params.get("seed")
        if seed is not None:
            mx.random.seed(int(seed))
        logger.info(
            f"MLX whole-step engine ready: {[p.key for p in perceptors]} "
            f"({tower_dtype}), {image_kind} image {side_x}x{side_y}, "
            f"sampler={self._base_cfg['sampler']}, cutn={self._base_cfg['cutn']}."
        )

    # ------------------------------------------------------------------
    # eligibility (fail loud at construction — seam map §4)
    # ------------------------------------------------------------------

    @staticmethod
    def _validate_config(params, image_rep, embedder) -> None:
        if sys.platform != "darwin":  # pragma: no cover - mlx import fails first
            raise _reject("MLX is Metal-only", "Use perceptor_backend=torch.")
        if params.animation_mode != "off":
            raise _reject(
                f"animation_mode={params.animation_mode!r} (still mode only)",
                "Use perceptor_backend=mlx (the M1 bridge) for animation.",
            )
        if not isinstance(image_rep, (PixelImage, RGBImage)):
            raise _reject(
                f"image model {type(image_rep).__name__} "
                "(PixelImage/RGBImage only)",
                "Use perceptor_backend=mlx for VQGAN, torch for LlamaGen.",
            )
        if embedder.cutout_sampler not in ("batched", "smart"):
            raise _reject(
                f"cutout_sampler={embedder.cutout_sampler!r}",
                "The classic sampler is torch-only: use cutout_sampler="
                "batched|smart, or perceptor_backend=torch.",
            )
        if params.optimizer != "adam":
            raise _reject(
                f"optimizer={params.optimizer!r}",
                "The whole-step path supports plain Adam only: use "
                "optimizer=adam, or perceptor_backend=mlx|torch for adamw_sf.",
            )
        for key in (
            "semantic_stabilization_weight",
            "semantic_init_weight",
            "depth_stabilization_weight",
        ):
            if not is_zero_weight(params.get(key) or ""):
                raise _reject(
                    f"{key} is set (needs torch-side embeddings/models)",
                    "Use perceptor_backend=mlx.",
                )
        for spec in _scene_prompt_specs(params):
            if spec.image_path() is not None:
                raise _reject(
                    f"scene prompt {spec.prompt_string!r} is a semantic "
                    "image prompt",
                    "Use perceptor_backend=mlx.",
                )
            if not isinstance(spec.mask, (MaskAll, MaskGeometric)):
                raise _reject(
                    f"scene prompt {spec.prompt_string!r} carries a "
                    f"{type(spec.mask).__name__} (geometric masks only)",
                    "Use perceptor_backend=mlx for image masks, torch for "
                    "semantic masks.",
                )
        # Video masks anywhere (prompts or direct losses) registered a
        # rotoscoper at parse time; rotoscopers advance at frame boundaries
        # even in still mode (ImageGuide.update:420-423), so baked mask
        # constants would go stale. Direct-loss IMAGE masks are fine — they
        # cross once as constants.
        if ROTOSCOPERS.rotoscopers:
            raise _reject(
                "video-mask rotoscopers are active",
                "Use perceptor_backend=mlx or torch.",
            )

    # ------------------------------------------------------------------
    # plan building (parse torch objects at the boundary, once)
    # ------------------------------------------------------------------

    def _direct_plan(self, aug) -> DirectLossPlan:
        kind = _DIRECT_LOSS_KINDS.get(type(aug))
        if kind is None:
            hint = (
                "Use perceptor_backend=mlx."
                if type(aug).__name__ == "DepthLoss"
                else "Use perceptor_backend=mlx or torch."
            )
            raise _reject(f"loss aug {type(aug).__name__} ({aug})", hint)
        if kind == "tv":
            return DirectLossPlan(name=str(aug), kind=kind, comp=None, mask=None)
        comp = _to_mx_constant(f"{aug}.comp", aug.comp)
        mask = None
        if aug.use_mask:
            mask_t = aug.mask
            height, width = self._base_cfg["side_y"], self._base_cfg["side_x"]
            if mask_t.shape[-2:] != (height, width):
                # the torch class's once-per-shape lazy resize
                # (MSELossClass.get_loss), moved to setup
                mask_t = TF.resize(mask_t, [height, width])
            mask = _to_mx_constant(f"{aug}.mask", mask_t)
        return DirectLossPlan(name=str(aug), kind=kind, comp=comp, mask=mask)

    def _prompt_meta_for(self, prompt) -> tuple:
        if type(prompt) is not Prompt:
            raise _reject(
                f"prompt {prompt!r} is a {type(prompt).__name__} (semantic "
                "image prompts / stabilization need torch-side embeddings)",
                "Use perceptor_backend=mlx.",
            )
        if getattr(prompt.mask, "embed_dependent", False):
            raise _reject(
                f"prompt {prompt!r} has a semantic [mask]",
                "Use perceptor_backend=torch.",
            )
        cached = self._prompt_meta.get(id(prompt))
        if cached is not None and cached[0] is prompt:
            return cached
        spec = parse_prompt_spec(prompt.prompt_string)
        if isinstance(spec.mask, MaskAll):
            mask_kind = "a"
        elif isinstance(spec.mask, MaskGeometric):
            mask_kind = spec.mask.key
        else:
            raise _reject(
                f"prompt {prompt!r} carries a {type(spec.mask).__name__}",
                "Use perceptor_backend=mlx for image masks, torch for "
                "semantic masks.",
            )
        embeds = prompt.embeds.detach().float()
        expected = (len(self._perceptors), self._out_dim_max)
        if tuple(embeds.shape) != expected:
            raise RuntimeError(
                f"prompt {prompt!r} carries embeddings of shape "
                f"{tuple(embeds.shape)}, expected {expected} — it was parsed "
                "under a different perceptor ensemble."
            )
        meta = (
            prompt,
            _to_mx_constant(f"{prompt}.embeds", embeds),
            mask_kind,
            spec.cutoff,
        )
        self._prompt_meta[id(prompt)] = meta
        return meta

    def _ensure_step(self, active_augs, active_prompts, gas: int) -> None:
        key = (
            tuple(id(aug) for aug in active_augs),
            tuple((id(p), recorded) for p, recorded in active_prompts),
            gas,
        )
        if key == self._step_key and self._step is not None:
            return
        self._direct_augs = tuple(active_augs)
        self._direct_plans = tuple(self._direct_plan(aug) for aug in active_augs)
        prompt_plans = []
        for prompt, recorded in active_prompts:
            _, embeds, mask_kind, _cutoff = self._prompt_meta_for(prompt)
            prompt_plans.append(
                PromptPlan(
                    name=str(prompt),
                    embeds=embeds,
                    mask_kind=mask_kind,
                    recorded=recorded,
                )
            )
        cfg = StepConfig(gas=gas, **self._base_cfg)
        cutter = (
            self._cutter
            if self._cutter is not None
            else make_cutter(cfg, self._aug_config)
        )
        self._step = build_step(
            params=self._params,
            opt=self._opt,
            cfg=cfg,
            direct_plans=self._direct_plans,
            prompt_plans=tuple(prompt_plans),
            towers=self._towers,
            batch_index=self._batch_index,
            group_cut_sizes=self._group_cut_sizes,
            tower_means=self._tower_means,
            tower_stds=self._tower_stds,
            cutter=cutter,
        )
        self._step_key = key

    # ------------------------------------------------------------------
    # the train() replacement
    # ------------------------------------------------------------------

    def train_step(
        self,
        i: int,
        prompts,
        interp_prompts,
        loss_augs,
        *,
        interp_steps: int = 0,
        gradient_accumulation_steps: int = 1,
        palette_gate: float = 1.0,
    ) -> dict:
        """
        One optimizer step — ``DirectImageGuide.train``'s contract: returns
        the step record ``{name: 0-dim scalar, ..., "TOTAL": total}`` with
        the EXACT torch record names, values lazy until reporting.

        ``palette_gate`` is phase scheduling's Limited Palette lock as a
        per-step host argument: 1.0 = palette trains, 0.0 = its gradient
        and update are gated off in-step (never a retrace). The guide
        computes it from the schedule table (pytti/phase_scheduling.py).
        """
        if palette_gate not in (0.0, 1.0):
            raise ValueError(
                f"palette_gate must be 0.0 or 1.0, got {palette_gate!r} — "
                "it is phase scheduling's lock gate, not a soft weight"
            )
        t = i / interp_steps if i < interp_steps else 1.0

        active_augs = [
            aug
            for aug in loss_augs
            if aug.enabled and not is_zero_weight(aug.weight)
        ]
        active_prompts: list[tuple] = []
        for prompt in prompts:
            self._prompt_meta_for(prompt)  # type/mask gates apply even if idle
            if prompt.enabled and not is_zero_weight(prompt.weight):
                active_prompts.append((prompt, True))
        if i < interp_steps:
            for prompt in interp_prompts:
                self._prompt_meta_for(prompt)
                if prompt.enabled and not is_zero_weight(prompt.weight):
                    active_prompts.append((prompt, False))

        self._ensure_step(active_augs, active_prompts, gradient_accumulation_steps)

        def host_vector(values) -> mx.array:
            return mx.array(np.asarray(values, dtype=np.float32))

        # weight_scale is the phase-scheduling multiplier the guide sets on
        # the Loss objects (1.0 when off) — folded into the evaluated
        # weight exactly like torch Loss.forward does
        aug_w = host_vector(
            [parametric_eval(a.weight) * a.weight_scale for a in active_augs]
        )
        aug_s = host_vector([parametric_eval(a.stop) for a in active_augs])
        p_w, p_s, p_thresh, p_scale = [], [], [], []
        for prompt, recorded in active_prompts:
            _, _, _, cutoff = self._prompt_meta_for(prompt)
            p_w.append(parametric_eval(prompt.weight))
            p_s.append(parametric_eval(prompt.stop))
            p_thresh.append(parametric_eval(cutoff))
            p_scale.append(t if recorded else 1.0 - t)

        total, aug_raws, image_raws, prompt_raws = self._step(
            aug_w,
            aug_s,
            host_vector(p_w),
            host_vector(p_s),
            host_vector(p_thresh),
            host_vector(p_scale),
            mx.array(np.float32(palette_gate)),
        )
        # one dispatch per step: materialize the new params/moments/records
        # (the MLX execution cadence — this is not a torch-style mid-graph
        # host sync; records stay unformatted until _report_losses)
        mx.eval(
            self._params, self._opt.state, total, aug_raws, image_raws, prompt_raws
        )

        # records in train()'s insertion order: loss augs, image losses,
        # prompts, TOTAL — zeros for idle entries, exactly like torch
        record: dict = {}
        aug_pos = {id(aug): j for j, aug in enumerate(active_augs)}
        for aug in loss_augs:
            j = aug_pos.get(id(aug))
            record[str(aug)] = mx.zeros(()) if j is None else aug_raws[j]
        for j, name in enumerate(self._image_loss_names):
            record[name] = image_raws[j]
        prompt_pos = {
            id(p): j for j, (p, recorded) in enumerate(active_prompts) if recorded
        }
        for prompt in prompts:
            j = prompt_pos.get(id(prompt))
            record[str(prompt)] = mx.zeros(()) if j is None else prompt_raws[j]
        record["TOTAL"] = total
        return record

    # ------------------------------------------------------------------
    # frame-boundary + save/restore seams
    # ------------------------------------------------------------------

    def reset_optimizer(self) -> None:
        """``set_optim(None)`` under mlx_full: fresh-Adam semantics
        (zero moments, zero step count), state structure preserved."""
        reset_adam_state(self._opt)

    def write_back(self, image_rep) -> None:
        """Load the MLX params tree into the torch module (the ONE torch
        exit): ``_save_frame``'s decode/PNG/breath/``.bak`` all read the
        module, so ``.bak`` stays torch-serialized and restore-compatible."""
        if image_rep is not self._image_rep:
            raise ValueError(
                "write_back called with a different image_rep than the one "
                "this engine was built from"
            )
        if self._image_kind == "pixel":
            state_dict = pixel_state_dict_from_params(self._params)
        else:
            state_dict = rgb_state_dict_from_params(self._params)
        image_rep.load_state_dict(state_dict)
