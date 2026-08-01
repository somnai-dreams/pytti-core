import math
from contextlib import contextmanager
from pathlib import Path

import schedulefree
import torch
from loguru import logger
from PIL import Image
from torch import nn, optim
from tqdm import tqdm

from pytti import (
    format_input,
    freeze_vram_usage,
    parametric_eval,
    print_vram_usage,
    set_t,
    vram_usage_mode,
)
from pytti.AudioParse import SpectralAudioParser
from pytti.image_models.differentiable_image import DifferentiableImage
from pytti.rotoscoper import update_rotoscopers
from pytti.Transforms import animate_video_source, zoom_2d, zoom_3d


def frame_filename(base_name: str, n: int) -> str:
    """Zero-padded frame name: sorts correctly and feeds ffmpeg %04d."""
    return f"{base_name}_{n:04d}.png"


def make_optimizer(params_iterable, config_optimizer: str, lr, **optimizer_params):
    """
    Build the training optimizer.

    'adam'     -> torch.optim.Adam, exactly the legacy behavior.
    'adamw_sf' -> schedule-free AdamW (Defazio et al., NeurIPS 2024): no LR
                  schedule needed, and the eval-mode Polyak-averaged iterate
                  is what output frames must be decoded from. Returned in
                  train mode, ready to step.
    """
    if config_optimizer == "adam":
        return optim.Adam(params_iterable, lr=lr, **optimizer_params)
    if config_optimizer == "adamw_sf":
        # warmup_steps=10: frames are ~50-step optimization bursts and
        # reset_lr_each_frame reconstructs the optimizer per frame, so a
        # short warmup ramp is re-run at every frame boundary.
        opt = schedulefree.AdamWScheduleFree(
            params_iterable, lr=lr, warmup_steps=10, **optimizer_params
        )
        opt.train()
        return opt
    raise ValueError(
        f"Unknown optimizer {config_optimizer!r}: expected 'adam' or 'adamw_sf'"
    )


def breath_alpha(n: int, num_scenes: int, steps_per_scene: int, save_every: int) -> float:
    """
    Blend factor for breath mode at frame n: 0.0 (pure init image) at the
    start ramping linearly to 1.0 (fully optimized) at the final frame.
    """
    total_frames = max(1, (num_scenes * steps_per_scene) // save_every)
    return min(n / total_frames, 1.0)


class DirectImageGuide:
    """
    Image guide that uses an optimizer and torch autograd to optimize an image representation
    Based on the BigGan+CLIP algorithm by advadnoun (https://twitter.com/advadnoun)
    image_rep: (DifferentiableImage) image representation
    embedder: (Module)               image embedder
    optimizer: (Class)               optimizer class to use. Defaults to Adam
    all other arguments are passed as kwargs to the optimizer.
    """

    def __init__(
        self,
        image_rep: DifferentiableImage,
        embedder: nn.Module,
        optimizer: optim.Optimizer = None,
        lr: float = None,
        params=None,
        base_name=None,
        video_frames=None,
        optical_flows=None,
        stabilization_augs=None,
        last_frame_semantic=None,
        semantic_init_prompt=None,
        init_augs=None,
        init_image_pil=None,
        **optimizer_params,
    ):
        self.image_rep = image_rep
        self.embedder = embedder
        self.params = params
        if lr is None:
            lr = image_rep.lr
        self.lr = lr
        self.optimizer_params = optimizer_params
        # bare guides (params=None, e.g. PixelImage's palette fit) always
        # use plain Adam
        self.optimizer_name = (
            "adam" if params is None else params.get("optimizer", "adam")
        )
        if optimizer is None:
            self.optimizer = make_optimizer(
                image_rep.parameters(), self.optimizer_name, lr, **optimizer_params
            )
        else:
            self.optimizer = optimizer

        # per-step loss records for the current scene: list of {name: value}
        # values are detached 0-dim scalars (torch tensors, or mx arrays
        # under perceptor_backend=mlx_full); format at report time only
        self.loss_history: list[dict] = []

        # perceptor_backend=mlx: the semantic losses (CLIP tower fwd+bwd +
        # prompt reduction) run on MLX via the bridge; samplers, augs, image
        # models, and the optimizer stay torch. None = the torch path.
        # perceptor_backend=mlx_full: the ENTIRE step runs on MLX (M2 —
        # docs/mlx-m2-seam-map.md); train() delegates to the engine and the
        # torch optimizer above is never stepped. Both None = the torch path.
        self.mlx_semantic_loss = None
        self.mlx_engine = None
        backend = (
            params.get("perceptor_backend", "torch")
            if params is not None
            else "torch"
        )
        if embedder is not None and backend == "mlx":
            # local import: only the mlx path pays for loading the towers
            from pytti.Perceptor.mlx_backend.bridge import MLXSemanticLoss

            self.mlx_semantic_loss = MLXSemanticLoss(embedder)
        elif embedder is not None and backend == "mlx_full":
            # local import: mlx is darwin-only, imported lazily
            from pytti.mlx_engine.engine import MLXStillEngine

            # fails loud at construction on any config the whole-step
            # engine can't take (animation, VQGAN, classic sampler, ...)
            self.mlx_engine = MLXStillEngine(
                image_rep=image_rep,
                embedder=embedder,
                params=params,
                lr=self.lr,
            )

        self.audio_parser = None
        if params is not None:
            if params.input_audio and params.input_audio_filters:
                self.audio_parser = SpectralAudioParser(
                    params.input_audio,
                    params.input_audio_offset,
                    params.frames_per_second,
                    params.input_audio_filters,
                )

        self.base_name = base_name
        self.video_frames = video_frames
        self.optical_flows = optical_flows
        self.stabilization_augs = stabilization_augs
        self.last_frame_semantic = last_frame_semantic
        self.semantic_init_prompt = semantic_init_prompt
        self.init_augs = init_augs
        self.init_image_pil = init_image_pil

    def run_steps(
        self,
        n_steps,
        prompts,
        interp_prompts,
        loss_augs,
        stop=-math.inf,
        interp_steps=0,
        i_offset=0,
        skipped_steps=0,
        gradient_accumulation_steps: int = 1,
    ):
        """
        runs the optimizer
        prompts: (ClipPrompt list) list of prompts
        n_steps: (positive integer) steps to run
        returns: the number of steps run
        """
        steps_run = 0
        for i in tqdm(range(n_steps)):
            self.update(i + i_offset, i + skipped_steps)
            losses = self.train(
                i + skipped_steps,
                prompts,
                interp_prompts,
                loss_augs,
                interp_steps=interp_steps,
                gradient_accumulation_steps=gradient_accumulation_steps,
            )
            steps_run = i + 1
            # only pay the device sync when an early-stop is actually set
            if stop != -math.inf and float(losses["TOTAL"]) <= stop:
                break
        return steps_run

    def set_optim(self, opt=None):
        if opt is not None:
            self.optimizer = opt
        elif self.mlx_engine is not None:
            # same cadence, MLX state: zero Adam moments + step count
            self.mlx_engine.reset_optimizer()
        else:
            # reset_lr_each_frame lands here once per frame; make_optimizer
            # re-enters train mode for schedule-free optimizers every time
            self.optimizer = make_optimizer(
                self.image_rep.parameters(),
                self.optimizer_name,
                self.lr,
                **self.optimizer_params,
            )

    @contextmanager
    def optimizer_eval(self):
        """
        Decode-for-output context. Schedule-free optimizers hold the fast
        (extrapolated) iterate in the parameters during training and only
        swap in the Polyak-averaged iterate on .eval() — decoding a frame
        outside this context silently emits the un-averaged image. No-op
        passthrough for plain Adam.
        """
        opt = self.optimizer
        if isinstance(opt, schedulefree.AdamWScheduleFree):
            opt.eval()
            try:
                yield
            finally:
                opt.train()
        else:
            yield

    def clear_loss_history(self):
        self.loss_history = []

    def train(
        self,
        i,
        prompts,
        interp_prompts,
        loss_augs,
        interp_steps=0,
        save_loss=True,
        gradient_accumulation_steps: int = 1,
    ):
        """
        steps the optimizer
        promts: (ClipPrompt list) list of prompts
        """
        if self.mlx_engine is not None:
            # M2 whole-step engine: decode -> cutouts -> towers -> losses ->
            # Adam all inside one compiled MLX function. Records keep the
            # torch names; values are lazy mx scalars (float() at reporting
            # only, preserving the no-mid-step-sync rule below).
            step_record = self.mlx_engine.train_step(
                i,
                prompts,
                interp_prompts,
                loss_augs,
                interp_steps=interp_steps,
                gradient_accumulation_steps=gradient_accumulation_steps,
            )
            if save_loss:
                self.loss_history.append(step_record)
            return {"TOTAL": step_record["TOTAL"]}

        self.optimizer.zero_grad()
        total_loss = None
        # Values stay 0-dim device tensors until a reporting path formats
        # them: float(tensor) is a device sync, and syncing mid-step flushes
        # the half-built MPS/CUDA command queue — measured ~800ms of pure
        # serialization per step at 512px (step 1.89s -> 1.09s without it).
        step_record: dict[str, torch.Tensor] = {}

        # interpolation ramp: prompts fade in (t) while the previous scene's
        # prompts fade out (1 - t)
        t = i / interp_steps if i < interp_steps else 1

        # One decode per microbatch; the aug/image-model losses ride along
        # with the first microbatch's graph so each step needs exactly
        # gradient_accumulation_steps decodes and backwards (no retained
        # graphs, no extra decode).
        microbatches = gradient_accumulation_steps if self.embedder is not None else 1
        for mb_i in range(microbatches):
            z_mb = self.image_rep.decode_training_tensor()
            mb_total = 0

            if mb_i == 0:
                for aug in loss_augs:
                    loss, loss_raw = aug(
                        format_input(z_mb, self.image_rep, aug), self.image_rep
                    )
                    mb_total = mb_total + loss
                    step_record[str(aug)] = loss_raw.detach()
                for aug in self.image_rep.image_loss():
                    loss, loss_raw = aug(self.image_rep)
                    mb_total = mb_total + loss
                    step_record[str(aug)] = loss_raw.detach()

            if self.embedder is not None and self.mlx_semantic_loss is not None:
                # MLX bridge: one fused fwd+bwd for all semantic prompts.
                # The interpolation ramp is folded in as per-prompt constants
                # (see bridge.py); records keep the torch path's names.
                semantic_total, semantic_records = self.mlx_semantic_loss(
                    self.image_rep,
                    z_mb,
                    prompts,
                    interp_prompts if i < interp_steps else [],
                    ramp=t,
                )
                step_record.update(semantic_records)
                mb_total = mb_total + semantic_total / gradient_accumulation_steps
            elif self.embedder is not None:
                image_embeds, offsets, sizes = self.embedder(
                    self.image_rep, input=z_mb
                )

                interp_total = 0
                if i < interp_steps:
                    for prompt in interp_prompts:
                        loss, _ = prompt(
                            format_input(image_embeds, self.embedder, prompt),
                            format_input(offsets, self.embedder, prompt),
                            format_input(sizes, self.embedder, prompt),
                        )
                        interp_total = interp_total + loss * (1 - t)

                prompt_total = 0
                for prompt in prompts:
                    loss, loss_raw = prompt(
                        format_input(image_embeds, self.embedder, prompt),
                        format_input(offsets, self.embedder, prompt),
                        format_input(sizes, self.embedder, prompt),
                    )
                    prompt_total = prompt_total + loss * t
                    step_record[str(prompt)] = loss_raw.detach()

                mb_total = mb_total + (
                    prompt_total + interp_total
                ) / gradient_accumulation_steps

            if isinstance(mb_total, torch.Tensor) and mb_total.requires_grad:
                mb_total.backward()
                mb_detached = mb_total.detach()
            else:
                mb_detached = torch.as_tensor(float(mb_total))
            total_loss = (
                mb_detached if total_loss is None else total_loss + mb_detached
            )

        self.optimizer.step()
        self.image_rep.update()
        self.optimizer.zero_grad()

        step_record["TOTAL"] = total_loss
        if save_loss:
            self.loss_history.append(step_record)

        return {"TOTAL": total_loss}

    # ------------------------------------------------------------------
    # per-step reporting / saving / animation
    # ------------------------------------------------------------------

    def _report_losses(self, i):
        logger.debug(f"Step {i} losses:")
        if self.loss_history:
            for name, value in self.loss_history[-1].items():
                logger.debug(f"  {name}: {float(value):.6f}")
        if self.params.approximate_vram_usage:
            logger.debug("VRAM Usage:")
            print_vram_usage()

    def _save_frame(self, i):
        # The ONE save path: everything decoded/serialized here must see the
        # averaged (eval) iterate under adamw_sf — the .bak included, so a
        # restore reproduces the frame that was saved.
        with self.optimizer_eval():
            img = self.image_rep
            params = self.params
            if self.mlx_engine is not None:
                # params live on MLX between saves: load them into the torch
                # module so decode/PNG/breath/.bak (and restore) see them
                self.mlx_engine.write_back(img)
            # NB: computed at call time — hydra chdirs into the run's output
            # directory before the render starts
            outpath = Path.cwd() / "images_out"
            im = img.decode_image()
            n = (i + 1) // params.save_every

            if params.breath_mode and self.init_image_pil is not None:
                # crossfade from the init image to the optimized output over
                # the whole render: frame 1 is (almost) the source, the last
                # frame is fully optimized
                num_scenes = max(
                    1, len([s for s in params.scenes.split("||") if s.strip()])
                )
                alpha = breath_alpha(
                    n, num_scenes, params.steps_per_scene, params.save_every
                )
                init_resized = self.init_image_pil.resize(im.size, Image.LANCZOS)
                im = Image.blend(init_resized, im, alpha=alpha)

            frame_dir = outpath / params.file_namespace
            frame_dir.mkdir(parents=True, exist_ok=True)
            im.save(frame_dir / frame_filename(self.base_name, n))

            if params.backups > 0:
                backup_dir = Path("backup") / params.file_namespace
                backup_dir.mkdir(parents=True, exist_ok=True)
                torch.save(
                    img.state_dict(), backup_dir / f"{self.base_name}_{n:04d}.bak"
                )
                if n > params.backups:
                    stale = (
                        backup_dir / f"{self.base_name}_{n - params.backups:04d}.bak"
                    )
                    if stale.exists():
                        stale.unlink()

    def update(self, i, stage_i):
        """
        Called once per optimization step, before train(): reporting, frame
        saving, and animation-frame advancement.
        """
        params = self.params
        if params is None:
            # bare guide (e.g. PixelImage.encode_image's palette fit)
            return
        img = self.image_rep

        j = i + 1
        if (params.display_every > 0) and (j % params.display_every == 0):
            self._report_losses(i)
        if (i > 0) and (params.save_every > 0) and (j % params.save_every == 0):
            self._save_frame(i)

        # animate
        ################
        t = (i - params.pre_animation_steps) / (
            params.steps_per_frame * params.frames_per_second
        )
        # advance expression time every step; bands only roll at frame
        # boundaries (so <band>_prev really is the previous frame's value)
        set_t(t)
        if i < params.pre_animation_steps:
            return
        if (i - params.pre_animation_steps) % params.steps_per_frame != 0:
            return

        if self.audio_parser is not None:
            band_dict = self.audio_parser.get_params(t)
            logger.debug(f"Time: {t:.4f} seconds, audio params: {band_dict}")
            set_t(t, band_dict)
        else:
            logger.debug(f"Time: {t:.4f} seconds")

        update_rotoscopers(
            ((i - params.pre_animation_steps) // params.steps_per_frame + 1)
            * params.frame_stride
        )
        if params.reset_lr_each_frame:
            self.set_optim(None)

        if params.animation_mode == "2D":
            tx, ty = parametric_eval(params.translate_x), parametric_eval(
                params.translate_y
            )
            theta = parametric_eval(params.rotate_2d)
            zx, zy = parametric_eval(params.zoom_x_2d), parametric_eval(
                params.zoom_y_2d
            )
            logger.debug(f"Translate: {tx}, {ty}  Rotate: {theta}  Zoom: {zx}, {zy}")

            next_step_pil = zoom_2d(
                img,
                (tx, ty),
                (zx, zy),
                theta,
                border_mode=params.infill_mode,
                sampling_mode=params.sampling_mode,
            )
        elif params.animation_mode == "3D":
            im = img.decode_image()
            with vram_usage_mode("Optical Flow Loss"):
                flow, next_step_pil = zoom_3d(
                    img,
                    (
                        params.translate_x,
                        params.translate_y,
                        params.translate_z_3d,
                    ),
                    params.rotate_3d,
                    params.field_of_view,
                    params.near_plane,
                    params.far_plane,
                    border_mode=params.infill_mode,
                    sampling_mode=params.sampling_mode,
                    stabilize=params.lock_camera,
                    device=params.device,
                )
                freeze_vram_usage()

            for optical_flow in self.optical_flows:
                optical_flow.set_last_step(im)
                optical_flow.set_target_flow(flow)
                optical_flow.set_enabled(True)
        elif params.animation_mode == "Video Source":
            _flow_im, next_step_pil = animate_video_source(
                i=i,
                img=img,
                video_frames=self.video_frames,
                optical_flows=self.optical_flows,
                base_name=self.base_name,
                pre_animation_steps=params.pre_animation_steps,
                frame_stride=params.frame_stride,
                steps_per_frame=params.steps_per_frame,
                file_namespace=params.file_namespace,
                reencode_each_frame=params.reencode_each_frame,
                lock_palette=params.lock_palette,
                save_every=params.save_every,
                infill_mode=params.infill_mode,
                sampling_mode=params.sampling_mode,
                device=params.device,
            )

        if params.animation_mode != "off":
            for aug in self.stabilization_augs:
                aug.set_comp(next_step_pil)
                aug.set_enabled(True)
            if self.last_frame_semantic is not None:
                self.last_frame_semantic.set_image(self.embedder, next_step_pil)
                self.last_frame_semantic.set_enabled(True)
            for aug in self.init_augs:
                aug.set_enabled(False)
            if self.semantic_init_prompt is not None:
                self.semantic_init_prompt.set_enabled(False)
