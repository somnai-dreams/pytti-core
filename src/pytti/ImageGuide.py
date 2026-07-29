import math
from pathlib import Path

import torch
from loguru import logger
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
        **optimizer_params,
    ):
        self.image_rep = image_rep
        self.embedder = embedder
        if lr is None:
            lr = image_rep.lr
        optimizer_params["lr"] = lr
        self.optimizer_params = optimizer_params
        if optimizer is None:
            self.optimizer = optim.Adam(image_rep.parameters(), **optimizer_params)
        else:
            self.optimizer = optimizer

        # per-step loss records for the current scene: list of {name: value}
        self.loss_history: list[dict[str, float]] = []

        self.audio_parser = None
        if params is not None:
            if params.input_audio and params.input_audio_filters:
                self.audio_parser = SpectralAudioParser(
                    params.input_audio,
                    params.input_audio_offset,
                    params.frames_per_second,
                    params.input_audio_filters,
                )

        self.params = params
        self.base_name = base_name
        self.video_frames = video_frames
        self.optical_flows = optical_flows
        self.stabilization_augs = stabilization_augs
        self.last_frame_semantic = last_frame_semantic
        self.semantic_init_prompt = semantic_init_prompt
        self.init_augs = init_augs

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
            if losses["TOTAL"] <= stop:
                break
        return steps_run

    def set_optim(self, opt=None):
        if opt is not None:
            self.optimizer = opt
        else:
            self.optimizer = optim.Adam(
                self.image_rep.parameters(), **self.optimizer_params
            )

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
        self.optimizer.zero_grad()
        total_loss = 0.0
        step_record: dict[str, float] = {}

        # interpolation ramp: prompts fade in (t) while the previous scene's
        # prompts fade out (1 - t)
        t = i / interp_steps if i < interp_steps else 1

        # ---- loss augs + image-model losses: one backward pass of their own.
        # (Previously these were computed once but backwarded once per
        # microbatch through a retained graph, holding memory and reporting
        # a TOTAL of zero.)
        z = self.image_rep.decode_training_tensor()
        aug_losses = {
            aug: aug(format_input(z, self.image_rep, aug), self.image_rep)
            for aug in loss_augs
        }
        image_losses = {aug: aug(self.image_rep) for aug in self.image_rep.image_loss()}

        aug_total = 0
        for name_losses in (aug_losses, image_losses):
            for aug, (loss, loss_raw) in name_losses.items():
                aug_total = aug_total + loss
                step_record[str(aug)] = float(loss_raw)
        if isinstance(aug_total, torch.Tensor) and aug_total.requires_grad:
            aug_total.backward()
        total_loss += float(aug_total)

        # ---- prompt (CLIP) losses: fresh cutouts per microbatch
        if self.embedder is not None:
            for _ in range(gradient_accumulation_steps):
                z_mb = self.image_rep.decode_training_tensor()
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
                    step_record[str(prompt)] = float(loss_raw)

                mb_total = (prompt_total + interp_total) / gradient_accumulation_steps
                if isinstance(mb_total, torch.Tensor) and mb_total.requires_grad:
                    mb_total.backward()
                total_loss += float(mb_total)

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
                logger.debug(f"  {name}: {value:.6f}")
        if self.params.approximate_vram_usage:
            logger.debug("VRAM Usage:")
            print_vram_usage()

    def _save_frame(self, i):
        img = self.image_rep
        params = self.params
        # NB: computed at call time — hydra chdirs into the run's output
        # directory before the render starts
        outpath = Path.cwd() / "images_out"
        im = img.decode_image()
        n = (i + 1) // params.save_every
        frame_dir = outpath / params.file_namespace
        frame_dir.mkdir(parents=True, exist_ok=True)
        im.save(frame_dir / f"{self.base_name}_{n}.png")

        if params.backups > 0:
            backup_dir = Path("backup") / params.file_namespace
            backup_dir.mkdir(parents=True, exist_ok=True)
            torch.save(img.state_dict(), backup_dir / f"{self.base_name}_{n}.bak")
            if n > params.backups:
                stale = backup_dir / f"{self.base_name}_{n - params.backups}.bak"
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
