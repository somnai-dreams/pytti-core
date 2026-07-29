import math

import torch
from torch import nn, optim
from tqdm import tqdm

from pytti import format_input
from pytti.AudioParse import SpectralAudioParser
from pytti.image_models.differentiable_image import DifferentiableImage


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
            self.update(
                model=self,
                img=self.image_rep,
                i=i + i_offset,
                stage_i=i + skipped_steps,
                params=self.params,
                base_name=self.base_name,
                optical_flows=self.optical_flows,
                video_frames=self.video_frames,
                stabilization_augs=self.stabilization_augs,
                last_frame_semantic=self.last_frame_semantic,
                embedder=self.embedder,
                init_augs=self.init_augs,
                semantic_init_prompt=self.semantic_init_prompt,
            )
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

    def update(self, model, img, i, stage_i, *args, **kwargs):
        """
        update hook called every step
        """
        pass
