import math

from torch import nn, optim
from tqdm import tqdm

from pytti import format_input
from pytti.AudioParse import SpectralAudioParser
from pytti.image_models.differentiable_image import DifferentiableImage


def unpack_dict(D, n=2):
    """
    Given a dictionary D whose values are n-tuples, return a tuple of n
    dictionaries, each mapping the same keys to one tuple slot.
    """
    ds = [{k: V[i] for k, V in D.items()} for i in range(n)]
    return tuple(ds)


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
        z = self.image_rep.decode_training_tensor()
        losses = []

        aug_losses = {
            aug: aug(format_input(z, self.image_rep, aug), self.image_rep)
            for aug in loss_augs
        }

        image_augs = self.image_rep.image_loss()
        image_losses = {aug: aug(self.image_rep) for aug in image_augs}

        losses, losses_raw = [], []
        # NB(known bug, fix in correctness slice): total_loss is never
        # accumulated across the microbatch loop, so the reported TOTAL is
        # always 0 and `stop` can never trigger. Preserved as-is for now.
        total_loss = 0

        for mb_i in range(gradient_accumulation_steps):
            t = 1
            interp_losses = [0]
            prompt_losses = {}
            if self.embedder is not None:
                image_embeds, offsets, sizes = self.embedder(self.image_rep, input=z)

                if i < interp_steps:
                    t = i / interp_steps
                    interp_losses = [
                        prompt(
                            format_input(image_embeds, self.embedder, prompt),
                            format_input(offsets, self.embedder, prompt),
                            format_input(sizes, self.embedder, prompt),
                        )[0]
                        * (1 - t)
                        for prompt in interp_prompts
                    ]

                prompt_losses = {
                    prompt: prompt(
                        format_input(image_embeds, self.embedder, prompt),
                        format_input(offsets, self.embedder, prompt),
                        format_input(sizes, self.embedder, prompt),
                    )
                    for prompt in prompts
                }

            losses, losses_raw = zip(
                *map(unpack_dict, [prompt_losses, aug_losses, image_losses])
            )
            losses = list(losses)
            losses_raw = list(losses_raw)

            for v in prompt_losses.values():
                v[0].mul_(t)

            total_loss_mb = sum(map(lambda x: sum(x.values()), losses)) + sum(
                interp_losses
            )

            total_loss_mb /= gradient_accumulation_steps
            total_loss_mb.backward(retain_graph=True)

        losses_raw.append({"TOTAL": total_loss})
        self.optimizer.step()
        self.image_rep.update()
        self.optimizer.zero_grad()

        if save_loss:
            self.loss_history.append(
                {
                    str(k): float(v)
                    for loss_dict in losses_raw
                    for k, v in loss_dict.items()
                }
            )

        return {"TOTAL": float(total_loss)}

    def update(self, model, img, i, stage_i, *args, **kwargs):
        """
        update hook called every step
        """
        pass
