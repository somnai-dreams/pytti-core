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
from pytti.image_models.pixel import PixelImage
from pytti.image_models.rgb_image import RGBImage
from pytti.LossAug.TVLossClass import TVLoss
from pytti.phase_scheduling import (
    init_weight_scale,
    palette_locked,
    scene_t_hat,
    tv_weight_scale,
)
from pytti.rotoscoper import update_rotoscopers
from pytti.structure_annealing import anneal_image_rep, anneal_schedule
from pytti.Transforms import animate_video_source, zoom_2d, zoom_3d


def frame_filename(base_name: str, n: int) -> str:
    """Zero-padded frame name: sorts correctly and feeds ffmpeg %04d."""
    return f"{base_name}_{n:04d}.png"


# auto_stop: steps between TOTAL-loss samples. Each sample is one float()
# host sync; every 10 steps it amortizes to nothing (the per-step hot path
# stays sync-free — see the comment in DirectImageGuide.train).
AUTO_STOP_CHECK_INTERVAL = 10


def plateau_improvement(samples, window_samples: int) -> float | None:
    """
    Relative improvement of a loss series over its trailing window.

    Pure function — the whole convergence decision lives here so it is
    testable against synthetic loss sequences without a render.

    samples: TOTAL-loss samples, oldest first (one per check interval).
    window_samples: trailing window length in samples; must be >= 2.

    Returns None until one full window of samples exists (a partial window
    is never judged). Otherwise the trailing window is split in half and

        improvement = (mean(older half) - mean(newer half))
                      / max(|mean(older half)|, 1e-8)

    Positive = still improving; near zero or negative = plateaued (or
    regressing — either way, no longer converging). The half-means make the
    verdict robust to per-sample noise from the stochastic cutouts.
    """
    if window_samples < 2:
        raise ValueError(
            "window_samples must be >= 2 to measure improvement, "
            f"got {window_samples}"
        )
    if len(samples) < window_samples:
        return None
    window = list(samples[-window_samples:])
    half = window_samples // 2
    older, newer = window[:half], window[half:]
    older_mean = sum(older) / len(older)
    newer_mean = sum(newer) / len(newer)
    return (older_mean - newer_mean) / max(abs(older_mean), 1e-8)


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
        # None = a bare guide with no semantic path (PixelImage's palette
        # fit, direct-loss-only tests); every embedder read below is
        # is-not-None guarded already
        embedder: nn.Module | None,
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
        output_size=None,
        **optimizer_params,
    ):
        self.image_rep = image_rep
        self.embedder = embedder
        self.params = params

        # coarse_to_fine stage 1: saved PNG frames are upscaled to the final
        # canvas (width, height) so the numbered frame sequence stays uniform
        # in size across the stage boundary. None = save at native dims.
        # .bak backups always stay at native dims (they are the state dict).
        self.output_size = tuple(output_size) if output_size is not None else None

        # coherence_weighting (structured_config): non-None = the sampler's
        # (side_x, side_y) canvas, which switches the per-cutout semantic
        # weight redistribution on in every torch-side semantic path
        # (Prompt.forward, the M1 mlx bridge). mlx_full reads the config
        # flag into its StepConfig instead and computes the same weights
        # in-graph. One value carries both the switch and the data it needs,
        # so they can never disagree.
        self.coherence_canvas = (
            tuple(image_rep.image_shape)
            if params is not None and bool(params.get("coherence_weighting", False))
            else None
        )

        # phase scheduling: fixed quality-phase schedules over normalized
        # scene time t_hat = step/steps_per_scene (the table lives in
        # pytti/phase_scheduling.py; applied per step in train())
        self.phase_scheduling = params is not None and bool(
            params.get("phase_scheduling", False)
        )
        if self.phase_scheduling and params.animation_mode != "off":
            raise ValueError(
                "phase_scheduling schedules quality phases over normalized "
                "scene time (t_hat = step / steps_per_scene); "
                f"animation_mode={params.animation_mode!r} re-anchors the "
                "image every frame, so scene-time schedules have no defined "
                "meaning there (animation semantics deferred). Set "
                "animation_mode: off or phase_scheduling: false."
            )
        if (
            params is not None
            and bool(params.get("auto_stop", False))
            and params.animation_mode != "off"
        ):
            raise ValueError(
                "auto_stop judges loss plateaus over a scene; animation "
                f"warps (animation_mode={params.animation_mode!r}) perturb "
                "the loss every frame, so a plateau verdict has no defined "
                "meaning there. Set animation_mode: off or auto_stop: false."
            )

        # structure annealing (pytti/structure_annealing.py): scene-local
        # step -> blend strength for every scheduled re-liquify cycle, or
        # None when off. The schedule is a pure function of the config, so
        # restores replay the remaining cycles at the same steps. workhorse
        # validates the full config surface before any model loads; the
        # checks here repeat the ones a directly-constructed guide could
        # violate (fail loud, never a silent no-op knob).
        self.anneal_schedule: dict[int, float] | None = None
        if params is not None and bool(params.get("structure_annealing", False)):
            if params.animation_mode != "off":
                raise ValueError(
                    "structure_annealing schedules re-liquify cycles over a "
                    "scene's steps; animation_mode="
                    f"{params.animation_mode!r} re-anchors the image every "
                    "frame, so the schedule has no defined meaning there. "
                    "Set animation_mode: off or structure_annealing: false."
                )
            if bool(params.get("auto_stop", False)):
                raise ValueError(
                    "auto_stop judges TOTAL-loss plateaus; "
                    "structure_annealing deliberately resets the loss "
                    "mid-run, so a plateau verdict has no defined meaning. "
                    "Set auto_stop: false or structure_annealing: false."
                )
            if params.get("optimizer", "adam") != "adam":
                raise ValueError(
                    "structure_annealing supports optimizer=adam only "
                    "(schedule-free parameters entangle a Polyak average a "
                    "host-side overwrite would desynchronize). Set "
                    "optimizer: adam or structure_annealing: false."
                )
            if not isinstance(image_rep, (PixelImage, RGBImage)):
                raise ValueError(
                    "structure_annealing has no re-liquify path for "
                    f"{type(image_rep).__name__}: only PixelImage and "
                    "RGBImage hold pixel-domain state (latent re-encode is "
                    "out of scope in v1). Set structure_annealing: false."
                )
            self.anneal_schedule = anneal_schedule(
                int(params.steps_per_scene),
                int(params.get("anneal_cycles", 3)),
                float(params.get("anneal_strength", 0.5)),
            )

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

        # phase-scheduling palette lock state. The torch and mlx-bridge
        # paths lock via PixelImage.lock_palette (the exact mechanism the
        # lock_palette config uses: decode reads the sorted snapshot, the
        # live palette leaves the graph); mlx_full gates the palette
        # gradient + update in-step via a per-step 0/1 host argument
        # instead, so the lock is never a compiled-graph structure change.
        # A palette already locked by config (lock_palette/target_palette
        # set use_palette_target before the guide is built) is left alone.
        self._phase_palette_locked = False
        self._phase_manages_palette = (
            self.phase_scheduling
            and self.mlx_engine is None
            and isinstance(image_rep, PixelImage)
            and not image_rep.use_palette_target
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
        returns: the steps this scene counts for against the caller's global
                 step counter: n_steps when the scene ran to its cap OR
                 auto_stop converged it early (the counter stays
                 scene-aligned either way, so later scenes' frame numbering
                 and save slots match a full-length run), i + 1 when the
                 legacy `stop` loss threshold broke the loop.

        auto_stop (self.params): plateau detection on the TOTAL loss. The
        sample window is local to this call, and this method runs once per
        scene (workhorse.py's scene loop), so multi-scene runs reset the
        window at every scene boundary by construction. Sampling starts
        after the interpolation ramp — a scene never stops mid-crossfade.
        """
        params = self.params
        if (
            self.anneal_schedule is not None
            and interp_steps > 0
            and interp_prompts is not prompts
            and min(self.anneal_schedule) < interp_steps
        ):
            # backstop for directly-constructed guides — workhorse rejects
            # this at config time (validate_structure_annealing). Scene 1
            # and coarse_to_fine pass the SAME list for both prompt args
            # (a self-crossfade), which is why identity is the test here.
            raise ValueError(
                "structure_annealing: the first anneal cycle (scene step "
                f"{min(self.anneal_schedule)}) falls inside this scene's "
                f"{interp_steps}-step interpolation crossfade, where the "
                "outgoing scene's prompts still dominate the loss — the "
                "re-liquified band would recompose toward the wrong scene. "
                "Lower interpolation_steps or anneal_cycles, or raise "
                "steps_per_scene."
            )
        auto_stop = params is not None and bool(params.get("auto_stop", False))
        if auto_stop:
            window_steps = int(params.get("auto_stop_window", 50))
            threshold = float(params.get("auto_stop_threshold", 0.002))
            if window_steps < 2 * AUTO_STOP_CHECK_INTERVAL:
                raise ValueError(
                    f"auto_stop_window={window_steps} is too short: the loss "
                    f"is sampled every {AUTO_STOP_CHECK_INTERVAL} steps and "
                    "the plateau detector needs at least two samples — use "
                    f"auto_stop_window >= {2 * AUTO_STOP_CHECK_INTERVAL}"
                )
            if not math.isfinite(threshold):
                raise ValueError(
                    f"auto_stop_threshold={threshold!r} must be finite"
                )
            window_samples = window_steps // AUTO_STOP_CHECK_INTERVAL
            total_samples: list[float] = []

        steps_run = 0
        for i in tqdm(range(n_steps)):
            self.update(i + i_offset, i + skipped_steps)
            if self.anneal_schedule is not None:
                # scene-local step, matching the schedule's domain (update()
                # above already saved this step's frame, so the cycle shows
                # from the NEXT saved frame on)
                scene_step = i + skipped_steps
                if scene_step >= min(self.anneal_schedule):
                    # the release is a function of scene POSITION, not of a
                    # cycle event, so a restore that resumes past the first
                    # cycle stays hold-free like the uninterrupted run
                    self._release_init_holds(i + i_offset)
                strength = self.anneal_schedule.get(scene_step)
                if strength is not None:
                    self._apply_structure_anneal(i + i_offset, strength)
            losses = self.train(
                i + skipped_steps,
                prompts,
                interp_prompts,
                loss_augs,
                interp_steps=interp_steps,
                gradient_accumulation_steps=gradient_accumulation_steps,
            )
            steps_run = i + 1
            # float(TOTAL) is a device sync; it is only paid when the legacy
            # early-stop is set (per step) or at an auto_stop sample point
            # (once per AUTO_STOP_CHECK_INTERVAL steps, past the interp ramp)
            sample_now = (
                auto_stop
                and i + skipped_steps >= interp_steps
                and (i + 1) % AUTO_STOP_CHECK_INTERVAL == 0
            )
            if stop == -math.inf and not sample_now:
                continue
            total = float(losses["TOTAL"])
            if stop != -math.inf and total <= stop:
                break
            if sample_now:
                total_samples.append(total)
                improvement = plateau_improvement(total_samples, window_samples)
                if improvement is not None and improvement < threshold:
                    logger.info(
                        f"auto_stop: converged at step {i + i_offset + 1} "
                        f"(of cap {i_offset + n_steps}) — relative TOTAL-loss "
                        f"improvement {improvement:.6f} < {threshold} over "
                        f"the trailing {window_steps} steps; stopping scene."
                    )
                    self._save_final_frame(i + i_offset, i_offset + n_steps)
                    return n_steps
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

    def _apply_phase_schedule(self, i, loss_augs) -> float:
        """
        Apply the phase_scheduling table (pytti/phase_scheduling.py) for
        scene-local step ``i``:

        - sets ``weight_scale`` on the smoothing (TV) loss and on the
          direct init-hold losses (``self.init_augs``) — both backends read
          it when evaluating the configured weight;
        - manages the Limited Palette lock for the final third of the
          scene: the torch/mlx-bridge paths flip ``PixelImage.lock_palette``
          at the transition; mlx_full instead consumes the returned gate.

        Returns the palette gate: 1.0 (open) or 0.0 (locked), passed to the
        mlx_full engine as a per-step host argument. run_steps passes
        ``i + skipped_steps`` and workhorse runs one scene per run_steps
        call, so ``i`` traverses [0, steps_per_scene) within every scene
        and the schedules reset at each scene boundary by construction.
        """
        t_hat = scene_t_hat(i, self.params.steps_per_scene)
        tv_scale = tv_weight_scale(t_hat)
        for aug in loss_augs:
            if isinstance(aug, TVLoss):
                aug.weight_scale = tv_scale
        init_scale = init_weight_scale(t_hat)
        for aug in self.init_augs or ():
            aug.weight_scale = init_scale
        locked = palette_locked(t_hat)
        if locked != self._phase_palette_locked:
            if self._phase_manages_palette:
                self.image_rep.lock_palette(locked)
            if isinstance(self.image_rep, PixelImage):
                logger.info(
                    "phase_scheduling: palette "
                    f"{'locked' if locked else 'unlocked'} at scene step {i} "
                    f"(t_hat {t_hat:.3f})"
                )
            self._phase_palette_locked = locked
        return 0.0 if locked else 1.0

    def _release_init_holds(self, global_step: int) -> None:
        """
        Disable every direct init-hold loss, idempotently (structure
        annealing only — pytti/structure_annealing.py). A direct init hold
        is a full-band pull toward a PRE-anneal image: left enabled it
        drags the re-liquified band straight back within a few steps and
        the cycles are inert (coarse_to_fine holds the previous stage's
        composition at weight 2 in exactly the stage the cycles run in).
        run_steps calls this from the first cycle's scene step ONWARD —
        position-based, so a restore resuming past the first cycle is
        hold-free exactly like the uninterrupted run. The hold has done
        its settling work by then; from here on CLIP owns composition.
        Both backends read ``enabled`` per step (mlx_full re-traces once).
        """
        released = [aug for aug in (self.init_augs or ()) if aug.enabled]
        if released:
            for aug in released:
                aug.set_enabled(False)
            logger.info(
                "structure_annealing: released the direct init hold at "
                f"step {global_step} "
                f"({', '.join(str(aug) for aug in released)}) — a hold "
                "toward the pre-anneal image would cancel the cycles"
            )

    def _apply_structure_anneal(self, global_step: int, strength: float) -> None:
        """
        One structure-annealing cycle (pytti/structure_annealing.py): blend
        the image's low-frequency band toward the configured source at
        ``strength``, releasing any direct init hold at the first cycle
        (a full-band pull toward a pre-anneal image would cancel the
        re-liquification — see the anneal module's docstring). Torch path:
        the live parameters are edited in place (identity preserved, so
        Adam's moments stay attached — KEPT through the cycle by design).
        mlx_full: the same torch-side operation runs as a host intervention
        between compiled steps — export the params tree into the torch
        module, anneal it, import it back; the engine's Adam state and RNG
        stream are untouched.
        """
        params = self.params
        if params is None:
            raise RuntimeError(
                "_apply_structure_anneal reached with params=None — the "
                "anneal schedule is only ever built from a params config"
            )
        if self.mlx_engine is not None:
            self.mlx_engine.write_back(self.image_rep)
        anneal_image_rep(
            self.image_rep,
            strength=strength,
            band=float(params.get("anneal_band", 0.15)),
            source=str(params.get("anneal_source", "noise")),
            init_spectrum_falloff=float(params.get("init_spectrum_falloff", 1.0)),
            init_spectrum_chroma=str(params.get("init_spectrum_chroma", "full")),
        )
        if self.mlx_engine is not None:
            self.mlx_engine.import_params(self.image_rep)
        logger.info(
            f"structure_annealing: re-liquified the low band at step "
            f"{global_step} (strength {strength:.3f}, "
            f"band {float(params.get('anneal_band', 0.15)):g}, "
            f"source {params.get('anneal_source', 'noise')})"
        )

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
        # phase scheduling runs first so BOTH backends see the same per-step
        # weight scales / palette gate (i is the scene-local step here)
        palette_gate = 1.0
        if self.phase_scheduling:
            palette_gate = self._apply_phase_schedule(i, loss_augs)

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
                palette_gate=palette_gate,
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
                    coherence_canvas=self.coherence_canvas,
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
                            coherence_canvas=self.coherence_canvas,
                        )
                        interp_total = interp_total + loss * (1 - t)

                prompt_total = 0
                for prompt in prompts:
                    loss, loss_raw = prompt(
                        format_input(image_embeds, self.embedder, prompt),
                        format_input(offsets, self.embedder, prompt),
                        format_input(sizes, self.embedder, prompt),
                        coherence_canvas=self.coherence_canvas,
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

    def decode_output_image(self):
        """
        Decode the current image as an OUTPUT image (PIL, native dims) with
        the same state discipline as _save_frame: the Polyak-averaged (eval)
        iterate under adamw_sf, and the live MLX params under mlx_full.
        coarse_to_fine's stage transition decodes through here.
        """
        with self.optimizer_eval():
            if self.mlx_engine is not None:
                self.mlx_engine.write_back(self.image_rep)
            return self.image_rep.decode_image()

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
            if self.output_size is not None and im.size != self.output_size:
                # coarse_to_fine stage 1: frames land on disk at the final
                # canvas size (the .bak below keeps native dims)
                im = im.resize(self.output_size, Image.BICUBIC)
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

    def _save_final_frame(self, i, scene_end):
        """
        Persist the converged state when auto_stop ends a scene early.

        update() saves BEFORE train(), so the newest frame on disk always
        predates the stop step's state. Save into the next save_every slot of
        this scene — or re-save the scene's last slot when the stop landed
        after it — so frame numbering matches a full-length run and a later
        scene's saves can never land on top of the converged frame.
        _save_frame handles optimizer_eval (adamw_sf) and the mlx_full
        write-back, so the frame and its .bak both hold the final state.

        i: the global step index the scene stopped at (0-based).
        scene_end: the global step index just past the scene's cap.
        """
        params = self.params
        if params.save_every <= 0:
            return
        # slots already written by update(): every n with n*save_every-1 <= i
        n_fired = (i + 1) // params.save_every
        # last slot a full-length scene would write
        n_scene_last = scene_end // params.save_every
        n_final = min(n_fired + 1, n_scene_last)
        if n_final < 1:
            # the scene is shorter than one save interval: a full-length run
            # would not have saved a frame either
            return
        self._save_frame(n_final * params.save_every - 1)

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
