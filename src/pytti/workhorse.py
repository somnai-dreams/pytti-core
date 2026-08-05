"""
CLI rendering entry point. Hydra composes the config, `do_run` executes it.
"""

import copy
import gc
import os
import random
import re
import socket
import urllib.parse
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import hydra
import torch
from loguru import logger
from omegaconf import DictConfig, OmegaConf, open_dict
from PIL import Image

from pytti import (
    empty_cache,
    fetch,
    is_zero_weight,
    print_vram_usage,
    reset_vram_usage,
    set_default_device,
    vram_profiling,
    vram_usage_mode,
)
from pytti.coarse_to_fine import (
    is_thumbnail_stage,
    resume_stage,
    stage_dims,
    stage_steps,
    validate_coarse_stages,
    validate_coarse_to_fine,
)

# importing the module registers ConfigSchema in Hydra's ConfigStore, which
# default.yaml composes as its base
from pytti.config import structured_config  # noqa: F401
from pytti.files import get_last_file, get_next_file
from pytti.image_models import (
    DifferentiableImage,
    LlamaGenImage,
    PixelImage,
    RGBImage,
    VQGANImage,
)
from pytti.image_models.init_noise import require_white_init, resolve_init_spectrum
from pytti.ImageGuide import DirectImageGuide
from pytti.LossAug.LossOrchestratorClass import (
    configure_init_image,
    configure_optical_flows,
    configure_stabilization_augs,
)
from pytti.Perceptor import load_clip
from pytti.Perceptor.Embedder import HDMultiClipEmbedder
from pytti.Perceptor.Prompt import parse_prompt
from pytti.prompt_spec import parse_prompt_spec
from pytti.rotoscoper import ROTOSCOPERS, get_frames
from pytti.warmup import (
    ensure_configs_exist,
    migrate_local_config,
    register_resolvers,
)

HUB_PROBE_TIMEOUT_S = 2.0


def _hub_host_port() -> tuple[str, int]:
    """Host/port of the HF hub endpoint (honors an HF_ENDPOINT override)."""
    endpoint = os.environ.get("HF_ENDPOINT", "").strip() or "https://huggingface.co"
    parts = urllib.parse.urlsplit(endpoint)
    if parts.hostname is None:
        raise ValueError(f"HF_ENDPOINT {endpoint!r} has no hostname")
    return parts.hostname, parts.port or (80 if parts.scheme == "http" else 443)


def hub_reachable(timeout_s: float = HUB_PROBE_TIMEOUT_S) -> bool:
    """One short TCP connect to the hub endpoint; False on DNS failure,
    no route, refusal, or timeout."""
    host, port = _hub_host_port()
    try:
        socket.create_connection((host, port), timeout=timeout_s).close()
    except OSError:
        return False
    return True


def configure_offline_fallback(probe: Callable[[], bool] = hub_reachable) -> bool:
    """
    Offline robustness (live user bug): with the machine offline,
    huggingface_hub still resolves tokenizers/configs against
    huggingface.co even when every file is cached, so a render hangs for
    minutes at step 0 before failing. Probe hub reachability ONCE (~2s
    budget) and pre-set offline mode when the hub is unreachable, so hub
    lookups go straight to the local cache. A cache MISS under offline
    mode still fails loud with the hub's LocalEntryNotFoundError naming
    the missing repo.

    Must run before anything imports huggingface_hub — its constants
    module freezes HF_HUB_OFFLINE/TRANSFORMERS_OFFLINE at import time
    (verified on 1.26.0), and pytti defers that import until load_clip.

    A pre-set HF_HUB_OFFLINE or TRANSFORMERS_OFFLINE (any value, even
    "0") is the user's decision: skipped, never overridden.

    Returns True when the probe engaged offline mode.
    """
    if (
        os.environ.get("HF_HUB_OFFLINE") is not None
        or os.environ.get("TRANSFORMERS_OFFLINE") is not None
    ):
        return False
    if probe():
        return False
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    host, port = _hub_host_port()
    logger.info(f"hub {host}:{port} unreachable — offline: using cached models only")
    return True


def parse_scenes(
    embedder,
    scenes,
    scene_prefix,
    scene_suffix,
):
    """
    Parses scenes separated by || and applies provided prefixes and suffixes to each scene.

    :param embedder: The embedder object
    :param params: The experiment parameters
    :return: The embedder and the prompts.
    """
    logger.info("Loading prompts...")
    prompts = [
        [
            parse_prompt(embedder, p.strip())
            for p in (scene_prefix + stage + scene_suffix).strip().split("|")
            if p.strip()
        ]
        for stage in scenes.split("||")
        if stage
    ]
    logger.info("Prompts loaded.")
    return embedder, prompts


def load_init_image(
    init_image_path=None,
    height: int = -1,
    width: int = -1,
):
    """
    If the user has specified an image to use as the initial image, load it. Otherwise, if the
    user has specified a width or height, create a blank image of the specified size

    :param init_image_path: A local path or URL describing where to load the image from
    :param height: height of the image to be generated.
    :return: the initial image and the size of the initial image.
    """
    if init_image_path:
        init_image_pil = Image.open(fetch(init_image_path)).convert("RGB")
        init_size = init_image_pil.size
        # automatic aspect ratio matching
        if width == -1:
            width = int(height * init_size[0] / init_size[1])
        if height == -1:
            height = int(width * init_size[1] / init_size[0])
    else:
        init_image_pil = None
    return init_image_pil, height, width


def load_video_source(
    video_path: str,
    pre_animation_steps: int,
    steps_per_frame: int,
    height: int,
    width: int,
    init_image_pil: Image.Image,
    params=None,
):
    """
    Loads a video file and returns a PIL image of the first frame

    :param video_path: The path to the video file
    :param pre_animation_steps: The number of frames to skip at the beginning of the video
    :param steps_per_frame: How many steps to take per frame
    :param height: the height of the output image
    :param width: the width of the output image
    :return: The video frames, the initial image, the height and width of the image.
    """
    logger.info(f"loading {video_path}...")
    video_frames = get_frames(video_path, params)
    pre_animation_steps = max(steps_per_frame, pre_animation_steps)
    if init_image_pil is None:
        init_image_pil = Image.fromarray(video_frames.get_data(0)).convert("RGB")
        init_size = init_image_pil.size
        if width == -1:
            width = int(height * init_size[0] / init_size[1])
        if height == -1:
            height = int(width * init_size[1] / init_size[0])
    return video_frames, init_image_pil, height, width


@dataclass
class PassSetup:
    """Everything Phase 3 builds for ONE optimization pass at one canvas
    size: the image rep plus every loss aug wired to its dims. A plain run
    builds exactly one; coarse_to_fine builds one per stage."""

    img: DifferentiableImage
    loss_augs: list
    init_augs: list
    semantic_init_prompt: object
    stabilization_augs: list
    last_frame_semantic: object
    optical_flows: list
    init_image_pil: Image.Image | None


def configure_pass(
    params,
    device,
    embedder,
    prompts,
    init_image_pil,
    video_frames,
    restore: bool,
    palette_source: PixelImage | None = None,
) -> PassSetup:
    """
    Phase 3 of a render: build the image representation and every loss aug
    at params' dims. Extracted from do_run so coarse_to_fine can run it once
    per stage (fresh rep + augs at each stage's dims); a plain run calls it
    exactly once with the unmodified config.

    palette_source: coarse-to-fine's Limited Palette carry — the stage-1
    PixelImage whose learned palette seeds the fresh image and (via
    lock_palette) pins the encode fit to exactly those colors instead of
    re-fitting a palette from scratch. A user-locked palette stays locked to
    the carried colors; an unlocked one resumes learning after the fit.
    """
    # The init_spectrum knob only governs a visible random start: with an
    # init_image (encoded over the random init in configure_init_image
    # below) or a restore (state reloaded from the .bak) it resolves to the
    # plain white draw — no wasted shaping, and no spurious VQGAN/LlamaGen
    # rejection of a start that is never shown.
    init_spectrum, init_spectrum_falloff = resolve_init_spectrum(
        params.init_spectrum,
        params.init_spectrum_falloff,
        has_init_image=init_image_pil is not None,
        restore=restore,
    )

    # set up image
    if params.image_model == "Limited Palette":
        img = PixelImage(
            width=params.width,
            height=params.height,
            scale=params.pixel_size,
            palette_size=params.palette_size,
            n_palettes=params.palettes,
            gamma=params.gamma,
            hdr_weight=params.hdr_weight,
            norm_weight=params.palette_normalization_weight,
            device=device,
        )
        img.encode_random(
            random_palette=params.random_initial_palette,
            init_spectrum=init_spectrum,
            init_spectrum_falloff=init_spectrum_falloff,
        )
        if params.target_palette.strip() != "":
            img.set_palette_target(
                Image.open(fetch(params.target_palette)).convert("RGB")
            )
        else:
            img.lock_palette(params.lock_palette)
    elif params.image_model == "Unlimited Palette":
        img = RGBImage(params.width, params.height, params.pixel_size, device=device)
        img.encode_random(
            init_spectrum=init_spectrum,
            init_spectrum_falloff=init_spectrum_falloff,
        )
    elif params.image_model == "VQGAN":
        # categorical token init has no spectrum to shape: reject a shaped
        # (and visible — see resolve above) init BEFORE the multi-GB model
        # download/load; encode_random repeats the same guard for direct use
        require_white_init("VQGANImage", init_spectrum)
        # namespaced cache (~/.cache/pytti/vqgan); fall back to the legacy
        # un-namespaced location if it already holds downloads
        model_artifacts_path = Path(params.models_parent_dir) / "pytti" / "vqgan"
        legacy_path = Path(params.models_parent_dir) / "vqgan"
        if not model_artifacts_path.exists() and legacy_path.exists():
            model_artifacts_path = legacy_path
        VQGANImage.init_vqgan(params.vqgan_model, model_artifacts_path, device=device)
        img = VQGANImage(params.width, params.height, params.pixel_size, device=device)
        img.encode_random(
            init_spectrum=init_spectrum,
            init_spectrum_falloff=init_spectrum_falloff,
        )
    elif params.image_model == "LlamaGen":
        if params.perceptor_backend != "torch":
            raise ValueError(
                f"image_model=LlamaGen is torch-only; perceptor_backend="
                f"{params.perceptor_backend!r} does not support it. "
                "Use perceptor_backend=torch."
            )
        # categorical token init has no spectrum to shape: reject a shaped
        # (and visible — see resolve above) init BEFORE the model
        # download/load; encode_random repeats the same guard for direct use
        require_white_init("LlamaGenImage", init_spectrum)
        # weights land in the HF hub cache, revision-pinned + sha-verified
        LlamaGenImage.init_llamagen(params.llamagen_model, device=device)
        img = LlamaGenImage(
            params.width, params.height, params.pixel_size, device=device
        )
        img.encode_random(
            init_spectrum=init_spectrum,
            init_spectrum_falloff=init_spectrum_falloff,
        )
    else:
        raise ValueError(
            f"Unrecognized image_model: {params.image_model!r}. "
            'Supported: "Limited Palette", "Unlimited Palette", "VQGAN", '
            '"LlamaGen".'
        )

    release_palette_after_fit = False
    if palette_source is not None:
        if not isinstance(img, PixelImage):
            raise ValueError(
                "palette_source only applies to Limited Palette, got "
                f"{type(img).__name__}"
            )
        img.copy_palette_from(palette_source)
        if params.target_palette.strip() == "":
            # hold exactly the carried colors through the encode fit below;
            # a config-locked palette stays locked to them, an unlocked one
            # is released after the fit and keeps learning in this pass.
            # unlock first: lock_palette(True) on an already-locked image
            # would re-snapshot the STALE target, not the carried palette
            img.lock_palette(False)
            img.lock_palette(True)
            release_palette_after_fit = not params.lock_palette

    #######################################

    loss_augs = []

    # set up init image
    (
        init_augs,
        semantic_init_prompt,
        loss_augs,
        img,
        embedder,
        prompts,
    ) = configure_init_image(
        init_image_pil,
        restore,
        img,
        params,
        loss_augs,
        embedder,
        prompts,
    )

    if release_palette_after_fit:
        img.lock_palette(False)

    # other image prompts
    for entry in params.direct_image_prompts.split("|"):
        if not entry.strip():
            continue
        spec = parse_prompt_spec(entry.strip())
        loss_augs.append(
            type(img)
            .get_preferred_loss()
            .build(
                spec.text,
                img.image_shape,
                weight=spec.weight,
                stop=spec.stop,
                mask=spec.mask,
                path=spec.image_path() or spec.text,
            )
        )

    # stabilization
    (
        loss_augs,
        img,
        init_image_pil,
        stabilization_augs,
    ) = configure_stabilization_augs(img, init_image_pil, params, loss_augs)

    if not is_zero_weight(params.semantic_stabilization_weight):
        last_frame_semantic = parse_prompt(
            embedder,
            f"stabilization:{params.semantic_stabilization_weight}",
            init_image_pil if init_image_pil else img.decode_image(),
        )
        last_frame_semantic.set_enabled(init_image_pil is not None)
        for scene in prompts:
            scene.append(last_frame_semantic)
    else:
        last_frame_semantic = None

    # optical flow
    img, loss_augs, optical_flows = configure_optical_flows(img, params, loss_augs)

    return PassSetup(
        img=img,
        loss_augs=loss_augs,
        init_augs=init_augs,
        semantic_init_prompt=semantic_init_prompt,
        stabilization_augs=stabilization_augs,
        last_frame_semantic=last_frame_semantic,
        optical_flows=optical_flows,
        init_image_pil=init_image_pil,
    )


def make_guide(
    params, setup: PassSetup, embedder, base_name, video_frames=None, output_size=None
) -> DirectImageGuide:
    """Phase 5: the main model object for one pass."""
    return DirectImageGuide(
        image_rep=setup.img,
        embedder=embedder,
        lr=params.learning_rate,
        params=params,
        base_name=base_name,
        optical_flows=setup.optical_flows,
        video_frames=video_frames,
        stabilization_augs=setup.stabilization_augs,
        last_frame_semantic=setup.last_frame_semantic,
        init_augs=setup.init_augs,
        semantic_init_prompt=setup.semantic_init_prompt,
        init_image_pil=setup.init_image_pil,
        output_size=output_size,
    )


def full_canvas(params, img: DifferentiableImage) -> tuple[int, int]:
    """
    The FULL-dims canvas (pixels) for a run whose current pass rendered
    `img`. Pixel/RGB canvases are exactly dims x pixel_size; VQGAN and
    LlamaGen floor to their model stride, recovered from the live rep
    (image_shape = toks x f).
    """
    width = params.width * params.pixel_size
    height = params.height * params.pixel_size
    if isinstance(img, (VQGANImage, LlamaGenImage)):
        f = img.image_shape[0] // img.toksX
        width, height = (width // f) * f, (height // f) * f
    return width, height


def run_coarse_to_fine(
    *, params, device, embedder, prompts, init_image_pil, base_name, restore
):
    """
    Staged still render (coarse_to_fine: true, coarse_stages stages) —
    stage semantics in pytti/coarse_to_fine.py's module docstring. Frame
    numbering is monotonic across stages: every guide shares
    base_name/save_every and each stage's run_steps starts at the global
    offset where the previous stage ended, so update()'s save slots and the
    .bak numbering continue the sequence (STUDIO's frame scan and the
    restore path never see a reset).

    Restore maps the restored global step back onto the stage ladder
    (coarse_to_fine.resume_stage): a step inside stage k rebuilds stage k's
    rep (the .bak at that slot holds stage-k dims) and replays the
    remainder of the ladder; resuming into stage k > 1 reloads the
    persisted stage-(k-1) transition image for the weight-2 hold.

    Thumbnail-stage sampling: any stage whose canvas short side is at or
    below the largest perceptor input runs with cutout_sampler=full
    (coarse_to_fine.is_thumbnail_stage) — random crops there upsample
    (nearly) the whole frame anyway, so the designed full-frame sampler is
    forced for exactly those stages, on torch and mlx alike (the embedder's
    live sampler attribute is what both the torch cutout path and the
    mlx_full engine construction read).

    Costs, documented: every stage's guide is a fresh construction, so
    mlx_full recompiles its whole-step graph once (and reloads its towers)
    at each transition; the mlx RNG also reseeds from the same config seed,
    so each stage replays the same cutout draw sequence — same-seed-N-runs
    semantics, not a correctness issue.
    """
    splits = stage_steps(params.steps_per_scene, params.coarse_stages)
    dims = stage_dims(params.width, params.height, params.coarse_stages)
    n_stages = len(splits)
    scene = prompts[0]
    backup_dir = Path("backup") / params.file_namespace

    # classic+mlx_full is rejected by the engine at guide construction on
    # plain runs; a thumbnail stage 1 forcing sampler=full would mask the
    # configured value and defer that rejection until AFTER stage 1
    # rendered — reject the configured pair up front instead (mirrors
    # mlx_engine.engine.MLXStillEngine._validate_config).
    if params.perceptor_backend == "mlx_full" and params.cutout_sampler == "classic":
        raise ValueError(
            "perceptor_backend=mlx_full does not support "
            "cutout_sampler=classic (torch-only): use "
            "cutout_sampler=batched|smart|full, or perceptor_backend=torch."
        )

    def transition_png(stage_number: int) -> Path:
        # the persisted seam written when stage `stage_number` completes
        # (always at the full canvas) — also the resume seam for restores
        # that land in stage_number + 1
        return backup_dir / f"{base_name}_coarse_{stage_number}.png"

    if restore:
        filename, restore_frame = get_last_file(
            str(backup_dir),
            f"^(?P<pre>{re.escape(base_name)}_)(?P<index>\\d*)(?P<post>\\.bak)$",
        )
        if filename is None:
            raise FileNotFoundError(
                f"restore=true but no .bak backups found under {backup_dir}"
            )
        logger.info(f"restoring from {filename}")
        bak_path = backup_dir / filename
        i_restore = restore_frame * params.save_every
    else:
        bak_path = None
        i_restore = 0

    stage_start, stage_done = resume_stage(i_restore, splits)

    configured_sampler = embedder.cutout_sampler
    max_cut_size = max(embedder.cut_sizes)
    prev_img = None  # previous stage's live rep (Limited Palette carry)
    stage_init_pil = init_image_pil  # stage 1 inits from the user's image
    canvas = None

    try:
        for stage_number in range(stage_start, n_stages + 1):
            idx = stage_number - 1
            w, h = dims[idx]
            offset = sum(splits[:idx])
            resumed_here = restore and stage_number == stage_start
            done = stage_done if stage_number == stage_start else 0

            p = copy.deepcopy(params)
            p.width, p.height = w, h
            p.steps_per_scene = splits[idx]
            if stage_number > 1:
                seam = transition_png(stage_number - 1)
                if resumed_here:
                    if not seam.is_file():
                        raise FileNotFoundError(
                            f"resuming coarse_to_fine inside stage "
                            f"{stage_number} needs the stage-"
                            f"{stage_number - 1} transition image at {seam}, "
                            "which is missing — restart the render without "
                            "restore=true"
                        )
                    stage_init_pil = Image.open(seam).convert("RGB")
                p.init_image = str(seam)  # names the hold in logs/records
                p.direct_init_weight = "2"  # structural hold on the previous stage
                p.semantic_init_weight = ""  # validated off for coarse_to_fine

            logger.info(
                f"coarse_to_fine stage {stage_number}/{n_stages}: {w}x{h} "
                f"for steps {offset}..{offset + splits[idx]} of "
                f"{params.steps_per_scene}"
                + (
                    ", previous stage held at direct init weight 2"
                    if stage_number > 1
                    else ""
                )
            )
            setup = configure_pass(
                p,
                device,
                embedder,
                prompts,
                stage_init_pil,
                None,
                resumed_here,
                palette_source=prev_img if isinstance(prev_img, PixelImage) else None,
            )
            if resumed_here and bak_path is not None:
                setup.img.load_state_dict(torch.load(bak_path))
            if canvas is None:
                canvas = full_canvas(params, setup.img)

            # thumbnail-stage sampling: set per stage on the live embedder
            # attribute — the torch cutout path reads it at every
            # make_cutouts call and the mlx_full engine snapshots it at
            # construction (in make_guide below), so one assignment covers
            # both backends.
            side_px = tuple(setup.img.image_shape)
            if is_thumbnail_stage(side_px, max_cut_size):
                embedder.cutout_sampler = "full"
                if configured_sampler != "full":
                    logger.info(
                        f"coarse_to_fine stage {stage_number}/{n_stages}: "
                        f"canvas {side_px[0]}x{side_px[1]}px is at or below "
                        f"the largest perceptor input ({max_cut_size}px), "
                        "so every random crop would upsample the whole "
                        "frame anyway — forcing cutout_sampler=full for "
                        f"this stage (configured: {configured_sampler})"
                    )
            else:
                embedder.cutout_sampler = configured_sampler

            is_final = stage_number == n_stages
            model = make_guide(
                p,
                setup,
                embedder,
                base_name,
                output_size=None if is_final else canvas,
            )
            model.run_steps(
                splits[idx] - done,
                scene,
                scene,
                setup.loss_augs,
                # later stages start mid-scene, not at a scene boundary:
                # no interp ramp
                interp_steps=p.interpolation_steps if stage_number == 1 else 0,
                i_offset=offset + done,
                skipped_steps=done,
                gradient_accumulation_steps=p.gradient_accumulation_steps,
            )
            if is_final:
                break

            # transition: decode -> bicubic upscale to the full canvas ->
            # persist; the next pass re-encodes it at its own dims
            seam = transition_png(stage_number)
            stage_init_pil = model.decode_output_image().resize(canvas, Image.BICUBIC)
            stage_init_pil.save(seam)
            logger.info(
                f"coarse_to_fine transition {stage_number}->"
                f"{stage_number + 1}: upscaled the stage-{stage_number} "
                f"result to {canvas[0]}x{canvas[1]} ({seam})"
            )
            prev_img = setup.img
            del model, setup
            gc.collect()
            empty_cache()
    finally:
        embedder.cutout_sampler = configured_sampler


@hydra.main(config_path="config", config_name="default", version_base=None)
def _hydra_main(cfg: DictConfig):
    params = cfg

    device = set_default_device(params.get("device"))
    if params.get("device") is None:
        with open_dict(params) as p:
            p.device = str(device)
    logger.debug(f"Using device {device}")

    # YAML 1.1 parses a literal `off` as boolean False; hydra then stringifies
    # it into the str-typed schema field as 'False'. Normalize BOTH forms —
    # the string form previously slipped through and rode the torch path as a
    # silent no-op animation mode (exposed by mlx_full's eligibility check).
    if params.animation_mode in (False, "False", "false"):
        params.animation_mode = "off"

    # save_every: 0 means "one frame per animation frame"
    if params.save_every <= 0:
        params.save_every = params.steps_per_frame
        logger.info(f"save_every auto-set to steps_per_frame ({params.save_every})")

    logger.debug(OmegaConf.to_container(cfg, resolve=True))
    latest = -1

    restore = params.restore
    # reencode: restore from the saved PNG instead of the model state backup
    # (useful if image settings changed); not yet exposed in config
    reencode = False
    restore_run = latest

    # NB: `backup/` dir probably not working at present
    if restore and restore_run == latest:
        _, restore_run = get_last_file(
            f"backup/{params.file_namespace}",
            f"^(?P<pre>{re.escape(params.file_namespace)}\\(?)(?P<index>\\d*)(?P<post>\\)?_\\d+\\.bak)$",
        )

    def do_run():
        # NB: must be computed after hydra has chdir'd into the run directory
        outpath = f"{os.getcwd()}/images_out"

        # Phase 1 - reset state
        ########################
        ROTOSCOPERS.clear_rotoscopers()
        vram_profiling(params.approximate_vram_usage)
        reset_vram_usage()
        # @markdown which frame to restore from
        restore_frame = latest

        # coarse_to_fine rejects every config it has no stage semantics for
        # BEFORE any model loads (coarse_stages is checked unconditionally:
        # a non-default value on a run that ignores it is a config lie)
        validate_coarse_stages(
            coarse_to_fine=params.coarse_to_fine,
            coarse_stages=params.coarse_stages,
        )
        if params.coarse_to_fine:
            validate_coarse_to_fine(
                animation_mode=params.animation_mode,
                n_scenes=len([s for s in params.scenes.split("||") if s.strip()]),
                breath_mode=params.breath_mode,
                semantic_init=params.semantic_init_weight not in ["", "0"],
                semantic_stabilization=not is_zero_weight(
                    params.semantic_stabilization_weight
                ),
            )
            # fail loud on a bad step split
            stage_steps(params.steps_per_scene, params.coarse_stages)

        # set up seed for deterministic RNG
        if params.seed is None:
            with open_dict(params) as p:
                p.seed = random.randint(0, 2**32 - 1)
            logger.info(f"Using random seed {params.seed}")
        torch.manual_seed(params.seed)

        # Phase 2 - load and parse
        ###########################

        # load CLIP — probing hub reachability first, so a dead network
        # reads the cache immediately instead of hanging on per-file HEADs
        configure_offline_fallback()
        load_clip(params, device=device)

        cutn = params.cutouts
        if params.gradient_accumulation_steps > 1:
            if cutn % params.gradient_accumulation_steps != 0:
                raise ValueError(
                    "To use gradient_accumulation_steps > 1, the cutouts parameter "
                    "must be an exact multiple of gradient_accumulation_steps."
                )
            cutn //= params.gradient_accumulation_steps
        logger.debug(cutn)

        embedder = HDMultiClipEmbedder(
            cutn=cutn,
            cut_pow=params.cut_pow,
            padding=params.cutout_border,
            border_mode=params.border_mode,
            cutout_sampler=params.cutout_sampler,
            device=device,
        )

        # load scenes
        with vram_usage_mode("Text Prompts"):
            embedder, prompts = parse_scenes(
                embedder,
                scenes=params.scenes,
                scene_prefix=params.scene_prefix,
                scene_suffix=params.scene_suffix,
            )

        # load init image
        init_image_pil, height, width = load_init_image(
            init_image_path=params.init_image,
            height=params.height,
            width=params.width,
        )

        # video source
        video_frames = None
        if params.animation_mode == "Video Source":
            video_frames, init_image_pil, height, width = load_video_source(
                video_path=params.video_path,
                pre_animation_steps=params.pre_animation_steps,
                steps_per_frame=params.steps_per_frame,
                height=params.height,
                width=params.width,
                init_image_pil=init_image_pil,
                params=params,
            )

        params.height, params.width = height, width

        # Phase 3/4 - filespace + base_name (shared by every pass, so
        # coarse_to_fine's two stages number one continuous sequence)
        #############################################################

        Path(f"{outpath}/{params.file_namespace}").mkdir(parents=True, exist_ok=True)
        Path(f"backup/{params.file_namespace}").mkdir(parents=True, exist_ok=True)
        if restore:
            base_name = (
                params.file_namespace
                if restore_run == 0
                else f"{params.file_namespace}({restore_run})"
            )
        elif not params.allow_overwrite:
            # finds the next available base_name to save files with
            _, i = get_next_file(
                f"{outpath}/{params.file_namespace}",
                f"^(?P<pre>{re.escape(params.file_namespace)}\\(?)(?P<index>\\d*)(?P<post>\\)?_1\\.png)$",
                [f"{params.file_namespace}_1.png", f"{params.file_namespace}(1)_1.png"],
            )
            base_name = (
                params.file_namespace if i == 0 else f"{params.file_namespace}({i})"
            )
        else:
            base_name = params.file_namespace

        if params.coarse_to_fine:
            run_coarse_to_fine(
                params=params,
                device=device,
                embedder=embedder,
                prompts=prompts,
                init_image_pil=init_image_pil,
                base_name=base_name,
                restore=restore,
            )
            return

        # Phase 3 - Setup Optimization
        ###############################

        setup = configure_pass(
            params, device, embedder, prompts, init_image_pil, video_frames, restore
        )
        img = setup.img

        # restore
        if restore:
            if not reencode:
                if restore_frame == latest:
                    filename, restore_frame = get_last_file(
                        f"backup/{params.file_namespace}",
                        f"^(?P<pre>{re.escape(base_name)}_)(?P<index>\\d*)(?P<post>\\.bak)$",
                    )
                else:
                    filename = f"{base_name}_{restore_frame}.bak"
                logger.info(f"restoring from {filename}")
                img.load_state_dict(
                    torch.load(f"backup/{params.file_namespace}/{filename}")
                )
            else:  # reencode
                if restore_frame == latest:
                    filename, restore_frame = get_last_file(
                        f"{outpath}/{params.file_namespace}",
                        f"^(?P<pre>{re.escape(base_name)}_)(?P<index>\\d*)(?P<post>\\.png)$",
                    )
                else:
                    filename = f"{base_name}_{restore_frame}.png"
                logger.info(f"restoring from {filename}")
                img.encode_image(
                    Image.open(f"{outpath}/{params.file_namespace}/{filename}").convert(
                        "RGB"
                    )
                )
            i = restore_frame * params.save_every
        else:
            i = 0

        # Phase 5 - setup optimizer
        ############################

        # make the main model object
        model = make_guide(params, setup, embedder, base_name, video_frames)

        # Run the training loop
        ########################

        # `i`: current iteration
        # `skip_X`: number of _X that have already been processed to completion (per the current iteration)
        # `last_scene`: previously processed scene/prompt (or current prompt if on first/only scene)
        skip_prompts = i // params.steps_per_scene
        skip_steps = i % params.steps_per_scene
        last_scene = prompts[0] if skip_prompts == 0 else prompts[skip_prompts - 1]
        for scene in prompts[skip_prompts:]:
            logger.info("Running prompt: " + " | ".join(map(str, scene)))
            i += model.run_steps(
                params.steps_per_scene - skip_steps,
                scene,
                last_scene,
                setup.loss_augs,
                interp_steps=params.interpolation_steps,
                i_offset=i,
                skipped_steps=skip_steps,
                gradient_accumulation_steps=params.gradient_accumulation_steps,
            )
            skip_steps = 0
            model.clear_loss_history()
            last_scene = scene

    try:
        gc.collect()
        empty_cache()
        do_run()
        logger.info("Complete.")
        gc.collect()
        empty_cache()
    except KeyboardInterrupt:
        pass
    except RuntimeError:
        print_vram_usage()
        raise


def _main():
    register_resolvers()
    ensure_configs_exist()
    migrate_local_config()
    _hydra_main()


if __name__ == "__main__":
    _main()
