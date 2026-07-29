"""
CLI rendering entry point. Hydra composes the config, `do_run` executes it.
"""

import gc
import os
import re
from pathlib import Path

import hydra
import torch
from loguru import logger
from omegaconf import DictConfig, OmegaConf, open_dict
from PIL import Image

from pytti import (
    empty_cache,
    fetch,
    print_vram_usage,
    reset_vram_usage,
    set_default_device,
    vram_profiling,
    vram_usage_mode,
)
from pytti.files import get_last_file, get_next_file
from pytti.image_models import PixelImage, RGBImage, VQGANImage
from pytti.ImageGuide import DirectImageGuide
from pytti.LossAug.LossOrchestratorClass import (
    configure_init_image,
    configure_optical_flows,
    configure_stabilization_augs,
)
from pytti.Perceptor import load_clip
from pytti.Perceptor.Embedder import HDMultiClipEmbedder
from pytti.Perceptor.Prompt import parse_prompt
from pytti.rotoscoper import ROTOSCOPERS, get_frames
from pytti.update_func import update
from pytti.warmup import ensure_configs_exist, register_resolvers


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


@hydra.main(config_path="config", config_name="default", version_base=None)
def _hydra_main(cfg: DictConfig):
    params = cfg

    device = set_default_device(params.get("device"))
    if params.get("device") is None:
        with open_dict(params) as p:
            p.device = str(device)
    logger.debug(f"Using device {device}")

    # literal "off" in yaml interpreted as False
    if params.animation_mode == False:  # noqa: E712
        params.animation_mode = "off"

    logger.debug(OmegaConf.to_container(cfg, resolve=True))
    latest = -1

    # @markdown check `restore` to restore from a previous run
    restore = params.get("restore") or False
    # @markdown check `reencode` if you are restoring with a modified image or modified image settings
    reencode = False
    # @markdown which run to restore
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

        # set up seed for deterministic RNG
        if params.seed is not None:
            torch.manual_seed(params.seed)

        # Phase 2 - load and parse
        ###########################

        # load CLIP
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

        # Phase 3 - Setup Optimization
        ###############################

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
            img.encode_random(random_palette=params.random_initial_palette)
            if params.target_palette.strip() != "":
                img.set_palette_target(
                    Image.open(fetch(params.target_palette)).convert("RGB")
                )
            else:
                img.lock_palette(params.lock_palette)
        elif params.image_model == "Unlimited Palette":
            img = RGBImage(
                params.width, params.height, params.pixel_size, device=device
            )
            img.encode_random()
        elif params.image_model == "VQGAN":
            model_artifacts_path = Path(params.models_parent_dir) / "vqgan"
            VQGANImage.init_vqgan(params.vqgan_model, model_artifacts_path, device=device)
            img = VQGANImage(
                params.width, params.height, params.pixel_size, device=device
            )
            img.encode_random()
        else:
            raise ValueError(
                f"Unrecognized image_model: {params.image_model!r}. "
                'Supported: "Limited Palette", "Unlimited Palette", "VQGAN".'
            )

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

        # other image prompts
        loss_augs.extend(
            type(img)
            .get_preferred_loss()
            .TargetImage(p.strip(), img.image_shape, is_path=True)
            for p in params.direct_image_prompts.split("|")
            if p.strip()
        )

        # stabilization
        (
            loss_augs,
            img,
            init_image_pil,
            stabilization_augs,
        ) = configure_stabilization_augs(img, init_image_pil, params, loss_augs)

        if params.semantic_stabilization_weight not in ["0", ""]:
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

        # Phase 4 - setup outputs
        ##########################

        # set up filespace
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
        model = DirectImageGuide(
            image_rep=img,
            embedder=embedder,
            lr=params.learning_rate,
            params=params,
            base_name=base_name,
            optical_flows=optical_flows,
            video_frames=video_frames,
            stabilization_augs=stabilization_augs,
            last_frame_semantic=last_frame_semantic,
            init_augs=init_augs,
            semantic_init_prompt=semantic_init_prompt,
        )
        model.update = update

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
                loss_augs,
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
    _hydra_main()


if __name__ == "__main__":
    _main()
