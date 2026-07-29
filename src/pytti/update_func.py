from pathlib import Path

import torch
from loguru import logger

from pytti import (
    freeze_vram_usage,
    parametric_eval,
    print_vram_usage,
    set_t,
    vram_usage_mode,
)
from pytti.rotoscoper import update_rotoscopers
from pytti.Transforms import (
    animate_video_source,
    zoom_2d,
    zoom_3d,
)


# Update is called each step.
def update(
    model,
    img,
    i,
    stage_i,
    params=None,
    base_name=None,
    optical_flows=None,
    video_frames=None,
    stabilization_augs=None,
    last_frame_semantic=None,
    embedder=None,
    init_augs=None,
    semantic_init_prompt=None,
):
    # NB: this must be computed at call time — hydra chdirs into the run's
    # output directory before the render starts.
    outpath = Path.cwd() / "images_out"

    def report_out(i):
        logger.debug(f"Step {i} losses:")
        for name, value in model.loss_history[-1].items() if model.loss_history else []:
            logger.debug(f"  {name}: {value:.6f}")
        if params.approximate_vram_usage:
            logger.debug("VRAM Usage:")
            print_vram_usage()

    def save_out(i, img, save_every):
        im = img.decode_image()
        n = (i + 1) // save_every
        frame_dir = outpath / params.file_namespace
        frame_dir.mkdir(parents=True, exist_ok=True)
        im.save(frame_dir / f"{base_name}_{n}.png")

        if params.backups > 0:
            backup_dir = Path("backup") / params.file_namespace
            backup_dir.mkdir(parents=True, exist_ok=True)
            torch.save(img.state_dict(), backup_dir / f"{base_name}_{n}.bak")
            if n > params.backups:
                stale = backup_dir / f"{base_name}_{n - params.backups}.bak"
                if stale.exists():
                    stale.unlink()

    j = i + 1

    if (params.display_every > 0) and (j % params.display_every == 0):
        report_out(i)

    if (i > 0) and (params.save_every > 0) and (j % params.save_every == 0):
        save_out(i, img, params.save_every)

    # animate
    ################
    t = (i - params.pre_animation_steps) / (
        params.steps_per_frame * params.frames_per_second
    )
    set_t(t, {})
    if i >= params.pre_animation_steps:
        if (i - params.pre_animation_steps) % params.steps_per_frame == 0:
            if model.audio_parser is not None:
                band_dict = model.audio_parser.get_params(t)
                logger.debug(f"Time: {t:.4f} seconds, audio params: {band_dict}")
                set_t(t, band_dict)
            else:
                logger.debug(f"Time: {t:.4f} seconds")

            update_rotoscopers(
                ((i - params.pre_animation_steps) // params.steps_per_frame + 1)
                * params.frame_stride
            )
            if params.reset_lr_each_frame:
                model.set_optim(None)

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

                for optical_flow in optical_flows:
                    optical_flow.set_last_step(im)
                    optical_flow.set_target_flow(flow)
                    optical_flow.set_enabled(True)
            elif params.animation_mode == "Video Source":
                flow_im, next_step_pil = animate_video_source(
                    i=i,
                    img=img,
                    video_frames=video_frames,
                    optical_flows=optical_flows,
                    base_name=base_name,
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
                for aug in stabilization_augs:
                    aug.set_comp(next_step_pil)
                    aug.set_enabled(True)
                if last_frame_semantic is not None:
                    last_frame_semantic.set_image(embedder, next_step_pil)
                    last_frame_semantic.set_enabled(True)
                for aug in init_augs:
                    aug.set_enabled(False)
                if semantic_init_prompt is not None:
                    semantic_init_prompt.set_enabled(False)
