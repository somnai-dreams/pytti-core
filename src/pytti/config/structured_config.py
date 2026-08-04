"""
The canonical pytti config schema — the single source of every setting and
its default. assets/default.yaml only selects this schema and the demo
values; presets override on top. Composed into every run via Hydra's
ConfigStore, so unknown keys and invalid values fail at startup with a
message instead of surfacing as AttributeErrors mid-render.
"""


from attrs import define, field
from hydra.core.config_store import ConfigStore
from omegaconf import MISSING

from pytti.config.model_names import (
    LLAMAGEN_MODEL_NAMES,
    VQGAN_MODEL_ALIASES,
    VQGAN_MODEL_NAMES,
)

# bump when the set of config keys changes; drives ./config migration
CONFIG_VERSION = 2


def _choice(valid_values):
    def validator(self, attribute, value):
        if value not in valid_values:
            raise ValueError(
                f"{value!r} is not a valid value for {attribute.name}. "
                f"Valid values: {list(valid_values)}"
            )

    return validator


@define(auto_attribs=True)
class AudioFilterConfig:
    variable_name: str = ""
    # None = unset; validated loudly when audio is actually used
    f_center: float | None = None
    f_width: float | None = None
    order: int = 5


@define(auto_attribs=True)
class ConfigSchema:
    #############
    ## Prompts ##
    #############

    scenes: str = MISSING
    scene_prefix: str = ""
    scene_suffix: str = ""

    direct_image_prompts: str = ""
    init_image: str = ""
    direct_init_weight: str = ""
    semantic_init_weight: str = ""

    ###################
    ## Image & model ##
    ###################

    image_model: str = field(
        default="Unlimited Palette",
        validator=_choice(
            ["Unlimited Palette", "Limited Palette", "VQGAN", "LlamaGen"]
        ),
    )
    vqgan_model: str = field(
        default="sflickr",
        validator=_choice(VQGAN_MODEL_NAMES + list(VQGAN_MODEL_ALIASES)),
    )
    # LlamaGen VQ variant: ds16 = the default look (f=16, same latent-grid
    # math as taming f16); ds8 = f=8, 4x the tokens, finer texture
    llamagen_model: str = field(
        default="ds16",
        validator=_choice(LLAMAGEN_MODEL_NAMES),
    )
    animation_mode: str = field(
        default="off", validator=_choice(["off", "2D", "3D", "Video Source"])
    )

    width: int = 512
    height: int = 512

    steps_per_scene: int = 100
    steps_per_frame: int = 50
    interpolation_steps: int = 0

    # Convergence auto-stop: end a scene early once the TOTAL loss plateaus;
    # steps_per_scene stays the hard cap. The loss is sampled every 10 steps
    # (one host sync per interval); the scene stops when relative improvement
    # over the trailing auto_stop_window steps falls below
    # auto_stop_threshold. Never fires before one full window of samples past
    # the interpolation ramp; multi-scene runs judge each scene independently.
    auto_stop: bool = False
    # trailing window, in steps, over which loss improvement is judged
    auto_stop_window: int = 50
    # relative improvement over the window below which the scene is converged
    auto_stop_threshold: float = 0.002

    # Phase scheduling: quality-phase behavior over normalized scene time
    # t_hat = step/steps_per_scene (still mode only — fails loud under any
    # animation_mode). One switch, three fixed schedules (the table lives in
    # src/pytti/phase_scheduling.py): the smoothing/TV weight ramps
    # 2x -> 0.5x of its configured value across each scene (structure early,
    # detail late), Limited Palette locks its palette for the final third
    # (clean crystallization), and the direct init-hold weight decays
    # 1x -> 0.5x (anchored start, freer finish). Scales multiply the
    # evaluated weight, so parametric weight expressions keep working.
    phase_scheduling: bool = False

    # Coarse-to-fine still rendering (still mode, single scene — fails loud
    # otherwise): stage 1 renders at half the configured dims (each dim
    # halved, rounded to /8, floored at 64) for the first 40% of
    # steps_per_scene; the result is bicubic-upscaled and re-encoded into a
    # fresh image rep at full dims (Limited Palette carries its learned
    # palette across), and stage 2 runs the remaining 60% with the stage-1
    # image as a weight-2 direct init hold plus the normal prompts. Frame
    # numbering and backups continue across the stage boundary (stage-1
    # frames save upscaled to the full canvas); auto_stop judges each stage
    # independently; phase_scheduling's t_hat spans each stage's own steps.
    coarse_to_fine: bool = False

    learning_rate: float | None = None
    reset_lr_each_frame: bool = True
    seed: int | None = None  # None = a fresh random seed each run
    # 16 is tuned for the smart sampler; classic/batched want ~40
    cutouts: int = 16
    cut_pow: float = 2
    cutout_border: float = 0.25
    border_mode: str = field(
        default="clamp",
        validator=_choice(["clamp", "mirror", "wrap", "black", "smear"]),
    )

    ##########
    # Camera #
    ##########

    field_of_view: int = 60
    near_plane: int = 1
    far_plane: int = 10000

    #######################
    ### Audioreactivity ###
    #######################

    input_audio: str = ""
    input_audio_offset: float = 0
    # None means no filters (omegaconf cannot ingest attrs list factories)
    input_audio_filters: list[AudioFilterConfig] | None = None

    ######################
    ### Induced Motion ###
    ######################

    #  _2d and _3d only apply to those animation modes
    translate_x: str = "0"
    translate_y: str = "0"
    translate_z_3d: str = "0"
    rotate_3d: str = "[1, 0, 0, 0]"
    rotate_2d: str = "0"
    zoom_x_2d: str = "0"
    zoom_y_2d: str = "0"

    sampling_mode: str = field(
        default="bicubic", validator=_choice(["nearest", "bilinear", "bicubic"])
    )
    infill_mode: str = field(
        default="wrap", validator=_choice(["mirror", "wrap", "black", "smear"])
    )

    pre_animation_steps: int = 50
    lock_camera: bool = True

    #######################
    ### Limited Palette ###
    #######################

    pixel_size: int = 1
    smoothing_weight: float = 0.02
    random_initial_palette: bool = False
    palette_size: int = 5
    palettes: int = 20
    gamma: float = 1
    hdr_weight: float = 0.01
    palette_normalization_weight: float = 0.2
    target_palette: str = ""
    lock_palette: bool = False

    #####################
    ### Stabilization ###
    #####################

    frames_per_second: int = 12

    direct_stabilization_weight: str = ""
    semantic_stabilization_weight: str = ""
    depth_stabilization_weight: str = ""
    edge_stabilization_weight: str = ""
    flow_stabilization_weight: str = ""

    #####################################
    ### animation_mode = Video Source ###
    #####################################

    video_path: str = ""
    frame_stride: int = 1
    reencode_each_frame: bool = True
    flow_long_term_samples: int = 1

    ############
    ### CLIP ###
    ############

    ViTB32: bool = True
    ViTB16: bool = False
    ViTL14: bool = False
    ViTL14_336px: bool = False
    # FARE robust CLIP: cleaner, perceptually-aligned gradients
    FARE4ViTB32: bool = False
    FARE2ViTL14: bool = False
    # SigLIP2: much stronger prompt semantics than the classic towers
    SigLIP2B16: bool = False
    SigLIP2SO400M: bool = False
    RN50: bool = False
    RN101: bool = False
    RN50x4: bool = False
    RN50x16: bool = False
    RN50x64: bool = False

    ###############
    ### Outputs ###
    ###############

    file_namespace: str = "default"
    allow_overwrite: bool = False
    display_every: int = 50
    # 0 = save one frame per animation frame (steps_per_frame). The 0-default
    # keeps saved frames locked to animation frames even when steps_per_frame
    # changes — a manual value is an explicit override.
    save_every: int = 0

    # crossfade saved frames from init_image to the optimized output over the
    # course of the render (requires init_image)
    breath_mode: bool = False

    backups: int = 3
    approximate_vram_usage: bool = False

    # resume from the latest backup in backup/<file_namespace>/
    restore: bool = False

    #################
    ### Model I/O ###
    #################

    # This is where pytti will expect to find model weights.
    # Each model will be assigned a separate subdirectory within this folder
    # If the expected model artifacts are not present, pytti will attempt to download them.
    models_parent_dir: str = "${user_cache:}"

    ##########################
    ### Performance tuning ###
    ##########################

    # smart (default) = designed two-population sampler: full-frame global
    # anchors + stratified detail cuts, sync-free. A/B verdict 2026-07-31:
    # at cutn 16 it BEAT batched@40 on held-out ViT-L/14 adherence
    # (0.375 vs 0.366) at ~2.3x less tower compute. batched = classic's
    # distribution in one grid_sample (wants cutn ~40); classic = the
    # original 2021 per-crop loop, kept as the legacy preset
    cutout_sampler: str = field(
        default="smart", validator=_choice(["classic", "batched", "smart"])
    )
    # adamw_sf = schedule-free AdamW (Polyak-averaged eval iterate)
    optimizer: str = field(default="adam", validator=_choice(["adam", "adamw_sf"]))

    # mlx = run the CLIP towers + prompt-loss reduction on MLX (Apple Metal,
    # fp16; macOS only, classic ViT towers only — docs/mlx-port-plan.md M1).
    # mlx_full = run the ENTIRE still-image step on MLX (M2): still mode,
    # Limited/Unlimited Palette, batched|smart sampler, plain adam, no
    # semantic masks / semantic image prompts / depth — anything else fails
    # loudly at startup naming the backend that supports it.
    # torch = the reference engine, every tower, every platform.
    perceptor_backend: str = field(
        default="torch", validator=_choice(["torch", "mlx", "mlx_full"])
    )

    gradient_accumulation_steps: int = 1
    # None = auto (cuda > mps > cpu); or e.g. "cuda:1", "mps", "cpu"
    device: str | None = None

    # version of the local ./config format; used for migration
    config_version: int = CONFIG_VERSION


def register():
    cs = ConfigStore.instance()
    cs.store(name="config_schema", node=ConfigSchema)


register()
